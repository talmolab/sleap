"""Tier 0.5 verify-only regression test for the sleap-io 0.8.0 + sleap-nn 0.3.0 upgrade.

GUARD: sleap-io 0.8.0 #480 — the SLEAP Analysis HDF5 export now spans the FULL
video length instead of stopping at the last labeled/predicted frame.

In sleap this is driven by :class:`SleapAnalysisAdaptor.write` (used by the GUI
``Export Analysis File`` command), which delegates to
``sio.save_analysis_h5(..., all_frames=True, preset="matlab")``. Pre-0.8.0 the
exported frame axis only ran from frame 0 to the last labeled frame; with #480
``to_analysis_arrays`` clamps the last frame to ``len(video) - 1`` so the dense
arrays cover every frame of the source video.

These tests build a ``Labels`` whose predictions live only on EARLY frames of a
known-length video, export through sleap's adaptor, and assert the exported
``track_occupancy`` / ``tracks`` / score datasets span the full video length
(while occupancy is still confined to the early frames).
"""

import json

import h5py
import numpy as np

from sleap_io import Labels, LabeledFrame, Skeleton, Track
from sleap_io.model.instance import PredictedInstance

from sleap.io.format.sleap_analysis import SleapAnalysisAdaptor


def _frame_axis_length(h5_path, dataset):
    """Return (frame_axis_length, dims_tuple) for a dataset, using its dims attr.

    The matlab preset stores arrays with different axis orders (e.g. ``tracks``
    is ``(track, xy, node, frame)`` while ``track_occupancy`` is
    ``(frame, track)``). We locate the frame axis by name from the stored
    ``dims`` attribute so the assertions don't hard-code positions.
    """
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset]
        dims = tuple(json.loads(ds.attrs["dims"]))
        frame_axis = dims.index("frame")
        return ds.shape[frame_axis], dims


def _occupancy_frame_first(h5_path):
    """Read ``track_occupancy`` and return it as a (frame, track) array."""
    with h5py.File(h5_path, "r") as f:
        ds = f["track_occupancy"]
        occ = ds[:]
        dims = tuple(json.loads(ds.attrs["dims"]))
    if dims.index("frame") != 0:
        occ = np.swapaxes(occ, 0, 1)
    return occ


def _tracks_frame_first(h5_path):
    """Read ``tracks`` and return it reordered to (frame, track, node, xy)."""
    with h5py.File(h5_path, "r") as f:
        ds = f["tracks"]
        data = ds[:]
        dims = tuple(json.loads(ds.attrs["dims"]))
    target = ("frame", "track", "node", "xy")
    axes = tuple(dims.index(name) for name in target)
    return np.transpose(data, axes)


def _make_early_frame_labels(video, *, n_labeled, tracked=True):
    """Build a Labels with predicted instances only on frames 0..n_labeled-1."""
    skeleton = Skeleton(["a", "b"])
    track = Track(name="track0") if tracked else None
    lfs = []
    for frame_idx in range(n_labeled):
        inst = PredictedInstance.from_numpy(
            np.array([[10.0 + frame_idx, 20.0], [30.0, 40.0 + frame_idx]]),
            skeleton=skeleton,
            point_scores=np.array([0.9, 0.8]),
            score=0.95,
            track=track,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    labels = Labels(
        labeled_frames=lfs,
        videos=[video],
        skeletons=[skeleton],
        tracks=[track] if tracked else [],
    )
    return labels


def test_analysis_h5_export_spans_full_video_length(small_robot_mp4_vid, tmp_path):
    """Core #480 guard: HDF5 frame axis == full video length, not last labeled+1."""
    video = small_robot_mp4_vid
    n_frames = len(video)
    n_labeled = 5  # predictions only on frames 0..4

    # Sanity: the video must be longer than the labeled span for this to be a
    # meaningful test of "spans full video length".
    assert n_frames > n_labeled

    labels = _make_early_frame_labels(video, n_labeled=n_labeled, tracked=True)
    assert max(lf.frame_idx for lf in labels) == n_labeled - 1

    out_path = str(tmp_path / "robot.analysis.h5")
    SleapAnalysisAdaptor.write(
        filename=out_path,
        source_object=labels,
        source_path="source.slp",
        video=video,
    )

    # Every dense dataset's frame axis must span the FULL video, not just the
    # labeled span (which would be n_labeled under the pre-0.8.0 behavior).
    for dataset in ("tracks", "track_occupancy", "point_scores", "instance_scores",
                    "tracking_scores"):
        frame_len, dims = _frame_axis_length(out_path, dataset)
        assert frame_len == n_frames, (
            f"{dataset} frame axis {frame_len} (dims={dims}) != "
            f"full video length {n_frames}"
        )
        # Explicit contrast with the old behavior (stop at last labeled frame).
        assert frame_len > n_labeled

    # Occupancy must still be confined to the early frames: every frame is a
    # row, but only the first n_labeled rows are occupied.
    occ = _occupancy_frame_first(out_path)  # (frame, track)
    assert occ.shape == (n_frames, 1)
    occupied_frames = np.flatnonzero(occ.any(axis=1))
    np.testing.assert_array_equal(occupied_frames, np.arange(n_labeled))

    # The trailing (unlabeled) frames are padded with NaN coordinates, while the
    # early labeled frames carry real coordinates.
    tracks = _tracks_frame_first(out_path)  # (frame, track, node, xy)
    assert tracks.shape[0] == n_frames
    assert np.isnan(tracks[n_labeled:]).all()
    assert not np.isnan(tracks[:n_labeled]).any()


def test_analysis_h5_full_length_realistic_subset(
    centered_pair_predictions_sorted, tmp_path
):
    """Realistic fixture: subset real predictions to early frames, full span kept."""
    labels = centered_pair_predictions_sorted
    video = labels.videos[0]
    n_frames = len(video)
    assert n_frames > 0

    # Keep only the first K frames of real predictions.
    k = 6
    labels.labeled_frames = [lf for lf in labels.labeled_frames if lf.frame_idx < k]
    assert len(labels.labeled_frames) == k
    last_labeled = max(lf.frame_idx for lf in labels)
    assert last_labeled < n_frames - 1  # there are real trailing frames to span

    out_path = str(tmp_path / "centered_pair.analysis.h5")
    SleapAnalysisAdaptor.write(
        filename=out_path,
        source_object=labels,
        source_path=None,
        video=video,
    )

    # Frame axis spans the entire 1100-frame video, not just the K early frames.
    occ_len, _ = _frame_axis_length(out_path, "track_occupancy")
    tracks_len, _ = _frame_axis_length(out_path, "tracks")
    assert occ_len == n_frames
    assert tracks_len == n_frames
    assert occ_len > k

    # Only the early frames are actually occupied.
    occ = _occupancy_frame_first(out_path)
    occupied_frames = np.flatnonzero(occ.any(axis=1))
    assert occupied_frames.max() == last_labeled
    assert len(occupied_frames) == k


def test_analysis_h5_full_length_untracked(small_robot_mp4_vid, tmp_path):
    """Untracked (no Track assignments) export also spans the full video length."""
    video = small_robot_mp4_vid
    n_frames = len(video)
    n_labeled = 4

    labels = _make_early_frame_labels(video, n_labeled=n_labeled, tracked=False)
    assert len(labels.tracks) == 0

    out_path = str(tmp_path / "robot_untracked.analysis.h5")
    SleapAnalysisAdaptor.write(
        filename=out_path,
        source_object=labels,
        source_path=None,
        video=video,
    )

    occ_len, _ = _frame_axis_length(out_path, "track_occupancy")
    tracks_len, _ = _frame_axis_length(out_path, "tracks")
    assert occ_len == n_frames
    assert tracks_len == n_frames
    assert occ_len > n_labeled

    occ = _occupancy_frame_first(out_path)
    occupied_frames = np.flatnonzero(occ.any(axis=1))
    np.testing.assert_array_equal(occupied_frames, np.arange(n_labeled))

    # A single synthetic track is created for the untracked single-instance case.
    with h5py.File(out_path, "r") as f:
        track_names = [n.decode("utf-8") for n in f["track_names"][:]]
    assert track_names == ["track_0"]

"""Tests for render-clip GUI plumbing.

Covers the bugs surfaced by rendering a multi-video predictions file:
  - Duplicated skeletons when a user-corrected ``Instance`` coexists with the
    ``PredictedInstance`` it was derived from.
  - Out-of-order output when ``Labels.labeled_frames`` is not stored in frame
    order (``sio.render_video`` writes frames in the order of ``frame_inds``).
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio

from sleap.gui.commands import _labels_with_visible_instances


def _skeleton() -> sio.Skeleton:
    return sio.Skeleton([sio.Node("a"), sio.Node("b")])


def _pred(skel: sio.Skeleton, xy: tuple[float, float]) -> sio.PredictedInstance:
    points = np.array([[xy[0], xy[1]], [xy[0] + 1.0, xy[1] + 1.0]])
    return sio.PredictedInstance.from_numpy(points, skeleton=skel, score=0.9)


def _user(
    skel: sio.Skeleton,
    xy: tuple[float, float],
    from_predicted: sio.PredictedInstance | None = None,
) -> sio.Instance:
    points = np.array([[xy[0], xy[1]], [xy[0] + 1.0, xy[1] + 1.0]])
    inst = sio.Instance.from_numpy(points, skeleton=skel)
    if from_predicted is not None:
        inst.from_predicted = from_predicted
    return inst


def test_labels_with_visible_instances_hides_used_predictions():
    skel = _skeleton()
    video_a = sio.Video(filename="a.mp4")
    video_b = sio.Video(filename="b.mp4")

    # Frame 0: prediction corrected by a user instance -> prediction hidden.
    pred_used = _pred(skel, (10.0, 20.0))
    user_corrected = _user(skel, (11.0, 21.0), from_predicted=pred_used)
    lf_corrected = sio.LabeledFrame(
        video=video_a, frame_idx=0, instances=[pred_used, user_corrected]
    )

    # Frame 1: only a raw prediction -> kept.
    pred_orphan = _pred(skel, (30.0, 40.0))
    lf_orphan = sio.LabeledFrame(
        video=video_a, frame_idx=1, instances=[pred_orphan]
    )

    # Frame 2 (different video): prediction + user pair, must also be filtered
    # because we pass video=None (means "filter all videos").
    pred_used_b = _pred(skel, (50.0, 60.0))
    user_corrected_b = _user(skel, (51.0, 61.0), from_predicted=pred_used_b)
    lf_other_video = sio.LabeledFrame(
        video=video_b,
        frame_idx=5,
        instances=[pred_used_b, user_corrected_b],
    )

    labels = sio.Labels(
        labeled_frames=[lf_corrected, lf_orphan, lf_other_video],
        videos=[video_a, video_b],
        skeletons=[skel],
    )

    # Filter only video_a: video_b frame should pass through unchanged.
    filtered = _labels_with_visible_instances(labels, video_a)

    assert filtered is not labels
    assert filtered.labeled_frames is not labels.labeled_frames
    # Original must not be mutated.
    assert [type(i).__name__ for i in lf_corrected.instances] == [
        "PredictedInstance",
        "Instance",
    ]

    # video_a/frame 0: prediction hidden, only user instance visible.
    out_corrected = next(
        lf
        for lf in filtered.labeled_frames
        if lf.video is video_a and lf.frame_idx == 0
    )
    assert len(out_corrected.instances) == 1
    assert isinstance(out_corrected.instances[0], sio.Instance)
    assert not isinstance(out_corrected.instances[0], sio.PredictedInstance)

    # video_a/frame 1: orphan prediction kept.
    out_orphan = next(
        lf
        for lf in filtered.labeled_frames
        if lf.video is video_a and lf.frame_idx == 1
    )
    assert len(out_orphan.instances) == 1
    assert isinstance(out_orphan.instances[0], sio.PredictedInstance)

    # video_b frame: passthrough (same LabeledFrame object).
    out_other = next(lf for lf in filtered.labeled_frames if lf.video is video_b)
    assert out_other is lf_other_video


def test_labels_with_visible_instances_uses_track_when_present():
    """With tracks, the "used" check goes through tracks, not ``from_predicted``."""
    skel = _skeleton()
    video = sio.Video(filename="a.mp4")
    track = sio.Track(name="t0")

    pred = _pred(skel, (10.0, 20.0))
    pred.track = track
    user = _user(skel, (11.0, 21.0))
    user.track = track

    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred, user])
    labels = sio.Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skel],
        tracks=[track],
    )

    filtered = _labels_with_visible_instances(labels, video)

    out = filtered.labeled_frames[0]
    assert len(out.instances) == 1
    assert isinstance(out.instances[0], sio.Instance)
    assert not isinstance(out.instances[0], sio.PredictedInstance)


def test_labels_with_visible_instances_preserves_top_level_state():
    skel = _skeleton()
    video = sio.Video(filename="a.mp4")
    track = sio.Track(name="t0")
    pred = _pred(skel, (0.0, 0.0))
    user = _user(skel, (0.1, 0.1), from_predicted=pred)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred, user])
    labels = sio.Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skel],
        tracks=[track],
        provenance={"source": "test"},
    )

    filtered = _labels_with_visible_instances(labels, video)

    # Shared references preserved; only labeled_frames is new.
    assert filtered.videos is labels.videos
    assert filtered.skeletons is labels.skeletons
    assert filtered.tracks is labels.tracks
    assert filtered.provenance == labels.provenance


def test_get_frame_indices_sorts_output(centered_pair_predictions):
    """``RenderClipDialog.get_frame_indices()`` must return frame indices in
    ascending order — ``Labels.labeled_frames`` storage is not guaranteed to
    be in frame order, and ``sio.render_video`` writes frames in the order it
    receives them.
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions

    # Shuffle labeled_frames to mimic the wild — sleap-io load order is not
    # guaranteed to be frame-sorted on all files (confirmed with multi-video
    # prediction files exported via the GUI).
    lfs = list(labels.labeled_frames)
    shuffled = [lfs[i] for i in (len(lfs) // 2, 0, *range(1, len(lfs) // 2))]
    shuffled += [lf for lf in lfs if lf not in shuffled]
    labels.labeled_frames = shuffled

    video = labels.videos[0]
    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        # Only the "all labeled frames" radio is exercised here — custom range
        # uses the same comprehension and sort.
        dialog.range_all.setChecked(True)
        indices = dialog.get_frame_indices()
    finally:
        dialog.deleteLater()

    assert indices == sorted(indices)
    assert len(indices) == sum(1 for lf in labels.labeled_frames if lf.video == video)


def test_match_source_fps_checkbox(centered_pair_predictions):
    """Dialog's match-source-fps checkbox should pin the FPS spinbox to the
    source video's reported FPS when checked, disable the spinbox, and restore
    manual editing when unchecked. Pre-corrections shipped a hardcoded
    ``fps=30`` default that silently slowed high-fps source footage (120+ fps
    in this project).
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]
    src_fps = video.backend.fps

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        assert dialog.match_source_fps.isChecked()
        assert dialog.fps.value() == int(round(src_fps))
        assert not dialog.fps.isEnabled()

        # Unchecking re-enables the spinbox without changing its value.
        dialog.match_source_fps.setChecked(False)
        assert dialog.fps.isEnabled()
        dialog.fps.setValue(15)
        assert dialog.get_export_params()["fps"] == 15

        # Re-checking pins back to source fps.
        dialog.match_source_fps.setChecked(True)
        assert dialog.fps.value() == int(round(src_fps))
        assert not dialog.fps.isEnabled()
        assert dialog.get_export_params()["fps"] == int(round(src_fps))
    finally:
        dialog.deleteLater()

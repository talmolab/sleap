"""Tier 0.5 verify-only smoke tests for the sleap-io 0.8.0 upgrade.

These lock in three "confirmed non-breakage" items from the upgrade audit so a
future sleap-io bump that *does* break them is caught cheaply:

1. IO-VER-RO  -- read-only views: only ``Labels.instances`` (the aggregate
   iterator property) is read-only. ``LabeledFrame.instances`` is a plain
   mutable ``list``; append / pop / whole-list reassignment all work. These are
   real mutation patterns used in ``sleap/gui/commands.py`` and
   ``sleap/sleap_io_adaptors/lf_labels_utils.py``.

2. IO-VER-SKEL -- structural skeleton dedup: sleap-io 0.8.0 (#447)
   canonicalizes structurally-identical skeletons in update/append/extend, so
   distinct-but-equal ``Skeleton`` objects collapse to one canonical skeleton
   with point data preserved. sleap's ``OpenSkeleton.delete_extra_skeletons``
   path stays a no-op on the resulting single-skeleton ``Labels``.

3. IO-VER-SLP -- .slp format 2.4 round-trip: saving a predictions fixture
   through sleap's save path (``sio.save_file``) and reloading preserves frame
   count, per-frame instance counts, frame indices, track names/count and
   sample point coordinates.
"""

import numpy as np
import sleap_io as sio
from sleap_io import Labels, LabeledFrame, Skeleton, Video
from sleap_io.model.instance import Instance


# ---------------------------------------------------------------------------
# 1. IO-VER-RO: read-only views boundary
# ---------------------------------------------------------------------------


def _make_lf(n_instances=1):
    """Build a standalone LabeledFrame with `n_instances` user instances."""
    skel = Skeleton(["a", "b"])
    video = Video.from_filename("video.mp4")
    instances = [
        Instance.from_numpy(
            np.array([[i, i], [i + 1, i + 1]], dtype="float32"), skeleton=skel
        )
        for i in range(n_instances)
    ]
    return LabeledFrame(video=video, frame_idx=0, instances=instances), skel, video


def test_labeled_frame_instances_is_mutable_list():
    """LabeledFrame.instances supports append / pop / reassignment (real GUI ops)."""
    lf, skel, _ = _make_lf(n_instances=1)

    # It is a plain list, not a read-only view/proxy.
    assert isinstance(lf.instances, list)
    assert len(lf.instances) == 1

    # append (gui/commands.py: lf.instances.append(new_instance))
    new_inst = Instance.from_numpy(
        np.array([[9, 9], [10, 10]], dtype="float32"), skeleton=skel
    )
    lf.instances.append(new_inst)
    assert len(lf.instances) == 2
    assert lf.instances[-1] is new_inst

    # pop (lf_labels_utils.py: lf_inst_to_remove.instances.pop(inst_idx))
    popped = lf.instances.pop()
    assert popped is new_inst
    assert len(lf.instances) == 1

    # whole-list reassignment (gui/commands.py: lf.instances = []; and
    # lf_labels_utils.py: frame.instances = list(frame.instances) + [instance])
    lf.instances = []
    assert lf.instances == []
    extra = Instance.from_numpy(
        np.array([[1, 1], [2, 2]], dtype="float32"), skeleton=skel
    )
    lf.instances = list(lf.instances) + [extra]
    assert lf.instances == [extra]


def test_labels_instances_property_is_read_only():
    """Labels.instances is a read-only aggregate property (no setter)."""
    prop = Labels.instances
    assert isinstance(prop, property)
    assert prop.fset is None  # documents the read-only boundary

    lf, _, _ = _make_lf(n_instances=2)
    labels = Labels([lf])

    # The aggregate property iterates over all per-frame instances.
    aggregated = list(labels.instances)
    assert len(aggregated) == 2
    assert aggregated == list(lf.instances)

    # Assigning to the aggregate property is rejected.
    try:
        labels.instances = [lf.instances[0]]
        raised = False
    except AttributeError:
        raised = True
    assert raised, "Labels.instances unexpectedly accepted assignment"


# ---------------------------------------------------------------------------
# 2. IO-VER-SKEL: structural skeleton dedup
# ---------------------------------------------------------------------------


def test_structural_skeleton_dedup_on_construction():
    """Distinct-but-structurally-identical skeletons collapse to one canonical."""
    node_names = ["x", "y", "z"]
    skel_a = Skeleton(list(node_names))
    skel_b = Skeleton(list(node_names))

    # Distinct objects (Skeleton is eq=False / identity-based), but structurally
    # identical with the same node order.
    assert skel_a is not skel_b
    assert skel_a.matches(skel_b, require_same_order=True)

    video = Video.from_filename("video.mp4")
    pts_a = np.array([[1, 1], [2, 2], [3, 3]], dtype="float32")
    pts_b = np.array([[4, 4], [5, 5], [6, 6]], dtype="float32")
    inst_a = Instance.from_numpy(pts_a, skeleton=skel_a)
    inst_b = Instance.from_numpy(pts_b, skeleton=skel_b)
    lf_a = LabeledFrame(video=video, frame_idx=0, instances=[inst_a])
    lf_b = LabeledFrame(video=video, frame_idx=1, instances=[inst_b])

    labels = Labels([lf_a, lf_b])

    # Dedup: exactly one canonical skeleton shared across both instances.
    assert len(labels.skeletons) == 1
    assert labels[0][0].skeleton is labels[1][0].skeleton
    assert labels.skeletons[0].node_names == node_names

    # Point data preserved through canonicalization (same node order => no
    # reordering of positional points).
    np.testing.assert_array_equal(labels[0][0].numpy(), pts_a)
    np.testing.assert_array_equal(labels[1][0].numpy(), pts_b)


def test_delete_extra_skeletons_noop_on_single_canonical_skeleton():
    """sleap's delete-extra-skeletons path is a safe no-op on one skeleton."""
    from sleap.gui.commands import OpenSkeleton

    node_names = ["x", "y", "z"]
    video = Video.from_filename("video.mp4")
    inst_a = Instance.from_numpy(
        np.array([[1, 1], [2, 2], [3, 3]], dtype="float32"),
        skeleton=Skeleton(list(node_names)),
    )
    inst_b = Instance.from_numpy(
        np.array([[4, 4], [5, 5], [6, 6]], dtype="float32"),
        skeleton=Skeleton(list(node_names)),
    )
    labels = Labels(
        [
            LabeledFrame(video=video, frame_idx=0, instances=[inst_a]),
            LabeledFrame(video=video, frame_idx=1, instances=[inst_b]),
        ]
    )

    # Precondition: dedup already produced a single canonical skeleton.
    assert len(labels.skeletons) == 1
    canonical = labels.skeletons[0]

    # delete_extra_skeletons short-circuits when len(skeletons) <= 1: no crash,
    # no change.
    OpenSkeleton.delete_extra_skeletons(labels)
    assert len(labels.skeletons) == 1
    assert labels.skeletons[0] is canonical


# ---------------------------------------------------------------------------
# 3. IO-VER-SLP: .slp format 2.4 round-trip
# ---------------------------------------------------------------------------


def test_slp_round_trip_preserves_predictions(min_tracks_2node_predictions, tmp_path):
    """Saving + reloading a predictions .slp preserves structure and points."""
    src: Labels = min_tracks_2node_predictions

    # Capture source invariants.
    src_n_frames = len(src.labeled_frames)
    src_frame_idxs = [lf.frame_idx for lf in src.labeled_frames]
    src_counts = [len(lf.instances) for lf in src.labeled_frames]
    src_total = sum(src_counts)
    src_track_names = [t.name for t in src.tracks]

    assert src_n_frames > 0
    assert src_total > 0
    assert len(src_track_names) >= 2  # 'female', 'male'

    # Sample point coordinates from the first non-empty frame.
    sample_lf = next(lf for lf in src.labeled_frames if len(lf.instances) > 0)
    sample_frame_idx = sample_lf.frame_idx
    sample_pts = sample_lf.instances[0].numpy().copy()

    # Save through sleap's save path (commands.py uses sio.save_file with
    # format=None for .slp) and reload.
    out = tmp_path / "round_trip.slp"
    sio.save_file(labels=src, filename=str(out), format=None)
    assert out.exists()
    dst: Labels = sio.load_file(str(out))

    # Frame count + indices.
    assert len(dst.labeled_frames) == src_n_frames
    assert [lf.frame_idx for lf in dst.labeled_frames] == src_frame_idxs

    # Per-frame and total instance counts.
    assert [len(lf.instances) for lf in dst.labeled_frames] == src_counts
    assert sum(len(lf.instances) for lf in dst.labeled_frames) == src_total

    # Track names + count.
    assert [t.name for t in dst.tracks] == src_track_names
    assert len(dst.tracks) == len(src_track_names)

    # Skeleton(s) survive as a single canonical skeleton with same nodes.
    assert len(dst.skeletons) == len(src.skeletons)
    assert dst.skeletons[0].node_names == src.skeletons[0].node_names

    # Sample point coordinates survive byte-faithfully (nan-aware exact compare).
    dst_lf = next(
        lf for lf in dst.labeled_frames if lf.frame_idx == sample_frame_idx
    )
    dst_pts = dst_lf.instances[0].numpy()
    np.testing.assert_array_equal(
        np.nan_to_num(dst_pts, nan=-1.0), np.nan_to_num(sample_pts, nan=-1.0)
    )

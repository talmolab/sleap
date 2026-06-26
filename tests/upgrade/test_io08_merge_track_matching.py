"""Tier 0.5 regression tests: sleap-io 0.8.0 track-matching default flip.

sleap-io 0.8.0 (talmolab/sleap-io#449) flipped the default track matcher of
``Labels.merge()`` / ``Labels.match()`` from ``"name"`` to ``"identity"``:

- ``"identity"`` (the new default) matches tracks ONLY by Python object identity
  (the same ``Track`` instance). Two distinct ``Track`` objects that happen to
  share a name are kept as separate tracks.
- ``"name"`` matches tracks by their ``name`` attribute, collapsing same-named
  tracks across the two projects into one.

SLEAP pins ``track="identity"`` explicitly on every ``Labels.merge()`` call site
in the GUI (``gui/learning/runners.py``, ``gui/dialogs/merge.py``,
``gui/commands.py``). This RESTORES SLEAP's original, pre-sleap-io-port merge
behavior: tracks were matched by object identity, so same-named tracks coming
from a separate file or inference run are NOT collapsed into the project's
tracks. (The ``"name"`` behavior was only ever an artifact of the 0.7.x sleap-io
default; 0.8.0's identity default lines back up with original SLEAP, and we pass
it explicitly so the behavior is pinned regardless of future default changes.)

These tests LOCK IN that behavior: they must pass against the current worktree.
If a future change re-introduces ``track="name"`` on these call sites, the
dialog/HITL tests here will regress to collapsed tracks and fail.
"""

import numpy as np

from sleap_io import (
    LabeledFrame,
    Labels,
    PredictedInstance,
    Skeleton,
    Track,
    Video,
)


def _labels_with_named_track(
    track_name: str,
    video: Video,
    frame_idx: int = 0,
    coords=((10.0, 10.0), (20.0, 20.0)),
    score: float = 0.9,
) -> Labels:
    """Build a one-frame Labels with a single predicted instance on a named track.

    A fresh ``Skeleton`` and a fresh ``Track`` (distinct object identity, but the
    given ``name``) are created each call, mirroring two independently-loaded
    projects that share semantically-meaningful track names.
    """
    skeleton = Skeleton(["A", "B"])
    track = Track(name=track_name)
    labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track])
    lf = LabeledFrame(video=video, frame_idx=frame_idx)
    pred = PredictedInstance.from_numpy(
        np.array(coords, dtype="float64"), skeleton=skeleton, score=score, track=track
    )
    lf.instances.append(pred)
    labels.append(lf)
    return labels


def _unique_track_objs(labels: Labels) -> int:
    """Count distinct ``Track`` objects referenced by instances across all frames."""
    seen = set()
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            if inst.track is not None:
                seen.add(id(inst.track))
    return len(seen)


class TestLibraryTrackMatchingContract:
    """Documents the sleap-io 0.8.0 breaking change (both matcher options).

    Two Labels each contain a ``Track(name="track_0")`` with a predicted instance
    on the SAME frame of the SAME video, with identical coordinates (so no spatial
    divergence under the merge's instance matcher).
    """

    def test_default_identity_keeps_two_tracks(self):
        """WITHOUT track= the 0.8.0 default ("identity") does NOT collapse names."""
        video = Video(filename="shared_vid.mp4")
        base = _labels_with_named_track("track_0", video)
        new = _labels_with_named_track("track_0", video)

        # Sanity: distinct Track objects that happen to share a name.
        assert base.tracks[0] is not new.tracks[0]
        assert base.tracks[0].name == new.tracks[0].name == "track_0"

        base.merge(new, frame="keep_both")  # no track= -> identity default

        # Identity default treats the two same-named tracks as distinct.
        assert len(base.tracks) == 2
        assert [t.name for t in base.tracks] == ["track_0", "track_0"]

        # keep_both kept both instances on the single shared frame...
        assert len(base.labeled_frames) == 1
        frame = base.labeled_frames[0]
        assert len(frame.instances) == 2
        # ...and they reference two distinct track objects (not collapsed).
        assert _unique_track_objs(base) == 2

    def test_track_name_option_collapses_to_one_track(self):
        """WITH track="name" same-named tracks collapse (the opt-in alternative)."""
        video = Video(filename="shared_vid.mp4")
        base = _labels_with_named_track("track_0", video)
        new = _labels_with_named_track("track_0", video)

        base.merge(new, frame="keep_both", track="name")

        # Name matching collapses the two "track_0" tracks into one.
        assert len(base.tracks) == 1
        assert base.tracks[0].name == "track_0"

        # Both instances kept (keep_both), now sharing a single track object.
        frame = base.labeled_frames[0]
        assert len(frame.instances) == 2
        assert _unique_track_objs(base) == 1
        # The surviving track is the base project's original track object.
        assert frame.instances[0].track is base.tracks[0]
        assert frame.instances[1].track is base.tracks[0]


class TestMergeDialogTrackIdentity:
    """sleap.gui.dialogs.merge.MergeDialog pins track="identity".

    Exercises the same public path tests/gui/test_merge.py uses for the dialog:
    construct the dialog (runs ``_perform_merge_analysis`` on a deepcopy) then call
    ``_perform_final_merge`` (the committing path that mutates ``base_labels``).
    Same-named tracks from the two projects must stay DISTINCT (identity), matching
    original pre-sleap-io-port SLEAP behavior.
    """

    def test_final_merge_keeps_same_named_tracks_distinct(self, qtbot):
        from sleap.gui.dialogs.merge import MergeDialog

        video = Video(filename="dialog_vid.mp4")
        # Distinct frames so there are no overlapping-frame conflicts -- this
        # isolates the track-matching behavior.
        base = _labels_with_named_track("track_0", video, frame_idx=0)
        new = _labels_with_named_track("track_0", video, frame_idx=1)

        # Constructing the dialog runs _perform_merge_analysis on a *deepcopy*,
        # so base_labels itself is untouched until the final merge.
        dlg = MergeDialog(base_labels=base, new_labels=new)

        # No overlapping frames -> clean merge, no conflicts.
        assert dlg.conflicts == []
        # base_labels is unchanged by the analysis pass.
        assert len(base.tracks) == 1
        assert len(base.labeled_frames) == 1

        # Commit the merge (the path wired to the "Finish Merge" button).
        dlg._perform_final_merge()

        # Identity matching: the two same-named tracks stay distinct.
        assert len(base.tracks) == 2
        assert [t.name for t in base.tracks] == ["track_0", "track_0"]
        # Both frames present; instances reference two distinct track objects.
        assert len(base.labeled_frames) == 2
        assert _unique_track_objs(base) == 2

    def test_name_option_would_collapse(self, qtbot):
        """Counter-check: track="name" WOULD collapse the same inputs to one track.

        Confirms the dialog's ``track="identity"`` choice is load-bearing -- the
        alternative matcher produces the (now-unwanted) single-track collapse.
        """
        video = Video(filename="dialog_vid.mp4")
        base = _labels_with_named_track("track_0", video, frame_idx=0)
        new = _labels_with_named_track("track_0", video, frame_idx=1)

        base.merge(new, frame="keep_both", track="name")  # the opt-in alternative

        assert len(base.tracks) == 1
        assert _unique_track_objs(base) == 1


class TestInferenceResultTrackIdentity:
    """InferenceTask.merge_results pins track="identity" for the post-inference path.

    Mirrors tests/gui/test_merge.py::TestInferenceResultMerging construction: build
    an InferenceTask with ``trained_job_paths=[]`` and a ``results`` list of
    LabeledFrames, then call ``merge_results()``. Same-named predicted tracks from a
    separate inference run must stay DISTINCT from the project's tracks (identity).
    """

    def test_merge_results_keeps_same_named_track_distinct(self):
        from sleap.gui.learning.runners import InferenceTask

        video = Video(filename="hitl_vid.mp4")
        # Existing project: one predicted instance on Track(name="track_0").
        labels = _labels_with_named_track("track_0", video, frame_idx=0)
        existing_track = labels.tracks[0]
        assert len(labels.tracks) == 1

        # Inference results: a NEW Track object with the SAME name on a new frame
        # (the typical re-run-tracking-and-merge-back HITL scenario).
        results_track = Track(name="track_0")
        skeleton = labels.skeleton
        res_lf = LabeledFrame(video=video, frame_idx=1)
        res_pred = PredictedInstance.from_numpy(
            np.array([[50.0, 50.0], [55.0, 55.0]]),
            skeleton=skeleton,
            score=0.95,
            track=results_track,
        )
        res_lf.instances.append(res_pred)
        assert results_track is not existing_track

        task = InferenceTask(
            trained_job_paths=[],
            labels=labels,
            results=[res_lf],
            inference_params={"_prediction_mode": "add"},
        )
        task.merge_results()

        # Identity matching: the same-named predicted track is kept as a distinct
        # track rather than collapsed into the project's existing one.
        assert len(labels.tracks) == 2
        assert [t.name for t in labels.tracks] == ["track_0", "track_0"]
        assert _unique_track_objs(labels) == 2
        # New predictions landed on a new frame on a track that is NOT the existing
        # project track (a separate identity).
        assert len(labels.labeled_frames) == 2
        new_frame = labels.find(video, frame_idx=1)[0]
        assert len(new_frame.instances) == 1
        assert new_frame.instances[0].track is not existing_track

    def test_name_option_would_collapse(self):
        """Counter-check: track="name" would collapse to a single "track_0".

        Confirms the runners.py ``track="identity"`` choice is load-bearing.
        Reproduced at the library level on equivalent inputs.
        """
        video = Video(filename="hitl_vid.mp4")
        labels = _labels_with_named_track("track_0", video, frame_idx=0)

        results_track = Track(name="track_0")
        skeleton = labels.skeleton
        res_lf = LabeledFrame(video=video, frame_idx=1)
        res_pred = PredictedInstance.from_numpy(
            np.array([[50.0, 50.0], [55.0, 55.0]]),
            skeleton=skeleton,
            score=0.95,
            track=results_track,
        )
        res_lf.instances.append(res_pred)
        new_labels = Labels([res_lf])

        labels.merge(new_labels, frame="keep_both", track="name")
        assert len(labels.tracks) == 1
        assert _unique_track_objs(labels) == 1

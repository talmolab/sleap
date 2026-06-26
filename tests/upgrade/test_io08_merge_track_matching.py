"""Tier 0.5 regression tests: sleap-io 0.8.0 track-matching default flip.

sleap-io 0.8.0 (talmolab/sleap-io#449) flipped the default track matcher of
``Labels.merge()`` / ``Labels.match()`` from ``"name"`` to ``"identity"``:

- ``"identity"`` (the new default) matches tracks ONLY by Python object identity
  (the same ``Track`` instance). Two distinct ``Track`` objects that happen to
  share a name are kept as separate tracks -- a correctness-first default that
  never collapses distinct, tracker-assigned identities.
- ``"name"`` (the pre-0.8.0 behavior) matches tracks by their ``name`` attribute,
  collapsing same-named tracks across the two projects into one.

The SLEAP Tier 0 fix pins ``track="name"`` on every ``Labels.merge()`` call site
in the GUI (``gui/learning/runners.py``, ``gui/dialogs/merge.py``,
``gui/commands.py``) to preserve the 1.6.x "merge same-named tracks" behavior
that identity/ID-classification and HITL workflows rely on.

These tests LOCK IN that behavior: they must pass against the current (fixed)
worktree. If a future change drops the ``track="name"`` pin, the dialog/HITL
tests here will regress to duplicated tracks and fail.
"""

import warnings

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
    """Documents the sleap-io 0.8.0 breaking change + the SLEAP fix rationale.

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

    def test_track_name_collapses_to_one_track(self):
        """WITH track="name" same-named tracks collapse (the Tier 0 fix behavior)."""
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


class TestMergeDialogTrackCollapse:
    """sleap.gui.dialogs.merge.MergeDialog must pin track="name" (Tier 0 fix).

    Exercises the same public path as tests/gui/test_merge.py would for the
    dialog: construct the dialog (runs ``_perform_merge_analysis`` on a deepcopy)
    then call ``_perform_final_merge`` (the committing path that mutates
    ``base_labels``). Same-named tracks from the two projects must collapse to a
    single track rather than duplicate.
    """

    def test_final_merge_collapses_same_named_tracks(self, qtbot):
        from sleap.gui.dialogs.merge import MergeDialog

        video = Video(filename="dialog_vid.mp4")
        # Distinct frames so there are no overlapping-frame conflicts and no
        # spatial-divergence warning -- this isolates the track-collapse behavior.
        base = _labels_with_named_track("track_0", video, frame_idx=0)
        new = _labels_with_named_track("track_0", video, frame_idx=1)

        # Constructing the dialog runs _perform_merge_analysis on a *deepcopy*,
        # so base_labels itself is untouched until the final merge.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # name-merge divergence would error here
            dlg = MergeDialog(base_labels=base, new_labels=new)

        # No overlapping frames -> clean merge, no conflicts.
        assert dlg.conflicts == []
        # base_labels is unchanged by the analysis pass.
        assert len(base.tracks) == 1
        assert len(base.labeled_frames) == 1

        # Commit the merge (the path wired to the "Finish Merge" button).
        dlg._perform_final_merge()

        # Same-named tracks collapsed: still exactly one "track_0".
        assert len(base.tracks) == 1
        assert base.tracks[0].name == "track_0"
        # Both frames present, all instances share the single track object.
        assert len(base.labeled_frames) == 2
        assert _unique_track_objs(base) == 1

    def test_final_merge_without_name_fix_would_duplicate(self, qtbot):
        """Counter-check: identity matching (the 0.8.0 default) WOULD duplicate.

        This is the regression the dialog's ``track="name"`` pin prevents. We
        reproduce the un-pinned merge directly on the dialog's inputs to show the
        fix is load-bearing: same inputs, identity default -> 2 tracks.
        """
        video = Video(filename="dialog_vid.mp4")
        base = _labels_with_named_track("track_0", video, frame_idx=0)
        new = _labels_with_named_track("track_0", video, frame_idx=1)

        base.merge(new, frame="keep_both")  # identity default, as if unpinned

        assert len(base.tracks) == 2
        assert _unique_track_objs(base) == 2


class TestInferenceResultTrackMatching:
    """InferenceTask.merge_results pins track="name" for the post-inference HITL path.

    Mirrors tests/gui/test_merge.py::TestInferenceResultMerging construction:
    build an InferenceTask with ``trained_job_paths=[]`` and a ``results`` list of
    LabeledFrames, then call ``merge_results()``. Same-named predicted tracks must
    merge into the project's existing same-named track instead of duplicating.
    """

    def test_merge_results_collapses_same_named_track(self):
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

        # The same-named predicted track merged into the project's existing track
        # instead of accumulating a duplicate.
        assert len(labels.tracks) == 1
        assert labels.tracks[0].name == "track_0"
        assert _unique_track_objs(labels) == 1
        # New predictions landed on a new frame, attached to the surviving track.
        assert len(labels.labeled_frames) == 2
        new_frame = labels.find(video, frame_idx=1)[0]
        assert len(new_frame.instances) == 1
        assert new_frame.instances[0].track is labels.tracks[0]

    def test_merge_results_without_name_fix_would_duplicate(self):
        """Counter-check: identity default would create a second "track_0".

        Documents the regression the runners.py ``track="name"`` pin prevents.
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

        # Identity default (no track=) -> the new "track_0" is appended as distinct.
        labels.merge(new_labels, frame="keep_both")
        assert len(labels.tracks) == 2
        assert _unique_track_objs(labels) == 2

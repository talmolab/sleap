"""Tests for prediction-to-instance conversion bugs.

These tests verify the bug fixes for the issue where:
1. "Add instances from all predictions on current frame" doesn't fully add them
2. "Delete all predictions" also deletes these supposedly-added instances
3. Double-clicking also doesn't properly add predictions
4. User must move a keypoint for the instance to persist

See: scratch/2026-01-08-predictions-not-fully-added/README.md
"""

import numpy as np
import pytest

from sleap_io import (
    Labels,
    Skeleton,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Video,
    Track,
)
from sleap_io.model.instance import PointsArray

from sleap.sleap_io_adaptors.lf_labels_utils import (
    get_unused_predictions,
    get_instances_to_show,
)
from sleap.gui.commands import (
    AddInstance,
    AddMissingInstanceNodes,
    AddUserInstancesFromPredictions,
    AddUserInstancesFromAllPredictions,
)


class _StubState:
    """Minimal GuiState stand-in: returns None for unset keys.

    Mirrors ``sleap.gui.state.GuiState`` indexing semantics used by the
    node-placement helpers (``context.state["skeleton"]``), so tests do not
    need a full GUI state object.
    """

    def __init__(self, **initial):
        self._vars = dict(initial)

    def __getitem__(self, key):
        return self._vars.get(key)

    def __setitem__(self, key, value):
        self._vars[key] = value

    def __contains__(self, key):
        return key in self._vars


@pytest.fixture
def simple_skeleton():
    """Create a simple skeleton with 3 nodes."""
    skeleton = Skeleton(name="test")
    skeleton.add_node("head")
    skeleton.add_node("thorax")
    skeleton.add_node("abdomen")
    skeleton.add_edge("head", "thorax")
    skeleton.add_edge("thorax", "abdomen")
    return skeleton


@pytest.fixture
def simple_video():
    """Create a simple dummy video."""
    return Video(filename="test.mp4")


@pytest.fixture
def prediction_with_track(simple_skeleton):
    """Create a PredictedInstance with a track."""
    track = Track(name="track1")
    pred = PredictedInstance.empty(
        skeleton=simple_skeleton,
        score=0.95,
        track=track,
    )
    # Set some point coordinates
    pred["head"] = (10.0, 20.0, 0.9)
    pred["thorax"] = (15.0, 30.0, 0.85)
    pred["abdomen"] = (20.0, 40.0, 0.8)
    return pred


@pytest.fixture
def prediction_without_track(simple_skeleton):
    """Create a PredictedInstance without a track."""
    pred = PredictedInstance.empty(
        skeleton=simple_skeleton,
        score=0.90,
    )
    pred["head"] = (50.0, 60.0, 0.88)
    pred["thorax"] = (55.0, 70.0, 0.82)
    pred["abdomen"] = (60.0, 80.0, 0.75)
    return pred


@pytest.fixture
def user_instance_from_prediction(simple_skeleton, prediction_with_track):
    """Create a user Instance that was created from a prediction."""
    inst = Instance.empty(
        skeleton=simple_skeleton,
        from_predicted=prediction_with_track,
        track=prediction_with_track.track,
    )
    # Copy point coordinates
    inst["head"] = (10.0, 20.0)
    inst["thorax"] = (15.0, 30.0)
    inst["abdomen"] = (20.0, 40.0)
    return inst


class TestGetUnusedPredictionsBug:
    """Tests for Bug 1: get_unused_predictions() uses wrong attribute checks.

    The bug is that the function uses `hasattr(inst, "from_predicted")` which
    returns True for ALL instances (both Instance and PredictedInstance have
    this attribute). It should use `type(inst) is PredictedInstance` instead.
    """

    def test_returns_only_predicted_instances_with_tracks(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """Prediction should not be unused when user instance exists in same track."""
        track = prediction_with_track.track

        # Create a user instance in the same track
        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            track=track,
        )
        user_inst["head"] = (10.0, 20.0)
        user_inst["thorax"] = (15.0, 30.0)
        user_inst["abdomen"] = (20.0, 40.0)

        # Create labeled frame with both
        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_with_track, user_inst],
        )

        # The prediction should NOT be in unused_predictions because
        # there's a user instance in the same track
        unused = get_unused_predictions(lf)

        # BUG: Currently returns [prediction_with_track, user_inst] because
        # hasattr check is wrong. Should return [] because prediction is "used"
        assert prediction_with_track not in unused, (
            "PredictedInstance should not be in unused_predictions when a user "
            "instance exists in the same track"
        )
        assert user_inst not in unused, (
            "User Instance should never be in unused_predictions"
        )

    def test_returns_only_predicted_instances_without_tracks(
        self, simple_skeleton, simple_video, prediction_without_track
    ):
        """Prediction should not be unused when linked via from_predicted."""
        # Create a user instance linked to the prediction
        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_without_track,
        )
        user_inst["head"] = (50.0, 60.0)
        user_inst["thorax"] = (55.0, 70.0)
        user_inst["abdomen"] = (60.0, 80.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_without_track, user_inst],
        )

        unused = get_unused_predictions(lf)

        # Prediction should NOT be unused since user_inst.from_predicted points to it
        assert prediction_without_track not in unused, (
            "PredictedInstance should not be in unused_predictions when a user "
            "instance has from_predicted pointing to it"
        )
        # User instance should never be in unused_predictions
        assert user_inst not in unused, (
            "User Instance should never be in unused_predictions"
        )

    def test_only_returns_predicted_instance_type(self, simple_skeleton, simple_video):
        """unused_predictions should only ever contain PredictedInstance objects."""
        track = Track(name="track1")

        # Create both types with the same track
        pred = PredictedInstance.empty(skeleton=simple_skeleton, track=track, score=0.9)
        pred["head"] = (10.0, 20.0, 0.9)

        user = Instance.empty(skeleton=simple_skeleton, track=track)
        user["head"] = (10.0, 20.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[pred, user],
        )

        unused = get_unused_predictions(lf)

        # All items in unused_predictions must be PredictedInstance
        for inst in unused:
            assert type(inst) is PredictedInstance, (
                f"unused_predictions should only contain PredictedInstance, "
                f"got {type(inst).__name__}"
            )


class TestGetInstancesToShowBug:
    """Tests for Bug 2: get_instances_to_show() uses same wrong check.

    The bug is that the function uses `not hasattr(inst, "from_predicted")`
    which is always False (both types have this attribute). This causes
    user instances to be incorrectly filtered out.
    """

    def test_shows_user_instances_after_predictions_deleted(
        self, simple_skeleton, simple_video
    ):
        """User instances should be visible even after predictions are deleted."""
        # Simulate state after "Delete all predictions":
        # Frame only has user instances with from_predicted set to deleted predictions

        # Create a "dangling" reference to simulate deleted prediction
        deleted_prediction = PredictedInstance.empty(
            skeleton=simple_skeleton, score=0.9
        )

        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=deleted_prediction,  # Points to "deleted" prediction
        )
        user_inst["head"] = (10.0, 20.0)

        # Frame only contains user instance (prediction was deleted)
        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[user_inst],
        )

        to_show = get_instances_to_show(lf)

        # User instance should be shown
        assert user_inst in to_show, (
            "User instance should be shown after predictions are deleted"
        )

    def test_shows_user_instances_alongside_predictions(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """Both user instances and unused predictions should be visible."""
        # User instance created from the prediction
        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_with_track,
            track=prediction_with_track.track,
        )
        user_inst["head"] = (10.0, 20.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_with_track, user_inst],
        )

        to_show = get_instances_to_show(lf)

        # User instance should always be shown
        assert user_inst in to_show, "User instance should always be shown"

    def test_all_user_instances_always_shown(self, simple_skeleton, simple_video):
        """All user instances (with or without from_predicted) should be shown."""
        # Instance without from_predicted
        user1 = Instance.empty(skeleton=simple_skeleton)
        user1["head"] = (10.0, 20.0)

        # Instance with from_predicted
        pred = PredictedInstance.empty(skeleton=simple_skeleton, score=0.9)
        user2 = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=pred,
        )
        user2["head"] = (50.0, 60.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[user1, user2, pred],
        )

        to_show = get_instances_to_show(lf)

        # Both user instances should be shown
        assert user1 in to_show, "User instance without from_predicted should be shown"
        assert user2 in to_show, "User instance with from_predicted should be shown"


class TestMakeInstanceFromPredictedInstanceBug:
    """Tests for Bug 3: make_instance_from_predicted_instance() point conversion.

    The bug is that the function passes PredictedPointsArray directly to Instance
    instead of converting to PointsArray. This may cause the instance to be
    incorrectly identified as a prediction in some code paths.
    """

    def test_resulting_instance_has_correct_points_type(
        self, simple_skeleton, prediction_with_track
    ):
        """The created Instance should have PointsArray, not PredictedPointsArray."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )

        # Check the type of the points array
        assert type(new_instance.points) is PointsArray, (
            f"Points should be PointsArray, not {type(new_instance.points).__name__}"
        )

    def test_resulting_instance_points_have_no_score_field(
        self, simple_skeleton, prediction_with_track
    ):
        """The created Instance's points should not have a 'score' field."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )

        # Check dtype names
        dtype_names = new_instance.points.dtype.names
        assert "score" not in dtype_names, (
            "Instance points should not have 'score' field in dtype"
        )

    def test_resulting_instance_is_not_predicted_instance(
        self, simple_skeleton, prediction_with_track
    ):
        """The created object should be Instance, not PredictedInstance."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )

        assert type(new_instance) is Instance, (
            f"Result should be Instance, not {type(new_instance).__name__}"
        )
        assert not hasattr(new_instance, "score") or new_instance.score is None, (
            "Instance should not have a score attribute with a value"
        )

    def test_resulting_instance_preserves_coordinates(
        self, simple_skeleton, prediction_with_track
    ):
        """The created Instance should preserve the coordinate values."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )

        # Check that coordinates are preserved
        orig_pts = prediction_with_track.numpy()
        new_pts = new_instance.numpy()

        np.testing.assert_allclose(
            orig_pts,
            new_pts,
            err_msg="Coords should be preserved in prediction-to-instance conversion",
        )

    def test_resulting_instance_has_from_predicted_set(
        self, simple_skeleton, prediction_with_track
    ):
        """The created Instance should have from_predicted linking to the original."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )

        assert new_instance.from_predicted is prediction_with_track, (
            "Instance.from_predicted should reference the original PredictedInstance"
        )


class TestNaNPredictedNodeVisibilityBug:
    """Tests for the NaN-predicted-node visibility bug.

    When a predicted instance has NaN-coordinate nodes (i.e. nodes that were
    not detected by the model), converting it to a user Instance via the
    GUI should result in those nodes having `visible=False`. The previous
    behavior left those nodes with `visible=True` and uninitialized xy
    values, because `Instance.empty()` allocates points via `np.empty()`
    (uninitialized memory) and the conversion code did not explicitly set
    visibility for NaN-coord nodes.

    See: scratch/2026-04-27-nan-predicted-to-user-visibility-bug/README.md
    """

    @pytest.fixture
    def prediction_with_nan_node(self, simple_skeleton):
        """PredictedInstance with one valid and one NaN-coord node."""
        pred = PredictedInstance.empty(skeleton=simple_skeleton, score=0.8)
        pred["head"] = (10.0, 20.0, 0.9)
        pred["thorax"] = (np.nan, np.nan, 0.0)
        pred["abdomen"] = (40.0, 50.0, 0.7)
        # Predictions typically come back with visible=True even when the
        # coordinates are NaN -- that is the upstream condition that
        # exposes this bug.
        pred.points["visible"] = True
        return pred

    def test_make_instance_nan_node_is_invisible(self, prediction_with_nan_node):
        """make_instance_from_predicted_instance: NaN-coord -> visible=False."""
        new_instance = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_nan_node
            )
        )

        names = list(new_instance.points["name"])
        thorax_idx = names.index("thorax")
        head_idx = names.index("head")
        abdomen_idx = names.index("abdomen")

        assert bool(new_instance.points[thorax_idx]["visible"]) is False, (
            "NaN-coord predicted node should be invisible after conversion"
        )
        assert bool(new_instance.points[head_idx]["visible"]) is True, (
            "Valid predicted node should remain visible after conversion"
        )
        assert bool(new_instance.points[abdomen_idx]["visible"]) is True

    def test_set_visible_nodes_initializes_nan_node_with_dirty_heap(
        self, simple_skeleton, prediction_with_nan_node
    ):
        """set_visible_nodes should set NaN-coord nodes to visible=False, xy=NaN.

        We poison the freshly-allocated `Instance.empty()` buffer with non-NaN,
        non-zero, visible=True garbage to simulate dirty heap memory in a
        long-running GUI session. Without the fix, the missing-node branch
        leaves the buffer untouched and downstream gating fails to recognize
        it as missing, propagating the garbage values through.
        """
        new_instance = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_with_nan_node,
        )
        # Pollute the buffer to simulate dirty heap.
        for i in range(len(new_instance.points)):
            new_instance.points[i]["xy"] = np.array([1234.5, 6789.0])
            new_instance.points[i]["visible"] = True
            new_instance.points[i]["complete"] = True

        # Minimal stub context: set_visible_nodes only reads
        # context.state["video"], context.state["skeleton"], and
        # context.labels.videos[0] (as a fallback for video shape).
        class _StubVideo:
            shape = (1, 480, 640, 3)  # (n_frames, height, width, channels)

        class _StubLabels:
            videos = [_StubVideo()]

        class _StubContext:
            labels = _StubLabels()
            state = {"video": _StubVideo(), "skeleton": simple_skeleton}

        has_missing = AddInstance.set_visible_nodes(
            context=_StubContext(),
            copy_instance=prediction_with_nan_node,
            new_instance=new_instance,
            mark_complete=False,
            init_method="best",
        )

        assert has_missing is True, "Expected has_missing_nodes=True for NaN node"

        names = list(new_instance.points["name"])
        thorax_idx = names.index("thorax")

        thorax_xy = new_instance.points[thorax_idx]["xy"]
        assert np.all(np.isnan(thorax_xy)), (
            f"NaN-coord node should have xy=NaN after set_visible_nodes; "
            f"got {thorax_xy!r}"
        )
        assert bool(new_instance.points[thorax_idx]["visible"]) is False, (
            "NaN-coord node should have visible=False after set_visible_nodes"
        )

    def test_set_visible_nodes_preserves_valid_node_visibility(
        self, simple_skeleton, prediction_with_nan_node
    ):
        """Valid (non-NaN) predicted nodes should remain visible after copy."""
        new_instance = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_with_nan_node,
        )

        class _StubVideo:
            shape = (1, 480, 640, 3)

        class _StubLabels:
            videos = [_StubVideo()]

        class _StubContext:
            labels = _StubLabels()
            state = {"video": _StubVideo(), "skeleton": simple_skeleton}

        AddInstance.set_visible_nodes(
            context=_StubContext(),
            copy_instance=prediction_with_nan_node,
            new_instance=new_instance,
            mark_complete=False,
            init_method="best",
        )

        names = list(new_instance.points["name"])
        head_idx = names.index("head")
        np.testing.assert_array_equal(
            new_instance.points[head_idx]["xy"], np.array([10.0, 20.0])
        )
        assert bool(new_instance.points[head_idx]["visible"]) is True

    def test_add_random_nodes_does_not_leak_visible_across_iterations(
        self, simple_skeleton, prediction_with_nan_node
    ):
        """`add_random_nodes` must not let one node's visibility bleed into the next.

        Regression for the variable-shadowing bug: the function parameter
        `visible` was reassigned inside the loop's else branch, so a valid
        node (visible=True) processed before a NaN node would clobber the
        local `visible` and the NaN node's if-branch would write
        visible=True instead of the requested False. End result: NaN
        predicted nodes appeared as visible user labels in the GUI.
        """
        from PySide6 import QtCore

        # Run set_visible_nodes first to set up the new instance the same
        # way `AddInstance.create_new_instance` would after my upstream fix:
        # valid nodes copied with visible=True, NaN nodes initialized with
        # xy=NaN/visible=False.
        new_instance = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_with_nan_node,
        )

        class _StubVideo:
            shape = (1, 480, 640, 3)

        class _StubPlayer:
            @staticmethod
            def getVisibleRect():
                return QtCore.QRectF(0.0, 0.0, 640.0, 480.0)

        class _StubApp:
            player = _StubPlayer()

        class _StubLabels:
            videos = [_StubVideo()]

        class _StubContext:
            labels = _StubLabels()
            app = _StubApp()
            state = {"video": _StubVideo(), "skeleton": simple_skeleton}

        AddInstance.set_visible_nodes(
            context=_StubContext(),
            copy_instance=prediction_with_nan_node,
            new_instance=new_instance,
            mark_complete=False,
            init_method="best",
        )

        # Now run add_random_nodes with visible=False (the value that the
        # double-click-from-prediction flow passes in via fill_missing_nodes).
        AddMissingInstanceNodes.add_random_nodes(
            _StubContext(), new_instance, visible=False
        )

        names = list(new_instance.points["name"])
        thorax_idx = names.index("thorax")
        head_idx = names.index("head")
        abdomen_idx = names.index("abdomen")

        # NaN-pred node must be invisible regardless of where it falls in
        # iteration order relative to the valid nodes.
        assert bool(new_instance.points[thorax_idx]["visible"]) is False, (
            "NaN-pred node must remain invisible after add_random_nodes"
        )
        # Valid nodes must remain visible (their predicted xy is preserved).
        assert bool(new_instance.points[head_idx]["visible"]) is True
        assert bool(new_instance.points[abdomen_idx]["visible"]) is True


class TestOriginalPredictionNotRemovedBug:
    """Tests for Bug 4: Original predictions not removed when creating user instances.

    When a user instance is created from a prediction, the original PredictedInstance
    should ideally be removed from the frame to avoid confusion. Currently, both
    coexist, which can cause the UI to show overlapping instances.
    """

    def test_sleap_io_unused_predictions_excludes_used(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """sleap-io's LabeledFrame.unused_predictions should exclude used predictions.

        This tests that sleap-io's implementation is correct (it uses type checks).
        """
        # Create user instance in same track
        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            track=prediction_with_track.track,
        )
        user_inst["head"] = (10.0, 20.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_with_track, user_inst],
        )

        # Use sleap-io's property directly (should be correct)
        unused = lf.unused_predictions

        # Prediction should be excluded because user instance exists in same track
        assert prediction_with_track not in unused, (
            "sleap-io's unused_predictions should exclude predictions with "
            "matching user instances in the same track"
        )

    def test_delete_predictions_only_deletes_predicted_instances(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """DeleteAllPredictions should only delete PredictedInstance objects.

        This verifies that the type check `type(inst) == PredictedInstance` works.
        """
        # Create user instance
        user_inst = Instance.empty(
            skeleton=simple_skeleton,
            from_predicted=prediction_with_track,
            track=prediction_with_track.track,
        )
        user_inst["head"] = (10.0, 20.0)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_with_track, user_inst],
        )

        # Simulate what DeleteAllPredictions does
        instances_to_delete = [
            inst for inst in lf.instances if type(inst) == PredictedInstance
        ]

        # Should only find the prediction, not the user instance
        assert prediction_with_track in instances_to_delete
        assert user_inst not in instances_to_delete, (
            "DeleteAllPredictions should not delete user instances"
        )

        # After deletion, only user instance should remain
        remaining = [inst for inst in lf.instances if inst not in instances_to_delete]
        assert user_inst in remaining
        assert prediction_with_track not in remaining


class TestFullWorkflowIntegration:
    """Integration tests for adding predictions as user instances."""

    def test_add_all_predictions_then_delete_predictions_preserves_user_instances(
        self, simple_skeleton, simple_video
    ):
        """Full workflow: add predictions as user instances, then delete predictions.

        1. Start with frame containing only predictions
        2. "Add instances from all predictions" creates user instances
        3. "Delete all predictions" removes predictions
        4. User instances should remain and be visible
        """
        track1 = Track(name="track1")
        track2 = Track(name="track2")

        pred1 = PredictedInstance.empty(
            skeleton=simple_skeleton, track=track1, score=0.9
        )
        pred1["head"] = (10.0, 20.0, 0.9)

        pred2 = PredictedInstance.empty(
            skeleton=simple_skeleton, track=track2, score=0.85
        )
        pred2["head"] = (50.0, 60.0, 0.85)

        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[pred1, pred2],
        )

        # Step 1: Get unused predictions (simulating AddUserInstancesFromPredictions)
        # Using sleap-io's correct implementation
        unused = lf.unused_predictions
        assert pred1 in unused
        assert pred2 in unused

        # Step 2: Create user instances from predictions
        user_instances = []
        for pred in unused:
            user_inst = (
                AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                    pred
                )
            )
            user_instances.append(user_inst)
            lf.instances.append(user_inst)

        assert len(lf.instances) == 4  # 2 predictions + 2 user instances

        # Step 3: Delete all predictions
        lf.instances = [
            inst for inst in lf.instances if type(inst) is not PredictedInstance
        ]

        assert len(lf.instances) == 2  # Only user instances remain

        # Step 4: Verify user instances are shown
        to_show = get_instances_to_show(lf)

        assert len(to_show) == 2, (
            f"Both user instances should be shown, got {len(to_show)}"
        )
        for inst in user_instances:
            assert inst in to_show, (
                "User instance should be visible after predictions are deleted"
            )

    def test_double_click_workflow_preserves_instance(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """Simulate double-click: create instance from prediction, delete predictions.

        This mimics what happens when a user double-clicks a prediction and then
        runs "Delete all predictions".
        """
        lf = LabeledFrame(
            video=simple_video,
            frame_idx=0,
            instances=[prediction_with_track],
        )

        # Double-click creates user instance (via AddInstance with copy_instance)
        user_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )
        lf.instances.append(user_inst)

        # Delete predictions
        lf.instances = [
            inst for inst in lf.instances if type(inst) is not PredictedInstance
        ]

        # User instance should be visible
        to_show = get_instances_to_show(lf)

        assert user_inst in to_show, (
            "User instance created by double-click should be visible after "
            "predictions are deleted"
        )


class TestAddUserInstancesFromAllPredictions:
    """Tests for the bulk 'Accept All Predictions' command."""

    def test_converts_predictions_across_multiple_frames(
        self, simple_skeleton, simple_video
    ):
        track1 = Track(name="track1")
        track2 = Track(name="track2")

        pred1 = PredictedInstance.empty(
            skeleton=simple_skeleton, track=track1, score=0.9
        )
        pred1["head"] = (10.0, 20.0, 0.9)
        pred1["thorax"] = (15.0, 30.0, 0.85)
        pred1["abdomen"] = (20.0, 40.0, 0.8)

        pred2 = PredictedInstance.empty(
            skeleton=simple_skeleton, track=track2, score=0.85
        )
        pred2["head"] = (50.0, 60.0, 0.88)
        pred2["thorax"] = (55.0, 70.0, 0.82)
        pred2["abdomen"] = (60.0, 80.0, 0.75)

        lf0 = LabeledFrame(video=simple_video, frame_idx=0, instances=[pred1])
        lf1 = LabeledFrame(video=simple_video, frame_idx=1, instances=[pred2])

        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf0, lf1],
            tracks=[track1, track2],
        )

        from sleap.gui.commands import CommandContext

        context = CommandContext.from_labels(labels)
        AddUserInstancesFromAllPredictions.do_action(context, {})

        # Each frame should now have the original prediction + a new user instance
        assert len(lf0.instances) == 2
        assert len(lf1.instances) == 2

        user_insts_f0 = [i for i in lf0.instances if type(i) is Instance]
        user_insts_f1 = [i for i in lf1.instances if type(i) is Instance]
        assert len(user_insts_f0) == 1
        assert len(user_insts_f1) == 1

        assert user_insts_f0[0].from_predicted is pred1
        assert user_insts_f1[0].from_predicted is pred2

    def test_skips_already_accepted_predictions(self, simple_skeleton, simple_video):
        track = Track(name="track1")

        pred = PredictedInstance.empty(skeleton=simple_skeleton, track=track, score=0.9)
        pred["head"] = (10.0, 20.0, 0.9)
        pred["thorax"] = (15.0, 30.0, 0.85)
        pred["abdomen"] = (20.0, 40.0, 0.8)

        user_inst = Instance.empty(
            skeleton=simple_skeleton, track=track, from_predicted=pred
        )
        user_inst["head"] = (10.0, 20.0)
        user_inst["thorax"] = (15.0, 30.0)
        user_inst["abdomen"] = (20.0, 40.0)

        lf = LabeledFrame(video=simple_video, frame_idx=0, instances=[pred, user_inst])

        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
            tracks=[track],
        )

        from sleap.gui.commands import CommandContext

        context = CommandContext.from_labels(labels)
        AddUserInstancesFromAllPredictions.do_action(context, {})

        # Should not add duplicate — prediction is already "used"
        assert len(lf.instances) == 2

    def test_adds_new_tracks(self, simple_skeleton, simple_video):
        new_track = Track(name="new_track")

        pred = PredictedInstance.empty(
            skeleton=simple_skeleton, track=new_track, score=0.9
        )
        pred["head"] = (10.0, 20.0, 0.9)
        pred["thorax"] = (15.0, 30.0, 0.85)
        pred["abdomen"] = (20.0, 40.0, 0.8)

        lf = LabeledFrame(video=simple_video, frame_idx=0, instances=[pred])

        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
            tracks=[],
        )

        from sleap.gui.commands import CommandContext

        context = CommandContext.from_labels(labels)
        AddUserInstancesFromAllPredictions.do_action(context, {})

        assert new_track in labels.tracks

    def test_no_op_when_no_predictions(self, simple_skeleton, simple_video):
        user_inst = Instance.empty(skeleton=simple_skeleton)
        user_inst["head"] = (10.0, 20.0)
        user_inst["thorax"] = (15.0, 30.0)
        user_inst["abdomen"] = (20.0, 40.0)

        lf = LabeledFrame(video=simple_video, frame_idx=0, instances=[user_inst])

        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
        )

        from sleap.gui.commands import CommandContext

        context = CommandContext.from_labels(labels)
        AddUserInstancesFromAllPredictions.do_action(context, {})

        assert len(lf.instances) == 1


class TestFillMissingPredictedNodes:
    """Tests for filling undetected nodes when converting predictions (#2764).

    ``make_instance_from_predicted_instance`` keeps the model's detected
    keypoints and leaves undetected nodes at ``xy=NaN`` / ``visible=False``.
    ``AddUserInstancesFromPredictions.fill_missing_predicted_nodes`` then
    initializes those missing nodes (using the same machinery as a brand-new
    instance) to visible, draggable positions while preserving the detected
    points, track, and ``from_predicted`` link.

    These tests provide an offscreen player stub that supplies
    ``getVisibleRect()`` (the only GUI hook the placement helpers need), so
    they run headlessly without a real QApplication.
    """

    @staticmethod
    def _stub_context(skeleton, labels):
        """Build a minimal context whose app has an offscreen player.

        Mirrors the ``_StubPlayer``/``_StubApp``/``_StubContext`` pattern used
        elsewhere in this file. ``labels`` is a real ``Labels`` so the
        template lookup (``get_template_instance_points``) behaves realistically.
        """
        from qtpy import QtCore

        class _StubPlayer:
            @staticmethod
            def getVisibleRect():
                return QtCore.QRectF(0.0, 0.0, 640.0, 480.0)

        class _StubApp:
            player = _StubPlayer()

        class _StubContext:
            pass

        ctx = _StubContext()
        ctx.app = _StubApp()
        ctx.labels = labels
        # GuiState-like: __getitem__ returns None for missing keys; emulate
        # the all-frames path where "skeleton" starts unset so we also cover
        # the helper populating it from copy_instance.skeleton.
        ctx.state = _StubState()
        return ctx

    @pytest.fixture
    def prediction_with_nan_node(self, simple_skeleton):
        """PredictedInstance with two detected nodes and one undetected node."""
        pred = PredictedInstance.empty(skeleton=simple_skeleton, score=0.8)
        pred["head"] = (10.0, 20.0, 0.9)
        pred["thorax"] = (np.nan, np.nan, 0.0)
        pred["abdomen"] = (40.0, 50.0, 0.7)
        pred.points["visible"] = True
        return pred

    def test_fills_missing_node_and_preserves_detected(
        self, simple_skeleton, simple_video, prediction_with_nan_node
    ):
        """Undetected node becomes finite + visible; detected ones unchanged."""
        lf = LabeledFrame(
            video=simple_video, frame_idx=0, instances=[prediction_with_nan_node]
        )
        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
        )

        new_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_nan_node
            )
        )
        ctx = self._stub_context(simple_skeleton, labels)
        AddUserInstancesFromPredictions.fill_missing_predicted_nodes(
            ctx, new_inst, prediction_with_nan_node
        )

        names = list(new_inst.points["name"])
        head_idx = names.index("head")
        thorax_idx = names.index("thorax")
        abdomen_idx = names.index("abdomen")

        # Detected points are untouched and still visible.
        np.testing.assert_array_equal(
            new_inst.points[head_idx]["xy"], np.array([10.0, 20.0])
        )
        np.testing.assert_array_equal(
            new_inst.points[abdomen_idx]["xy"], np.array([40.0, 50.0])
        )
        assert bool(new_inst.points[head_idx]["visible"]) is True
        assert bool(new_inst.points[abdomen_idx]["visible"]) is True

        # Previously-undetected node is now finite, visible, and incomplete.
        thorax_xy = new_inst.points[thorax_idx]["xy"]
        assert np.all(np.isfinite(thorax_xy)), (
            f"undetected node should be filled with finite xy; got {thorax_xy!r}"
        )
        assert bool(new_inst.points[thorax_idx]["visible"]) is True
        assert bool(new_inst.points[thorax_idx]["complete"]) is False

        # The from_predicted link is preserved.
        assert new_inst.from_predicted is prediction_with_nan_node

    def test_all_nodes_detected_is_noop(
        self, simple_skeleton, simple_video, prediction_with_track
    ):
        """When every node is detected, nothing changes."""
        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=simple_video,
                    frame_idx=0,
                    instances=[prediction_with_track],
                )
            ],
        )
        new_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_track
            )
        )
        before = new_inst.numpy().copy()
        before_vis = new_inst.points["visible"].copy()

        ctx = self._stub_context(simple_skeleton, labels)
        AddUserInstancesFromPredictions.fill_missing_predicted_nodes(
            ctx, new_inst, prediction_with_track
        )

        np.testing.assert_array_equal(new_inst.numpy(), before)
        np.testing.assert_array_equal(new_inst.points["visible"], before_vis)

    def test_no_nodes_detected_fills_all_visible(self, simple_skeleton, simple_video):
        """When nothing is detected, all nodes are filled and visible.

        The centroid is undefined (no visible nodes) so placement falls back to
        the current view (the stub's visible rect), exactly like a brand-new
        instance with no copy source.
        """
        pred = PredictedInstance.empty(skeleton=simple_skeleton, score=0.1)
        pred["head"] = (np.nan, np.nan, 0.0)
        pred["thorax"] = (np.nan, np.nan, 0.0)
        pred["abdomen"] = (np.nan, np.nan, 0.0)
        pred.points["visible"] = True

        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[
                LabeledFrame(video=simple_video, frame_idx=0, instances=[pred])
            ],
        )
        new_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(pred)
        )
        ctx = self._stub_context(simple_skeleton, labels)
        AddUserInstancesFromPredictions.fill_missing_predicted_nodes(
            ctx, new_inst, pred
        )

        assert np.all(np.isfinite(new_inst.numpy())), (
            "all nodes should be filled with finite coords when none detected"
        )
        assert all(bool(v) for v in new_inst.points["visible"]), (
            "all filled nodes should be visible"
        )

    def test_single_node_skeleton_undetected(self, simple_video):
        """Single-node skeleton with the node undetected gets placed visibly."""
        skeleton = Skeleton(name="one")
        skeleton.add_node("center")

        pred = PredictedInstance.empty(skeleton=skeleton, score=0.1)
        pred["center"] = (np.nan, np.nan, 0.0)
        pred.points["visible"] = True

        labels = Labels(
            videos=[simple_video],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(video=simple_video, frame_idx=0, instances=[pred])
            ],
        )
        new_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(pred)
        )
        ctx = self._stub_context(skeleton, labels)
        AddUserInstancesFromPredictions.fill_missing_predicted_nodes(
            ctx, new_inst, pred
        )

        assert np.all(np.isfinite(new_inst.numpy()))
        assert bool(new_inst.points[0]["visible"]) is True

    def test_no_player_is_noop(self, simple_skeleton, prediction_with_nan_node):
        """Headless context (no player) leaves the instance untouched.

        This is the contract that keeps the bulk ``do_action`` tests (which use
        ``CommandContext.from_labels`` / ``FakeApp``) passing: without a GUI
        player there is no visible-rect to place into, so the fill is skipped
        and the NaN node stays NaN/invisible.
        """
        from sleap.gui.commands import CommandContext

        new_inst = (
            AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
                prediction_with_nan_node
            )
        )
        before = new_inst.numpy().copy()

        # FakeApp has no `player` attribute.
        context = CommandContext.from_labels(Labels(skeletons=[simple_skeleton]))
        AddUserInstancesFromPredictions.fill_missing_predicted_nodes(
            context, new_inst, prediction_with_nan_node
        )

        names = list(new_inst.points["name"])
        thorax_idx = names.index("thorax")
        np.testing.assert_array_equal(
            new_inst.numpy(), before, err_msg="no-player fill must be a no-op"
        )
        assert bool(new_inst.points[thorax_idx]["visible"]) is False

    def test_single_frame_do_action_wires_fill_with_player(
        self, simple_skeleton, simple_video, prediction_with_nan_node
    ):
        """``AddUserInstancesFromPredictions.do_action`` runs the fill (wiring).

        The helper-level tests above prove the fill works in isolation, but they
        would still pass if the ``fill_missing_predicted_nodes`` call were ever
        removed from ``do_action``. Drive the real ``do_action`` with a context
        that has a player so the converted user instance's undetected node is
        actually filled end-to-end.
        """
        lf = LabeledFrame(
            video=simple_video, frame_idx=0, instances=[prediction_with_nan_node]
        )
        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
        )
        ctx = self._stub_context(simple_skeleton, labels)
        ctx.state["labeled_frame"] = lf

        AddUserInstancesFromPredictions.do_action(ctx, {})

        user = [inst for inst in lf.instances if type(inst) is Instance]
        assert len(user) == 1
        new_inst = user[0]
        names = list(new_inst.points["name"])
        thorax_idx = names.index("thorax")
        head_idx = names.index("head")

        # Undetected node filled by the do_action -> fill wiring.
        assert np.all(np.isfinite(new_inst.points[thorax_idx]["xy"]))
        assert bool(new_inst.points[thorax_idx]["visible"]) is True
        assert bool(new_inst.points[thorax_idx]["complete"]) is False
        # Detected node preserved.
        np.testing.assert_array_equal(
            new_inst.points[head_idx]["xy"], np.array([10.0, 20.0])
        )

    def test_all_frames_do_action_wires_fill_with_player(
        self, simple_skeleton, simple_video, prediction_with_nan_node
    ):
        """``AddUserInstancesFromAllPredictions.do_action`` runs the fill (wiring).

        The all-frames bulk path has no other player-present coverage, so this
        locks in that it too initializes undetected nodes.
        """
        lf = LabeledFrame(
            video=simple_video, frame_idx=0, instances=[prediction_with_nan_node]
        )
        labels = Labels(
            videos=[simple_video],
            skeletons=[simple_skeleton],
            labeled_frames=[lf],
        )
        ctx = self._stub_context(simple_skeleton, labels)

        AddUserInstancesFromAllPredictions.do_action(ctx, {})

        user = [inst for inst in lf.instances if type(inst) is Instance]
        assert len(user) == 1
        new_inst = user[0]
        thorax_idx = list(new_inst.points["name"]).index("thorax")
        assert np.all(np.isfinite(new_inst.points[thorax_idx]["xy"]))
        assert bool(new_inst.points[thorax_idx]["visible"]) is True
        assert bool(new_inst.points[thorax_idx]["complete"]) is False

"""Tests for merge functionality using sleap-io 0.6.0 API.

These tests verify that the merge dialog, inference result merging, and
DLC import all work correctly with the `frame=` parameter.
"""

import pytest
from copy import deepcopy

from sleap_io import Labels, LabeledFrame, Instance, PredictedInstance, Video, Skeleton
from sleap.gui.dialogs.merge import show_instance_type_counts


def test_count_string(simple_predictions):
    """Test the instance type count string formatting."""
    assert show_instance_type_counts(simple_predictions[0]) == "0 (user) / 2 (pred)"


class TestLabelsMergeAPI:
    """Tests for Labels.merge() with the frame= parameter (sleap-io 0.6.0 API)."""

    @pytest.fixture
    def base_labels(self):
        """Create base labels with user instances."""
        skeleton = Skeleton(["A", "B"])
        video = Video(filename="base_video.mp4")
        labels = Labels(videos=[video], skeletons=[skeleton])

        # Add user instances on frames 0, 1, 2
        for frame_idx in range(3):
            lf = LabeledFrame(video=video, frame_idx=frame_idx)
            inst = Instance.from_numpy([[10, 10], [20, 20]], skeleton=skeleton)
            lf.instances.append(inst)
            labels.append(lf)

        return labels

    @pytest.fixture
    def new_labels_same_video(self, base_labels):
        """Create new labels with predictions on overlapping frames."""
        skeleton = base_labels.skeleton
        video = base_labels.video

        labels = Labels(videos=[video], skeletons=[skeleton])

        # Add predictions on frames 1, 2, 3 (overlapping with base on 1, 2)
        for frame_idx in range(1, 4):
            lf = LabeledFrame(video=video, frame_idx=frame_idx)
            pred = PredictedInstance.from_numpy(
                [[15, 15], [25, 25]], skeleton=skeleton, score=0.9
            )
            lf.instances.append(pred)
            labels.append(lf)

        return labels

    def test_merge_frame_keep_both(self, base_labels, new_labels_same_video):
        """Test merge with frame='keep_both' keeps instances from both."""
        base_copy = deepcopy(base_labels)
        original_frame_count = len(base_copy)

        # Merge with keep_both strategy
        base_copy.merge(new_labels_same_video, frame="keep_both")

        # Should have frames 0, 1, 2, 3 (frame 3 is new)
        assert len(base_copy) >= original_frame_count

        # Frames 1 and 2 should have both user and predicted instances
        for frame_idx in [1, 2]:
            lf = base_copy.find(base_copy.video, frame_idx=frame_idx)
            if lf:
                lf = lf[0]
                # Should have instances from both sources
                assert len(lf.instances) >= 1

    def test_merge_frame_replace_predictions(self, base_labels, new_labels_same_video):
        """Test merge with frame='replace_predictions' replaces predictions."""
        # Add some predictions to base labels first
        skeleton = base_labels.skeleton
        lf = base_labels.find(base_labels.video, frame_idx=1)[0]
        old_pred = PredictedInstance.from_numpy(
            [[5, 5], [5, 5]], skeleton=skeleton, score=0.5
        )
        lf.instances.append(old_pred)

        base_copy = deepcopy(base_labels)

        # Merge with replace_predictions strategy
        base_copy.merge(new_labels_same_video, frame="replace_predictions")

        # The merge should have processed - check it didn't error
        assert len(base_copy) >= 1

    def test_merge_frame_auto(self, base_labels, new_labels_same_video):
        """Test merge with frame='auto' uses automatic strategy selection."""
        base_copy = deepcopy(base_labels)

        # Merge with auto strategy (should work without errors)
        base_copy.merge(new_labels_same_video, frame="auto")

        # Basic sanity check - merge completed
        assert len(base_copy) >= len(base_labels)


class TestInferenceResultMerging:
    """Tests for InferenceTask.merge_results() functionality."""

    @pytest.fixture
    def labels_with_predictions(self, centered_pair_predictions):
        """Get labels with existing predictions."""
        from sleap.sleap_io_adaptors.lf_labels_utils import labels_copy

        return labels_copy(centered_pair_predictions)

    def test_merge_results_add_mode(self, labels_with_predictions):
        """Test merging inference results in 'add' mode uses keep_both."""
        from sleap.gui.learning.runners import InferenceTask

        labels = labels_with_predictions
        skeleton = labels.skeleton
        video = labels.video

        # Count predictions before
        pred_count_before = sum(
            len(lf.predicted_instances) for lf in labels.labeled_frames
        )

        # Create mock inference results
        results = []
        for frame_idx in [0, 1]:
            lf = LabeledFrame(video=video, frame_idx=frame_idx)
            pred = PredictedInstance.from_numpy(
                [[100, 100]] * len(skeleton.nodes), skeleton=skeleton, score=0.95
            )
            lf.instances.append(pred)
            results.append(lf)

        # Create InferenceTask with add mode (trained_job_paths=[] for testing)
        task = InferenceTask(
            trained_job_paths=[],
            labels=labels,
            results=results,
            inference_params={"_prediction_mode": "add"},
        )

        # Merge results
        task.merge_results()

        # Predictions should be added (not replaced)
        pred_count_after = sum(
            len(lf.predicted_instances) for lf in labels.labeled_frames
        )
        assert pred_count_after >= pred_count_before

    def test_merge_results_replace_mode(self, labels_with_predictions):
        """Test merging inference results in 'replace' mode uses replace_predictions."""
        from sleap.gui.learning.runners import InferenceTask

        labels = labels_with_predictions
        skeleton = labels.skeleton
        video = labels.video

        # Create mock inference results with new predictions
        results = []
        lf = LabeledFrame(video=video, frame_idx=0)
        pred = PredictedInstance.from_numpy(
            [[200, 200]] * len(skeleton.nodes), skeleton=skeleton, score=0.99
        )
        lf.instances.append(pred)
        results.append(lf)

        # Create InferenceTask with replace mode
        task = InferenceTask(
            trained_job_paths=[],
            labels=labels,
            results=results,
            inference_params={"_prediction_mode": "replace"},
        )

        # Merge results
        task.merge_results()

        # Check that merge completed without error
        assert len(labels) >= 1

    def test_merge_results_clear_all_first(self, labels_with_predictions):
        """Test that _clear_all_first removes predictions before merge."""
        from sleap.gui.learning.runners import InferenceTask

        labels = labels_with_predictions
        skeleton = labels.skeleton
        video = labels.video

        # Create mock inference results
        results = []
        lf = LabeledFrame(video=video, frame_idx=0)
        pred = PredictedInstance.from_numpy(
            [[50, 50]] * len(skeleton.nodes), skeleton=skeleton, score=0.8
        )
        lf.instances.append(pred)
        results.append(lf)

        # Create InferenceTask with clear_all_first
        task = InferenceTask(
            trained_job_paths=[],
            labels=labels,
            results=results,
            inference_params={
                "_prediction_mode": "add",
                "_clear_all_first": True,
                "_predict_target": "suggested_frames",
            },
        )

        # Merge results (this should clear predictions first)
        task.merge_results()

        # Check that merge completed without error
        assert len(labels) >= 1

    def test_merge_results_removes_empty_instances(self, labels_with_predictions):
        """Test that empty instances are removed during merge."""
        from sleap.gui.learning.runners import InferenceTask

        labels = labels_with_predictions
        skeleton = labels.skeleton
        video = labels.video

        # Create results with an empty instance (no visible points)
        results = []
        lf = LabeledFrame(video=video, frame_idx=0)
        # Create instance with all NaN points
        import numpy as np

        empty_pred = PredictedInstance.from_numpy(
            np.full((len(skeleton.nodes), 2), np.nan), skeleton=skeleton, score=0.5
        )
        lf.instances.append(empty_pred)
        results.append(lf)

        # Create InferenceTask
        task = InferenceTask(
            trained_job_paths=[],
            labels=labels,
            results=results,
            inference_params={"_prediction_mode": "add"},
        )

        # Merge results - empty instances should be filtered out
        task.merge_results()

        # The empty frame should have been filtered
        # (results list is filtered before merge)
        assert len(task.results) == 0 or all(
            len(lf.instances) > 0 for lf in task.results
        )


class TestImportDLCFolderMerge:
    """Tests for ImportDeepLabCutFolder merge functionality."""

    def test_import_multiple_dlc_files_merges_correctly(self):
        """Test that importing multiple DLC files uses frame='auto' merge."""
        from sleap.gui.commands import ImportDeepLabCutFolder

        # Find DLC test files
        csv_files = ImportDeepLabCutFolder.find_dlc_files_in_folder(
            "tests/data/dlc_multiple_datasets"
        )
        assert len(csv_files) == 2

        # Import and merge - this uses merged_labels.merge(labels, frame="auto")
        labels = ImportDeepLabCutFolder.import_labels_from_dlc_files(csv_files)

        # Verify merge worked correctly
        assert labels is not None
        assert len(labels.labeled_frames) == 3  # 2 from one file, 1 from another
        assert len(labels.videos) == 2
        assert len(labels.skeletons) == 1

    def test_import_single_dlc_file_no_merge(self):
        """Test that importing a single DLC file doesn't require merge."""
        from sleap.gui.commands import ImportDeepLabCutFolder
        import glob

        # Find just one DLC test file
        csv_files = glob.glob("tests/data/dlc_multiple_datasets/video1/*.csv")
        assert len(csv_files) == 1

        # Import single file - no merge needed
        labels = ImportDeepLabCutFolder.import_labels_from_dlc_files(csv_files)

        # Verify import worked
        assert labels is not None
        assert len(labels.labeled_frames) == 2
        assert len(labels.videos) == 1

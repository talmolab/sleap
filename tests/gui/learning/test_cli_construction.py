"""Tests for CLI construction.

This module tests that CLI arguments are correctly constructed for
training and inference commands.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile


from sleap_io import Video, Labels

from sleap.gui.learning.runners import (
    VideoItemForInference,
    DatasetItemForInference,
    ItemsForInference,
    InferenceTask,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_video():
    """Create a mock Video object."""
    video = MagicMock(spec=Video)
    video.filename = "/path/to/video.mp4"
    video.backend = MagicMock()
    video.backend.dataset = None
    video.backend.input_format = None
    return video


@pytest.fixture
def mock_video_with_backend():
    """Create a mock Video with backend attributes."""
    video = MagicMock(spec=Video)
    video.filename = "/path/to/video.h5"
    video.backend = MagicMock()
    video.backend.dataset = "images"
    video.backend.input_format = "channels_last"
    return video


@pytest.fixture
def mock_embedded_video():
    """Create a mock Video whose images are embedded in a `.pkg.slp`.

    Matches what sleap-io reports for an embedded project: `filename` is the
    `.pkg.slp` itself and `backend.dataset` names the embedded video.
    """
    video = MagicMock(spec=Video)
    video.filename = "/path/to/train.pkg.slp"
    video.backend = MagicMock()
    video.backend.dataset = "video3/video"
    video.backend.input_format = "channels_last"
    return video


@pytest.fixture
def mock_image_sequence_video():
    """Create a mock Video backed by a list of image files."""
    video = MagicMock(spec=Video)
    video.filename = ["/path/to/imgs/img0.png", "/path/to/imgs/img1.png"]
    video.backend = MagicMock()
    video.backend.dataset = None
    video.backend.input_format = None
    return video


@pytest.fixture
def mock_labels(mock_video):
    """Create a mock Labels object."""
    labels = MagicMock(spec=Labels)
    labels.videos = [mock_video]
    return labels


# =============================================================================
# VideoItemForInference CLI Args Tests
# =============================================================================


class TestVideoItemForInferenceCLI:
    """Tests for VideoItemForInference CLI argument construction."""

    def test_basic_cli_args(self, mock_video):
        """Basic CLI args should include the video data path and frames."""
        item = VideoItemForInference(
            video=mock_video,
            frames=[0, 1, 2, 3, 4],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        cli_args = item.cli_args

        assert "--data_path" in cli_args
        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/video.mp4"
        assert "/path/to/labels.slp" not in cli_args
        assert "--frames" in cli_args
        # Frames should be comma-separated
        frames_idx = cli_args.index("--frames") + 1
        assert cli_args[frames_idx] == "0,1,2,3,4"

    def test_cli_args_never_include_video_index(self, mock_video):
        """CLI args should not include --video_index even with a labels_path.

        Regression test: `--data_path` is the video file itself, so there is
        exactly one video and no index to select. Passing a `.slp` data path plus
        `--video_index`/`--frames` routed sleap-nn through `LabelsProvider`, which
        can only subset frames that are *already labeled* in the project -- so
        "predict on entire video" silently capped at the labeled-frame count.
        """
        item = VideoItemForInference(
            video=mock_video,
            frames=[0, 1, 2],
            labels_path="/path/to/labels.slp",
            video_idx=2,
        )

        cli_args = item.cli_args

        assert "--video_index" not in cli_args
        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/video.mp4"

    def test_entire_video_range_targets_video_file(self, mock_video):
        """Entire-video mode on a sparsely labeled project must target the video.

        Regression test for the frame-count cap: a project with 20 labeled frames
        out of 2560 asked for `--frames 0,-1499` against the `.slp`, and sleap-nn
        logged `source=LabelsProvider | frames=20`. The full frame range must go to
        the video file so `VideoProvider` reads every requested index.
        """
        # "Entire video" encodes the full range as [0, -num_frames] (see
        # `MainWindow._get_frame_selection`); 1500 frames requested here.
        item = VideoItemForInference(
            video=mock_video,
            frames=[0, -1500],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        cli_args = item.cli_args

        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/video.mp4"
        assert not cli_args[data_idx].endswith(".slp")
        assert "--video_index" not in cli_args
        frames_idx = cli_args.index("--frames") + 1
        assert cli_args[frames_idx] == "0,-1499"

    def test_embedded_pkg_slp_keeps_labels_path_and_video_index(
        self, mock_embedded_video
    ):
        """An embedded `.pkg.slp` video must stay on the project + --video_index.

        There is no separate video file for an embedded video -- `video.filename` is
        the `.pkg.slp` itself -- so pointing `--data_path` at it would hand sleap-nn a
        `.slp` with no `--video_index`, running unscoped across every video in the
        project. The frames that exist for an embedded video are exactly the ones
        stored in the project, so `LabelsProvider` is the right source here.
        """
        item = VideoItemForInference(
            video=mock_embedded_video,
            frames=[0, -80],
            labels_path="/path/to/train.pkg.slp",
            video_idx=3,
        )

        assert item.uses_labels_path
        assert item.path == "/path/to/train.pkg.slp"

        cli_args = item.cli_args
        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/train.pkg.slp"
        assert "--video_index" in cli_args
        assert cli_args[cli_args.index("--video_index") + 1] == "3"

    def test_image_sequence_video_keeps_labels_path(self, mock_image_sequence_video):
        """A video backed by a list of images must stay on the project path.

        `Video.filename` is a list for image-sequence backends, which cannot be
        expressed as a single `--data_path` (and would be stringified into a Python
        list repr on the command line).
        """
        item = VideoItemForInference(
            video=mock_image_sequence_video,
            frames=[0, -2],
            labels_path="/path/to/labels.slp",
            video_idx=1,
        )

        assert item.uses_labels_path
        assert item.path == "/path/to/labels.slp"

        cli_args = item.cli_args
        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/labels.slp"
        assert "[" not in cli_args[data_idx]
        assert "--video_index" in cli_args

    def test_absolute_path_not_applied_to_list_filename(
        self, mock_image_sequence_video
    ):
        """use_absolute_path must not try to abspath() a list of image paths."""
        item = VideoItemForInference(
            video=mock_image_sequence_video,
            frames=[0, -2],
            labels_path="/path/to/labels.slp",
            video_idx=0,
            use_absolute_path=True,
        )

        # Would raise TypeError if abspath() were handed the list.
        assert item.path == "/path/to/labels.slp"

    def test_cli_args_with_hdf5_backend(self, mock_video_with_backend):
        """CLI args should include video_dataset and video_input_format for HDF5."""
        item = VideoItemForInference(
            video=mock_video_with_backend,
            frames=[0],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        cli_args = item.cli_args

        assert "--video_dataset" in cli_args
        dataset_idx = cli_args.index("--video_dataset") + 1
        assert cli_args[dataset_idx] == "images"

        assert "--video_input_format" in cli_args
        format_idx = cli_args.index("--video_input_format") + 1
        assert cli_args[format_idx] == "channels_last"

    def test_cli_args_frame_range(self, mock_video):
        """CLI args should handle frame ranges (negative end marker)."""
        # Negative values indicate range endpoints
        item = VideoItemForInference(
            video=mock_video,
            frames=[0, -100],  # Frames 0 to 99
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        cli_args = item.cli_args

        assert "--frames" in cli_args
        frames_idx = cli_args.index("--frames") + 1
        # -100 becomes -99 (endpoint adjustment)
        assert "-99" in cli_args[frames_idx]

    def test_remote_url_video_not_mangled_by_absolute_path(self):
        """A remote video URL must pass through verbatim.

        `os.path.abspath()` would rewrite `https://h/v.mp4` into
        `<cwd>/https:/h/v.mp4`. sleap-nn passes URL data paths through as-is.
        """
        video = MagicMock(spec=Video)
        video.filename = "https://host/videos/vid.mp4"
        video.backend = MagicMock()
        video.backend.dataset = None
        video.backend.input_format = None

        item = VideoItemForInference(
            video=video,
            frames=[0, -100],
            labels_path="/path/to/labels.slp",
            video_idx=0,
            use_absolute_path=True,
        )

        assert not item.uses_labels_path
        assert item.path == "https://host/videos/vid.mp4"

    def test_no_labels_path_still_uses_video(self, mock_video):
        """An empty labels_path must not be used as the data path."""
        for labels_path in (None, ""):
            item = VideoItemForInference(
                video=mock_video,
                frames=[0],
                labels_path=labels_path,
                video_idx=0,
            )
            assert not item.uses_labels_path
            assert item.path == mock_video.filename

    def test_no_frames_omits_frames_arg(self, mock_video):
        """frames=None means all frames, which is `--frames` omitted, not a crash."""
        item = VideoItemForInference(
            video=mock_video,
            frames=None,
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        cli_args = item.cli_args  # Used to raise TypeError.

        assert "--frames" not in cli_args
        data_idx = cli_args.index("--data_path") + 1
        assert cli_args[data_idx] == "/path/to/video.mp4"

    def test_path_property_with_labels(self, mock_video):
        """path property should return the video filename even with a labels_path."""
        item = VideoItemForInference(
            video=mock_video,
            frames=[0],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        assert item.path == mock_video.filename

    def test_path_property_without_labels(self, mock_video):
        """path property should return video filename when no labels_path."""
        item = VideoItemForInference(
            video=mock_video,
            frames=[0],
            labels_path=None,
            video_idx=0,
        )

        assert item.path == mock_video.filename


# =============================================================================
# DatasetItemForInference CLI Args Tests
# =============================================================================


class TestDatasetItemForInferenceCLI:
    """Tests for DatasetItemForInference CLI argument construction."""

    def test_user_labeled_frames_cli_args(self):
        """User labeled frames should add --only_labeled_frames."""
        item = DatasetItemForInference(
            labels_path="/path/to/labels.slp",
            frame_filter="user",
        )

        cli_args = item.cli_args

        assert "--data_path" in cli_args
        assert "/path/to/labels.slp" in cli_args
        assert "--only_labeled_frames" in cli_args
        assert "--only_suggested_frames" not in cli_args

    def test_suggested_frames_cli_args(self):
        """Suggested frames should add --only_suggested_frames."""
        item = DatasetItemForInference(
            labels_path="/path/to/labels.slp",
            frame_filter="suggested",
        )

        cli_args = item.cli_args

        assert "--data_path" in cli_args
        assert "--only_suggested_frames" in cli_args
        assert "--only_labeled_frames" not in cli_args

    def test_predicted_frames_cli_args(self):
        """Predicted frames should add --only_predicted_frames."""
        item = DatasetItemForInference(
            labels_path="/path/to/labels.slp",
            frame_filter="predicted",
        )

        cli_args = item.cli_args

        assert "--data_path" in cli_args
        assert "--only_predicted_frames" in cli_args
        assert "--only_labeled_frames" not in cli_args
        assert "--only_suggested_frames" not in cli_args

    def test_path_property(self):
        """path property should return labels_path."""
        item = DatasetItemForInference(
            labels_path="/path/to/labels.slp",
            frame_filter="user",
        )

        assert item.path == "/path/to/labels.slp"

    def test_absolute_path(self):
        """use_absolute_path should return absolute path."""
        with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
            temp_path = f.name

        item = DatasetItemForInference(
            labels_path=temp_path,
            frame_filter="user",
            use_absolute_path=True,
        )

        assert Path(item.path).is_absolute()


# =============================================================================
# ItemsForInference Tests
# =============================================================================


class TestItemsForInference:
    """Tests for ItemsForInference class."""

    def test_from_video_frames_dict(self, mock_video, mock_labels):
        """from_video_frames_dict should create correct items."""
        video_frames_dict = {mock_video: [0, 1, 2, 3, 4]}

        items = ItemsForInference.from_video_frames_dict(
            video_frames_dict=video_frames_dict,
            total_frame_count=5,
            labels=mock_labels,
            labels_path="/path/to/labels.slp",
        )

        assert len(items) == 1
        assert items.total_frame_count == 5
        assert isinstance(items.items[0], VideoItemForInference)

    def test_from_video_frames_dict_targets_video_not_labels(
        self, mock_video, mock_labels
    ):
        """Frame-range items must run on the video file, not the project .slp.

        `labels_path`/`video_idx` are still recorded on the item (they identify the
        project the video came from), they just no longer choose `--data_path`.
        """
        items = ItemsForInference.from_video_frames_dict(
            video_frames_dict={mock_video: [0, -2560]},
            total_frame_count=2560,
            labels=mock_labels,
            labels_path="/path/to/labels.slp",
        )

        item = items.items[0]
        assert item.path == "/path/to/video.mp4"
        assert item.labels_path == "/path/to/labels.slp"
        assert item.video_idx == 0
        assert "--video_index" not in item.cli_args

    def test_from_video_frames_dict_multiple_videos(self, mock_labels):
        """from_video_frames_dict should handle multiple videos."""
        video1 = MagicMock(spec=Video)
        video1.filename = "/path/to/video1.mp4"
        video1.backend = MagicMock()
        video1.backend.dataset = None
        video1.backend.input_format = None

        video2 = MagicMock(spec=Video)
        video2.filename = "/path/to/video2.mp4"
        video2.backend = MagicMock()
        video2.backend.dataset = None
        video2.backend.input_format = None

        mock_labels.videos = [video1, video2]

        video_frames_dict = {
            video1: [0, 1, 2],
            video2: [0, 1, 2, 3, 4],
        }

        items = ItemsForInference.from_video_frames_dict(
            video_frames_dict=video_frames_dict,
            total_frame_count=8,
            labels=mock_labels,
            labels_path="/path/to/labels.slp",
        )

        assert len(items) == 2

    def test_from_video_frames_dict_skips_empty(self, mock_video, mock_labels):
        """from_video_frames_dict should skip videos with no frames."""
        video_frames_dict = {mock_video: []}

        items = ItemsForInference.from_video_frames_dict(
            video_frames_dict=video_frames_dict,
            total_frame_count=0,
            labels=mock_labels,
            labels_path="/path/to/labels.slp",
        )

        assert len(items) == 0


# =============================================================================
# InferenceTask CLI Construction Tests
# =============================================================================


class TestInferenceTaskCLI:
    """Tests for InferenceTask.make_predict_cli_call."""

    @pytest.fixture
    def inference_task(self, mock_labels):
        """Create a basic inference task."""
        return InferenceTask(
            trained_job_paths=["/path/to/model/training_config.yaml"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

    @pytest.fixture
    def video_item(self, mock_video):
        """Create a video item for inference."""
        return VideoItemForInference(
            video=mock_video,
            frames=[0, 1, 2],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

    def test_basic_cli_call(self, inference_task, video_item):
        """Basic CLI call should include sleap track and model paths."""
        cli_args, output_path = inference_task.make_predict_cli_call(
            video_item, output_path="/path/to/output.slp"
        )

        assert cli_args[0] == "sleap"
        assert cli_args[1] == "predict"
        assert "--model_paths" in cli_args
        # Model path should be parent directory (strip training_config.yaml)
        model_idx = cli_args.index("--model_paths") + 1
        # Normalize path separators for cross-platform compatibility
        assert Path(cli_args[model_idx]).as_posix() == "/path/to/model"
        assert "-o" in cli_args
        assert "/path/to/output.slp" in cli_args

    def test_batch_size_in_cli(self, inference_task, video_item):
        """Batch size should be included in CLI args."""
        inference_task.inference_params["_batch_size"] = 8

        cli_args, _ = inference_task.make_predict_cli_call(
            video_item, output_path="/tmp/out.slp"
        )

        assert "--batch_size" in cli_args
        batch_idx = cli_args.index("--batch_size") + 1
        assert cli_args[batch_idx] == "8"

    def test_max_instances_in_cli(self, inference_task, video_item):
        """Max instances should be included in CLI args."""
        inference_task.inference_params["_max_instances"] = 5

        cli_args, _ = inference_task.make_predict_cli_call(
            video_item, output_path="/tmp/out.slp"
        )

        assert "--max_instances" in cli_args
        max_idx = cli_args.index("--max_instances") + 1
        assert cli_args[max_idx] == "5"

    def test_max_instances_none_not_in_cli(self, inference_task, video_item):
        """Max instances=None should NOT be in CLI args."""
        inference_task.inference_params["_max_instances"] = None

        cli_args, _ = inference_task.make_predict_cli_call(
            video_item, output_path="/tmp/out.slp"
        )

        assert "--max_instances" not in cli_args


# =============================================================================
# Tracker CLI Construction Tests
# =============================================================================


class TestTrackerCLI:
    """Tests for tracker CLI argument construction."""

    @pytest.fixture
    def inference_task_with_tracker(self, mock_labels):
        """Create an inference task with tracker params."""
        return InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "flow",
                "tracking.match": "greedy",
                "tracking.track_window": 10,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "oks",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

    @pytest.fixture
    def video_item(self, mock_video):
        """Create a video item for inference."""
        return VideoItemForInference(
            video=mock_video,
            frames=[0, 1, 2],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

    def test_tracker_none_no_tracking_args(self, mock_labels, video_item):
        """tracker=none should not add tracking args."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={"tracking.tracker": "none"},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--tracking" not in cli_args

    def test_flow_tracker_cli_args(self, inference_task_with_tracker, video_item):
        """Flow tracker should add correct CLI args."""
        cli_args, _ = inference_task_with_tracker.make_predict_cli_call(
            video_item, output_path="/tmp/out.slp"
        )

        assert "--tracking" in cli_args
        assert "--use_flow" in cli_args
        assert "--track_matching_method" in cli_args
        match_idx = cli_args.index("--track_matching_method") + 1
        assert cli_args[match_idx] == "greedy"
        assert "--tracking_window_size" in cli_args
        window_idx = cli_args.index("--tracking_window_size") + 1
        assert cli_args[window_idx] == "10"

    def test_simple_tracker_no_flow(self, mock_labels, video_item):
        """Simple tracker should not add --use_flow."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "hungarian",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "centroids",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--tracking" in cli_args
        assert "--use_flow" not in cli_args

    def test_max_tracks_in_cli(self, mock_labels, video_item):
        """Max tracks should add --candidates_method and --max_tracks."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "flow",
                "tracking.match": "greedy",
                "tracking.track_window": 10,
                "tracking.max_tracks": 3,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "oks",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--candidates_method" in cli_args
        cand_idx = cli_args.index("--candidates_method") + 1
        assert cli_args[cand_idx] == "local_queues"
        assert "--max_tracks" in cli_args
        max_idx = cli_args.index("--max_tracks") + 1
        assert cli_args[max_idx] == "3"

    def test_post_connect_single_breaks_cli(self, mock_labels, video_item):
        """post_connect_single_breaks=1 should add correct args."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "flow",
                "tracking.match": "greedy",
                "tracking.track_window": 10,
                "tracking.max_tracks": 5,
                "tracking.post_connect_single_breaks": 1,
                "tracking.robust": 1.0,
                "tracking.similarity": "oks",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--post_connect_single_breaks" in cli_args
        assert "--tracking_target_instance_count" in cli_args
        count_idx = cli_args.index("--tracking_target_instance_count") + 1
        assert cli_args[count_idx] == "5"

    def test_similarity_oks_cli(self, inference_task_with_tracker, video_item):
        """similarity=oks should add --features keypoints and --scoring_method oks."""
        cli_args, _ = inference_task_with_tracker.make_predict_cli_call(
            video_item, output_path="/tmp/out.slp"
        )

        assert "--features" in cli_args
        feat_idx = cli_args.index("--features") + 1
        assert cli_args[feat_idx] == "keypoints"
        assert "--scoring_method" in cli_args
        score_idx = cli_args.index("--scoring_method") + 1
        assert cli_args[score_idx] == "oks"

    def test_similarity_centroids_cli(self, mock_labels, video_item):
        """similarity=centroids should add correct args."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "greedy",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "centroids",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--features" in cli_args
        feat_idx = cli_args.index("--features") + 1
        assert cli_args[feat_idx] == "centroids"
        assert "--scoring_method" in cli_args
        score_idx = cli_args.index("--scoring_method") + 1
        assert cli_args[score_idx] == "euclidean_dist"

    def test_similarity_iou_cli(self, mock_labels, video_item):
        """similarity=iou should add correct args."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "greedy",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "iou",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--features" in cli_args
        feat_idx = cli_args.index("--features") + 1
        assert cli_args[feat_idx] == "bboxes"
        assert "--scoring_method" in cli_args
        score_idx = cli_args.index("--scoring_method") + 1
        assert cli_args[score_idx] == "iou"

    def test_similarity_centroid_cli(self, mock_labels, video_item):
        """similarity=centroid (non-flow tracker dropdown value) should map the
        same as centroids, not be silently dropped."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "greedy",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "centroid",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--features" in cli_args
        feat_idx = cli_args.index("--features") + 1
        assert cli_args[feat_idx] == "centroids"
        assert "--scoring_method" in cli_args
        score_idx = cli_args.index("--scoring_method") + 1
        assert cli_args[score_idx] == "euclidean_dist"

    def test_similarity_instance_cli(self, mock_labels, video_item):
        """similarity=instance (non-flow tracker "object keypoint" dropdown
        value) should map the same as oks, not be silently dropped."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "greedy",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 1.0,
                "tracking.similarity": "instance",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--features" in cli_args
        feat_idx = cli_args.index("--features") + 1
        assert cli_args[feat_idx] == "keypoints"
        assert "--scoring_method" in cli_args
        score_idx = cli_args.index("--scoring_method") + 1
        assert cli_args[score_idx] == "oks"

    def test_robust_quantile_cli(self, mock_labels, video_item):
        """robust != 1.0 should add robust quantile args."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={
                "tracking.tracker": "simple",
                "tracking.match": "greedy",
                "tracking.track_window": 5,
                "tracking.max_tracks": None,
                "tracking.post_connect_single_breaks": 0,
                "tracking.robust": 0.75,
                "tracking.similarity": "oks",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        assert "--scoring_reduction" in cli_args
        red_idx = cli_args.index("--scoring_reduction") + 1
        assert cli_args[red_idx] == "robust_quantile"
        assert "--robust_best_instance" in cli_args
        robust_idx = cli_args.index("--robust_best_instance") + 1
        assert cli_args[robust_idx] == "0.75"


# =============================================================================
# Model Path Handling Tests
# =============================================================================


class TestModelPathHandling:
    """Tests for model path handling in CLI construction."""

    @pytest.fixture
    def video_item(self, mock_video):
        """Create a video item for inference."""
        return VideoItemForInference(
            video=mock_video,
            frames=[0],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

    def test_yaml_path_stripped_to_parent(self, mock_labels, video_item):
        """YAML config path should be stripped to parent directory."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model/training_config.yaml"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        model_idx = cli_args.index("--model_paths") + 1
        # Should be parent dir, not the yaml file
        # Normalize for cross-platform
        assert Path(cli_args[model_idx]).as_posix() == "/path/to/model"

    def test_json_path_stripped_to_parent(self, mock_labels, video_item):
        """JSON config path should be stripped to parent directory."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model/config.json"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        model_idx = cli_args.index("--model_paths") + 1
        # Normalize for cross-platform
        assert Path(cli_args[model_idx]).as_posix() == "/path/to/model"

    def test_directory_path_unchanged(self, mock_labels, video_item):
        """Directory path should be used as-is."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        model_idx = cli_args.index("--model_paths") + 1
        assert cli_args[model_idx] == "/path/to/model"

    def test_multiple_model_paths(self, mock_labels, video_item):
        """Multiple model paths should all be included."""
        task = InferenceTask(
            trained_job_paths=[
                "/path/to/centroid_model",
                "/path/to/centered_instance_model",
            ],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, _ = task.make_predict_cli_call(video_item, output_path="/tmp/out.slp")

        # Count --model_paths occurrences
        model_paths_count = cli_args.count("--model_paths")
        assert model_paths_count == 2

        # Both paths should be present
        assert "/path/to/centroid_model" in cli_args
        assert "/path/to/centered_instance_model" in cli_args


# =============================================================================
# Output Path Generation Tests
# =============================================================================


class TestWritePipelineFiles:
    """Tests for write_pipeline_files function."""

    def test_train_script_does_not_contain_zmq(self, tmp_path, mock_labels):
        """Train script should not contain zmq overrides.

        Regression test for issue #2562: The export training package feature
        crashed because the code tried to access zmq config fields that don't
        exist in the GUI-generated config. The fix removes zmq overrides entirely
        since they're unnecessary for Colab training.
        """
        from sleap.gui.learning.runners import write_pipeline_files
        from sleap.gui.learning.configs import ConfigFileInfo
        from omegaconf import OmegaConf

        # Create a minimal config WITHOUT zmq fields (simulating GUI form data)
        # This is exactly the scenario that caused the bug
        config_dict = {
            "trainer_config": {
                "ckpt_dir": str(tmp_path / "models"),
                "run_name": "test_run",
                "save_ckpt": True,
            },
            "data_config": {
                "train_labels_path": ["labels.slp"],
            },
            "model_config": {
                "head_configs": {},
            },
        }
        config = OmegaConf.create(config_dict)

        # Create ConfigFileInfo with the minimal config
        cfg_info = ConfigFileInfo(
            config=config,
            path="test_config.yaml",
            filename="test_config.yaml",
            head_name="centroid",
            dont_retrain=False,
        )

        # Create output directory
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        labels_path = tmp_path / "labels.slp"
        labels_path.touch()

        # Create video item for inference
        video_item = VideoItemForInference(
            video=mock_labels.videos[0],
            frames=[0, 1, 2],
            labels_path=str(labels_path),
            video_idx=0,
        )

        items_for_inference = ItemsForInference(
            items=[video_item],
            total_frame_count=3,
        )

        # Mock the sleap_nn functions to avoid requiring the full sleap-nn install
        # verify_training_cfg is imported inside the function from sleap_nn
        # filter_cfg is imported at module level from sleap.gui.config_utils
        with patch(
            "sleap_nn.config.training_job_config.verify_training_cfg",
            side_effect=lambda cfg: cfg,
        ), patch(
            "sleap.gui.config_utils.filter_cfg",
            side_effect=lambda cfg: cfg,
        ):
            # Call write_pipeline_files
            write_pipeline_files(
                output_dir=str(output_dir),
                labels_filename=str(labels_path),
                config_info_list=[cfg_info],
                inference_params={},
                items_for_inference=items_for_inference,
            )

        # Read the generated train script
        train_script_path = output_dir / "train-script.sh"
        assert train_script_path.exists(), "train-script.sh should be created"
        train_script_content = train_script_path.read_text()

        # CRITICAL ASSERTION: train script should NOT contain zmq
        assert "zmq" not in train_script_content, (
            f"Train script should not contain zmq overrides. "
            f"Content:\n{train_script_content}"
        )

        # Verify the script has basic required content
        assert "sleap train" in train_script_content
        assert "--config-name" in train_script_content
        assert "--config-dir" in train_script_content

    def test_train_script_run_name_with_equals_survives_shell_and_hydra(
        self, tmp_path, mock_labels
    ):
        """Regression test for issue #2611.

        Auto-generated run names include "n=<num_labeled_frames>" (e.g.
        "260212_121024.centroid.n=181"), which contains a literal "=". The
        fix in #2612 wrapped Hydra override values in shell single quotes,
        but bash strips shell-level quotes before "sleap" ever sees the
        argument, so Hydra's own override parser still choked on the
        embedded "=" with `OverrideParseException`. The values must carry
        literal quote characters that survive shell word-splitting and are
        still present when Hydra parses argv.
        """
        import shlex

        from sleap.gui.learning.runners import write_pipeline_files
        from sleap.gui.learning.configs import ConfigFileInfo
        from omegaconf import OmegaConf
        from hydra.core.override_parser.overrides_parser import OverridesParser

        # No custom run_name, so write_pipeline_files auto-generates one that
        # includes "n=<num_user_labeled_frames>" (matches training behavior).
        config_dict = {
            "trainer_config": {
                "ckpt_dir": str(tmp_path / "models"),
                "run_name": "",
                "save_ckpt": True,
            },
            "data_config": {
                "train_labels_path": ["labels.slp"],
            },
            "model_config": {
                "head_configs": {},
            },
        }
        config = OmegaConf.create(config_dict)

        cfg_info = ConfigFileInfo(
            config=config,
            path="test_config.yaml",
            filename="test_config.yaml",
            head_name="centroid",
            dont_retrain=False,
        )

        output_dir = tmp_path / "export"
        output_dir.mkdir()
        labels_path = tmp_path / "labels.slp"
        labels_path.touch()

        video_item = VideoItemForInference(
            video=mock_labels.videos[0],
            frames=[0, 1, 2],
            labels_path=str(labels_path),
            video_idx=0,
        )
        items_for_inference = ItemsForInference(
            items=[video_item],
            total_frame_count=3,
        )

        with patch(
            "sleap_nn.config.training_job_config.verify_training_cfg",
            side_effect=lambda cfg: cfg,
        ), patch(
            "sleap.gui.config_utils.filter_cfg",
            side_effect=lambda cfg: cfg,
        ):
            write_pipeline_files(
                output_dir=str(output_dir),
                labels_filename=str(labels_path),
                config_info_list=[cfg_info],
                inference_params={},
                items_for_inference=items_for_inference,
                num_user_labeled_frames=181,
            )

        train_script_content = (output_dir / "train-script.sh").read_text()
        train_line = next(
            line for line in train_script_content.splitlines() if "sleap train" in line
        )

        # Sanity check: the auto-generated run_name does contain "=".
        assert "run_name=" in train_line
        assert ".n=181" in train_line

        # Emulate bash's word-splitting/quote-stripping (shlex.split matches
        # bash here since we only use single/double quotes, no globs/vars),
        # then feed the resulting argv to Hydra's real override parser --
        # this is the exact failure point from the traceback in #2611.
        argv = shlex.split(train_line)
        overrides = [a for a in argv if a.startswith("trainer_config.")]
        assert len(overrides) == 2

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(overrides)  # must not raise
        values = {p.key_or_group: p.value() for p in parsed}
        assert values["trainer_config.run_name"].endswith(".centroid.n=181")


class TestOutputPathGeneration:
    """Tests for output path generation."""

    @pytest.fixture
    def video_item(self, mock_video):
        """Create a video item for inference."""
        return VideoItemForInference(
            video=mock_video,
            frames=[0],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

    def test_explicit_output_path(self, mock_labels, video_item):
        """Explicit output path should be used."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, output_path = task.make_predict_cli_call(
            video_item, output_path="/custom/output.slp"
        )

        assert output_path == "/custom/output.slp"
        assert "-o" in cli_args
        o_idx = cli_args.index("-o") + 1
        assert cli_args[o_idx] == "/custom/output.slp"

    def test_auto_output_path_includes_timestamp(self, mock_labels, video_item):
        """Auto-generated output path should include timestamp."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/tmp/test_labels.slp",
        )

        with patch("sleap.gui.learning.runners.os.makedirs"):
            cli_args, output_path = task.make_predict_cli_call(video_item)

        # Should contain predictions and timestamp pattern
        assert "predictions" in output_path
        assert ".predictions.slp" in output_path

    def test_auto_output_path_named_after_video(self, mock_labels, video_item):
        """Auto-generated output path should be named after the video, once.

        The video name used to be added as a *prefix* to the `.slp` basename
        (`video.mp4_labels.slp....`) because `--data_path` was the project `.slp`.
        Now that `--data_path` is the video itself, the basename carries the video
        name directly -- so it must neither be duplicated nor lost.
        """
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/tmp/test_labels.slp",
        )

        with patch("sleap.gui.learning.runners.os.makedirs"):
            _, output_path = task.make_predict_cli_call(video_item)

        name = Path(output_path).name
        assert name.startswith("video.mp4.")
        assert name.endswith(".predictions.slp")
        assert name.count("video.mp4") == 1
        assert "video_None" not in name
        assert "test_labels.slp" not in name
        # Predictions still land in a `predictions` dir next to the project file.
        assert Path(output_path).parent.as_posix() == "/tmp/predictions"

    def test_auto_output_path_without_labels_filename_uses_video_dir(
        self, mock_labels, video_item
    ):
        """With no project filename, predictions land next to the video."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename=None,
        )

        _, output_path = task.make_predict_cli_call(video_item)

        assert Path(output_path).parent.as_posix() == "/path/to"
        assert Path(output_path).name.startswith("video.mp4.")

    def test_auto_output_path_disambiguates_embedded_videos(
        self, mock_labels, mock_embedded_video
    ):
        """Embedded videos share one filename, so the index must disambiguate.

        Every video in a `.pkg.slp` reports the same `filename` (the project file), so
        per-video runs would otherwise collide on a single output path.
        """
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/tmp/train.pkg.slp",
        )
        item = VideoItemForInference(
            video=mock_embedded_video,
            frames=[0, -80],
            labels_path="/tmp/train.pkg.slp",
            video_idx=3,
        )

        with patch("sleap.gui.learning.runners.os.makedirs"):
            _, output_path = task.make_predict_cli_call(item)

        name = Path(output_path).name
        assert name.startswith("video3_train.pkg.slp.")
        assert name.endswith(".predictions.slp")

    def test_auto_output_path_for_dataset_item_unchanged(self, mock_labels):
        """Annotation-status items should still be named after the project .slp."""
        task = InferenceTask(
            trained_job_paths=["/path/to/model"],
            inference_params={},
            labels=mock_labels,
            labels_filename="/tmp/test_labels.slp",
        )
        item = DatasetItemForInference(
            labels_path="/tmp/test_labels.slp", frame_filter="user"
        )

        with patch("sleap.gui.learning.runners.os.makedirs"):
            _, output_path = task.make_predict_cli_call(item)

        name = Path(output_path).name
        assert name.startswith("test_labels.slp.")
        assert name.endswith(".predictions.slp")


# =============================================================================
# Full CLI Integration Tests
# =============================================================================


class TestFullCLIIntegration:
    """Integration tests for complete CLI construction."""

    def test_full_inference_cli_with_all_options(self, mock_labels, mock_video):
        """Full inference CLI should include all configured options."""
        video_item = VideoItemForInference(
            video=mock_video,
            frames=[0, 1, 2, 3, 4],
            labels_path="/path/to/labels.slp",
            video_idx=0,
        )

        task = InferenceTask(
            trained_job_paths=[
                "/path/to/centroid/training_config.yaml",
                "/path/to/centered_instance/training_config.yaml",
            ],
            inference_params={
                "_batch_size": 16,
                "_max_instances": 10,
                "tracking.tracker": "flow",
                "tracking.match": "hungarian",
                "tracking.track_window": 15,
                "tracking.max_tracks": 5,
                "tracking.post_connect_single_breaks": 1,
                "tracking.robust": 0.8,
                "tracking.similarity": "oks",
            },
            labels=mock_labels,
            labels_filename="/path/to/labels.slp",
        )

        cli_args, output_path = task.make_predict_cli_call(
            video_item, output_path="/output/predictions.slp"
        )

        # Check base command
        assert cli_args[0] == "sleap"
        assert cli_args[1] == "predict"

        # Check data args
        assert "--data_path" in cli_args
        assert "--frames" in cli_args

        # Check model paths (2 models)
        assert cli_args.count("--model_paths") == 2

        # Check batch size
        assert "--batch_size" in cli_args
        assert "16" in cli_args

        # Check max instances
        assert "--max_instances" in cli_args
        assert "10" in cli_args

        # Check tracker args
        assert "--tracking" in cli_args
        assert "--use_flow" in cli_args
        assert "--track_matching_method" in cli_args
        assert "hungarian" in cli_args
        assert "--tracking_window_size" in cli_args
        assert "15" in cli_args
        assert "--max_tracks" in cli_args
        assert "5" in cli_args
        assert "--post_connect_single_breaks" in cli_args
        assert "--tracking_target_instance_count" in cli_args

        # Check similarity
        assert "--features" in cli_args
        assert "keypoints" in cli_args
        assert "--scoring_method" in cli_args
        assert "oks" in cli_args

        # Check robust
        assert "--scoring_reduction" in cli_args
        assert "robust_quantile" in cli_args
        assert "--robust_best_instance" in cli_args
        assert "0.8" in cli_args

        # Check output
        assert "-o" in cli_args
        assert "/output/predictions.slp" in cli_args

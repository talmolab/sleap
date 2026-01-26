"""Tests for sleap.sleap_io_adaptors.lf_labels_utils module."""

import pytest
from sleap_io import Labels, LabeledFrame, Video

from sleap.sleap_io_adaptors.lf_labels_utils import iterate_labeled_frames


@pytest.fixture
def labels_with_frames():
    """Create a Labels object with labeled frames at indices 10, 20, 30, 40, 50."""
    video = Video(filename="test_video.mp4")
    labels = Labels(videos=[video])

    # Create labeled frames at specific indices
    frame_indices = [10, 20, 30, 40, 50]
    for idx in frame_indices:
        lf = LabeledFrame(video=video, frame_idx=idx)
        labels.append(lf)

    return labels, video, frame_indices


class TestIterateLabeledFrames:
    """Tests for iterate_labeled_frames function."""

    def test_forward_iteration_basic(self, labels_with_frames):
        """Test basic forward iteration from beginning."""
        labels, video, frame_indices = labels_with_frames

        frames = list(iterate_labeled_frames(labels, video))
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices == [10, 20, 30, 40, 50]

    def test_forward_iteration_from_frame(self, labels_with_frames):
        """Test forward iteration starting from a specific frame."""
        labels, video, frame_indices = labels_with_frames

        # From frame 30, next should be 40
        frames = list(iterate_labeled_frames(labels, video, from_frame_idx=30))
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 40, "First frame should be 40 (next after 30)"
        assert result_indices == [40, 50, 10, 20, 30]

    def test_forward_iteration_from_unlabeled_frame(self, labels_with_frames):
        """Test forward iteration from a frame that isn't labeled."""
        labels, video, frame_indices = labels_with_frames

        # From frame 35, next should be 40
        frames = list(iterate_labeled_frames(labels, video, from_frame_idx=35))
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 40, "First frame should be 40 (next after 35)"

    def test_reverse_iteration_basic(self, labels_with_frames):
        """Test basic reverse iteration from end."""
        labels, video, frame_indices = labels_with_frames

        frames = list(iterate_labeled_frames(labels, video, reverse=True))
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices == [50, 40, 30, 20, 10]

    def test_reverse_iteration_from_frame(self, labels_with_frames):
        """Test reverse iteration starting from a specific frame.

        This is the main regression test for issue #2578.
        """
        labels, video, frame_indices = labels_with_frames

        # From frame 30, previous should be 20 (NOT 10!)
        frames = list(
            iterate_labeled_frames(labels, video, from_frame_idx=30, reverse=True)
        )
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 20, (
            f"First frame should be 20 (immediate previous to 30), "
            f"got {result_indices[0]}. Bug #2578."
        )
        assert result_indices == [20, 10, 50, 40, 30]

    def test_reverse_iteration_from_unlabeled_frame(self, labels_with_frames):
        """Test reverse iteration from a frame that isn't labeled.

        Additional regression test for issue #2578.
        """
        labels, video, frame_indices = labels_with_frames

        # From frame 35, previous should be 30 (NOT 20!)
        frames = list(
            iterate_labeled_frames(labels, video, from_frame_idx=35, reverse=True)
        )
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 30, (
            f"First frame should be 30 (immediate previous to 35), "
            f"got {result_indices[0]}. Bug #2578."
        )

    def test_reverse_iteration_wraps_at_beginning(self, labels_with_frames):
        """Test that reverse iteration wraps to end when at first frame."""
        labels, video, frame_indices = labels_with_frames

        # From frame 10 (first labeled), previous should wrap to 50
        frames = list(
            iterate_labeled_frames(labels, video, from_frame_idx=10, reverse=True)
        )
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 50, "Should wrap to last frame (50) when at first"

    def test_forward_iteration_wraps_at_end(self, labels_with_frames):
        """Test that forward iteration wraps to beginning when at last frame."""
        labels, video, frame_indices = labels_with_frames

        # From frame 50 (last labeled), next should wrap to 10
        frames = list(iterate_labeled_frames(labels, video, from_frame_idx=50))
        result_indices = [lf.frame_idx for lf in frames]

        assert result_indices[0] == 10, "Should wrap to first frame (10) when at last"

    def test_empty_labels(self):
        """Test iteration with no labeled frames."""
        video = Video(filename="test_video.mp4")
        labels = Labels(videos=[video])

        frames = list(iterate_labeled_frames(labels, video))
        assert frames == []

    def test_single_frame(self):
        """Test iteration with only one labeled frame."""
        video = Video(filename="test_video.mp4")
        labels = Labels(videos=[video])
        lf = LabeledFrame(video=video, frame_idx=25)
        labels.append(lf)

        # Forward
        frames = list(iterate_labeled_frames(labels, video, from_frame_idx=20))
        assert [f.frame_idx for f in frames] == [25]

        # Reverse
        frames = list(
            iterate_labeled_frames(labels, video, from_frame_idx=30, reverse=True)
        )
        assert [f.frame_idx for f in frames] == [25]

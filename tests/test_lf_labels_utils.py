"""Tests for sleap.sleap_io_adaptors.lf_labels_utils module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sleap.sleap_io_adaptors.lf_labels_utils import (
    make_video_callback,
    find_path_using_paths,
    fix_paths_with_saved_prefix,
)


class TestVideoCallbackImageSequences:
    """Tests for video_callback handling of image sequences."""

    def test_is_sequence_detection(self):
        """Test that image sequences (list filenames) are correctly identified."""
        # Create mock videos - one regular, one image sequence
        mock_regular_video = MagicMock()
        mock_regular_video.filename = "tests/data/videos/small_robot.mp4"

        mock_sequence_video = MagicMock()
        mock_sequence_video.filename = [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
            "tests/data/videos/robot2.jpg",
        ]

        video_list = [mock_regular_video, mock_sequence_video]

        # Create callback (non-GUI mode)
        callback = make_video_callback(search_paths=[], use_gui=False)

        # Call the callback - it should not raise
        callback(video_list)

        # Verify the filenames are still accessible
        assert mock_regular_video.filename == "tests/data/videos/small_robot.mp4"
        assert isinstance(mock_sequence_video.filename, list)

    def test_missing_detection_for_sequences(self):
        """Test that missing detection works for image sequences."""
        # Create mock video with missing image sequence
        mock_video = MagicMock()
        mock_video.filename = [
            "/nonexistent/path/frame0.jpg",
            "/nonexistent/path/frame1.jpg",
        ]

        video_list = [mock_video]
        context = {"changed_on_load": False}

        callback = make_video_callback(
            search_paths=[], use_gui=False, context=context
        )
        callback(video_list)

        # The video should still have its original filename since no replacement found
        assert mock_video.filename == [
            "/nonexistent/path/frame0.jpg",
            "/nonexistent/path/frame1.jpg",
        ]

    def test_existing_sequence_not_marked_missing(self):
        """Test that existing image sequences are not marked as missing."""
        # Create mock video with existing image sequence
        mock_video = MagicMock()
        mock_video.filename = [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
            "tests/data/videos/robot2.jpg",
        ]

        video_list = [mock_video]
        context = {"changed_on_load": False}

        callback = make_video_callback(
            search_paths=[], use_gui=False, context=context
        )
        callback(video_list)

        # The filename should remain unchanged
        assert mock_video.filename == [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
            "tests/data/videos/robot2.jpg",
        ]

    def test_mixed_videos_and_sequences(self):
        """Test callback with both regular videos and image sequences."""
        mock_regular = MagicMock()
        mock_regular.filename = "tests/data/videos/small_robot.mp4"

        mock_sequence = MagicMock()
        mock_sequence.filename = [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
        ]

        video_list = [mock_regular, mock_sequence]
        context = {"changed_on_load": False}

        callback = make_video_callback(
            search_paths=[], use_gui=False, context=context
        )
        callback(video_list)

        # Both should retain their original format
        assert isinstance(mock_regular.filename, str)
        assert isinstance(mock_sequence.filename, list)


class TestFindPathUsingPaths:
    """Tests for find_path_using_paths function."""

    def test_returns_list_unchanged(self):
        """Test that image sequence lists are returned unchanged."""
        filename = [
            "/some/path/frame0.jpg",
            "/some/path/frame1.jpg",
        ]
        result = find_path_using_paths(filename, ["tests/data/videos"])

        # Should return the list unchanged
        assert result == filename
        assert isinstance(result, list)

    def test_finds_regular_file(self):
        """Test that regular files can be found in search paths."""
        filename = "/nonexistent/small_robot.mp4"
        result = find_path_using_paths(filename, ["tests/data/videos"])

        # Should find the file in the search path
        assert result == "tests/data/videos/small_robot.mp4"

    def test_returns_original_if_not_found(self):
        """Test that original filename is returned if not found."""
        filename = "/nonexistent/totally_missing.mp4"
        result = find_path_using_paths(filename, ["tests/data/videos"])

        # Should return original since file doesn't exist
        assert result == filename


class TestFixPathsWithSavedPrefix:
    """Tests for fix_paths_with_saved_prefix function."""

    def test_skips_image_sequences(self):
        """Test that image sequences are skipped during prefix conversion."""
        filenames = [
            "/old/path/video.mp4",
            ["/old/path/frame0.jpg", "/old/path/frame1.jpg"],  # Sequence
        ]
        missing = [True, True]

        # Mock the config to return a prefix conversion
        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.util.get_config_yaml"
        ) as mock_config:
            mock_config.return_value = {"/old/path": "/new/path"}
            fix_paths_with_saved_prefix(filenames, missing)

        # The sequence should be unchanged (skipped)
        assert filenames[1] == ["/old/path/frame0.jpg", "/old/path/frame1.jpg"]

    def test_handles_empty_sequence(self):
        """Test that empty sequences don't cause errors."""
        filenames = [
            "/some/path/video.mp4",
            [],  # Empty sequence
        ]
        missing = [True, True]

        # Should not raise
        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.util.get_config_yaml"
        ) as mock_config:
            mock_config.return_value = None
            fix_paths_with_saved_prefix(filenames, missing)

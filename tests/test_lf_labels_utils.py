"""Tests for sleap.sleap_io_adaptors.lf_labels_utils module."""

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

        callback = make_video_callback(search_paths=[], use_gui=False, context=context)
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

        callback = make_video_callback(search_paths=[], use_gui=False, context=context)
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

        callback = make_video_callback(search_paths=[], use_gui=False, context=context)
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
        # Use Path for comparison to handle Windows vs Unix path separators
        assert Path(result) == Path("tests/data/videos/small_robot.mp4")

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


class TestVideoCallbackSearchPaths:
    """Tests for video_callback search path functionality."""

    def test_search_paths_find_missing_video(self):
        """Test that search paths can find a missing video."""
        mock_video = MagicMock()
        mock_video.filename = "/nonexistent/path/small_robot.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        callback = make_video_callback(
            search_paths=["tests/data/videos"], use_gui=False, context=context
        )
        callback(video_list)

        # The video should have been found and replaced
        assert "small_robot.mp4" in mock_video.replace_filename.call_args[0][0]
        assert context["changed_on_load"] is True

    def test_search_paths_no_match(self):
        """Test that search paths that don't contain the file leave it unchanged."""
        mock_video = MagicMock()
        mock_video.filename = "/nonexistent/path/totally_missing.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        callback = make_video_callback(
            search_paths=["tests/data/videos"], use_gui=False, context=context
        )
        callback(video_list)

        # The video filename should be unchanged (still missing)
        mock_video.replace_filename.assert_called_once_with(
            "/nonexistent/path/totally_missing.mp4"
        )


class TestVideoCallbackExtensionMatching:
    """Tests for non-GUI extension matching in video_callback."""

    def test_extension_matching_replaces_paths(self):
        """Test that when extensions match, paths are replaced in order."""
        mock_video1 = MagicMock()
        mock_video1.filename = "/old/path/video1.mp4"

        mock_video2 = MagicMock()
        mock_video2.filename = "/old/path/video2.mp4"

        video_list = [mock_video1, mock_video2]
        context = {"changed_on_load": False}

        # Provide exactly the same number of new paths with matching extensions
        new_paths = [
            "tests/data/videos/small_robot.mp4",
            "tests/data/videos/centered_pair_small.mp4",
        ]

        callback = make_video_callback(
            search_paths=new_paths, use_gui=False, context=context
        )
        callback(video_list)

        # Both videos should be replaced with the new paths
        mock_video1.replace_filename.assert_called_once()
        mock_video2.replace_filename.assert_called_once()
        assert context["changed_on_load"] is True

    def test_extension_mismatch_no_replacement(self):
        """Test that mismatched extensions prevent replacement."""
        mock_video = MagicMock()
        mock_video.filename = "/old/path/video.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        # Provide new path with different extension
        new_paths = ["/new/path/video.avi"]

        callback = make_video_callback(
            search_paths=new_paths, use_gui=False, context=context
        )
        callback(video_list)

        # Video should keep original path (extensions don't match)
        mock_video.replace_filename.assert_called_once_with("/old/path/video.mp4")

    def test_extension_matching_skips_sequences(self):
        """Test that image sequences are skipped during extension matching."""
        mock_video = MagicMock()
        mock_video.filename = "/old/path/video.mp4"

        mock_sequence = MagicMock()
        mock_sequence.filename = ["/old/path/frame0.jpg", "/old/path/frame1.jpg"]

        video_list = [mock_video, mock_sequence]
        context = {"changed_on_load": False}

        # Provide two new paths (same count as videos)
        new_paths = ["/new/path/video.mp4", "/new/path/other.jpg"]

        callback = make_video_callback(
            search_paths=new_paths, use_gui=False, context=context
        )
        callback(video_list)

        # Regular video gets replaced, sequence stays as-is
        mock_video.replace_filename.assert_called_once()
        # Sequence should be called with original list
        mock_sequence.replace_filename.assert_called_once()

    def test_extension_matching_with_image_sequence_extensions(self):
        """Test extension extraction from image sequences."""
        mock_sequence = MagicMock()
        mock_sequence.filename = ["/old/path/frame0.tif", "/old/path/frame1.tif"]

        video_list = [mock_sequence]
        context = {"changed_on_load": False}

        # Provide path with matching tif extension
        new_paths = ["/new/path/frames.tif"]

        callback = make_video_callback(
            search_paths=new_paths, use_gui=False, context=context
        )
        callback(video_list)

        # Sequence should not be replaced (skipped for extension matching)
        mock_sequence.replace_filename.assert_called_once()


class TestVideoCallbackEmptySequence:
    """Tests for handling empty image sequences."""

    def test_empty_sequence_marked_missing(self):
        """Test that empty sequences are marked as missing."""
        mock_video = MagicMock()
        mock_video.filename = []  # Empty sequence

        video_list = [mock_video]
        context = {"changed_on_load": False}

        callback = make_video_callback(search_paths=[], use_gui=False, context=context)
        callback(video_list)

        # Empty sequence should remain empty
        mock_video.replace_filename.assert_called_once_with([])


class TestFixPathsWithSavedPrefixExisting:
    """Additional tests for fix_paths_with_saved_prefix."""

    def test_skips_existing_files(self):
        """Test that existing files are not modified."""
        filenames = [
            "tests/data/videos/small_robot.mp4",  # Exists
            "/nonexistent/video.mp4",  # Missing
        ]
        missing = None  # Let function determine missing status

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.util.get_config_yaml"
        ) as mock_config:
            mock_config.return_value = {"tests/data": "/other/path"}
            fix_paths_with_saved_prefix(filenames, missing)

        # Existing file should not be modified
        assert filenames[0] == "tests/data/videos/small_robot.mp4"

    def test_existing_sequence_not_modified(self):
        """Test that existing image sequences are not modified."""
        filenames = [
            ["tests/data/videos/robot0.jpg", "tests/data/videos/robot1.jpg"],  # Exists
        ]
        missing = None  # Let function determine missing status

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.util.get_config_yaml"
        ) as mock_config:
            mock_config.return_value = {"tests/data": "/other/path"}
            fix_paths_with_saved_prefix(filenames, missing)

        # Existing sequence should not be modified
        assert filenames[0] == [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
        ]

    def test_prefix_conversion_applied(self):
        """Test that prefix conversion is applied to missing files."""
        # Create a file that will exist after prefix conversion
        filenames = ["/old/prefix/videos/small_robot.mp4"]
        missing = [True]

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.util.get_config_yaml"
        ) as mock_config:
            mock_config.return_value = {"/old/prefix": "tests/data"}
            fix_paths_with_saved_prefix(filenames, missing)

        # File should be converted and found
        assert Path(filenames[0]) == Path("tests/data/videos/small_robot.mp4")
        assert missing[0] is False


class TestVideoCallbackGUI:
    """Tests for GUI code path in video_callback."""

    def test_gui_dialog_shown_when_missing(self):
        """Test that GUI dialog is shown when there are missing videos."""
        mock_video = MagicMock()
        mock_video.filename = "/nonexistent/path/video.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            # Simulate user accepting the dialog
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec_.return_value = True
            MockDialog.return_value = mock_dialog_instance

            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            callback(video_list)

            # Dialog should have been created with display_filenames
            MockDialog.assert_called_once()
            call_args = MockDialog.call_args
            # First argument is display_filenames list
            assert call_args[0][0] == ["/nonexistent/path/video.mp4"]
            # is_sequence should be passed
            assert call_args[1]["is_sequence"] == [False]

    def test_gui_dialog_abort_returns_true(self):
        """Test that aborting the dialog returns True (stop)."""
        mock_video = MagicMock()
        mock_video.filename = "/nonexistent/path/video.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            # Simulate user aborting the dialog
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec_.return_value = False
            MockDialog.return_value = mock_dialog_instance

            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            result = callback(video_list)

            # Should return True (abort signal)
            assert result is True

    def test_gui_dialog_with_image_sequence(self):
        """Test that image sequences show directory path in dialog."""
        mock_sequence = MagicMock()
        mock_sequence.filename = [
            "/nonexistent/images/frame0.jpg",
            "/nonexistent/images/frame1.jpg",
        ]

        video_list = [mock_sequence]
        context = {"changed_on_load": False}

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec_.return_value = True
            MockDialog.return_value = mock_dialog_instance

            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            callback(video_list)

            # Dialog should show parent directory for sequences
            call_args = MockDialog.call_args
            display_filenames = call_args[0][0]
            # Should show "/nonexistent/images" (parent dir)
            assert display_filenames[0] == "/nonexistent/images"
            assert call_args[1]["is_sequence"] == [True]

    def test_gui_dialog_remaps_sequence_paths(self):
        """Test that sequence paths are remapped after user selects new directory."""
        mock_sequence = MagicMock()
        original_frames = [
            "/old/images/frame0.jpg",
            "/old/images/frame1.jpg",
        ]
        mock_sequence.filename = original_frames.copy()

        video_list = [mock_sequence]
        context = {"changed_on_load": False}

        def side_effect_exec():
            # Simulate user selecting a new directory
            # The dialog modifies display_filenames in place
            call_args = MockDialog.call_args
            display_filenames = call_args[0][0]
            missing = call_args[0][1]
            # Simulate user found the directory
            display_filenames[0] = "/new/images"
            missing[0] = False
            return True

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec_.side_effect = side_effect_exec
            MockDialog.return_value = mock_dialog_instance

            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            callback(video_list)

            # The replace_filename should be called with remapped paths
            mock_sequence.replace_filename.assert_called_once()
            new_paths = mock_sequence.replace_filename.call_args[0][0]
            assert new_paths == ["/new/images/frame0.jpg", "/new/images/frame1.jpg"]

    def test_gui_not_shown_when_no_missing(self):
        """Test that GUI dialog is not shown when all files exist."""
        mock_video = MagicMock()
        mock_video.filename = "tests/data/videos/small_robot.mp4"

        video_list = [mock_video]
        context = {"changed_on_load": False}

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            callback(video_list)

            # Dialog should not be created since file exists
            MockDialog.assert_not_called()

    def test_gui_dialog_mixed_videos_and_sequences(self):
        """Test GUI dialog with both regular videos and image sequences."""
        mock_video = MagicMock()
        mock_video.filename = "/nonexistent/video.mp4"

        mock_sequence = MagicMock()
        mock_sequence.filename = [
            "/nonexistent/images/frame0.jpg",
            "/nonexistent/images/frame1.jpg",
        ]

        video_list = [mock_video, mock_sequence]
        context = {"changed_on_load": False}

        with patch(
            "sleap.sleap_io_adaptors.lf_labels_utils.MissingFilesDialog"
        ) as MockDialog:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec_.return_value = True
            MockDialog.return_value = mock_dialog_instance

            callback = make_video_callback(
                search_paths=[], use_gui=True, context=context
            )
            callback(video_list)

            call_args = MockDialog.call_args
            display_filenames = call_args[0][0]
            # Regular video shows full path, sequence shows directory
            assert display_filenames[0] == "/nonexistent/video.mp4"
            assert display_filenames[1] == "/nonexistent/images"
            assert call_args[1]["is_sequence"] == [False, True]


class TestFindPathUsingPathsEdgeCases:
    """Additional edge case tests for find_path_using_paths."""

    def test_empty_search_paths(self):
        """Test with empty search paths list."""
        filename = "/nonexistent/video.mp4"
        result = find_path_using_paths(filename, [])
        assert result == filename

    def test_search_path_is_file_not_dir(self):
        """Test that file paths in search_paths are handled."""
        filename = "/nonexistent/small_robot.mp4"
        # Pass a file path instead of directory
        result = find_path_using_paths(filename, ["tests/data/videos/small_robot.mp4"])
        # Should return original since search path is not a directory
        assert result == filename

    def test_empty_filename_list(self):
        """Test with empty list filename."""
        filename = []
        result = find_path_using_paths(filename, ["tests/data/videos"])
        assert result == []

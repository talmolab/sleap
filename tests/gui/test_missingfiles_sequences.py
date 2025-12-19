"""Tests for MissingFilesDialog image sequence handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from sleap_io import Video


class TestFindPathUsingPaths:
    """Tests for find_path_using_paths function with list inputs."""

    def test_find_path_with_list_returns_unchanged(self):
        """Test that list inputs (image sequences) are returned unchanged."""
        from sleap.sleap_io_adaptors.lf_labels_utils import find_path_using_paths

        frame_list = ["/old/path/frame001.jpg", "/old/path/frame002.jpg"]
        search_paths = ["tests/data/videos"]

        result = find_path_using_paths(frame_list, search_paths)

        # List should be returned unchanged
        assert result == frame_list
        assert result is frame_list  # Same object

    def test_find_path_with_string_searches_directories(self):
        """Test that string paths are searched in directories."""
        from sleap.sleap_io_adaptors.lf_labels_utils import find_path_using_paths

        # Use a file that exists
        filename = "nonexistent/small_robot.mp4"
        search_paths = ["tests/data/videos"]

        result = find_path_using_paths(filename, search_paths)

        # Should find the file in search path (use Path for cross-platform)
        expected = str(Path("tests/data/videos/small_robot.mp4"))
        assert result == expected

    def test_find_path_returns_original_if_not_found(self):
        """Test that original path is returned if file not found."""
        from sleap.sleap_io_adaptors.lf_labels_utils import find_path_using_paths

        filename = "nonexistent/truly_missing.mp4"
        search_paths = ["tests/data/videos"]

        result = find_path_using_paths(filename, search_paths)

        # Should return original
        assert result == filename


class TestFixPathsWithSavedPrefixSequences:
    """Tests for fix_paths_with_saved_prefix with image sequences."""

    def test_fix_paths_with_image_sequence(self):
        """Test fixing paths for image sequences."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        # Create a list with an image sequence
        filenames = [
            [
                "foo/robot0.jpg",
                "foo/robot1.jpg",
                "foo/robot2.jpg",
            ]
        ]
        missing = [True]
        path_prefix_conversions = {"foo": "tests/data/videos"}

        fix_paths_with_saved_prefix(
            filenames, missing=missing, path_prefix_conversions=path_prefix_conversions
        )

        # Check that paths were fixed
        assert filenames[0] == [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
            "tests/data/videos/robot2.jpg",
        ]
        assert missing[0] is False

    def test_fix_paths_with_empty_sequence(self):
        """Test that empty sequences are skipped."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        filenames = [[]]  # Empty sequence
        missing = [True]
        path_prefix_conversions = {"foo": "tests/data/videos"}

        fix_paths_with_saved_prefix(
            filenames, missing=missing, path_prefix_conversions=path_prefix_conversions
        )

        # Should remain unchanged
        assert filenames[0] == []
        assert missing[0] is True

    def test_fix_paths_with_empty_first_frame(self):
        """Test that sequences with empty first frame are skipped."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        filenames = [["", "foo/robot1.jpg"]]  # Empty first frame
        missing = [True]
        path_prefix_conversions = {"foo": "tests/data/videos"}

        fix_paths_with_saved_prefix(
            filenames, missing=missing, path_prefix_conversions=path_prefix_conversions
        )

        # Should remain unchanged because first frame is empty
        assert filenames[0] == ["", "foo/robot1.jpg"]

    def test_fix_paths_sequence_not_missing(self):
        """Test that non-missing sequences are skipped."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        filenames = [["foo/robot0.jpg", "foo/robot1.jpg"]]
        missing = [False]  # Not missing
        path_prefix_conversions = {"foo": "tests/data/videos"}

        fix_paths_with_saved_prefix(
            filenames, missing=missing, path_prefix_conversions=path_prefix_conversions
        )

        # Should remain unchanged
        assert filenames[0] == ["foo/robot0.jpg", "foo/robot1.jpg"]

    def test_fix_paths_regular_video_not_missing_no_missing_list(self):
        """Test regular video handling when missing list is None and file exists."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        filenames = ["tests/data/videos/small_robot.mp4"]  # File exists
        path_prefix_conversions = {"old": "new"}

        fix_paths_with_saved_prefix(
            filenames, missing=None, path_prefix_conversions=path_prefix_conversions
        )

        # Should remain unchanged because file exists
        assert filenames[0] == "tests/data/videos/small_robot.mp4"

    def test_fix_paths_none_conversions(self):
        """Test that None conversions returns early."""
        from sleap.sleap_io_adaptors.lf_labels_utils import fix_paths_with_saved_prefix

        filenames = ["foo/video.mp4"]
        missing = [True]

        # Mock get_config_yaml to return None
        with patch("sleap.util.get_config_yaml", return_value=None):
            fix_paths_with_saved_prefix(filenames, missing=missing)

        # Should remain unchanged
        assert filenames[0] == "foo/video.mp4"


class TestVideoCallbackSequences:
    """Tests for video_callback with image sequence handling."""

    def test_video_callback_detects_image_sequences(self):
        """Test that video_callback correctly detects image sequences."""
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        # Create mock videos
        mock_video1 = MagicMock(spec=Video)
        mock_video1.filename = "tests/data/videos/small_robot.mp4"
        mock_video1.backend = MagicMock()
        mock_video1.backend._cached_shape = None
        mock_video1.backend_metadata = {"shape": [10, 480, 640, 3]}

        mock_video2 = MagicMock(spec=Video)
        mock_video2.filename = [
            "tests/data/videos/robot0.jpg",
            "tests/data/videos/robot1.jpg",
        ]
        mock_video2.backend = MagicMock()
        mock_video2.backend._cached_shape = None
        mock_video2.backend_metadata = {"shape": [2, 100, 100, 3]}

        video_list = [mock_video1, mock_video2]
        callback = make_video_callback(use_gui=False)

        # Should not raise - both videos exist
        result = callback(video_list)

        # Should not abort
        assert result is None

    def test_video_callback_search_path_finds_file(self):
        """Test that search_paths can find a missing video file."""
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        mock_video = MagicMock(spec=Video)
        mock_video.filename = "/nonexistent/small_robot.mp4"  # File "missing"
        mock_video.backend = MagicMock()
        mock_video.backend._cached_shape = None
        mock_video.backend_metadata = {"shape": [10, 480, 640, 3]}

        video_list = [mock_video]
        callback = make_video_callback(
            search_paths=["tests/data/videos"], use_gui=False
        )

        result = callback(video_list)
        assert result is None

        # replace_filename should have been called with the found path
        mock_video.replace_filename.assert_called_once()

    def test_video_callback_only_replaces_changed_filenames(self):
        """Test that replace_filename is only called when filename changed."""
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        mock_video = MagicMock(spec=Video)
        mock_video.filename = "tests/data/videos/small_robot.mp4"
        mock_video.backend = MagicMock()
        mock_video.backend._cached_shape = None
        mock_video.backend_metadata = {"shape": [10, 480, 640, 3]}

        video_list = [mock_video]
        callback = make_video_callback(use_gui=False)

        callback(video_list)

        # replace_filename should NOT be called since path didn't change
        mock_video.replace_filename.assert_not_called()

    def test_video_callback_empty_sequence(self):
        """Test handling of empty image sequence."""
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        mock_video = MagicMock(spec=Video)
        mock_video.filename = []  # Empty sequence
        mock_video.backend = MagicMock()
        mock_video.backend._cached_shape = None
        mock_video.backend_metadata = {"shape": [0, 100, 100, 3]}

        video_list = [mock_video]
        callback = make_video_callback(use_gui=False)

        result = callback(video_list)
        assert result is None

    def test_video_callback_path_object_in_sequence(self):
        """Test handling of Path objects in sequence filenames."""
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        mock_video = MagicMock(spec=Video)
        # Use Path objects instead of strings
        mock_video.filename = [
            Path("tests/data/videos/robot0.jpg"),
            Path("tests/data/videos/robot1.jpg"),
        ]
        mock_video.backend = MagicMock()
        mock_video.backend._cached_shape = None
        mock_video.backend_metadata = {"shape": [2, 100, 100, 3]}

        video_list = [mock_video]
        callback = make_video_callback(use_gui=False)

        # Should handle Path objects without error
        result = callback(video_list)
        assert result is None


class TestFilenamesPrefixChangeSequences:
    """Tests for filenames_prefix_change with image sequence support."""

    def test_prefix_change_with_sequence(self):
        """Test prefix change for image sequences."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/robot0.jpg"]  # Display path for sequence
        original_filenames = [
            ["foo/robot0.jpg", "foo/robot1.jpg", "foo/robot2.jpg"]
        ]  # Full sequence
        is_sequence = [True]
        missing = [True]

        filenames_prefix_change(
            filenames,
            "foo",
            "tests/data/videos",
            missing=missing,
            is_sequence=is_sequence,
            original_filenames=original_filenames,
        )

        # Display path should be updated
        assert filenames[0] == "tests/data/videos/robot0.jpg"
        assert missing[0] is False

    def test_prefix_change_sequence_first_frame_not_found(self):
        """Test sequence skipped if first frame doesn't exist at new location."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/missing.jpg"]
        original_filenames = [["foo/missing.jpg", "foo/missing2.jpg"]]
        is_sequence = [True]
        missing = [True]

        filenames_prefix_change(
            filenames,
            "foo",
            "tests/data/videos",  # robot0.jpg exists but missing.jpg doesn't
            missing=missing,
            is_sequence=is_sequence,
            original_filenames=original_filenames,
        )

        # Should remain unchanged
        assert filenames[0] == "foo/missing.jpg"
        assert missing[0] is True

    def test_prefix_change_handles_non_list_original_filename(self):
        """Test handling when original_filenames entry is not a list."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/robot0.jpg"]
        original_filenames = ["foo/robot0.jpg"]  # Not a list
        is_sequence = [True]
        missing = [True]

        filenames_prefix_change(
            filenames,
            "foo",
            "tests/data/videos",
            missing=missing,
            is_sequence=is_sequence,
            original_filenames=original_filenames,
        )

        # Should handle gracefully - treat single item as list
        assert filenames[0] == "tests/data/videos/robot0.jpg"

    def test_prefix_change_no_original_filenames(self):
        """Test that is_sequence and original_filenames default to sensible values."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/small_robot.mp4", "foo/does_not_exist.mp4"]

        # Don't pass is_sequence or original_filenames
        filenames_prefix_change(filenames, "foo", "tests/data/videos")

        # First path should be fixed (file exists)
        assert filenames[0] == "tests/data/videos/small_robot.mp4"
        # Second should remain (file doesn't exist)
        assert filenames[1] == "foo/does_not_exist.mp4"

    def test_prefix_change_empty_filenames(self):
        """Test with empty filenames list."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = []
        filenames_prefix_change(filenames, "foo", "bar")
        assert filenames == []

    def test_prefix_change_empty_old_prefix(self):
        """Test with empty old_prefix."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/video.mp4"]
        filenames_prefix_change(filenames, "", "bar")
        assert filenames == ["foo/video.mp4"]

    def test_prefix_change_empty_new_prefix(self):
        """Test with empty new_prefix."""
        from sleap.io.pathutils import filenames_prefix_change

        filenames = ["foo/video.mp4"]
        filenames_prefix_change(filenames, "foo", "")
        assert filenames == ["foo/video.mp4"]


class TestMissingFilesDialogSequences:
    """Tests for MissingFilesDialog with image sequence support."""

    def test_dialog_init_with_sequences(self, qtbot):
        """Test dialog initialization with sequence parameters."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/frame001.jpg", "/other/video.mp4"]
        is_sequence = [True, False]
        original_filenames = [["/path/frame001.jpg", "/path/frame002.jpg"], None]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
            original_filenames=original_filenames,
        )
        qtbot.addWidget(dialog)

        assert dialog.is_sequence == is_sequence
        assert dialog.original_filenames == original_filenames

    def test_dialog_init_defaults(self, qtbot):
        """Test dialog initialization with default sequence parameters."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/video1.mp4", "/path/video2.mp4"]

        dialog = MissingFilesDialog(filenames)
        qtbot.addWidget(dialog)

        # Should default to all False
        assert dialog.is_sequence == [False, False]
        # Should default to filenames
        assert dialog.original_filenames == filenames

    def test_set_filename_calls_prefix_change_with_sequence_params(self, qtbot):
        """Test that setFilename passes sequence params to prefix_change."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["m:/old/video.mp4"]
        is_sequence = [False]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # Mock the prefix change function to verify parameters
        with patch("sleap.io.pathutils.filenames_prefix_change") as mock_prefix:
            dialog.setFilename(0, "tests/data/videos/small_robot.mp4", confirm=False)

            # Verify is_sequence was passed
            mock_prefix.assert_called_once()
            call_kwargs = mock_prefix.call_args[1]
            assert "is_sequence" in call_kwargs
            assert call_kwargs["is_sequence"] == is_sequence

    def test_locate_file_sequence_caption(self, qtbot):
        """Test that locateFile uses correct caption for sequences."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/frame001.jpg"]
        is_sequence = [True]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # Mock FileDialog.open to return None (cancel)
        with patch(
            "sleap.gui.dialogs.missingfiles.FileDialog.open", return_value=("", None)
        ):
            # Should not raise, should return early
            dialog.locateFile(0)

    def test_locate_file_regular_video_caption(self, qtbot):
        """Test that locateFile uses correct caption for regular videos."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/video.mp4"]
        is_sequence = [False]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # Mock FileDialog.open to return None (cancel)
        with patch(
            "sleap.gui.dialogs.missingfiles.FileDialog.open", return_value=("", None)
        ):
            dialog.locateFile(0)

    def test_locate_file_with_new_file_selection(self, qtbot):
        """Test locateFile when user selects a new file."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/old/path/video.mp4"]
        is_sequence = [False]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # Mock FileDialog.open to return a new path
        with patch(
            "sleap.gui.dialogs.missingfiles.FileDialog.open",
            return_value=("tests/data/videos/small_robot.mp4", None),
        ):
            with patch.object(dialog, "setFilename") as mock_set:
                dialog.locateFile(0)
                mock_set.assert_called_once_with(0, "tests/data/videos/small_robot.mp4")

    def test_locate_file_duplicate_prevention(self, qtbot):
        """Test that locateFile prevents duplicate file selection."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/video1.mp4", "tests/data/videos/small_robot.mp4"]
        is_sequence = [False, False]

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # Mock FileDialog.open to return an already existing path
        # Also mock QMessageBox to prevent popup during test
        with patch(
            "sleap.gui.dialogs.missingfiles.FileDialog.open",
            return_value=("tests/data/videos/small_robot.mp4", None),
        ):
            with patch(
                "sleap.gui.dialogs.missingfiles.QtWidgets.QMessageBox"
            ) as mock_msgbox:
                with patch.object(dialog, "setFilename") as mock_set:
                    # Should not call setFilename due to duplicate
                    dialog.locateFile(0)
                    mock_set.assert_not_called()
                    # Verify the message box was shown
                    mock_msgbox.assert_called_once()

    def test_locate_file_sequence_allows_duplicates(self, qtbot):
        """Test that sequences don't have duplicate prevention."""
        from sleap.gui.dialogs.missingfiles import MissingFilesDialog

        filenames = ["/path/frame001.jpg", "/other/frame001.jpg"]
        is_sequence = [True, True]  # Both are sequences

        dialog = MissingFilesDialog(
            filenames,
            is_sequence=is_sequence,
        )
        qtbot.addWidget(dialog)

        # For sequences, duplicate check is skipped
        with patch(
            "sleap.gui.dialogs.missingfiles.FileDialog.open",
            return_value=("tests/data/videos/robot0.jpg", None),
        ):
            with patch.object(dialog, "setFilename") as mock_set:
                dialog.locateFile(0)
                # Should call setFilename (no duplicate check for sequences)
                mock_set.assert_called_once()


class TestIntegration:
    """Integration tests using real Video objects."""

    def test_real_image_video_loading(self, small_robot_single_image_vid):
        """Test with real ImageVideo backend."""
        video = small_robot_single_image_vid

        # Verify it's an ImageVideo (list of filenames)
        assert isinstance(video.filename, list)
        assert len(video.filename) == 3

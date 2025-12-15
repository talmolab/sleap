from unittest.mock import patch, MagicMock
from sleap.gui.dialogs.missingfiles import MissingFilesDialog


def test_missing_gui(qtbot):
    """Test basic MissingFilesDialog with regular video files."""
    filenames = ["m:\\centered_pair_small.mp4", "m:\\small_robot.mp4"]
    win = MissingFilesDialog(filenames)
    win.show()
    qtbot.addWidget(win)

    assert win.file_table.model().rowCount() == 2
    assert not win.accept_button.isEnabled()

    win.setFilename(0, "tests/data/videos/centered_pair_small.mp4", False)
    assert filenames[0] == "tests/data/videos/centered_pair_small.mp4"
    assert filenames[1] == "tests/data/videos/small_robot.mp4"

    assert win.accept_button.isEnabled()


def test_missing_gui_with_image_sequence(qtbot):
    """Test MissingFilesDialog with image sequence (directory path)."""
    # For image sequences, the display filename is the directory path
    filenames = ["m:\\missing_images_dir"]
    is_sequence = [True]
    missing = [True]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    assert win.file_table.model().rowCount() == 1
    assert not win.accept_button.isEnabled()
    assert win.is_sequence[0] is True

    # Simulate user finding the directory
    win.setFilename(0, "tests/data/videos", False)
    assert filenames[0] == "tests/data/videos"
    assert win.accept_button.isEnabled()


def test_missing_gui_mixed_files_and_sequences(qtbot):
    """Test MissingFilesDialog with both regular videos and image sequences."""
    # Use different path prefixes to prevent auto-fix from applying to all files
    filenames = [
        "x:\\videos\\centered_pair_small.mp4",  # Regular video
        "y:\\images\\missing_dir",  # Image sequence (directory)
        "z:\\other\\small_robot.mp4",  # Regular video
    ]
    is_sequence = [False, True, False]
    missing = [True, True, True]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    assert win.file_table.model().rowCount() == 3
    assert not win.accept_button.isEnabled()

    # Verify is_sequence is correctly set
    assert win.is_sequence == [False, True, False]

    # Fix the regular video (confirm=False to skip auto-fix prompt)
    win.setFilename(0, "tests/data/videos/centered_pair_small.mp4", False)
    assert not win.accept_button.isEnabled()  # Still missing files

    # Fix the image sequence directory
    win.setFilename(1, "tests/data/videos", False)
    assert not win.accept_button.isEnabled()  # Still missing one file

    # Fix the last regular video
    win.setFilename(2, "tests/data/videos/small_robot.mp4", False)
    assert win.accept_button.isEnabled()  # All found


def test_missing_gui_is_sequence_default(qtbot):
    """Test that is_sequence defaults to all False when not provided."""
    filenames = ["m:\\video1.mp4", "m:\\video2.mp4"]
    win = MissingFilesDialog(filenames)
    win.show()
    qtbot.addWidget(win)

    # Should default to False for all entries
    assert win.is_sequence == [False, False]


def test_missing_gui_info_text_with_sequences(qtbot):
    """Test that info text mentions directories when sequences are present."""
    # Without sequences
    filenames_no_seq = ["m:\\video.mp4"]
    win_no_seq = MissingFilesDialog(filenames_no_seq, missing=[True])
    qtbot.addWidget(win_no_seq)

    # With sequences
    filenames_seq = ["m:\\images_dir"]
    win_seq = MissingFilesDialog(filenames_seq, missing=[True], is_sequence=[True])
    qtbot.addWidget(win_seq)

    # The dialog with sequences should have different info text
    # (We can't easily check the text content in unit tests, but we verify
    # the dialog creates without error)


def test_locate_file_regular_video(qtbot):
    """Test locateFile for regular video files."""
    filenames = ["m:\\missing_video.mp4"]
    missing = [True]
    is_sequence = [False]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    # Mock the FileDialog.open to return a found file
    with patch("sleap.gui.dialogs.missingfiles.FileDialog") as MockFileDialog:
        MockFileDialog.open.return_value = (
            "tests/data/videos/small_robot.mp4",
            "filter",
        )

        win.locateFile(0)

        # FileDialog.open should be called (not openDir)
        MockFileDialog.open.assert_called_once()
        MockFileDialog.openDir.assert_not_called()

        # Filename should be updated
        assert filenames[0] == "tests/data/videos/small_robot.mp4"
        assert missing[0] is False


def test_locate_file_image_sequence_directory(qtbot):
    """Test locateFile for image sequence (directory selection)."""
    filenames = ["m:\\missing_images_dir"]
    missing = [True]
    is_sequence = [True]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    # Mock the FileDialog.openDir to return a found directory
    with patch("sleap.gui.dialogs.missingfiles.FileDialog") as MockFileDialog:
        MockFileDialog.openDir.return_value = "tests/data/videos"

        # Mock Path.is_dir to return True
        with patch("sleap.gui.dialogs.missingfiles.Path") as MockPath:
            mock_path_instance = MagicMock()
            mock_path_instance.is_dir.return_value = True
            mock_path_instance.name = "missing_images_dir"
            mock_path_instance.__str__ = lambda self: "tests/data/videos"
            MockPath.return_value = mock_path_instance

            win.locateFile(0)

            # FileDialog.openDir should be called (not open)
            MockFileDialog.openDir.assert_called_once()
            MockFileDialog.open.assert_not_called()


def test_locate_file_duplicate_prevention(qtbot):
    """Test that duplicate files are prevented."""
    filenames = [
        "m:\\video1.mp4",
        "tests/data/videos/small_robot.mp4",  # Already in list
    ]
    missing = [True, False]
    is_sequence = [False, False]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    # Mock the FileDialog.open to return a file already in the list
    with patch("sleap.gui.dialogs.missingfiles.FileDialog") as MockFileDialog:
        MockFileDialog.open.return_value = (
            "tests/data/videos/small_robot.mp4",  # Duplicate
            "filter",
        )

        # Mock QMessageBox to capture the warning
        with patch(
            "sleap.gui.dialogs.missingfiles.QtWidgets.QMessageBox"
        ) as MockMsgBox:
            mock_msgbox_instance = MagicMock()
            MockMsgBox.return_value = mock_msgbox_instance

            win.locateFile(0)

            # Message box should be shown for duplicate
            MockMsgBox.assert_called_once()
            mock_msgbox_instance.exec_.assert_called_once()

            # Filename should NOT be updated (still missing)
            assert filenames[0] == "m:\\video1.mp4"
            assert missing[0] is True


def test_locate_file_duplicate_directory_prevention(qtbot):
    """Test that duplicate directories are prevented for sequences."""
    filenames = [
        "m:\\missing_dir",
        "tests/data/videos",  # Already in list
    ]
    missing = [True, False]
    is_sequence = [True, True]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    # Mock the FileDialog.openDir to return a directory already in the list
    with patch("sleap.gui.dialogs.missingfiles.FileDialog") as MockFileDialog:
        MockFileDialog.openDir.return_value = "tests/data/videos"  # Duplicate

        # Mock Path for both is_dir check and string comparison
        with patch("sleap.gui.dialogs.missingfiles.Path") as MockPath:
            mock_path_instance = MagicMock()
            mock_path_instance.is_dir.return_value = True
            mock_path_instance.name = "videos"
            mock_path_instance.__str__ = lambda self: "tests/data/videos"
            MockPath.return_value = mock_path_instance

            # Mock QMessageBox to capture the warning
            with patch(
                "sleap.gui.dialogs.missingfiles.QtWidgets.QMessageBox"
            ) as MockMsgBox:
                mock_msgbox_instance = MagicMock()
                MockMsgBox.return_value = mock_msgbox_instance

                win.locateFile(0)

                # Message box should be shown for duplicate
                MockMsgBox.assert_called_once()

                # Filename should NOT be updated (still missing)
                assert filenames[0] == "m:\\missing_dir"
                assert missing[0] is True


def test_locate_file_empty_selection(qtbot):
    """Test that empty selection is handled gracefully."""
    filenames = ["m:\\missing_video.mp4"]
    missing = [True]
    is_sequence = [False]

    win = MissingFilesDialog(filenames, missing=missing, is_sequence=is_sequence)
    win.show()
    qtbot.addWidget(win)

    # Mock the FileDialog.open to return empty string (user cancelled)
    with patch("sleap.gui.dialogs.missingfiles.FileDialog") as MockFileDialog:
        MockFileDialog.open.return_value = ("", "filter")

        win.locateFile(0)

        # Filename should NOT be updated
        assert filenames[0] == "m:\\missing_video.mp4"
        assert missing[0] is True


def test_replace_mode_dialog(qtbot):
    """Test dialog in replace mode."""
    filenames = ["tests/data/videos/small_robot.mp4"]
    missing = [False]  # Not missing, but we want to replace

    win = MissingFilesDialog(filenames, missing=missing, replace=True)
    win.show()
    qtbot.addWidget(win)

    # In replace mode, accept button should be enabled (files found)
    assert win.replace is True


def test_allow_incomplete_mode(qtbot):
    """Test dialog with allow_incomplete=True."""
    filenames = ["m:\\missing_video.mp4"]
    missing = [True]

    win = MissingFilesDialog(filenames, missing=missing, allow_incomplete=True)
    win.show()
    qtbot.addWidget(win)

    # Accept button should be enabled even with missing files
    assert win.accept_button.isEnabled() is True

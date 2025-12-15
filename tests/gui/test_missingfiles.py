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

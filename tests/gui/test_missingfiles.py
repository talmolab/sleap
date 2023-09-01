from sleap.gui.dialogs.missingfiles import MissingFilesDialog


def test_missing_gui(qtbot):

    filenames = ["m:\\centered_pair_small.mp4", "m:\\small_robot.mp4"]
    win = MissingFilesDialog(filenames)
    win.show()
    qtbot.addWidget(win)

    assert win.file_table.model().rowCount() == 2
    assert win.accept_button.isEnabled() == False

    win.setFilename(0, "tests/data/videos/centered_pair_small.mp4", False)
    assert filenames[0] == "tests/data/videos/centered_pair_small.mp4"
    assert filenames[1] == "tests/data/videos/small_robot.mp4"

    assert win.accept_button.isEnabled() == True

import pytest

from sleap.gui.widgets.imagedir import QtImageDirectoryWidget


@pytest.mark.exclude_from_linux_pip_test  # Fails with core dump on linux
def test_imagedir_widget(qtbot):
    window = QtImageDirectoryWidget(
        directory="tests/data/videos/",
        filters=[("JPEG", "*.jpg"), ("Robot 1", "*1.jpg")],
    )

    window.show()
    qtbot.addWidget(window)

    assert window.windowTitle() == "robot2.jpg"

    window.setFilter(1)
    assert window.windowTitle() == "robot1.jpg"

    window.setFilter(0)
    assert window.windowTitle() == "robot2.jpg"

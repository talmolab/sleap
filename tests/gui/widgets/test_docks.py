"""Module for testing dock widgets for the `MainWindow`."""

import pytest

from sleap.gui.app import MainWindow
from sleap.gui.widgets.docks import (
    InstancesDock,
    SuggestionsDock,
    VideosDock,
    SkeletonDock,
)


def test_videos_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = VideosDock(main_window)

    assert dock.name == "Videos"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


def test_skeleton_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SkeletonDock(main_window)

    assert dock.name == "Skeleton"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


def test_suggestions_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SuggestionsDock(main_window)

    assert dock.name == "Labeling Suggestions"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


def test_instances_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = InstancesDock(main_window)

    assert dock.name == "Instances"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


if __name__ == "__main__":
    pytest.main([f"{__file__}::test_instances_dock"])

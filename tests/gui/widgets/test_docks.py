"""Module for testing dock widgets for the `MainWindow`."""

from pathlib import Path

import numpy as np
import pytest

from sleap import Labels, Video
from sleap.gui.app import MainWindow
from sleap.gui.commands import AddInstance, OpenSkeleton
from sleap.gui.widgets.docks import (
    InstancesDock,
    SuggestionsDock,
    VideosDock,
    SkeletonDock,
)


def test_videos_dock(
    qtbot,
    centered_pair_predictions: Labels,
    small_robot_mp4_vid: Video,
    centered_pair_vid: Video,
    small_robot_3_frame_vid: Video,
):
    """Test the `DockWidget` class."""

    # Add some extra videos to the labels
    labels = centered_pair_predictions
    labels.add_video(small_robot_3_frame_vid)
    labels.add_video(centered_pair_vid)
    labels.add_video(small_robot_mp4_vid)
    assert len(labels.videos) == 4

    # Create the dock
    main_window = MainWindow()

    # Use commands to set the labels instead of setting it directly
    # To make sure other dependent instances like color_manager are also set
    main_window.commands.loadLabelsObject(labels)

    video_state = labels.videos[-1]
    main_window.state["video"] = video_state
    dock = VideosDock(main_window)

    # Test that the dock was created correctly
    assert dock.name == "Videos"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()

    # Test that videos can be removed

    # No videos selected, won't remove anything
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 4

    # Select the last video, should remove that one and update state

    dock.main_window.videos_dock.table.selectRowItem(small_robot_mp4_vid)
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 3
    assert video_state not in labels.videos
    assert main_window.state["video"] == labels.videos[-1]

    # Select the last two videos, should remove those two and update state
    idxs = [1, 2]
    videos_to_be_removed = [labels.videos[i] for i in idxs]
    main_window.state["selected_batch_video"] = idxs
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 1
    assert (
        videos_to_be_removed[0] not in labels.videos
        and videos_to_be_removed[1] not in labels.videos
    )
    assert main_window.state["video"] == labels.videos[-1]


def test_skeleton_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SkeletonDock(main_window)

    assert dock.name == "Skeleton"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()

    # This method should get called when we click the load button, but let's just call
    # the non-gui parts directly
    fn = Path(
        OpenSkeleton.get_template_skeleton_filename(context=dock.main_window.commands)
    )
    assert fn.name == f"{dock.skeleton_templates.currentText()}.json"


def test_suggestions_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SuggestionsDock(main_window)

    assert dock.name == "Labeling Suggestions"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


def test_instances_dock(qtbot, centered_pair_predictions: Labels):
    """Test the `DockWidget` class."""
    main_window = MainWindow(labels=centered_pair_predictions)
    labels = main_window.labels
    context = main_window.commands
    lf = context.state["labeled_frame"]
    dock = InstancesDock(main_window)

    assert dock.name == "Instances"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()

    # Test new instance button

    offset = 10

    # Find instance that we will copy from
    (
        copy_instance,
        from_predicted,
        from_prev_frame,
    ) = AddInstance.find_instance_to_copy_from(
        context, copy_instance=None, init_method="best"
    )
    n_instance = len(lf.instances)
    dock.main_window._buttons["new instance"].click()

    # Check that new instance was added with offset
    assert len(lf.instances) == n_instance + 1
    new_inst = lf.instances[-1]
    diff = np.nan_to_num(new_inst.numpy() - copy_instance.numpy(), nan=offset)
    assert np.all(diff == offset)

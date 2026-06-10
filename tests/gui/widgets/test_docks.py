"""Module for testing dock widgets for the `MainWindow`."""

from pathlib import Path

import numpy as np

from sleap import Labels, Video
from sleap.gui.app import MainWindow
from sleap.gui.commands import AddInstance, OpenSkeleton
from sleap.gui.state import INSTANCE_HIDDEN_KEY, VIEW_ONLY_INSTANCE_KEY
from sleap.gui.widgets.docks import (
    InstancesDock,
    SkeletonDock,
    SuggestionsDock,
    VideosDock,
)
from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video


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
    labels_add_video(labels, small_robot_3_frame_vid)
    labels_add_video(labels, centered_pair_vid)
    labels_add_video(labels, small_robot_mp4_vid)
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

    # Test that the video edit buttons are wired up
    assert "add videos" in dock.main_window._buttons
    assert "replace videos" in dock.main_window._buttons
    assert "remove video" in dock.main_window._buttons

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


def _qt_instance_for(player, instance):
    """Return the `QtInstance` on the canvas for the given `Instance`."""
    for qt_inst in player.view.all_instances:
        if qt_inst.instance is instance:
            return qt_inst
    return None


def test_instances_dock_visibility_columns(qtbot, centered_pair_predictions: Labels):
    """The Instances dock exposes visibility/view-only checkbox columns that
    toggle per-instance rendering on the canvas."""
    from qtpy import QtCore

    main_window = MainWindow(labels=centered_pair_predictions)

    # Navigate to a frame with multiple instances.
    target = centered_pair_predictions.labeled_frames[13]
    main_window.state["frame_idx"] = target.frame_idx

    model = main_window.instances_dock.table.model()
    assert "visibility" in model.properties
    assert "view only" in model.properties

    assert model.rowCount() >= 2
    inst0 = model.original_items[0]
    inst1 = model.original_items[1]

    qt0 = _qt_instance_for(main_window.player, inst0)
    qt1 = _qt_instance_for(main_window.player, inst1)
    assert qt0 is not None and qt1 is not None

    # Default: both visible.
    assert qt0.isVisible()
    assert qt1.isVisible()

    vis_col = model.properties.index("visibility")
    view_col = model.properties.index("view only")

    # Uncheck visibility on row 0 -> that instance hides, the other stays.
    model.setData(
        model.index(0, vis_col), QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole
    )
    assert not qt0.isVisible()
    assert qt1.isVisible()

    # Re-check -> reappears.
    model.setData(model.index(0, vis_col), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
    assert qt0.isVisible()

    # View-only on row 1 -> only instance 1 visible.
    model.setData(model.index(1, view_col), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
    assert not qt0.isVisible()
    assert qt1.isVisible()

    # Clicking a visibility box exits view-only -> both visible again.
    model.setData(model.index(0, vis_col), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
    assert qt0.isVisible()
    assert qt1.isVisible()

    # Visibility/view-only reset on frame change.
    other = next(
        lf
        for lf in centered_pair_predictions.labeled_frames
        if lf.frame_idx != target.frame_idx and len(lf.instances) >= 1
    )
    model.setData(model.index(0, view_col), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
    main_window.state["frame_idx"] = other.frame_idx
    assert main_window.state[VIEW_ONLY_INSTANCE_KEY] is None
    assert main_window.state[INSTANCE_HIDDEN_KEY] == set()


def test_instances_dock_visibility_replot_and_global_toggle(
    qtbot, centered_pair_predictions: Labels
):
    """Per-instance visibility survives a same-frame replot and is overridden by
    the global "show instances" toggle (regressions for #2755)."""
    from qtpy import QtCore

    main_window = MainWindow(labels=centered_pair_predictions)
    target = centered_pair_predictions.labeled_frames[13]
    main_window.state["frame_idx"] = target.frame_idx

    model = main_window.instances_dock.table.model()
    assert model.rowCount() >= 2
    inst0 = model.original_items[0]
    inst1 = model.original_items[1]
    vis_col = model.properties.index("visibility")

    # Hide instance 0 via its visibility box.
    model.setData(
        model.index(0, vis_col), QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole
    )
    assert id(inst0) in main_window.state[INSTANCE_HIDDEN_KEY]

    # A same-frame replot (what a marker-size/add-instance change triggers) must
    # NOT reset the hide -- only a real frame change does.
    main_window.plotFrame()
    assert id(inst0) in main_window.state[INSTANCE_HIDDEN_KEY]
    qt0 = _qt_instance_for(main_window.player, inst0)
    qt1 = _qt_instance_for(main_window.player, inst1)
    assert qt0 is not None and not qt0.isVisible()
    assert qt1 is not None and qt1.isVisible()

    # Global "show instances" off hides everything, even the still-visible inst1.
    main_window.state["show instances"] = False
    qt0 = _qt_instance_for(main_window.player, inst0)
    qt1 = _qt_instance_for(main_window.player, inst1)
    assert qt0 is not None and not qt0.isVisible()
    assert qt1 is not None and not qt1.isVisible()

    # Turning it back on restores per-instance state (inst0 hidden, inst1 shown).
    main_window.state["show instances"] = True
    qt0 = _qt_instance_for(main_window.player, inst0)
    qt1 = _qt_instance_for(main_window.player, inst1)
    assert qt0 is not None and not qt0.isVisible()
    assert qt1 is not None and qt1.isVisible()

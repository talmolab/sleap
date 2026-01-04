from sleap.gui.widgets.video import (
    GraphicsView,
    QtVideoPlayer,
    QtTextWithBackground,
    VisibleBoundingBox,
    QtInstance,
)

from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QColor, QWheelEvent
import numpy as np


def test_gui_video(qtbot):
    vp = QtVideoPlayer()
    vp.show()
    qtbot.addWidget(vp)

    assert vp.close()


def test_gui_video_instances(qtbot, small_robot_mp4_vid, centered_pair_labels):
    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)

    test_frame_idx = 63
    labeled_frames = centered_pair_labels.labeled_frames

    def plot_instances(vp, idx):
        for instance in labeled_frames[test_frame_idx].instances:
            vp.addInstance(instance=instance)

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()

    vp.show()
    vp.plot()

    # Check that all instances are included in viewer
    assert len(vp.instances) == len(labeled_frames[test_frame_idx].instances)

    # All instances should be selectable
    assert vp.selectable_instances == vp.instances

    vp.zoomToFit()

    # Check that we zoomed correctly
    assert vp.view.zoomFactor > 1

    vp.instances[0].updatePoints(complete=True)

    # Check that node is marked as complete
    nodes = [item for item in vp.instances[0].childItems() if hasattr(item, "point")]
    assert all((node.point[3] for node in nodes))  # point[3] = complete

    # Check that selection via keyboard works
    assert vp.view.getSelectionIndex() is None
    qtbot.keyClick(vp, QtCore.Qt.Key_1)
    assert vp.view.getSelectionIndex() == 0
    qtbot.keyClick(vp, QtCore.Qt.Key_2)
    assert vp.view.getSelectionIndex() == 1

    # Check that updatedSelection signal is emitted
    with qtbot.waitSignal(vp.view.updatedSelection, timeout=10):
        qtbot.keyClick(vp, QtCore.Qt.Key_1)

    # Check that selection by Instance works
    for inst in labeled_frames[test_frame_idx].instances:
        vp.view.selectInstance(inst)
        assert vp.view.getSelectionInstance() == inst

    # Check that sequence selection works
    with qtbot.waitCallback() as cb:
        vp.view.selectInstance(None)
        vp.onSequenceSelect(2, cb)
        qtbot.keyClick(vp, QtCore.Qt.Key_2)
        qtbot.keyClick(vp, QtCore.Qt.Key_1)

    inst_1 = vp.selectable_instances[1].instance
    inst_0 = vp.selectable_instances[0].instance
    assert cb.args[0] == [inst_1, inst_0]

    assert vp.close()


def test_getInstancesBoundingRect():
    rect = GraphicsView.getInstancesBoundingRect([])
    assert rect.isNull()


def test_QtTextWithBackground(qtbot):
    scene = QtWidgets.QGraphicsScene()
    view = QtWidgets.QGraphicsView()
    view.setScene(scene)

    txt = QtTextWithBackground()

    txt.setDefaultTextColor(QColor("yellow"))
    bg_color = txt.getBackgroundColor()
    assert bg_color.lightness() == 0

    txt.setDefaultTextColor(QColor("black"))
    bg_color = txt.getBackgroundColor()
    assert bg_color.lightness() == 255

    scene.addItem(txt)
    qtbot.addWidget(view)


def test_VisibleBoundingBox(qtbot, centered_pair_labels):
    vp = QtVideoPlayer(centered_pair_labels.video)

    test_idx = 27
    for instance in centered_pair_labels.labeled_frames[test_idx].instances:
        vp.addInstance(instance)

    inst = vp.instances[0]

    # Check if type of bounding box is correct
    assert type(inst.box) == VisibleBoundingBox

    # Scale the bounding box
    start_top_left = inst.box.rect().topLeft()
    start_bottom_right = inst.box.rect().bottomRight()
    initial_width = inst.box.rect().width()
    initial_height = inst.box.rect().height()

    dx = 5
    dy = 10

    end_top_left = QtCore.QPointF(start_top_left.x() - dx, start_top_left.y() - dy)
    end_bottom_right = QtCore.QPointF(
        start_bottom_right.x() + dx, start_bottom_right.y() + dy
    )

    inst.box.setRect(QtCore.QRectF(end_top_left, end_bottom_right))

    # Check if bounding box scaled appropriately
    assert inst.box.rect().width() - initial_width == 2 * dx
    assert inst.box.rect().height() - initial_height == 2 * dy


def test_wheelEvent(qtbot):
    """Test the wheelEvent method of the GraphicsView class."""
    graphics_view = GraphicsView()

    # Create a QWheelEvent
    position = QtCore.QPointF(100, 100)  # The position of the wheel event
    global_position = QtCore.QPointF(100, 100)  # The global position of the wheel event
    pixel_delta = QtCore.QPoint(0, 120)  # The distance in pixels the wheel is rotated
    angle_delta = QtCore.QPoint(0, 120)  # The distance in degrees the wheel is rotated
    buttons = QtCore.Qt.MouseButton.NoButton  # The mouse buttons
    modifiers = QtCore.Qt.KeyboardModifier.NoModifier  # The keyboard modifiers
    phase = QtCore.Qt.ScrollPhase.ScrollUpdate  # The scroll phase
    inverted = False  # The inverted flag
    source = QtCore.Qt.MouseEventSource.MouseEventNotSynthesized  # The source

    event = QWheelEvent(
        position,
        global_position,
        pixel_delta,
        angle_delta,
        buttons,
        modifiers,
        phase,
        inverted,
        source,
    )

    # Call the wheelEvent method
    print(
        "Testing GraphicsView.wheelEvent which will result in exit code 127 "
        "originating from a segmentation fault if it fails."
    )
    graphics_view.wheelEvent(event)


def test_nan_coordinates_bounding_rect(qtbot, centered_pair_labels):
    """Test that NaN coordinates don't create NaN bounding rects.

    Regression test for issue #2427 where NaN coordinates in predicted instances
    caused the GUI to freeze on Linux systems with Qt 6.10+.
    """
    from sleap_io.model.instance import PredictedInstance

    vp = QtVideoPlayer(centered_pair_labels.video)

    # Get a labeled frame with instances
    test_frame = centered_pair_labels.labeled_frames[0]
    original_instance = test_frame.instances[0]

    # Test 1: Instance with some NaN coordinates (failed keypoint detection)
    points_with_nan = original_instance.numpy().copy()
    points_with_nan[0] = [np.nan, np.nan]  # First keypoint has NaN
    points_with_nan[1] = [np.nan, np.nan]  # Second keypoint has NaN

    predicted_instance = PredictedInstance(
        points=points_with_nan, skeleton=original_instance.skeleton, score=0.5
    )

    # Create QtInstance directly to test bounding rect calculation
    qt_instance = QtInstance(instance=predicted_instance, player=vp)

    # Verify bounding rect doesn't have NaN values (issue #2427)
    bounding_rect = qt_instance.getPointsBoundingRect()
    # Should return either a valid rect or null rect, but never NaN rect
    assert not np.isnan(bounding_rect.x()), "Bounding rect x is NaN"
    assert not np.isnan(bounding_rect.y()), "Bounding rect y is NaN"
    assert not np.isnan(bounding_rect.width()), "Bounding rect width is NaN"
    assert not np.isnan(bounding_rect.height()), "Bounding rect height is NaN"

    # Test 2: Instance with all NaN coordinates (complete detection failure)
    n_nodes = len(original_instance.skeleton.nodes)
    all_nan_points = np.full((n_nodes, 2), np.nan)
    all_nan_instance = PredictedInstance(
        points=all_nan_points, skeleton=original_instance.skeleton, score=0.1
    )

    qt_instance_all_nan = QtInstance(instance=all_nan_instance, player=vp)
    bounding_rect_all_nan = qt_instance_all_nan.getPointsBoundingRect()

    # Should return a null rect (which Qt handles gracefully)
    assert bounding_rect_all_nan.isNull() or bounding_rect_all_nan.isEmpty(), (
        "All-NaN instance should have null bounding rect"
    )

    # Verify no NaN values in the rect
    assert not np.isnan(bounding_rect_all_nan.x()), "All-NaN rect x is NaN"
    assert not np.isnan(bounding_rect_all_nan.y()), "All-NaN rect y is NaN"


def test_navigate_highlight(qtbot, small_robot_mp4_vid, centered_pair_labels):
    """Test the navigation highlight feature for Size Distribution click-to-navigate."""
    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)

    test_frame_idx = 63
    labeled_frames = centered_pair_labels.labeled_frames
    frame_instances = labeled_frames[test_frame_idx].instances

    def plot_instances(vp, idx):
        for instance in frame_instances:
            vp.addInstance(instance=instance)

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()
    vp.show()
    vp.plot()

    # Ensure we have instances
    assert len(vp.instances) >= 2

    # Initially, no instances should have navigation highlight
    for inst in vp.instances:
        assert not inst.navigate_highlight

    # Highlight the second instance via player method (using Instance object)
    vp.highlightNavigatedInstance(frame_instances[1])

    # Check that only the second instance is highlighted
    assert not vp.instances[0].navigate_highlight
    assert vp.instances[1].navigate_highlight
    assert vp.instances[1].navigate_box.opacity() == 0.5

    # Highlight a different instance (first)
    vp.highlightNavigatedInstance(frame_instances[0])

    # Check that now only the first instance is highlighted
    assert vp.instances[0].navigate_highlight
    assert not vp.instances[1].navigate_highlight

    # Clear all highlights via player method (pass None)
    vp.highlightNavigatedInstance(None)

    # All instances should now have no highlight
    for inst in vp.instances:
        assert not inst.navigate_highlight
        assert inst.navigate_box.opacity() == 0

    # Test via GraphicsView directly
    vp.view.highlightNavigatedInstance(frame_instances[0])
    assert vp.instances[0].navigate_highlight

    vp.view.clearNavigateHighlight()
    assert not vp.instances[0].navigate_highlight

    assert vp.close()

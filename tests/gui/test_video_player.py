import numpy as np
from sleap import Instance, Skeleton
from sleap.gui.widgets.video import (
    QtVideoPlayer,
    GraphicsView,
    QtInstance,
    QtVideoPlayer,
)

import PySide2.QtCore as QtCore


def test_gui_video(qtbot):
    vp = QtVideoPlayer()
    vp.show()
    qtbot.addWidget(vp)

    assert vp.close()

    # Click the button 20 times
    # for i in range(20):
    #     qtbot.mouseClick(vp.btn, QtCore.Qt.LeftButton)


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
    assert all((node.point.complete for node in nodes))

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

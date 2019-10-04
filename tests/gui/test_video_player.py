from sleap.gui.video import QtVideoPlayer

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
            vp.addInstance(instance=instance, color=(0, 0, 128))

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()

    vp.show()
    vp.plot()

    # Check that all instances are included in viewer
    assert len(vp.instances) == len(labeled_frames[test_frame_idx].instances)

    vp.zoomToFit()

    # Check that we zoomed correctly
    assert vp.view.zoomFactor > 1

    vp.instances[0].updatePoints(complete=True)

    # Check that node is marked as complete
    assert vp.instances[0].childItems()[3].point.complete

    # Check that selection via keyboard works
    assert vp.view.getSelection() == None
    qtbot.keyClick(vp, QtCore.Qt.Key_1)
    assert vp.view.getSelection() == 0
    qtbot.keyClick(vp, QtCore.Qt.Key_QuoteLeft)
    assert vp.view.getSelection() == 1

    # Check that selection by Instance works
    for inst in labeled_frames[test_frame_idx].instances:
        vp.view.selectInstance(inst)
        assert vp.view.getSelectionInstance() == inst

    # Check that sequence selection works
    with qtbot.waitCallback() as cb:
        vp.view.clearSelection()
        vp.onSequenceSelect(2, cb)
        qtbot.keyClick(vp, QtCore.Qt.Key_2)
        qtbot.keyClick(vp, QtCore.Qt.Key_1)
    assert cb.args[0] == [1, 0]

    assert vp.close()

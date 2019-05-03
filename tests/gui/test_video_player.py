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

    test_frame_idx = 0
    labeled_frames = [_ for _ in centered_pair_labels if _.frame_idx == test_frame_idx]

    def plot_instances(vp, idx):
        for instance in labeled_frames[idx].instances:        
            vp.addInstance(instance=instance, color=(0,0,128))

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()

    vp.show()
    vp.plot()

    # Check that all instances are included in viewer
    assert len(vp.instances) == len(labeled_frames[0].instances)

    vp.zoomToFit()

    # Check that we zoomed correctly
    assert(vp.view.zoomFactor > 2)
    
    vp.instances[0].updatePoints(complete=True)
    
    # Check that node is marked as complete
    assert vp.instances[0].childItems()[1].point.complete

    assert vp.close()
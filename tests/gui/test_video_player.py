from sleap.gui.video import VideoPlayer, QtVideoPlayer

import PySide2.QtCore as QtCore

def test_gui_video(qtbot):
    vp = QtVideoPlayer()
    vp.show()
    qtbot.addWidget(vp)

    # Click the button 20 times
    # for i in range(20):
    #     qtbot.mouseClick(vp.btn, QtCore.Qt.LeftButton)

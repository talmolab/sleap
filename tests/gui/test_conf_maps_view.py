from sleap.gui.video import VideoPlayer, QtVideoPlayer
from sleap.gui.confmaps import QtConfMaps

from sleap.io.video import Video

TEST_H5_FILE = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
TEST_H5_DSET = "/confmaps"
TEST_H5_INPUT_FORMAT = "channels_first"




import PySide2.QtCore as QtCore

def test_gui_video(qtbot):
    
    hdf5_conf = Video.from_hdf5(file=TEST_H5_FILE, dataset=TEST_H5_DSET, input_format=TEST_H5_INPUT_FORMAT)
    
    vp = QtVideoPlayer()
    vp.show()
    conf_maps = QtConfMaps(hdf5_conf.get_frame(1))
    vp.view.scene.addItem(conf_maps)

    # Click the button 20 times
    # for i in range(20):
    #     qtbot.mouseClick(vp.btn, QtCore.Qt.LeftButton)

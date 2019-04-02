from sleap.gui.video import VideoPlayer, QtVideoPlayer
from sleap.gui.confmapsplot import ConfMapsPlot

from sleap.io.video import Video

TEST_H5_FILE = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
TEST_H5_DSET = "/confmaps"
TEST_H5_INPUT_FORMAT = "channels_first"




import PySide2.QtCore as QtCore

def test_gui_conf_maps(qtbot):
    
    hdf5_conf = Video.from_hdf5(file=TEST_H5_FILE, dataset=TEST_H5_DSET, input_format=TEST_H5_INPUT_FORMAT)
    
    vp = QtVideoPlayer()
    vp.show()
    conf_maps = ConfMapsPlot(hdf5_conf.get_frame(1))
    vp.view.scene.addItem(conf_maps)
    
    # make sure we're showing all the channels
    assert len(conf_maps.childItems()) == 6
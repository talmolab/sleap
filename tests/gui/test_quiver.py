from sleap.gui.video import VideoPlayer, QtVideoPlayer
from sleap.gui.quiverplot import MultiQuiverPlot

from sleap.io.video import Video

TEST_H5_FILE = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
TEST_H5_DSET = "/pafs"
TEST_H5_INPUT_FORMAT = "channels_first"


import PySide2.QtCore as QtCore

def test_gui_quiver(qtbot):
    
    hdf5_conf = Video.from_hdf5(file=TEST_H5_FILE, dataset=TEST_H5_DSET, input_format=TEST_H5_INPUT_FORMAT)
    
    vp = QtVideoPlayer()
    vp.show()
    affinity_fields = MultiQuiverPlot(
                    frame=hdf5_conf.get_frame(0),
                    show=[0,1],
                    decimation=1
                    )
    vp.view.scene.addItem(affinity_fields)
    
    # make sure we're showing all the channels we selected
    assert len(affinity_fields.childItems()) == 2
    # make sure we're showing all arrows in first channel
    assert len(affinity_fields.childItems()[0].childItems()) == 1148

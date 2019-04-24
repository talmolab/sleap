from sleap.gui.video import QtVideoPlayer
from sleap.gui.confmapsplot import ConfMapsPlot

from sleap.io.video import Video

import PySide2.QtCore as QtCore

def test_gui_conf_maps(qtbot, hdf5_confmaps):
    
    vp = QtVideoPlayer()
    vp.show()
    conf_maps = ConfMapsPlot(hdf5_confmaps.get_frame(1))
    vp.view.scene.addItem(conf_maps)
    
    # make sure we're showing all the channels
    assert len(conf_maps.childItems()) == 6
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.gui.overlays.confmaps import ConfMapsPlot


def test_gui_conf_maps(qtbot, hdf5_confmaps):

    vp = QtVideoPlayer()
    vp.show()
    conf_maps = ConfMapsPlot(hdf5_confmaps.get_frame(1), show_box=False)
    vp.view.scene.addItem(conf_maps)

    # make sure we're showing all the channels
    assert len(conf_maps.childItems()) == 6

    assert vp.close()

import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("li"), reason="exclude_from_linux_pip_test"
)  # Fails with core dump on linux
def test_gui_conf_maps(qtbot, hdf5_confmaps):

    from sleap.gui.widgets.video import QtVideoPlayer
    from sleap.gui.overlays.confmaps import ConfMapsPlot

    vp = QtVideoPlayer()
    vp.show()
    conf_maps = ConfMapsPlot(hdf5_confmaps.get_frame(1), show_box=False)
    vp.view.scene.addItem(conf_maps)

    # make sure we're showing all the channels
    assert len(conf_maps.childItems()) == 6

    assert vp.close()

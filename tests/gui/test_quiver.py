from sleap.gui.widgets.video import QtVideoPlayer
from sleap.gui.overlays.pafs import MultiQuiverPlot


def test_gui_quiver(qtbot, hdf5_affinity):

    vp = QtVideoPlayer()
    vp.show()
    affinity_fields = MultiQuiverPlot(
        frame=hdf5_affinity.get_frame(0)[265:275, 238:248], show=[0, 1], decimation=1
    )
    vp.view.scene.addItem(affinity_fields)

    # make sure we're showing all the channels we selected
    assert len(affinity_fields.childItems()) == 2
    # make sure we're showing all arrows in first channel
    assert len(affinity_fields.childItems()[0].points) == 480

    assert vp.close()

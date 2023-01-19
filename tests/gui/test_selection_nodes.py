from sleap.gui.widgets.video import *
import pytestqt
from qtpy import QtCore, QtWidgets, QtVideoPlayer
from qtpy.QtGui import QColor

def test_AddNodes(qtbot, small_robot_mp4_vid, centered_pair_labels):
    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)

    test_frame_idx = 63
    labeled_frames = centered_pair_labels.labeled_frames

    def plot_instances(vp, idx):
        for instance in labeled_frames[test_frame_idx].instances:
            vp.addInstance(instance=instance)

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()

    vp.show()
    vp.plot()

    # Check that the bounding box type is correct
    inst = vp.instances[1]  # QtInstance
    nodes = inst.get_all_nodes()


    # Gets node's coordinates
    node_1_point = QtCore.QPoint(nodes[0].scenePos().x(), nodes[0].scenePos().y())
    node_2_point = QtCore.QPoint(nodes[1].scenePos().x(), nodes[1].scenePos().y())

    # Selects first Node
    qtbot.mouseMove(vp, node_1_point)
    qtbot.keyPress(vp, QtCore.Qt.ShiftModifier)
    qtbot.mousePress(vp, QtCore.Qt.LeftButton)


    # Check if first node is added
    selected_list = vp.view.selected_nodes

    assert len(selected_list) == 1
    assert selected_list[0] == nodes[0]

    # Select Second Node
    qtbot.mouseRelease(vp, QtCore.Qt.LeftButton)
    qtbot.mouseMove(vp, node_2_point)
    qtbot.mousePress(vp, QtCore.Qt.LeftButton)

    assert len(selected_list) == 2
    assert selected_list[0] == nodes[0]
    assert selected_list[1] == nodes[1]



if __name__ == "__main__":
    import pytest

    pytest.main([r"tests\gui\test_selction_nodes.py::test_AddNodes"])

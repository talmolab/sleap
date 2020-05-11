import numpy as np
from sleap.gui.app import MainWindow
from sleap.gui.commands import *


def test_grid_system_midpoint_gui(qtbot, midpoint_grid_labels):
    app = MainWindow()
    app.loadLabelsObject(midpoint_grid_labels)

    assert len(app.state["labeled_frame"]) == 1
    lf = app.state["labeled_frame"]

    assert len(lf) == 1
    inst = lf[0]

    assert inst.points[0].x == -0.5
    assert inst.points[0].y == -0.5
    assert inst.points[1].x == 0.0
    assert inst.points[1].y == 0.0
    assert inst.points[2].x == -0.5
    assert inst.points[2].y == 0.5

    # app.player  # QtVideoPlayer
    # app.player.view  # GraphicsView(QGraphicsView)
    # app.player.view.instances  # List["QtInstance"]
    qt_inst = app.player.view.instances[0]
    # qt_inst.nodes[node_name] # QtNode

    qt_node = qt_inst.nodes["(-0.5, -0.5)"]  # QtNode
    assert qt_node.scenePos().x() == -0.5
    assert qt_node.scenePos().y() == -0.5

    qt_node = qt_inst.nodes["(0, 0)"]  # QtNode
    assert qt_node.scenePos().x() == 0
    assert qt_node.scenePos().y() == 0

    qt_node = qt_inst.nodes["(-0.5, 0.5)"]  # QtNode
    assert qt_node.scenePos().x() == -0.5
    assert qt_node.scenePos().y() == 0.5


def test_grid_system_legacy_gui(qtbot, legacy_grid_labels):
    app = MainWindow()
    app.loadLabelsObject(legacy_grid_labels)

    assert len(app.state["labeled_frame"]) == 1
    lf = app.state["labeled_frame"]

    assert len(lf) == 1
    inst = lf[0]

    assert inst.points[0].x == -1
    assert inst.points[0].y == -1
    assert inst.points[1].x == -0.5
    assert inst.points[1].y == -0.5
    assert inst.points[2].x == -1
    assert inst.points[2].y == 0

    # app.player  # QtVideoPlayer
    # app.player.view  # GraphicsView(QGraphicsView)
    # app.player.view.instances  # List["QtInstance"]
    qt_inst = app.player.view.instances[0]

    node_names = ["(-0.5, -0.5)", "(0, 0)", "(-0.5, 0.5)"]
    # qt_inst.nodes[node_name] # QtNode
    qt_node = qt_inst.nodes["(-0.5, -0.5)"]  # QtNode
    assert qt_node.scenePos().x() == -1
    assert qt_node.scenePos().y() == -1

    qt_node = qt_inst.nodes["(0, 0)"]  # QtNode
    assert qt_node.scenePos().x() == -0.5
    assert qt_node.scenePos().y() == -0.5

    qt_node = qt_inst.nodes["(-0.5, 0.5)"]  # QtNode
    assert qt_node.scenePos().x() == -1
    assert qt_node.scenePos().y() == 0


def test_grid_system_midpoint_labels(midpoint_grid_labels):
    inst = midpoint_grid_labels[0][0]
    np.testing.assert_array_equal(
        inst.points_array, [[-0.5, -0.5], [0, 0], [-0.5, 0.5]]
    )


def test_grid_system_legacy_labels(legacy_grid_labels):
    inst = legacy_grid_labels[0][0]
    np.testing.assert_array_equal(inst.points_array, [[-1, -1], [-0.5, -0.5], [-1, 0]])

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
from PySide2.QtWidgets import QLabel, QPushButton
from PySide2.QtWidgets import QTableWidget, QTableWidgetItem
from PySide2.QtWidgets import QMenu, QAction
from PySide2.QtCore import Qt

import matplotlib.pyplot as plt
import numpy as np

from sleap.gui.video import QtVideoPlayer, QtInstance, QtEdge, QtNode
from sleap.io.generated import GeneratedLabels


class MainWindow(QMainWindow):
    def __init__(self, labels: GeneratedLabels, video=None, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.labels = labels
        self.video = video if video is not None else labels.video
        self.player = QtVideoPlayer(video=None, callbacks=[self.newFrame,])
        self.setCentralWidget(self.player)

        # lines(7)*255
        self.cmap = np.array([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ])


        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction("New...")
        fileMenu.addAction("Open project")
        fileMenu.addAction("Open video")

        viewMenu = self.menuBar().addMenu("View")

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Documentation")
        helpMenu.addAction("Keyboard reference")
        helpMenu.addAction("About")

        self.statusBar() # Initialize status bar

        def _make_dock(name, widgets=[]):
            dock = QDockWidget(name)
            dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            dock_widget = QWidget()
            layout = QVBoxLayout()
            for widget in widgets:
                layout.addWidget(widget)
            dock_widget.setLayout(layout)
            dock.setWidget(dock_widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            viewMenu.addAction(dock.toggleViewAction())
            return layout

        skeleton_layout = _make_dock("Skeleton")

        gb = QGroupBox("Nodes")
        vb = QVBoxLayout()
        self.skeletonNodesTable = QTableWidget()
        self.skeletonNodesTable.setColumnCount(2)
        self.skeletonNodesTable.setVerticalHeaderLabels(["id",])
        self.skeletonNodesTable.setHorizontalHeaderLabels(["name", "symmetry"])
        hb = QHBoxLayout()
        hb.addWidget(QPushButton("New node"))
        hb.addWidget(QPushButton("Delete node"))
        vb.addWidget(self.skeletonNodesTable)
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        gb = QGroupBox("Edges")
        vb = QVBoxLayout()
        self.skeletonEdgesTable = QTableWidget()
        self.skeletonEdgesTable.setColumnCount(2)
        self.skeletonEdgesTable.setHorizontalHeaderLabels(["source", "destination"])
        self.skeletonEdgesTable.setVerticalHeaderLabels(["id",])
        # self.skeletonEdgesTable.verticalHeader().hide()
        hb = QHBoxLayout()
        hb.addWidget(QPushButton("New edge"))
        hb.addWidget(QPushButton("Delete edge"))
        vb.addWidget(self.skeletonEdgesTable)
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        ######
        points_layout = _make_dock("Points")
        self.pointsTable = QTableWidget()
        self.pointsTable.setColumnCount(6)
        self.pointsTable.setHorizontalHeaderLabels(["frameIdx", "instanceId", "x", "y", "node", "visible"])
        self.pointsTable.setVerticalHeaderLabels(["id",])
        points_layout.addWidget(self.pointsTable)

        self.loadLabels()

    def loadLabels(self):
        # data_path = "" # get from file dialog
        # self.labels = GeneratedLabels(data_path)
        self.video = self.labels.video

        # Populate skeleton
        self.skeletonNodesTable.setRowCount(len(labels.skeleton.node_names))
        for i, node in enumerate(labels.skeleton.node_names):
            self.skeletonNodesTable.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
            self.skeletonNodesTable.setItem(i, 0, QTableWidgetItem(f"{node}"))
            self.skeletonNodesTable.setItem(i, 1, QTableWidgetItem(""))

        self.skeletonEdgesTable.setRowCount(len(labels.skeleton.graph.edges()))
        for i, (src, dst) in enumerate(labels.skeleton.graph.edges()):
            self.skeletonEdgesTable.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
            self.skeletonEdgesTable.setItem(i, 0, QTableWidgetItem(f"{src}"))
            self.skeletonEdgesTable.setItem(i, 1, QTableWidgetItem(f"{dst}"))

        # Populate points
        self.pointsTable.setRowCount(len(labels.points))
        for i, row in self.labels.points.iterrows():
            self.pointsTable.setVerticalHeaderItem(i, QTableWidgetItem(f"{int(row['id'])}"))
            for j, k in enumerate(["frameIdx", "instanceId", "x", "y", "node", "visible"]):
                self.pointsTable.setItem(i, j, QTableWidgetItem(f"{row[k]}"))

        # Show video
        self.player.load_video(self.labels.video)
        
    def newFrame(self, player, idx):
        frame_instances = self.labels.get_frame_instances(idx)

        for i, instance in enumerate(frame_instances):
            qt_instance = QtInstance(instance=instance, color=self.cmap[i])
            player.view.scene.addItem(qt_instance)

        self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.labels.video)} | Instances (current/total): {len(frame_instances)}/{self.labels.points.instanceId.nunique()}")


if __name__ == "__main__":

    data_path = "../../tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    labels = GeneratedLabels(data_path)

    app = QApplication([])
    app.setApplicationName("sLEAP Label")
    window = MainWindow(labels)
    window.showMaximized()
    app.exec_()


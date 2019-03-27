from PySide2 import QtCore
from PySide2.QtCore import Qt

from PySide2.QtGui import QKeyEvent

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout
from PySide2.QtWidgets import QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
from PySide2.QtWidgets import QTableWidget, QTableView, QTableWidgetItem
from PySide2.QtWidgets import QMenu, QAction
from PySide2.QtWidgets import QFileDialog, QMessageBox

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point
from sleap.io.video import Video, HDF5Video, MediaVideo
from sleap.io.dataset import Labels, LabeledFrame
from sleap.gui.video import QtVideoPlayer, QtInstance, QtEdge, QtNode
from sleap.gui.dataviews import VideosTable, SkeletonNodesTable, SkeletonEdgesTable, LabeledFrameTable


class MainWindow(QMainWindow):
    labels: Labels
    skeleton: Skeleton
    video: Video

    def __init__(self, data_path=None, video=None, import_data=None, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

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

        self.labels = Labels()
        self.skeleton = Skeleton()
        self.labeled_frame = None
        self.video = None

        self.initialize_gui()

        if data_path is not None:
            pass

        if import_data is not None:
            self.importData(import_data)

        # TODO: auto-add video to clean project if no data provided
        # TODO: auto-select video if data provided, or add it to project
        if video is not None:
            self.addVideo(video)

    def initialize_gui(self):

        ####### Menus #######
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction("New project").triggered.connect(self.newProject)
        fileMenu.addAction("Open project").triggered.connect(self.openProject)
        fileMenu.addAction("Save").triggered.connect(self.saveProject)
        fileMenu.addAction("Save as...").triggered.connect(self.saveProjectAs)
        fileMenu.addSeparator()
        fileMenu.addAction("Import...").triggered.connect(self.importData)
        fileMenu.addAction("Export...").triggered.connect(self.exportData)
        fileMenu.addSeparator()
        fileMenu.addAction("&Quit").triggered.connect(self.close)

        videoMenu = self.menuBar().addMenu("Video")
        # videoMenu.addAction("Check video encoding").triggered.connect(self.checkVideoEncoding)
        # videoMenu.addAction("Reencode for seeking").triggered.connect(self.reencodeForSeeking)
        # videoMenu.addSeparator()
        videoMenu.addAction("Add video").triggered.connect(self.addVideo)
        videoMenu.addAction("Add folder").triggered.connect(self.addVideoFolder)
        videoMenu.addAction("Next video").triggered.connect(self.nextVideo)
        videoMenu.addAction("Previous video").triggered.connect(self.previousVideo)
        videoMenu.addSeparator()
        videoMenu.addAction("Extract clip...").triggered.connect(self.extractClip)

        viewMenu = self.menuBar().addMenu("View")

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Documentation").triggered.connect(self.openDocumentation)
        helpMenu.addAction("Keyboard reference").triggered.connect(self.openKeyRef)
        helpMenu.addAction("About").triggered.connect(self.openAbout)

        ####### Video player #######
        self.player = QtVideoPlayer()
        self.player.callbacks.append(self.newFrame)
        self.setCentralWidget(self.player)

        ####### Status bar #######
        self.statusBar() # Initialize status bar

        ####### Helpers #######
        def _make_dock(name, widgets=[], tab_with=None):
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
            if tab_with is not None:
                self.tabifyDockWidget(tab_with, dock)
            return layout

        ####### Videos #######
        videos_layout = _make_dock("Videos")
        self.videosTable = VideosTable()
        videos_layout.addWidget(self.videosTable)
        hb = QHBoxLayout()
        btn = QPushButton("Add video")
        btn.clicked.connect(self.addVideo); hb.addWidget(btn)
        btn = QPushButton("Remove video")
        btn.clicked.connect(self.removeVideo); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        videos_layout.addWidget(hbw)

        ####### Skeleton #######
        skeleton_layout = _make_dock("Skeleton", tab_with=videos_layout.parent().parent())

        gb = QGroupBox("Nodes")
        vb = QVBoxLayout()
        self.skeletonNodesTable = SkeletonNodesTable(self.skeleton)
        vb.addWidget(self.skeletonNodesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New node")
        btn.clicked.connect(self.newNode); hb.addWidget(btn)
        btn = QPushButton("Delete node")
        btn.clicked.connect(self.deleteNode); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        gb = QGroupBox("Edges")
        vb = QVBoxLayout()
        self.skeletonEdgesTable = SkeletonEdgesTable(self.skeleton)
        vb.addWidget(self.skeletonEdgesTable)
        hb = QHBoxLayout()
        self.skeletonEdgesSrc = QComboBox(); self.skeletonEdgesSrc.setEditable(False); self.skeletonEdgesSrc.currentIndexChanged.connect(self.selectSkeletonEdgeSrc)
        hb.addWidget(self.skeletonEdgesSrc)
        self.skeletonEdgesDst = QComboBox(); self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        btn = QPushButton("Add edge")
        btn.clicked.connect(self.newEdge); hb.addWidget(btn)
        btn = QPushButton("Delete edge")
        btn.clicked.connect(self.deleteEdge); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        ####### Instances #######
        instances_layout = _make_dock("Instances")
        self.instancesTable = LabeledFrameTable()
        instances_layout.addWidget(self.instancesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New instance")
        btn.clicked.connect(self.newInstance); hb.addWidget(btn)
        btn = QPushButton("Delete instance")
        btn.clicked.connect(self.deleteInstance); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        instances_layout.addWidget(hbw)

        ####### Points #######
        # points_layout = _make_dock("Points", tab_with=instances_layout.parent().parent())
        # self.pointsTable = _make_table(["id", "frameIdx", "instanceId", "x", "y", "node", "visible"])
        # # self.pointsTable = _make_table_df(self.labels.points)
        # points_layout.addWidget(self.pointsTable)

        ####### Training #######
        training_layout = _make_dock("Training")
        gb = QGroupBox("Data representation")
        fl = QFormLayout()
        self.dataRange = QComboBox(); self.dataRange.addItems(["[0, 1]", "[-1, 1]"]); self.dataRange.setEditable(False)
        fl.addRow("Range:", self.dataRange)
        # TODO: range ([0, 1], [-1, 1])
        # TODO: normalization (z-score, CLAHE)
        self.dataScale = QDoubleSpinBox(); self.dataScale.setMinimum(0.25); self.dataScale.setValue(1.0)
        fl.addRow("Scale:", self.dataScale)
        
        gb.setLayout(fl)
        training_layout.addWidget(gb)

        gb = QGroupBox("Augmentation")
        fl = QFormLayout()
        self.augmentationRotation = QDoubleSpinBox(); self.augmentationRotation.setRange(0, 180); self.augmentationRotation.setValue(15.0)
        fl.addRow("Rotation:", self.augmentationRotation)
        self.augmentationFlipH = QCheckBox()
        fl.addRow("Flip (horizontal):", self.augmentationFlipH)
        # self.augmentationScaling = QDoubleSpinBox(); self.augmentationScaling.setRange(0.1, 2); self.augmentationScaling.setValue(1.0)
        # fl.addRow("Scaling:", self.augmentationScaling)
        gb.setLayout(fl)
        training_layout.addWidget(gb)

        gb = QGroupBox("Confidence maps")
        fl = QFormLayout()
        self.confmapsArchitecture = QComboBox(); self.confmapsArchitecture.addItems(["leap_cnn", "unet", "hourglass", "stacked_hourglass"]); self.confmapsArchitecture.setCurrentIndex(1); self.confmapsArchitecture.setEditable(False)
        fl.addRow("Architecture:", self.confmapsArchitecture)
        self.confmapsFilters = QSpinBox(); self.confmapsFilters.setMinimum(1); self.confmapsFilters.setValue(32)
        fl.addRow("Filters:", self.confmapsFilters)
        self.confmapsDepth = QSpinBox(); self.confmapsDepth.setMinimum(1); self.confmapsDepth.setValue(3)
        fl.addRow("Depth:", self.confmapsDepth)
        self.confmapsSigma = QDoubleSpinBox(); self.confmapsSigma.setMinimum(0.1); self.confmapsSigma.setValue(5.0)
        fl.addRow("Sigma:", self.confmapsSigma)
        btn = QPushButton("Train"); btn.clicked.connect(self.trainConfmaps)
        fl.addRow(btn)
        gb.setLayout(fl)
        training_layout.addWidget(gb)

        gb = QGroupBox("PAFs")
        fl = QFormLayout()
        self.pafsArchitecture = QComboBox(); self.pafsArchitecture.addItems(["leap_cnn", "unet", "hourglass", "stacked_hourglass"]); self.pafsArchitecture.setEditable(False)
        fl.addRow("Architecture:", self.pafsArchitecture)
        self.pafsFilters = QSpinBox(); self.pafsFilters.setMinimum(1); self.pafsFilters.setValue(32)
        fl.addRow("Filters:", self.pafsFilters)
        self.pafsDepth = QSpinBox(); self.pafsDepth.setMinimum(1); self.pafsDepth.setValue(3)
        fl.addRow("Depth:", self.pafsDepth)
        self.pafsSigma = QDoubleSpinBox(); self.pafsSigma.setMinimum(0.1); self.pafsSigma.setValue(5.0)
        fl.addRow("Sigma:", self.pafsSigma)
        btn = QPushButton("Train"); btn.clicked.connect(self.trainPAFs)
        fl.addRow(btn)
        gb.setLayout(fl)
        training_layout.addWidget(gb)


    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            event.ignore() # Kicks the event up to parent

    def importData(self, filename=None):
        show_msg = False
        # if filename is None:
        if not isinstance(filename, str):
            filters = ["JSON labels (*.json)", "HDF5 dataset (*.h5 *.hdf5)"]
            # filename, selected_filter = QFileDialog.getOpenFileName(self, dir="C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot", caption="Import labeled data...", filter=";;".join(filters))
            filename, selected_filter = QFileDialog.getOpenFileName(self, dir=None, caption="Import labeled data...", filter=";;".join(filters))
            show_msg = True
        
        if len(filename) == 0: return

        if filename.endswith(".json"):
            self.labels = Labels.load_json(filename)
            if show_msg:
                msgBox = QMessageBox(text=f"Imported {len(self.labels)} labeled frames.")
                msgBox.exec_()

            # Update UI tables
            self.videosTable.model().videos = self.labels.videos
            if len(self.labels.labels) > 0:
                if len(self.labels.labels[0].instances) > 0:
                    self.skeleton = self.labels.labels[0].instances[0].skeleton
                    self.skeletonNodesTable.model().skeleton = self.skeleton
                    self.skeletonEdgesTable.model().skeleton = self.skeleton
                    self.skeletonEdgesSrc.clear()
                    self.skeletonEdgesDst.clear()
                    self.skeletonEdgesSrc.addItems(self.skeleton.nodes)
                    
            # Load first video
            self.loadVideo(self.labels.videos[0])
            


    def addVideo(self, filename=None):
        # Browse for file
        if not isinstance(filename, str):
            filters = ["Media (*.mp4 *.avi)", "HDF5 dataset (*.h5 *.hdf5)"]
            filename, selected_filter = QFileDialog.getOpenFileName(self, caption="Add video...", filter=";;".join(filters))
            if len(filename) == 0: return

            # TODO: auto-detect HDF5 datasets and/or ask user

        # TODO: check for duplicate videos by filename? class methods for __eq__?

        # Instantiate video object
        video = Video.from_filename(filename)

        # Add to labels
        self.labels.add_video(video)

        # Load if no video currently loaded
        if self.video is None:
            self.loadVideo(video)

        # TODO: Update data model/view!

    def removeVideo(self):
        # Get selected video
        idx = self.videosTable.currentIndex()
        if not idx.isValid(): return
        video = self.labels.videos[idx.row()]

        # Count labeled frames for this video
        n = len(self.labels.find(video))

        # Warn if there are labels that will be deleted
        if n > 0:
            response = QMessageBox.critical(self, "Removing video with labels", f"{n} labeled frames in this video will be deleted, are you sure you want to remove this video?", QMessageBox.Yes, QMessageBox.No)
            if response == QMessageBox.No:
                return

        # Remove video
        self.labels.remove_video(video)

        # TODO: Update data model?

        # Update view if this was the current video
        if self.video == video:
            if len(self.labels.videos) == 0:
                self.player.reset()
                # TODO: update statusbar
            else:
                new_idx = min(idx.row(), len(self.labels.videos) - 1)
                self.loadVideo(self.labels.videos[new_idx])

    def loadVideo(self, video:Video):
        # Update current video instance
        self.video = video

        # Load video in player widget
        self.player.load_video(self.video)

        # Jump to last labeled frame
        last_label = self.labels.find_last(self.video)
        if last_label is not None:
            self.player.plot(last_label.frame_idx)


    def newNode(self):
        # Find new part name
        part_name = "new_part"
        i = 1
        while part_name in self.skeleton:
            part_name = f"new_part_{i}"
            i += 1

        # Add the node to the skeleton
        self.skeleton.add_node(part_name)

        # TODO: Update data model(s)?

        # TODO: Move this to unified data model
        # Update source edges dropdown
        self.skeletonEdgesSrc.clear()
        self.skeletonEdgesSrc.addItems(self.skeleton.nodes)
        self.skeletonEdgesDst.clear()
        

    def deleteNode(self):
        # Get selected node
        idx = self.skeletonNodesTable.currentIndex()
        if not idx.isValid(): return
        node = self.skeleton.nodes[idx.row()]

        # Check if there are instances with the skeleton that owns the node to be deleted
        affected_instances = list(self.labels.instances(skeleton=self.skeleton))

        # TODO: update instances correctly
        if len(affected_instances) > 0:
            return

        # Remove
        self.skeleton.delete_node(node)

        # TODO: Update data model(s)?

        # TODO: Move this to unified data model
        # Update source edges dropdown
        self.skeletonEdgesSrc.clear()
        self.skeletonEdgesSrc.addItems(self.skeleton.nodes)
        self.skeletonEdgesDst.clear()

        # TODO: Replot instances?

    def selectSkeletonEdgeSrc(self):
        # TODO: Move this to unified data model
        # TODO: Hook to signal emitted for updates to the node names

        # Get selected source node
        # src_node = self.skeletonEdgesSrc.currentText()
        src_node = self.skeleton.nodes[self.skeletonEdgesSrc.currentIndex()]

        # Find destination nodes
        dst_nodes = set(self.skeleton.nodes) - {src_node}

        # Find valid edges
        valid_edges = {(src_node, dst_node) for dst_node in dst_nodes} - set(self.skeleton.edges)

        # Filter down to valid destination nodes
        valid_dst_nodes = [dst_node for src_node, dst_node in valid_edges]

        # Update destination edges dropdown
        self.skeletonEdgesDst.clear()
        self.skeletonEdgesDst.addItems(valid_dst_nodes)

    def newEdge(self):
        # TODO: Move this to unified data model

        # Get selected nodes
        src_node = self.skeletonEdgesDst.currentText()
        dst_node = self.skeletonEdgesSrc.currentText()

        # Check if they're in the graph
        if src_node not in self.skeleton or dst_node not in self.skeleton:
            return

        # Add edge
        self.skeleton.add_edge(source=src_node, destination=dst_node)


    def deleteEdge(self):
        # TODO: Move this to unified data model

        # Get selected edge
        idx = self.skeletonEdgesTable.currentIndex()
        if not idx.isValid(): return
        edge = self.skeleton.edges[idx.row()]

        # Delete edge
        self.skeleton.delete_edge(source=edge[0], destination=edge[1])

        # TODO: Update things


    def newInstance(self):
        if self.labeled_frame is None:
            return

        new_instance = Instance(skeleton=self.skeleton)
        for node in self.skeleton.nodes:
            new_instance[node] = Point(x=np.random.rand() * self.video.width * 0.5, y=np.random.rand() * self.video.height * 0.5, visible=True)
        self.labeled_frame.instances.append(new_instance)

        if self.labeled_frame not in self.labels.labels:
            self.labels.append(self.labeled_frame)

        self.player.plot()

    def deleteInstance(self):
        pass

    def newProject(self):
        pass
    def openProject(self):
        pass
    def saveProject(self):
        pass
    def saveProjectAs(self):
        pass
    def exportData(self):
        pass
    # def close(self):
        # pass
    def checkVideoEncoding(self):
        pass
    def reencodeForSeeking(self):
        pass
    def addVideoFolder(self):
        pass
    def nextVideo(self):
        pass
    def previousVideo(self):
        pass
    def extractClip(self):
        pass
    def openDocumentation(self):
        pass
    def openKeyRef(self):
        pass
    def openAbout(self):
        pass


    def trainConfmaps(self):
        from sleap.nn.datagen import generate_images, generate_confidence_maps
        from sleap.nn.training import train

        imgs, keys = generate_images(self.labels)
        confmaps, _keys, points = generate_confidence_maps(self.labels)

        self.confmapModel = train(imgs, confmaps, test_size=0.1, batch_norm=False, num_filters=64, batch_size=4, num_epochs=100, steps_per_epoch=100)

    def trainPAFs(self):
        pass


    def newFrame(self, player, frame_idx):

        labeled_frame = [label for label in self.labels.labels if label.video == self.video and label.frame_idx == frame_idx]
        self.labeled_frame = labeled_frame[0] if len(labeled_frame) > 0 else LabeledFrame(video=self.video, frame_idx=frame_idx)
        self.instancesTable.model().labeled_frame = self.labeled_frame

        for i, instance in enumerate(self.labeled_frame.instances):
            qt_instance = QtInstance(instance=instance, color=self.cmap[i])
            player.view.scene.addItem(qt_instance)

        # self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.video)}  |  Labeled frames (video/total): {self.labels.instances[self.labels.instances.videoId == 1].frameIdx.nunique()}/{len(self.labels)}  |  Instances (frame/total): {len(frame_instances)}/{self.labels.points.instanceId.nunique()}")
        self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.video)}")


def main(*args, **kwargs):
    app = QApplication([])
    app.setApplicationName("sLEAP Label")
    window = MainWindow(*args, **kwargs)
    window.showMaximized()
    app.exec_()

if __name__ == "__main__":

    main(import_data="tests/data/json_format_v1/centered_pair.json")

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

from sleap.gui.video import QtVideoPlayer, QtInstance, QtEdge, QtNode
from sleap.io.video import Video, HDF5Video, MediaVideo
from sleap.io.labels import Labels

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    https://stackoverflow.com/a/45262758/1939934
    """

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                # if(index.column() != 0):
                #     return str('%.2f'%self._data.values[index.row()][index.column()])
                # else:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[section]
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return str(self._data.index[section])
        return None

    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        return flags


class MainWindow(QMainWindow):
    def __init__(self, data_path=None, video=None, *args, **kwargs):
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
        self.initialize_gui()

        if data_path is not None:
            self.importData(data_path)

        # TODO: auto-add video to clean project if no data provided
        # TODO: auto-select video if data provided, or add it to project
        if video is not None:
            self.addVideo(video)

    def initialize_gui(self):

        ####### Menus #######
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction("New project")
        fileMenu.addAction("Open project")
        fileMenu.addAction("Save")
        fileMenu.addAction("Save as...")
        fileMenu.addSeparator()
        fileMenu.addAction("Import...").triggered.connect(self.importData)
        fileMenu.addAction("Export...")
        fileMenu.addSeparator()
        fileMenu.addAction("&Quit").triggered.connect(self.close)

        videoMenu = self.menuBar().addMenu("Video")
        videoMenu.addAction("Check video encoding")
        videoMenu.addAction("Reencode for seeking")
        videoMenu.addSeparator()
        videoMenu.addAction("Add video").triggered.connect(self.addVideo)
        videoMenu.addAction("Add folder")
        videoMenu.addAction("Next video")
        videoMenu.addAction("Previous video")
        videoMenu.addSeparator()
        videoMenu.addAction("Extract clip")

        viewMenu = self.menuBar().addMenu("View")

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Documentation")
        helpMenu.addAction("Keyboard reference")
        helpMenu.addAction("About")

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

        def _make_table(cols):
            table = QTableWidget()
            table.setColumnCount(len(cols))
            table.setHorizontalHeaderLabels(cols)
            table.verticalHeader().hide()
            return table

        def _make_table_df(df):
            table = QTableView()
            table.setModel(PandasModel(df))
            return table

        ####### Videos #######
        videos_layout = _make_dock("Videos")
        self.videosTable = _make_table_df(self.labels.videos)
        videos_layout.addWidget(self.videosTable)

        ####### Skeleton #######
        skeleton_layout = _make_dock("Skeleton", tab_with=videos_layout.parent().parent())

        gb = QGroupBox("Nodes")
        vb = QVBoxLayout()
        self.skeletonNodesTable = _make_table(["id", "name", "symmetry"])
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
        self.skeletonEdgesTable = _make_table(["source", "destination"])
        vb.addWidget(self.skeletonEdgesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New edge")
        btn.clicked.connect(self.newEdge); hb.addWidget(btn)
        btn = QPushButton("Delete edge")
        btn.clicked.connect(self.deleteEdge); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        ####### Instances #######
        instances_layout = _make_dock("Instances")
        # self.instancesTable = _make_table(["id", "videoId", "frameIdx", "complete", "trackId"])
        self.instancesTable = _make_table_df(self.labels.instances)
        instances_layout.addWidget(self.instancesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New instance")
        btn.clicked.connect(self.newInstance); hb.addWidget(btn)
        btn = QPushButton("Delete instance")
        btn.clicked.connect(self.deleteInstance); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        instances_layout.addWidget(hbw)

        ####### Points #######
        points_layout = _make_dock("Points", tab_with=instances_layout.parent().parent())
        # self.pointsTable = _make_table(["id", "frameIdx", "instanceId", "x", "y", "node", "visible"])
        self.pointsTable = _make_table_df(self.labels.points)
        points_layout.addWidget(self.pointsTable)

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
            self.labels = Labels(filename)
            if show_msg:
                msgBox = QMessageBox(text=f"Imported {len(self.labels)} labeled frames.")
                msgBox.exec_()

            # Update UI tables
            self.videosTable.setModel(PandasModel(self.labels.videos))
            self.instancesTable.setModel(PandasModel(self.labels.instances))
            self.pointsTable.setModel(PandasModel(self.labels.points))

            # Load first video
            vid = self.labels.videos.iloc[0]
            if vid.format == "media":
                self.loadVideo(MediaVideo(vid.filepath, grayscale=vid.channels == 1))


    def addVideo(self, filename=None):
        if not isinstance(filename, str):
            filters = ["Media (*.mp4 *.avi)", "HDF5 dataset (*.h5 *.hdf5)"]
            filename, selected_filter = QFileDialog.getOpenFileName(self, dir=None, caption="Add video...", filter=";;".join(filters))
            if len(filename) == 0: return

        self.loadVideo(Video.from_media(filename))

    def removeVideo(self):
        pass

    def loadVideo(self, video:Video):
        # TODO: use video id
        self.video = video
        self.player.load_video(self.video)


    def newNode(self):
        pass
    def deleteNode(self):
        pass

    def newEdge(self):
        pass
    def deleteEdge(self):
        pass

    def newInstance(self):
        pass
    def deleteInstance(self):
        pass


    def trainConfmaps(self):
        from sleap.nn.datagen import generate_images, generate_confidence_maps
        from sleap.nn.training import train

        imgs, keys = generate_images(self.labels)
        confmaps, _keys, points = generate_confidence_maps(self.labels)

        self.confmapModel = train(imgs, confmaps, test_size=0.1, batch_norm=False, num_filters=64, batch_size=4, num_epochs=100, steps_per_epoch=100)


    def newFrame(self, player, frame_idx):
        pass
        # TODO: use video id
        # frame_instances = self.labels.get_frame_instances(self.labels.videos.id.loc[0], frame_idx)

        # for i, instance in enumerate(frame_instances):
        #     qt_instance = QtInstance(instance=instance, color=self.cmap[i])
        #     player.view.scene.addItem(qt_instance)

        # self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.video)}  |  Labeled frames (video/total): {self.labels.instances[self.labels.instances.videoId == 1].frameIdx.nunique()}/{len(self.labels)}  |  Instances (frame/total): {len(frame_instances)}/{self.labels.points.instanceId.nunique()}")
        # self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.video)}")


if __name__ == "__main__":

    # from sleap.io.video import HDF5Video
    # vid = HDF5Video("../../tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5", "/box", input_format="channels_first")
    
    # TODO: unfuck this
    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        data_path = "D:/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"

    vid_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.mp4"
    app = QApplication([])
    app.setApplicationName("sLEAP Label")
    # window = MainWindow()
    # window = MainWindow(video=vid)
    # window = MainWindow(data_path=data_path)
    window = MainWindow(video=vid_path)
    window.showMaximized()
    app.exec_()

    # window.loadVideo(Video.from_media(vid_path))


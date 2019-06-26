from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt

from PySide2.QtGui import QKeyEvent, QKeySequence

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout
from PySide2.QtWidgets import QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
from PySide2.QtWidgets import QTableWidget, QTableView, QTableWidgetItem
from PySide2.QtWidgets import QMenu, QAction
from PySide2.QtWidgets import QFileDialog, QMessageBox

import os
import sys
import copy
import yaml
from pkg_resources import Requirement, resource_filename
from pathlib import PurePath

import numpy as np
import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, PredictedInstance, Point, LabeledFrame, Track
from sleap.io.video import Video, HDF5Video, MediaVideo
from sleap.io.dataset import Labels
from sleap.gui.video import QtVideoPlayer
from sleap.gui.tracks import TrackTrailManager
from sleap.gui.dataviews import VideosTable, SkeletonNodesTable, SkeletonEdgesTable, \
    LabeledFrameTable, SkeletonNodeModel, SuggestionsTable
from sleap.gui.importvideos import ImportVideos
from sleap.gui.formbuilder import YamlFormWidget
from sleap.gui.suggestions import VideoFrameSuggestions

OPEN_IN_NEW = True

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
        self.video_idx = None
        self.mark_idx = None
        self.filename = None
        self._menu_actions = dict()
        self._buttons = dict()

        self._trail_manager = None

        self._show_labels = True
        self._show_edges = True
        self._show_trails = False
        self._color_predicted = False
        self._auto_zoom = False

        self.changestack_clear()
        self.initialize_gui()

        if data_path is not None:
            pass

        if import_data is not None:
            self.importData(import_data)

        # TODO: auto-add video to clean project if no data provided
        # TODO: auto-select video if data provided, or add it to project
        if video is not None:
            self.addVideo(video)

    def changestack_push(self, change=None):
        """Add to stack of changes made by user."""
        # Currently the change doesn't store any data, and we're only using this
        # to determine if there are unsaved changes. Eventually we could use this
        # to support undo/redo.
        self._change_stack.append(change)

    def changestack_savepoint(self):
        self.changestack_push("SAVE")

    def changestack_clear(self):
        self._change_stack = list()

    def changestack_start_atomic(self, change=None):
        pass

    def changestack_end_atomic(self):
        pass

    def changestack_has_changes(self) -> bool:
        # True iff there are no unsaved changed
        if len(self._change_stack) == 0: return False
        if self._change_stack[-1] == "SAVE": return False
        return True

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, x):
        self._filename = x
        if x is not None: self.setWindowTitle(x)

    def initialize_gui(self):

        shortcut_yaml = resource_filename(Requirement.parse("sleap"),"sleap/config/shortcuts.yaml")
        with open(shortcut_yaml, 'r') as f:
            shortcuts = yaml.load(f, Loader=yaml.SafeLoader)

        for action in shortcuts:
            key_string = shortcuts.get(action, None)
            key_string = "" if key_string is None else key_string
            if "." in key_string:
                shortcuts[action] = eval(key_string)

        ####### Video player #######
        self.player = QtVideoPlayer()
        self.player.changedPlot.connect(self.newFrame)
        self.player.changedData.connect(lambda inst: self.changestack_push("viewer change"))
        self.player.view.instanceDoubleClicked.connect(self.newInstance)
        self.setCentralWidget(self.player)

        ####### Status bar #######
        self.statusBar() # Initialize status bar

        ####### Menus #######
        fileMenu = self.menuBar().addMenu("File")
        self._menu_actions["new"] = fileMenu.addAction("&New Project", self.newProject, shortcuts["new"])
        self._menu_actions["open"] = fileMenu.addAction("&Open Project...", self.openProject, shortcuts["open"])
        self._menu_actions["save"] = fileMenu.addAction("&Save", self.saveProject, shortcuts["save"])
        self._menu_actions["save as"] = fileMenu.addAction("Save As...", self.saveProjectAs, shortcuts["save as"])
        fileMenu.addSeparator()
        self._menu_actions["close"] = fileMenu.addAction("Quit", self.close, shortcuts["close"])

        videoMenu = self.menuBar().addMenu("Video")
        # videoMenu.addAction("Check video encoding").triggered.connect(self.checkVideoEncoding)
        # videoMenu.addAction("Reencode for seeking").triggered.connect(self.reencodeForSeeking)
        # videoMenu.addSeparator()
        self._menu_actions["add videos"] = videoMenu.addAction("Add Videos...", self.addVideo, shortcuts["add videos"])
        self._menu_actions["next video"] = videoMenu.addAction("Next Video", self.nextVideo, shortcuts["next video"])
        self._menu_actions["prev video"] = videoMenu.addAction("Previous Video", self.previousVideo, shortcuts["prev video"])
        videoMenu.addSeparator()
        self._menu_actions["mark frame"] = videoMenu.addAction("Mark Frame", self.markFrame, shortcuts["mark frame"])
        self._menu_actions["goto marked"] = videoMenu.addAction("Go to Marked Frame", self.goMarkedFrame, shortcuts["goto marked"])
        self._menu_actions["extract clip"] = videoMenu.addAction("Extract Clip...", self.extractClip, shortcuts["extract clip"])

        labelMenu = self.menuBar().addMenu("Labels")
        self._menu_actions["add instance"] = labelMenu.addAction("Add Instance", self.newInstance, shortcuts["add instance"])
        self._menu_actions["delete instance"] = labelMenu.addAction("Delete Instance", self.deleteSelectedInstance, shortcuts["delete instance"])
        self.track_menu = labelMenu.addMenu("Set Instance Track")
        self._menu_actions["transpose"] = labelMenu.addAction("Transpose Instance Tracks", self.transposeInstance, shortcuts["transpose"])
        self._menu_actions["select next"] = labelMenu.addAction("Select Next Instance", self.player.view.nextSelection, shortcuts["select next"])
        self._menu_actions["clear selection"] = labelMenu.addAction("Clear Selection", self.player.view.clearSelection, shortcuts["clear selection"])
        labelMenu.addSeparator()
        self._menu_actions["goto next"] = labelMenu.addAction("Next Labeled Frame", self.nextLabeledFrame, shortcuts["goto next"])
        self._menu_actions["goto prev"] = labelMenu.addAction("Previous Labeled Frame", self.previousLabeledFrame, shortcuts["goto prev"])

        self._menu_actions["goto next suggestion"] = labelMenu.addAction("Next Suggestion", self.nextSuggestedFrame, shortcuts["goto next suggestion"])
        self._menu_actions["goto prev suggestion"] = labelMenu.addAction("Previous Suggestion", lambda:self.nextSuggestedFrame(-1), shortcuts["goto prev suggestion"])

        labelMenu.addSeparator()
        self._menu_actions["show labels"] = labelMenu.addAction("Show Node Names", self.toggleLabels, shortcuts["show labels"])
        self._menu_actions["show edges"] = labelMenu.addAction("Show Edges", self.toggleEdges, shortcuts["show edges"])
        self._menu_actions["show trails"] = labelMenu.addAction("Show Trails", self.toggleTrails, shortcuts["show trails"])

        self.trailLengthMenu = labelMenu.addMenu("Trail Length")
        for length_option in (4, 10, 20):
            menu_item = self.trailLengthMenu.addAction(f"{length_option}",
                            lambda x=length_option: self.setTrailLength(x))
            menu_item.setCheckable(True)

        self._menu_actions["color predicted"] = labelMenu.addAction("Color Predicted Instances", self.toggleColorPredicted, shortcuts["color predicted"])
        labelMenu.addSeparator()
        self._menu_actions["fit"] = labelMenu.addAction("Fit Instances to View", self.toggleAutoZoom, shortcuts["fit"])

        self._menu_actions["show labels"].setCheckable(True); self._menu_actions["show labels"].setChecked(self._show_labels)
        self._menu_actions["show edges"].setCheckable(True); self._menu_actions["show edges"].setChecked(self._show_edges)
        self._menu_actions["show trails"].setCheckable(True); self._menu_actions["show trails"].setChecked(self._show_trails)
        self._menu_actions["color predicted"].setCheckable(True); self._menu_actions["color predicted"].setChecked(self._color_predicted)
        self._menu_actions["fit"].setCheckable(True)

        predictionMenu = self.menuBar().addMenu("Predict")
        self._menu_actions["active learning"] = predictionMenu.addAction("Run Active Learning...", self.runActiveLearning)
        self._menu_actions["visualize models"] = predictionMenu.addAction("Visualize Model Outputs...", self.visualizeOutputs)
        self._menu_actions["import predictions"] = predictionMenu.addAction("Import Predictions...", self.importPredictions)
        self._menu_actions["import predictions"].setEnabled(False)

        viewMenu = self.menuBar().addMenu("View")

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Documentation", self.openDocumentation)
        helpMenu.addAction("Keyboard Reference", self.openKeyRef)
        helpMenu.addAction("About", self.openAbout)

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
        btn = QPushButton("Show video")
        btn.clicked.connect(self.activateSelectedVideo); hb.addWidget(btn)
        self._buttons["show video"] = btn
        btn = QPushButton("Add videos")
        btn.clicked.connect(self.addVideo); hb.addWidget(btn)
        btn = QPushButton("Remove video")
        btn.clicked.connect(self.removeVideo); hb.addWidget(btn)
        self._buttons["remove video"] = btn
        hbw = QWidget(); hbw.setLayout(hb)
        videos_layout.addWidget(hbw)

        self.videosTable.doubleClicked.connect(self.activateSelectedVideo)

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
        self._buttons["delete node"] = btn
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
        self.skeletonEdgesSrc.setModel(SkeletonNodeModel(self.skeleton))
        hb.addWidget(self.skeletonEdgesSrc)
        hb.addWidget(QLabel("to"))
        self.skeletonEdgesDst = QComboBox(); self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        self.skeletonEdgesDst.setModel(SkeletonNodeModel(self.skeleton, lambda: self.skeletonEdgesSrc.currentText()))
        btn = QPushButton("Add edge")
        btn.clicked.connect(self.newEdge); hb.addWidget(btn)
        self._buttons["add edge"] = btn
        btn = QPushButton("Delete edge")
        btn.clicked.connect(self.deleteEdge); hb.addWidget(btn)
        self._buttons["delete edge"] = btn
        hbw = QWidget(); hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        hb = QHBoxLayout()
        btn = QPushButton("Load Skeleton")
        btn.clicked.connect(self.openSkeleton); hb.addWidget(btn)
        btn = QPushButton("Save Skeleton")
        btn.clicked.connect(self.saveSkeleton); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        skeleton_layout.addWidget(hbw)

        # update edge UI when change to nodes
        self.skeletonNodesTable.model().dataChanged.connect(self.updateEdges)
        self.skeletonNodesTable.model().dataChanged.connect(self.changestack_push)

        ####### Instances #######
        instances_layout = _make_dock("Instances")
        self.instancesTable = LabeledFrameTable(labels=self.labels)
        instances_layout.addWidget(self.instancesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New instance")
        btn.clicked.connect(lambda x: self.newInstance()); hb.addWidget(btn)
        btn = QPushButton("Delete instance")
        btn.clicked.connect(self.deleteSelectedInstance); hb.addWidget(btn)
        self._buttons["delete instance"] = btn
        hbw = QWidget(); hbw.setLayout(hb)
        instances_layout.addWidget(hbw)

        def update_instance_table_selection():
            cur_video_instance = self.player.view.getSelection()
            if cur_video_instance is None: cur_video_instance = -1
            table_index = self.instancesTable.model().createIndex(cur_video_instance, 0)
            self.instancesTable.setCurrentIndex(table_index)

        self.instancesTable.selectionChangedSignal.connect(lambda row: self.player.view.selectInstance(row, from_all=True, signal=False))
        self.player.view.updatedSelection.connect(update_instance_table_selection)

        # update track UI when change to track name
        self.instancesTable.model().dataChanged.connect(self.updateTrackMenu)
        self.instancesTable.model().dataChanged.connect(self.changestack_push)

        ####### Suggestions #######
        suggestions_layout = _make_dock("Labeling Suggestions")
        self.suggestionsTable = SuggestionsTable(labels=self.labels)
        suggestions_layout.addWidget(self.suggestionsTable)

        hb = QHBoxLayout()
        btn = QPushButton("Prev")
        btn.clicked.connect(lambda:self.nextSuggestedFrame(-1)); hb.addWidget(btn)
        self.suggested_count_label = QLabel()
        hb.addWidget(self.suggested_count_label)
        btn = QPushButton("Next")
        btn.clicked.connect(lambda:self.nextSuggestedFrame()); hb.addWidget(btn)
        hbw = QWidget(); hbw.setLayout(hb)
        suggestions_layout.addWidget(hbw)

        suggestions_yaml = resource_filename(Requirement.parse("sleap"),"sleap/config/suggestions.yaml")
        form_wid = YamlFormWidget(yaml_file=suggestions_yaml, title="Generate Suggestions")
        form_wid.mainAction.connect(self.generateSuggestions)
        suggestions_layout.addWidget(form_wid)

        self.suggestionsTable.doubleClicked.connect(lambda table_idx: self.gotoVideoAndFrame(*self.labels.get_suggestions()[table_idx.row()]))

        #
        # Set timer to update state of gui at regular intervals
        #
        self.update_gui_timer = QtCore.QTimer()
        self.update_gui_timer.timeout.connect(self.update_gui_state)
        self.update_gui_timer.start(0)

        ####### Points #######
        # points_layout = _make_dock("Points", tab_with=instances_layout.parent().parent())
        # self.pointsTable = _make_table(["id", "frameIdx", "instanceId", "x", "y", "node", "visible"])
        # # self.pointsTable = _make_table_df(self.labels.points)
        # points_layout.addWidget(self.pointsTable)

        ####### Training #######
#         training_layout = _make_dock("Training")
#         gb = QGroupBox("Data representation")
#         fl = QFormLayout()
#         self.dataRange = QComboBox(); self.dataRange.addItems(["[0, 1]", "[-1, 1]"]); self.dataRange.setEditable(False)
#         fl.addRow("Range:", self.dataRange)
#         # TODO: range ([0, 1], [-1, 1])
#         # TODO: normalization (z-score, CLAHE)
#         self.dataScale = QDoubleSpinBox(); self.dataScale.setMinimum(0.25); self.dataScale.setValue(1.0)
#         fl.addRow("Scale:", self.dataScale)
# 
#         gb.setLayout(fl)
#         training_layout.addWidget(gb)
# 
#         gb = QGroupBox("Augmentation")
#         fl = QFormLayout()
#         self.augmentationRotation = QDoubleSpinBox(); self.augmentationRotation.setRange(0, 180); self.augmentationRotation.setValue(15.0)
#         fl.addRow("Rotation:", self.augmentationRotation)
#         self.augmentationFlipH = QCheckBox()
#         fl.addRow("Flip (horizontal):", self.augmentationFlipH)
#         # self.augmentationScaling = QDoubleSpinBox(); self.augmentationScaling.setRange(0.1, 2); self.augmentationScaling.setValue(1.0)
#         # fl.addRow("Scaling:", self.augmentationScaling)
#         gb.setLayout(fl)
#         training_layout.addWidget(gb)
# 
#         gb = QGroupBox("Confidence maps")
#         fl = QFormLayout()
#         self.confmapsArchitecture = QComboBox(); self.confmapsArchitecture.addItems(["leap_cnn", "unet", "hourglass", "stacked_hourglass"]); self.confmapsArchitecture.setCurrentIndex(1); self.confmapsArchitecture.setEditable(False)
#         fl.addRow("Architecture:", self.confmapsArchitecture)
#         self.confmapsFilters = QSpinBox(); self.confmapsFilters.setMinimum(1); self.confmapsFilters.setValue(32)
#         fl.addRow("Filters:", self.confmapsFilters)
#         self.confmapsDepth = QSpinBox(); self.confmapsDepth.setMinimum(1); self.confmapsDepth.setValue(3)
#         fl.addRow("Depth:", self.confmapsDepth)
#         self.confmapsSigma = QDoubleSpinBox(); self.confmapsSigma.setMinimum(0.1); self.confmapsSigma.setValue(5.0)
#         fl.addRow("Sigma:", self.confmapsSigma)
#         btn = QPushButton("Train"); btn.clicked.connect(self.trainConfmaps)
#         fl.addRow(btn)
#         gb.setLayout(fl)
#         training_layout.addWidget(gb)
# 
#         gb = QGroupBox("PAFs")
#         fl = QFormLayout()
#         self.pafsArchitecture = QComboBox(); self.pafsArchitecture.addItems(["leap_cnn", "unet", "hourglass", "stacked_hourglass"]); self.pafsArchitecture.setEditable(False)
#         fl.addRow("Architecture:", self.pafsArchitecture)
#         self.pafsFilters = QSpinBox(); self.pafsFilters.setMinimum(1); self.pafsFilters.setValue(32)
#         fl.addRow("Filters:", self.pafsFilters)
#         self.pafsDepth = QSpinBox(); self.pafsDepth.setMinimum(1); self.pafsDepth.setValue(3)
#         fl.addRow("Depth:", self.pafsDepth)
#         self.pafsSigma = QDoubleSpinBox(); self.pafsSigma.setMinimum(0.1); self.pafsSigma.setValue(5.0)
#         fl.addRow("Sigma:", self.pafsSigma)
#         btn = QPushButton("Train"); btn.clicked.connect(self.trainPAFs)
#         fl.addRow(btn)
#         gb.setLayout(fl)
#         training_layout.addWidget(gb)

    def update_gui_state(self):
        has_selected_instance = (self.player.view.getSelection() is not None)
        has_unsaved_changes = self.changestack_has_changes()
        has_multiple_videos = (self.labels is not None and len(self.labels.videos) > 1)
        has_labeled_frames = (len(self.labels.find(self.video)) > 0)
        has_suggestions = (len(self.labels.suggestions) > 0)
        has_multiple_instances = (self.labeled_frame is not None and len(self.labeled_frame.instances) > 1)
        # todo: exclude predicted instances from count
        has_nodes_selected = (self.skeletonEdgesSrc.currentIndex() > -1 and
                             self.skeletonEdgesDst.currentIndex() > -1)

        # Update menus

        self.track_menu.setEnabled(has_selected_instance)
        self._menu_actions["clear selection"].setEnabled(has_selected_instance)
        self._menu_actions["delete instance"].setEnabled(has_selected_instance)

        self._menu_actions["transpose"].setEnabled(has_multiple_instances)

        self._menu_actions["save"].setEnabled(has_unsaved_changes)
        self._menu_actions["goto marked"].setEnabled(self.mark_idx is not None)

        self._menu_actions["next video"].setEnabled(has_multiple_videos)
        self._menu_actions["prev video"].setEnabled(has_multiple_videos)

        self._menu_actions["goto next"].setEnabled(has_labeled_frames)
        self._menu_actions["goto prev"].setEnabled(has_labeled_frames)

        self._menu_actions["goto next suggestion"].setEnabled(has_suggestions)
        self._menu_actions["goto prev suggestion"].setEnabled(has_suggestions)

        # Update buttons
        self._buttons["add edge"].setEnabled(has_nodes_selected)
        self._buttons["delete edge"].setEnabled(self.skeletonEdgesTable.currentIndex().isValid())
        self._buttons["delete node"].setEnabled(self.skeletonNodesTable.currentIndex().isValid())
        self._buttons["show video"].setEnabled(self.videosTable.currentIndex().isValid())
        self._buttons["remove video"].setEnabled(self.videosTable.currentIndex().isValid())
        self._buttons["delete instance"].setEnabled(self.instancesTable.currentIndex().isValid())

    def update_data_views(self):
        self.videosTable.model().videos = self.labels.videos

        self.skeletonNodesTable.model().skeleton = self.skeleton
        self.skeletonEdgesTable.model().skeleton = self.skeleton
        self.skeletonEdgesSrc.model().skeleton = self.skeleton
        self.skeletonEdgesDst.model().skeleton = self.skeleton

        self.instancesTable.model().labels = self.labels
        self.instancesTable.model().labeled_frame = self.labeled_frame

        self.suggestionsTable.model().labels = self.labels

        # update count of suggested frames w/ labeled instances
        suggestion_status_text = ""
        suggestion_list = self.labels.get_suggestions()
        if len(suggestion_list):
            suggestion_label_counts = [self.labels.instance_count(video, frame_idx)
                for (video, frame_idx) in suggestion_list]
            labeled_count = len(suggestion_list) - suggestion_label_counts.count(0)
            suggestion_status_text = f"{labeled_count}/{len(suggestion_list)} labeled"
        self.suggested_count_label.setText(suggestion_status_text)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            event.ignore() # Kicks the event up to parent

    def plotFrame(self, *args, **kwargs):
        """Wrap call to player.plot so we can redraw/update things."""
        self.player.plot(*args, **kwargs)
        self.player.showLabels(self._show_labels)
        self.player.showEdges(self._show_edges)
        if self._auto_zoom:
            self.player.zoomToFit()

    def importData(self, filename=None, do_load=True):
        show_msg = False

        if len(filename) == 0: return

        def gui_video_callback(video_list, new_paths=[os.path.dirname(filename)]):
            from PySide2.QtWidgets import QFileDialog, QMessageBox

            has_shown_prompt = False # have we already alerted user about missing files?

            # Check each video
            for video_item in video_list:
                if "backend" in video_item and "filename" in video_item["backend"]:
                    current_filename = video_item["backend"]["filename"]
                    # check if we can find video
                    if not os.path.exists(current_filename):
                        is_found = False

                        current_basename = os.path.basename(current_filename)
                        # handle unix, windows, or mixed paths
                        if current_basename.find("/") > -1:
                            current_basename = current_basename.split("/")[-1]
                        if current_basename.find("\\") > -1:
                            current_basename = current_basename.split("\\")[-1]
                        
                        # First see if we can find the file in another directory,
                        # and if not, prompt the user to find the file.

                        # We'll check in the current working directory, and if the user has
                        # already found any missing videos, check in the directory of those.
                        for path_dir in new_paths:
                            check_path = os.path.join(path_dir, current_basename)
                            if os.path.exists(check_path):
                                # we found the file in a different directory
                                video_item["backend"]["filename"] = check_path
                                is_found = True
                                break

                        # if we found this file, then move on to the next file
                        if is_found: continue

                        # Since we couldn't find the file on our own, prompt the user.

                        if not has_shown_prompt:
                            QMessageBox(text=f"We're unable to locate one or more video files for this project. Please locate {current_basename}.").exec_()
                            has_shown_prompt = True

                        current_root, current_ext = os.path.splitext(current_basename)
                        caption = f"Please locate {current_basename}..."
                        filters = [f"{current_root} file (*{current_ext})", "Any File (*.*)"]
                        new_filename, _ = QFileDialog.getOpenFileName(self, dir=None, caption=caption, filter=";;".join(filters))
                        # if we got an answer, then update filename for video
                        if len(new_filename):
                            video_item["backend"]["filename"] = new_filename
                            # keep track of the directory chosen by user
                            new_paths.append(os.path.dirname(new_filename))

        has_loaded = False
        labels = None
        if type(filename) == Labels:
            labels = filename
            filename = None
            has_loaded = True
        elif filename.endswith(".json"):
            labels = Labels.load_json(filename, video_callback=gui_video_callback)
            has_loaded = True
        elif filename.endswith(".mat"):
            labels = Labels.load_mat(filename)
            has_loaded = True
        elif filename.endswith(".csv"):
            # for now, the only csv we support is the DeepLabCut format
            labels = Labels.load_deeplabcut_csv(filename)
            has_loaded = True

        if do_load:
            self.labels = labels
            self.filename = filename

            if has_loaded:
                self.changestack_clear()
                self._trail_manager = TrackTrailManager(self.labels, self.player.view.scene)
                self.setTrailLength(self._trail_manager.trail_length)

                if show_msg:
                    msgBox = QMessageBox(text=f"Imported {len(self.labels)} labeled frames.")
                    msgBox.exec_()

                if len(self.labels.labels) > 0:
                    if len(self.labels.labels[0].instances) > 0:
                        # TODO: add support for multiple skeletons
                        self.skeleton = self.labels.labels[0].instances[0].skeleton

                # Update UI tables
                self.update_data_views()

                # Load first video
                self.loadVideo(self.labels.videos[0], 0)

                # Update track menu options
                self.updateTrackMenu()
        else:
            return labels

    def updateTrackMenu(self):
        self.track_menu.clear()
        for track in self.labels.tracks:
            key_command = ""
            if self.labels.tracks.index(track) < 9:
                key_command = Qt.CTRL + Qt.Key_0 + self.labels.tracks.index(track) + 1
            self.track_menu.addAction(f"{track.name}", lambda x=track:self.setInstanceTrack(x), key_command)
        self.track_menu.addAction("New Track", self.addTrack, Qt.CTRL + Qt.Key_0)

    def activateSelectedVideo(self, x):
        # Get selected video
        idx = self.videosTable.currentIndex()
        if not idx.isValid(): return
        self.loadVideo(self.labels.videos[idx.row()], idx.row())

    def addVideo(self, filename=None):
        # Browse for file
        video = None
        if isinstance(filename, str):
            video = Video.from_filename(filename)
            # Add to labels
            self.labels.add_video(video)
        else:
            import_list = ImportVideos().go()
            for import_item in import_list:
                # Create Video object
                video = Video.from_filename(**import_item["params"])
                # Add to labels
                self.labels.add_video(video)
                self.changestack_push("add video")

        # Load if no video currently loaded
        if self.video is None:
            self.loadVideo(video, len(self.labels.videos)-1)

        # Update data model/view
        self.update_data_views()

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
        self.changestack_push("remove video")

        # Update data model
        self.update_data_views()

        # Update view if this was the current video
        if self.video == video:
            if len(self.labels.videos) == 0:
                self.player.reset()
                # TODO: update statusbar
            else:
                new_idx = min(idx.row(), len(self.labels.videos) - 1)
                self.loadVideo(self.labels.videos[new_idx], new_idx)

    def loadVideo(self, video:Video, video_idx: int = None):
        # Clear video frame mark
        self.mark_idx = None

        # Update current video instance
        self.video = video
        self.video_idx = video_idx if video_idx is not None else self.labels.videos.index(video)

        # Load video in player widget
        self.player.load_video(self.video)

        # Annotate labeled frames on seekbar
        self.updateSeekbarMarks()

        # Jump to last labeled frame
        last_label = self.labels.find_last(self.video)
        if last_label is not None:
            self.plotFrame(last_label.frame_idx)

    def openSkeleton(self):
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = QFileDialog.getOpenFileName(self, dir=None, caption="Open skeleton...", filter=";;".join(filters))

        if len(filename) == 0: return

        if filename.endswith(".json"):
            self.skeleton = Skeleton.load_json(filename)
        elif filename.endswith((".h5", ".hdf5")):
            sk_list = Skeleton.load_all_hdf5(filename)
            if len(sk_list):
                self.skeleton = sk_list[0]

        # Update data model
        self.update_data_views()

    def saveSkeleton(self):
        default_name = "skeleton.json"
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = QFileDialog.getSaveFileName(self, caption="Save As...", dir=default_name, filter=";;".join(filters))

        if len(filename) == 0: return

        if filename.endswith(".json"):
            self.skeleton.save_json(filename)
        elif filename.endswith((".h5", ".hdf5")):
            self.skeleton.save_hdf5(filename)

    def newNode(self):
        # Find new part name
        part_name = "new_part"
        i = 1
        while part_name in self.skeleton:
            part_name = f"new_part_{i}"
            i += 1

        # Add the node to the skeleton
        self.skeleton.add_node(part_name)
        self.changestack_push("new node")

        # Update data model
        self.update_data_views()

        self.plotFrame()

    def deleteNode(self):
        # Get selected node
        idx = self.skeletonNodesTable.currentIndex()
        if not idx.isValid(): return
        node = self.skeleton.nodes[idx.row()]

        # Remove
        self.skeleton.delete_node(node)
        self.changestack_push("delete node")

        # Update data model
        self.update_data_views()

        # Replot instances
        self.plotFrame()

    def selectSkeletonEdgeSrc(self):
        self.skeletonEdgesDst.model().skeleton = self.skeleton

    def updateEdges(self):
        self.update_data_views()
        self.plotFrame()

    def newEdge(self):
        # TODO: Move this to unified data model

        # Get selected nodes
        src_node = self.skeletonEdgesSrc.currentText()
        dst_node = self.skeletonEdgesDst.currentText()

        # Check if they're in the graph
        if src_node not in self.skeleton or dst_node not in self.skeleton:
            return

        # Add edge
        self.skeleton.add_edge(source=src_node, destination=dst_node)
        self.changestack_push("new edge")

        # Update data model
        self.update_data_views()

        self.plotFrame()


    def deleteEdge(self):
        # TODO: Move this to unified data model

        # Get selected edge
        idx = self.skeletonEdgesTable.currentIndex()
        if not idx.isValid(): return
        edge = self.skeleton.edges[idx.row()]

        # Delete edge
        self.skeleton.delete_edge(source=edge[0], destination=edge[1])
        self.changestack_push("delete edge")

        # Update data model
        self.update_data_views()

        self.plotFrame()

    def updateSeekbarMarks(self):
        # If there are tracks, mark whether track has instance in frame.
        if len(self.labels.tracks):
            self.player.seekbar.setTracksFromLabels(self.labels)
        # Otherwise, mark which frames have any instances.
        else:
            # list of frame_idx for simple markers for labeled frames
            labeled_marks = [lf.frame_idx for lf in self.labels.find(self.video) if len(lf.instances)]
            user_labeled = [lf.frame_idx for lf in self.labels.find(self.video) if len(lf.user_instances)]
            # "f" for suggestions with instances and "o" for those without
            # "f" means "filled", "o" means "open"
            # "p" for suggestions with only predicted instances
            def mark_type(frame):
                if frame in user_labeled:
                    return "f"
                elif frame in labeled_marks:
                    return "p"
                else:
                    return "o"
            # list of (type, frame) tuples for suggestions
            suggestion_marks = [(mark_type(frame_idx), frame_idx)
                for frame_idx in self.labels.get_video_suggestions(self.video)]
            # combine marks for labeled frame and marks for suggested frames
            all_marks = labeled_marks + suggestion_marks

            self.player.seekbar.setMarks(all_marks)

    def generateSuggestions(self, params):
        new_suggestions = dict()
        for video in self.labels.videos:
            new_suggestions[video] = VideoFrameSuggestions.suggest(video, params)
        self.labels.set_suggestions(new_suggestions)
        self.update_data_views()
        self.updateSeekbarMarks()

    def runActiveLearning(self):
        from sleap.gui.active import ActiveLearningDialog

        ret = ActiveLearningDialog(self.filename, self.labels).exec_()

        if ret:
            # we ran active learning so update display/ui
            self.plotFrame()
            self.updateSeekbarMarks()
            self.changestack_push("new predictions")

    def visualizeOutputs(self):
        filters = ["HDF5 output (*.h5 *.hdf5)"]
        models_dir = None
        if self.filename is not None:
            models_dir = os.path.join(os.path.dirname(self.filename), "models/")
        filename, selected_filter = QFileDialog.getOpenFileName(self, dir=models_dir, caption="Import model outputs...", filter=";;".join(filters))

        if len(filename) == 0: return
    
        # show confmaps and pafs
        from sleap.gui.confmapsplot import show_confmaps_from_h5
        from sleap.gui.quiverplot import show_pafs_from_h5
        
        conf_win = show_confmaps_from_h5(filename)
        conf_win.move(200, 200)
        paf_win = show_pafs_from_h5(filename)
        paf_win.move(220+conf_win.rect().width(), 200)

    def importPredictions(self):
        filters = ["JSON labels (*.json)", "HDF5 dataset (*.h5 *.hdf5)", "Matlab dataset (*.mat)", "DeepLabCut csv (*.csv)"]
        filename, selected_filter = QFileDialog.getOpenFileName(self, dir=None, caption="Import labeled data...", filter=";;".join(filters))

        if len(filename) == 0: return
        
        # load predicted labels
        predicted_labels = self.importData(filename, do_load=False)
        
        # add to current labels project
        # TODO: the objects (videos and skeletons) won't match, so we need to fix this

        # update video for predicted frame to match video object in labels project
        for lf in predicted_labels.labeled_frames:
            for video in self.labels.videos:
                if lf.video.filename == video.filename:
                    lf.video = video
        self.labels.labeled_frames.extend(predicted_labels.labeled_frames)
        print(f"total lf: {len(self.labels.labeled_frames)}")
        # update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self.changestack_push("new predictions")

    def newInstance(self, copy_instance=None):
        if self.labeled_frame is None:
            return

        # FIXME: filter by skeleton type

        used_tracks = [inst.track
                       for inst in self.labeled_frame.instances
                       if type(inst) == Instance
                       ]
        unused_predictions = [inst
                              for inst in self.labeled_frame.instances
                              if inst.track not in used_tracks
                              and type(inst) == PredictedInstance
                              ]

        from_prev_frame = False
        if copy_instance is None:
            selected_idx = self.player.view.getSelection()
            if selected_idx is not None:
                # If the user has selected an instance, copy that one.
                copy_instance = self.labeled_frame.instances[selected_idx]
            elif len(unused_predictions):
                # If there are predicted instances that don't correspond to an instance
                # in this frame, use the first predicted instance without matching instance.
                copy_instance = unused_predictions[0]
            else:
                # Otherwise, if there are instances in previous frames,
                # copy the points from one of those instances.
                prev_idx = self.previousLabeledFrameIndex()
                if prev_idx is not None:
                    prev_instances = self.getInstancesFromFrameIdx(prev_idx)
                    if len(prev_instances) > len(self.labeled_frame.instances):
                        # If more instances in previous frame than current, then use the
                        # first unmatched instance.
                        copy_instance = prev_instances[len(self.labeled_frame.instances)]
                        from_prev_frame = True
                    elif len(self.labeled_frame.instances):
                        # Otherwise, if there are already instances in current frame,
                        # copy the points from the last instance added to frame.
                        copy_instance = self.labeled_frame.instances[-1]
                    elif len(prev_instances):
                        # Otherwise use the last instance added to previous frame.
                        copy_instance = prev_instances[-1]
                        from_prev_frame = True
        new_instance = Instance(skeleton=self.skeleton)
        for node in self.skeleton.nodes:
            if copy_instance is not None and node in copy_instance.nodes:
                new_instance[node] = copy.copy(copy_instance[node])
            else:
                # mark the node as not "visible" if we're copying from a predicted instance without this node
                is_visible = copy_instance is None or not hasattr(copy_instance, "score")#type(copy_instance) != PredictedInstance
                new_instance[node] = Point(x=np.random.rand() * self.video.width * 0.5, y=np.random.rand() * self.video.height * 0.5, visible=is_visible)
        # If we're copying a predicted instance or from another frame, copy the track
        if hasattr(copy_instance, "score") or from_prev_frame:
            new_instance.track = copy_instance.track

        # Add the instance
        self.labeled_frame.instances.append(new_instance)
        self.changestack_push("new instance")

        if self.labeled_frame not in self.labels.labels:
            self.labels.append(self.labeled_frame)
            self.changestack_push("new labeled frame")

        # update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self.updateTrackMenu()

    def deleteSelectedInstance(self):
        selected_inst = self.player.view.getSelectionInstance()
        if selected_inst is None: return

        self.labeled_frame.instances.remove(selected_inst)
        self.changestack_push("delete instance")

        self.plotFrame()
        self.updateSeekbarMarks()

    def addTrack(self):
        track_numbers_used = [int(track.name)
                                for track in self.labels.tracks
                                if track.name.isnumeric()]
        next_number = max(track_numbers_used, default=0) + 1
        new_track = Track(spawned_on=self.player.frame_idx, name=next_number)

        self.changestack_start_atomic("add track")
        self.labels.tracks.append(new_track)
        self.changestack_push("new track")
        self.setInstanceTrack(new_track)
        self.changestack_push("set track")
        self.changestack_end_atomic()

        # update track menu and seekbar
        self.updateTrackMenu()
        self.updateSeekbarMarks()

    def setInstanceTrack(self, new_track):
        idx = self.player.view.getSelection()
        if idx is None: return

        selected_instance = self.labeled_frame.instances_to_show[idx]

        old_track = selected_instance.track

        # When setting track for an instance that doesn't already have a track set,
        # we don't want to update *every* instance that also doesn't have a track.
        # Instead, we'll use index in instance list as pseudo-track.
        if old_track is None:
            old_track = idx

        self._swap_tracks(new_track, old_track)

        self.player.view.selectInstance(idx)

    def _swap_tracks(self, new_track, old_track):
        start, end = self.player.seekbar.getSelection()
        if start < end:
            # If range is selected in seekbar, use that
            frame_range = range(start, end)
        else:
            # Otherwise, range is current to last frame
            frame_range = range(self.player.frame_idx, self.video.frames)

        # get all instances in old/new tracks
        old_track_instances = self._get_track_instances(old_track, frame_range)
        new_track_instances = self._get_track_instances(new_track, frame_range)

        # swap new to old tracks on all instances
        for instance in old_track_instances:
            instance.track = new_track
        # old_track can be `Track` or int
        # If int, it's index in instance list which we'll use as a pseudo-track,
        # but we won't set instances currently on new_track to old_track.
        if type(old_track) == Track:
            for instance in new_track_instances:
                instance.track = old_track

        self.changestack_push("swap tracks")

        self.plotFrame()
        self.updateSeekbarMarks()

    def _get_track_instances(self, track, frame_range=None):
        """Get instances for a given track.

        Args:
            track: the `Track` or int ("pseudo-track" index to instance list)
            frame_range (optional):
                If specific, only return instances on frames in range.
                If None, return all instances for given track.
        Returns:
            list of `Instance` objects
        """

        def does_track_match(inst, tr, labeled_frame):
            match = False
            if type(tr) == Track and inst.track is tr:
                match = True
            elif (type(tr) == int and labeled_frame.instances.index(inst) == tr
                    and inst.track is None):
                match = True
            return match

        track_instances = [instance
                            for labeled_frame in self.labels.labeled_frames
                            for instance in labeled_frame.instances
                            if does_track_match(instance, track, labeled_frame)
                                and (frame_range is None or labeled_frame.frame_idx in frame_range)]
        return track_instances

    def transposeInstance(self):
        # We're currently identifying instances by numeric index, so it's
        # impossible to (e.g.) have a single instance which we identify
        # as the second instance in some other frame.

        # For the present, we can only "transpose" if there are multiple instances.
        if len(self.labeled_frame.instances) < 2: return
        # If there are just two instances, transpose them.
        if len(self.labeled_frame.instances) == 2:
            self._transpose_instances((0,1))
        # If there are more than two, then we need the user to select the instances.
        else:
            self.player.onSequenceSelect(seq_len = 2,
                                         on_success = self._transpose_instances,
                                         on_each = self._transpose_message,
                                         on_failure = lambda x:self.updateStatusMessage()
                                         )

    def _transpose_message(self, instance_ids:list):
        word = "next" if len(instance_ids) else "first"
        self.updateStatusMessage(f"Please select the {word} instance to transpose...")

    def _transpose_instances(self, instance_ids:list):
        if len(instance_ids) != 2: return

        idx_0 = instance_ids[0]
        idx_1 = instance_ids[1]
        # Swap order in array (just for this frame) for when we don't have tracks
        instance_0 = self.labeled_frame.instances_to_show[idx_0]
        instance_1 = self.labeled_frame.instances_to_show[idx_1]
        # Swap tracks for current and subsequent frames
        old_track, new_track = instance_0.track, instance_1.track
        if old_track is not None and new_track is not None:
            self._swap_tracks(new_track, old_track)

        # instance_0.track, instance_1.track = instance_1.track, instance_0.track

        self.plotFrame()

    def newProject(self):
        window = MainWindow()
        window.showMaximized()

    def openProject(self, first_open=False):
        filters = ["JSON labels (*.json)", "HDF5 dataset (*.h5 *.hdf5)", "Matlab dataset (*.mat)", "DeepLabCut csv (*.csv)"]
        filename, selected_filter = QFileDialog.getOpenFileName(self, dir=None, caption="Import labeled data...", filter=";;".join(filters))

        if len(filename) == 0: return

        if OPEN_IN_NEW and not first_open:
            new_window = MainWindow()
            new_window.showMaximized()
            new_window.importData(filename)
        else:
            self.importData(filename)

    def saveProject(self):
        if self.filename is not None:
            filename = self.filename
            if filename.endswith(".json"):
                Labels.save_json(labels = self.labels, filename = filename)
            # Mark savepoint in change stack
            self.changestack_savepoint()
            # Redraw. Not sure why, but sometimes we need to do this.
            self.plotFrame()
        else:
            # No filename (must be new project), so treat as "Save as"
            self.saveProjectAs()

    def saveProjectAs(self):
        default_name = self.filename if self.filename is not None else "untitled.json"
        p = PurePath(default_name)
        default_name = str(p.with_name(f"{p.stem} copy{p.suffix}"))

        filters = ["JSON labels (*.json)", "HDF5 dataset (*.h5 *.hdf5)"]
        filename, selected_filter = QFileDialog.getSaveFileName(self, caption="Save As...", dir=default_name, filter=";;".join(filters))

        if len(filename) == 0: return

        if filename.endswith(".json"):
            Labels.save_json(labels = self.labels, filename = filename)
            self.filename = filename
            # Mark savepoint in change stack
            self.changestack_savepoint()
            # Redraw. Not sure why, but sometimes we need to do this.
            self.plotFrame()
        else:
            QMessageBox(text=f"File not saved. Only .json currently implemented.").exec_()

    def closeEvent(self, event):
        if not self.changestack_has_changes():
            # No unsaved changes, so accept event (close)
            event.accept()
        else:
            msgBox = QMessageBox()
            msgBox.setText("Do you want to save the changes to this project?")
            msgBox.setInformativeText("If you don't save, your changes will be lost.")
            msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msgBox.setDefaultButton(QMessageBox.Save)

            ret_val = msgBox.exec_()

            if ret_val == QMessageBox.Cancel:
                # cancel close by ignoring event
                event.ignore()
            elif ret_val == QMessageBox.Discard:
                # don't save, just close
                event.accept()
            elif ret_val == QMessageBox.Save:
                # save
                self.saveProject()
                # accept avent (close)
                event.accept()

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
        new_idx = self.video_idx+1
        new_idx = 0 if new_idx >= len(self.labels.videos) else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def previousVideo(self):
        new_idx = self.video_idx-1
        new_idx = len(self.labels.videos)-1 if new_idx < 0 else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def markFrame(self):
        self.mark_idx = self.player.frame_idx

    def goMarkedFrame(self):
        self.plotFrame(self.mark_idx)

    def extractClip(self):
        start, end = self.player.seekbar.getSelection()
        if start < end:
            clip_labeled_frames = [copy.copy(lf) for lf in self.labels.labeled_frames if lf.frame_idx in range(start, end)]
            clip_video_frames = self.video.get_frames(range(start, end))
            clip_video = Video.from_numpy(clip_video_frames)

            # adjust video and frame_idx for clip labeled frames
            for lf in clip_labeled_frames:
                lf.video = clip_video
                lf.frame_idx -= start

            # create new labels object with clip
            clip_labels = Labels(clip_labeled_frames)

            # clip_window = QtVideoPlayer(video=clip_video)
            clip_window = MainWindow()
            clip_window.showMaximized()
            clip_window.importData(clip_labels)
            clip_window.show()

    def previousLabeledFrameIndex(self):
        prev_idx = None
        cur_idx = self.player.frame_idx
        frame_indexes = [frame.frame_idx for frame in self.labels.find(self.video)]
        frame_indexes.sort()
        if len(frame_indexes):
            prev_idx = max(filter(lambda idx: idx < cur_idx, frame_indexes), default=frame_indexes[-1])
        return prev_idx

    def previousLabeledFrame(self):
        prev_idx = self.previousLabeledFrameIndex()
        if prev_idx is not None:
            self.plotFrame(prev_idx)

    def nextLabeledFrame(self):
        cur_idx = self.player.frame_idx
        frame_indexes = [frame.frame_idx for frame in self.labels.find(self.video)]
        frame_indexes.sort()
        if len(frame_indexes):
            next_idx = min(filter(lambda idx: idx > cur_idx, frame_indexes), default=frame_indexes[0])
            self.plotFrame(next_idx)

    def nextSuggestedFrame(self, seek_direction=1):
        next_video, next_frame = self.labels.get_next_suggestion(self.video, self.player.frame_idx, seek_direction)
        if next_video is not None:
            self.gotoVideoAndFrame(next_video, next_frame)
        if next_frame is not None:
            selection_idx = self.labels.get_suggestions().index((next_video, next_frame))
            self.suggestionsTable.selectRow(selection_idx)

    def gotoVideoAndFrame(self, video, frame_idx):
        if video != self.video:
            # switch to the other video
            self.loadVideo(video)
        self.plotFrame(frame_idx)

    def toggleLabels(self):
        self._show_labels = not self._show_labels
        self._menu_actions["show labels"].setChecked(self._show_labels)
        self.player.showLabels(self._show_labels)

    def toggleEdges(self):
        self._show_edges = not self._show_edges
        self._menu_actions["show edges"].setChecked(self._show_edges)
        self.player.showEdges(self._show_edges)

    def toggleTrails(self):
        self._show_trails = not self._show_trails
        self._menu_actions["show trails"].setChecked(self._show_trails)
        self.plotFrame()

    def setTrailLength(self, trail_length):
        self._trail_manager.trail_length = trail_length

        for menu_item in self.trailLengthMenu.children():
            if menu_item.text() == str(trail_length):
                menu_item.setChecked(True)
            else:
                menu_item.setChecked(False)

        if self.video is not None: self.plotFrame()

    def toggleColorPredicted(self):
        self._color_predicted = not self._color_predicted
        self._menu_actions["color predicted"].setChecked(self._color_predicted)
        self.plotFrame()

    def toggleAutoZoom(self):
        self._auto_zoom = not self._auto_zoom
        self._menu_actions["fit"].setChecked(self._auto_zoom)
        if not self._auto_zoom:
            self.player.view.clearZoom()
        self.plotFrame()

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

    def getInstancesFromFrameIdx(self, frame_idx):
        labeled_frame = [label for label in self.labels.labels if label.video == self.video and label.frame_idx == frame_idx]
        instances = labeled_frame[0].instances if len(labeled_frame) > 0 else []
        return instances

    def newFrame(self, player, frame_idx):

        labeled_frame = [label for label in self.labels.labels if label.video == self.video and label.frame_idx == frame_idx]
        self.labeled_frame = labeled_frame[0] if len(labeled_frame) > 0 else LabeledFrame(video=self.video, frame_idx=frame_idx)

        self.update_data_views()

        count_no_track = 0
        for i, instance in enumerate(self.labeled_frame.instances_to_show):
            if instance.track in self.labels.tracks:
                track_idx = self.labels.tracks.index(instance.track)
            else:
                # Instance without track
                track_idx = len(self.labels.tracks) + count_no_track
                count_no_track += 1
            is_predicted = hasattr(instance,"score")

            player.addInstance(instance=instance,
                               color=self.cmap[track_idx%len(self.cmap)],
                               predicted=is_predicted,
                               color_predicted=self._color_predicted)

        if self._trail_manager is not None and self._show_trails:
            self._trail_manager.add_trails_to_scene(frame_idx)

        player.view.updatedViewer.emit()

        self.updateStatusMessage()

    def updateStatusMessage(self, message = None):
        if message is None:
            message = f"Frame: {self.player.frame_idx+1}/{len(self.video)}"

        self.statusBar().showMessage(message)
        # self.statusBar().showMessage(f"Frame: {self.player.frame_idx+1}/{len(self.video)}  |  Labeled frames (video/total): {self.labels.instances[self.labels.instances.videoId == 1].frameIdx.nunique()}/{len(self.labels)}  |  Instances (frame/total): {len(frame_instances)}/{self.labels.points.instanceId.nunique()}")

def main(*args, **kwargs):
    app = QApplication([])
    app.setApplicationName("sLEAP Label")

    window = MainWindow(*args, **kwargs)
    window.showMaximized()

    if "import_data" not in kwargs:
        window.openProject(first_open=True)

    app.exec_()

if __name__ == "__main__":

    kwargs = dict()
    if len(sys.argv) > 1:
        kwargs["import_data"] = sys.argv[1]

    main(**kwargs)

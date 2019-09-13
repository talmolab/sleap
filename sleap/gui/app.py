from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt

from PySide2.QtGui import QKeyEvent, QKeySequence

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout
from PySide2.QtWidgets import QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
from PySide2.QtWidgets import QTableWidget, QTableView, QTableWidgetItem
from PySide2.QtWidgets import QMenu, QAction
from PySide2.QtWidgets import QFileDialog, QMessageBox

import copy
import operator
import os
import sys
import yaml

from pkg_resources import Requirement, resource_filename
from pathlib import PurePath

import numpy as np
import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, PredictedInstance, Point, LabeledFrame, Track
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.gui.video import QtVideoPlayer
from sleap.gui.dataviews import VideosTable, SkeletonNodesTable, SkeletonEdgesTable, \
    LabeledFrameTable, SkeletonNodeModel, SuggestionsTable
from sleap.gui.importvideos import ImportVideos
from sleap.gui.formbuilder import YamlFormWidget
from sleap.gui.suggestions import VideoFrameSuggestions

from sleap.gui.overlays.tracks import TrackColorManager, TrackTrailOverlay
from sleap.gui.overlays.instance import InstanceOverlay

OPEN_IN_NEW = True

class MainWindow(QMainWindow):
    labels: Labels
    skeleton: Skeleton
    video: Video

    def __init__(self, data_path=None, video=None, import_data=None, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.labels = Labels()
        self.skeleton = Skeleton()
        self.labeled_frame = None
        self.video = None
        self.video_idx = None
        self.mark_idx = None
        self.filename = None
        self._menu_actions = dict()
        self._buttons = dict()
        self._child_windows = dict()

        self._color_palette = "standard"
        self._color_manager = TrackColorManager(self.labels, self._color_palette)

        self.overlays = dict()

        self._show_labels = True
        self._show_edges = True
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
        self.player = QtVideoPlayer(color_manager=self._color_manager)
        self.player.changedPlot.connect(self.newFrame)
        self.player.changedData.connect(lambda inst: self.changestack_push("viewer change"))
        self.player.view.instanceDoubleClicked.connect(self.doubleClickInstance)
        self.player.seekbar.selectionChanged.connect(lambda: self.updateStatusMessage())
        self.setCentralWidget(self.player)

        ####### Status bar #######
        self.statusBar() # Initialize status bar

        self.load_overlays()

        ####### Menus #######

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        self._menu_actions["new"] = fileMenu.addAction("&New Project", self.newProject, shortcuts["new"])
        self._menu_actions["open"] = fileMenu.addAction("&Open Project...", self.openProject, shortcuts["open"])
        fileMenu.addSeparator()
        self._menu_actions["add videos"] = fileMenu.addAction("Add Videos...", self.addVideo, shortcuts["add videos"])
        fileMenu.addSeparator()
        self._menu_actions["save"] = fileMenu.addAction("&Save", self.saveProject, shortcuts["save"])
        self._menu_actions["save as"] = fileMenu.addAction("Save As...", self.saveProjectAs, shortcuts["save as"])
        fileMenu.addSeparator()
        self._menu_actions["close"] = fileMenu.addAction("Quit", self.close, shortcuts["close"])

        ### Go Menu ###

        goMenu = self.menuBar().addMenu("Go")

        self._menu_actions["goto next"] = goMenu.addAction("Next Labeled Frame", self.nextLabeledFrame, shortcuts["goto next"])
        self._menu_actions["goto prev"] = goMenu.addAction("Previous Labeled Frame", self.previousLabeledFrame, shortcuts["goto prev"])

        self._menu_actions["goto next user"] = goMenu.addAction("Next User Labeled Frame", self.nextUserLabeledFrame, shortcuts["goto next user"])

        self._menu_actions["goto next suggestion"] = goMenu.addAction("Next Suggestion", self.nextSuggestedFrame, shortcuts["goto next suggestion"])
        self._menu_actions["goto prev suggestion"] = goMenu.addAction("Previous Suggestion", lambda:self.nextSuggestedFrame(-1), shortcuts["goto prev suggestion"])

        self._menu_actions["goto next track"] = goMenu.addAction("Next Track Spawn Frame", self.nextTrackFrame, shortcuts["goto next track"])

        goMenu.addSeparator()

        self._menu_actions["next video"] = goMenu.addAction("Next Video", self.nextVideo, shortcuts["next video"])
        self._menu_actions["prev video"] = goMenu.addAction("Previous Video", self.previousVideo, shortcuts["prev video"])

        goMenu.addSeparator()

        self._menu_actions["goto frame"] = goMenu.addAction("Go to Frame...", self.gotoFrame, shortcuts["goto frame"])
        self._menu_actions["mark frame"] = goMenu.addAction("Mark Frame", self.markFrame, shortcuts["mark frame"])
        self._menu_actions["goto marked"] = goMenu.addAction("Go to Marked Frame", self.goMarkedFrame, shortcuts["goto marked"])


        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")

        viewMenu.addSeparator()
        self._menu_actions["color predicted"] = viewMenu.addAction("Color Predicted Instances", self.toggleColorPredicted, shortcuts["color predicted"])

        self.paletteMenu = viewMenu.addMenu("Color Palette")
        for palette_name in self._color_manager.palette_names:
            menu_item = self.paletteMenu.addAction(f"{palette_name}",
                            lambda x=palette_name: self.setPalette(x))
            menu_item.setCheckable(True)
        self.setPalette("standard")

        viewMenu.addSeparator()

        self._menu_actions["show labels"] = viewMenu.addAction("Show Node Names", self.toggleLabels, shortcuts["show labels"])
        self._menu_actions["show edges"] = viewMenu.addAction("Show Edges", self.toggleEdges, shortcuts["show edges"])
        self._menu_actions["show trails"] = viewMenu.addAction("Show Trails", self.toggleTrails, shortcuts["show trails"])

        self.trailLengthMenu = viewMenu.addMenu("Trail Length")
        for length_option in (4, 10, 20):
            menu_item = self.trailLengthMenu.addAction(f"{length_option}",
                            lambda x=length_option: self.setTrailLength(x))
            menu_item.setCheckable(True)

        viewMenu.addSeparator()

        self._menu_actions["fit"] = viewMenu.addAction("Fit Instances to View", self.toggleAutoZoom, shortcuts["fit"])

        viewMenu.addSeparator()

        # set menu checkmarks
        self._menu_actions["show labels"].setCheckable(True); self._menu_actions["show labels"].setChecked(self._show_labels)
        self._menu_actions["show edges"].setCheckable(True); self._menu_actions["show edges"].setChecked(self._show_edges)
        self._menu_actions["show trails"].setCheckable(True); self._menu_actions["show trails"].setChecked(self.overlays["trails"].show)
        self._menu_actions["color predicted"].setCheckable(True); self._menu_actions["color predicted"].setChecked(self.overlays["instance"].color_predicted)
        self._menu_actions["fit"].setCheckable(True)

        ### Label Menu ###

        labelMenu = self.menuBar().addMenu("Labels")
        self._menu_actions["add instance"] = labelMenu.addAction("Add Instance", self.newInstance, shortcuts["add instance"])
        self._menu_actions["delete instance"] = labelMenu.addAction("Delete Instance", self.deleteSelectedInstance, shortcuts["delete instance"])

        labelMenu.addSeparator()

        self.track_menu = labelMenu.addMenu("Set Instance Track")
        self._menu_actions["transpose"] = labelMenu.addAction("Transpose Instance Tracks", self.transposeInstance, shortcuts["transpose"])
        self._menu_actions["delete track"] = labelMenu.addAction("Delete Instance and Track", self.deleteSelectedInstanceTrack, shortcuts["delete track"])

        labelMenu.addSeparator()

        self._menu_actions["select next"] = labelMenu.addAction("Select Next Instance", self.player.view.nextSelection, shortcuts["select next"])
        self._menu_actions["clear selection"] = labelMenu.addAction("Clear Selection", self.player.view.clearSelection, shortcuts["clear selection"])

        labelMenu.addSeparator()

        ### Predict Menu ###

        predictionMenu = self.menuBar().addMenu("Predict")
        self._menu_actions["active learning"] = predictionMenu.addAction("Run Active Learning...", self.runActiveLearning, shortcuts["learning"])
        self._menu_actions["inference"] = predictionMenu.addAction("Run Inference...", self.runInference)
        self._menu_actions["learning expert"] = predictionMenu.addAction("Expert Controls...", self.runLearningExpert)
        predictionMenu.addSeparator()
        self._menu_actions["negative sample"] = predictionMenu.addAction("Mark Negative Training Sample...", self.markNegativeAnchor)
        predictionMenu.addSeparator()
        self._menu_actions["visualize models"] = predictionMenu.addAction("Visualize Model Outputs...", self.visualizeOutputs)
        self._menu_actions["import predictions"] = predictionMenu.addAction("Import Predictions...", self.importPredictions)
        predictionMenu.addSeparator()
        self._menu_actions["remove predictions"] = predictionMenu.addAction("Delete All Predictions...", self.deletePredictions)
        self._menu_actions["remove clip predictions"] = predictionMenu.addAction("Delete Predictions from Clip...", self.deleteClipPredictions, shortcuts["delete clip"])
        self._menu_actions["remove area predictions"] = predictionMenu.addAction("Delete Predictions from Area...", self.deleteAreaPredictions, shortcuts["delete area"])
        self._menu_actions["remove score predictions"] = predictionMenu.addAction("Delete Predictions with Low Score...", self.deleteLowScorePredictions)
        self._menu_actions["remove frame limit predictions"] = predictionMenu.addAction("Delete Predictions beyond Frame Limit...", self.deleteFrameLimitPredictions)
        predictionMenu.addSeparator()
        self._menu_actions["export frames"] = predictionMenu.addAction("Export Training Package...", self.exportLabeledFrames)
        self._menu_actions["export clip"] = predictionMenu.addAction("Export Labeled Clip...", self.exportLabeledClip, shortcuts["export clip"])

        ############

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
        self.update_gui_timer.start(0.1)

    def load_overlays(self):
        self.overlays["trails"] = TrackTrailOverlay(
                                    labels = self.labels,
                                    scene = self.player.view.scene,
                                    color_manager = self._color_manager)

        self.overlays["instance"] = InstanceOverlay(
                                    labels = self.labels,
                                    player = self.player,
                                    color_manager = self._color_manager)

    def update_gui_state(self):
        has_selected_instance = (self.player.view.getSelection() is not None)
        has_unsaved_changes = self.changestack_has_changes()
        has_multiple_videos = (self.labels is not None and len(self.labels.videos) > 1)
        has_labeled_frames = self.labels is not None and any((lf.video == self.video for lf in self.labels))
        has_suggestions = self.labels is not None and (len(self.labels.suggestions) > 0)
        has_tracks = self.labels is not None and (len(self.labels.tracks) > 0)
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

        self._menu_actions["goto next track"].setEnabled(has_tracks)

        # Update buttons
        self._buttons["add edge"].setEnabled(has_nodes_selected)
        self._buttons["delete edge"].setEnabled(self.skeletonEdgesTable.currentIndex().isValid())
        self._buttons["delete node"].setEnabled(self.skeletonNodesTable.currentIndex().isValid())
        self._buttons["show video"].setEnabled(self.videosTable.currentIndex().isValid())
        self._buttons["remove video"].setEnabled(self.videosTable.currentIndex().isValid())
        self._buttons["delete instance"].setEnabled(self.instancesTable.currentIndex().isValid())

    def update_data_views(self):
        if len(self.skeleton.nodes) == 0 and len(self.labels.skeletons):
             self.skeleton = self.labels.skeletons[0]

        self.videosTable.model().videos = self.labels.videos

        self.skeletonNodesTable.model().skeleton = self.skeleton
        self.skeletonEdgesTable.model().skeleton = self.skeleton
        self.skeletonEdgesSrc.model().skeleton = self.skeleton
        self.skeletonEdgesDst.model().skeleton = self.skeleton

        self.instancesTable.model().labels = self.labels
        self.instancesTable.model().labeled_frame = self.labeled_frame
        self.instancesTable.model().color_manager = self._color_manager

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
        if self.video is None: return

        self.player.plot(*args, **kwargs)
        self.player.showLabels(self._show_labels)
        self.player.showEdges(self._show_edges)
        if self._auto_zoom:
            self.player.zoomToFit()

    def importData(self, filename=None, do_load=True):
        show_msg = False

        if len(filename) == 0: return

        gui_video_callback = Labels.make_gui_video_callback(
                                    search_paths=[os.path.dirname(filename)])

        has_loaded = False
        labels = None
        if type(filename) == Labels:
            labels = filename
            filename = None
            has_loaded = True
        else:
            try:
                labels = Labels.load_file(filename, video_callback=gui_video_callback)
                has_loaded = True
            except ValueError as e:
                print(e)
                QMessageBox(text=f"Unable to load {filename}.").exec_()

        if do_load:

            self.labels = labels
            self.filename = filename

            if has_loaded:
                self.changestack_clear()
                self._color_manager.labels = self.labels
                self._color_manager.set_palette(self._color_palette)

                self.load_overlays()

                self.setTrailLength(self.overlays["trails"].trail_length)

                if show_msg:
                    msgBox = QMessageBox(text=f"Imported {len(self.labels)} labeled frames.")
                    msgBox.exec_()

                if len(self.labels.skeletons):
                    # TODO: add support for multiple skeletons
                    self.skeleton = self.labels.skeletons[0]

                # Update UI tables
                self.update_data_views()

                # Load first video
                if len(self.labels.videos):
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

        if self.skeleton not in self.labels:
            self.labels.skeletons.append(self.skeleton)
            self.changestack_push("new skeleton")

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
        self.player.seekbar.setTracksFromLabels(self.labels, self.video)

    def generateSuggestions(self, params):
        new_suggestions = dict()
        for video in self.labels.videos:
            new_suggestions[video] = VideoFrameSuggestions.suggest(
                                            video=video,
                                            labels=self.labels,
                                            params=params)

        self.labels.set_suggestions(new_suggestions)

        self.update_data_views()
        self.updateSeekbarMarks()

    def _frames_for_prediction(self):

        def remove_user_labeled(video, frames, user_labeled_frames=self.labels.user_labeled_frames):
            if len(frames) == 0: return frames
            video_user_labeled_frame_idxs = [lf.frame_idx for lf in user_labeled_frames
                                             if lf.video == video]
            return list(set(frames) - set(video_user_labeled_frame_idxs))

        selection = dict()
        selection["frame"] = {self.video: [self.player.frame_idx]}
        selection["clip"] = {self.video: list(range(*self.player.seekbar.getSelection()))}
        selection["video"] = {self.video: list(range(self.video.num_frames))}

        selection["suggestions"] = {
            video:remove_user_labeled(video, self.labels.get_video_suggestions(video))
            for video in self.labels.videos}

        selection["random"] = {
            video: remove_user_labeled(video, VideoFrameSuggestions.random(video=video))
            for video in self.labels.videos}

        return selection

    def _show_learning_window(self, mode):
        from sleap.gui.active import ActiveLearningDialog

        if self._child_windows.get(mode, None) is None:
            self._child_windows[mode] = ActiveLearningDialog(self.filename, self.labels, mode)
            self._child_windows[mode].learningFinished.connect(self.learningFinished)

        self._child_windows[mode].frame_selection = self._frames_for_prediction()
        self._child_windows[mode].open()

    def learningFinished(self):
        # we ran active learning so update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self.update_data_views()
        self.changestack_push("new predictions")

    def runLearningExpert(self):
        self._show_learning_window("expert")

    def runInference(self):
        self._show_learning_window("inference")

    def runActiveLearning(self):
        self._show_learning_window("learning")

    def visualizeOutputs(self):
        filters = ["Model (*.json)", "HDF5 output (*.h5 *.hdf5)"]

        # Default to opening from models directory from project
        models_dir = None
        if self.filename is not None:
            models_dir = os.path.join(os.path.dirname(self.filename), "models/")

        # Show dialog
        filename, selected_filter = QFileDialog.getOpenFileName(self, dir=models_dir, caption="Import model outputs...", filter=";;".join(filters))

        if len(filename) == 0: return

        if selected_filter == filters[0]:
            # Model as overlay datasource
            # This will show live inference results

            from sleap.gui.overlays.base import DataOverlay
            overlay = DataOverlay.from_model(filename, self.video, player=self.player)

            self.overlays["inference"] = overlay

        else:
            # HDF5 as overlay datasource
            # This will show saved inference results from previous run

            show_confmaps = True
            show_pafs = False

            if show_confmaps:
                from sleap.gui.overlays.confmaps import ConfmapOverlay
                confmap_overlay = ConfmapOverlay.from_h5(filename, player=self.player)
                self.player.changedPlot.connect(lambda parent, idx: confmap_overlay.add_to_scene(None, idx))

            if show_pafs:
                from sleap.gui.overlays.pafs import PafOverlay
                paf_overlay = PafOverlay.from_h5(filename, player=self.player)
                self.player.changedPlot.connect(lambda parent, idx: paf_overlay.add_to_scene(None, idx))

        self.plotFrame()

    def deletePredictions(self):

        predicted_instances = [(lf, inst) for lf in self.labels for inst in lf if type(inst) == PredictedInstance]

        resp = QMessageBox.critical(self,
                "Removing predicted instances",
                f"There are {len(predicted_instances)} predicted instances. "
                "Are you sure you want to delete these?",
                QMessageBox.Yes, QMessageBox.No)

        if resp == QMessageBox.No: return

        for lf, inst in predicted_instances:
            self.labels.remove_instance(lf, inst)

        self.plotFrame()
        self.updateSeekbarMarks()
        self.changestack_push("removed predictions")

    def deleteClipPredictions(self):

        predicted_instances = [(lf, inst)
                for lf in self.labels.find(self.video, frame_idx = range(*self.player.seekbar.getSelection()))
                for inst in lf
                if type(inst) == PredictedInstance]

        # If user selected an instance, then only delete for that track.
        selected_inst = self.player.view.getSelectionInstance()
        if selected_inst is not None:
            track = selected_inst.track
            if track == None:
                # If user selected an instance without a track, delete only
                # that instance and only on the current frame.
                predicted_instances = [(self.labeled_frame, selected_inst)]
            else:
                # Filter by track
                predicted_instances = list(filter(lambda x: x[1].track == track, predicted_instances))

        resp = QMessageBox.critical(self,
                "Removing predicted instances",
                f"There are {len(predicted_instances)} predicted instances. "
                "Are you sure you want to delete these?",
                QMessageBox.Yes, QMessageBox.No)

        if resp == QMessageBox.No: return

        # Delete the instances
        for lf, inst in predicted_instances:
            self.labels.remove_instance(lf, inst)

        self.plotFrame()
        self.updateSeekbarMarks()
        self.changestack_push("removed predictions")

    def deleteAreaPredictions(self):

        # Callback to delete after area has been selected
        def delete_area_callback(x0, y0, x1, y1):

            self.updateStatusMessage()

            # Make sure there was an area selected
            if x0==x1 or y0==y1: return

            min_corner = (x0, y0)
            max_corner = (x1, y1)

            def is_bounded(inst):
                points_array = inst.points_array(invisible_as_nan=True)
                valid_points = points_array[~np.isnan(points_array).any(axis=1)]

                is_gt_min = np.all(valid_points >= min_corner)
                is_lt_max = np.all(valid_points <= max_corner)
                return is_gt_min and is_lt_max

            # Find all instances contained in selected area
            predicted_instances = [(lf, inst) for lf in self.labels.find(self.video)
                                    for inst in lf
                                    if type(inst) == PredictedInstance
                                    and is_bounded(inst)]

            self._delete_confirm(predicted_instances)

        # Prompt the user to select area
        self.updateStatusMessage(f"Please select the area from which to remove instances. This will be applied to all frames.")
        self.player.onAreaSelection(delete_area_callback)

    def deleteLowScorePredictions(self):
        score_thresh, okay = QtWidgets.QInputDialog.getDouble(
                                self,
                                "Delete Instances with Low Score...",
                                "Score Below:",
                                1,
                                0, 100)
        if okay:
            # Find all instances contained in selected area
            predicted_instances = [(lf, inst) for lf in self.labels.find(self.video)
                                    for inst in lf
                                    if type(inst) == PredictedInstance
                                    and inst.score < score_thresh]

            self._delete_confirm(predicted_instances)

    def deleteFrameLimitPredictions(self):
        count_thresh, okay = QtWidgets.QInputDialog.getInt(
                                self,
                                "Limit Instances in Frame...",
                                "Maximum instances in a frame:",
                                3,
                                1, 100)
        if okay:
            predicted_instances = []
            # Find all instances contained in selected area
            for lf in self.labels.find(self.video):
                if len(lf.instances) > count_thresh:
                    # Get all but the count_thresh many instances with the highest score
                    extra_instances = sorted(lf.instances,
                                            key=operator.attrgetter('score')
                                            )[:-count_thresh]
                    predicted_instances.extend([(lf, inst) for inst in extra_instances])

            self._delete_confirm(predicted_instances)

    def _delete_confirm(self, lf_inst_list):

        # Confirm that we want to delete
        resp = QMessageBox.critical(self,
                "Removing predicted instances",
                f"There are {len(lf_inst_list)} predicted instances that would be deleted. "
                "Are you sure you want to delete these?",
                QMessageBox.Yes, QMessageBox.No)

        if resp == QMessageBox.No: return

        # Delete the instances
        for lf, inst in lf_inst_list:
            self.labels.remove_instance(lf, inst)

        # Update visuals
        self.plotFrame()
        self.updateSeekbarMarks()
        self.changestack_push("removed predictions")

    def markNegativeAnchor(self):
        def click_callback(x, y):
            self.updateStatusMessage()
            self.labels.add_negative_anchor(self.video, self.player.frame_idx, (x, y))
            self.changestack_push("add negative anchors")

        # Prompt the user to select area
        self.updateStatusMessage(f"Please click where you want a negative sample...")
        self.player.onPointSelection(click_callback)

    def importPredictions(self):
        filters = ["HDF5 dataset (*.h5 *.hdf5)", "JSON labels (*.json *.json.zip)"]
        filenames, selected_filter = QFileDialog.getOpenFileNames(self, dir=None, caption="Import labeled data...", filter=";;".join(filters))

        if len(filenames) == 0: return

        for filename in filenames:
            gui_video_callback = Labels.make_gui_video_callback(
                                    search_paths=[os.path.dirname(filename)])

            if filename.endswith((".h5", ".hdf5")):
                new_labels = Labels.load_hdf5(
                                filename,
                                match_to=self.labels,
                                video_callback=gui_video_callback)

            elif filename.endswith((".json", ".json.zip")):
                new_labels = Labels.load_json(
                                filename,
                                match_to=self.labels,
                                video_callback=gui_video_callback)

            self.labels.extend_from(new_labels)

            for vid in new_labels.videos:
                print(f"Labels imported for {vid.filename}")
                print(f"  frames labeled: {len(new_labels.find(vid))}")

        # update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self.update_data_views()
        self.changestack_push("new predictions")

    def doubleClickInstance(self, instance):
        # When a predicted instance is double-clicked, add a new instance
        if hasattr(instance, "score"):
            self.newInstance(copy_instance = instance)

        # When a regular instance is double-clicked, add any missing points
        else:
            # the rect that's currently visibile in the window view
            in_view_rect = self.player.view.mapToScene(self.player.view.rect()).boundingRect()

            for node in self.skeleton.nodes:
                if node.name not in instance.node_names or instance[node].isnan():
                    # pick random points within currently zoomed view
                    x = in_view_rect.x() + (in_view_rect.width() * 0.1) \
                        + (np.random.rand() * in_view_rect.width() * 0.8)
                    y = in_view_rect.y() + (in_view_rect.height() * 0.1) \
                        + (np.random.rand() * in_view_rect.height() * 0.8)
                    # set point for node
                    instance[node] = Point(x=x, y=y, visible=False)

            self.plotFrame()

    def newInstance(self, copy_instance=None):
        if self.labeled_frame is None:
            return

        # FIXME: filter by skeleton type

        from_predicted = copy_instance
        unused_predictions = self.labeled_frame.unused_predictions

        from_prev_frame = False
        if copy_instance is None:
            selected_idx = self.player.view.getSelection()
            if selected_idx is not None:
                # If the user has selected an instance, copy that one.
                copy_instance = self.labeled_frame.instances[selected_idx]
                from_predicted = copy_instance
            elif len(unused_predictions):
                # If there are predicted instances that don't correspond to an instance
                # in this frame, use the first predicted instance without matching instance.
                copy_instance = unused_predictions[0]
                from_predicted = copy_instance
            else:
                # Otherwise, if there are instances in previous frames,
                # copy the points from one of those instances.
                prev_idx = self.previousLabeledFrameIndex()
                if prev_idx is not None:
                    prev_instances = self.labels.find(self.video, prev_idx, return_new=True)[0].instances
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
        from_predicted = from_predicted if hasattr(from_predicted, "score") else None
        new_instance = Instance(skeleton=self.skeleton, from_predicted=from_predicted)

        # the rect that's currently visibile in the window view
        in_view_rect = self.player.view.mapToScene(self.player.view.rect()).boundingRect()

        # go through each node in skeleton
        for node in self.skeleton.node_names:
            # if we're copying from a skeleton that has this node
            if copy_instance is not None and node in copy_instance and not copy_instance[node].isnan():
                # just copy x, y, and visible
                # we don't want to copy a PredictedPoint or score attribute
                new_instance[node] = Point(
                                        x=copy_instance[node].x,
                                        y=copy_instance[node].y,
                                        visible=copy_instance[node].visible)
            else:
                # pick random points within currently zoomed view
                x = in_view_rect.x() + (in_view_rect.width() * 0.1) \
                    + (np.random.rand() * in_view_rect.width() * 0.8)
                y = in_view_rect.y() + (in_view_rect.height() * 0.1) \
                    + (np.random.rand() * in_view_rect.height() * 0.8)
                # mark the node as not "visible" if we're copying from a predicted instance without this node
                is_visible = copy_instance is None or not hasattr(copy_instance, "score")
                # set point for node
                new_instance[node] = Point(x=x, y=y, visible=is_visible)

        # If we're copying a predicted instance or from another frame, copy the track
        if hasattr(copy_instance, "score") or from_prev_frame:
            new_instance.track = copy_instance.track

        # Add the instance
        self.labels.add_instance(self.labeled_frame, new_instance)
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

        self.labels.remove_instance(self.labeled_frame, selected_inst)
        self.changestack_push("delete instance")

        self.plotFrame()
        self.updateSeekbarMarks()

    def deleteSelectedInstanceTrack(self):
        selected_inst = self.player.view.getSelectionInstance()
        if selected_inst is None: return

        # to do: range of frames?

        track = selected_inst.track
        self.labels.remove_instance(self.labeled_frame, selected_inst)

        if track is not None:
            # remove any instance on this track
            for lf in self.labels.find(self.video):
                for inst in filter(lambda inst: inst.track == track, lf.instances):
                    self.labels.remove_instance(lf, inst)

        self.changestack_push("delete track")

        self.plotFrame()
        self.updateTrackMenu()
        self.updateSeekbarMarks()

    def addTrack(self):
        track_numbers_used = [int(track.name)
                                for track in self.labels.tracks
                                if track.name.isnumeric()]
        next_number = max(track_numbers_used, default=0) + 1
        new_track = Track(spawned_on=self.player.frame_idx, name=next_number)

        self.changestack_start_atomic("add track")
        self.labels.add_track(self.video, new_track)
        self.changestack_push("new track")
        self.setInstanceTrack(new_track)
        self.changestack_push("set track")
        self.changestack_end_atomic()

        # update track menu and seekbar
        self.updateTrackMenu()
        self.updateSeekbarMarks()

    def setInstanceTrack(self, new_track):
        vis_idx = self.player.view.getSelection()
        if vis_idx is None: return

        selected_instance = self.labeled_frame.instances_to_show[vis_idx]
        idx = self.labeled_frame.index(selected_instance)

        old_track = selected_instance.track

        # When setting track for an instance that doesn't already have a track set,
        # just set for selected instance.
        if old_track is None:
            # Move anything already in the new track out of it
            new_track_instances = self.labels.find_track_instances(
                    video = self.video,
                    track = new_track,
                    frame_range = (self.player.frame_idx, self.player.frame_idx+1))
            for instance in new_track_instances:
                instance.track = None
            # Move selected instance into new track
            self.labels.track_set_instance(self.labeled_frame, selected_instance, new_track)

        # When the instance does already have a track, then we want to update
        # the track for a range of frames.
        else:

            # Determine range that should be affected
            if self.player.seekbar.hasSelection():
                # If range is selected in seekbar, use that
                frame_range = tuple(*self.player.seekbar.getSelection())
            else:
                # Otherwise, range is current to last frame
                frame_range = (self.player.frame_idx, self.video.frames)

            # Do the swap
            self.labels.track_swap(self.video, new_track, old_track, frame_range)

        self.changestack_push("swap tracks")

        # Update visuals
        self.plotFrame()
        self.updateSeekbarMarks()

        # Make sure the originally selected instance is still selected
        self.player.view.selectInstance(idx)

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

        # Swap tracks for current and subsequent frames when we do have tracks
        old_track, new_track = instance_0.track, instance_1.track
        if old_track is not None and new_track is not None:
            frame_range = (self.player.frame_idx, self.video.frames)
            self.labels.track_swap(self.video, new_track, old_track, frame_range)

        self.changestack_push("swap tracks")

        # Update visuals
        self.plotFrame()
        self.updateSeekbarMarks()

    def newProject(self):
        window = MainWindow()
        window.showMaximized()

    def openProject(self, first_open=False):
        filters = ["JSON labels (*.json *.json.zip)", "HDF5 dataset (*.h5 *.hdf5)", "Matlab dataset (*.mat)", "DeepLabCut csv (*.csv)"]
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

            if filename.endswith((".json", ".json.zip")):
                compress = filename.endswith(".zip")
                Labels.save_json(labels = self.labels, filename = filename,
                                    compress = compress)
            elif filename.endswith(".h5"):
                Labels.save_hdf5(labels = self.labels, filename = filename)

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

        filters = ["JSON labels (*.json)", "Compressed JSON (*.zip)", "HDF5 dataset (*.h5)"]
        filename, selected_filter = QFileDialog.getSaveFileName(self,
                                        caption="Save As...",
                                        dir=default_name,
                                        filter=";;".join(filters))

        if len(filename) == 0: return

        if filename.endswith((".json", ".zip")):
            compress = filename.endswith(".zip")
            Labels.save_json(labels = self.labels, filename = filename, compress = compress)
            self.filename = filename
            # Mark savepoint in change stack
            self.changestack_savepoint()
            # Redraw. Not sure why, but sometimes we need to do this.
            self.plotFrame()
        elif filename.endswith(".h5"):
            Labels.save_hdf5(labels = self.labels, filename = filename)
            self.filename = filename
            # Mark savepoint in change stack
            self.changestack_savepoint()
            # Redraw. Not sure why, but sometimes we need to do this.
            self.plotFrame()
        else:
            QMessageBox(text=f"File not saved. Try saving as json.").exec_()

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

    def nextVideo(self):
        new_idx = self.video_idx+1
        new_idx = 0 if new_idx >= len(self.labels.videos) else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def previousVideo(self):
        new_idx = self.video_idx-1
        new_idx = len(self.labels.videos)-1 if new_idx < 0 else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def gotoFrame(self):
        frame_number, okay = QtWidgets.QInputDialog.getInt(
                                self,
                                "Go To Frame...",
                                "Frame Number:",
                                self.player.frame_idx+1,
                                1, self.video.frames)
        if okay:
            self.plotFrame(frame_number-1)

    def markFrame(self):
        self.mark_idx = self.player.frame_idx

    def goMarkedFrame(self):
        self.plotFrame(self.mark_idx)

    def exportLabeledClip(self):
        from sleap.io.visuals import save_labeled_video
        if self.player.seekbar.hasSelection():

            fps, okay = QtWidgets.QInputDialog.getInt(
                                    self,
                                    "Frames per second",
                                    "Frames per second:",
                                    getattr(self.video, "fps", 30),
                                    1, 300)
            if not okay: return

            filename, _ = QFileDialog.getSaveFileName(self, caption="Save Video As...", dir=self.filename + ".avi", filter="AVI Video (*.avi)")

            if len(filename) == 0: return

            save_labeled_video(
                    labels=self.labels,
                    video=self.video,
                    filename=filename,
                    frames=list(range(*self.player.seekbar.getSelection())),
                    fps=fps,
                    gui_progress=True
                    )

    def exportLabeledFrames(self):
        filename, _ = QFileDialog.getSaveFileName(self, caption="Save Labeled Frames As...", dir=self.filename)
        if len(filename) == 0: return
        Labels.save_json(self.labels, filename, save_frame_data=True)

    def previousLabeledFrameIndex(self):
        cur_idx = self.player.frame_idx
        frames = self.labels.frames(self.video, from_frame_idx=cur_idx, reverse=True)

        try:
            next_idx = next(frames).frame_idx
        except:
            return

    def previousLabeledFrame(self):
        prev_idx = self.previousLabeledFrameIndex()
        if prev_idx is not None:
            self.plotFrame(prev_idx)

    def nextLabeledFrame(self):
        cur_idx = self.player.frame_idx

        frames = self.labels.frames(self.video, from_frame_idx=cur_idx)

        try:
            next_idx = next(frames).frame_idx
        except:
            return

        self.plotFrame(next_idx)

    def nextUserLabeledFrame(self):
        cur_idx = self.player.frame_idx

        frames = self.labels.frames(self.video, from_frame_idx=cur_idx)
        # Filter to frames with user instances
        frames = filter(lambda lf: lf.has_user_instances, frames)

        try:
            next_idx = next(frames).frame_idx
        except:
            return

        self.plotFrame(next_idx)

    def nextSuggestedFrame(self, seek_direction=1):
        next_video, next_frame = self.labels.get_next_suggestion(self.video, self.player.frame_idx, seek_direction)
        if next_video is not None:
            self.gotoVideoAndFrame(next_video, next_frame)
        if next_frame is not None:
            selection_idx = self.labels.get_suggestions().index((next_video, next_frame))
            self.suggestionsTable.selectRow(selection_idx)

    def nextTrackFrame(self):
        cur_idx = self.player.frame_idx
        video_tracks = {inst.track for lf in self.labels.find(self.video) for inst in lf if inst.track is not None}
        next_idx = min([track.spawned_on for track in video_tracks if track.spawned_on > cur_idx], default=-1)
        if next_idx > -1:
            self.plotFrame(next_idx)

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
        self.overlays["trails"].show = not self.overlays["trails"].show
        self._menu_actions["show trails"].setChecked(self.overlays["trails"].show)
        self.plotFrame()

    def setTrailLength(self, trail_length):
        self.overlays["trails"].trail_length = trail_length
        self._menu_check_single(self.trailLengthMenu, trail_length)

        if self.video is not None: self.plotFrame()

    def setPalette(self, palette):
        self._color_manager.set_palette(palette)
        self._menu_check_single(self.paletteMenu, palette)
        if self.video is not None: self.plotFrame()
        self.updateSeekbarMarks()

    def _menu_check_single(self, menu, item_text):
        for menu_item in menu.children():
            if menu_item.text() == str(item_text):
                menu_item.setChecked(True)
            else:
                menu_item.setChecked(False)

    def toggleColorPredicted(self):
        self.overlays["instance"].color_predicted = not self.overlays["instance"].color_predicted
        self._menu_actions["color predicted"].setChecked(self.overlays["instance"].color_predicted)
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

    def newFrame(self, player, frame_idx, selected_idx):
        """Called each time a new frame is drawn."""

        # Store the current LabeledFrame (or make new, empty object)
        self.labeled_frame = self.labels.find(self.video, frame_idx, return_new=True)[0]

        # Show instances, etc, for this frame
        for overlay in self.overlays.values():
            overlay.add_to_scene(self.video, frame_idx)

        # Select instance if there was already selection
        if selected_idx > -1:
            player.view.selectInstance(selected_idx)

        # Update related displays
        self.updateStatusMessage()
        self.update_data_views()

        # Trigger event after the overlays have been added
        player.view.updatedViewer.emit()

    def updateStatusMessage(self, message = None):
        if message is None:
            message = f"Frame: {self.player.frame_idx+1}/{len(self.video)}"
            if self.player.seekbar.hasSelection():
                start, end = self.player.seekbar.getSelection()
                message += f" (selection: {start}-{end})"
            message += f"    Labeled Frames: "
            if self.video is not None:
                message += f"{len(self.labels.get_video_user_labeled_frames(self.video))}"
                if len(self.labels.videos) > 1:
                    message += " in video, "
            if len(self.labels.videos) > 1:
                message += f"{len(self.labels.user_labeled_frames)} in project"

        self.statusBar().showMessage(message)

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

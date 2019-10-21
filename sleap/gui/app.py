"""
Main GUI application for labeling, active learning, and proofreading.
"""

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt, QEvent

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
from PySide2.QtWidgets import QLabel, QPushButton, QComboBox
from PySide2.QtWidgets import QMessageBox

import re
import operator
import os
import sys

from pkg_resources import Requirement, resource_filename
from pathlib import PurePath

from typing import Callable, Dict, Iterator, Optional

import numpy as np

from sleap.skeleton import Skeleton
from sleap.instance import Instance, PredictedInstance, Point, Track
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.info.summary import StatisticSeries
from sleap.gui.video import QtVideoPlayer
from sleap.gui.dataviews import (
    VideosTable,
    SkeletonNodesTable,
    SkeletonEdgesTable,
    LabeledFrameTable,
    SkeletonNodeModel,
    SuggestionsTable,
)
from sleap.gui.importvideos import ImportVideos
from sleap.gui.filedialog import FileDialog
from sleap.gui.formbuilder import YamlFormWidget
from sleap.gui.merge import MergeDialog
from sleap.gui.shortcuts import Shortcuts, ShortcutDialog
from sleap.gui.suggestions import VideoFrameSuggestions

from sleap.gui.overlays.tracks import (
    TrackColorManager,
    TrackTrailOverlay,
    TrackListOverlay,
)
from sleap.gui.overlays.instance import InstanceOverlay
from sleap.gui.overlays.anchors import NegativeAnchorOverlay

OPEN_IN_NEW = True


class MainWindow(QMainWindow):
    """The SLEAP GUI application.

    Each project (`Labels` dataset) that you have loaded in the GUI will
    have its own `MainWindow` object.

    Attributes:
        labels: The :class:`Labels` dataset. If None, a new, empty project
            (i.e., :class:`Labels` object) will be created.
        skeleton: The active :class:`Skeleton` for the project in the gui
        video: The active :class:`Video` in view in the gui
    """

    labels: Labels
    skeleton: Skeleton
    video: Video

    def __init__(self, labels_path: Optional[str] = None, *args, **kwargs):
        """Initialize the app.

        Args:
            labels_path: Path to saved :class:`Labels` dataset.

        Returns:
            None.
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        self.labels = Labels()
        self.skeleton = Skeleton()
        self.labeled_frame = None
        self.video = None
        self.video_idx = None
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
        self._initialize_gui()

        if labels_path:
            self.loadProject(labels_path)

    def event(self, e: QEvent) -> bool:
        """Custom event handler.

        We use this to ignore events that would clear status bar.

        Args:
            e: The event.
        Returns:
            True if we ignore event, otherwise returns whatever the usual
            event handler would return.
        """
        if e.type() == QEvent.StatusTip:
            if e.tip() == "":
                return True
        return super().event(e)

    def changestack_push(self, change: bool = None):
        """Adds to stack of changes made by user."""
        # Currently the change doesn't store any data, and we're only using this
        # to determine if there are unsaved changes. Eventually we could use this
        # to support undo/redo.
        self._change_stack.append(change)

    def changestack_savepoint(self):
        """Marks that project was just saved."""
        self.changestack_push("SAVE")

    def changestack_clear(self):
        """Clears stack of changes."""
        self._change_stack = list()

    def changestack_start_atomic(self):
        """Marks that we want to track a set of changes as a single change."""
        self.changestack_push("ATOMIC_START")

    def changestack_end_atomic(self):
        """Marks that we want finished the set of changes to track together."""
        self.changestack_push("ATOMIC_END")

    def changestack_has_changes(self) -> bool:
        """Returns whether there are any unsaved changes."""
        # True iff there are no unsaved changed
        if len(self._change_stack) == 0:
            return False
        if self._change_stack[-1] == "SAVE":
            return False
        return True

    @property
    def filename(self):
        """Returns filename for current project."""
        return self._filename

    @filename.setter
    def filename(self, x):
        """Sets filename for current project. Doesn't load file."""
        self._filename = x
        if x is not None:
            self.setWindowTitle(x)

    def _initialize_gui(self):
        """Creates menus, dock windows, starts timers to update gui state."""

        self._create_video_player()
        self.statusBar()
        self.load_overlays()
        self._create_menus()
        self._create_dock_windows()

        # Create timer to update state of gui at regular intervals
        self.update_gui_timer = QtCore.QTimer()
        self.update_gui_timer.timeout.connect(self._update_gui_state)
        self.update_gui_timer.start(0.1)

    def _create_video_player(self):
        """Creates and connects :class:`QtVideoPlayer` for gui."""
        self.player = QtVideoPlayer(color_manager=self._color_manager)
        self.player.changedPlot.connect(self._after_plot_update)
        self.player.changedData.connect(
            lambda inst: self.changestack_push("viewer change")
        )
        self.player.view.instanceDoubleClicked.connect(self.doubleClickInstance)
        self.player.seekbar.selectionChanged.connect(lambda: self.updateStatusMessage())
        self.setCentralWidget(self.player)

    def _create_menus(self):
        """Creates main application menus."""
        shortcuts = Shortcuts()

        def _menu_item(menu, key: str, name: str, action: Callable):
            menu_item = menu.addAction(name, action, shortcuts[key])
            self._menu_actions[key] = menu_item

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        _menu_item(fileMenu, "new", "New Project", self.newProject)
        _menu_item(fileMenu, "open", "Open Project...", self.openProject)
        _menu_item(
            fileMenu, "import predictions", "Import Labels...", self.importPredictions
        )

        fileMenu.addSeparator()
        _menu_item(fileMenu, "add videos", "Add Videos...", self.addVideo)

        fileMenu.addSeparator()
        _menu_item(fileMenu, "save", "Save", self.saveProject)
        _menu_item(fileMenu, "save as", "Save As...", self.saveProjectAs)

        fileMenu.addSeparator()
        _menu_item(fileMenu, "close", "Quit", self.close)

        ### Go Menu ###

        goMenu = self.menuBar().addMenu("Go")

        _menu_item(
            goMenu, "goto next labeled", "Next Labeled Frame", self.nextLabeledFrame
        )
        _menu_item(
            goMenu,
            "goto prev labeled",
            "Previous Labeled Frame",
            self.previousLabeledFrame,
        )
        _menu_item(
            goMenu,
            "goto next user",
            "Next User Labeled Frame",
            self.nextUserLabeledFrame,
        )
        _menu_item(
            goMenu, "goto next suggestion", "Next Suggestion", self.nextSuggestedFrame
        )
        _menu_item(
            goMenu,
            "goto prev suggestion",
            "Previous Suggestion",
            lambda: self.nextSuggestedFrame(-1),
        )
        _menu_item(
            goMenu,
            "goto next track spawn",
            "Next Track Spawn Frame",
            self.nextTrackFrame,
        )

        goMenu.addSeparator()

        _menu_item(goMenu, "next video", "Next Video", self.nextVideo)
        _menu_item(goMenu, "prev video", "Previous Video", self.previousVideo)

        goMenu.addSeparator()

        _menu_item(goMenu, "goto frame", "Go to Frame...", self.gotoFrame)

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

        viewMenu.addSeparator()
        _menu_item(
            viewMenu,
            "color predicted",
            "Color Predicted Instances",
            self.toggleColorPredicted,
        )

        self.paletteMenu = viewMenu.addMenu("Color Palette")
        for palette_name in self._color_manager.palette_names:
            menu_item = self.paletteMenu.addAction(
                f"{palette_name}", lambda x=palette_name: self.setPalette(x)
            )
            menu_item.setCheckable(True)
        self.setPalette("standard")

        viewMenu.addSeparator()

        self.seekbarHeaderMenu = viewMenu.addMenu("Seekbar Header")
        headers = (
            "None",
            "Point Displacement (sum)",
            "Point Displacement (max)",
            "Instance Score (sum)",
            "Instance Score (min)",
            "Point Score (sum)",
            "Point Score (min)",
            "Number of predicted points",
        )
        for header in headers:
            menu_item = self.seekbarHeaderMenu.addAction(
                header, lambda x=header: self.setSeekbarHeader(x)
            )
            menu_item.setCheckable(True)
        self.setSeekbarHeader("None")

        viewMenu.addSeparator()

        _menu_item(viewMenu, "show labels", "Show Node Names", self.toggleLabels)
        _menu_item(viewMenu, "show edges", "Show Edges", self.toggleEdges)
        _menu_item(viewMenu, "show trails", "Show Trails", self.toggleTrails)

        self.trailLengthMenu = viewMenu.addMenu("Trail Length")
        for length_option in (4, 10, 20):
            menu_item = self.trailLengthMenu.addAction(
                f"{length_option}", lambda x=length_option: self.setTrailLength(x)
            )
            menu_item.setCheckable(True)

        viewMenu.addSeparator()

        _menu_item(viewMenu, "fit", "Fit Instances to View", self.toggleAutoZoom)

        viewMenu.addSeparator()

        # set menu checkmarks
        self._menu_actions["show labels"].setCheckable(True)
        self._menu_actions["show labels"].setChecked(self._show_labels)
        self._menu_actions["show edges"].setCheckable(True)
        self._menu_actions["show edges"].setChecked(self._show_edges)
        self._menu_actions["show trails"].setCheckable(True)
        self._menu_actions["show trails"].setChecked(self.overlays["trails"].show)
        self._menu_actions["color predicted"].setCheckable(True)
        self._menu_actions["color predicted"].setChecked(
            self.overlays["instance"].color_predicted
        )
        self._menu_actions["fit"].setCheckable(True)

        ### Label Menu ###

        labelMenu = self.menuBar().addMenu("Labels")
        _menu_item(labelMenu, "add instance", "Add Instance", self.newInstance)
        _menu_item(
            labelMenu, "delete instance", "Delete Instance", self.deleteSelectedInstance
        )

        labelMenu.addSeparator()

        self.track_menu = labelMenu.addMenu("Set Instance Track")
        _menu_item(
            labelMenu, "transpose", "Transpose Instance Tracks", self.transposeInstance
        )
        _menu_item(
            labelMenu,
            "delete track",
            "Delete Instance and Track",
            self.deleteSelectedInstanceTrack,
        )

        labelMenu.addSeparator()

        _menu_item(
            labelMenu,
            "select next",
            "Select Next Instance",
            self.player.view.nextSelection,
        )
        _menu_item(
            labelMenu,
            "clear selection",
            "Clear Selection",
            self.player.view.clearSelection,
        )

        labelMenu.addSeparator()

        ### Predict Menu ###

        predictionMenu = self.menuBar().addMenu("Predict")

        _menu_item(
            predictionMenu,
            "active learning",
            "Run Active Learning...",
            lambda: self.showLearningDialog("learning"),
        )
        _menu_item(
            predictionMenu,
            "inference",
            "Run Inference...",
            lambda: self.showLearningDialog("inference"),
        )
        _menu_item(
            predictionMenu,
            "learning expert",
            "Expert Controls...",
            lambda: self.showLearningDialog("expert"),
        )

        predictionMenu.addSeparator()
        _menu_item(
            predictionMenu,
            "negative sample",
            "Mark Negative Training Sample...",
            self.markNegativeAnchor,
        )
        _menu_item(
            predictionMenu,
            "clear negative samples",
            "Clear Current Frame Negative Samples",
            self.clearFrameNegativeAnchors,
        )

        predictionMenu.addSeparator()
        _menu_item(
            predictionMenu,
            "visualize models",
            "Visualize Model Outputs...",
            self.visualizeOutputs,
        )

        predictionMenu.addSeparator()
        _menu_item(
            predictionMenu,
            "remove predictions",
            "Delete All Predictions...",
            self.deletePredictions,
        )
        _menu_item(
            predictionMenu,
            "remove clip predictions",
            "Delete Predictions from Clip...",
            self.deleteClipPredictions,
        )
        _menu_item(
            predictionMenu,
            "remove area predictions",
            "Delete Predictions from Area...",
            self.deleteAreaPredictions,
        )
        _menu_item(
            predictionMenu,
            "remove score predictions",
            "Delete Predictions with Low Score...",
            self.deleteLowScorePredictions,
        )
        _menu_item(
            predictionMenu,
            "remove frame limit predictions",
            "Delete Predictions beyond Frame Limit...",
            self.deleteFrameLimitPredictions,
        )

        predictionMenu.addSeparator()
        _menu_item(
            predictionMenu,
            "export frames",
            "Export Training Package...",
            self.exportLabeledFrames,
        )
        _menu_item(
            predictionMenu,
            "export clip",
            "Export Labeled Clip...",
            self.exportLabeledClip,
        )

        ############

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Keyboard Reference", self.openKeyRef)

    def _create_dock_windows(self):
        """Create dock windows and connects them to gui."""

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
            self.viewMenu.addAction(dock.toggleViewAction())
            if tab_with is not None:
                self.tabifyDockWidget(tab_with, dock)
            return layout

        ####### Videos #######
        videos_layout = _make_dock("Videos")
        self.videosTable = VideosTable()
        videos_layout.addWidget(self.videosTable)
        hb = QHBoxLayout()
        btn = QPushButton("Show video")
        btn.clicked.connect(self.activateSelectedVideo)
        hb.addWidget(btn)
        self._buttons["show video"] = btn
        btn = QPushButton("Add videos")
        btn.clicked.connect(self.addVideo)
        hb.addWidget(btn)
        btn = QPushButton("Remove video")
        btn.clicked.connect(self.removeVideo)
        hb.addWidget(btn)
        self._buttons["remove video"] = btn
        hbw = QWidget()
        hbw.setLayout(hb)
        videos_layout.addWidget(hbw)

        self.videosTable.doubleClicked.connect(self.activateSelectedVideo)

        ####### Skeleton #######
        skeleton_layout = _make_dock(
            "Skeleton", tab_with=videos_layout.parent().parent()
        )

        gb = QGroupBox("Nodes")
        vb = QVBoxLayout()
        self.skeletonNodesTable = SkeletonNodesTable(self.skeleton)
        vb.addWidget(self.skeletonNodesTable)
        hb = QHBoxLayout()
        btn = QPushButton("New node")
        btn.clicked.connect(self.newNode)
        hb.addWidget(btn)
        btn = QPushButton("Delete node")
        btn.clicked.connect(self.deleteNode)
        hb.addWidget(btn)
        self._buttons["delete node"] = btn
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        def _update_edge_src():
            self.skeletonEdgesDst.model().skeleton = self.skeleton

        gb = QGroupBox("Edges")
        vb = QVBoxLayout()
        self.skeletonEdgesTable = SkeletonEdgesTable(self.skeleton)
        vb.addWidget(self.skeletonEdgesTable)
        hb = QHBoxLayout()
        self.skeletonEdgesSrc = QComboBox()
        self.skeletonEdgesSrc.setEditable(False)
        self.skeletonEdgesSrc.currentIndexChanged.connect(_update_edge_src)
        self.skeletonEdgesSrc.setModel(SkeletonNodeModel(self.skeleton))
        hb.addWidget(self.skeletonEdgesSrc)
        hb.addWidget(QLabel("to"))
        self.skeletonEdgesDst = QComboBox()
        self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        self.skeletonEdgesDst.setModel(
            SkeletonNodeModel(
                self.skeleton, lambda: self.skeletonEdgesSrc.currentText()
            )
        )
        btn = QPushButton("Add edge")
        btn.clicked.connect(self.newEdge)
        hb.addWidget(btn)
        self._buttons["add edge"] = btn
        btn = QPushButton("Delete edge")
        btn.clicked.connect(self.deleteEdge)
        hb.addWidget(btn)
        self._buttons["delete edge"] = btn
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        hb = QHBoxLayout()
        btn = QPushButton("Load Skeleton")
        btn.clicked.connect(self.openSkeleton)
        hb.addWidget(btn)
        btn = QPushButton("Save Skeleton")
        btn.clicked.connect(self.saveSkeleton)
        hb.addWidget(btn)
        hbw = QWidget()
        hbw.setLayout(hb)
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
        btn.clicked.connect(lambda x: self.newInstance())
        hb.addWidget(btn)
        btn = QPushButton("Delete instance")
        btn.clicked.connect(self.deleteSelectedInstance)
        hb.addWidget(btn)
        self._buttons["delete instance"] = btn
        hbw = QWidget()
        hbw.setLayout(hb)
        instances_layout.addWidget(hbw)

        def update_instance_table_selection():
            inst_selected = self.player.view.getSelectionInstance()

            if not inst_selected:
                return

            idx = -1
            if inst_selected in self.labeled_frame.instances_to_show:
                idx = self.labeled_frame.instances_to_show.index(inst_selected)

            table_row_idx = self.instancesTable.model().createIndex(idx, 0)
            self.instancesTable.setCurrentIndex(table_row_idx)

        self.instancesTable.selectionChangedSignal.connect(
            lambda inst: self.player.view.selectInstance(inst, signal=False)
        )
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
        btn.clicked.connect(lambda: self.nextSuggestedFrame(-1))
        hb.addWidget(btn)
        self.suggested_count_label = QLabel()
        hb.addWidget(self.suggested_count_label)
        btn = QPushButton("Next")
        btn.clicked.connect(lambda: self.nextSuggestedFrame())
        hb.addWidget(btn)
        hbw = QWidget()
        hbw.setLayout(hb)
        suggestions_layout.addWidget(hbw)

        suggestions_yaml = resource_filename(
            Requirement.parse("sleap"), "sleap/config/suggestions.yaml"
        )
        form_wid = YamlFormWidget(
            yaml_file=suggestions_yaml, title="Generate Suggestions"
        )
        form_wid.mainAction.connect(self.generateSuggestions)
        suggestions_layout.addWidget(form_wid)

        self.suggestionsTable.doubleClicked.connect(
            lambda table_idx: self.gotoVideoAndFrame(
                *self.labels.get_suggestions()[table_idx.row()]
            )
        )

    def load_overlays(self):
        """Load all standard video overlays."""
        self.overlays["track_labels"] = TrackListOverlay(self.labels, self.player)
        self.overlays["negative"] = NegativeAnchorOverlay(self.labels, self.player)
        self.overlays["trails"] = TrackTrailOverlay(self.labels, self.player)
        self.overlays["instance"] = InstanceOverlay(self.labels, self.player)

    def _update_gui_state(self):
        """Enable/disable gui items based on current state."""
        has_selected_instance = self.player.view.getSelection() is not None
        has_unsaved_changes = self.changestack_has_changes()
        has_multiple_videos = self.labels is not None and len(self.labels.videos) > 1
        has_labeled_frames = self.labels is not None and any(
            (lf.video == self.video for lf in self.labels)
        )
        has_suggestions = self.labels is not None and (len(self.labels.suggestions) > 0)
        has_tracks = self.labels is not None and (len(self.labels.tracks) > 0)
        has_multiple_instances = (
            self.labeled_frame is not None and len(self.labeled_frame.instances) > 1
        )
        # todo: exclude predicted instances from count
        has_nodes_selected = (
            self.skeletonEdgesSrc.currentIndex() > -1
            and self.skeletonEdgesDst.currentIndex() > -1
        )
        control_key_down = QApplication.queryKeyboardModifiers() == Qt.ControlModifier

        # Update menus

        self.track_menu.setEnabled(has_selected_instance)
        self._menu_actions["clear selection"].setEnabled(has_selected_instance)
        self._menu_actions["delete instance"].setEnabled(has_selected_instance)

        self._menu_actions["transpose"].setEnabled(has_multiple_instances)

        self._menu_actions["save"].setEnabled(has_unsaved_changes)

        self._menu_actions["next video"].setEnabled(has_multiple_videos)
        self._menu_actions["prev video"].setEnabled(has_multiple_videos)

        self._menu_actions["goto next labeled"].setEnabled(has_labeled_frames)
        self._menu_actions["goto prev labeled"].setEnabled(has_labeled_frames)

        self._menu_actions["goto next suggestion"].setEnabled(has_suggestions)
        self._menu_actions["goto prev suggestion"].setEnabled(has_suggestions)

        self._menu_actions["goto next track spawn"].setEnabled(has_tracks)

        # Update buttons
        self._buttons["add edge"].setEnabled(has_nodes_selected)
        self._buttons["delete edge"].setEnabled(
            self.skeletonEdgesTable.currentIndex().isValid()
        )
        self._buttons["delete node"].setEnabled(
            self.skeletonNodesTable.currentIndex().isValid()
        )
        self._buttons["show video"].setEnabled(
            self.videosTable.currentIndex().isValid()
        )
        self._buttons["remove video"].setEnabled(
            self.videosTable.currentIndex().isValid()
        )
        self._buttons["delete instance"].setEnabled(
            self.instancesTable.currentIndex().isValid()
        )

        # Update overlays
        self.overlays["track_labels"].visible = (
            control_key_down and has_selected_instance
        )

    def _update_data_views(self, *update):
        """Update data used by data view table models.

        Args:
            Accepts names of what data to update as unnamed string arguments:
                "video", "skeleton", "labels", "frame", "suggestions"
            If no arguments are given, then everything is updated.
        Returns:
             None.
        """
        update = update or ("video", "skeleton", "labels", "frame", "suggestions")

        if len(self.skeleton.nodes) == 0 and len(self.labels.skeletons):
            self.skeleton = self.labels.skeletons[0]

        if "video" in update:
            self.videosTable.model().items = self.labels.videos

        if "skeleton" in update:
            self.skeletonNodesTable.model().skeleton = self.skeleton
            self.skeletonEdgesTable.model().skeleton = self.skeleton
            self.skeletonEdgesSrc.model().skeleton = self.skeleton
            self.skeletonEdgesDst.model().skeleton = self.skeleton

        if "labels" in update:
            self.instancesTable.model().labels = self.labels
            self.instancesTable.model().color_manager = self._color_manager

        if "frame" in update:
            self.instancesTable.model().labeled_frame = self.labeled_frame

        if "suggestions" in update:
            self.suggestionsTable.model().labels = self.labels

            # update count of suggested frames w/ labeled instances
            suggestion_status_text = ""
            suggestion_list = self.labels.get_suggestions()
            if len(suggestion_list):
                suggestion_label_counts = [
                    self.labels.instance_count(video, frame_idx)
                    for (video, frame_idx) in suggestion_list
                ]
                labeled_count = len(suggestion_list) - suggestion_label_counts.count(0)
                suggestion_status_text = (
                    f"{labeled_count}/{len(suggestion_list)} labeled"
                )
            self.suggested_count_label.setText(suggestion_status_text)

    def plotFrame(self, *args, **kwargs):
        """Plots (or replots) current frame."""
        if self.video is None:
            return

        self.player.plot(*args, **kwargs)
        self.player.showLabels(self._show_labels)
        self.player.showEdges(self._show_edges)
        if self._auto_zoom:
            self.player.zoomToFit()

    def _after_plot_update(self, player, frame_idx, selected_inst):
        """Called each time a new frame is drawn."""

        # Store the current LabeledFrame (or make new, empty object)
        self.labeled_frame = self.labels.find(self.video, frame_idx, return_new=True)[0]

        # Show instances, etc, for this frame
        for overlay in self.overlays.values():
            overlay.add_to_scene(self.video, frame_idx)

        # Select instance if there was already selection
        if selected_inst is not None:
            player.view.selectInstance(selected_inst)

        # Update related displays
        self.updateStatusMessage()
        self._update_data_views("frame")

        # Trigger event after the overlays have been added
        player.view.updatedViewer.emit()

    def updateStatusMessage(self, message: Optional[str] = None):
        """Updates status bar."""
        if message is None:
            message = f"Frame: {self.player.frame_idx+1}/{len(self.video)}"
            if self.player.seekbar.hasSelection():
                start, end = self.player.seekbar.getSelection()
                message += f" (selection: {start}-{end})"

            if len(self.labels.videos) > 1:
                message += f" of video {self.labels.videos.index(self.video)}"

            message += f"    Labeled Frames: "
            if self.video is not None:
                message += (
                    f"{len(self.labels.get_video_user_labeled_frames(self.video))}"
                )
                if len(self.labels.videos) > 1:
                    message += " in video, "
            if len(self.labels.videos) > 1:
                message += f"{len(self.labels.user_labeled_frames)} in project"

        self.statusBar().showMessage(message)

    def loadProject(self, filename: Optional[str] = None):
        """
        Loads given labels file into GUI.

        Args:
            filename: The path to the saved labels dataset. If None,
                then don't do anything.

        Returns:
            None:
        """
        show_msg = False

        if len(filename) == 0:
            return

        gui_video_callback = Labels.make_gui_video_callback(
            search_paths=[os.path.dirname(filename)]
        )

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

        self.labels = labels
        self.filename = filename

        if has_loaded:
            self.changestack_clear()
            self._color_manager.labels = self.labels
            self._color_manager.set_palette(self._color_palette)

            self.load_overlays()

            self.setTrailLength(self.overlays["trails"].trail_length)

            if show_msg:
                msgBox = QMessageBox(
                    text=f"Imported {len(self.labels)} labeled frames."
                )
                msgBox.exec_()

            if len(self.labels.skeletons):
                # TODO: add support for multiple skeletons
                self.skeleton = self.labels.skeletons[0]

            # Update UI tables
            self._update_data_views()

            # Load first video
            if len(self.labels.videos):
                self.loadVideo(self.labels.videos[0], 0)

            # Update track menu options
            self.updateTrackMenu()

    def updateTrackMenu(self):
        """Updates track menu options."""
        self.track_menu.clear()
        for track in self.labels.tracks:
            key_command = ""
            if self.labels.tracks.index(track) < 9:
                key_command = Qt.CTRL + Qt.Key_0 + self.labels.tracks.index(track) + 1
            self.track_menu.addAction(
                f"{track.name}", lambda x=track: self.setInstanceTrack(x), key_command
            )
        self.track_menu.addAction("New Track", self.addTrack, Qt.CTRL + Qt.Key_0)

    def activateSelectedVideo(self, x):
        """Activates video selected in table."""
        # Get selected video
        idx = self.videosTable.currentIndex()
        if not idx.isValid():
            return
        self.loadVideo(self.labels.videos[idx.row()], idx.row())

    def addVideo(self, filename: Optional[str] = None):
        """Shows gui for adding video to project.

        Args:
            filename: If given, then we just load this video. If not given,
                then we show dialog for importing videos.

        Returns:
            None.
        """
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
            self.loadVideo(video, len(self.labels.videos) - 1)

        # Update data model/view
        self._update_data_views("video")

    def removeVideo(self):
        """Removes video (selected in table) from project."""
        # Get selected video
        idx = self.videosTable.currentIndex()
        if not idx.isValid():
            return
        video = self.labels.videos[idx.row()]

        # Count labeled frames for this video
        n = len(self.labels.find(video))

        # Warn if there are labels that will be deleted
        if n > 0:
            response = QMessageBox.critical(
                self,
                "Removing video with labels",
                f"{n} labeled frames in this video will be deleted, "
                "are you sure you want to remove this video?",
                QMessageBox.Yes,
                QMessageBox.No,
            )
            if response == QMessageBox.No:
                return

        # Remove video
        self.labels.remove_video(video)
        self.changestack_push("remove video")

        # Update data model
        self._update_data_views()

        # Update view if this was the current video
        if self.video == video:
            if len(self.labels.videos) == 0:
                self.player.reset()
                # TODO: update statusbar
            else:
                new_idx = min(idx.row(), len(self.labels.videos) - 1)
                self.loadVideo(self.labels.videos[new_idx], new_idx)

    def loadVideo(self, video: Video, video_idx: int = None):
        """Activates video in gui."""

        # Update current video instance
        self.video = video
        self.video_idx = (
            video_idx if video_idx is not None else self.labels.videos.index(video)
        )

        # Load video in player widget
        self.player.load_video(self.video)

        # Annotate labeled frames on seekbar
        self.updateSeekbarMarks()

        # Jump to last labeled frame
        last_label = self.labels.find_last(self.video)
        if last_label is not None:
            self.plotFrame(last_label.frame_idx)

    def openSkeleton(self):
        """Shows gui for loading saved skeleton into project."""
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = FileDialog.open(
            self, dir=None, caption="Open skeleton...", filter=";;".join(filters)
        )

        if len(filename) == 0:
            return

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
        self._update_data_views()

    def saveSkeleton(self):
        """Shows gui for saving skeleton from project."""
        default_name = "skeleton.json"
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = FileDialog.save(
            self, caption="Save As...", dir=default_name, filter=";;".join(filters)
        )

        if len(filename) == 0:
            return

        if filename.endswith(".json"):
            self.skeleton.save_json(filename)
        elif filename.endswith((".h5", ".hdf5")):
            self.skeleton.save_hdf5(filename)

    def newNode(self):
        """Adds new node to skeleton."""
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
        self._update_data_views()

        self.plotFrame()

    def deleteNode(self):
        """Removes (currently selected) node from skeleton."""
        # Get selected node
        idx = self.skeletonNodesTable.currentIndex()
        if not idx.isValid():
            return
        node = self.skeleton.nodes[idx.row()]

        # Remove
        self.skeleton.delete_node(node)
        self.changestack_push("delete node")

        # Update data model
        self._update_data_views()

        # Replot instances
        self.plotFrame()

    def updateEdges(self):
        """Called when edges in skeleton have been changed."""
        self._update_data_views()
        self.plotFrame()

    def newEdge(self):
        """Adds new edge to skeleton."""
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
        self._update_data_views()

        self.plotFrame()

    def deleteEdge(self):
        """Removes (currently selected) edge from skeleton."""
        # TODO: Move this to unified data model

        # Get selected edge
        idx = self.skeletonEdgesTable.currentIndex()
        if not idx.isValid():
            return
        edge = self.skeleton.edges[idx.row()]

        # Delete edge
        self.skeleton.delete_edge(source=edge[0], destination=edge[1])
        self.changestack_push("delete edge")

        # Update data model
        self._update_data_views()

        self.plotFrame()

    def updateSeekbarMarks(self):
        """Updates marks on seekbar."""
        self.player.seekbar.setTracksFromLabels(self.labels, self.video)

    def setSeekbarHeader(self, graph_name):
        """Updates graph shown in seekbar header."""
        data_obj = StatisticSeries(self.labels)
        header_functions = {
            "Point Displacement (sum)": data_obj.get_point_displacement_series,
            "Point Displacement (max)": data_obj.get_point_displacement_series,
            "Instance Score (sum)": data_obj.get_instance_score_series,
            "Instance Score (min)": data_obj.get_instance_score_series,
            "Point Score (sum)": data_obj.get_point_score_series,
            "Point Score (min)": data_obj.get_point_score_series,
            "Number of predicted points": data_obj.get_point_count_series,
        }

        self._menu_check_single(self.seekbarHeaderMenu, graph_name)

        if graph_name == "None":
            self.player.seekbar.clearHeader()
        else:
            if graph_name in header_functions:
                kwargs = dict(video=self.video)
                reduction_name = re.search("\((sum|max|min)\)", graph_name)
                if reduction_name is not None:
                    kwargs["reduction"] = reduction_name.group(1)
                series = header_functions[graph_name](**kwargs)
                self.player.seekbar.setHeaderSeries(series)
            else:
                print(f"Could not find function for {header_functions}")

    def generateSuggestions(self, params: Dict):
        """Generates suggestions using given params dictionary."""
        new_suggestions = dict()
        for video in self.labels.videos:
            new_suggestions[video] = VideoFrameSuggestions.suggest(
                video=video, labels=self.labels, params=params
            )

        self.labels.set_suggestions(new_suggestions)

        self._update_data_views("suggestions")
        self.updateSeekbarMarks()

    def _frames_for_prediction(self):
        """Builds options for frames on which to run inference.

        Args:
            None.
        Returns:
            Dictionary, keys are names of options (e.g., "clip", "random"),
            values are {video: list of frame indices} dictionaries.
        """

        def remove_user_labeled(
            video, frames, user_labeled_frames=self.labels.user_labeled_frames
        ):
            if len(frames) == 0:
                return frames
            video_user_labeled_frame_idxs = [
                lf.frame_idx for lf in user_labeled_frames if lf.video == video
            ]
            return list(set(frames) - set(video_user_labeled_frame_idxs))

        selection = dict()
        selection["frame"] = {self.video: [self.player.frame_idx]}
        selection["clip"] = {
            self.video: list(range(*self.player.seekbar.getSelection()))
        }
        selection["video"] = {self.video: list(range(self.video.num_frames))}

        selection["suggestions"] = {
            video: remove_user_labeled(video, self.labels.get_video_suggestions(video))
            for video in self.labels.videos
        }

        selection["random"] = {
            video: remove_user_labeled(video, VideoFrameSuggestions.random(video=video))
            for video in self.labels.videos
        }

        return selection

    def showLearningDialog(self, mode: str):
        """Helper function to show active learning dialog in given mode.

        Args:
            mode: A string representing mode for dialog, which could be:
            * "active"
            * "inference"
            * "expert"

        Returns:
            None.
        """
        from sleap.gui.active import ActiveLearningDialog

        if "inference" in self.overlays:
            QMessageBox(
                text="In order to use this function you must first quit and "
                "re-open SLEAP to release resources used by visualizing "
                "model outputs."
            ).exec_()
            return

        if self._child_windows.get(mode, None) is None:
            self._child_windows[mode] = ActiveLearningDialog(
                self.filename, self.labels, mode
            )
            self._child_windows[mode].learningFinished.connect(self.learningFinished)

        self._child_windows[mode].frame_selection = self._frames_for_prediction()
        self._child_windows[mode].open()

    def learningFinished(self):
        """Called when active learning (or inference) finishes."""
        # we ran active learning so update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self._update_data_views()
        self.changestack_push("new predictions")

    def visualizeOutputs(self):
        """Gui for adding overlay with live visualization of predictions."""
        filters = ["Model (*.json)", "HDF5 output (*.h5 *.hdf5)"]

        # Default to opening from models directory from project
        models_dir = None
        if self.filename is not None:
            models_dir = os.path.join(os.path.dirname(self.filename), "models/")

        # Show dialog
        filename, selected_filter = FileDialog.open(
            self,
            dir=models_dir,
            caption="Import model outputs...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return

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
                self.player.changedPlot.connect(
                    lambda parent, idx: confmap_overlay.add_to_scene(None, idx)
                )

            if show_pafs:
                from sleap.gui.overlays.pafs import PafOverlay

                paf_overlay = PafOverlay.from_h5(filename, player=self.player)
                self.player.changedPlot.connect(
                    lambda parent, idx: paf_overlay.add_to_scene(None, idx)
                )

        self.plotFrame()

    def deletePredictions(self):
        """Deletes all predicted instances in project."""

        predicted_instances = [
            (lf, inst)
            for lf in self.labels
            for inst in lf
            if type(inst) == PredictedInstance
        ]

        self._delete_confirm(predicted_instances)

    def deleteClipPredictions(self):
        """Deletes all instances within selected range of video frames."""

        predicted_instances = [
            (lf, inst)
            for lf in self.labels.find(
                self.video, frame_idx=range(*self.player.seekbar.getSelection())
            )
            for inst in lf
            if type(inst) == PredictedInstance
        ]

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
                predicted_instances = list(
                    filter(lambda x: x[1].track == track, predicted_instances)
                )

        self._delete_confirm(predicted_instances)

    def deleteAreaPredictions(self):
        """Gui for deleting instances within some rect on frame images."""

        # Callback to delete after area has been selected
        def delete_area_callback(x0, y0, x1, y1):

            self.updateStatusMessage()

            # Make sure there was an area selected
            if x0 == x1 or y0 == y1:
                return

            min_corner = (x0, y0)
            max_corner = (x1, y1)

            def is_bounded(inst):
                points_array = inst.points_array
                valid_points = points_array[~np.isnan(points_array).any(axis=1)]

                is_gt_min = np.all(valid_points >= min_corner)
                is_lt_max = np.all(valid_points <= max_corner)
                return is_gt_min and is_lt_max

            # Find all instances contained in selected area
            predicted_instances = [
                (lf, inst)
                for lf in self.labels.find(self.video)
                for inst in lf
                if type(inst) == PredictedInstance and is_bounded(inst)
            ]

            self._delete_confirm(predicted_instances)

        # Prompt the user to select area
        self.updateStatusMessage(
            f"Please select the area from which to remove instances. This will be applied to all frames."
        )
        self.player.onAreaSelection(delete_area_callback)

    def deleteLowScorePredictions(self):
        """Gui for deleting instances below some score threshold."""
        score_thresh, okay = QtWidgets.QInputDialog.getDouble(
            self, "Delete Instances with Low Score...", "Score Below:", 1, 0, 100
        )
        if okay:
            # Find all instances contained in selected area
            predicted_instances = [
                (lf, inst)
                for lf in self.labels.find(self.video)
                for inst in lf
                if type(inst) == PredictedInstance and inst.score < score_thresh
            ]

            self._delete_confirm(predicted_instances)

    def deleteFrameLimitPredictions(self):
        """Gui for deleting instances beyond some number in each frame."""
        count_thresh, okay = QtWidgets.QInputDialog.getInt(
            self,
            "Limit Instances in Frame...",
            "Maximum instances in a frame:",
            3,
            1,
            100,
        )
        if okay:
            predicted_instances = []
            # Find all instances contained in selected area
            for lf in self.labels.find(self.video):
                if len(lf.instances) > count_thresh:
                    # Get all but the count_thresh many instances with the highest score
                    extra_instances = sorted(
                        lf.instances, key=operator.attrgetter("score")
                    )[:-count_thresh]
                    predicted_instances.extend([(lf, inst) for inst in extra_instances])

            self._delete_confirm(predicted_instances)

    def _delete_confirm(self, lf_inst_list):
        """Helper function to confirm before deleting instances.

        Args:
            lf_inst_list: A list of (labeled frame, instance) tuples.
        """

        # Confirm that we want to delete
        resp = QMessageBox.critical(
            self,
            "Removing predicted instances",
            f"There are {len(lf_inst_list)} predicted instances that would be deleted. "
            "Are you sure you want to delete these?",
            QMessageBox.Yes,
            QMessageBox.No,
        )

        if resp == QMessageBox.No:
            return

        # Delete the instances
        for lf, inst in lf_inst_list:
            self.labels.remove_instance(lf, inst)

        # Update visuals
        self.plotFrame()
        self.updateSeekbarMarks()
        self.changestack_push("removed predictions")

    def markNegativeAnchor(self):
        """Allows user to add negative training sample anchor."""

        def click_callback(x, y):
            self.updateStatusMessage()
            self.labels.add_negative_anchor(self.video, self.player.frame_idx, (x, y))
            self.changestack_push("add negative anchors")
            self.plotFrame()

        # Prompt the user to select area
        self.updateStatusMessage(f"Please click where you want a negative sample...")
        self.player.onPointSelection(click_callback)

    def clearFrameNegativeAnchors(self):
        """Removes negative training sample anchors on current frame."""
        self.labels.remove_negative_anchors(self.video, self.player.frame_idx)
        self.changestack_push("remove negative anchors")
        self.plotFrame()

    def importPredictions(self):
        """Starts gui for importing another dataset into currently one."""
        filters = ["HDF5 dataset (*.h5 *.hdf5)", "JSON labels (*.json *.json.zip)"]
        filenames, selected_filter = FileDialog.openMultiple(
            self, dir=None, caption="Import labeled data...", filter=";;".join(filters)
        )

        if len(filenames) == 0:
            return

        for filename in filenames:
            gui_video_callback = Labels.make_gui_video_callback(
                search_paths=[os.path.dirname(filename)]
            )

            new_labels = Labels.load_file(filename, video_callback=gui_video_callback)

            # Merging data is handled by MergeDialog
            MergeDialog(base_labels=self.labels, new_labels=new_labels).exec_()

        # update display/ui
        self.plotFrame()
        self.updateSeekbarMarks()
        self._update_data_views()
        self.changestack_push("new predictions")

    def doubleClickInstance(self, instance: Instance):
        """
        Handles when the user has double-clicked an instance.

        If prediction, then copy to new user-instance.
        If already user instance, then add any missing nodes (in case
        skeleton has been changed after instance was created).

        Args:
            instance: The :class:`Instance` that was double-clicked.
        """
        # When a predicted instance is double-clicked, add a new instance
        if hasattr(instance, "score"):
            self.newInstance(copy_instance=instance)

        # When a regular instance is double-clicked, add any missing points
        else:
            # the rect that's currently visibile in the window view
            in_view_rect = self.player.view.mapToScene(
                self.player.view.rect()
            ).boundingRect()

            for node in self.skeleton.nodes:
                if node not in instance.nodes or instance[node].isnan():
                    # pick random points within currently zoomed view
                    x = (
                        in_view_rect.x()
                        + (in_view_rect.width() * 0.1)
                        + (np.random.rand() * in_view_rect.width() * 0.8)
                    )
                    y = (
                        in_view_rect.y()
                        + (in_view_rect.height() * 0.1)
                        + (np.random.rand() * in_view_rect.height() * 0.8)
                    )
                    # set point for node
                    instance[node] = Point(x=x, y=y, visible=False)

            self.plotFrame()

    def newInstance(self, copy_instance: Optional[Instance] = None):
        """
        Creates a new instance, copying node coordinates as appropriate.

        Args:
            copy_instance: The :class:`Instance` (or
                :class:`PredictedInstance`) which we want to copy.
        """
        if self.labeled_frame is None:
            return

        # FIXME: filter by skeleton type

        from_predicted = copy_instance
        from_prev_frame = False

        if copy_instance is None:
            selected_idx = self.player.view.getSelection()
            if selected_idx is not None:
                # If the user has selected an instance, copy that one.
                copy_instance = self.labeled_frame.instances[selected_idx]
                from_predicted = copy_instance

        if copy_instance is None:
            unused_predictions = self.labeled_frame.unused_predictions
            if len(unused_predictions):
                # If there are predicted instances that don't correspond to an instance
                # in this frame, use the first predicted instance without matching instance.
                copy_instance = unused_predictions[0]
                from_predicted = copy_instance

        if copy_instance is None:
            # Otherwise, if there are instances in previous frames,
            # copy the points from one of those instances.
            prev_idx = self.previousLabeledFrameIndex()

            if prev_idx is not None:
                prev_instances = self.labels.find(
                    self.video, prev_idx, return_new=True
                )[0].instances
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

        # Now create the new instance
        new_instance = Instance(skeleton=self.skeleton, from_predicted=from_predicted)

        # Get the rect that's currently visibile in the window view
        in_view_rect = self.player.view.mapToScene(
            self.player.view.rect()
        ).boundingRect()

        # go through each node in skeleton
        for node in self.skeleton.node_names:
            # if we're copying from a skeleton that has this node
            if (
                copy_instance is not None
                and node in copy_instance
                and not copy_instance[node].isnan()
            ):
                # just copy x, y, and visible
                # we don't want to copy a PredictedPoint or score attribute
                new_instance[node] = Point(
                    x=copy_instance[node].x,
                    y=copy_instance[node].y,
                    visible=copy_instance[node].visible,
                )
            else:
                # pick random points within currently zoomed view
                x = (
                    in_view_rect.x()
                    + (in_view_rect.width() * 0.1)
                    + (np.random.rand() * in_view_rect.width() * 0.8)
                )
                y = (
                    in_view_rect.y()
                    + (in_view_rect.height() * 0.1)
                    + (np.random.rand() * in_view_rect.height() * 0.8)
                )
                # mark the node as not "visible" if we're copying from a predicted instance without this node
                is_visible = copy_instance is None or not hasattr(
                    copy_instance, "score"
                )
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
        """Deletes currently selected instance."""
        selected_inst = self.player.view.getSelectionInstance()
        if selected_inst is None:
            return

        self.labels.remove_instance(self.labeled_frame, selected_inst)
        self.changestack_push("delete instance")

        self.plotFrame()
        self.updateSeekbarMarks()

    def deleteSelectedInstanceTrack(self):
        """Deletes all instances from track of currently selected instance."""
        selected_inst = self.player.view.getSelectionInstance()
        if selected_inst is None:
            return

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
        """Creates new track and moves selected instance into this track."""
        track_numbers_used = [
            int(track.name) for track in self.labels.tracks if track.name.isnumeric()
        ]
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

    def setInstanceTrack(self, new_track: "Track"):
        """Sets track for selected instance."""
        vis_idx = self.player.view.getSelection()
        if vis_idx is None:
            return

        selected_instance = self.labeled_frame.instances_to_show[vis_idx]
        idx = self.labeled_frame.index(selected_instance)

        old_track = selected_instance.track

        # When setting track for an instance that doesn't already have a track set,
        # just set for selected instance.
        if old_track is None:
            # Move anything already in the new track out of it
            new_track_instances = self.labels.find_track_instances(
                video=self.video,
                track=new_track,
                frame_range=(self.player.frame_idx, self.player.frame_idx + 1),
            )
            for instance in new_track_instances:
                instance.track = None
            # Move selected instance into new track
            self.labels.track_set_instance(
                self.labeled_frame, selected_instance, new_track
            )

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
        """Transposes tracks for two instances.

        If there are only two instances, then this swaps tracks.
        Otherwise, it allows user to select the instances for which we want
        to swap tracks.
        """
        # We're currently identifying instances by numeric index, so it's
        # impossible to (e.g.) have a single instance which we identify
        # as the second instance in some other frame.

        # For the present, we can only "transpose" if there are multiple instances.
        if len(self.labeled_frame.instances) < 2:
            return
        # If there are just two instances, transpose them.
        if len(self.labeled_frame.instances) == 2:
            self._transpose_instances((0, 1))
        # If there are more than two, then we need the user to select the instances.
        else:
            self.player.onSequenceSelect(
                seq_len=2,
                on_success=self._transpose_instances,
                on_each=self._transpose_message,
                on_failure=lambda x: self.updateStatusMessage(),
            )

    def _transpose_message(self, instance_ids: list):
        word = "next" if len(instance_ids) else "first"
        self.updateStatusMessage(f"Please select the {word} instance to transpose...")

    def _transpose_instances(self, instance_ids: list):
        if len(instance_ids) != 2:
            return

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
        """Create a new project in a new window."""
        window = MainWindow()
        window.showMaximized()

    def openProject(self, first_open: bool = False):
        """
        Allows use to select and then open a saved project.

        Args:
            first_open: Whether this is the first window opened. If True,
                then the new project is loaded into the current window
                rather than a new application window.

        Returns:
            None.
        """
        filters = [
            "HDF5 dataset (*.h5 *.hdf5)",
            "JSON labels (*.json *.json.zip)",
            "Matlab dataset (*.mat)",
            "DeepLabCut csv (*.csv)",
        ]

        filename, selected_filter = FileDialog.open(
            self, dir=None, caption="Import labeled data...", filter=";;".join(filters)
        )

        if len(filename) == 0:
            return

        if OPEN_IN_NEW and not first_open:
            new_window = MainWindow()
            new_window.showMaximized()
            new_window.loadProject(filename)
        else:
            self.loadProject(filename)

    def saveProject(self):
        """Show gui to save project (or save as if not yet saved)."""
        if self.filename is not None:
            self._trySave(self.filename)
        else:
            # No filename (must be new project), so treat as "Save as"
            self.saveProjectAs()

    def saveProjectAs(self):
        """Show gui to save project as a new file."""
        default_name = self.filename if self.filename is not None else "untitled"
        p = PurePath(default_name)
        default_name = str(p.with_name(f"{p.stem} copy{p.suffix}"))

        filters = [
            "HDF5 dataset (*.h5)",
            "JSON labels (*.json)",
            "Compressed JSON (*.zip)",
        ]
        filename, selected_filter = FileDialog.save(
            self, caption="Save As...", dir=default_name, filter=";;".join(filters)
        )

        if len(filename) == 0:
            return

        if self._trySave(filename):
            # If save was successful
            self.filename = filename

    def _trySave(self, filename):
        """Helper function which attempts save and handles errors."""
        success = False
        try:
            Labels.save_file(labels=self.labels, filename=filename)
            success = True
            # Mark savepoint in change stack
            self.changestack_savepoint()

        except Exception as e:
            message = f"An error occured when attempting to save:\n {e}\n\n"
            message += "Try saving your project with a different filename or in a different format."
            QtWidgets.QMessageBox(text=message).exec_()

        # Redraw. Not sure why, but sometimes we need to do this.
        self.plotFrame()

        return success

    def closeEvent(self, event):
        """Closes application window, prompting for saving as needed."""
        if not self.changestack_has_changes():
            # No unsaved changes, so accept event (close)
            event.accept()
        else:
            msgBox = QMessageBox()
            msgBox.setText("Do you want to save the changes to this project?")
            msgBox.setInformativeText("If you don't save, your changes will be lost.")
            msgBox.setStandardButtons(
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
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
        """Activates next video in project."""
        new_idx = self.video_idx + 1
        new_idx = 0 if new_idx >= len(self.labels.videos) else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def previousVideo(self):
        """Activates previous video in project."""
        new_idx = self.video_idx - 1
        new_idx = len(self.labels.videos) - 1 if new_idx < 0 else new_idx
        self.loadVideo(self.labels.videos[new_idx], new_idx)

    def gotoFrame(self):
        """Shows gui to go to frame by number."""
        frame_number, okay = QtWidgets.QInputDialog.getInt(
            self,
            "Go To Frame...",
            "Frame Number:",
            self.player.frame_idx + 1,
            1,
            self.video.frames,
        )
        if okay:
            self.plotFrame(frame_number - 1)

    def exportLabeledClip(self):
        """Shows gui for exporting clip with visual annotations."""
        from sleap.io.visuals import save_labeled_video

        if self.player.seekbar.hasSelection():

            fps, okay = QtWidgets.QInputDialog.getInt(
                self,
                "Frames per second",
                "Frames per second:",
                getattr(self.video, "fps", 30),
                1,
                300,
            )
            if not okay:
                return

            filename, _ = FileDialog.save(
                self,
                caption="Save Video As...",
                dir=self.filename + ".avi",
                filter="AVI Video (*.avi)",
            )

            if len(filename) == 0:
                return

            save_labeled_video(
                labels=self.labels,
                video=self.video,
                filename=filename,
                frames=list(range(*self.player.seekbar.getSelection())),
                fps=fps,
                gui_progress=True,
            )

    def exportLabeledFrames(self):
        """Gui for exporting the training dataset of labels/frame images."""
        filters = ["HDF5 dataset (*.h5)", "Compressed JSON dataset (*.json *.json.zip)"]
        filename, _ = FileDialog.save(
            self,
            caption="Save Labeled Frames As...",
            dir=self.filename + ".h5",
            filters=";;".join(filters),
        )
        if len(filename) == 0:
            return

        Labels.save_file(
            self.labels, filename, default_suffix="h5", save_frame_data=True
        )

    def _plot_if_next(self, frame_iterator: Iterator) -> bool:
        """Plots next frame (if there is one) from iterator.

        Arguments:
            frame_iterator: The iterator from which we'll try to get next
            :class:`LabeledFrame`.

        Returns:
            True if we went to next frame.
        """
        try:
            next_lf = next(frame_iterator)
        except StopIteration:
            return False

        self.plotFrame(next_lf.frame_idx)
        return True

    def previousLabeledFrameIndex(self):
        cur_idx = self.player.frame_idx
        frames = self.labels.frames(self.video, from_frame_idx=cur_idx, reverse=True)

        try:
            next_idx = next(frames).frame_idx
        except:
            return

        return next_idx

    def previousLabeledFrame(self):
        """Goes to labeled frame prior to current frame."""
        frames = self.labels.frames(
            self.video, from_frame_idx=self.player.frame_idx, reverse=True
        )
        self._plot_if_next(frames)

    def nextLabeledFrame(self):
        """Goes to labeled frame after current frame."""
        frames = self.labels.frames(self.video, from_frame_idx=self.player.frame_idx)
        self._plot_if_next(frames)

    def nextUserLabeledFrame(self):
        """Goes to next labeled frame with user instances."""
        frames = self.labels.frames(self.video, from_frame_idx=self.player.frame_idx)
        # Filter to frames with user instances
        frames = filter(lambda lf: lf.has_user_instances, frames)
        self._plot_if_next(frames)

    def nextSuggestedFrame(self, seek_direction=1):
        """Goes to next (or previous) suggested frame."""
        next_video, next_frame = self.labels.get_next_suggestion(
            self.video, self.player.frame_idx, seek_direction
        )
        if next_video is not None:
            self.gotoVideoAndFrame(next_video, next_frame)
        if next_frame is not None:
            selection_idx = self.labels.get_suggestions().index(
                (next_video, next_frame)
            )
            self.suggestionsTable.selectRow(selection_idx)

    def nextTrackFrame(self):
        """Goes to next frame on which a track starts."""
        cur_idx = self.player.frame_idx
        track_ranges = self.labels.get_track_occupany(self.video)
        next_idx = min(
            [
                track_range.start
                for track_range in track_ranges.values()
                if track_range.start is not None and track_range.start > cur_idx
            ],
            default=-1,
        )
        if next_idx > -1:
            self.plotFrame(next_idx)

    def gotoVideoAndFrame(self, video: Video, frame_idx: int):
        """Activates video and goes to frame."""
        if video != self.video:
            # switch to the other video
            self.loadVideo(video)
        self.plotFrame(frame_idx)

    def toggleLabels(self):
        """Toggles whether skeleton node labels are shown in video overlay."""
        self._show_labels = not self._show_labels
        self._menu_actions["show labels"].setChecked(self._show_labels)
        self.player.showLabels(self._show_labels)

    def toggleEdges(self):
        """Toggles whether skeleton edges are shown in video overlay."""
        self._show_edges = not self._show_edges
        self._menu_actions["show edges"].setChecked(self._show_edges)
        self.player.showEdges(self._show_edges)

    def toggleTrails(self):
        """Toggles whether track trails are shown in video overlay."""
        self.overlays["trails"].show = not self.overlays["trails"].show
        self._menu_actions["show trails"].setChecked(self.overlays["trails"].show)
        self.plotFrame()

    def setTrailLength(self, trail_length: int):
        """Sets length of track trails to show in video overlay."""
        self.overlays["trails"].trail_length = trail_length
        self._menu_check_single(self.trailLengthMenu, trail_length)

        if self.video is not None:
            self.plotFrame()

    def setPalette(self, palette: str):
        """Sets color palette used for track colors."""
        self._color_manager.set_palette(palette)
        self._menu_check_single(self.paletteMenu, palette)
        if self.video is not None:
            self.plotFrame()
        self.updateSeekbarMarks()

    def _menu_check_single(self, menu, item_text):
        """Helper method to select exactly one submenu item."""
        for menu_item in menu.children():
            if menu_item.text() == str(item_text):
                menu_item.setChecked(True)
            else:
                menu_item.setChecked(False)

    def toggleColorPredicted(self):
        """Toggles whether predicted instances are shown in track colors."""
        val = self.overlays["instance"].color_predicted
        self.overlays["instance"].color_predicted = not val
        self._menu_actions["color predicted"].setChecked(
            self.overlays["instance"].color_predicted
        )
        self.plotFrame()

    def toggleAutoZoom(self):
        """Toggles whether to zoom viewer to fit labeled instances."""
        self._auto_zoom = not self._auto_zoom
        self._menu_actions["fit"].setChecked(self._auto_zoom)
        if not self._auto_zoom:
            self.player.view.clearZoom()
        self.plotFrame()

    def openKeyRef(self):
        """Shows gui for viewing/modifying keyboard shortucts."""
        ShortcutDialog().exec_()


def main(*args, **kwargs):
    """Starts new instance of app."""
    app = QApplication([])
    app.setApplicationName("SLEAP Label")

    window = MainWindow(*args, **kwargs)
    window.showMaximized()

    if not kwargs.get("labels_path", None):
        window.openProject(first_open=True)

    app.exec_()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels_path", help="Path to labels file", type=str, default=None, nargs="?"
    )
    parser.add_argument(
        "--nonnative",
        help="Don't use native file dialogs",
        action="store_const",
        const=True,
        default=False,
    )

    args = parser.parse_args()

    if args.nonnative:
        os.environ["USE_NON_NATIVE_FILE"] = "1"

    main(labels_path=args.labels_path)

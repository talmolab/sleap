"""
Main GUI application for labeling, training/inference, and proofreading.
"""


import re
import os
import random

from typing import Callable, List, Optional, Tuple

from PySide2 import QtCore, QtGui
from PySide2.QtCore import Qt, QEvent

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
from PySide2.QtWidgets import QLabel, QPushButton, QComboBox
from PySide2.QtWidgets import QMessageBox

import sleap
from sleap.skeleton import Skeleton
from sleap.instance import Instance
from sleap.io.dataset import Labels
from sleap.info.summary import StatisticSeries
from sleap.gui.commands import CommandContext, UpdateTopic
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.gui.widgets.slider import set_slider_marks_from_labels
from sleap.gui.dataviews import (
    GenericTableView,
    VideosTableModel,
    SkeletonNodesTableModel,
    SkeletonEdgesTableModel,
    SuggestionsTableModel,
    LabeledFrameTableModel,
    SkeletonNodeModel,
)

from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import YamlFormWidget
from sleap.gui.shortcuts import Shortcuts
from sleap.gui.dialogs.shortcuts import ShortcutDialog
from sleap.gui.state import GuiState
from sleap.gui.overlays.tracks import TrackTrailOverlay, TrackListOverlay
from sleap.gui.color import ColorManager
from sleap.gui.overlays.instance import InstanceOverlay

from sleap.prefs import prefs


class MainWindow(QMainWindow):
    """The SLEAP GUI application.

    Each project (`Labels` dataset) that you have loaded in the GUI will
    have its own `MainWindow` object.

    Attributes:
        labels: The :class:`Labels` dataset. If None, a new, empty project
            (i.e., :class:`Labels` object) will be created.
        state: Object that holds GUI state, e.g., current video, frame,
            whether to show node labels, etc.
    """

    def __init__(self, labels_path: Optional[str] = None, *args, **kwargs):
        """Initialize the app.

        Args:
            labels_path: Path to saved :class:`Labels` dataset.

        Returns:
            None.
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        self.state = GuiState()
        self.labels = Labels()

        self.commands = CommandContext(
            state=self.state, app=self, update_callback=self.on_data_update
        )

        self._menu_actions = dict()
        self._buttons = dict()
        self._child_windows = dict()

        self.overlays = dict()

        self.state.connect("filename", self.setWindowTitle)

        self.state["skeleton"] = Skeleton()
        self.state["labeled_frame"] = None
        self.state["filename"] = None
        self.state["show labels"] = True
        self.state["show edges"] = True
        self.state["edge style"] = "Line"
        self.state["fit"] = False
        self.state["color predicted"] = prefs["color predicted"]

        self._initialize_gui()

        if labels_path:
            self.loadProjectFile(labels_path)

    def setWindowTitle(self, value):
        """Sets window title (if value is not None)."""
        if value is not None:
            super(MainWindow, self).setWindowTitle(
                f"{value} - SLEAP Label v{sleap.version.__version__}"
            )

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

    def closeEvent(self, event):
        """Closes application window, prompting for saving as needed."""
        if not self.state["has_changes"]:
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
                self.commands.saveProject()
                # accept event (closes window)
                event.accept()

    @property
    def labels(self):
        return self.state["labels"]

    @labels.setter
    def labels(self, value):
        self.state["labels"] = value

    def _initialize_gui(self):
        """Creates menus, dock windows, starts timers to update gui state."""

        self._create_color_manager()
        self._create_video_player()
        self.statusBar()

        self._create_menus()
        self._create_dock_windows()

        self.load_overlays()

        # Create timer to update state of gui at regular intervals
        self.update_gui_timer = QtCore.QTimer()
        self.update_gui_timer.timeout.connect(self._update_gui_state)
        self.update_gui_timer.start(1)

    def _create_video_player(self):
        """Creates and connects :class:`QtVideoPlayer` for gui."""
        self.player = QtVideoPlayer(
            color_manager=self.color_manager, state=self.state, context=self.commands
        )
        self.player.changedPlot.connect(self._after_plot_update)

        self.player.view.instanceDoubleClicked.connect(self.doubleClickInstance)
        self.player.seekbar.selectionChanged.connect(lambda: self.updateStatusMessage())
        self.setCentralWidget(self.player)

        def switch_frame(video):
            # Jump to last labeled frame
            last_label = self.labels.find_last(video)
            if last_label is not None:
                self.state["frame_idx"] = last_label.frame_idx
            else:
                self.state["frame_idx"] = 0

        self.state.connect(
            "video", callbacks=[switch_frame, lambda x: self.updateSeekbarMarks()]
        )

    def _create_color_manager(self):
        self.color_manager = ColorManager(self.labels)
        self.color_manager.palette = self.state.get("palette", default="standard")

    def _create_menus(self):
        """Creates main application menus."""
        shortcuts = Shortcuts()

        # add basic menu item
        def add_menu_item(menu, key: str, name: str, action: Callable):
            menu_item = menu.addAction(name, action, shortcuts[key])
            self._menu_actions[key] = menu_item

        # set menu checkmarks
        def connect_check(key):
            self._menu_actions[key].setCheckable(True)
            self._menu_actions[key].setChecked(self.state[key])
            self.state.connect(key, self._menu_actions[key].setChecked)

        # add checkable menu item connected to state variable
        def add_menu_check_item(menu, key: str, name: str):
            add_menu_item(menu, key, name, lambda: self.state.toggle(key))
            connect_check(key)

        # check and uncheck submenu items
        def _menu_check_single(menu, item_text):
            """Helper method to select exactly one submenu item."""
            for menu_item in menu.children():
                if menu_item.text() == str(item_text):
                    menu_item.setChecked(True)
                else:
                    menu_item.setChecked(False)

        # add submenu with checkable items
        def add_submenu_choices(menu, title, options, key):
            submenu = menu.addMenu(title)

            self.state.connect(key, lambda x: _menu_check_single(submenu, x))

            for option in options:
                submenu_item = submenu.addAction(
                    f"{option}", lambda x=option: self.state.set(key, x)
                )
                submenu_item.setCheckable(True)

            self.state.emit(key)

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        add_menu_item(fileMenu, "new", "New Project", self.commands.newProject)
        add_menu_item(fileMenu, "open", "Open Project...", self.commands.openProject)

        import_types_menu = fileMenu.addMenu("Import...")
        add_menu_item(
            import_types_menu,
            "import_coco",
            "COCO dataset...",
            self.commands.importCoco,
        )
        add_menu_item(
            import_types_menu,
            "import_dlc",
            "DeepLabCut dataset...",
            self.commands.importDLC,
        )
        add_menu_item(
            import_types_menu,
            "import_dpk",
            "DeepPoseKit dataset...",
            self.commands.importDPK,
        )
        add_menu_item(
            import_types_menu,
            "import_leap",
            "LEAP Matlab dataset...",
            self.commands.importLEAP,
        )
        add_menu_item(
            import_types_menu,
            "import_analysis",
            "SLEAP Analysis HDF5...",
            self.commands.importAnalysisFile,
        )

        add_menu_item(
            fileMenu,
            "import predictions",
            "Merge into Project...",
            self.commands.importPredictions,
        )

        fileMenu.addSeparator()
        add_menu_item(fileMenu, "add videos", "Add Videos...", self.commands.addVideo)
        add_menu_item(
            fileMenu, "replace videos", "Replace Videos...", self.commands.replaceVideo
        )

        fileMenu.addSeparator()
        add_menu_item(fileMenu, "save", "Save", self.commands.saveProject)
        add_menu_item(fileMenu, "save as", "Save As...", self.commands.saveProjectAs)

        fileMenu.addSeparator()
        add_menu_item(
            fileMenu,
            "export analysis",
            "Export Analysis HDF5...",
            self.commands.exportAnalysisFile,
        )

        fileMenu.addSeparator()
        add_menu_item(fileMenu, "close", "Quit", self.close)

        ### Go Menu ###

        goMenu = self.menuBar().addMenu("Go")

        add_menu_item(
            goMenu,
            "goto next labeled",
            "Next Labeled Frame",
            self.commands.nextLabeledFrame,
        )
        add_menu_item(
            goMenu,
            "goto prev labeled",
            "Previous Labeled Frame",
            self.commands.previousLabeledFrame,
        )
        add_menu_item(
            goMenu,
            "goto next user",
            "Next User Labeled Frame",
            self.commands.nextUserLabeledFrame,
        )
        add_menu_item(
            goMenu,
            "goto next suggestion",
            "Next Suggestion",
            self.commands.nextSuggestedFrame,
        )
        add_menu_item(
            goMenu,
            "goto prev suggestion",
            "Previous Suggestion",
            self.commands.prevSuggestedFrame,
        )
        add_menu_item(
            goMenu,
            "goto next track spawn",
            "Next Track Spawn Frame",
            self.commands.nextTrackFrame,
        )

        goMenu.addSeparator()

        def next_vid():
            self.state.increment_in_list("video", self.labels.videos)

        def prev_vid():
            self.state.increment_in_list("video", self.labels.videos, reverse=True)

        add_menu_item(goMenu, "next video", "Next Video", next_vid)
        add_menu_item(goMenu, "prev video", "Previous Video", prev_vid)

        goMenu.addSeparator()

        add_menu_item(goMenu, "goto frame", "Go to Frame...", self.commands.gotoFrame)
        add_menu_item(
            goMenu, "select to frame", "Select to Frame...", self.commands.selectToFrame
        )

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

        viewMenu.addSeparator()
        add_menu_check_item(viewMenu, "color predicted", "Color Predicted Instances")

        add_submenu_choices(
            menu=viewMenu,
            title="Color Palette",
            options=self.color_manager.palette_names,
            key="palette",
        )

        distinctly_color_options = ("instances", "nodes", "edges")

        add_submenu_choices(
            menu=viewMenu,
            title="Apply Distinct Colors To",
            options=distinctly_color_options,
            key="distinctly_color",
        )

        self.state["palette"] = prefs["palette"]
        self.state["distinctly_color"] = "instances"

        viewMenu.addSeparator()

        add_menu_check_item(viewMenu, "show labels", "Show Node Names")
        add_menu_check_item(viewMenu, "show edges", "Show Edges")

        add_submenu_choices(
            menu=viewMenu,
            title="Edge Style",
            options=("Line", "Wedge"),
            key="edge style",
        )

        add_submenu_choices(
            menu=viewMenu,
            title="Trail Length",
            options=(0, 10, 20, 50, 100, 200, 500),
            key="trail_length",
        )

        viewMenu.addSeparator()

        add_menu_check_item(viewMenu, "fit", "Fit Instances to View")

        viewMenu.addSeparator()

        seekbar_header_options = (
            "None",
            "Point Displacement (sum)",
            "Point Displacement (max)",
            "Primary Point Displacement (sum)",
            "Primary Point Displacement (max)",
            "Instance Score (sum)",
            "Instance Score (min)",
            "Point Score (sum)",
            "Point Score (min)",
            "Number of predicted points",
        )

        add_submenu_choices(
            menu=viewMenu,
            title="Seekbar Header",
            options=seekbar_header_options,
            key="seekbar_header",
        )

        self.state["seekbar_header"] = "None"
        self.state.connect("seekbar_header", self.setSeekbarHeader)

        viewMenu.addSeparator()

        ### Label Menu ###

        instance_adding_methods = dict(
            best="Best",
            template="Average Instance",
            force_directed="Force Directed",
            random="Random",
            prior_frame="Copy prior frame",
            prediction="Copy predictions",
        )

        def new_instance_menu_action():
            method_key = [
                key
                for (key, val) in instance_adding_methods.items()
                if val == self.state["instance_init_method"]
            ]
            if method_key:
                self.commands.newInstance(init_method=method_key[0])

        labelMenu = self.menuBar().addMenu("Labels")
        add_menu_item(
            labelMenu, "add instance", "Add Instance", new_instance_menu_action
        )

        add_submenu_choices(
            menu=labelMenu,
            title="Instance Placement Method",
            options=instance_adding_methods.values(),
            key="instance_init_method",
        )
        self.state["instance_init_method"] = instance_adding_methods["best"]

        add_menu_item(
            labelMenu,
            "delete instance",
            "Delete Instance",
            self.commands.deleteSelectedInstance,
        )

        labelMenu.addSeparator()

        self.track_menu = labelMenu.addMenu("Set Instance Track")
        add_menu_item(
            labelMenu,
            "transpose",
            "Transpose Instance Tracks",
            self.commands.transposeInstance,
        )
        add_menu_item(
            labelMenu,
            "delete track",
            "Delete Instance and Track",
            self.commands.deleteSelectedInstanceTrack,
        )

        labelMenu.addSeparator()

        add_menu_item(
            labelMenu,
            "custom delete",
            "Custom Instance Delete...",
            self.commands.deleteDialog,
        )
        labelMenu.addSeparator()

        add_menu_item(
            labelMenu,
            "select next",
            "Select Next Instance",
            lambda: self.state.increment_in_list(
                "instance", self.state["labeled_frame"].instances_to_show
            ),
        )
        add_menu_item(
            labelMenu,
            "clear selection",
            "Clear Selection",
            lambda: self.state.set("instance", None),
        )

        labelMenu.addSeparator()

        ### Predict Menu ###

        predictionMenu = self.menuBar().addMenu("Predict")

        add_menu_item(
            predictionMenu,
            "training",
            "Run Training...",
            lambda: self.showLearningDialog("training"),
        )
        add_menu_item(
            predictionMenu,
            "inference",
            "Run Inference...",
            lambda: self.showLearningDialog("inference"),
        )

        predictionMenu.addSeparator()
        add_menu_item(
            predictionMenu,
            "visualize models",
            "Visualize Model Outputs...",
            self.visualizeOutputs,
        )

        predictionMenu.addSeparator()

        add_menu_item(
            predictionMenu,
            "add instances from all frame predictions",
            "Add Instances from All Predictions on Current Frame",
            self.commands.addUserInstancesFromPredictions,
        )

        predictionMenu.addSeparator()

        add_menu_item(
            predictionMenu,
            "delete frame predictions",
            "Delete Predictions on Current Frame",
            self.commands.deleteFramePredictions,
        )
        add_menu_item(
            predictionMenu,
            "delete all predictions",
            "Delete All Predictions...",
            self.commands.deletePredictions,
        )
        add_menu_item(
            predictionMenu,
            "delete clip predictions",
            "Delete Predictions from Clip...",
            self.commands.deleteClipPredictions,
        )
        add_menu_item(
            predictionMenu,
            "delete area predictions",
            "Delete Predictions from Area...",
            self.commands.deleteAreaPredictions,
        )
        add_menu_item(
            predictionMenu,
            "delete score predictions",
            "Delete Predictions with Low Score...",
            self.commands.deleteLowScorePredictions,
        )
        add_menu_item(
            predictionMenu,
            "delete frame limit predictions",
            "Delete Predictions beyond Frame Limit...",
            self.commands.deleteFrameLimitPredictions,
        )

        predictionMenu.addSeparator()
        add_menu_item(
            predictionMenu,
            "export frames",
            "Export Training Package...",
            self.commands.exportDatasetWithImages,
        )
        add_menu_item(
            predictionMenu,
            "export clip",
            "Export Labeled Clip...",
            self.commands.exportLabeledClip,
        )

        ############

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Keyboard Shortcuts", self.openKeyRef)

    def process_events_then(self, action: Callable):
        """Decorates a function with a call to first process events."""

        def wrapped_function(*args):
            QApplication.instance().processEvents()
            action(*args)

        return wrapped_function

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

            key = f"hide {name.lower()} dock"
            if key in prefs and prefs[key]:
                dock.hide()

            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.viewMenu.addAction(dock.toggleViewAction())

            if tab_with is not None:
                self.tabifyDockWidget(tab_with, dock)

            return layout

        def _add_button(to, label, action, key=None):
            key = key or label.lower()
            btn = QPushButton(label)
            btn.clicked.connect(action)
            to.addWidget(btn)
            self._buttons[key] = btn
            return btn

        ####### Videos #######
        videos_layout = _make_dock("Videos")
        self.videosTable = GenericTableView(
            state=self.state,
            row_name="video",
            is_activatable=True,
            model=VideosTableModel(items=self.labels.videos, context=self.commands),
        )
        videos_layout.addWidget(self.videosTable)

        hb = QHBoxLayout()
        _add_button(hb, "Show Video", self.videosTable.activateSelected)
        _add_button(hb, "Add Videos", self.commands.addVideo)
        _add_button(hb, "Remove Video", self.commands.removeVideo)

        hbw = QWidget()
        hbw.setLayout(hb)
        videos_layout.addWidget(hbw)

        ####### Skeleton #######
        skeleton_layout = _make_dock(
            "Skeleton", tab_with=videos_layout.parent().parent()
        )

        gb = QGroupBox("Nodes")
        vb = QVBoxLayout()
        self.skeletonNodesTable = GenericTableView(
            state=self.state,
            row_name="node",
            model=SkeletonNodesTableModel(
                items=self.state["skeleton"], context=self.commands
            ),
        )

        vb.addWidget(self.skeletonNodesTable)
        hb = QHBoxLayout()
        _add_button(hb, "New Node", self.commands.newNode)
        _add_button(hb, "Delete Node", self.commands.deleteNode)

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        def _update_edge_src():
            self.skeletonEdgesDst.model().skeleton = self.state["skeleton"]

        gb = QGroupBox("Edges")
        vb = QVBoxLayout()
        self.skeletonEdgesTable = GenericTableView(
            state=self.state,
            row_name="edge",
            model=SkeletonEdgesTableModel(
                items=self.state["skeleton"], context=self.commands
            ),
        )

        vb.addWidget(self.skeletonEdgesTable)
        hb = QHBoxLayout()
        self.skeletonEdgesSrc = QComboBox()
        self.skeletonEdgesSrc.setEditable(False)
        self.skeletonEdgesSrc.currentIndexChanged.connect(_update_edge_src)
        self.skeletonEdgesSrc.setModel(SkeletonNodeModel(self.state["skeleton"]))
        hb.addWidget(self.skeletonEdgesSrc)
        hb.addWidget(QLabel("to"))
        self.skeletonEdgesDst = QComboBox()
        self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        self.skeletonEdgesDst.setModel(
            SkeletonNodeModel(
                self.state["skeleton"], lambda: self.skeletonEdgesSrc.currentText()
            )
        )

        def new_edge():
            src_node = self.skeletonEdgesSrc.currentText()
            dst_node = self.skeletonEdgesDst.currentText()
            self.commands.newEdge(src_node, dst_node)

        _add_button(hb, "Add Edge", new_edge)
        _add_button(hb, "Delete Edge", self.commands.deleteEdge)

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        gb.setLayout(vb)
        skeleton_layout.addWidget(gb)

        hb = QHBoxLayout()
        _add_button(hb, "Load Skeleton", self.commands.openSkeleton)
        _add_button(hb, "Save Skeleton", self.commands.saveSkeleton)

        hbw = QWidget()
        hbw.setLayout(hb)
        skeleton_layout.addWidget(hbw)

        ####### Instances #######
        instances_layout = _make_dock("Instances")
        self.instancesTable = GenericTableView(
            state=self.state,
            row_name="instance",
            name_prefix="",
            model=LabeledFrameTableModel(
                items=self.state["labeled_frame"], context=self.commands
            ),
        )
        instances_layout.addWidget(self.instancesTable)

        hb = QHBoxLayout()
        _add_button(hb, "New Instance", lambda x: self.commands.newInstance())
        _add_button(hb, "Delete Instance", self.commands.deleteSelectedInstance)

        hbw = QWidget()
        hbw.setLayout(hb)
        instances_layout.addWidget(hbw)

        ####### Suggestions #######
        suggestions_layout = _make_dock("Labeling Suggestions")
        self.suggestionsTable = GenericTableView(
            state=self.state,
            is_sortable=True,
            model=SuggestionsTableModel(
                items=self.labels.suggestions, context=self.commands
            ),
        )

        suggestions_layout.addWidget(self.suggestionsTable)

        hb = QHBoxLayout()

        _add_button(
            hb,
            "Prev",
            self.process_events_then(self.commands.prevSuggestedFrame),
            "goto previous suggestion",
        )

        self.suggested_count_label = QLabel()
        hb.addWidget(self.suggested_count_label)

        _add_button(
            hb,
            "Next",
            self.process_events_then(self.commands.nextSuggestedFrame),
            "goto next suggestion",
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        suggestions_layout.addWidget(hbw)

        self.suggestions_form_widget = YamlFormWidget.from_name(
            "suggestions", title="Generate Suggestions",
        )
        self.suggestions_form_widget.mainAction.connect(
            self.process_events_then(self.commands.generateSuggestions)
        )
        suggestions_layout.addWidget(self.suggestions_form_widget)

        def goto_suggestion(*args):
            selected_frame = self.suggestionsTable.getSelectedRowItem()
            self.commands.gotoVideoAndFrame(
                selected_frame.video, selected_frame.frame_idx
            )

        self.suggestionsTable.doubleClicked.connect(goto_suggestion)

        self.state.connect("suggestion_idx", self.suggestionsTable.selectRow)

    def load_overlays(self):
        """Load all standard video overlays."""
        self.overlays["track_labels"] = TrackListOverlay(self.labels, self.player)
        self.overlays["trails"] = TrackTrailOverlay(self.labels, self.player)
        self.overlays["instance"] = InstanceOverlay(
            self.labels, self.player, self.state
        )

        def overlay_state_connect(overlay, state_key, overlay_attribute=None):
            overlay_attribute = overlay_attribute or state_key
            self.state.connect(
                state_key,
                callbacks=[
                    lambda x: setattr(overlay, overlay_attribute, x),
                    self.plotFrame,
                ],
            )

        overlay_state_connect(self.overlays["trails"], "trail_length")

        overlay_state_connect(self.color_manager, "palette")
        overlay_state_connect(self.color_manager, "distinctly_color")
        overlay_state_connect(self.color_manager, "color predicted", "color_predicted")
        self.state.connect("palette", lambda x: self.updateSeekbarMarks())

        # update the skeleton tables since we may want to redraw colors
        for state_var in ("palette", "distinctly_color", "edge style"):
            self.state.connect(
                state_var, lambda x: self.on_data_update([UpdateTopic.skeleton])
            )

        # Set defaults
        self.state["trail_length"] = prefs["trail length"]

        # Emit signals for default that may have been set earlier
        self.state.emit("palette")
        self.state.emit("distinctly_color")
        self.state.emit("color predicted")

    def _update_gui_state(self):
        """Enable/disable gui items based on current state."""
        has_selected_instance = self.state["instance"] is not None
        has_selected_video = self.state["selected_video"] is not None
        has_selected_node = self.state["selected_node"] is not None
        has_selected_edge = self.state["selected_edge"] is not None

        has_frame_range = bool(self.state["has_frame_range"])
        has_unsaved_changes = bool(self.state["has_changes"])
        has_multiple_videos = self.labels is not None and len(self.labels.videos) > 1
        has_labeled_frames = self.labels is not None and any(
            (lf.video == self.state["video"] for lf in self.labels)
        )
        has_suggestions = self.labels is not None and bool(self.labels.suggestions)
        has_tracks = self.labels is not None and (len(self.labels.tracks) > 0)
        has_multiple_instances = (
            self.state["labeled_frame"] is not None
            and len(self.state["labeled_frame"].instances) > 1
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

        self._menu_actions["delete clip predictions"].setEnabled(has_frame_range)
        self._menu_actions["export clip"].setEnabled(has_frame_range)

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
        self._buttons["delete edge"].setEnabled(has_selected_edge)
        self._buttons["delete node"].setEnabled(has_selected_node)
        self._buttons["show video"].setEnabled(has_selected_video)
        self._buttons["remove video"].setEnabled(has_selected_video)
        self._buttons["delete instance"].setEnabled(has_selected_instance)

        # Update overlays
        self.overlays["track_labels"].visible = (
            control_key_down and has_selected_instance
        )

    def on_data_update(self, what: List[UpdateTopic]):
        def _has_topic(topic_list):
            if UpdateTopic.all in what:
                return True
            for topic in topic_list:
                if topic in what:
                    return True
            return False

        if _has_topic(
            [
                UpdateTopic.frame,
                UpdateTopic.skeleton,
                UpdateTopic.project_instances,
                UpdateTopic.tracks,
            ]
        ):
            self.plotFrame()

        if _has_topic(
            [
                UpdateTopic.frame,
                UpdateTopic.project_instances,
                UpdateTopic.tracks,
                UpdateTopic.suggestions,
            ]
        ):
            self.updateSeekbarMarks()

        if _has_topic(
            [UpdateTopic.frame, UpdateTopic.project_instances, UpdateTopic.tracks]
        ):
            self.updateTrackMenu()

        if _has_topic([UpdateTopic.video]):
            self.videosTable.model().items = self.labels.videos

        if _has_topic([UpdateTopic.skeleton]):
            self.skeletonNodesTable.model().items = self.state["skeleton"]
            self.skeletonEdgesTable.model().items = self.state["skeleton"]
            self.skeletonEdgesSrc.model().skeleton = self.state["skeleton"]
            self.skeletonEdgesDst.model().skeleton = self.state["skeleton"]

            if self.labels.skeletons:
                self.suggestions_form_widget.set_field_options(
                    "node", self.labels.skeletons[0].node_names
                )

        if _has_topic([UpdateTopic.project, UpdateTopic.on_frame]):
            self.instancesTable.model().items = self.state["labeled_frame"]

        if _has_topic([UpdateTopic.suggestions]):
            self.suggestionsTable.model().items = self.labels.suggestions

        if _has_topic([UpdateTopic.project_instances, UpdateTopic.suggestions]):
            # update count of suggested frames w/ labeled instances
            suggestion_status_text = ""
            suggestion_list = self.labels.get_suggestions()
            if suggestion_list:
                suggestion_label_counts = [
                    self.labels.instance_count(item.video, item.frame_idx)
                    for item in suggestion_list
                ]
                labeled_count = len(suggestion_list) - suggestion_label_counts.count(0)
                suggestion_status_text = (
                    f"{labeled_count}/{len(suggestion_list)} labeled"
                )
            self.suggested_count_label.setText(suggestion_status_text)

    def plotFrame(self, *args, **kwargs):
        """Plots (or replots) current frame."""
        if self.state["video"] is None:
            return

        self.player.plot()

    def _after_plot_update(self, player, frame_idx, selected_inst):
        """Called each time a new frame is drawn."""

        # Store the current LabeledFrame (or make new, empty object)
        self.state["labeled_frame"] = self.labels.find(
            self.state["video"], frame_idx, return_new=True
        )[0]

        # Show instances, etc, for this frame
        for overlay in self.overlays.values():
            overlay.add_to_scene(self.state["video"], frame_idx)

        # Select instance if there was already selection
        if selected_inst is not None:
            player.view.selectInstance(selected_inst)

        if self.state["fit"]:
            player.zoomToFit()

        # Update related displays
        self.updateStatusMessage()
        self.on_data_update([UpdateTopic.on_frame])

        # Trigger event after the overlays have been added
        player.view.updatedViewer.emit()

    def updateStatusMessage(self, message: Optional[str] = None):
        """Updates status bar."""

        current_video = self.state["video"]
        frame_idx = self.state["frame_idx"] or 0

        spacer = "        "

        if message is None:
            message = f"Frame: {frame_idx+1:,}/{len(current_video):,}"
            if self.player.seekbar.hasSelection():
                start, end = self.state["frame_range"]
                message += f" (selection: {start+1:,}-{end:,})"

            if len(self.labels.videos) > 1:
                message += f" of video {self.labels.videos.index(current_video)+1}"

            message += f"{spacer}Labeled Frames: "
            if current_video is not None:
                message += str(
                    self.labels.get_labeled_frame_count(current_video, "user")
                )

                if len(self.labels.videos) > 1:
                    message += " in video, "
            if len(self.labels.videos) > 1:
                project_user_frame_count = self.labels.get_labeled_frame_count(
                    filter="user"
                )
                message += f"{project_user_frame_count} in project"

            if current_video is not None:
                pred_frame_count = self.labels.get_labeled_frame_count(
                    current_video, "predicted"
                )
                if pred_frame_count:
                    message += f"{spacer}Predicted Frames: {pred_frame_count:,}"
                    message += (
                        f" ({pred_frame_count/current_video.num_frames*100:.2f}%)"
                    )
                    message += " in video"

        self.statusBar().showMessage(message)

    def loadProjectFile(self, filename: Optional[str] = None):
        """
        Loads given labels file into GUI.

        Args:
            filename: The path to the saved labels dataset. If None,
                then don't do anything.

        Returns:
            None:
        """
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
                labels = Labels.load_file(filename, video_search=gui_video_callback)
                has_loaded = True
            except ValueError as e:
                print(e)
                QMessageBox(text=f"Unable to load {filename}.").exec_()

        if has_loaded:
            self.loadLabelsObject(labels, filename)

    def loadLabelsObject(self, labels: Labels, filename: Optional[str] = None):
        """
        Loads a `Labels` object into the GUI, replacing any currently loaded.

        Args:
            labels: The `Labels` object to load.
            filename: The filename where this file is saved, if any.

        Returns:
            None.

        """
        self.state["labels"] = labels
        self.state["filename"] = filename

        self.commands.changestack_clear()
        self.color_manager.labels = self.labels
        self.color_manager.set_palette(self.state["palette"])

        self.load_overlays()

        if len(self.labels.skeletons):
            self.state["skeleton"] = self.labels.skeletons[0]

        # Load first video
        if len(self.labels.videos):
            self.state["video"] = self.labels.videos[0]

        self.on_data_update([UpdateTopic.project, UpdateTopic.all])

    def updateTrackMenu(self):
        """Updates track menu options."""
        self.track_menu.clear()
        for track in self.labels.tracks:
            key_command = ""
            if self.labels.tracks.index(track) < 9:
                key_command = Qt.CTRL + Qt.Key_0 + self.labels.tracks.index(track) + 1
            self.track_menu.addAction(
                f"{track.name}",
                lambda x=track: self.commands.setInstanceTrack(x),
                key_command,
            )
        self.track_menu.addAction(
            "New Track", self.commands.addTrack, Qt.CTRL + Qt.Key_0
        )

    def updateSeekbarMarks(self):
        """Updates marks on seekbar."""
        set_slider_marks_from_labels(
            self.player.seekbar, self.labels, self.state["video"], self.color_manager
        )

    def setSeekbarHeader(self, graph_name):
        """Updates graph shown in seekbar header."""
        data_obj = StatisticSeries(self.labels)
        header_functions = {
            "Point Displacement (sum)": data_obj.get_point_displacement_series,
            "Point Displacement (max)": data_obj.get_point_displacement_series,
            "Primary Point Displacement (sum)": data_obj.get_primary_point_displacement_series,
            "Primary Point Displacement (max)": data_obj.get_primary_point_displacement_series,
            "Instance Score (sum)": data_obj.get_instance_score_series,
            "Instance Score (min)": data_obj.get_instance_score_series,
            "Point Score (sum)": data_obj.get_point_score_series,
            "Point Score (min)": data_obj.get_point_score_series,
            "Number of predicted points": data_obj.get_point_count_series,
        }

        if graph_name == "None":
            self.player.seekbar.clearHeader()
        else:
            if graph_name in header_functions:
                kwargs = dict(video=self.state["video"])
                reduction_name = re.search("\((sum|max|min)\)", graph_name)
                if reduction_name is not None:
                    kwargs["reduction"] = reduction_name.group(1)
                series = header_functions[graph_name](**kwargs)
                self.player.seekbar.setHeaderSeries(series)
            else:
                print(f"Could not find function for {header_functions}")

    def _frames_for_prediction(self):
        """Builds options for frames on which to run inference.

        Args:
            None.
        Returns:
            Dictionary, keys are names of options (e.g., "clip", "random"),
            values are {video: list of frame indices} dictionaries.
        """

        user_labeled_frames = self.labels.user_labeled_frames

        def remove_user_labeled(video, frame_idxs):
            if len(frame_idxs) == 0:
                return frame_idxs
            video_user_labeled_frame_idxs = {
                lf.frame_idx for lf in user_labeled_frames if lf.video == video
            }
            return list(set(frame_idxs) - video_user_labeled_frame_idxs)

        current_video = self.state["video"]

        selection = dict()
        selection["frame"] = {current_video: [self.state["frame_idx"]]}

        # Use negative number in list for range (i.e., "0,-123" means "0-123")
        # The ranges should be [X, Y) like standard Python ranges
        def encode_range(a: int, b: int) -> Tuple[int, int]:
            return a, -b

        clip_range = self.state.get("frame_range", default=(0, 0))

        selection["clip"] = {current_video: encode_range(*clip_range)}
        selection["video"] = {current_video: encode_range(0, current_video.num_frames)}

        selection["suggestions"] = {
            video: remove_user_labeled(video, self.labels.get_video_suggestions(video))
            for video in self.labels.videos
        }

        selection["random"] = {
            video: remove_user_labeled(
                video, random.sample(range(video.frames), min(20, video.frames))
            )
            for video in self.labels.videos
        }

        if len(self.labels.videos) > 1:
            selection["random_video"] = {
                current_video: remove_user_labeled(
                    current_video,
                    random.sample(
                        range(current_video.frames), min(20, current_video.frames)
                    ),
                )
            }

        if user_labeled_frames:
            selection["user"] = {
                video: [lf.frame_idx for lf in user_labeled_frames if lf.video == video]
                for video in self.labels.videos
            }

        return selection

    def showLearningDialog(self, mode: str):
        """Helper function to show learning dialog in given mode.

        Args:
            mode: A string representing mode for dialog, which could be:
            * "training"
            * "inference"

        Returns:
            None.
        """
        from sleap.gui.learning.dialog import LearningDialog

        if "inference" in self.overlays:
            QMessageBox(
                text="In order to use this function you must first quit and "
                "re-open SLEAP to release resources used by visualizing "
                "model outputs."
            ).exec_()
            return

        if not self.state["filename"] or self.state["has_changes"]:
            QMessageBox(
                text="You have unsaved changes. Please save before running training or inference."
            ).exec_()
            return

        if self._child_windows.get(mode, None) is None:
            self._child_windows[mode] = LearningDialog(
                mode, self.state["filename"], self.labels,
            )
            self._child_windows[mode].learningFinished.connect(self.learningFinished)

        self._child_windows[mode].update_file_lists()

        self._child_windows[mode].frame_selection = self._frames_for_prediction()
        self._child_windows[mode].open()

    def learningFinished(self, new_count: int):
        """Called when inference finishes."""
        # we ran inference so update display/ui
        self.on_data_update([UpdateTopic.all])
        if new_count > 0:
            self.commands.changestack_push("new predictions")

    def visualizeOutputs(self):
        """Gui for adding overlay with live visualization of predictions."""
        filters = ["Model (*.json)"]

        # Default to opening from models directory from project
        models_dir = None
        if self.state["filename"] is not None:
            models_dir = os.path.join(
                os.path.dirname(self.state["filename"]), "models/"
            )

        # Show dialog
        filename, selected_filter = FileDialog.open(
            self,
            dir=models_dir,
            caption="Import model outputs...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return

        # Model as overlay datasource
        # This will show live inference results

        from sleap.gui.overlays.base import DataOverlay

        overlay = DataOverlay.from_model(
            filename, self.state["video"], player=self.player
        )

        self.overlays["inference"] = overlay

        self.plotFrame()

    def doubleClickInstance(self, instance: Instance, event: QtGui.QMouseEvent = None):
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
            mark_complete = False
            # Mark the nodes as "complete" if shift-key is down
            if event is not None and event.modifiers() & Qt.ShiftModifier:
                mark_complete = True

            self.commands.newInstance(
                copy_instance=instance, mark_complete=mark_complete
            )

        # When a regular instance is double-clicked, add any missing points
        else:
            self.commands.completeInstanceNodes(instance)

    def openKeyRef(self):
        """Shows gui for viewing/modifying keyboard shortucts."""
        ShortcutDialog().exec_()


def main():
    """Starts new instance of app."""

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

    app = QApplication([])
    app.setApplicationName(f"SLEAP Label v{sleap.version.__version__}")

    window = MainWindow(labels_path=args.labels_path)
    window.showMaximized()

    # if not args.labels_path:
    #     window.commands.openProject(first_open=True)

    app.exec_()


if __name__ == "__main__":
    main()

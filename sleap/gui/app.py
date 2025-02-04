"""Main GUI application for labeling, training/inference, and proofreading.

Each open project is an instance of :py:class:`MainWindow`.

The main window contains a :py:class:`QtVideoPlayer` widget for showing
video frames (the video player widget contains both a graphics view widget
that shows the frame image and a seekbar widget for navigation). The main
window also contains various "data views"--tables which can be docked
in the window as well as a status bar.

When a new instance of :py:class:`MainWindow` is created, it creates
all of these widgets, sets up the menus, and also creates

- single :py:class:`GuiState` object
- single :py:class:`CommandContext` object
- single :py:class:`ColorManager` object
- multiple overlay objects (subclasses of :py:class:`BaseOverlay`)

A timer is started (runs via Qt event loop) which enables/disables
various menu items and buttons based on current state (e.g., you
can't delete an instance if no instance is selected).

Shortcuts are loaded using :py:class:`Shortcuts` class. Preferences
are loaded by importing `prefs`, a singleton instance of
:py:class:`Preferences`.

:py:class:`GuiState` is used for storing "global" state for the project
(e.g., :py:class:`Labels` object, the current frame, current instance,
whether to show track trails, etc.). every menu command with state
(e.g., check/uncheck) should be connected to a state variable.

:py:class:`CommandContext` has methods which can be triggered
by menu items/buttons/etc in the GUI to perform various actions. The
command context enforces a pattern for implementing each command in
its own class, it keeps track of whether there are unsaved changes
(and in the future would make it easier to implement undo/redo), and
it handles triggering the relevant updates in the GUI based on the
effects of the command (these are passed using `UpdateTopic` enum and
handed by :py:method:`MainWindow.on_data_update()`).

:py:class:`ColorManager` loads color palettes, keeps track of current
palette, and should always be queried for how to draw instances--this
ensures consistency (e.g.) between color of instances drawn on video
frame and instances listed in data view table.
"""

import os
import platform
import random
import re
import traceback
from logging import getLogger
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import sys
import subprocess

from qtpy import QtCore, QtGui
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QApplication, QMainWindow, QMessageBox

import sleap
from sleap.gui.color import ColorManager
from sleap.gui.commands import CommandContext, UpdateTopic
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog
from sleap.gui.dialogs.metrics import MetricsTableDialog
from sleap.gui.dialogs.shortcuts import ShortcutDialog
from sleap.gui.overlays.instance import InstanceOverlay
from sleap.gui.overlays.tracks import TrackListOverlay, TrackTrailOverlay
from sleap.gui.shortcuts import Shortcuts
from sleap.gui.state import GuiState
from sleap.gui.web import ReleaseChecker, ping_analytics
from sleap.gui.widgets.docks import (
    InstancesDock,
    SkeletonDock,
    SuggestionsDock,
    VideosDock,
)
from sleap.gui.widgets.slider import set_slider_marks_from_labels
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.info.summary import StatisticSeries
from sleap.instance import Instance
from sleap.io.dataset import Labels
from sleap.io.video import available_video_exts
from sleap.prefs import prefs
from sleap.skeleton import Skeleton
from sleap.util import parse_uri_path, get_config_file


logger = getLogger(__name__)


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

    def __init__(
        self,
        labels_path: Optional[str] = None,
        labels: Optional[Labels] = None,
        reset: bool = False,
        no_usage_data: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the app.

        Args:
            labels_path: Path to saved :class:`Labels` dataset.
            reset: If `True`, reset preferences to default (including window state).
            no_usage_data: If `True`, launch GUI without sharing usage data regardless
                of stored preferences.
        """
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setAcceptDrops(True)

        self.state = GuiState()
        self.labels = labels or Labels()

        self.commands = CommandContext(
            state=self.state, app=self, update_callback=self.on_data_update
        )

        self.shortcuts = Shortcuts()

        self._menu_actions = dict()
        self._buttons = dict()
        self._child_windows = dict()

        self.overlays = dict()

        self.state.connect("filename", self.setWindowTitle)

        self.state["skeleton"] = Skeleton()
        self.state["labeled_frame"] = None
        self.state["last_interacted_frame"] = None
        self.state["filename"] = None
        self.state["show non-visible nodes"] = prefs["show non-visible nodes"]
        self.state["show instances"] = True
        self.state["show labels"] = True
        self.state["show edges"] = True
        self.state["edge style"] = prefs["edge style"]
        self.state["fit"] = False
        self.state["color predicted"] = prefs["color predicted"]
        self.state["trail_length"] = prefs["trail length"]
        self.state["trail_shade"] = prefs["trail shade"]
        self.state["marker size"] = prefs["marker size"]
        self.state["propagate track labels"] = prefs["propagate track labels"]
        self.state["node label size"] = prefs["node label size"]
        self.state["share usage data"] = prefs["share usage data"]
        self.state["skeleton_preview_image"] = None
        self.state["skeleton_description"] = "No skeleton loaded yet"
        if no_usage_data:
            self.state["share usage data"] = False
        self.state["clipboard_track"] = None
        self.state["clipboard_instance"] = None

        self.state.connect("marker size", self.plotFrame)
        self.state.connect("node label size", self.plotFrame)
        self.state.connect("show non-visible nodes", self.plotFrame)

        self.release_checker = ReleaseChecker()

        if self.state["share usage data"]:
            ping_analytics()

        self._initialize_gui()

        if reset:
            print("Reseting GUI state and preferences...")
            prefs.reset_to_default()
        elif len(prefs["window state"]) > 0:
            print("Restoring GUI state...")
            self.restoreState(prefs["window state"])

        if labels_path is not None:
            self.commands.loadProjectFile(filename=labels_path)
        elif labels is not None:
            self.commands.loadLabelsObject(labels=labels)
        else:
            self.state["project_loaded"] = False

    def setWindowTitle(self, value):
        """Sets window title (if value is not None)."""
        if value is not None:
            super(MainWindow, self).setWindowTitle(
                f"{value} - SLEAP v{sleap.version.__version__}"
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
        """Close application window, prompting for saving as needed."""
        # Save window state.
        prefs["window state"] = self.saveState()
        prefs["marker size"] = self.state["marker size"]
        prefs["show non-visible nodes"] = self.state["show non-visible nodes"]
        prefs["node label size"] = self.state["node label size"]
        prefs["edge style"] = self.state["edge style"]
        prefs["propagate track labels"] = self.state["propagate track labels"]
        prefs["color predicted"] = self.state["color predicted"]
        prefs["trail length"] = self.state["trail_length"]
        prefs["trail shade"] = self.state["trail_shade"]
        prefs["share usage data"] = self.state["share usage data"]

        # Save preferences.
        prefs.save()

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

    def dragEnterEvent(self, event):
        # TODO: Parse filenames and accept only if valid ext (or folder)
        mime_format = 'application/x-qt-windows-mime;value="FileName"'
        if mime_format in event.mimeData().formats():
            # This only returns the first filename if multiple files are dropped:
            filename = event.mimeData().data(mime_format).data().decode()
            event.acceptProposedAction()

    def dropEvent(self, event):
        # Parse filenames
        filenames = event.mimeData().data("text/uri-list").data().decode()
        filenames = [parse_uri_path(f.strip()) for f in filenames.strip().split("\n")]

        exts = [Path(f).suffix for f in filenames]

        if len(exts) == 1 and exts[0].lower() == ".slp":
            if self.state["project_loaded"]:
                # Merge
                self.commands.mergeProject(filenames=filenames)
            else:
                # Load
                self.commands.openProject(filename=filenames[0], first_open=True)

        elif all([ext.lower()[1:] in available_video_exts() for ext in exts]):
            # Import videos
            self.commands.showImportVideos(filenames=filenames)

        else:
            raise TypeError(
                f"Invalid file type(s) dropped: {', '.join(exts)} \n"
                f"Supported formats: .slp, .{', .'.join(available_video_exts())}"
            )

    @property
    def labels(self) -> Labels:
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

        self._load_overlays()

        # Create timer to update state of gui at 20 millisec. intervals
        self.update_gui_timer = QtCore.QTimer()
        self.update_gui_timer.timeout.connect(self._update_gui_state)
        self.update_gui_timer.start(20)

    def _create_video_player(self):
        """Creates and connects :class:`QtVideoPlayer` for gui."""
        self.player = QtVideoPlayer(
            color_manager=self.color_manager, state=self.state, context=self.commands
        )
        self.player.changedPlot.connect(self._after_plot_change)
        self.player.updatedPlot.connect(self._after_plot_update)

        self.player.view.instanceDoubleClicked.connect(
            self._handle_instance_double_click
        )
        self.player.seekbar.selectionChanged.connect(lambda: self.updateStatusMessage())
        self.setCentralWidget(self.player)

        def switch_frame(video):
            """Jump to last labeled frame"""
            last_label = self.labels.find_last(video)
            if last_label is not None:
                self.state["frame_idx"] = last_label.frame_idx
            else:
                self.state["frame_idx"] = 0

        def update_frame_chunk_suggestions(video):
            """Set upper limit of frame_chunk spinbox to number frames in video."""
            method_layout = (
                self.suggestions_dock.suggestions_form_widget.form_layout.fields[
                    "method"
                ]
            )
            frame_chunk_layout = method_layout.page_layouts["frame chunk"]
            frame_to_spinbox = frame_chunk_layout.fields["frame_to"]
            frame_from_spinbox = frame_chunk_layout.fields["frame_from"]
            if video is not None:
                frame_to_spinbox.setMaximum(video.num_frames)
                frame_from_spinbox.setMaximum(video.num_frames)

        self.state.connect(
            "video",
            callbacks=[
                switch_frame,
                lambda x: self._update_seekbar_marks(),
                update_frame_chunk_suggestions,
            ],
        )

    def _create_color_manager(self):
        self.color_manager = ColorManager(self.labels)
        self.color_manager.palette = self.state.get("palette", default="standard")

    def _create_menus(self):
        """Creates main application menus."""
        # shortcuts = Shortcuts()

        # add basic menu item
        def add_menu_item(menu, key: str, name: str, action: Callable):
            menu_item = menu.addAction(name, action, self.shortcuts[key])
            self._menu_actions[key] = menu_item
            return menu_item

        # set menu checkmarks
        def connect_check(key):
            self._menu_actions[key].setCheckable(True)
            self._menu_actions[key].setChecked(self.state[key])
            self.state.connect(
                key, lambda checked: self._menu_actions[key].setChecked(checked)
            )

        # add checkable menu item connected to state variable
        def add_menu_check_item(menu, key: str, name: str):
            menu_item = add_menu_item(menu, key, name, lambda: self.state.toggle(key))
            connect_check(key)
            return menu_item

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
            "import_dlc_folder",
            "Multiple DeepLabCut datasets from folder...",
            self.commands.importDLCFolder,
        )
        add_menu_item(
            import_types_menu,
            "import_dpk",
            "DeepPoseKit dataset...",
            self.commands.importDPK,
        )
        add_menu_item(
            import_types_menu,
            "import_at",
            "AlphaTracker dataset...",
            self.commands.importAT,
        )
        add_menu_item(
            import_types_menu,
            "import_nwb",
            "NWB dataset...",
            self.commands.importNWB,
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
            self.commands.mergeProject,
        )

        fileMenu.addSeparator()
        add_menu_item(fileMenu, "add videos", "Add Videos...", self.commands.addVideo)
        add_menu_item(
            fileMenu, "replace videos", "Replace Videos...", self.commands.replaceVideo
        )

        fileMenu.addSeparator()
        add_menu_item(fileMenu, "save", "Save", self.commands.saveProject)
        add_menu_item(fileMenu, "save as", "Save As...", self.commands.saveProjectAs)

        export_analysis_menu = fileMenu.addMenu("Export Analysis HDF5...")
        add_menu_item(
            export_analysis_menu,
            "export_analysis_current",
            "Current Video...",
            self.commands.exportAnalysisFile,
        )
        add_menu_item(
            export_analysis_menu,
            "export_analysis_video",
            "All Videos...",
            lambda: self.commands.exportAnalysisFile(all_videos=True),
        )

        export_csv_menu = fileMenu.addMenu("Export Analysis CSV...")
        add_menu_item(
            export_csv_menu,
            "export_csv_current",
            "Current Video...",
            self.commands.exportCSVFile,
        )
        add_menu_item(
            export_csv_menu,
            "export_csv_all",
            "All Videos...",
            lambda: self.commands.exportCSVFile(all_videos=True),
        )

        add_menu_item(fileMenu, "export_nwb", "Export NWB...", self.commands.exportNWB)

        fileMenu.addSeparator()
        add_menu_item(
            fileMenu, "reset prefs", "Reset preferences to defaults...", self.resetPrefs
        )

        add_menu_item(
            fileMenu,
            "open preference directory",
            "Open Preferences Directory...",
            self.openPrefs,
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
            "goto last interacted",
            "Last Interacted Frame",
            self.commands.lastInteractedFrame,
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

        goMenu.addSeparator()

        add_menu_item(
            goMenu,
            "select next",
            "Select Next Instance",
            lambda: self.state.increment_in_list(
                "instance", self.state["labeled_frame"].instances_to_show
            ),
        )
        add_menu_item(
            goMenu,
            "clear selection",
            "Clear Selection",
            lambda: self.state.set("instance", None),
        )

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

        viewMenu.addSeparator()
        add_menu_check_item(viewMenu, "fit", "Fit Instances to View")

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

        add_menu_check_item(viewMenu, "show instances", "Show Instances")
        add_menu_check_item(
            viewMenu, "show non-visible nodes", "Show Non-Visible Nodes"
        )
        add_menu_check_item(viewMenu, "show labels", "Show Node Names")
        add_menu_check_item(viewMenu, "show edges", "Show Edges")

        add_submenu_choices(
            menu=viewMenu,
            title="Edge Style",
            options=("Line", "Wedge"),
            key="edge style",
        )

        # XXX
        add_submenu_choices(
            menu=viewMenu,
            title="Node Marker Size",
            options=prefs["node marker sizes"],
            key="marker size",
        )

        add_submenu_choices(
            menu=viewMenu,
            title="Node Label Size",
            options=prefs["node label sizes"],
            key="node label size",
        )

        viewMenu.addSeparator()
        add_submenu_choices(
            menu=viewMenu,
            title="Trail Length",
            options=TrackTrailOverlay.get_length_options(),
            key="trail_length",
        )
        add_submenu_choices(
            menu=viewMenu,
            title="Trail Shade",
            options=tuple(TrackTrailOverlay.get_shade_options().keys()),
            key="trail_shade",
        )

        viewMenu.addSeparator()
        add_menu_item(
            viewMenu,
            "export clip",
            "Render Video Clip with Instances...",
            self.commands.exportLabeledClip,
        )
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
            """Determine which action to use when using Ctrl + I or menu Add Instance.

            We always add an offset of 10.
            """
            method_key = [
                key
                for (key, val) in instance_adding_methods.items()
                if val == self.state["instance_init_method"]
            ]
            if method_key:
                self.commands.newInstance(init_method=method_key[0], offset=10)

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

        add_menu_item(
            labelMenu,
            "custom delete",
            "Custom Instance Delete...",
            self.commands.deleteDialog,
        )

        labelMenu.addSeparator()

        add_menu_item(
            labelMenu,
            "add instances from all frame predictions",
            "Add Instances from All Predictions on Current Frame",
            self.commands.addUserInstancesFromPredictions,
        )

        labelMenu.addSeparator()

        labelMenu.addAction(
            "Copy Instance",
            self.commands.copyInstance,
            Qt.CTRL | Qt.Key_C,
        )
        labelMenu.addAction(
            "Paste Instance",
            self.commands.pasteInstance,
            Qt.CTRL | Qt.Key_V,
        )

        labelMenu.addSeparator()

        add_menu_item(
            labelMenu,
            "delete frame predictions",
            "Delete Predictions on Current Frame",
            self.commands.deleteFramePredictions,
        )
        add_menu_item(
            labelMenu,
            "delete all predictions",
            "Delete All Predictions...",
            self.commands.deletePredictions,
        )
        add_menu_item(
            labelMenu,
            "delete clip predictions",
            "Delete Predictions from Clip...",
            self.commands.deleteClipPredictions,
        )
        add_menu_item(
            labelMenu,
            "delete area predictions",
            "Delete Predictions from Area...",
            self.commands.deleteAreaPredictions,
        )
        add_menu_item(
            labelMenu,
            "delete score predictions",
            "Delete Predictions with Low Score...",
            self.commands.deleteLowScorePredictions,
        )
        add_menu_item(
            labelMenu,
            "delete max instance predictions",
            "Delete Predictions beyond Max Instances...",
            self.commands.deleteInstanceLimitPredictions,
        )
        add_menu_item(
            labelMenu,
            "delete frame limit predictions",
            "Delete Predictions beyond Frame Limit...",
            self.commands.deleteFrameLimitPredictions,
        )

        ### Tracks Menu ###

        tracksMenu = self.menuBar().addMenu("Tracks")
        self.track_menu = tracksMenu.addMenu("Set Instance Track")
        add_menu_check_item(
            tracksMenu, "propagate track labels", "Propagate Track Labels"
        ).setToolTip(
            "If enabled, setting a track will also apply to subsequent instances of "
            "the same track."
        )
        add_menu_item(
            tracksMenu,
            "transpose",
            "Transpose Instance Tracks",
            self.commands.transposeInstance,
        )

        tracksMenu.addSeparator()

        add_menu_item(
            tracksMenu,
            "delete track",
            "Delete Instance and Track",
            self.commands.deleteSelectedInstanceTrack,
        )
        self.delete_tracks_menu = tracksMenu.addMenu("Delete Track")
        self.delete_tracks_menu.setEnabled(False)

        self.delete_multiple_tracks_menu = tracksMenu.addMenu("Delete Multiple Tracks")
        self.delete_multiple_tracks_menu.setToolTip(
            "Delete either only 'Unused' tracks or 'All' tracks, and update instances. Instances are not removed."
        )

        add_menu_item(
            self.delete_multiple_tracks_menu,
            "delete unused tracks",
            "Unused",
            lambda: self.commands.deleteMultipleTracks(delete_all=False),
        )

        add_menu_item(
            self.delete_multiple_tracks_menu,
            "delete all tracks",
            "All",
            lambda: self.commands.deleteMultipleTracks(delete_all=True),
        )

        tracksMenu.addSeparator()

        tracksMenu.addAction(
            "Copy Instance Track",
            self.commands.copyInstanceTrack,
            Qt.CTRL | Qt.SHIFT | Qt.Key_C,
        )
        tracksMenu.addAction(
            "Paste Instance Track",
            self.commands.pasteInstanceTrack,
            Qt.CTRL | Qt.SHIFT | Qt.Key_V,
        )

        tracksMenu.addSeparator()

        seekbar_header_options = (
            "None",
            "Point Displacement (sum)",
            "Point Displacement (max)",
            "Primary Point Displacement (sum)",
            "Primary Point Displacement (max)",
            "Tracking Score (mean)",
            "Tracking Score (min)",
            "Instance Score (sum)",
            "Instance Score (min)",
            "Point Score (sum)",
            "Point Score (min)",
            "Number of predicted points",
            "Min Centroid Proximity",
        )

        add_submenu_choices(
            menu=tracksMenu,
            title="Seekbar Header",
            options=seekbar_header_options,
            key="seekbar_header",
        )

        self.state["seekbar_header"] = "None"
        self.state.connect("seekbar_header", self._set_seekbar_header)

        ### Predict Menu ###

        predictionMenu = self.menuBar().addMenu("Predict")
        predictionMenu.setToolTipsVisible(True)

        add_menu_item(
            predictionMenu,
            "training",
            "Run Training...",
            lambda: self._show_learning_dialog("training"),
        )
        add_menu_item(
            predictionMenu,
            "inference",
            "Run Inference...",
            lambda: self._show_learning_dialog("inference"),
        )

        predictionMenu.addSeparator()

        add_menu_item(
            predictionMenu,
            "show metrics",
            "Evaluation Metrics for Trained Models...",
            self._show_metrics_dialog,
        )

        add_menu_item(
            predictionMenu,
            "visualize models",
            "Visualize Model Outputs...",
            self._handle_model_overlay_command,
        )

        predictionMenu.addSeparator()

        labels_package_menu = predictionMenu.addMenu("Export Labels Package...")
        add_menu_item(
            labels_package_menu,
            "export user labels package",
            "Labeled frames",
            self.commands.exportUserLabelsPackage,
        ).setToolTip(
            "Export user-labeled frames with image data into a single SLP file.\n\n"
            "Use this for archiving a dataset with labeled frames only."
        )
        add_menu_item(
            labels_package_menu,
            "export labels package",
            "Labeled + suggested frames (recommended)",
            self.commands.exportTrainingPackage,
        ).setToolTip(
            "Export user-labeled frames and suggested frames with image data into a "
            "single SLP file.\n\n"
            "Use this for human-in-the-loop training to enable remote inference on "
            "unlabeled frames."
        )
        add_menu_item(
            labels_package_menu,
            "export full package",
            "Labeled + predicted + suggested frames",
            self.commands.exportFullPackage,
        ).setToolTip(
            "Export all frames (including predictions) and suggested frames with image "
            "data into a single SLP file.\n\n"
            "Use this when you need to store images for predicted frames, such as for "
            "proofreading or reproducibility."
        )

        predictionMenu.addSeparator()
        add_menu_item(
            predictionMenu,
            "training on colab",
            "Train on Google Colab...",
            lambda: self.commands.openWebsite(
                "https://colab.research.google.com/github/talmolab/sleap/blob/main/docs/notebooks/Training_and_inference_using_Google_Drive.ipynb"
            ),
        )

        ############

        helpMenu = self.menuBar().addMenu("Help")

        helpMenu.addAction(
            "Documentation", lambda: self.commands.openWebsite("https://sleap.ai")
        )
        helpMenu.addAction(
            "GitHub",
            lambda: self.commands.openWebsite("https://github.com/talmolab/sleap"),
        )
        helpMenu.addAction(
            "Releases",
            lambda: self.commands.openWebsite(
                "https://github.com/talmolab/sleap/releases"
            ),
        )

        helpMenu.addSeparator()

        helpMenu.addAction("Latest versions:", self.commands.checkForUpdates)
        self.state["stable_version_menu"] = helpMenu.addAction(
            "  Stable: N/A", self.commands.openStableVersion
        )
        self.state["stable_version_menu"].setEnabled(False)
        self.state["prerelease_version_menu"] = helpMenu.addAction(
            "  Prerelease: N/A", self.commands.openPrereleaseVersion
        )
        self.state["prerelease_version_menu"].setEnabled(False)
        self.commands.checkForUpdates()

        helpMenu.addSeparator()
        usageMenu = helpMenu.addMenu("Improve SLEAP")
        add_menu_check_item(usageMenu, "share usage data", "Share usage data")
        usageMenu.addAction(
            "What is usage data?",
            lambda: self.commands.openWebsite("https://sleap.ai/help.html#usage-data"),
        )

        helpMenu.addSeparator()
        helpMenu.addAction("Keyboard Shortcuts", self._show_keyboard_shortcuts_window)

    def process_events_then(self, action: Callable):
        """Decorates a function with a call to first process events."""

        def wrapped_function(*args):
            QApplication.instance().processEvents()
            action(*args)

        return wrapped_function

    def _create_dock_windows(self):
        """Create dock windows and connect them to GUI."""

        self.videos_dock = VideosDock(self)
        self.skeleton_dock = SkeletonDock(self, tab_with=self.videos_dock)
        self.suggestions_dock = SuggestionsDock(self, tab_with=self.videos_dock)
        self.instances_dock = InstancesDock(self, tab_with=self.videos_dock)

        # Bring videos tab forward.
        self.videos_dock.wgt_layout.parent().parent().raise_()

    def _load_overlays(self):
        """Load all standard video overlays."""
        self.overlays["track_labels"] = TrackListOverlay(
            labels=self.labels, player=self.player
        )
        self.overlays["trails"] = TrackTrailOverlay(
            labels=self.labels,
            player=self.player,
            trail_shade=self.state["trail_shade"],
            trail_length=self.state["trail_length"],
        )
        self.overlays["instance"] = InstanceOverlay(
            labels=self.labels, player=self.player, state=self.state
        )

        # When gui state changes, we also want to set corresponding attribute
        # on overlay (or color manager shared by overlays) so that they can
        # update/redraw as needed.
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
        overlay_state_connect(self.overlays["trails"], "trail_shade")

        overlay_state_connect(self.color_manager, "palette")
        overlay_state_connect(self.color_manager, "distinctly_color")
        overlay_state_connect(self.color_manager, "color predicted", "color_predicted")
        self.state.connect("palette", lambda x: self._update_seekbar_marks())

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
        has_selected_node = self.state["selected_node"] is not None
        has_selected_edge = self.state["selected_edge"] is not None
        has_selected_video = self.state["selected_video"] is not None
        has_video = self.state["video"] is not None

        has_frame_range = bool(self.state["has_frame_range"])
        has_unsaved_changes = bool(self.state["has_changes"])
        has_videos = self.labels is not None and len(self.labels.videos) > 0
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
            self.skeleton_dock.skeletonEdgesSrc.currentIndex() > -1
            and self.skeleton_dock.skeletonEdgesDst.currentIndex() > -1
        )
        control_key_down = QApplication.queryKeyboardModifiers() == Qt.ControlModifier

        # Update menus

        self.track_menu.setEnabled(has_selected_instance)
        self.delete_tracks_menu.setEnabled(has_tracks)
        self._menu_actions["clear selection"].setEnabled(has_selected_instance)
        self._menu_actions["delete instance"].setEnabled(has_selected_instance)

        self._menu_actions["delete clip predictions"].setEnabled(has_frame_range)
        # self._menu_actions["export clip"].setEnabled(has_frame_range)

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
        self._buttons["toggle grayscale"].setEnabled(has_video)
        self._buttons["show video"].setEnabled(has_selected_video)
        self._buttons["remove video"].setEnabled(has_video)
        self._buttons["delete instance"].setEnabled(has_selected_instance)
        self.suggestions_dock.suggestions_form_widget.buttons[
            "generate_button"
        ].setEnabled(has_videos)

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
            self._update_seekbar_marks()

        if _has_topic(
            [UpdateTopic.frame, UpdateTopic.project_instances, UpdateTopic.tracks]
        ):
            self._update_track_menu()

        if _has_topic([UpdateTopic.video]):
            self.videos_dock.table.model().items = self.labels.videos

        if _has_topic([UpdateTopic.skeleton]):
            self.skeleton_dock.nodes_table.model().items = self.state["skeleton"]
            self.skeleton_dock.edges_table.model().items = self.state["skeleton"]
            self.skeleton_dock.skeletonEdgesSrc.model().skeleton = self.state[
                "skeleton"
            ]
            self.skeleton_dock.skeletonEdgesDst.model().skeleton = self.state[
                "skeleton"
            ]

            if self.labels.skeletons:
                self.suggestions_dock.suggestions_form_widget.set_field_options(
                    "node", self.labels.skeletons[0].node_names
                )

        if _has_topic([UpdateTopic.project, UpdateTopic.on_frame]):
            self.instances_dock.table.model().items = self.state["labeled_frame"]

        if _has_topic([UpdateTopic.suggestions]):
            self.suggestions_dock.table.model().items = self.labels.suggestions

        if _has_topic([UpdateTopic.project_instances, UpdateTopic.suggestions]):
            # update count of suggested frames w/ labeled instances
            suggestion_status_text = ""
            suggestion_list = self.labels.get_suggestions()
            if suggestion_list:
                labeled_count = 0
                for suggestion in suggestion_list:
                    lf = self.labels.get(
                        (suggestion.video, suggestion.frame_idx), use_cache=True
                    )
                    if lf is not None and lf.has_user_instances:
                        labeled_count += 1
                prc = (labeled_count / len(suggestion_list)) * 100
                suggestion_status_text = (
                    f"{labeled_count}/{len(suggestion_list)} labeled ({prc:.1f}%)"
                )
            self.suggestions_dock.suggested_count_label.setText(suggestion_status_text)

        if _has_topic([UpdateTopic.frame, UpdateTopic.project_instances]):
            self.state["last_interacted_frame"] = self.state["labeled_frame"]

    def plotFrame(self, *args, **kwargs):
        """Plots (or replots) current frame."""
        if self.state["video"] is None:
            return

        self.player.plot()

    def _after_plot_update(self, frame_idx):
        """Run after plot is updated, but stay on same frame."""
        overlay: TrackTrailOverlay = self.overlays["trails"]
        overlay.redraw(self.state["video"], frame_idx)

    def _after_plot_change(self, player, frame_idx, selected_inst):
        """Called each time a new frame is drawn."""

        # Store the current frame_idx and LabeledFrame (or make new, empty object)
        self.state["frame_idx"] = frame_idx
        self.state["labeled_frame"] = (
            self.labels.find(self.state["video"], frame_idx, return_new=True)[0]
            if frame_idx is not None
            else None
        )

        # Show instances, etc, for this frame
        for overlay in self.overlays.values():
            overlay.redraw(self.state["video"], frame_idx)

        # Select instance if there was already selection
        if selected_inst is not None:
            player.view.selectInstance(selected_inst)
        else:
            self.state["instance"] = None

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
            message = ""
            if len(self.labels.videos) > 0 and current_video is not None:
                message += f"Video {self.labels.videos.index(current_video)+1}/"
                message += f"{len(self.labels.videos)}"
                message += spacer

            if current_video is not None:
                message += f"Frame: {frame_idx+1:,}/{len(current_video):,}"

            if self.player.seekbar.hasSelection():
                start, end = self.state["frame_range"]
                message += spacer
                message += f"Selection: {start+1:,}-{end:,} ({end-start+1:,} frames)"

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

            lf = self.state["labeled_frame"]
            # TODO: revisit with LabeledFrame.unused_predictions() & instances_to_show()
            n_instances = 0 if lf is None else len(lf.instances_to_show)
            message += f"{spacer}Current frame: {n_instances} instances"
            if (n_instances > 0) and not self.state["show instances"]:
                hide_key = self.shortcuts["show instances"].toString()
                message += f" [Hidden] Press '{hide_key}' to toggle."
                self.statusBar().setStyleSheet("color: red")
            else:
                self.statusBar().setStyleSheet("")

        self.statusBar().showMessage(message)

    def resetPrefs(self):
        """Reset preferences to defaults."""
        prefs.reset_to_default()
        msg = QMessageBox()
        msg.setText(
            "Note: Some preferences may not take effect until application is restarted."
        )
        msg.exec_()

    def openPrefs(self):
        """Open preference file directory"""
        pref_path = get_config_file("preferences.yaml")
        # Make sure the pref_path is a directory rather than a file
        if pref_path.is_file():
            pref_path = pref_path.parent
        # Open the file explorer at the folder containing the preferences.yaml file
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(pref_path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(pref_path)])
        else:
            subprocess.Popen(["xdg-open", str(pref_path)])

    def _update_track_menu(self):
        """Updates track menu options."""
        self.track_menu.clear()
        self.delete_tracks_menu.clear()

        # Create a dictionary mapping track indices to Qt.Key values
        key_mapping = {
            0: Qt.Key_1,
            1: Qt.Key_2,
            2: Qt.Key_3,
            3: Qt.Key_4,
            4: Qt.Key_5,
            5: Qt.Key_6,
            6: Qt.Key_7,
            7: Qt.Key_8,
            8: Qt.Key_9,
            9: Qt.Key_0,
        }
        for track_ind, track in enumerate(self.labels.tracks):
            key_command = ""
            if track_ind < 9:
                key_command = Qt.CTRL | key_mapping[track_ind]
            self.track_menu.addAction(
                f"{track.name}",
                lambda x=track: self.commands.setInstanceTrack(x),
                key_command,
            )
            self.delete_tracks_menu.addAction(
                f"{track.name}", lambda x=track: self.commands.deleteTrack(x)
            )
        self.track_menu.addAction(
            "New Track", self.commands.addTrack, Qt.CTRL | Qt.Key_0
        )

    def _update_seekbar_marks(self):
        """Updates marks on seekbar."""
        set_slider_marks_from_labels(
            self.player.seekbar, self.labels, self.state["video"], self.color_manager
        )

    def _set_seekbar_header(self, graph_name: str):
        """Updates graph shown in seekbar header based on menu selection."""
        data_obj = StatisticSeries(self.labels)
        header_functions = {
            "Point Displacement (sum)": data_obj.get_point_displacement_series,
            "Point Displacement (max)": data_obj.get_point_displacement_series,
            "Primary Point Displacement (sum)": data_obj.get_primary_point_displacement_series,
            "Primary Point Displacement (max)": data_obj.get_primary_point_displacement_series,
            "Tracking Score (mean)": data_obj.get_tracking_score_series,
            "Tracking Score (min)": data_obj.get_tracking_score_series,
            "Instance Score (sum)": data_obj.get_instance_score_series,
            "Instance Score (min)": data_obj.get_instance_score_series,
            "Point Score (sum)": data_obj.get_point_score_series,
            "Point Score (min)": data_obj.get_point_score_series,
            "Number of predicted points": data_obj.get_point_count_series,
            "Min Centroid Proximity": data_obj.get_min_centroid_proximity_series,
        }

        if graph_name == "None":
            self.player.seekbar.clearHeader()
        else:
            if graph_name in header_functions:
                kwargs = dict(video=self.state["video"])
                reduction_name = re.search("\\((sum|max|min|mean)\\)", graph_name)
                if reduction_name is not None:
                    kwargs["reduction"] = reduction_name.group(1)
                series = header_functions[graph_name](**kwargs)
                self.player.seekbar.setHeaderSeries(series)
            else:
                print(f"Could not find function for {header_functions}")

    def _get_frames_for_prediction(self):
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
        selection["all_videos"] = {
            video: encode_range(0, video.num_frames) for video in self.labels.videos
        }

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

    def _show_learning_dialog(self, mode: str):
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
                text=(
                    "You have unsaved changes. Please save before running training or "
                    "inference."
                )
            ).exec_()
            return

        if self._child_windows.get(mode, None) is None:
            # Re-use existing dialog widget.
            self._child_windows[mode] = LearningDialog(
                mode,
                self.state["filename"],
                self.labels,
            )
            self._child_windows[mode]._handle_learning_finished.connect(
                self._handle_learning_finished
            )
        else:
            # Update data in existing dialog widget.
            self._child_windows[mode].labels = self.labels
            self._child_windows[mode].labels_filename = self.state["filename"]
            self._child_windows[mode].skeleton = self.labels.skeleton

        self._child_windows[mode].update_file_lists()

        self._child_windows[mode].frame_selection = self._get_frames_for_prediction()
        self._child_windows[mode].open()

    def _handle_learning_finished(self, new_count: int):
        """Called when inference finishes."""
        if (
            len(self.labels.skeletons) > 0
            and self.state["skeleton"] not in self.labels.skeletons
        ):
            # Update the GUI state skeleton if the labels skeleton changed after merge.
            self.state["skeleton"] = self.labels.skeletons[-1]
        # we ran inference so update display/ui
        self.on_data_update([UpdateTopic.all])
        if new_count > 0:
            self.commands.changestack_push("new predictions")

    def _show_metrics_dialog(self):
        self._child_windows["metrics"] = MetricsTableDialog(self.state["filename"])
        self._child_windows["metrics"].show()

    def _handle_model_overlay_command(self):
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

        predictor = DataOverlay.make_predictor(filename)
        show_pafs = False

        # If multi-head model with both confmaps and pafs,
        # ask user which to show.
        if (
            predictor.confidence_maps_key_name
            and predictor.part_affinity_fields_key_name
        ):
            results = FormBuilderModalDialog(form_name="head_type_form").get_results()
            show_pafs = "Part Affinity" in results["head_type"]

        overlay = DataOverlay.from_predictor(
            predictor=predictor,
            video=self.state["video"],
            player=self.player,
            show_pafs=show_pafs,
        )

        self.overlays["inference"] = overlay

        self.plotFrame()

    def _handle_instance_double_click(
        self, instance: Instance, event: QtGui.QMouseEvent = None
    ):
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

    def _show_keyboard_shortcuts_window(self):
        """Shows gui for viewing/modifying keyboard shortucts."""
        ShortcutDialog().exec_()


def create_sleap_label_parser():
    """Creates parser for `sleap-label` command line arguments.

    Returns:
        argparse.ArgumentParser: The parser.
    """

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
    parser.add_argument(
        "--profiling",
        help="Enable performance profiling",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--reset",
        help=(
            "Reset GUI state and preferences. Use this flag if the GUI appears "
            "incorrectly or fails to open."
        ),
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--no-usage-data",
        help=("Launch the GUI without sharing usage data regardless of preferences."),
        action="store_const",
        const=True,
        default=False,
    )

    return parser


def create_app():
    """Creates Qt application."""

    app = QApplication([])
    app.setApplicationName(f"SLEAP v{sleap.version.__version__}")
    app.setWindowIcon(QtGui.QIcon(sleap.util.get_package_file("gui/icon.png")))

    return app


def main(args: Optional[list] = None, labels: Optional[Labels] = None):
    """Starts new instance of app."""

    parser = create_sleap_label_parser()
    args = parser.parse_args(args)

    if args.nonnative:
        os.environ["USE_NON_NATIVE_FILE"] = "1"

    if platform.system() == "Darwin":
        # TODO: Remove this workaround when we update to qtpy >= 5.15.
        # https://bugreports.qt.io/browse/QTBUG-87014
        # https://stackoverflow.com/q/64818879
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

    app = create_app()

    window = MainWindow(
        labels_path=args.labels_path,
        labels=labels,
        reset=args.reset,
        no_usage_data=args.no_usage_data,
    )
    window.showMaximized()

    # Disable GPU in GUI process. This does not affect subprocesses.
    try:
        sleap.use_cpu_only()
    except RuntimeError:  # Visible devices cannot be modified after being initialized
        logger.warning(
            "Running processes on the GPU. Restarting your GUI should allow switching "
            "back to CPU-only mode.\n"
            "Received the following error when trying to switch back to CPU-only mode:"
        )
        traceback.print_exc()

    # Print versions.
    print()
    print("Software versions:")
    sleap.versions()
    print()
    print("Happy SLEAPing! :)")

    if args.profiling:
        import cProfile

        cProfile.runctx("app.exec_()", globals=globals(), locals=locals())
    else:
        app.exec_()

    pass

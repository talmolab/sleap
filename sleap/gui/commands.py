"""
Module for gui command context and commands objects.

Each open project (i.e., `MainWindow`) will have its own `CommandContext`.
The context enables commands to access and modify the `GuiState` and `Labels`,
as well as potentially maintaining a command history (so we can add support for
undo!). See `sleap.gui.app` for how the context is created and used.

Every command will have both a method in `CommandContext` (this is what should
be used to trigger the command, e.g., connected to the menu action) and a
class which inherits from `AppCommand` (or a more specialized class such as
`NavCommand`, `GoIteratorCommand`, or `EditCommand`). Note that this code relies
on inheritance, so some care and attention is required.

A typical command will override the `ask` and `do_action` methods. If the
command updates something which affects the GUI, it should override the `topic`
attribute (this then gets passed back to the `update_callback` from the context.
If a command doesn't require any input from the user, then it doesn't need to
override the `ask` method.

If it's not possible to separate the GUI "ask" and the non-GUI "do" code, then
instead of `ask` and `do_action` you should add an `ask_and_do` method
(for instance, `DeleteDialogCommand` and `MergeProject` show dialogues which
handle both the GUI and the action). Ideally we'd endorse separation of "ask"
and "do" for all commands (this is important if we're going to implement undo)--
for now it's at least easy to see where this separation is violated.
"""

import attr
import operator
import os
import re
import sys
import subprocess

from abc import ABC
from enum import Enum
from glob import glob
from pathlib import PurePath
from typing import Callable, Dict, Iterator, List, Optional, Type, Tuple

import numpy as np

from PySide2 import QtCore, QtWidgets, QtGui

from PySide2.QtWidgets import QMessageBox, QProgressDialog

from sleap.gui.dialogs.delete import DeleteDialog
from sleap.skeleton import Skeleton
from sleap.instance import Instance, PredictedInstance, Point, Track, LabeledFrame
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.gui.dialogs.importvideos import ImportVideos
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.missingfiles import MissingFilesDialog
from sleap.gui.dialogs.merge import MergeDialog
from sleap.gui.dialogs.message import MessageDialog
from sleap.gui.suggestions import VideoFrameSuggestions
from sleap.gui.state import GuiState


# whether we support multiple project windows (i.e., "open" opens new window)
OPEN_IN_NEW = True


class UpdateTopic(Enum):
    """Topics so context can tell callback what was updated by the command."""

    all = 1
    video = 2
    skeleton = 3
    labels = 4
    on_frame = 5
    suggestions = 6
    tracks = 7
    frame = 8
    project = 9
    project_instances = 10


class AppCommand:
    """Base class for specific commands.

    Note that this is not an abstract base class. For specific commands, you
    should override `ask` and/or `do_action` methods, or add an `ask_and_do`
    method. In many cases you'll want to override the `topics` and `does_edits`
    attributes. That said, these are not virtual methods/attributes and have
    are implemented in the base class with default behaviors (i.e., doing
    nothing).

    You should not override `execute` or `do_with_signal`.

    Attributes:
        topics: List of `UpdateTopic` items. Override this to indicate what
            should be updated after command is executed.
        does_edits: Whether command will modify data that could be saved.
    """

    topics: List[UpdateTopic] = []
    does_edits: bool = False

    def execute(self, context: "CommandContext", params: dict = None):
        """Entry point for running command.

        This calls internal methods to gather information required for
        execution, perform the action, and notify about changes.

        Ideally, any information gathering should be performed in the `ask`
        method, and be added to the `params` dictionary which then gets
        passed to `do_action`. The `ask` method should not modify state.

        (This will make it easier to add support for undo,
        using an `undo_action` which will be given the same `params`
        dictionary.)

        If it's not possible to easily separate information gathering from
        performing the action, the child class should implement `ask_and_do`,
        which it turn should call `do_with_signal` to notify about changes.

        Args:
            context: This is the `CommandContext` in which the command will
                execute. Commands will use this to access `MainWindow`,
                `GuiState`, and `Labels`.
            params: Dictionary of any params for command.
        """
        params = params or dict()

        if hasattr(self, "ask_and_do") and callable(self.ask_and_do):
            self.ask_and_do(context, params)
        else:
            okay = self.ask(context, params)
            if okay:
                self.do_with_signal(context, params)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        """Method for information gathering.

        Returns:
            Whether to perform action. By default returns True, but this is
            where we should return False if we prompt user for confirmation
            and they abort.
        """
        return True

    @staticmethod
    def do_action(context: "CommandContext", params: dict):
        """Method for performing action."""
        pass

    @classmethod
    def do_with_signal(cls, context: "CommandContext", params: dict):
        """Wrapper to perform action and notify/track changes.

        Don't override this method!
        """
        cls.do_action(context, params)
        if cls.topics:
            context.signal_update(cls.topics)
        if cls.does_edits:
            context.changestack_push(cls.__name__)


@attr.s(auto_attribs=True)
class FakeApp:
    """Use if you want to execute commands independently of the GUI app."""

    labels: Labels


@attr.s(auto_attribs=True, eq=False)
class CommandContext:
    """
    Context within in which commands are executed.

    When you create a new command, you should both create a class for the
    command (which inherits from `CommandClass`) and add a distinct method
    for the command in the `CommandContext` class. This method is what should
    be connected/called from other code to invoke the command.

    Attributes:
        state: The `GuiState` object used to store state and pass messages.
        app: The `MainWindow`, available for commands that modify the app.
        update_callback: A callback to receive update notifications.
            This function should accept a list of `UpdateTopic` items.
    """

    state: GuiState
    app: "MainWindow"

    update_callback: Optional[Callable] = None
    _change_stack: List = attr.ib(default=attr.Factory(list))

    @classmethod
    def from_labels(cls, labels: Labels) -> "CommandContext":
        """Creates a command context for use independently of GUI app."""
        state = GuiState()
        app = FakeApp(labels)
        return cls(state=state, app=app)

    @property
    def labels(self) -> Labels:
        """Alias to app.labels."""
        return self.app.labels

    def signal_update(self, what: List[UpdateTopic]):
        """Calls the update callback after data has been changed."""
        if callable(self.update_callback):
            self.update_callback(what)

    def changestack_push(self, change: str = ""):
        """Adds to stack of changes made by user."""
        # Currently the change doesn't store any data, and we're only using this
        # to determine if there are unsaved changes. Eventually we could use this
        # to support undo/redo.
        self._change_stack.append(change)
        # print(len(self._change_stack))
        self.state["has_changes"] = True

    def changestack_savepoint(self):
        """Marks that project was just saved."""
        self.changestack_push("SAVE")
        self.state["has_changes"] = False

    def changestack_clear(self):
        """Clears stack of changes."""
        self._change_stack = list()
        self.state["has_changes"] = False

    @property
    def has_any_changes(self):
        return len(self._change_stack) > 0

    def execute(self, command: Type[AppCommand], **kwargs):
        """Execute command in this context, passing named arguments."""
        command().execute(context=self, params=kwargs)

    # File commands

    def newProject(self):
        """Create a new project in a new window."""
        self.execute(NewProject)

    def openProject(self, filename: Optional[str] = None, first_open: bool = False):
        """
        Allows use to select and then open a saved project.

        Args:
            filename: Filename of the project to be opened. If None, a file browser
                dialog will prompt the user for a path.
            first_open: Whether this is the first window opened. If True,
                then the new project is loaded into the current window
                rather than a new application window.

        Returns:
            None.
        """
        self.execute(OpenProject, filename=filename, first_open=first_open)

    def importDPK(self):
        """Imports DeepPoseKit datasets."""
        self.execute(ImportDeepPoseKit)

    def importCoco(self):
        """Imports COCO datasets."""
        self.execute(ImportCoco)

    def importDLC(self):
        """Imports DeepLabCut datasets."""
        self.execute(ImportDeepLabCut)

    def importDLCFolder(self):
        """Imports multiple DeepLabCut datasets."""
        self.execute(ImportDeepLabCutFolder)

    def importLEAP(self):
        """Imports LEAP matlab datasets."""
        self.execute(ImportLEAP)

    def importAnalysisFile(self):
        """Imports SLEAP analysis hdf5 files."""
        self.execute(ImportAnalysisFile)

    def saveProject(self):
        """Show gui to save project (or save as if not yet saved)."""
        self.execute(SaveProject)

    def saveProjectAs(self):
        """Show gui to save project as a new file."""
        self.execute(SaveProjectAs)

    def exportAnalysisFile(self):
        """Shows gui for exporting analysis h5 file."""
        self.execute(ExportAnalysisFile)

    def exportLabeledClip(self):
        """Shows gui for exporting clip with visual annotations."""
        self.execute(ExportLabeledClip)

    def exportUserLabelsPackage(self):
        """Gui for exporting the dataset with user-labeled images."""
        self.execute(ExportUserLabelsPackage)

    def exportTrainingPackage(self):
        """Gui for exporting the dataset with user-labeled images and suggestions."""
        self.execute(ExportTrainingPackage)

    def exportFullPackage(self):
        """Gui for exporting the dataset with any labeled frames and suggestions."""
        self.execute(ExportFullPackage)

    # Navigation Commands

    def previousLabeledFrame(self):
        """Goes to labeled frame prior to current frame."""
        self.execute(GoPreviousLabeledFrame)

    def nextLabeledFrame(self):
        """Goes to labeled frame after current frame."""
        self.execute(GoNextLabeledFrame)

    def nextUserLabeledFrame(self):
        """Goes to next labeled frame with user instances."""
        self.execute(GoNextUserLabeledFrame)

    def lastInteractedFrame(self):
        """Goes to last frame that user interacted with."""
        self.execute(GoLastInteractedFrame)

    def nextSuggestedFrame(self):
        """Goes to next suggested frame."""
        self.execute(GoNextSuggestedFrame)

    def prevSuggestedFrame(self):
        """Goes to previous suggested frame."""
        self.execute(GoPrevSuggestedFrame)

    def addCurrentFrameAsSuggestion(self):
        """Add current frame as a suggestion."""
        self.execute(AddSuggestion)

    def removeSuggestion(self):
        """Remove the selected frame from suggestions."""
        self.execute(RemoveSuggestion)

    def clearSuggestions(self):
        """Clear all suggestions."""
        self.execute(ClearSuggestions)

    def nextTrackFrame(self):
        """Goes to next frame on which a track starts."""
        self.execute(GoNextTrackFrame)

    def gotoFrame(self):
        """Shows gui to go to frame by number."""
        self.execute(GoFrameGui)

    def selectToFrame(self):
        """Shows gui to go to frame by number."""
        self.execute(SelectToFrameGui)

    def gotoVideoAndFrame(self, video: Video, frame_idx: int):
        """Activates video and goes to frame."""
        NavCommand.go_to(self, frame_idx, video)

    # Editing Commands

    def addVideo(self):
        """Shows gui for adding videos to project."""
        self.execute(AddVideo)

    def showImportVideos(self, filenames: List[str]):
        """Show video importer GUI without the file browser."""
        self.execute(ShowImportVideos, filenames=filenames)

    def replaceVideo(self):
        """Shows gui for replacing videos to project."""
        self.execute(ReplaceVideo)

    def removeVideo(self):
        """Removes selected video from project."""
        self.execute(RemoveVideo)

    def openSkeleton(self):
        """Shows gui for loading saved skeleton into project."""
        self.execute(OpenSkeleton)

    def saveSkeleton(self):
        """Shows gui for saving skeleton from project."""
        self.execute(SaveSkeleton)

    def newNode(self):
        """Adds new node to skeleton."""
        self.execute(NewNode)

    def deleteNode(self):
        """Removes (currently selected) node from skeleton."""
        self.execute(DeleteNode)

    def setNodeName(self, skeleton, node, name):
        """Changes name of node in skeleton."""
        self.execute(SetNodeName, skeleton=skeleton, node=node, name=name)

    def setNodeSymmetry(self, skeleton, node, symmetry: str):
        """Sets node symmetry in skeleton."""
        self.execute(SetNodeSymmetry, skeleton=skeleton, node=node, symmetry=symmetry)

    def updateEdges(self):
        """Called when edges in skeleton have been changed."""
        self.signal_update([UpdateTopic.skeleton])

    def newEdge(self, src_node, dst_node):
        """Adds new edge to skeleton."""
        self.execute(NewEdge, src_node=src_node, dst_node=dst_node)

    def deleteEdge(self):
        """Removes (currently selected) edge from skeleton."""
        self.execute(DeleteEdge)

    def deletePredictions(self):
        """Deletes all predicted instances in project."""
        self.execute(DeleteAllPredictions)

    def deleteFramePredictions(self):
        """Deletes all predictions on current frame."""
        self.execute(DeleteFramePredictions)

    def deleteClipPredictions(self):
        """Deletes all predictions within selected range of video frames."""
        self.execute(DeleteClipPredictions)

    def deleteAreaPredictions(self):
        """Gui for deleting instances within some rect on frame images."""
        self.execute(DeleteAreaPredictions)

    def deleteLowScorePredictions(self):
        """Gui for deleting instances below some score threshold."""
        self.execute(DeleteLowScorePredictions)

    def deleteFrameLimitPredictions(self):
        """Gui for deleting instances beyond some number in each frame."""
        self.execute(DeleteFrameLimitPredictions)

    def completeInstanceNodes(self, instance: Instance):
        """Adds missing nodes to given instance."""
        self.execute(AddMissingInstanceNodes, instance=instance)

    def newInstance(
        self,
        copy_instance: Optional[Instance] = None,
        init_method: str = "best",
        location: Optional[QtCore.QPoint] = None,
        mark_complete: bool = False,
    ):
        """
        Creates a new instance, copying node coordinates as appropriate.

        Args:
            copy_instance: The :class:`Instance` (or
                :class:`PredictedInstance`) which we want to copy.
            init_method: Method to use for positioning nodes.
            location: The location where instance should be added (if node init
                method supports custom location).
        """
        self.execute(
            AddInstance,
            copy_instance=copy_instance,
            init_method=init_method,
            location=location,
            mark_complete=mark_complete,
        )

    def setPointLocations(
        self, instance: Instance, nodes_locations: Dict["Node", Tuple[int, int]]
    ):
        """Sets locations for node(s) for an instance."""
        self.execute(
            SetInstancePointLocations,
            instance=instance,
            nodes_locations=nodes_locations,
        )

    def setInstancePointVisibility(
        self, instance: Instance, node: "Node", visible: bool
    ):
        """Toggles visibility set for a node for an instance."""
        self.execute(
            SetInstancePointVisibility, instance=instance, node=node, visible=visible
        )

    def addUserInstancesFromPredictions(self):
        self.execute(AddUserInstancesFromPredictions)

    def deleteSelectedInstance(self):
        """Deletes currently selected instance."""
        self.execute(DeleteSelectedInstance)

    def deleteSelectedInstanceTrack(self):
        """Deletes all instances from track of currently selected instance."""
        self.execute(DeleteSelectedInstanceTrack)

    def deleteDialog(self):
        """Deletes using options selected in a dialog."""
        self.execute(DeleteDialogCommand)

    def addTrack(self):
        """Creates new track and moves selected instance into this track."""
        self.execute(AddTrack)

    def setInstanceTrack(self, new_track: "Track"):
        """Sets track for selected instance."""
        self.execute(SetSelectedInstanceTrack, new_track=new_track)

    def deleteTrack(self, track: "Track"):
        """Delete a track and remove from all instances."""
        self.execute(DeleteTrack, track=track)

    def deleteAllTracks(self):
        """Delete all tracks."""
        self.execute(DeleteAllTracks)

    def setTrackName(self, track: "Track", name: str):
        """Sets name for track."""
        self.execute(SetTrackName, track=track, name=name)

    def transposeInstance(self):
        """Transposes tracks for two instances.

        If there are only two instances, then this swaps tracks.
        Otherwise, it allows user to select the instances for which we want
        to swap tracks.
        """
        self.execute(TransposeInstances)

    def mergeProject(self, filenames: Optional[List[str]] = None):
        """Starts gui for importing another dataset into currently one."""
        self.execute(MergeProject, filenames=filenames)

    def generateSuggestions(self, params: Dict):
        """Generates suggestions using given params dictionary."""
        self.execute(GenerateSuggestions, **params)

    def openWebsite(self, url):
        """Open a website from URL using the native system browser."""
        self.execute(OpenWebsite, url=url)

    def checkForUpdates(self):
        """Check for updates online."""
        self.execute(CheckForUpdates)

    def openStableVersion(self):
        """Open the current stable version."""
        self.execute(OpenStableVersion)

    def openPrereleaseVersion(self):
        """Open the current prerelease version."""
        self.execute(OpenPrereleaseVersion)


# File Commands


class NewProject(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        window = context.app.__class__()
        window.showMaximized()


class OpenProject(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):
        filename = params["filename"]

        do_open_in_new = OPEN_IN_NEW and not params.get("first_open", False)

        # If no project has been loaded in this window and no changes have been
        # made by user, then it's an empty project window so we'll load project
        # into this window rather than creating a new window.
        if not context.state["project_loaded"] and not context.has_any_changes:
            do_open_in_new = False

        if do_open_in_new:
            new_window = context.app.__class__()
            new_window.showMaximized()
            new_window.loadProjectFile(filename)
        else:
            context.app.loadProjectFile(filename)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        if params["filename"] is None:
            filters = [
                "SLEAP HDF5 dataset (*.slp *.h5 *.hdf5)",
                "JSON labels (*.json *.json.zip)",
            ]

            filename, selected_filter = FileDialog.open(
                context.app,
                dir=None,
                caption="Import labeled data...",
                filter=";;".join(filters),
            )

            if len(filename) == 0:
                return False

            params["filename"] = filename
        return True


class ImportDeepPoseKit(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):

        labels = Labels.from_deepposekit(
            filename=params["filename"],
            video_path=params["video_path"],
            skeleton_path=params["skeleton_path"],
        )

        new_window = context.app.__class__()
        new_window.showMaximized()
        new_window.loadLabelsObject(labels=labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        filters = ["HDF5 (*.h5 *.hdf5)"]

        filename, selected_filter = FileDialog.open(
            context.app,
            dir=None,
            caption="Import DeepPoseKit dataset...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        file_dir = os.path.dirname(filename)
        paths = [
            os.path.join(file_dir, "video.mp4"),
            os.path.join(file_dir, "skeleton.csv"),
        ]

        missing = [not os.path.exists(path) for path in paths]

        if sum(missing):
            okay = MissingFilesDialog(filenames=paths, missing=missing).exec_()

            if not okay or sum(missing):
                return False

        params["filename"] = filename
        params["video_path"] = paths[0]
        params["skeleton_path"] = paths[1]

        return True


class ImportLEAP(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):

        labels = Labels.load_leap_matlab(
            filename=params["filename"],
        )

        new_window = context.app.__class__()
        new_window.showMaximized()
        new_window.loadLabelsObject(labels=labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        filters = ["Matlab (*.mat)"]

        filename, selected_filter = FileDialog.open(
            context.app,
            dir=None,
            caption="Import LEAP Matlab dataset...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename

        return True


class ImportCoco(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):

        labels = Labels.load_coco(
            filename=params["filename"], img_dir=params["img_dir"], use_missing_gui=True
        )

        new_window = context.app.__class__()
        new_window.showMaximized()
        new_window.loadLabelsObject(labels=labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        filters = ["JSON (*.json)"]

        filename, selected_filter = FileDialog.open(
            context.app,
            dir=None,
            caption="Import COCO dataset...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename
        params["img_dir"] = os.path.dirname(filename)

        return True


class ImportDeepLabCut(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):

        labels = Labels.load_deeplabcut(filename=params["filename"])

        new_window = context.app.__class__()
        new_window.showMaximized()
        new_window.loadLabelsObject(labels=labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        filters = ["DeepLabCut dataset (*.yaml *.csv)"]

        filename, selected_filter = FileDialog.open(
            context.app,
            dir=None,
            caption="Import DeepLabCut dataset...",
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename

        return True


class ImportDeepLabCutFolder(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):
        csv_files = ImportDeepLabCutFolder.find_dlc_files_in_folder(
            params["folder_name"]
        )
        if csv_files:
            win = MessageDialog(
                f"Importing {len(csv_files)} DeepLabCut datasets...", context.app
            )
            merged_labels = ImportDeepLabCutFolder.import_labels_from_dlc_files(
                csv_files
            )
            win.hide()

            new_window = context.app.__class__()
            new_window.showMaximized()
            new_window.loadLabelsObject(labels=merged_labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        folder_name = FileDialog.openDir(
            context.app,
            dir=None,
            caption="Select a folder with DeepLabCut datasets...",
        )

        if len(folder_name) == 0:
            return False
        params["folder_name"] = folder_name
        return True

    @staticmethod
    def find_dlc_files_in_folder(folder_name: str) -> List[str]:
        return glob(f"{folder_name}/*/*.csv")

    @staticmethod
    def import_labels_from_dlc_files(csv_files: List[str]) -> Labels:
        merged_labels = None
        for csv_file in csv_files:
            labels = Labels.load_file(csv_file, as_format="deeplabcut")
            if merged_labels is None:
                merged_labels = labels
            else:
                merged_labels.extend_from(labels, unify=True)
        return merged_labels


class ImportAnalysisFile(AppCommand):
    @staticmethod
    def do_action(context: "CommandContext", params: dict):
        from sleap.io.format import read

        labels = read(
            params["filename"],
            for_object="labels",
            as_format="analysis",
            video=params["video"],
        )

        new_window = context.app.__class__()
        new_window.showMaximized()
        new_window.loadLabelsObject(labels=labels)

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        filename, selected_filter = FileDialog.open(
            context.app,
            dir=None,
            caption="Import SLEAP Analysis HDF5...",
            filter="SLEAP Analysis HDF5 (*.h5 *.hdf5)",
        )

        if len(filename) == 0:
            return False

        QtWidgets.QMessageBox(text="Please locate the video for this dataset.").exec_()

        video_param_list = ImportVideos().ask()

        if not video_param_list:
            return False

        params["filename"] = filename
        params["video"] = ImportVideos.create_video(video_param_list[0])

        return True


def get_new_version_filename(filename: str) -> str:
    """Increment version number in filenames that end in `.v###.slp`."""
    p = PurePath(filename)

    match = re.match(".*\\.v(\\d+)\\.slp", filename)
    if match is not None:
        old_ver = match.group(1)
        new_ver = str(int(old_ver) + 1).zfill(len(old_ver))
        filename = filename.replace(f".v{old_ver}.slp", f".v{new_ver}.slp")
        filename = str(PurePath(filename))
    else:
        filename = str(p.with_name(f"{p.stem} copy{p.suffix}"))

    return filename


class SaveProjectAs(AppCommand):
    @staticmethod
    def _try_save(context, labels: Labels, filename: str):
        """Helper function which attempts save and handles errors."""
        success = False
        try:
            Labels.save_file(labels=labels, filename=filename)
            success = True
            # Mark savepoint in change stack
            context.changestack_savepoint()

        except Exception as e:
            message = (
                f"An error occured when attempting to save:\n {e}\n\n"
                "Try saving your project with a different filename or in a different "
                "format."
            )
            QtWidgets.QMessageBox(text=message).exec_()

        # Redraw. Not sure why, but sometimes we need to do this.
        context.app.plotFrame()

        return success

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        if cls._try_save(context, context.state["labels"], params["filename"]):
            # If save was successful
            context.state["filename"] = params["filename"]

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        default_name = context.state["filename"]
        if default_name:
            default_name = get_new_version_filename(default_name)
        else:
            default_name = "labels.v000.slp"
        filters = ["SLEAP labels dataset (*.slp)"]
        filename, selected_filter = FileDialog.save(
            context.app,
            caption="Save As...",
            dir=default_name,
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename
        return True


class ExportAnalysisFile(AppCommand):
    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        from sleap.io.format.sleap_analysis import SleapAnalysisAdaptor

        SleapAnalysisAdaptor.write(params["output_path"], context.labels)

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        default_name = context.state["filename"] or "labels"
        p = PurePath(default_name)
        default_name = str(p.with_name(f"{p.stem}.analysis.h5"))

        filename, selected_filter = FileDialog.save(
            context.app,
            caption="Export Analysis File...",
            dir=default_name,
            filter="SLEAP Analysis HDF5 (*.h5)",
        )

        if len(filename) == 0:
            return False

        params["output_path"] = filename
        return True


class SaveProject(SaveProjectAs):
    @classmethod
    def ask(cls, context: CommandContext, params: dict) -> bool:
        if context.state["filename"] is not None:
            params["filename"] = context.state["filename"]
            return True

        # No filename (must be new project), so treat as "Save as"
        return SaveProjectAs.ask(context, params)


def open_file(filename: str):
    """Opens file in native system file browser or registered application.

    Args:
        filename: Path to file or folder.

    Notes:
        Source: https://stackoverflow.com/a/16204023
    """
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


class ExportLabeledClip(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        from sleap.io.visuals import save_labeled_video

        save_labeled_video(
            filename=params["filename"],
            labels=context.state["labels"],
            video=context.state["video"],
            frames=list(params["frames"]),
            fps=params["fps"],
            color_manager=params["color_manager"],
            show_edges=params["show edges"],
            scale=params["scale"],
            crop_size_xy=params["crop"],
            gui_progress=True,
        )

        if params["open_when_done"]:
            # Open the file using default video playing app
            open_file(params["filename"])

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:

        from sleap.gui.dialogs.export_clip import ExportClipDialog

        dialog = ExportClipDialog()

        # Set default fps from video (if video has fps attribute)
        dialog.form_widget.set_form_data(
            dict(fps=getattr(context.state["video"], "fps", 30))
        )

        # Show modal dialog and get form results
        export_options = dialog.get_results()

        # Check if user hit cancel
        if export_options is None:
            return False

        # Use VideoWriter to determine default video type to use
        from sleap.io.videowriter import VideoWriter

        # For OpenCV we default to avi since the bundled ffmpeg
        # makes mp4's that most programs can't open (VLC can).
        default_out_filename = context.state["filename"] + ".avi"

        # But if we can write mpegs using sci-kit video, use .mp4
        # since it has trouble writing .avi files.
        if VideoWriter.can_use_skvideo():
            default_out_filename = context.state["filename"] + ".mp4"

        # Ask where use wants to save video file
        filename, _ = FileDialog.save(
            context.app,
            caption="Save Video As...",
            dir=default_out_filename,
            filter="Video (*.avi *mp4)",
        )

        # Check if user hit cancel
        if len(filename) == 0:
            return False

        params["filename"] = filename
        params["fps"] = export_options["fps"]
        params["scale"] = export_options["scale"]
        params["open_when_done"] = export_options["open_when_done"]

        params["crop"] = None

        # Determine crop size relative to original size and scale
        # (crop size should be *final* output size, thus already scaled).
        w = int(context.state["video"].width * params["scale"])
        h = int(context.state["video"].height * params["scale"])
        if export_options["crop"] == "Half":
            params["crop"] = (w // 2, h // 2)
        elif export_options["crop"] == "Quarter":
            params["crop"] = (w // 4, h // 4)

        if export_options["use_gui_visuals"]:
            params["color_manager"] = context.app.color_manager
        else:
            params["color_manager"] = None

        params["show edges"] = context.state.get("show edges", default=True)

        # If user selected a clip, use that; otherwise include all frames.
        if context.state["has_frame_range"]:
            params["frames"] = range(*context.state["frame_range"])
        else:
            params["frames"] = range(context.state["video"].frames)

        return True


def export_dataset_gui(
    labels: Labels, filename: str, all_labeled: bool = False, suggested: bool = False
) -> str:
    """Export dataset with image data and display progress GUI dialog.

    Args:
        labels: `sleap.Labels` dataset to export.
        filename: Output filename. Should end in `.pkg.slp`.
        all_labeled: If `True`, export all labeled frames, including frames with no user
            instances.
        suggested: If `True`, include image data for suggested frames.
    """
    win = QProgressDialog("Exporting dataset with frame images...", "Cancel", 0, 1)

    def update_progress(n, n_total):
        if win.wasCanceled():
            return False
        win.setMaximum(n_total)
        win.setValue(n)
        win.setLabelText(
            "Exporting dataset with frame images...<br>"
            f"{n}/{n_total} (<b>{(n/n_total)*100:.1f}%</b>)"
        )
        QtWidgets.QApplication.instance().processEvents()
        return True

    Labels.save_file(
        labels,
        filename,
        default_suffix="slp",
        save_frame_data=True,
        all_labeled=all_labeled,
        suggested=suggested,
        progress_callback=update_progress,
    )

    if win.wasCanceled():
        # Delete output if saving was canceled.
        os.remove(filename)
        return "canceled"

    win.hide()

    return filename


class ExportDatasetWithImages(AppCommand):
    all_labeled = False
    suggested = False

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        export_dataset_gui(
            labels=context.state["labels"],
            filename=params["filename"],
            all_labeled=cls.all_labeled,
            suggested=cls.suggested,
        )

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        filters = [
            "SLEAP HDF5 dataset (*.slp *.h5)",
            "Compressed JSON dataset (*.json *.json.zip)",
        ]

        dirname = os.path.dirname(context.state["filename"])
        basename = os.path.basename(context.state["filename"])

        new_basename = f"{os.path.splitext(basename)[0]}.pkg.slp"
        new_filename = os.path.join(dirname, new_basename)

        filename, _ = FileDialog.save(
            context.app,
            caption="Save Labeled Frames As...",
            dir=new_filename,
            filters=";;".join(filters),
        )
        if len(filename) == 0:
            return False

        params["filename"] = filename
        return True


class ExportUserLabelsPackage(ExportDatasetWithImages):
    all_labeled = False
    suggested = False


class ExportTrainingPackage(ExportDatasetWithImages):
    all_labeled = False
    suggested = True


class ExportFullPackage(ExportDatasetWithImages):
    all_labeled = True
    suggested = True


# Navigation Commands


class GoIteratorCommand(AppCommand):
    @staticmethod
    def _plot_if_next(context, frame_iterator: Iterator) -> bool:
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

        context.state["frame_idx"] = next_lf.frame_idx
        return True

    @staticmethod
    def _get_frame_iterator(context: CommandContext):
        raise NotImplementedError("Call to virtual method.")

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        frames = cls._get_frame_iterator(context)
        cls._plot_if_next(context, frames)


class GoPreviousLabeledFrame(GoIteratorCommand):
    @staticmethod
    def _get_frame_iterator(context: CommandContext):
        return context.labels.frames(
            context.state["video"],
            from_frame_idx=context.state["frame_idx"],
            reverse=True,
        )


class GoNextLabeledFrame(GoIteratorCommand):
    @staticmethod
    def _get_frame_iterator(context: CommandContext):
        return context.labels.frames(
            context.state["video"], from_frame_idx=context.state["frame_idx"]
        )


class GoNextUserLabeledFrame(GoIteratorCommand):
    @staticmethod
    def _get_frame_iterator(context: CommandContext):
        frames = context.labels.frames(
            context.state["video"], from_frame_idx=context.state["frame_idx"]
        )
        # Filter to frames with user instances
        frames = filter(lambda lf: lf.has_user_instances, frames)
        return frames


class NavCommand(AppCommand):
    @staticmethod
    def go_to(context, frame_idx: int, video: Optional[Video] = None):
        if video is not None:
            context.state["video"] = video
        context.state["frame_idx"] = frame_idx


class GoLastInteractedFrame(NavCommand):
    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        if context.state["last_interacted_frame"] is not None:
            cls.go_to(
                context,
                frame_idx=context.state["last_interacted_frame"].frame_idx,
                video=context.state["last_interacted_frame"].video,
            )


class GoNextSuggestedFrame(NavCommand):
    seek_direction = 1

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):

        next_suggestion_frame = context.labels.get_next_suggestion(
            context.state["video"], context.state["frame_idx"], cls.seek_direction
        )
        if next_suggestion_frame is not None:
            cls.go_to(
                context, next_suggestion_frame.frame_idx, next_suggestion_frame.video
            )
            selection_idx = context.labels.get_suggestions().index(
                next_suggestion_frame
            )
            context.state["suggestion_idx"] = selection_idx


class GoPrevSuggestedFrame(GoNextSuggestedFrame):
    seek_direction = -1


class GoNextTrackFrame(NavCommand):
    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        video = context.state["video"]
        cur_idx = context.state["frame_idx"]
        track_ranges = context.labels.get_track_occupancy(video)

        later_tracks = [
            (track_range.start, track)
            for track, track_range in track_ranges.items()
            if track_range.start is not None and track_range.start > cur_idx
        ]

        later_tracks.sort(key=operator.itemgetter(0))

        if later_tracks:
            next_idx, next_track = later_tracks[0]
            cls.go_to(context, next_idx)

            # Select the instance in the new track
            lf = context.labels.find(video, next_idx, return_new=True)[0]
            track_instances = [
                inst for inst in lf.instances_to_show if inst.track == next_track
            ]
            if track_instances:
                context.state["instance"] = track_instances[0]


class GoFrameGui(NavCommand):
    @classmethod
    def do_action(cls, context: "CommandContext", params: dict):
        cls.go_to(context, params["frame_idx"])

    @classmethod
    def ask(cls, context: "CommandContext", params: dict) -> bool:
        frame_number, okay = QtWidgets.QInputDialog.getInt(
            context.app,
            "Go To Frame...",
            "Frame Number:",
            context.state["frame_idx"] + 1,
            1,
            context.state["video"].frames,
        )
        params["frame_idx"] = frame_number - 1

        return okay


class SelectToFrameGui(NavCommand):
    @classmethod
    def do_action(cls, context: "CommandContext", params: dict):
        context.app.player.setSeekbarSelection(
            params["from_frame_idx"], params["to_frame_idx"]
        )

    @classmethod
    def ask(cls, context: "CommandContext", params: dict) -> bool:
        frame_number, okay = QtWidgets.QInputDialog.getInt(
            context.app,
            "Select To Frame...",
            "Frame Number:",
            context.state["frame_idx"] + 1,
            1,
            context.state["video"].frames,
        )
        params["from_frame_idx"] = context.state["frame_idx"]
        params["to_frame_idx"] = frame_number - 1

        return okay


# Editing Commands


class EditCommand(AppCommand):
    """Class for commands which change data in project."""

    does_edits = True


class AddVideo(EditCommand):
    topics = [UpdateTopic.video]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        import_list = params["import_list"]

        new_videos = ImportVideos.create_videos(import_list)
        video = None
        for video in new_videos:
            # Add to labels
            context.labels.add_video(video)
            context.changestack_push("add video")

        # Load if no video currently loaded
        if context.state["video"] is None:
            context.state["video"] = video

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        """Shows gui for adding video to project."""
        params["import_list"] = ImportVideos().ask()

        return len(params["import_list"]) > 0


class ShowImportVideos(EditCommand):
    topics = [UpdateTopic.video]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        filenames = params["filenames"]
        import_list = ImportVideos().ask(filenames=filenames)
        new_videos = ImportVideos.create_videos(import_list)
        video = None
        for video in new_videos:
            # Add to labels
            context.labels.add_video(video)
            context.changestack_push("add video")

        # Load if no video currently loaded
        if context.state["video"] is None:
            context.state["video"] = video


class ReplaceVideo(EditCommand):
    topics = [UpdateTopic.video]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        new_paths = params["new_video_paths"]

        for video, new_path in zip(context.labels.videos, new_paths):
            if new_path != video.backend.filename:
                video.backend.filename = new_path
                video.backend.reset()

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        """Shows gui for replacing videos in project."""
        paths = [video.backend.filename for video in context.labels.videos]

        okay = MissingFilesDialog(filenames=paths, replace=True).exec_()

        if not okay:
            return False

        params["new_video_paths"] = paths

        return True


class RemoveVideo(EditCommand):
    topics = [UpdateTopic.video]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        video = params["video"]
        # Remove video
        context.labels.remove_video(video)

        # Update view if this was the current video
        if context.state["video"] == video:
            if len(context.labels.videos) > 0:
                context.state["video"] = context.labels.videos[-1]
            else:
                context.state["video"] = None

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        video = context.state["selected_video"]
        if video is None:
            return False

        # Count labeled frames for this video
        n = len(context.labels.find(video))

        # Warn if there are labels that will be deleted
        if n > 0:
            response = QMessageBox.critical(
                context.app,
                "Removing video with labels",
                f"{n} labeled frames in this video will be deleted, "
                "are you sure you want to remove this video?",
                QMessageBox.Yes,
                QMessageBox.No,
            )
            if response == QMessageBox.No:
                return False

        params["video"] = video
        return True


class OpenSkeleton(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = FileDialog.open(
            context.app, dir=None, caption="Open skeleton...", filter=";;".join(filters)
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename
        return True

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        filename = params["filename"]
        if filename.endswith(".json"):
            context.state["skeleton"] = Skeleton.load_json(filename)
        elif filename.endswith((".h5", ".hdf5")):
            sk_list = Skeleton.load_all_hdf5(filename)
            if len(sk_list):
                context.state["skeleton"] = sk_list[0]

        if context.state["skeleton"] not in context.labels:
            context.labels.skeletons.append(context.state["skeleton"])


class SaveSkeleton(AppCommand):
    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        default_name = "skeleton.json"
        filters = ["JSON skeleton (*.json)", "HDF5 skeleton (*.h5 *.hdf5)"]
        filename, selected_filter = FileDialog.save(
            context.app,
            caption="Save As...",
            dir=default_name,
            filter=";;".join(filters),
        )

        if len(filename) == 0:
            return False

        params["filename"] = filename
        return True

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        filename = params["filename"]
        if filename.endswith(".json"):
            context.state["skeleton"].save_json(filename)
        elif filename.endswith((".h5", ".hdf5")):
            context.state["skeleton"].save_hdf5(filename)


class NewNode(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        # Find new part name
        part_name = "new_part"
        i = 1
        while part_name in context.state["skeleton"]:
            part_name = f"new_part_{i}"
            i += 1

        # Add the node to the skeleton
        context.state["skeleton"].add_node(part_name)


class DeleteNode(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        node = context.state["selected_node"]
        context.state["skeleton"].delete_node(node)


class SetNodeName(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        node = params["node"]
        name = params["name"]
        skeleton = params["skeleton"]

        if name in skeleton.node_names:
            # Merge
            context.labels.merge_nodes(name, node.name)
        else:
            # Simple relabel
            skeleton.relabel_node(node.name, name)


class SetNodeSymmetry(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        node = params["node"]
        symmetry = params["symmetry"]
        skeleton = params["skeleton"]
        if symmetry and node != symmetry:
            skeleton.add_symmetry(node, symmetry)
        else:
            # Value was cleared by user, so delete symmetry
            symmetric_to = skeleton.get_symmetry(node)
            if symmetric_to is not None:
                skeleton.delete_symmetry(node, symmetric_to)


class NewEdge(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        src_node = params["src_node"]
        dst_node = params["dst_node"]

        # Check if they're in the graph
        if (
            src_node not in context.state["skeleton"]
            or dst_node not in context.state["skeleton"]
        ):
            return

        # Add edge
        context.state["skeleton"].add_edge(source=src_node, destination=dst_node)


class DeleteEdge(EditCommand):
    topics = [UpdateTopic.skeleton]

    @staticmethod
    def ask(context: "CommandContext", params: dict) -> bool:
        params["edge"] = context.state["selected_edge"]
        return True

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        edge = params["edge"]
        # Delete edge
        context.state["skeleton"].delete_edge(**edge)


class InstanceDeleteCommand(EditCommand):
    topics = [UpdateTopic.project_instances]

    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        raise NotImplementedError("Call to virtual method.")

    @staticmethod
    def _confirm_deletion(context: CommandContext, lf_inst_list: List) -> bool:
        """Helper function to confirm before deleting instances.

        Args:
            lf_inst_list: A list of (labeled frame, instance) tuples.
        """

        title = "Deleting instances"
        message = (
            f"There are {len(lf_inst_list)} instances which "
            f"would be deleted. Are you sure you want to delete these?"
        )

        # Confirm that we want to delete
        resp = QMessageBox.critical(
            context.app, title, message, QMessageBox.Yes, QMessageBox.No
        )

        if resp == QMessageBox.No:
            return False

        return True

    @staticmethod
    def _do_deletion(context: CommandContext, lf_inst_list: List[int]):
        # Delete the instances
        lfs_to_remove = []
        for lf, inst in lf_inst_list:
            context.labels.remove_instance(lf, inst, in_transaction=True)
            if len(lf.instances) == 0:
                lfs_to_remove.append(lf)

        context.labels.remove_frames(lfs_to_remove)

        # Update caches since we skipped doing this after each deletion
        context.labels.update_cache()

        # Update visuals
        context.changestack_push("delete instances")

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        cls._do_deletion(context, params["lf_instance_list"])

    @classmethod
    def ask(cls, context: CommandContext, params: dict) -> bool:
        lf_inst_list = cls.get_frame_instance_list(context, params)
        params["lf_instance_list"] = lf_inst_list

        return cls._confirm_deletion(context, lf_inst_list)


class DeleteAllPredictions(InstanceDeleteCommand):
    @staticmethod
    def get_frame_instance_list(
        context: CommandContext, params: dict
    ) -> List[Tuple[LabeledFrame, Instance]]:
        return [
            (lf, inst)
            for lf in context.labels
            for inst in lf
            if type(inst) == PredictedInstance
        ]


class DeleteFramePredictions(InstanceDeleteCommand):
    @staticmethod
    def _confirm_deletion(self, *args, **kwargs):
        # Don't require confirmation when deleting from current frame
        return True

    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        predicted_instances = [
            (lf, inst)
            for lf in context.labels.find(
                context.state["video"], frame_idx=context.state["frame_idx"]
            )
            for inst in lf
            if type(inst) == PredictedInstance
        ]

        return predicted_instances


class DeleteClipPredictions(InstanceDeleteCommand):
    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        predicted_instances = [
            (lf, inst)
            for lf in context.labels.find(
                context.state["video"], frame_idx=range(*context.state["frame_range"])
            )
            for inst in lf
            if type(inst) == PredictedInstance
        ]
        return predicted_instances


class DeleteAreaPredictions(InstanceDeleteCommand):
    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        min_corner = params["min_corner"]
        max_corner = params["max_corner"]

        def is_bounded(inst):
            points_array = inst.points_array
            valid_points = points_array[~np.isnan(points_array).any(axis=1)]

            is_gt_min = np.all(valid_points >= min_corner)
            is_lt_max = np.all(valid_points <= max_corner)
            return is_gt_min and is_lt_max

        # Find all instances contained in selected area
        predicted_instances = [
            (lf, inst)
            for lf in context.labels.find(context.state["video"])
            for inst in lf
            if type(inst) == PredictedInstance and is_bounded(inst)
        ]

        return predicted_instances

    @classmethod
    def ask_and_do(cls, context: CommandContext, params: dict):
        # Callback to delete after area has been selected
        def delete_area_callback(x0, y0, x1, y1):
            context.app.updateStatusMessage()

            # Make sure there was an area selected
            if x0 == x1 or y0 == y1:
                return

            params["min_corner"] = (x0, y0)
            params["max_corner"] = (x1, y1)

            predicted_instances = cls.get_frame_instance_list(context, params)

            if cls._confirm_deletion(context, predicted_instances):
                params["lf_instance_list"] = predicted_instances
                cls.do_with_signal(context, params)

        # Prompt the user to select area
        context.app.updateStatusMessage(
            f"Please select the area from which to remove instances. This will be applied to all frames."
        )
        context.app.player.onAreaSelection(delete_area_callback)


class DeleteLowScorePredictions(InstanceDeleteCommand):
    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        score_thresh = params["score_threshold"]
        predicted_instances = [
            (lf, inst)
            for lf in context.labels.find(context.state["video"])
            for inst in lf
            if type(inst) == PredictedInstance and inst.score < score_thresh
        ]
        return predicted_instances

    @classmethod
    def ask(cls, context: CommandContext, params: dict) -> bool:
        score_thresh, okay = QtWidgets.QInputDialog.getDouble(
            context.app, "Delete Instances with Low Score...", "Score Below:", 1, 0, 100
        )
        if okay:
            params["score_threshold"] = score_thresh
            return super().ask(context, params)


class DeleteFrameLimitPredictions(InstanceDeleteCommand):
    @staticmethod
    def get_frame_instance_list(context: CommandContext, params: dict):
        count_thresh = params["count_threshold"]
        predicted_instances = []
        # Find all instances contained in selected area
        for lf in context.labels.find(context.state["video"]):
            if len(lf.predicted_instances) > count_thresh:
                # Get all but the count_thresh many instances with the highest score
                extra_instances = sorted(
                    lf.predicted_instances, key=operator.attrgetter("score")
                )[:-count_thresh]
                predicted_instances.extend([(lf, inst) for inst in extra_instances])
        return predicted_instances

    @classmethod
    def ask(cls, context: CommandContext, params: dict) -> bool:
        count_thresh, okay = QtWidgets.QInputDialog.getInt(
            context.app,
            "Limit Instances in Frame...",
            "Maximum instances in a frame:",
            3,
            1,
            100,
        )
        if okay:
            params["count_threshold"] = count_thresh
            return super().ask(context, params)


class TransposeInstances(EditCommand):
    topics = [UpdateTopic.project_instances, UpdateTopic.tracks]

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        instances = params["instances"]

        if len(instances) != 2:
            return

        # Swap tracks for current and subsequent frames when we have tracks
        old_track, new_track = instances[0].track, instances[1].track
        if old_track is not None and new_track is not None:
            frame_range = (context.state["frame_idx"], context.state["video"].frames)
            context.labels.track_swap(
                context.state["video"], new_track, old_track, frame_range
            )

    @classmethod
    def ask_and_do(cls, context: CommandContext, params: dict):
        def on_each(instances: list):
            word = "next" if len(instances) else "first"
            context.app.updateStatusMessage(
                f"Please select the {word} instance to transpose..."
            )

        def on_success(instances: list):
            params["instances"] = instances
            cls.do_with_signal(context, params)

        if len(context.state["labeled_frame"].instances) < 2:
            return
        # If there are just two instances, transpose them.
        if len(context.state["labeled_frame"].instances) == 2:
            params["instances"] = context.state["labeled_frame"].instances
            cls.do_with_signal(context, params)
        # If there are more than two, then we need the user to select the instances.
        else:
            context.app.player.onSequenceSelect(
                seq_len=2,
                on_success=on_success,
                on_each=on_each,
                on_failure=lambda x: context.app.updateStatusMessage(),
            )


class DeleteSelectedInstance(EditCommand):
    topics = [UpdateTopic.frame, UpdateTopic.project_instances, UpdateTopic.suggestions]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        selected_inst = context.state["instance"]
        if selected_inst is None:
            return

        context.labels.remove_instance(context.state["labeled_frame"], selected_inst)


class DeleteSelectedInstanceTrack(EditCommand):
    topics = [
        UpdateTopic.project_instances,
        UpdateTopic.tracks,
        UpdateTopic.suggestions,
    ]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        selected_inst = context.state["instance"]
        if selected_inst is None:
            return

        track = selected_inst.track
        context.labels.remove_instance(context.state["labeled_frame"], selected_inst)

        if track is not None:
            # remove any instance on this track
            for lf in context.labels.find(context.state["video"]):
                track_instances = filter(lambda inst: inst.track == track, lf.instances)
                for inst in track_instances:
                    context.labels.remove_instance(lf, inst)


class DeleteDialogCommand(EditCommand):
    topics = [
        UpdateTopic.project_instances,
    ]

    @staticmethod
    def ask_and_do(context: CommandContext, params: dict):
        if DeleteDialog(context).exec_():
            context.signal_update([UpdateTopic.project_instances])


class AddTrack(EditCommand):
    topics = [UpdateTopic.tracks]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        track_numbers_used = [
            int(track.name) for track in context.labels.tracks if track.name.isnumeric()
        ]
        next_number = max(track_numbers_used, default=0) + 1
        new_track = Track(spawned_on=context.state["frame_idx"], name=str(next_number))

        context.labels.add_track(context.state["video"], new_track)

        context.execute(SetSelectedInstanceTrack, new_track=new_track)


class SetSelectedInstanceTrack(EditCommand):
    topics = [UpdateTopic.tracks]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        selected_instance = context.state["instance"]
        new_track = params["new_track"]
        if selected_instance is None:
            return

        # When setting track for an instance that doesn't already have a track set,
        # just set for selected instance.
        if (
            selected_instance.track is None
            or not context.state["propagate track labels"]
        ):
            # Move anything already in the new track out of it
            new_track_instances = context.labels.find_track_occupancy(
                video=context.state["video"],
                track=new_track,
                frame_range=(
                    context.state["frame_idx"],
                    context.state["frame_idx"] + 1,
                ),
            )
            for instance in new_track_instances:
                instance.track = None
            # Move selected instance into new track
            context.labels.track_set_instance(
                context.state["labeled_frame"], selected_instance, new_track
            )

        # When the instance does already have a track, then we want to update
        # the track for a range of frames.
        else:
            old_track = selected_instance.track

            # Determine range that should be affected
            if context.state["has_frame_range"]:
                # If range is selected in seekbar, use that
                frame_range = tuple(context.state["frame_range"])
            else:
                # Otherwise, range is current to last frame
                frame_range = (
                    context.state["frame_idx"],
                    context.state["video"].frames,
                )

            # Do the swap
            context.labels.track_swap(
                context.state["video"], new_track, old_track, frame_range
            )

        # Make sure the originally selected instance is still selected
        context.state["instance"] = selected_instance


class DeleteTrack(EditCommand):
    topics = [UpdateTopic.tracks]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        track = params["track"]
        context.labels.remove_track(track)


class DeleteAllTracks(EditCommand):
    topics = [UpdateTopic.tracks]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        context.labels.remove_all_tracks()


class SetTrackName(EditCommand):
    topics = [UpdateTopic.tracks, UpdateTopic.frame]

    @staticmethod
    def do_action(context: CommandContext, params: dict):
        track = params["track"]
        name = params["name"]
        track.name = name


class GenerateSuggestions(EditCommand):
    topics = [UpdateTopic.suggestions]

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):

        # TODO: Progress bar
        win = MessageDialog(
            "Generating list of suggested frames... " "This may take a few minutes.",
            context.app,
        )

        new_suggestions = VideoFrameSuggestions.suggest(
            labels=context.labels, params=params
        )

        context.labels.set_suggestions(new_suggestions)

        win.hide()


class AddSuggestion(EditCommand):
    topics = [UpdateTopic.suggestions]

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        context.labels.add_suggestion(
            context.state["video"], context.state["frame_idx"]
        )


class RemoveSuggestion(EditCommand):
    topics = [UpdateTopic.suggestions]

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        selected_frame = context.app.suggestionsTable.getSelectedRowItem()
        if selected_frame is not None:
            context.labels.remove_suggestion(
                selected_frame.video, selected_frame.frame_idx
            )


class ClearSuggestions(EditCommand):
    topics = [UpdateTopic.suggestions]

    @staticmethod
    def ask(context: CommandContext, params: dict) -> bool:
        if len(context.labels.suggestions) == 0:
            return False

        # Warn that suggestions will be cleared

        response = QMessageBox.warning(
            context.app,
            "Clearing all suggestions",
            "Are you sure you want to remove all suggestions from the project?",
            QMessageBox.Yes,
            QMessageBox.No,
        )
        if response == QMessageBox.No:
            return False

        return True

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        context.labels.clear_suggestions()


class MergeProject(EditCommand):
    topics = [UpdateTopic.all]

    @classmethod
    def ask_and_do(cls, context: CommandContext, params: dict):
        filenames = params["filenames"]
        if filenames is None:
            filters = [
                "SLEAP HDF5 dataset (*.slp *.h5 *.hdf5)",
                "SLEAP JSON dataset (*.json *.json.zip)",
            ]

            filenames, selected_filter = FileDialog.openMultiple(
                context.app,
                dir=None,
                caption="Import labeled data...",
                filter=";;".join(filters),
            )

        if len(filenames) == 0:
            return

        for filename in filenames:
            gui_video_callback = Labels.make_gui_video_callback(
                search_paths=[os.path.dirname(filename)]
            )

            new_labels = Labels.load_file(filename, video_search=gui_video_callback)

            # Merging data is handled by MergeDialog
            MergeDialog(base_labels=context.labels, new_labels=new_labels).exec_()

        cls.do_with_signal(context, params)


class AddInstance(EditCommand):
    topics = [UpdateTopic.frame, UpdateTopic.project_instances, UpdateTopic.suggestions]

    @staticmethod
    def get_previous_frame_index(context: CommandContext) -> Optional[int]:
        frames = context.labels.frames(
            context.state["video"],
            from_frame_idx=context.state["frame_idx"],
            reverse=True,
        )

        try:
            next_idx = next(frames).frame_idx
        except:
            return

        return next_idx

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        copy_instance = params.get("copy_instance", None)
        init_method = params.get("init_method", "best")
        location = params.get("location", None)
        mark_complete = params.get("mark_complete", False)

        if context.state["labeled_frame"] is None:
            return

        if len(context.state["skeleton"]) == 0:
            return

        from_predicted = copy_instance
        from_prev_frame = False

        if init_method == "best" and copy_instance is None:
            selected_inst = context.state["instance"]
            if selected_inst is not None:
                # If the user has selected an instance, copy that one.
                copy_instance = selected_inst
                from_predicted = copy_instance

        if (
            init_method == "best" and copy_instance is None
        ) or init_method == "prediction":
            unused_predictions = context.state["labeled_frame"].unused_predictions
            if len(unused_predictions):
                # If there are predicted instances that don't correspond to an instance
                # in this frame, use the first predicted instance without matching instance.
                copy_instance = unused_predictions[0]
                from_predicted = copy_instance

        if (
            init_method == "best" and copy_instance is None
        ) or init_method == "prior_frame":
            # Otherwise, if there are instances in previous frames,
            # copy the points from one of those instances.
            prev_idx = cls.get_previous_frame_index(context)

            if prev_idx is not None:
                prev_instances = context.labels.find(
                    context.state["video"], prev_idx, return_new=True
                )[0].instances
                if len(prev_instances) > len(context.state["labeled_frame"].instances):
                    # If more instances in previous frame than current, then use the
                    # first unmatched instance.
                    copy_instance = prev_instances[
                        len(context.state["labeled_frame"].instances)
                    ]
                    from_prev_frame = True
                elif init_method == "best" and (
                    context.state["labeled_frame"].instances
                ):
                    # Otherwise, if there are already instances in current frame,
                    # copy the points from the last instance added to frame.
                    copy_instance = context.state["labeled_frame"].instances[-1]
                elif len(prev_instances):
                    # Otherwise use the last instance added to previous frame.
                    copy_instance = prev_instances[-1]
                    from_prev_frame = True

        from_predicted = from_predicted if hasattr(from_predicted, "score") else None

        # Now create the new instance
        new_instance = Instance(
            skeleton=context.state["skeleton"],
            from_predicted=from_predicted,
            frame=context.state["labeled_frame"],
        )

        has_missing_nodes = False

        # go through each node in skeleton
        for node in context.state["skeleton"].node_names:
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
                    complete=mark_complete,
                )
            else:
                has_missing_nodes = True

        if has_missing_nodes:
            # mark the node as not "visible" if we're copying from a predicted instance without this node
            is_visible = copy_instance is None or (not hasattr(copy_instance, "score"))

            if init_method == "force_directed":
                AddMissingInstanceNodes.add_force_directed_nodes(
                    context=context,
                    instance=new_instance,
                    visible=is_visible,
                    center_point=location,
                )
            elif init_method == "random":
                AddMissingInstanceNodes.add_random_nodes(
                    context=context, instance=new_instance, visible=is_visible
                )
            elif init_method == "template":
                AddMissingInstanceNodes.add_nodes_from_template(
                    context=context,
                    instance=new_instance,
                    visible=is_visible,
                    center_point=location,
                )
            else:
                AddMissingInstanceNodes.add_best_nodes(
                    context=context, instance=new_instance, visible=is_visible
                )

        # If we're copying a predicted instance or from another frame, copy the track
        if hasattr(copy_instance, "score") or from_prev_frame:
            new_instance.track = copy_instance.track

        # Add the instance
        context.labels.add_instance(context.state["labeled_frame"], new_instance)

        if context.state["labeled_frame"] not in context.labels.labels:
            context.labels.append(context.state["labeled_frame"])


class SetInstancePointLocations(EditCommand):
    """Sets locations for node(s) for an instance.

    Note: It's important that this command does *not* update the visual
    scene, since this would redraw the frame and create new visual objects.
    The calling code is responsible for updating the visual scene.

    Params:
        instance: The instance
        nodes_locations: A dictionary of data to set
        * keys are nodes (or node names)
        * values are (x, y) coordinate tuples.
    """

    topics = []

    @classmethod
    def do_action(cls, context: "CommandContext", params: dict):
        instance = params["instance"]
        nodes_locations = params["nodes_locations"]

        for node, (x, y) in nodes_locations.items():
            if node in instance:
                instance[node].x = x
                instance[node].y = y


class SetInstancePointVisibility(EditCommand):
    """Toggles visibility set for a node for an instance.

    Note: It's important that this command does *not* update the visual
    scene, since this would redraw the frame and create new visual objects.
    The calling code is responsible for updating the visual scene.

    Params:
        instance: The instance
        node: The `Node` (or name string)
        visible: Whether to set or clear visibility for node
    """

    topics = []

    @classmethod
    def do_action(cls, context: "CommandContext", params: dict):
        instance = params["instance"]
        node = params["node"]
        visible = params["visible"]

        instance[node].visible = visible


class AddMissingInstanceNodes(EditCommand):
    topics = [UpdateTopic.frame]

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        instance = params["instance"]
        visible = params.get("visible", False)

        cls.add_best_nodes(context, instance, visible)

    @classmethod
    def add_best_nodes(cls, context, instance, visible):
        # Try placing missing nodes using a "template" instance
        cls.add_nodes_from_template(context, instance, visible)

        # If the "template" instance has missing nodes (i.e., a node that isn't
        # labeled on any of the instances we used to generate the template),
        # then adding nodes from the template may still result in missing nodes.
        # So we'll use random placement for anything that's still missing.
        cls.add_random_nodes(context, instance, visible)

    @classmethod
    def add_random_nodes(cls, context, instance, visible):
        # TODO: Move this to Instance so we can do this on-demand
        # the rect that's currently visible in the window view
        in_view_rect = context.app.player.getVisibleRect()

        for node in context.state["skeleton"].nodes:
            if node not in instance.nodes or instance[node].isnan():
                # pick random points within currently zoomed view
                x, y = cls.get_xy_in_rect(in_view_rect)
                # set point for node
                instance[node] = Point(x=x, y=y, visible=visible)

    @staticmethod
    def get_xy_in_rect(rect: QtCore.QRectF):
        """Returns random x, y coordinates within given rect."""
        x = rect.x() + (rect.width() * 0.1) + (np.random.rand() * rect.width() * 0.8)
        y = rect.y() + (rect.height() * 0.1) + (np.random.rand() * rect.height() * 0.8)
        return x, y

    @staticmethod
    def get_rect_center_xy(rect: QtCore.QRectF):
        """Returns x, y at center of rect."""

    @classmethod
    def add_nodes_from_template(
        cls,
        context,
        instance,
        visible: bool = False,
        center_point: QtCore.QPoint = None,
    ):
        from sleap.info import align

        # Get the "template" instance
        template_points = context.labels.get_template_instance_points(
            skeleton=instance.skeleton
        )

        # Align the template on to the current instance with missing points
        if instance.points:
            aligned_template = align.align_instance_points(
                source_points_array=template_points,
                target_points_array=instance.points_array,
            )
        else:
            template_mean = np.nanmean(template_points, axis=0)

            center_point = center_point or context.app.player.getVisibleRect().center()
            center = np.array([center_point.x(), center_point.y()])

            aligned_template = template_points + (center - template_mean)

        # Make missing points from the aligned template
        for i, node in enumerate(instance.skeleton.nodes):
            if node not in instance:
                x, y = aligned_template[i]
                instance[node] = Point(x=x, y=y, visible=visible)

    @classmethod
    def add_force_directed_nodes(
        cls, context, instance, visible, center_point: QtCore.QPoint = None
    ):
        import networkx as nx

        center_point = center_point or context.app.player.getVisibleRect().center()
        center_tuple = (center_point.x(), center_point.y())

        node_positions = nx.spring_layout(
            G=context.state["skeleton"].graph, center=center_tuple, scale=50
        )

        for node, pos in node_positions.items():
            instance[node] = Point(x=pos[0], y=pos[1], visible=visible)


class AddUserInstancesFromPredictions(EditCommand):
    topics = [UpdateTopic.frame, UpdateTopic.project_instances]

    @staticmethod
    def make_instance_from_predicted_instance(
        copy_instance: PredictedInstance,
    ) -> Instance:
        # create the new instance
        new_instance = Instance(
            skeleton=copy_instance.skeleton,
            from_predicted=copy_instance,
            frame=copy_instance.frame,
        )

        # go through each node in skeleton
        for node in new_instance.skeleton.node_names:
            # if we're copying from a skeleton that has this node
            if node in copy_instance and not copy_instance[node].isnan():
                # just copy x, y, and visible
                # we don't want to copy a PredictedPoint or score attribute
                new_instance[node] = Point(
                    x=copy_instance[node].x,
                    y=copy_instance[node].y,
                    visible=copy_instance[node].visible,
                    complete=False,
                )

        # copy the track
        new_instance.track = copy_instance.track

        return new_instance

    @classmethod
    def do_action(cls, context: CommandContext, params: dict):
        if context.state["labeled_frame"] is None:
            return

        new_instances = []
        unused_predictions = context.state["labeled_frame"].unused_predictions
        for predicted_instance in unused_predictions:
            new_instances.append(
                cls.make_instance_from_predicted_instance(predicted_instance)
            )

        # Add the instances
        for new_instance in new_instances:
            context.labels.add_instance(context.state["labeled_frame"], new_instance)


def open_website(url: str):
    """Open website in default browser.

    Args:
        url: URL to open.
    """
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


class OpenWebsite(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        open_website(params["url"])


class CheckForUpdates(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        success = context.app.release_checker.check_for_releases()
        if success:
            stable = context.app.release_checker.latest_stable
            prerelease = context.app.release_checker.latest_prerelease
            context.state["stable_version_menu"].setText(f"  Stable: {stable.version}")
            context.state["stable_version_menu"].setEnabled(True)
            context.state["prerelease_version_menu"].setText(
                f"  Prerelease: {prerelease.version}"
            )
            context.state["prerelease_version_menu"].setEnabled(True)

    # TODO: Provide GUI feedback about result.


class OpenStableVersion(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        rls = context.app.release_checker.latest_stable
        if rls is not None:
            context.openWebsite(rls.url)


class OpenPrereleaseVersion(AppCommand):
    @staticmethod
    def do_action(context: CommandContext, params: dict):
        rls = context.app.release_checker.latest_prerelease
        if rls is not None:
            context.openWebsite(rls.url)


def copy_to_clipboard(text: str):
    """Copy a string to the system clipboard.

    Args:
        text: String to copy to clipboard.
    """
    clipboard = QtWidgets.QApplication.clipboard()
    clipboard.clear(mode=clipboard.Clipboard)
    clipboard.setText(text, mode=clipboard.Clipboard)

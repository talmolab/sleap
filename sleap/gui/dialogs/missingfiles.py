"""
Gui for prompting the user to locate one or more missing files.
"""

import os

from pathlib import Path, PurePath
from typing import Callable, List

from qtpy import QtWidgets, QtCore, QtGui
from sleap_io.io.video_reading import ImageVideo

from sleap.io import pathutils
from sleap.gui.dialogs.filedialog import FileDialog


def verify_sequence_frames(frame_paths: List[str], new_dir: str) -> List[str]:
    """Verify which frames from a sequence exist in a new directory.

    Uses ImageVideo.find_images() to efficiently find available images
    in the target directory.

    Args:
        frame_paths: Original list of frame file paths.
        new_dir: Directory to check for frames.

    Returns:
        List of frame filenames that are missing from the new directory.
    """
    # Get all images available in the new directory
    available_images = set(Path(p).name for p in ImageVideo.find_images(new_dir))

    # Check which frames are missing
    missing_frames = []
    for frame_path in frame_paths:
        frame_name = Path(frame_path).name
        if frame_name not in available_images:
            missing_frames.append(frame_name)

    return missing_frames


class MissingFilesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        filenames: List[str],
        missing: List[bool] = None,
        replace: bool = False,
        allow_incomplete: bool = False,
        is_sequence: List[bool] = None,
        original_filenames: List = None,
        *args,
        **kwargs,
    ):
        """
        Creates dialog window for finding missing files.

        Any changes made by user will be reflected in filenames list.

        Args:
            filenames: List of filenames to find, needn't all be missing.
                For image sequences, this should contain the first frame path
                (for display purposes and distinguishability).
            missing: Corresponding list, whether each file is missing. If
                not given, then we'll check whether each file exists.
            replace: Whether we are replacing files (already found) or
                locating files (not already found). Affects text in dialog.
            allow_incomplete: Whether to enable "accept" button when there
                are still missing files.
            is_sequence: List indicating which entries are image sequences.
                If None, all entries are treated as regular video files.
            original_filenames: For image sequences, contains the full list
                of frame paths. Used for remapping frames to new directory.

        Returns:
            None.
        """

        super(MissingFilesDialog, self).__init__(*args, **kwargs)

        if not missing:
            missing = pathutils.list_file_missing(filenames)

        # Initialize sequence tracking
        self.is_sequence = is_sequence or [False] * len(filenames)
        self.original_filenames = original_filenames or filenames

        self.filenames = filenames
        self.missing = missing
        self.replace = replace

        missing_count = sum(missing)

        layout = QtWidgets.QVBoxLayout()

        if replace:
            info_text = "Double-click on a file to replace it..."
        else:
            info_text = (
                f"{missing_count} file(s) which could not be found. "
                "Please double-click on a file to locate it..."
            )
        info_label = QtWidgets.QLabel(info_text)
        layout.addWidget(info_label)

        self.file_table = MissingFileTable(filenames, missing)
        self.file_table.doubleClicked.connect(_qt_row_index_call(self.locateFile))
        layout.addWidget(self.file_table)

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Abort", QtWidgets.QDialogButtonBox.RejectRole)
        self.accept_button = buttons.addButton(
            "Continue", QtWidgets.QDialogButtonBox.AcceptRole
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        if not allow_incomplete:
            self.accept_button.setEnabled(False)

        layout.addWidget(buttons)

        self.setLayout(layout)

    def locateFile(self, idx: int):
        """Shows dialog for user to locate a specific missing file.

        For image sequences, the user selects the first frame file.
        The directory is extracted and all frames are remapped to that directory.
        All frames are verified to exist before marking as found.
        """
        old_filename = self.filenames[idx]
        _, old_ext = os.path.splitext(old_filename)

        if self.is_sequence[idx]:
            # Image sequence: user selects first frame file
            caption = f"Please locate first frame: {Path(old_filename).name}..."
            filters = [f"Image files (*{old_ext})", "Any File (*.*)"]
        else:
            caption = f"Please locate {old_filename}..."
            filters = [f"Missing file type (*{old_ext})", "Any File (*.*)"]

        filters = [filters[0]] if self.replace else filters
        new_filename, _ = FileDialog.open(
            None, dir=None, caption=caption, filter=";;".join(filters)
        )

        if not new_filename:
            return

        path_new_filename = Path(new_filename)

        if self.is_sequence[idx]:
            # Image sequence: verify all frames exist in the new directory
            new_dir = path_new_filename.parent
            original_frames = self.original_filenames[idx]

            if not isinstance(original_frames, list):
                original_frames = [original_frames]

            # Use ImageVideo.find_images() to efficiently verify frames
            missing_frames = verify_sequence_frames(original_frames, str(new_dir))

            if missing_frames:
                # Warn user that not all frames were found
                total_frames = len(original_frames)
                missing_count = len(missing_frames)
                sample_missing = missing_frames[:3]
                sample_str = ", ".join(sample_missing)
                if missing_count > 3:
                    sample_str += f"... and {missing_count - 3} more"

                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Frames",
                    f"Not all frames found in directory.\n\n"
                    f"Missing {missing_count} of {total_frames} frames:\n"
                    f"{sample_str}\n\n"
                    f"Please select a directory containing all frames.",
                )
                return

            # All frames found - update the display filename (first frame path)
            self.setFilename(idx, str(new_dir / Path(old_filename).name))
        else:
            # Regular video file
            paths = [str(PurePath(fn)) for fn in self.filenames]
            if str(path_new_filename) in paths:
                # Do not allow same video to be imported more than once.
                QtWidgets.QMessageBox(
                    text=(
                        f"The file <b>{path_new_filename.name}</b> cannot be added to "
                        "the project multiple times."
                    )
                ).exec_()
                return

            # Try using this change to find other missing files
            self.setFilename(idx, new_filename)

        # Redraw the table
        self.file_table.reset()

    def setFilename(self, idx: int, filename: str, confirm: bool = True):
        """Applies change after user finds missing file.

        For image sequences, auto-prefix verifies all frames exist before applying.
        """
        old_filename = self.filenames[idx]

        self.filenames[idx] = filename
        self.missing[idx] = False

        old_prefix, new_prefix = pathutils.find_changed_subpath(
            old_filename, filename
        )

        # See if we can apply same change to find other missing files.
        # We'll ask for confirmation for making these changes.
        confirm_callback = None
        if confirm:

            def confirm_callback():
                return self.confirmAutoReplace(old_prefix, new_prefix)

        # Apply prefix change - sequences will have all frames verified
        pathutils.filenames_prefix_change(
            self.filenames,
            old_prefix,
            new_prefix,
            self.missing,
            confirm_callback,
            is_sequence=self.is_sequence,
            original_filenames=self.original_filenames,
        )

        # If there are no missing files still, enable the "accept" button
        if sum(self.missing) == 0:
            self.accept_button.setEnabled(True)

    def confirmAutoReplace(self, old, new):
        message = (
            f"Other missing files can be found by replacing\n\n"
            f"{old}\n\nwith\n\n{new}\n\nWould you like to apply this "
            f"change?"
        )

        response = QtWidgets.QMessageBox.question(
            self,
            "Apply change to other paths",
            message,
            QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        return response == QtWidgets.QMessageBox.Yes

    def finish(self):
        self.accept()


def _qt_row_index_call(funct: Callable):
    def decorated_funct(qt_idx):
        if not qt_idx.isValid():
            return
        return funct(qt_idx.row())

    return decorated_funct


class MissingFileTable(QtWidgets.QTableView):
    """
    Qt table view for missing files.

    Arguments are passed through to the table view object.
    """

    def __init__(self, *args, **kwargs):
        super(MissingFileTable, self).__init__()
        self.setModel(MissingFileTableModel(*args, **kwargs))
        self.resizeColumnsToContents()

    def reset(self):
        super(MissingFileTable, self).reset()
        self.resizeColumnsToContents()


class MissingFileTableModel(QtCore.QAbstractTableModel):
    """Qt table model for missing files.

    Args:
        filenames: Filenames to show, needn't all be missing.
        missing: Corresponding list, whether each file is missing.
    """

    _props = ["filename"]

    def __init__(self, filenames: List[str], missing: List[bool]):
        super(MissingFileTableModel, self).__init__()
        self.filenames = filenames
        self.missing = missing

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Required by Qt."""
        if not index.isValid():
            return None

        idx = index.row()
        prop = self._props[index.column()]

        if idx >= self.rowCount():
            return None

        if role == QtCore.Qt.DisplayRole:
            if prop == "filename":
                return self.filenames[idx]

        elif role == QtCore.Qt.ForegroundRole:
            return QtGui.QColor("red") if self.missing[idx] else None

        return None

    def rowCount(self, *args):
        """Required by Qt."""
        return len(self.filenames)

    def columnCount(self, *args):
        """Required by Qt."""
        return len(self._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None

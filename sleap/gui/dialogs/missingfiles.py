"""
Gui for prompting the user to locate one or more missing files.
"""

import os

from typing import Callable, List

from PySide2 import QtWidgets, QtCore, QtGui

from sleap.io import pathutils
from sleap.gui.dialogs.filedialog import FileDialog


class MissingFilesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        filenames: List[str],
        missing: List[bool] = None,
        replace: bool = False,
        allow_incomplete: bool = False,
        *args,
        **kwargs,
    ):
        """
        Creates dialog window for finding missing files.

        Any changes made by user will be reflected in filenames list.

        Args:
            filenames: List of filenames to find, needn't all be missing.
            missing: Corresponding list, whether each file is missing. If
                not given, then we'll check whether each file exists.
            replace: Whether we are replacing files (already found) or
                locating files (not already found). Affects text in dialog.
            allow_incomplete: Whether to enable "accept" button when there
                are still missing files.

        Returns:
            None.
        """

        super(MissingFilesDialog, self).__init__(*args, **kwargs)

        if not missing:
            missing = pathutils.list_file_missing(filenames)

        self.filenames = filenames
        self.missing = missing

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
        """Shows dialog for user to locate a specific missing file."""
        old_filename = self.filenames[idx]
        _, old_ext = os.path.splitext(old_filename)

        caption = f"Please locate {old_filename}..."
        filters = [f"Missing file type (*{old_ext})", "Any File (*.*)"]
        new_filename, _ = FileDialog.open(
            None, dir=None, caption=caption, filter=";;".join(filters)
        )

        if new_filename:
            # Try using this change to find other missing files
            self.setFilename(idx, new_filename)

            # Redraw the table
            self.file_table.reset()

    def setFilename(self, idx: int, filename: str, confirm: bool = True):
        """Applies change after user finds missing file."""
        old_filename = self.filenames[idx]

        self.filenames[idx] = filename
        self.missing[idx] = False

        old_prefix, new_prefix = pathutils.find_changed_subpath(old_filename, filename)

        # See if we can apply same change to find other missing files.
        # We'll ask for confirmation for making these changes.
        confirm_callback = None
        if confirm:
            confirm_callback = lambda: self.confirmAutoReplace(old_prefix, new_prefix)

        pathutils.filenames_prefix_change(
            self.filenames, old_prefix, new_prefix, self.missing, confirm_callback
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


# if __name__ == "__main__":
#     app = QtWidgets.QApplication()
#     win = MissingFilesDialog(["m:/centered_pair_small.mp4", "m:/small_robot.mp4"])
#     result = win.exec_()
#     print(result)

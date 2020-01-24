"""
Gui for keyboard shortcuts.
"""
from PySide2 import QtWidgets
from PySide2.QtGui import QKeySequence

import yaml

from typing import Dict, List, Union

from sleap import util


class ShortcutDialog(QtWidgets.QDialog):
    """
    Dialog window for reviewing and modifying the keyboard shortcuts.
    """

    _column_len = 13

    def __init__(self, *args, **kwargs):
        super(ShortcutDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Keyboard Shortcuts")
        self.load_shortcuts()
        self.make_form()

    def accept(self):
        """Triggered when form is accepted; saves the shortcuts."""
        for action, widget in self.key_widgets.items():
            self.shortcuts[action] = widget.keySequence().toString()
        self.shortcuts.save()

        super(ShortcutDialog, self).accept()

    def load_shortcuts(self):
        """Loads shortcuts object."""
        self.shortcuts = Shortcuts()

    def make_form(self):
        """Creates the form with fields for all shortcuts."""
        self.key_widgets = dict()  # dict to store QKeySequenceEdit widgets

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.make_shortcuts_widget())
        layout.addWidget(
            QtWidgets.QLabel(
                "Any changes to keyboard shortcuts will not take effect "
                "until you quit and re-open the application."
            )
        )
        layout.addWidget(self.make_buttons_widget())
        self.setLayout(layout)

    def make_buttons_widget(self) -> QtWidgets.QDialogButtonBox:
        """Makes the form buttons."""
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        return buttons

    def make_shortcuts_widget(self) -> QtWidgets.QWidget:
        """Makes the widget will fields for all shortcuts."""
        shortcuts = self.shortcuts

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        # show shortcuts in columns
        for a in range(0, len(shortcuts), self._column_len):
            b = min(len(shortcuts), a + self._column_len)
            column_widget = self.make_column_widget(shortcuts[a:b])
            layout.addWidget(column_widget)
        widget.setLayout(layout)
        return widget

    def make_column_widget(self, shortcuts: List) -> QtWidgets.QWidget:
        """Makes a single column of shortcut fields.

        Args:
            shortcuts: The list of shortcuts to include in this column.

        Returns:
            The widget.
        """
        column_widget = QtWidgets.QWidget()
        column_layout = QtWidgets.QFormLayout()
        for action in shortcuts:
            item = QtWidgets.QKeySequenceEdit(shortcuts[action])
            column_layout.addRow(action.title(), item)
            self.key_widgets[action] = item
        column_widget.setLayout(column_layout)
        return column_widget


class Shortcuts(object):
    """
    Class for accessing keyboard shortcuts.

    Shortcuts are saved in `sleap/config/shortcuts.yaml`

    When instantiated, this reads in the shortcuts from the file.
    """

    _shortcuts = None
    _names = (
        "new",
        "open",
        "save",
        "save as",
        "close",
        "add videos",
        "next video",
        "prev video",
        "goto frame",
        "mark frame",
        "goto marked",
        "add instance",
        "delete instance",
        "delete track",
        "transpose",
        "select next",
        "clear selection",
        "goto next labeled",
        "goto prev labeled",
        "goto next user",
        "goto next suggestion",
        "goto prev suggestion",
        "goto next track spawn",
        "show labels",
        "show edges",
        "show trails",
        "color predicted",
        "fit",
        "learning",
        "export clip",
        "delete frame predictions",
        "delete clip predictions",
        "delete area predictions",
    )

    def __init__(self):
        shortcuts = util.get_config_yaml("shortcuts.yaml")

        for action in shortcuts:
            key_string = shortcuts.get(action, None)
            key_string = "" if key_string is None else key_string

            if not key_string.strip():
                shortcuts[action] = ""
                continue

            try:
                shortcuts[action] = eval(key_string)
            except:
                shortcuts[action] = QKeySequence.fromString(key_string)

        self._shortcuts = shortcuts

    def save(self):
        """Saves all shortcuts to shortcut file."""
        util.save_config_yaml("shortcuts.yaml", self._shortcuts)

    def __getitem__(self, idx: Union[slice, int, str]) -> Union[str, Dict[str, str]]:
        """
        Returns shortcut value, accessed by range, index, or key.

        Args:
            idx: Index (range, int, or str) of shortcut to access.

        Returns:
            If idx is int or string, then return value is the shortcut string.
            If idx is range, then return value is dictionary in which keys
            are shortcut name and value are shortcut strings.
        """
        if isinstance(idx, slice):
            # dict with names and values
            return {self._names[i]: self[i] for i in range(*idx.indices(len(self)))}
        elif isinstance(idx, int):
            # value
            idx = self._names[idx]
            return self[idx]
        else:
            # value
            if idx in self._shortcuts:
                return self._shortcuts[idx]
        return ""

    def __setitem__(self, idx: Union[str, int], val: str):
        """Sets shortcut by index."""
        if type(idx) == int:
            idx = self._names[idx]
            self[idx] = val
        else:
            self._shortcuts[idx] = val

    def __len__(self):
        """Returns number of shortcuts."""
        return len(self._names)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    win = ShortcutDialog()
    win.show()
    app.exec_()

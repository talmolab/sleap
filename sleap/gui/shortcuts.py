from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import Qt
from PySide2.QtGui import QKeySequence

import sys
import yaml

from pkg_resources import Requirement, resource_filename


class ShortcutDialog(QtWidgets.QDialog):

    _column_len = 13

    def __init__(self, *args, **kwargs):
        super(ShortcutDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Keyboard Shortcuts")
        self.load_shortcuts()
        self.make_form()

    def accept(self):
        for action, widget in self.key_widgets.items():
            self.shortcuts[action] = widget.keySequence().toString()
        self.shortcuts.save()

        super(ShortcutDialog, self).accept()

    def load_shortcuts(self):
        self.shortcuts = Shortcuts()

    def make_form(self):
        self.key_widgets = dict()  # dict to store QKeySequenceEdit widgets

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.make_shortcuts_widget())
        layout.addWidget(
            QtWidgets.QLabel(
                "Any changes to keyboard shortcuts will not take effect until you quit and re-open the application."
            )
        )
        layout.addWidget(self.make_buttons_widget())
        self.setLayout(layout)

    def make_buttons_widget(self):
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        return buttons

    def make_shortcuts_widget(self):
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

    def make_column_widget(self, shortcuts):
        column_widget = QtWidgets.QWidget()
        column_layout = QtWidgets.QFormLayout()
        for action in shortcuts:
            item = QtWidgets.QKeySequenceEdit(shortcuts[action])
            column_layout.addRow(action.title(), item)
            self.key_widgets[action] = item
        column_widget.setLayout(column_layout)
        return column_widget


def dict_cut(d, a, b):
    return dict(list(d.items())[a:b])


class Shortcuts:

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
        "delete clip",
        "delete area",
    )

    def __init__(self):
        shortcut_yaml = resource_filename(
            Requirement.parse("sleap"), "sleap/config/shortcuts.yaml"
        )
        with open(shortcut_yaml, "r") as f:
            shortcuts = yaml.load(f, Loader=yaml.SafeLoader)

        for action in shortcuts:
            key_string = shortcuts.get(action, None)
            key_string = "" if key_string is None else key_string

            try:
                shortcuts[action] = eval(key_string)
            except:
                shortcuts[action] = QKeySequence.fromString(key_string)

        self._shortcuts = shortcuts

    def save(self):
        shortcut_yaml = resource_filename(
            Requirement.parse("sleap"), "sleap/config/shortcuts.yaml"
        )
        with open(shortcut_yaml, "w") as f:
            yaml.dump(self._shortcuts, f)

    def __getitem__(self, idx):
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

    def __setitem__(self, idx, val):
        if type(idx) == int:
            idx = self._names[idx]
            self[idx] = val
        else:
            self._shortcuts[idx] = val

    def __len__(self):
        return len(self._names)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    win = ShortcutDialog()
    win.show()
    app.exec_()

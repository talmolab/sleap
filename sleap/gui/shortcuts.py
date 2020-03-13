"""
Class for accessing/setting keyboard shortcuts.
"""

from typing import Dict, Union

from PySide2.QtGui import QKeySequence

from sleap import util


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
    from PySide2 import QtWidgets

    from sleap.gui.dialogs.shortcuts import ShortcutDialog

    app = QtWidgets.QApplication()
    win = ShortcutDialog()
    win.show()
    app.exec_()

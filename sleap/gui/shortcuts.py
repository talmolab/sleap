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
        "select to frame",
        "add instance",
        "delete instance",
        "delete track",
        "transpose",
        "select next",
        "clear selection",
        "goto next labeled",
        "goto prev labeled",
        "goto last interacted",
        "goto next user",
        "goto next suggestion",
        "goto prev suggestion",
        "goto next track spawn",
        "show instances",
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
        "frame next",
        "frame prev",
        "frame next medium step",
        "frame prev medium step",
        "frame next large step",
        "frame prev large step",
    )

    def __init__(self):
        shortcuts = util.get_config_yaml("shortcuts.yaml")
        defaults = util.get_config_yaml("shortcuts.yaml", get_defaults=True)

        self._shortcuts = self._process_shortcut_dict(shortcuts)
        self._defaults = self._process_shortcut_dict(defaults)

    def _process_shortcut_dict(self, shortcuts: dict) -> dict:
        for action in shortcuts.keys():

            # Ignore shortcuts which aren't in currently supported list
            if action not in self._names:
                continue

            # Use "" if there's no shortcut set
            key_string = shortcuts.get(action, None)
            key_string = "" if key_string is None else key_string

            if not key_string.strip():
                shortcuts[action] = ""
                continue

            try:
                shortcuts[action] = eval(key_string)
            except:
                shortcuts[action] = QKeySequence.fromString(key_string)
        return shortcuts

    def save(self):
        """Saves all shortcuts to shortcut file."""
        data = dict()
        for key, val in self._shortcuts.items():
            # Only save shortcuts with names in supported list
            if key not in self._names:
                continue

            data[key] = val.toString() if isinstance(val, QKeySequence) else val

        util.save_config_yaml("shortcuts.yaml", data)

    def reset_to_default(self):
        """Reset shortcuts to default and save."""
        self._shortcuts = util.get_config_yaml("shortcuts.yaml", get_defaults=True)
        self.save()

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

        # if idx not in self._names:
        #     print(f"No shortcut matching '{idx}'")

        return self._shortcuts.get(idx, self._defaults.get(idx, ""))

    def __setitem__(self, idx: Union[str, int], val: str):
        """Sets shortcut by index."""
        if type(idx) == int:
            key = self._names[idx]
            self[key] = val
        else:
            if idx not in self._names:
                raise KeyError(f"No shortcut matching '{idx}'")

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

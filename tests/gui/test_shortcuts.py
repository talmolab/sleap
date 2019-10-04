from PySide2.QtGui import QKeySequence

from sleap.gui.shortcuts import Shortcuts


def test_shortcuts():
    shortcuts = Shortcuts()

    assert shortcuts["new"] == shortcuts[0]
    assert shortcuts["new"] == QKeySequence.fromString("Ctrl+N")
    shortcuts["new"] = QKeySequence.fromString("Ctrl+Shift+N")
    assert shortcuts["new"] == QKeySequence.fromString("Ctrl+Shift+N")
    assert list(shortcuts[0:2].keys()) == ["new", "open"]

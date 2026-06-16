from qtpy.QtGui import QKeySequence

from sleap.gui.shortcuts import Shortcuts


def test_shortcuts():
    shortcuts = Shortcuts()

    assert shortcuts["new"] == shortcuts[0]
    assert shortcuts["new"] == QKeySequence.fromString("Ctrl+N")
    shortcuts["new"] = QKeySequence.fromString("Ctrl+Shift+N")
    assert shortcuts["new"] == QKeySequence.fromString("Ctrl+Shift+N")
    assert list(shortcuts[0:2].keys()) == ["new", "open"]

    # "propagate track labels" toggle shortcut (discussion #1638)
    assert shortcuts["propagate track labels"] == QKeySequence.fromString("P")


def test_show_non_visible_nodes_shortcut():
    # "show non-visible nodes" toggle shortcut (#2781)
    shortcuts = Shortcuts()

    assert shortcuts["show non-visible nodes"] == QKeySequence.fromString("Shift+V")
    assert shortcuts["show non-visible nodes"] != QKeySequence()


def test_no_duplicate_shortcuts():
    """No two named actions may share the same non-empty key sequence."""
    shortcuts = Shortcuts()

    seen = set()
    for i in range(len(shortcuts)):
        value = shortcuts[i]
        # Unbound shortcuts come back as "" (str); bound ones as QKeySequence.
        key_string = value.toString() if isinstance(value, QKeySequence) else value
        if not key_string:
            continue
        assert key_string not in seen, f"Duplicate shortcut: {key_string}"
        seen.add(key_string)

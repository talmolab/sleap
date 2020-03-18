import os

from PySide2 import QtWidgets

from sleap.gui.dialogs.filedialog import FileDialog


def test_non_native_dialog():
    d = dict()
    FileDialog._non_native_if_set(d)
    assert "options" not in d

    os.environ["USE_NON_NATIVE_FILE"] = "1"

    FileDialog._non_native_if_set(d)
    assert d["options"] == QtWidgets.QFileDialog.DontUseNativeDialog

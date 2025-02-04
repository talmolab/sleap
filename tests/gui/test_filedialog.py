import os
import sys

from qtpy import QtWidgets

from sleap.gui.dialogs.filedialog import os_specific_method, FileDialog


def test_non_native_dialog():
    @os_specific_method
    def dummy_function(cls, *args, **kwargs):
        """This function returns the `kwargs` modified by the wrapper.

        Args:
            cls: The `FileDialog` class.

        Returns:
            kwargs: Modified by the wrapper.
        """
        return kwargs

    FileDialog.dummy_function = dummy_function
    save_env_non_native = os.environ.get("USE_NON_NATIVE_FILE", None)
    os.environ["USE_NON_NATIVE_FILE"] = ""
    d = dict()

    # Wrapper doesn't mutate `d` outside of scope, so need to return `modified_d`
    modified_d = FileDialog.dummy_function(FileDialog, d)
    is_linux = sys.platform.startswith("linux")
    if is_linux:
        assert modified_d["options"] == QtWidgets.QFileDialog.DontUseNativeDialog
    else:
        assert "options" not in modified_d

    os.environ["USE_NON_NATIVE_FILE"] = "1"
    modified_d = FileDialog.dummy_function(FileDialog, d)
    assert modified_d["options"] == QtWidgets.QFileDialog.DontUseNativeDialog

    if save_env_non_native is not None:
        os.environ["USE_NON_NATIVE_FILE"] = save_env_non_native

"""
Wrappers for Qt File Dialogs.

The main improvement is logic which determines whether to use native or non-
native file dialogs. Native dialogs are usually better but don't work correctly
on (some?) Ubuntu systems.
"""

import os, re, sys

from functools import wraps
from pathlib import Path
from typing import Callable
from qtpy import QtWidgets


def os_specific_method(func) -> Callable:
    """Check if native dialog should be used and update kwargs based on OS.

    Native Mac/Win file dialogs add file extension based on selected file type but
    non-native dialog (used for Linux) does not do this by default.
    """

    @wraps(func)
    def set_dialog_type(cls, *args, **kwargs):
        is_linux = sys.platform.startswith("linux")
        env_var_set = os.environ.get("USE_NON_NATIVE_FILE", False)
        cls.is_non_native = is_linux or env_var_set

        if cls.is_non_native:
            kwargs["options"] = kwargs.get("options", 0)
            if not kwargs["options"]:
                kwargs["options"] = QtWidgets.QFileDialog.DontUseNativeDialog

        # Make sure we don't send empty options argument
        if "options" in kwargs and not kwargs["options"]:
            del kwargs["options"]

        return func(cls, *args, **kwargs)

    return set_dialog_type


class FileDialog:
    """Substitute for QFileDialog; see class methods for details."""

    is_non_native = False

    @classmethod
    @os_specific_method
    def open(cls, *args, **kwargs):
        """
        Wrapper for `QFileDialog.getOpenFileName()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        return QtWidgets.QFileDialog.getOpenFileName(*args, **kwargs)

    @classmethod
    @os_specific_method
    def openMultiple(cls, *args, **kwargs):
        """
        Wrapper for `QFileDialog.getOpenFileNames()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        return QtWidgets.QFileDialog.getOpenFileNames(*args, **kwargs)

    @classmethod
    @os_specific_method
    def save(cls, *args, **kwargs):
        """Wrapper for `QFileDialog.getSaveFileName()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """

        # The non-native file dialog doesn't add file extensions from the
        # file-type menu in the dialog, so we need to do this ourselves.
        if cls.is_non_native and "filter" in kwargs and "dir" in kwargs:
            filename = kwargs["dir"]
            filters = kwargs["filter"].split(";;")
            if filters:
                if ".slp" in filters[0] and not filename.endswith(".slp"):
                    kwargs["dir"] = f"{filename}.slp"

        filename, filter = QtWidgets.QFileDialog.getSaveFileName(*args, **kwargs)

        # Make sure filename has appropriate file extension.
        if cls.is_non_native and filter:
            fn = Path(filename)
            # Get extension from filter as list of "*.ext"
            match = re.findall("\*(\.[a-zA-Z0-9]+)", filter)
            if len(match) > 0:
                # Add first filter extension if none of the filter extensions match
                add_extension = True
                for filter_ext in reversed(match):
                    if fn.suffix == filter_ext:
                        add_extension = False
                if add_extension:
                    filename = f"{filename}{filter_ext}"

        return filename, filter

    @classmethod
    @os_specific_method
    def openDir(cls, *args, **kwargs):
        """Wrapper for `QFileDialog.getExistingDirectory()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        return QtWidgets.QFileDialog.getExistingDirectory(*args, **kwargs)

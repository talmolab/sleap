"""
Wrappers for Qt File Dialogs.

The main improvement is logic which determines whether to use native or non-
native file dialogs. Native dialogs are usually better but don't work correctly
on (some?) Ubuntu systems.
"""

import os, re, sys
from pathlib import Path

from qtpy import QtWidgets


class FileDialog:
    """Substitute for QFileDialog; see class methods for details."""

    @classmethod
    def open(cls, *args, **kwargs):
        """
        Wrapper for `QFileDialog.getOpenFileName()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        cls._non_native_if_set(kwargs)
        return QtWidgets.QFileDialog.getOpenFileName(*args, **kwargs)

    @classmethod
    def openMultiple(cls, *args, **kwargs):
        """
        Wrapper for `QFileDialog.getOpenFileNames()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        cls._non_native_if_set(kwargs)
        return QtWidgets.QFileDialog.getOpenFileNames(*args, **kwargs)

    @classmethod
    def save(cls, *args, **kwargs):
        """Wrapper for `QFileDialog.getSaveFileName()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        is_non_native = cls._non_native_if_set(kwargs)

        # The non-native file dialog doesn't add file extensions from the
        # file-type menu in the dialog, so we need to do this ourselves.
        if is_non_native and "filter" in kwargs and "dir" in kwargs:
            filename = kwargs["dir"]
            filters = kwargs["filter"].split(";;")
            if filters:
                if ".slp" in filters[0] and not filename.endswith(".slp"):
                    kwargs["dir"] = f"{filename}.slp"

        filename, filter = QtWidgets.QFileDialog.getSaveFileName(*args, **kwargs)

        # Make sure filename has appropriate file extension.
        if is_non_native and filter:
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
    def openDir(cls, *args, **kwargs):
        """Wrapper for `QFileDialog.getExistingDirectory()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        return QtWidgets.QFileDialog.getExistingDirectory(*args, **kwargs)

    @staticmethod
    def _non_native_if_set(kwargs) -> bool:
        is_non_native = False
        is_linux = sys.platform.startswith("linux")
        env_var_set = os.environ.get("USE_NON_NATIVE_FILE", False)

        if is_linux or env_var_set:
            is_non_native = True
            kwargs["options"] = kwargs.get("options", 0)
            kwargs["options"] |= QtWidgets.QFileDialog.DontUseNativeDialog

        # Make sure we don't send empty options argument
        if "options" in kwargs and not kwargs["options"]:
            del kwargs["options"]

        return is_non_native

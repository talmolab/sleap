"""
Wrappers for Qt File Dialogs.
"""

import os, sys

from PySide2 import QtWidgets


class FileDialog(object):
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
        cls._non_native_if_set(kwargs)
        return QtWidgets.QFileDialog.getSaveFileName(*args, **kwargs)

    @classmethod
    def openDir(cls, *args, **kwargs):
        """Wrapper for `QFileDialog.getExistingDirectory()`

        Uses non-native file dialog if USE_NON_NATIVE_FILE env var set.

        Passes along everything except empty "options" arg.
        """
        return QtWidgets.QFileDialog.getExistingDirectory(*args, **kwargs)

    @staticmethod
    def _non_native_if_set(kwargs):
        is_linux = sys.platform.startswith("linux")
        env_var_set = os.environ.get("USE_NON_NATIVE_FILE", False)

        if is_linux or env_var_set:
            kwargs["options"] = kwargs.get("options", 0)
            kwargs["options"] |= QtWidgets.QFileDialog.DontUseNativeDialog

        # Make sure we don't send empty options argument
        if "options" in kwargs and not kwargs["options"]:
            del kwargs["options"]

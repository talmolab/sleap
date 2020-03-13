"""
Module to show a non-blocking modal dialog box with a string message.
"""

from PySide2 import QtWidgets, QtCore


class MessageDialog(QtWidgets.QDialog):
    def __init__(self, message, *args, **kwargs):
        super(MessageDialog, self).__init__(*args, **kwargs)

        layout = QtWidgets.QFormLayout()
        layout.addRow(message, None)
        self.setLayout(layout)
        self.setModal(True)
        self.show()

        # Hacky but the text doesn't show unless we call processEvents a few times.
        QtWidgets.QApplication.instance().processEvents()
        QtWidgets.QApplication.instance().processEvents()
        QtWidgets.QApplication.instance().processEvents()

"""
GUI for displaying the new announcement.
"""

from typing import List
from qtpy import QtWidgets


class BulletinDialog(QtWidgets.QDialog):
    """
    Dialog window to display the announcement.
    """

    app: "MainWindow"

    def __init__(self, *args, **kwargs):
        super(BulletinDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Announcement")
        self.info_msg()

    def info_msg(self):
        """Display information about changes."""
        msg = QtWidgets.QMessageBox()
        information = self.app.state["announcement"]
        msg.setText(information)
        msg.exec_()

"""
Generic module to ask user permission to complete an action.
"""

from PySide2.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
)


class QueryDialog(QDialog):
    """Opens `QDialog` to ask user permission to complete an action.

    Args:
        title: Text to be displayed in the header of dialog box.
        message: Test to be displayed in the body of dialog box.
    """

    def __init__(self, title: str, message: str, *args, **kwargs):
        """Initialize and display `QDialog`."""
        super().__init__(*args, **kwargs)

        self.user_response = False

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

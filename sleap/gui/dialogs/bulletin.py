"""
GUI for displaying the new announcement.
"""

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QMainWindow,
    QHBoxLayout,
    QWidget,
    QLabel,
)
from qtpy.QtCore import QObject, Signal, Slot, QThread, Property


class BulletinWorker(QThread):
    text_updated = Signal(str)

    def __init__(self, content, parent=None):
        super(BulletinWorker, self).__init__(parent)
        self.content = content

    def run(self):
        self.text_updated.emit(self.content)


class BulletinDialog(QDialog):
    def __init__(self, parent=None):
        super(BulletinDialog, self).__init__(parent)

        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    @Slot(str)
    def updateText(self, text):
        self.label.setText(text)

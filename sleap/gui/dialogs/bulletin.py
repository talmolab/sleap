"""
GUI for displaying the new announcement.
"""

import os
import sleap
import sleap.gui.web
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Property, Signal, QObject, QUrl
from qtpy.QtWebChannel import QWebChannel
from qtpy.QtWebEngineWidgets import QWebEngineView
from sleap.gui.commands import CommandContext
from sleap.io.dataset import Labels


class BulletinDialog(QObject):
    textChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_text = ""

    def get_text(self):
        return self.m_text

    def set_text(self, text):
        if self.m_text == text:
            return
        self.m_text = text
        self.textChanged.emit(self.m_text)

    text = Property(str, fget=get_text, fset=set_text, notify=textChanged)

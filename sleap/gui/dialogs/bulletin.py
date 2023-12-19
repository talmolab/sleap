"""
GUI for displaying the new announcement.
"""

import os 
# os.environ['QT_API'] = 'pyside6'
from qtpy.QtCore import Signal, Qt
from qtpy import QtWebEngineWidgets #import QWebEngineView
from qtpy.QtCore import Property, Signal, QObject, QUrl
from qtpy.QtWebChannel import QWebChannel
from qtpy import QtWidgets
from pathlib import Path


class BulletinWorker(QtWidgets.QMainWindow):
    def __init__(self, content, parent=None):
        super(BulletinWorker, self).__init__(parent)
        self._content = content
        # Set the window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

    def show_bulletin(self):

        self.document = Document()

        # Set the webchannel
        self.channel = QWebChannel()
        self.channel.registerObject("content", self.document)

        self.document.set_text(self._content)
        self.view = QtWebEngineWidgets.QWebEngineView()
        self.view.page().setWebChannel(self.channel)

        filename = str(Path(__file__).resolve().parent / "bulletin/markdown.html")
        url = QUrl.fromLocalFile(filename)
        self.view.load(url)

        # Set the central window with view
        self.setCentralWidget(self.view)
        self.show()


class Document(QObject):
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

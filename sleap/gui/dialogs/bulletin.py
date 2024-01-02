"""
GUI for displaying the new announcement.
"""

from qtpy.QtCore import Signal, Qt
from qtpy.QtWebEngineWidgets import QWebEngineView
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
        self.setWindowTitle("What's New?")
        self.setGeometry(0, 0, 900, 750)
        self.center_on_screen()

    def center_on_screen(self):
        # Get the screen geometry
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()

        # Calculate the center of the screen
        center_x = (screen_geometry.width() - self.width()) // 2
        center_y = (screen_geometry.height() - self.height()) // 2

        # Move the window to the center of the screen
        self.move(center_x, center_y)

    def show_bulletin(self):

        self.document = Document()

        # Set the webchannel
        self.channel = QWebChannel()
        self.channel.registerObject("content", self.document)

        self.document.set_text(self._content)
        self.view = QWebEngineView()
        self.view.page().setWebChannel(self.channel)

        filename = str(Path(__file__).resolve().parent / "bulletin/markdown.html")
        url = QUrl.fromLocalFile(filename)
        self.view.load(url)

        self.view.setGeometry(0, 0, 600, 400)
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


# class MyWebEngineView(QWebEngineView):
#     def createWindow(self, type):
#         new_view = MyWebEngineView(self)
#         new_view.show()
#         return new_view

#     def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
#         if navigation_type == QWebEnginePage.NavigationTypeLinkClicked:
#             # Emit the linkClicked signal when a link is clicked
#             self.page().mainFrame().javaScriptWindowObjectCleared.connect(
#                 lambda: self.page().mainFrame().addToJavaScriptWindowObject("linkHandler", self)
#             )
#             self.page().runJavaScript("linkHandler.linkClicked('%s');" % url.toString())
#             return False
#         return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

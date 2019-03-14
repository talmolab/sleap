from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PySide2.QtWidgets import QDockWidget
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QMenu, QAction

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

from sleap.gui.video import VideoPlayer, QtVideoPlayer

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # self.video = VideoPlayer()
        self.video = QtVideoPlayer()
        self.setCentralWidget(self.video)

        self.menuBar().addMenu("File")
        viewMenu = self.menuBar().addMenu("View")
        self.menuBar().addMenu("Help")


        project_dock = QDockWidget("Project")
        project_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        project_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Test"))
        layout.addWidget(QPushButton("Test"))
        project_widget.setLayout(layout)
        project_dock.setWidget(project_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, project_dock)
        viewMenu.addAction(project_dock.toggleViewAction())

        skeleton_dock = QDockWidget("Skeleton")
        skeleton_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        skeleton_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Test"))
        layout.addWidget(QPushButton("Test"))
        skeleton_widget.setLayout(layout)
        skeleton_dock.setWidget(skeleton_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, skeleton_dock)
        viewMenu.addAction(skeleton_dock.toggleViewAction())


        self.show()




if __name__ == "__main__":

    app = QApplication([])
    app.setApplicationName("sLEAP Label")
    window = MainWindow()

    from sleap.io.video import HDF5Video
    window.video.video = HDF5Video("C:/code/sleap/tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5", "/box", input_format="channels_first")
    window.video.plot()

    app.exec_()


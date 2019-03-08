from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PySide2.QtWidgets import QAction

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np


class VideoPlayer(QWidget):
    def __init__(self, video=None, *args, **kwargs):
        super(VideoPlayer, self).__init__(*args, **kwargs)

        self.video = video

        self.figure = plt.figure()
        self.ax = self.figure.add_axes([0,0,1,1])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.canvas = FigureCanvas(self.figure)

        # set the layout
        self.layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        # self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.plot()

    def plot(self):

        if self.video is not None:
            frame = self.video.get_frame(0)
        else:
            frame = np.zeros((2,2), dtype="uint8")

        # self.figure.clear()

        self.img = self.ax.imshow(frame.squeeze(), cmap="gray")

        self.canvas.draw()




if __name__ == "__main__":


    from sleap.io.video import HDF5Video

    vid = HDF5Video("C:/code/sleap/tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5", "/box", input_format="channels_first")

    app = QApplication([])
    # app.setApplicationName("sLEAP Label")
    window = VideoPlayer(video=vid)
    window.show()
    app.exec_()


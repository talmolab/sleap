from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PySide2.QtWidgets import QAction

from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtGui import QImage, QPixmap, QPainterPath, QPen
from PySide2.QtCore import Qt, QRectF, Signal
# from PySide2.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np


class VideoPlayer(QWidget):
    def __init__(self, video=None, *args, **kwargs):
        super(VideoPlayer, self).__init__(*args, **kwargs)

        self.video = video

        # https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
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




class GraphicsView(QGraphicsView):

    leftMouseButtonPressed = Signal(float, float)
    rightMouseButtonPressed = Signal(float, float)
    leftMouseButtonReleased = Signal(float, float)
    rightMouseButtonReleased = Signal(float, float)
    leftMouseButtonDoubleClicked = Signal(float, float)
    rightMouseButtonDoubleClicked = Signal(float, float)

    def __init__(self, *args, **kwargs):
        """ https://github.com/marcel-goldschen-ohm/PyQtImageViewer/blob/master/QtImageViewer.py """
        QGraphicsView.__init__(self)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self._pixmapHandle = None

        self.aspectRatioMode = Qt.KeepAspectRatio
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.zoomStack = []
        self.canZoom = True
        self.canPan = True

        self.zoomFactor = 1
        anchor_mode = QGraphicsView.AnchorUnderMouse
        # anchor_mode = QGraphicsView.AnchorViewCenter
        self.setTransformationAnchor(anchor_mode)

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
        else:
            self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).
            self.scale(self.zoomFactor, self.zoomFactor)
            # TODO: fix this so that it's a single operation
            #   Maybe adjust the self.sceneRect() to account for zooming?
            # print(self.mapFromScene(self.sceneRect()))

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()


    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self.canPan:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.setDragMode(QGraphicsView.RubberBandDrag)
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()
                selectionBBox = self.scene.selectionArea().boundingRect().intersected(viewBBox)
                self.scene.setSelectionArea(QPainterPath())  # Clear current selection area.
                if selectionBBox.isValid() and (selectionBBox != viewBBox):
                    self.zoomStack.append(selectionBBox)
                    self.updateViewer()
            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.zoomStack = []  # Clear zoom stack.
                self.updateViewer()
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def wheelEvent(self, event):

        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9

        self.zoomFactor = max(factor * self.zoomFactor, 1)
        self.updateViewer()

        # transform = self.transform()

        # print(self.sceneRect())
        # print(self.transform())

        # scale = max(transform.m11(), transform.m22())
        # if scale * factor < 1.0:
        #     factor = 1.0

        # self.scale(factor, factor)

        # print()


class QtVideoPlayer(QWidget):
    def __init__(self, video=None, *args, **kwargs):
        super(QtVideoPlayer, self).__init__(*args, **kwargs)

        self.video = video

        # self.scene = QGraphicsScene()
        # self.view = QGraphicsView(self.scene)
        self.view = GraphicsView()

        btn = QPushButton("Plot")
        btn.clicked.connect(self.plot)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.layout.addWidget(btn)
        self.setLayout(self.layout)
        self.view.show()
        self.plot()

        # https://stackoverflow.com/questions/19113532/qgraphicsview-zooming-in-and-out-under-mouse-position-using-mouse-wheel

    def plot(self):

        if self.video is not None:
            frame = self.video.get_frame(np.random.randint(0,len(self.video)))
        else:
            frame = np.zeros((2,2), dtype="uint8")

        # https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
        # https://stackoverflow.com/questions/55063499/pyqt5-convert-cv2-image-to-qimage
        image = QImage(frame.copy().data, frame.shape[1], frame.shape[0], frame.shape[1], QImage.Format_Grayscale8)
        # TODO: handle RGB and other formats
        self.view.setImage(image)

        pen = QPen(Qt.red, 3)
        lineItem = self.view.scene.addLine(100, 100, 250, 250, pen)
        lineItem.setFlag(QGraphicsItem.ItemIsMovable)
        # https://stackoverflow.com/questions/36689957/movable-qgraphicslineitem-bounding-box
        # https://doc.qt.io/qtforpython/PySide2/QtWidgets/QGraphicsItem.html#PySide2.QtWidgets.QGraphicsItem
        # https://doc.qt.io/qtforpython/overviews/qtwidgets-graphicsview-dragdroprobot-example.html#drag-and-drop-robot-example




if __name__ == "__main__":


    from sleap.io.video import HDF5Video

    vid = HDF5Video("C:/code/sleap/tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5", "/box", input_format="channels_first")

    app = QApplication([])
    # app.setApplicationName("sLEAP Label")
    # window = VideoPlayer(video=vid)
    window = QtVideoPlayer(video=vid)
    window.show()
    app.exec_()


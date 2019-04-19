from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QLabel, QPushButton, QSlider
from PySide2.QtWidgets import QAction

from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtGui import QImage, QPixmap, QPainter, QPainterPath
from PySide2.QtGui import QPen, QBrush, QColor
from PySide2.QtGui import QKeyEvent
from PySide2.QtCore import Qt, Signal, Slot
from PySide2.QtCore import QRectF, QLineF, QPointF
# from PySide2.QtCore import pyqtSignal

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt

import numpy as np

from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
# The PySide2.QtWidgets.QGraphicsObject class provides a base class for all graphics items that require signals, slots and properties.
from PySide2.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point
from sleap.io.video import Video, HDF5Video


import qimage2ndarray

class VideoPlayer(QWidget):
    def __init__(self, video: Video = None, *args, **kwargs):
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
        # brush = QBrush(QColor.black())
        self.scene.setBackgroundBrush(QBrush(QColor(Qt.black)))

        self._pixmapHandle = None

        self.setRenderHint(QPainter.Antialiasing)
        # self.setCacheMode(QGraphicsView.CacheNone)

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

        # self.scene.render()

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clear(self):
        self._pixmapHandle = None
        self.scene.clear()

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            # pixmap = QPixmap.fromImage(image)
            pixmap = QPixmap(image)
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

            rect = self.sceneRect()
            # print(self.transform())
            # print(rect)
            # print(self.rect())
            self.fitInView(rect, self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).
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

        # https://stackoverflow.com/questions/19113532/qgraphicsview-zooming-in-and-out-under-mouse-position-using-mouse-wheel
        # 

    def keyPressEvent(self, event):
        event.ignore() # Kicks the event up to parent

    def keyReleaseEvent(self, event):
        event.ignore() # Kicks the event up to parent


class QtVideoPlayer(QWidget):
    def __init__(self, video: Video = None, callbacks=[], *args, **kwargs):
        super(QtVideoPlayer, self).__init__(*args, **kwargs)

        self.frame_idx = -1
        self.callbacks = callbacks
        self.view = GraphicsView()

        # btn = QPushButton("Plot")
        # btn.clicked.connect(lambda x: self.plot(np.random.randint(0,len(self.video))))
        
        self.seekbar = QSlider(Qt.Horizontal)
        self.seekbar.valueChanged.connect(lambda evt: self.plot(self.seekbar.value()))
        self.seekbar.setEnabled(False)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.layout.addWidget(self.seekbar)
        # self.layout.addWidget(btn)
        self.setLayout(self.layout)
        self.view.show()

        if video is not None:
            self.load_video(video)
        
    def load_video(self, video: Video, initial_frame=0, plot=True):

        self.video = video
        self.frame_idx = initial_frame

        # Is this necessary?
        self.view.scene.setSceneRect(0, 0, video.width, video.height)

        # self.seekbar.setTickInterval(1)
        self.seekbar.setValue(self.frame_idx)
        self.seekbar.setMinimum(0)
        self.seekbar.setMaximum(self.video.frames - 1)
        self.seekbar.setEnabled(True)

        if plot:
            self.plot(initial_frame)

    def reset(self):
        self.video = None
        self.frame_idx = None
        self.view.clear()
        self.seekbar.setMaximum(0)
        self.seekbar.setEnabled(False)
        

    def plot(self, idx=None):

        if self.video is None:
            return

        # Refresh by default
        if idx is None:
            idx = self.frame_idx

        # Get image data
        frame = self.video.get_frame(idx)

        # Update index
        self.frame_idx = idx
        self.seekbar.setValue(self.frame_idx)

        # Clear existing objects
        self.view.clear()

        # Convert ndarray to QImage
        # TODO: handle RGB and other formats
        # https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
        # https://stackoverflow.com/questions/55063499/pyqt5-convert-cv2-image-to-qimage
        # image = QImage(frame.copy().data, frame.shape[1], frame.shape[0], frame.shape[1], QImage.Format_Grayscale8)
        # image = QImage(frame.copy().data, frame.shape[1], frame.shape[0], QImage.Format_Grayscale8)

        # Magic bullet:
        image = qimage2ndarray.array2qimage(frame)

        # Display image
        self.view.setImage(image)

        # Handle callbacks
        for callback in self.callbacks:
            callback(self, idx)

    def nextFrame(self, dt=1):
        self.plot((self.frame_idx + abs(dt)) % self.video.frames)

    def prevFrame(self, dt=1):
        self.plot((self.frame_idx - abs(dt)) % self.video.frames)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Left:
            self.prevFrame()
        elif event.key() == Qt.Key.Key_Right:
            self.nextFrame()
        elif event.key() == Qt.Key.Key_Home:
            self.plot(0)
        elif event.key() == Qt.Key.Key_End:
            self.plot(self.video.frames - 1)
        else:
            event.ignore() # Kicks the event up to parent
            # print(event.key())

class QtNode(QGraphicsEllipseItem):
    # pointUpdated = Signal(Point)

    def __init__(self, parent, point:Point, radius=1.5, node_name:str = None, *args, **kwargs):
        self.point = point
        self.radius = radius
        self.edges = []
        self.dragParent = False

        super(QtNode, self).__init__(-self.radius, -self.radius, self.radius*2, self.radius*2, parent=parent, *args, **kwargs)
        self.setPos(self.point.x, self.point.y)
        if node_name is not None:
            self.setToolTip(node_name)

    def updatePoint(self):
        self.point.x = self.scenePos().x()
        self.point.y = self.scenePos().y()

        for edge in self.edges:
            edge.updateEdge(self)

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.MetaModifier:
            self.dragParent = True
            self.parentObject().mousePressEvent(event)
        else:
            self.dragParent = False
            super(QtNode, self).mousePressEvent(event)
            self.updatePoint()

    def mouseMoveEvent(self, event):
        #print(event)
        if self.dragParent:
            self.parentObject().mouseMoveEvent(event)
        else:
            super(QtNode, self).mouseMoveEvent(event)
            self.updatePoint()

    def mouseReleaseEvent(self, event):
        #print(event)
        if self.dragParent:
            self.parentObject().mouseReleaseEvent(event)
            self.parentObject().setSelected(False)
            self.parentObject().updatePoints()
        else:
            super(QtNode, self).mouseReleaseEvent(event)
            self.updatePoint()


class QtEdge(QGraphicsLineItem):
    def __init__(self, parent, src:QtNode, dst:QtNode, *args, **kwargs):
        self.src = src
        self.dst = dst

        super(QtEdge, self).__init__(self.src.point.x, self.src.point.y, self.dst.point.x, self.dst.point.y, parent=parent, *args, **kwargs)
        
    def updateEdge(self, node):
        if node == self.src:
            line = self.line()
            line.setP1(node.scenePos())
            self.setLine(line)

        elif node == self.dst: 
            line = self.line()
            line.setP2(node.scenePos())
            self.setLine(line)


class QtInstance(QGraphicsObject):
    def __init__(self, skeleton:Skeleton = None, instance: Instance = None, color=(0, 114, 189), markerRadius=1.5, *args, **kwargs):
        super(QtInstance, self).__init__(*args, **kwargs)
        self.skeleton = skeleton if instance is None else instance.skeleton
        self.instance = instance
        self.nodes = {}
        self.edges = []
        self.color = color
        self.markerRadius = markerRadius
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        
        col_line = QColor(*self.color)
        pen = QPen(col_line, 1)
        pen.setCosmetic(True) # https://stackoverflow.com/questions/13120486/adjusting-qpen-thickness-when-scaling-qgraphicsview

        pen_missing = QPen(col_line, 1)
        pen_missing.setCosmetic(True)

        col_fill = QColor(*self.color, a=128)
        brush = QBrush(col_fill)

        col_fill_missing = QColor(*self.color, a=0)
        brush_missing = QBrush(col_fill_missing)

        for (node, point) in self.instance.nodes_points():
            if point.visible:
                node_item = QtNode(parent=self, point=point, radius=self.markerRadius, node_name = node.name)
                node_item.setPen(pen)
                node_item.setBrush(brush)
            else:
                node_item = QtNode(parent=self, point=point, radius=self.markerRadius * 0.5, node_name = node.name)
                node_item.setPen(pen_missing)
                node_item.setBrush(brush_missing)
            node_item.setFlag(QGraphicsItem.ItemIsMovable)

            self.nodes[node.name] = node_item

        for (src, dst) in self.skeleton.edge_names:
            # Make sure that both nodes are present in this instance before drawing edge
            if src in self.nodes and dst in self.nodes:
                edge_item = QtEdge(parent=self, src=self.nodes[src], dst=self.nodes[dst])
                edge_item.setPen(pen)
                self.nodes[src].edges.append(edge_item)
                self.nodes[dst].edges.append(edge_item)
                self.edges.append(edge_item)

    def updatePoints(self):
        # Update the position for each node
        for node_item in self.nodes.values():
            node_item.point.x = node_item.scenePos().x()
            node_item.point.y = node_item.scenePos().y()
            node_item.setPos(node_item.point.x, node_item.point.y)
        # Reset the scene position (changes when we drag entire skeleton)
        self.setPos(0, 0)
        # Update the position for each edge
        for edge_item in self.edges:
            edge_item.updateEdge(edge_item.src)
            edge_item.updateEdge(edge_item.dst)

    def boundingRect(self):
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass


if __name__ == "__main__":

    import h5py

    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    vid = HDF5Video(data_path, "/box", input_format="channels_first")

    app = QApplication([])
    # app.setApplicationName("sLEAP Label")
    # window = VideoPlayer(video=vid)
    window = QtVideoPlayer(video=vid)

    # lines(7)*255
    cmap = np.array([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ])

    skeleton = Skeleton("Fly")
    skeleton.add_node(name="head")
    skeleton.add_node(name="neck")
    skeleton.add_node(name="thorax")
    skeleton.add_node(name="abdomen")
    skeleton.add_node(name="left-wing")
    skeleton.add_node(name="right-wing")
    skeleton.add_edge(source="head", destination="neck")
    skeleton.add_edge(source="neck", destination="thorax")
    skeleton.add_edge(source="thorax", destination="abdomen")
    skeleton.add_edge(source="thorax", destination="left-wing")
    skeleton.add_edge(source="thorax", destination="right-wing")
    # skeleton.add_symmetry(node1="left-wing", node2="right-wing")
    node_names = list(skeleton.graph.nodes)

    scale = 0.5
    with h5py.File(data_path, "r") as f:
        # skeleton = Skeleton.load_hdf5(f["skeleton"])
        frames = {k: f["frames"][k][:].flatten() for k in ["videoId", "frameIdx"]}
        points = {k: f["points"][k][:].flatten() for k in ["id", "frameIdx", "instanceId", "x", "y", "node", "visible"]}

    # points["frameIdx"] -= 1
    points["x"] *= scale
    points["y"] *= scale
    points["node"] = points["node"].astype("uint8") - 1
    points["visible"] = points["visible"].astype("bool")

    def plot_instances(vp, idx):

        # Find instances in frame idx
        is_in_frame = points["frameIdx"] == frames["frameIdx"][idx]
        if not is_in_frame.any():
            return

        frame_instance_ids = np.unique(points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (points["instanceId"] == instance_id)
            instance_points = {node_names[n]: Point(x, y, visible=v) for x, y, n, v in
                                            zip(*[points[k][is_instance] for k in ["x", "y", "node", "visible"]])
                                            if n < len(node_names)}

            # Plot instance
            instance = Instance(skeleton=skeleton, points=instance_points)
            qt_instance = QtInstance(instance=instance, color=cmap[i%len(cmap)])
            vp.view.scene.addItem(qt_instance)

    window.callbacks.append(plot_instances)


    window.show()
    window.plot()
    
    app.exec_()


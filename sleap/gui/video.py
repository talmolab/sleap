"""
Module for showing and manipulating skeleton instances within a video.

All interactions should go through `QtVideoPlayer`.

Example usage:
    >>> my_video = Video(...)
    >>> my_instance = Instance(...)
    >>> color = (r, g, b)

    >>> vp = QtVideoPlayer(video = my_video)
    >>> vp.addInstance(instance = my_instance, color)
"""

from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QLabel, QPushButton, QSlider
from PySide2.QtWidgets import QAction

from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtGui import QImage, QPixmap, QPainter, QPainterPath, QTransform
from PySide2.QtGui import QPen, QBrush, QColor, QFont
from PySide2.QtGui import QKeyEvent
from PySide2.QtCore import Qt, Signal, Slot
from PySide2.QtCore import QRectF, QLineF, QPointF, QMarginsF, QSizeF

import math
import numpy as np

from typing import Callable

from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
# The PySide2.QtWidgets.QGraphicsObject class provides a base class for all graphics items that require signals, slots and properties.
from PySide2.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsRectItem

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point
from sleap.io.video import Video, HDF5Video
from sleap.gui.slider import VideoSlider

import qimage2ndarray


class QtVideoPlayer(QWidget):
    """
    Main QWidget for displaying video with skeleton instances.

    Args:
        video (optional): the :class:`Video` to display
        
    Signals:
        changedPlot: Emitted whenever the plot is redrawn
    """

    changedPlot = Signal(QWidget, int)

    def __init__(self, video: Video = None, *args, **kwargs):
        super(QtVideoPlayer, self).__init__(*args, **kwargs)

        self.frame_idx = -1
        self.view = GraphicsView()

        self.seekbar = VideoSlider()
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
        """
        Load video into viewer.

        Args:
            video: the :class:`Video` to display
            initial_frame (optional): display this frame of video
            plot: If True, plot the video frame. Otherwise, just load the data.
        """

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
        """ Reset viewer by removing all video data.
        """
        self.video = None
        self.frame_idx = None
        self.view.clear()
        self.seekbar.setMaximum(0)
        self.seekbar.setEnabled(False)

    @property
    def instances(self):
        return self.view.instances
        
    def addInstance(self, instance, **kwargs):
        """Add a skeleton instance to the video.
        
        Args:
            instance: this can be either a `QtInstance` or an `Instance`
            
            Any other named args are passed along if/when creating QtInstance.
        """
        if type(instance) == Instance:
            instance = QtInstance(instance=instance, **kwargs)
        if type(instance) != QtInstance: return
    
        self.view.scene.addItem(instance)

        # connect signal so we can adjust QtNodeLabel positions after zoom
        self.view.updatedViewer.connect(instance.updatePoints)

    def plot(self, idx=None):
        """
        Do the actual plotting of the video frame.

        Args:
            idx (optional): Go to frame idx. If None, stay on current frame.
        """

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

        # Emit signal (it's better to use the signal than a callback)
        self.changedPlot.emit(self, idx)

    def nextFrame(self, dt=1):
        """ Go to next frame.
        """
        self.plot((self.frame_idx + abs(dt)) % self.video.frames)

    def prevFrame(self, dt=1):
        """ Go to previous frame.
        """
        self.plot((self.frame_idx - abs(dt)) % self.video.frames)

    def showLabels(self, show):
        """ Show/hide node labels for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.instances:
            inst.showLabels(show)

    def showEdges(self, show):
        """ Show/hide node edges for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.instances:
            inst.showEdges(show)

    def toggleLabels(self):
        """ Toggle current show/hide state of node labels for all instances.
        """
        for inst in self.instances:
            inst.toggleLabels()

    def toggleEdges(self):
        """ Toggle current show/hide state of edges for all instances.
        """
        for inst in self.instances:
            inst.toggleEdges()

    def zoomToFit(self):
        """ Zoom view to fit all instances
        """
        zoom_rect = self.view.instancesBoundingRect(margin=20)
        if not zoom_rect.size().isEmpty():
            self.view.zoomToRect(zoom_rect)

    def onSequenceSelect(self, seq_len: int, on_success: Callable,
                         on_each = None, on_failure = None):
        """
        Collect a sequence of instances (through user selection) and call `on_success`.
        If the user cancels (by unselecting without new selection), call `on_failure`.
        
        Args:
            seq_len: number of instances we expect user to select
            on_success: callback after use has selected desired number of instances
            on_failure (optional): callback if user cancels selection
            
        Note:
            If successful, we call
            >>> on_success(sequence_of_selected_instance_indexes)
        """
        
        indexes = []
        if self.view.getSelection() is not None:
            indexes.append(self.view.getSelection())
        
        # Define function that will be called when user selects another instance
        def handle_selection(seq_len=seq_len,
                             indexes=indexes,
                             on_success=on_success,
                             on_each=on_each,
                             on_failure=on_failure):
            # Get the index of the currently selected instance
            new_idx = self.view.getSelection()
            # If something is selected, add it to the list
            if new_idx is not None:
                indexes.append(new_idx)
            # If nothing is selected, then remove this handler and trigger on_failure
            else:
                self.view.updatedSelection.disconnect(handle_selection)
                if callable(on_failure):
                    on_failure(indexes)
                return
            
            # If we have all the instances we want in our sequence, we're done
            if len(indexes) >= seq_len:
                # remove this handler
                self.view.updatedSelection.disconnect(handle_selection)
                # trigger success, passing the list of selected indexes
                on_success(indexes)
            # If we're still in progress...
            else:
                if callable(on_each):
                    on_each(indexes)

        self.view.updatedSelection.connect(handle_selection)

        if callable(on_each):
            on_each(indexes)

    def keyPressEvent(self, event: QKeyEvent):
        """ Custom event handler.
        Move between frames, toggle display of edges/labels, and select instances.
        """
        if event.key() == Qt.Key.Key_Left:
            self.prevFrame()
        elif event.key() == Qt.Key.Key_Right:
            self.nextFrame()
        elif event.key() == Qt.Key.Key_Home:
            self.plot(0)
        elif event.key() == Qt.Key.Key_End:
            self.plot(self.video.frames - 1)
        elif event.key() == Qt.Key.Key_Escape:
            self.view.clearSelection()
        elif event.key() == Qt.Key.Key_QuoteLeft:
            self.view.nextSelection()
        elif event.key() < 128 and chr(event.key()).isnumeric():
            # decrement by 1 since instances are 0-indexed
            self.view.selectInstance(int(chr(event.key()))-1)
        else:
            event.ignore() # Kicks the event up to parent
            # print(event.key())


class GraphicsView(QGraphicsView):
    """
    QGraphicsView used by QtVideoPlayer.

    This contains elements for display of video and event handlers for zoom/selection.
    
    Signals:
        updatedViewer: Emitted after update to view (e.g., zoom)
            Used internally so we know when to update points for each instance.
        updatedSelection: Emitted after the user has selected/unselected an instance

        leftMouseButtonPressed
        rightMouseButtonPressed
        leftMouseButtonReleased
        rightMouseButtonReleased
        leftMouseButtonDoubleClicked
        rightMouseButtonDoubleClicked
    """

    updatedViewer = Signal()
    updatedSelection = Signal()
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
        """ Clears the displayed frame from the scene.
        """
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

        base_w_scale = self.width() / self.sceneRect().width()
        base_h_scale = self.height() / self.sceneRect().height()
        base_scale = min(base_w_scale, base_h_scale)

        transform = QTransform()
        transform.scale(base_scale * self.zoomFactor, base_scale * self.zoomFactor)
        self.setTransform(transform)
        self.updatedViewer.emit()

    @property
    def instances(self):
        """
        Returns a list of instances.

        Order in list should match the order in which instances were added to scene.
        """
        return [item for item in self.scene.items(Qt.SortOrder.AscendingOrder)
                if type(item) == QtInstance]

    def clearSelection(self):
        """ Clear instance skeleton selection.
        """
        for instance in self.instances:
            instance.selected = False
        # signal that the selection has changed (so we can update visual display)
        self.updatedSelection.emit()

    def nextSelection(self):
        """ Select next instance (or first, if none currently selected).
        """
        instances = self.instances
        if len(instances) == 0: return
        select_inst = instances[0] # default to selecting first instance
        select_idx = 0
        for idx, instance in enumerate(instances):
            if instance.selected:
                instance.selected = False
                select_idx = (idx+1)%len(instances)
                select_inst = instances[select_idx]
                break
        select_inst.selected = True
        # signal that the selection has changed (so we can update visual display)
        self.updatedSelection.emit()

    def selectInstance(self, select_idx):
        """
        Select a particular skeleton instance.

        Args:
            select_idx: index of skeleton to select
        """
        instances = self.instances
        if select_idx < len(instances):
            for idx, instance in enumerate(instances):
                instance.selected = (select_idx == idx)
        # signal that the selection has changed (so we can update visual display)
        self.updatedSelection.emit()

    def getSelection(self):
        """ Returns the index of the currently selected instance.
        If no instance selected, returns None.
        """
        instances = self.instances
        if len(instances) == 0: return None
        select_inst = instances[0] # default to selecting first instance
        select_idx = 0
        for idx, instance in enumerate(instances):
            if instance.selected:
                return idx

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()


    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        scenePos = self.mapToScene(event.pos())
        # keep track of click location
        self._down_pos = event.pos()
        # behavior depends on which button is pressed
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
        # check if mouse moved during click
        has_moved = (event.pos() != self._down_pos)
        if event.button() == Qt.LeftButton:
            # Check if this was just a tap (not a drag)
            if not has_moved:
                # When just a tap, see if there's an item underneath to select
                clicked = self.scene.items(scenePos, Qt.IntersectsItemBoundingRect)
                clicked_instances = [item for item in clicked if type(item) == QtInstance]
                # We only handle single instance selection so pick at most one from list
                clicked_instance = clicked_instances[0] if len(clicked_instances) else None
                for idx, instance in enumerate(self.instances):
                    instance.selected = (instance == clicked_instance)
                    # If we want to allow selection of multiple instances, do this:
                    # instance.selected = (instance in clicked)
                self.updatedSelection.emit()
            # finish drag
            self.setDragMode(QGraphicsView.NoDrag)
            # pass along event
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                zoom_rect = self.scene.selectionArea().boundingRect()
                self.scene.setSelectionArea(QPainterPath()) # clear selection
                self.zoomToRect(zoom_rect)
            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def zoomToRect(self, zoom_rect: QRectF):
        """
        Method to zoom scene to a given rectangle.

        The rect can either be given relative to the current zoom
        (this is useful if it's the rect drawn by user) or it can be
        given in absolute coordinates for displayed frame.

        Args:
            zoom_rect: The `QRectF` to which we want to zoom.
            relative: Controls whether rect is relative to current zoom.
        """

        if zoom_rect.isNull(): return

        scale_h = self.scene.height()/zoom_rect.height()
        scale_w = self.scene.width()/zoom_rect.width()
        scale = min(scale_h, scale_w)
        
        self.zoomFactor = scale
        self.updateViewer()    
        self.centerOn(zoom_rect.center())

    def clearZoom(self):
        """ Clear zoom stack. Doesn't update display.
        """
        self.zoomFactor = 1

    def instancesBoundingRect(self, margin=0):
        """
        Returns a rect which contains all displayed skeleton instances.

        Args:
            margin: Margin for padding the rect.
        Returns:
            The `QRectF` which contains the skeleton instances.
        """
        rect = QRectF()
        for item in self.instances:
            rect = rect.united(item.boundingRect())
        if margin > 0:
            rect = rect.marginsAdded(QMarginsF(margin, margin, margin, margin))
        return rect

    def mouseDoubleClickEvent(self, event):
        """ Custom event handler. Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.clearZoom()
                self.updateViewer()
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def wheelEvent(self, event):
        """ Custom event handler. Zoom in/out based on scroll wheel change.
        """
        # zoom on wheel when no mouse buttons are pressed
        if event.buttons() == Qt.NoButton:
            angle = event.angleDelta().y()
            factor = 1.1 if angle > 0 else 0.9

            self.zoomFactor = max(factor * self.zoomFactor, 1)
            self.updateViewer()
        # trigger default event handler so event will pass to children
        QGraphicsView.wheelEvent(self, event)

    def keyPressEvent(self, event):
        event.ignore() # Kicks the event up to parent

    def keyReleaseEvent(self, event):
        event.ignore() # Kicks the event up to parent


class QtNodeLabel(QGraphicsTextItem):
    """
    QGraphicsTextItem to handle display of node text label.

    Args:
        node: The `QtNode` to which this label is attached.
        parent: The `QtInstance` which will contain this item.
    """

    def __init__(self, node, parent, *args, **kwargs):
        self.node = node
        self.text = node.name
        self._parent = parent
        super(QtNodeLabel, self).__init__(self.text, parent=parent, *args, **kwargs)

        self._anchor_x = self.pos().x()
        self._anchor_x = self.pos().y()

        self._base_font = QFont()
        self._base_font.setPixelSize(12)
        self.setFont(self._base_font)

        # set color to match node color
        self.setDefaultTextColor(self.node.pen().color())
        # don't rescale when view is scaled (i.e., zoom)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        self.adjustStyle()

    def adjustPos(self, *args, **kwargs):
        """ Update the position of the label based on the position of the node.

        Args:
            Accepts arbitrary arguments so we can connect to various signals.
        """
        node = self.node
        self._anchor_x = node.point.x
        self._anchor_y = node.point.y

        # Calculate position for label within the largest arc made by edges.
        shift_angle = 0
        if len(node.edges):
            edge_angles = sorted([edge.angle_to(node) for edge in node.edges])

            edge_angles.append(edge_angles[0] + math.pi*2)
            # Calculate size and bisector for each arc between adjacent edges
            edge_arcs = [(edge_angles[i+1]-edge_angles[i],
                            edge_angles[i+1]/2+edge_angles[i]/2)
                         for i in range(len(edge_angles)-1)]
            max_arc = sorted(edge_arcs)[-1]
            shift_angle = max_arc[1] # this is the angle of the bisector
            shift_angle %= 2*math.pi

        # Use the _shift_factor to control how the label is positioned
        # relative to the node.
        # Shift factor of -1 means we shift label up/left by its height/width.
        self._shift_factor_x = (math.cos(shift_angle)*.6) -.5
        self._shift_factor_y = (math.sin(shift_angle)*.6) -.5

        # Since item doesn't scale when view is transformed (i.e., zoom)
        # we need to calculate bounding size in view manually.
        height = self.boundingRect().height()
        width = self.boundingRect().width()

        scene = self.scene()
        if scene is not None:
            # Get the current scaling for the view and apply this to size of label
            view = scene.views()[0]
            height = height / view.viewportTransform().m11()
            width = width / view.viewportTransform().m22()

        self.setPos(self._anchor_x + width*self._shift_factor_x,
                    self._anchor_y + height*self._shift_factor_y)

        # Now apply these changes to the visual display
        self.adjustStyle()

    def adjustStyle(self):
        """ Update visual display of the label and its node.
        """
        if self.node.point.complete:
            self._base_font.setBold(True)
            self.setFont(self._base_font)
            complete_color = QColor(80, 194, 159) # greenish
            self.setDefaultTextColor(complete_color)
            # FIXME: Adjust style of node here as well?
            # self.node.setBrush(complete_color)
        else:
            self._base_font.setBold(False)
            self.setFont(self._base_font)
            self.setDefaultTextColor(QColor(232, 45, 32)) # redish

    def boundingRect(self):
        """ Method required by Qt.
        """
        return super(QtNodeLabel, self).boundingRect()

    def paint(self, *args, **kwargs):
        """ Method required by Qt.
        """
        super(QtNodeLabel, self).paint(*args, **kwargs)

    def mousePressEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.node.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.node.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.node.mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.node.wheelEvent(event)


class QtNode(QGraphicsEllipseItem):
    """
    QGraphicsEllipseItem to handle display of skeleton instance node.

    Args:
        parent: The `QtInstance` which will contain this item.
        point: The `Point` where this node is located.
            Note that this is a mutable object so we're able to directly access
            the very same `Point` object that's defined outside our class.
        radius: Radius of the visual node item.
        color: Color of the visual node item.
        callbacks: List of functions to call after we update to the `Point`.
    """
    def __init__(self, parent, point:Point, radius:float, color:list, node_name:str = None, callbacks = None, *args, **kwargs):
        self.point = point
        self.radius = radius
        self.color = color
        self.edges = []
        self.name = node_name
        self.callbacks = [] if callbacks is None else callbacks
        self.dragParent = False

        super(QtNode, self).__init__(-self.radius, -self.radius, self.radius*2, self.radius*2, parent=parent, *args, **kwargs)

        if node_name is not None:
            self.setToolTip(node_name)

        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setFlag(QGraphicsItem.ItemIsMovable)

        col_line = QColor(*self.color)
        
        self.pen_default = QPen(col_line, 1)
        self.pen_default.setCosmetic(True) # https://stackoverflow.com/questions/13120486/adjusting-qpen-thickness-when-scaling-qgraphicsview
        self.pen_missing = QPen(col_line, 1)
        self.pen_missing.setCosmetic(True)
        self.brush = QBrush(QColor(*self.color, a=128))
        self.brush_missing = QBrush(QColor(*self.color, a=0))

        self.setPos(self.point.x, self.point.y)
        self.updatePoint()

    def calls(self):
        """ Method to call all callbacks.
        """
        for callback in self.callbacks:
            if callable(callback):
                callback(self)

    def updatePoint(self):
        """ Method to update data for node/edge after user manipulates visual point.
        """
        self.point.x = self.scenePos().x()
        self.point.y = self.scenePos().y()

        if self.point.visible:
            radius = self.radius
            self.setPen(self.pen_default)
            self.setBrush(self.brush)
        else:
            radius = self.radius / 2.
            self.setPen(self.pen_missing)
            self.setBrush(self.brush_missing)
        self.setRect(-radius, -radius, radius*2, radius*2)

        for edge in self.edges:
            edge.updateEdge(self)
            # trigger callbacks for other connected nodes
            edge.connected_to(self).calls()

        # trigger callbacks for this node
        self.calls()

    def mousePressEvent(self, event):
        """ Custom event handler for mouse press.
        """
        if event.button() == Qt.LeftButton:
            # Alt-click to drag instance
            if event.modifiers() == Qt.AltModifier:
                self.dragParent = True
                self.parentObject().setFlag(QGraphicsItem.ItemIsMovable)
                # set origin to point clicked so that we can rotate around this point
                self.parentObject().setTransformOriginPoint(self.scenePos())
                self.parentObject().mousePressEvent(event)
            # Shift-click to mark all points as complete
            elif event.modifiers() == Qt.ShiftModifier:
                self.parentObject().updatePoints(complete=True)
            else:
                self.dragParent = False
                super(QtNode, self).mousePressEvent(event)
                self.updatePoint()
            
            self.point.complete = True
        elif event.button() == Qt.RightButton:
            # Right-click to toggle node as missing from this instance
            self.point.visible = not self.point.visible
            self.updatePoint()
        elif event.button() == Qt.MidButton:
            pass

    def mouseMoveEvent(self, event):
        """ Custom event handler for mouse move.
        """
        #print(event)
        if self.dragParent:
            self.parentObject().mouseMoveEvent(event)
        else:
            super(QtNode, self).mouseMoveEvent(event)
            self.updatePoint()

    def mouseReleaseEvent(self, event):
        """ Custom event handler for mouse release.
        """
        #print(event)
        if self.dragParent:
            self.parentObject().mouseReleaseEvent(event)
            self.parentObject().setSelected(False)
            self.parentObject().setFlag(QGraphicsItem.ItemIsMovable, False)
            self.parentObject().updatePoints()
        else:
            super(QtNode, self).mouseReleaseEvent(event)
            self.updatePoint()

    def wheelEvent(self, event):
        """Custom event handler for mouse scroll wheel."""
        if self.dragParent:
            angle = event.delta() / 20 + self.parentObject().rotation()
            self.parentObject().setRotation(angle)


class QtEdge(QGraphicsLineItem):
    """
    QGraphicsLineItem to handle display of edge between skeleton instance nodes.

    Args:
        src: The `QtNode` source node for the edge.
        dst: The `QtNode` destination node for the edge.
    """
    def __init__(self, parent, src:QtNode, dst:QtNode, color, *args, **kwargs):
        self.src = src
        self.dst = dst

        super(QtEdge, self).__init__(self.src.point.x, self.src.point.y, self.dst.point.x, self.dst.point.y, parent=parent, *args, **kwargs)

        pen = QPen(QColor(*color), 1)
        pen.setCosmetic(True)
        self.setPen(pen)

    def connected_to(self, node):
        """
        Return the other node along the edge.

        Args:
            node: One of the edge's nodes.
        Returns:
            The other node (or None if edge doesn't have node).
        """
        if node == self.src:
            return self.dst
        elif node == self.dst:
            return self.src

        return None

    def angle_to(self, node):
        """
        Returns the angle from one edge node to the other.

        Args:
            node: The node from which we're measuring the angle.
        Returns:
            Angle (in radians) to the other node.
        """
        to = self.connected_to(node)
        if to is not None:
            x = to.point.x - node.point.x
            y = to.point.y - node.point.y
            return math.atan2(y, x)

    def updateEdge(self, node):
        """
        Updates the visual display of node.

        Args:
            node: The node to update.
        """
        if node == self.src:
            line = self.line()
            line.setP1(node.scenePos())
            self.setLine(line)

        elif node == self.dst:
            line = self.line()
            line.setP2(node.scenePos())
            self.setLine(line)


class QtInstance(QGraphicsObject):
    """
    QGraphicsObject for skeleton instances.

    This object stores the data for one skeleton instance
    and handles the events to manipulate the skeleton within
    a video frame (i.e., moving, rotating, marking nodes).

    It should be instatiated with a `Skeleton` or `Instance`
    and added to the relevant `QGraphicsScene`.

    When instantiated, it creates `QtNode`, `QtEdge`, and
    `QtNodeLabel` items as children of itself.
    """
    def __init__(self, skeleton:Skeleton = None, instance: Instance = None, color=(0, 114, 189), markerRadius=4, *args, **kwargs):
        super(QtInstance, self).__init__(*args, **kwargs)
        self.skeleton = skeleton if instance is None else instance.skeleton
        self.instance = instance
        self.nodes = {}
        self.edges = []
        self.edges_shown = True
        self.labels = {}
        self.labels_shown = True
        self.color = color
        self.markerRadius = markerRadius
        self._selected = False
        self._bounding_rect = QRectF()
        #self.setFlag(QGraphicsItem.ItemIsMovable)
        #self.setFlag(QGraphicsItem.ItemIsSelectable)

        # Add box to go around instance
        self.box = QGraphicsRectItem(parent=self)
        box_pen = QPen(QColor(*self.color), 1)
        box_pen.setStyle(Qt.DashLine)
        box_pen.setCosmetic(True)
        self.box.setPen(box_pen)

        # Add nodes
        for (node, point) in self.instance.nodes_points():
            node_item = QtNode(parent=self, point=point, node_name=node.name,
                               color=self.color, radius=self.markerRadius)
            self.nodes[node.name] = node_item

        # Add edges
        for (src, dst) in self.skeleton.edge_names:
            # Make sure that both nodes are present in this instance before drawing edge
            if src in self.nodes and dst in self.nodes:
                edge_item = QtEdge(parent=self, src=self.nodes[src], dst=self.nodes[dst],
                                   color=self.color)
                self.nodes[src].edges.append(edge_item)
                self.nodes[dst].edges.append(edge_item)
                self.edges.append(edge_item)

        # Add labels to nodes
        # We do this after adding edges so that we can position labels to avoid overlap
        for node in self.nodes.values():
            node_label = QtNodeLabel(node, parent=self)
            node_label.adjustPos()

            self.labels[node.name] = node_label
            # add callback to adjust position of label after node has moved
            node.callbacks.append(node_label.adjustPos)
            node.callbacks.append(self.updateBox)

        # Update size of box so it includes all the nodes/edges
        self.updateBox()

    def updatePoints(self, complete:bool = False):
        """
        Updates data and display for all points in skeleton.

        This is called any time the skeleton is manipulated as a whole.

        Args:
            complete (optional): If set, we mark the state of all
                nodes in the skeleton to "complete".
        Returns:
            None.
        """

        # Update the position for each node
        for node_item in self.nodes.values():
            node_item.point.x = node_item.scenePos().x()
            node_item.point.y = node_item.scenePos().y()
            node_item.setPos(node_item.point.x, node_item.point.y)
            if complete: node_item.point.complete = True
        # Wait to run callbacks until all noces are updated
        # Otherwise the label positions aren't correct since
        # they depend on the edge vectors to old node positions.
        for node_item in self.nodes.values():
            node_item.calls()
        # Reset the scene position and rotation (changes when we drag entire skeleton)
        self.setPos(0, 0)
        self.setRotation(0)
        # Update the position for each edge
        for edge_item in self.edges:
            edge_item.updateEdge(edge_item.src)
            edge_item.updateEdge(edge_item.dst)
        # Update box for instance selection
        self.updateBox()

    def getPointsBoundingRect(self):
        """Returns a rect which contains all the nodes in the skeleton."""
        rect = None
        for item in self.edges:
            rect = item.boundingRect() if rect is None else rect.united(item.boundingRect())
        return rect

    def updateBox(self, *args, **kwargs):
        """
        Updates the box drawn around a selected skeleton.

        This updates both the box attribute stored and the visual box.
        The box attribute is used to determine whether a click should
        select this instance.
        """
        # Only show box if instance is selected
        op = .7 if self._selected else 0
        self.box.setOpacity(op)
        # Update the position for the box
        rect = self.getPointsBoundingRect()
        self._bounding_rect = rect
        rect = rect.marginsAdded(QMarginsF(10, 10, 10, 10))
        self.box.setRect(rect)

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, selected:bool):
        self._selected = selected
        # Update the selection box for this skeleton instance
        self.updateBox()

    def toggleLabels(self):
        """ Toggle whether or not labels are shown for this skeleton instance.
        """
        self.showLabels(not self.labels_shown)

    def showLabels(self, show):
        """
        Draws/hides the labels for this skeleton instance.

        Args:
            show: Show labels if True, hide them otherwise.
        """
        op = 1 if show else 0
        for label in self.labels.values():
            label.setOpacity(op)
        self.labels_shown = show

    def toggleEdges(self):
        """ Toggle whether or not edges are shown for this skeleton instance.
        """
        self.showEdges(not self.edges_shown)

    def showEdges(self, show = True):
        """
        Draws/hides the edges for this skeleton instance.

        Args:
            show: Show edges if True, hide them otherwise.
        """
        op = 1 if show else 0
        for edge in self.edges:
            edge.setOpacity(op)
        self.edges_shown = show

    def boundingRect(self):
        """ Method required Qt to determine bounding rect for item.
        """
        return self._bounding_rect

    def paint(self, painter, option, widget=None):
        """ Method required by Qt.
        """
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
            vp.addInstance(instance=instance, color=cmap[i%len(cmap)])

    window.changedPlot.connect(plot_instances)


    window.show()
    window.plot()

    app.exec_()


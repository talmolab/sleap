"""
Module for showing and manipulating skeleton instances within a video.

All interactions should go through `QtVideoPlayer`.

Example usage:
    >>> my_video = Video(...)
    >>> my_instance = Instance(...)

    >>> vp = QtVideoPlayer(video=my_video)
    >>> vp.addInstance(instance=my_instance, color=(r, g, b))

"""

from PySide2 import QtWidgets, QtCore

from PySide2.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
)
from PySide2.QtGui import QImage, QPixmap, QPainter, QPainterPath, QTransform
from PySide2.QtGui import QPen, QBrush, QColor, QFont
from PySide2.QtGui import QKeyEvent
from PySide2.QtCore import Qt, QRectF, QPointF, QMarginsF

import math

from typing import Callable, Dict, List, Optional, Tuple, Union

from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject

from PySide2.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsTextItem,
    QGraphicsRectItem,
)

from sleap.skeleton import Node
from sleap.instance import Instance, Point
from sleap.io.video import Video
from sleap.gui.slider import VideoSlider
from sleap.gui.state import GuiState

import qimage2ndarray


class QtVideoPlayer(QWidget):
    """
    Main QWidget for displaying video with skeleton instances.

    Signals:
        * changedPlot: Emitted whenever the plot is redrawn

    Attributes:
        video: The :class:`Video` to display
        color_manager: A :class:`TrackColorManager` object which determines
            which color to show the instances.

    """

    changedPlot = QtCore.Signal(QWidget, int, Instance)

    def __init__(
        self,
        video: Video = None,
        color_manager=None,
        state=None,
        context=None,
        *args,
        **kwargs,
    ):
        super(QtVideoPlayer, self).__init__(*args, **kwargs)

        self._shift_key_down = False

        self.color_manager = color_manager
        self.state = state or GuiState()
        self.context = context
        self.view = GraphicsView(self.state)
        self.video = None

        self.seekbar = VideoSlider(color_manager=self.color_manager)
        self.seekbar.keyPress.connect(self.keyPressEvent)
        self.seekbar.keyRelease.connect(self.keyReleaseEvent)
        self.seekbar.setEnabled(False)

        self.splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.seekbar)
        self.seekbar.heightUpdated.connect(lambda: self.splitter.refresh())

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)

        self.seekbar.valueChanged.connect(
            lambda e: self.state.set("frame_idx", self.seekbar.value())
        )

        def update_selection_state(a, b):
            self.state.set("frame_range", (a, b))
            self.state.set("has_frame_range", (a < b))

        self.seekbar.selectionChanged.connect(update_selection_state)

        self.state.connect("frame_idx", lambda idx: self.plot())
        self.state.connect("frame_idx", lambda idx: self.seekbar.setValue(idx))
        self.state.connect("instance", self.view.selectInstance)

        self.state.connect("show labels", self.plot)
        self.state.connect("show edges", self.plot)
        self.state.connect("video", self.load_video)
        self.state.connect("fit", self.setFitZoom)

        self.view.show()

        if video is not None:
            self.load_video(video)

    def load_video(self, video: Video, plot=True):
        """
        Load video into viewer.

        Args:
            video: the :class:`Video` to display
            plot: If True, plot the video frame. Otherwise, just load the data.
        """

        self.video = video

        # Is this necessary?
        self.view.scene.setSceneRect(0, 0, video.width, video.height)

        self.seekbar.setMinimum(0)
        self.seekbar.setMaximum(self.video.last_frame_idx)
        self.seekbar.setEnabled(True)

        if plot:
            self.plot()

    def reset(self):
        """ Reset viewer by removing all video data.
        """
        self.video = None
        self.state["frame_idx"] = None
        self.view.clear()
        self.seekbar.setMaximum(0)
        self.seekbar.setEnabled(False)

    @property
    def instances(self):
        """Returns list of all `QtInstance` objects in view."""
        return self.view.instances

    @property
    def selectable_instances(self):
        """Returns list of selectable `QtInstance` objects in view."""
        return self.view.selectable_instances

    @property
    def predicted_instances(self):
        """Returns list of predicted `QtInstance` objects in view."""
        return self.view.predicted_instances

    @property
    def scene(self):
        """Returns `QGraphicsScene` for viewer."""
        return self.view.scene

    def addInstance(self, instance, **kwargs):
        """Add a skeleton instance to the video.

        Args:
            instance: this can be either a `QtInstance` or an `Instance`

            Any other named args are passed along if/when creating QtInstance.
        """
        # Check if instance is an Instance (or subclass of Instance)
        if issubclass(type(instance), Instance):
            instance = QtInstance(instance=instance, player=self, **kwargs)
        if type(instance) != QtInstance:
            return

        self.view.scene.addItem(instance)

        # connect signal so we can adjust QtNodeLabel positions after zoom
        self.view.updatedViewer.connect(instance.updatePoints)

    def plot(self, *args):
        """
        Do the actual plotting of the video frame.
        """
        if self.video is None:
            return

        idx = self.state["frame_idx"] or 0

        # Get image data
        try:
            frame = self.video.get_frame(idx)
        except:
            frame = None

        if frame is not None:
            # Clear existing objects
            self.view.clear()

            # Convert ndarray to QImage
            image = qimage2ndarray.array2qimage(frame)

            # Display image
            self.view.setImage(image)

            # Emit signal
            self.changedPlot.emit(self, idx, self.state["instance"])

    def showLabels(self, show):
        """ Show/hide node labels for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.selectable_instances:
            inst.showLabels(show)

    def showEdges(self, show):
        """ Show/hide node edges for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.selectable_instances:
            inst.showEdges(show)

    def zoomToFit(self):
        """ Zoom view to fit all instances.
        """
        zoom_rect = self.view.instancesBoundingRect(margin=20)
        if not zoom_rect.size().isEmpty():
            self.view.zoomToRect(zoom_rect)

    def setFitZoom(self, value):
        """Zooms or unzooms current view to fit all instances."""
        if self.video:
            if value:
                self.zoomToFit()
            else:
                self.view.clearZoom()
            self.plot()

    def getVisibleRect(self):
        """Returns `QRectF` with currently visible portion of frame image."""
        return self.view.mapToScene(self.view.rect()).boundingRect()

    def onSequenceSelect(
        self,
        seq_len: int,
        on_success: Callable,
        on_each: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
    ):
        """
        Collect a sequence of instances (through user selection).

        When the sequence is complete, the `on_success` callback is called.
        After each selection in sequence, the `on_each` callback is called
        (if given). If the user cancels (by unselecting without new
        selection), the `on_failure` callback is called (if given).

        Note:
            If successful, we call
            >>> on_success(list_of_instances)

        Args:
            seq_len: Number of instances we want to collect in sequence.
            on_success: Callback for when user has selected desired number of
                instances.
            on_each: Callback after user selects each instance.
            on_failure: Callback if user cancels process before selecting
                enough instances.

        """

        selected_instances = []
        if self.view.getSelectionInstance() is not None:
            selected_instances.append(self.view.getSelectionInstance())

        # Define function that will be called when user selects another instance
        def handle_selection(
            seq_len=seq_len,
            selected_instances=selected_instances,
            on_success=on_success,
            on_each=on_each,
            on_failure=on_failure,
        ):
            # Get the index of the currently selected instance
            new_instance = self.view.getSelectionInstance()
            # If something is selected, add it to the list
            if new_instance is not None:
                selected_instances.append(new_instance)
            # If nothing is selected, then remove this handler and trigger on_failure
            else:
                self.view.updatedSelection.disconnect(handle_selection)
                if callable(on_failure):
                    on_failure(selected_instances)
                return

            # If we have all the instances we want in our sequence, we're done
            if len(selected_instances) >= seq_len:
                # remove this handler
                self.view.updatedSelection.disconnect(handle_selection)
                # trigger success, passing the list of selected instances
                on_success(selected_instances)
            # If we're still in progress...
            else:
                if callable(on_each):
                    on_each(selected_instances)

        self.view.updatedSelection.connect(handle_selection)

        if callable(on_each):
            on_each(selected_instances)

    @staticmethod
    def _signal_once(signal: QtCore.Signal, callback: Callable):
        """
        Connects callback for next occurrence of signal.

        Args:
            signal: The signal on which we want callback to be called.
            callback: The function that should be called just once, the next
                time the signal is emitted.

        Returns:
            None.
        """

        def call_once(*args):
            signal.disconnect(call_once)
            callback(*args)

        signal.connect(call_once)

    def onPointSelection(self, callback: Callable):
        """
        Starts mode for user to click point, callback called when finished.

        Args:
            callback: The function called after user clicks point, should
                take x and y as arguments.

        Returns:
            None.
        """
        self.view.click_mode = "point"
        self.view.setCursor(Qt.CrossCursor)
        self._signal_once(self.view.pointSelected, callback)

    def onAreaSelection(self, callback: Callable):
        """
        Starts mode for user to select area, callback called when finished.

        Args:
            callback: The function called after user clicks point, should
                take x0, y0, x1, y1 as arguments.

        Returns:
            None.
        """
        self.view.click_mode = "area"
        self.view.setCursor(Qt.CrossCursor)
        self._signal_once(self.view.areaSelected, callback)

    def keyReleaseEvent(self, event: QKeyEvent):
        """
        Custom event handler, tracks when user releases modifier (shift) key.
        """
        if event.key() == Qt.Key.Key_Shift:
            self._shift_key_down = False
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Custom event handler, allows navigation and selection within view.
        """
        frame_t0 = self.state["frame_idx"]

        if event.key() == Qt.Key.Key_Shift:
            self._shift_key_down = True
        elif event.key() == Qt.Key.Key_Left:
            self.state.increment("frame_idx", step=-1, mod=self.video.frames)
        elif event.key() == Qt.Key.Key_Right:
            self.state.increment("frame_idx", step=1, mod=self.video.frames)
        elif event.key() == Qt.Key.Key_Up:
            self.state.increment("frame_idx", step=-50, mod=self.video.frames)
        elif event.key() == Qt.Key.Key_Down:
            self.state.increment("frame_idx", step=50, mod=self.video.frames)
        elif event.key() == Qt.Key.Key_Space:
            self.state.increment("frame_idx", step=500, mod=self.video.frames)
        elif event.key() == Qt.Key.Key_Home:
            self.state["frame_idx"] = 0
        elif event.key() == Qt.Key.Key_End:
            self.state["frame_idx"] = self.video.frames - 1
        elif event.key() == Qt.Key.Key_Escape:
            self.view.click_mode = ""
            self.state["instance"] = None
        elif event.key() == Qt.Key.Key_QuoteLeft:
            self.state.increment_in_list("instance", self.selectable_instances)
        elif event.key() < 128 and chr(event.key()).isnumeric():
            # decrement by 1 since instances are 0-indexed
            idx = int(chr(event.key())) - 1
            if 0 <= idx < len(self.selectable_instances):
                instance = self.selectable_instances[idx].instance
                self.state["instance"] = instance
        else:
            event.ignore()  # Kicks the event up to parent

        # If user is holding down shift and action resulted in moving to another frame
        if self._shift_key_down and frame_t0 != self.state["frame_idx"]:
            # If there's no select, start seekbar selection at frame before action
            start, end = self.seekbar.getSelection()
            if start == end:
                self.seekbar.startSelection(frame_t0)
            # Set endpoint to frame after action
            self.seekbar.endSelection(self.state["frame_idx"], update=True)


class GraphicsView(QGraphicsView):
    """
    Custom `QGraphicsView` used by `QtVideoPlayer`.

    This contains elements for display of video and event handlers for zoom
    and selection of instances in view.

    Signals:
        * updatedViewer: Emitted after update to view (e.g., zoom).
            Used internally so we know when to update points for each instance.
        * updatedSelection: Emitted after the user has (un)selected an instance.
        * instanceDoubleClicked: Emitted after an instance is double-clicked.
            Passes the :class:`Instance` that was double-clicked.
        * areaSelected: Emitted after user selects an area when in "area"
            click mode. Passes x0, y0, x1, y1 for selected box coordinates.
        * pointSelected: Emitted after user clicks a point (in "point" click
            mode.) Passes x, y coordinates of point.
        * leftMouseButtonPressed: Emitted by event handler.
        * rightMouseButtonPressed: Emitted by event handler.
        * leftMouseButtonReleased: Emitted by event handler.
        * rightMouseButtonReleased: Emitted by event handler.
        * leftMouseButtonDoubleClicked: Emitted by event handler.
        * rightMouseButtonDoubleClicked: Emitted by event handler.

    """

    updatedViewer = QtCore.Signal()
    updatedSelection = QtCore.Signal()
    instanceDoubleClicked = QtCore.Signal(Instance)
    areaSelected = QtCore.Signal(float, float, float, float)
    pointSelected = QtCore.Signal(float, float)
    leftMouseButtonPressed = QtCore.Signal(float, float)
    rightMouseButtonPressed = QtCore.Signal(float, float)
    leftMouseButtonReleased = QtCore.Signal(float, float)
    rightMouseButtonReleased = QtCore.Signal(float, float)
    leftMouseButtonDoubleClicked = QtCore.Signal(float, float)
    rightMouseButtonDoubleClicked = QtCore.Signal(float, float)

    def __init__(self, state=None, *args, **kwargs):
        """ https://github.com/marcel-goldschen-ohm/PyQtImageViewer/blob/master/QtImageViewer.py """
        QGraphicsView.__init__(self)
        self.state = state or GuiState()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        # brush = QBrush(QColor.black())
        self.scene.setBackgroundBrush(QBrush(QColor(Qt.black)))

        self._pixmapHandle = None

        self.setRenderHint(QPainter.Antialiasing)

        self.aspectRatioMode = Qt.KeepAspectRatio
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canZoom = True
        self.canPan = True
        self.click_mode = ""

        self.zoomFactor = 1
        anchor_mode = QGraphicsView.AnchorUnderMouse
        self.setTransformationAnchor(anchor_mode)

    def hasImage(self) -> bool:
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clear(self):
        """ Clears the displayed frame from the scene.
        """
        self._pixmapHandle = None
        self.scene.clear()

    def setImage(self, image: Union[QImage, QPixmap]):
        """
        Set the scene's current image pixmap to the input QImage or QPixmap.

        Args:
            image: The QPixmap or QImage to display.

        Raises:
            RuntimeError: If the input image is not QImage or QPixmap

        Returns:
            None.
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap(image)
        else:
            raise RuntimeError(
                "ImageViewer.setImage: Argument must be a QImage or QPixmap."
            )
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def updateViewer(self):
        """ Apply current zoom. """
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
    def instances(self) -> List["QtInstance"]:
        """
        Returns a list of instances.

        Order should match the order in which instances were added to scene.
        """
        return list(filter(lambda x: not x.predicted, self.all_instances))

    @property
    def predicted_instances(self) -> List["QtInstance"]:
        """
        Returns a list of predicted instances.

        Order should match the order in which instances were added to scene.
        """
        return list(filter(lambda x: not x.predicted, self.all_instances))

    @property
    def selectable_instances(self) -> List["QtInstance"]:
        """
        Returns a list of instances which user can select.

        Order should match the order in which instances were added to scene.
        """
        return list(filter(lambda x: x.selectable, self.all_instances))

    @property
    def all_instances(self) -> List["QtInstance"]:
        """
        Returns a list of all `QtInstance` objects in scene.

        Order should match the order in which instances were added to scene.
        """
        scene_items = self.scene.items(Qt.SortOrder.AscendingOrder)
        return list(filter(lambda x: isinstance(x, QtInstance), scene_items))

    def selectInstance(self, select: Union[Instance, int]):
        """
        Select a particular instance in view.

        Args:
            select: Either `Instance` or index of instance in view.

        Returns:
            None
        """
        for idx, instance in enumerate(self.all_instances):
            instance.selected = select == idx or select == instance.instance
        self.updatedSelection.emit()

    def getSelectionIndex(self) -> Optional[int]:
        """ Returns the index of the currently selected instance.
        If no instance selected, returns None.
        """
        instances = self.all_instances
        if len(instances) == 0:
            return None
        for idx, instance in enumerate(instances):
            if instance.selected:
                return idx

    def getSelectionInstance(self) -> Optional[Instance]:
        """ Returns the currently selected instance.
        If no instance selected, returns None.
        """
        instances = self.all_instances
        if len(instances) == 0:
            return None
        for idx, instance in enumerate(instances):
            if instance.selected:
                return instance.instance

    def getTopInstanceAt(self, scenePos) -> Optional[Instance]:
        """Returns topmost instance at position in scene."""
        # Get all items at scenePos
        clicked = self.scene.items(scenePos, Qt.IntersectsItemBoundingRect)

        # Filter by selectable instances
        def is_selectable(item):
            return type(item) == QtInstance and item.selectable

        clicked = list(filter(is_selectable, clicked))

        if len(clicked):
            return clicked[0].instance

        return None

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

            if event.modifiers() == Qt.NoModifier:

                if self.click_mode == "area":
                    self.setDragMode(QGraphicsView.RubberBandDrag)
                elif self.click_mode == "point":
                    self.setDragMode(QGraphicsView.NoDrag)
                elif self.canPan:
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
        has_moved = event.pos() != self._down_pos
        if event.button() == Qt.LeftButton:

            if self.click_mode == "":
                # Check if this was just a tap (not a drag)
                if not has_moved:
                    self.state["instance"] = self.getTopInstanceAt(scenePos)

            elif self.click_mode == "area":
                # Check if user was selecting rectangular area
                selection_rect = self.scene.selectionArea().boundingRect()

                self.areaSelected.emit(
                    selection_rect.left(),
                    selection_rect.top(),
                    selection_rect.right(),
                    selection_rect.bottom(),
                )
            elif self.click_mode == "point":
                self.pointSelected.emit(scenePos.x(), scenePos.y())

            self.click_mode = ""
            self.unsetCursor()

            # finish drag
            self.setDragMode(QGraphicsView.NoDrag)
            # pass along event
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                zoom_rect = self.scene.selectionArea().boundingRect()
                self.scene.setSelectionArea(QPainterPath())  # clear selection
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
        """

        if zoom_rect.isNull():
            return

        scale_h = self.scene.height() / zoom_rect.height()
        scale_w = self.scene.width() / zoom_rect.width()
        scale = min(scale_h, scale_w)

        self.zoomFactor = scale
        self.updateViewer()
        self.centerOn(zoom_rect.center())

    def clearZoom(self):
        """ Clear zoom stack. Doesn't update display.
        """
        self.zoomFactor = 1

    def instancesBoundingRect(self, margin: float = 0) -> QRectF:
        """
        Returns a rect which contains all displayed skeleton instances.

        Args:
            margin: Margin for padding the rect.
        Returns:
            The `QRectF` which contains the skeleton instances.
        """
        rect = QRectF()
        for item in self.all_instances:
            rect = rect.united(item.boundingRect())
        if margin > 0:
            rect = rect.marginsAdded(QMarginsF(margin, margin, margin, margin))
        return rect

    def mouseDoubleClickEvent(self, event):
        """ Custom event handler, clears zoom.
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

        # Trigger wheelEvent for all child elements. This is a bit of a hack.
        # We can't use QGraphicsView.wheelEvent(self, event) since that will scroll view.
        # We want to trigger for all children, since wheelEvent should continue rotating
        # an skeleton even if the skeleton node/node label is no longer under the cursor.
        # Note that children expect a QGraphicsSceneWheelEvent event, which is why we're
        # explicitly ignoring TypeErrors. Everything seems to work fine since we don't
        # care about the mouse position; if we did, we'd need to map pos to scene.
        for child in self.items():
            try:
                child.wheelEvent(event)
            except TypeError:
                pass

    def keyPressEvent(self, event):
        """Custom event hander, disables default QGraphicsView behavior."""
        event.ignore()  # Kicks the event up to parent

    def keyReleaseEvent(self, event):
        """Custom event hander, disables default QGraphicsView behavior."""
        event.ignore()  # Kicks the event up to parent


class QtNodeLabel(QGraphicsTextItem):
    """
    QGraphicsTextItem to handle display of node text label.

    Args:
        node: The `QtNode` to which this label is attached.
        parent: The `QtInstance` which will contain this item.
        predicted: Whether this is for a predicted point.
    """

    def __init__(
        self,
        node: Node,
        parent: QGraphicsObject,
        predicted: bool = False,
        *args,
        **kwargs,
    ):
        self.node = node
        self.text = node.name
        self.predicted = predicted
        self._parent_instance = parent
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
        self._anchor_x = node.pos().x()
        self._anchor_y = node.pos().y()

        # Calculate position for label within the largest arc made by edges.
        shift_angle = 0
        if len(node.edges):
            edge_angles = sorted([edge.angle_to(node) for edge in node.edges])

            edge_angles.append(edge_angles[0] + math.pi * 2)
            # Calculate size and bisector for each arc between adjacent edges
            edge_arcs = [
                (
                    edge_angles[i + 1] - edge_angles[i],
                    edge_angles[i + 1] / 2 + edge_angles[i] / 2,
                )
                for i in range(len(edge_angles) - 1)
            ]
            max_arc = sorted(edge_arcs)[-1]
            shift_angle = max_arc[1]  # this is the angle of the bisector
            shift_angle %= 2 * math.pi

        # Use the _shift_factor to control how the label is positioned
        # relative to the node.
        # Shift factor of -1 means we shift label up/left by its height/width.
        self._shift_factor_x = (math.cos(shift_angle) * 0.6) - 0.5
        self._shift_factor_y = (math.sin(shift_angle) * 0.6) - 0.5

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

        self.setPos(
            self._anchor_x + width * self._shift_factor_x,
            self._anchor_y + height * self._shift_factor_y,
        )

        # Now apply these changes to the visual display
        self.adjustStyle()

    def adjustStyle(self):
        """ Update visual display of the label and its node.
        """

        complete_color = (
            QColor(80, 194, 159) if self.node.point.complete else QColor(232, 45, 32)
        )

        if self.predicted:
            self._base_font.setBold(False)
            self.setFont(self._base_font)
            self.setDefaultTextColor(QColor(128, 128, 128))
        elif not self.node.point.visible:
            self._base_font.setBold(False)
            self.setFont(self._base_font)
            # self.setDefaultTextColor(self.node.pen().color())
            self.setDefaultTextColor(complete_color)
        elif self.node.point.complete:
            self._base_font.setBold(True)
            self.setFont(self._base_font)
            self.setDefaultTextColor(complete_color)  # greenish
            # FIXME: Adjust style of node here as well?
            # self.node.setBrush(complete_color)
        else:
            self._base_font.setBold(False)
            self.setFont(self._base_font)
            self.setDefaultTextColor(complete_color)  # redish

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
        self.setCursor(Qt.ArrowCursor)
        self.node.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.node.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ Pass events along so that clicking label is like clicking node.
        """
        self.unsetCursor()
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
        node: The :class:`Node` corresponding to this visual node.
        point: The :class:`Point` where this node is located.
            Note that this is a mutable object so we're able to directly access
            the very same `Point` object that's defined outside our class.
        radius: Radius of the visual node item.
        color: Color of the visual node item.
        predicted: Whether this point is predicted.
        color_predicted: Whether to color predicted points.
        show_non_visible: Whether to show points where `visible` is False.
        callbacks: List of functions to call after we update to the `Point`.
    """

    def __init__(
        self,
        parent: QGraphicsObject,
        node: Node,
        point: Point,
        radius: float,
        color: list,
        predicted=False,
        color_predicted=False,
        show_non_visible=True,
        callbacks=None,
        *args,
        **kwargs,
    ):
        self._parent_instance = parent
        self.point = point
        self.node = node
        self.radius = radius
        self.color = color
        self.edges = []
        self.name = node.name
        self.predicted = predicted
        self.color_predicted = color_predicted
        self.show_non_visible = show_non_visible
        self.callbacks = [] if callbacks is None else callbacks
        self.dragParent = False

        super(QtNode, self).__init__(
            -self.radius,
            -self.radius,
            self.radius * 2,
            self.radius * 2,
            parent=parent,
            *args,
            **kwargs,
        )

        if self.name is not None:
            self.setToolTip(self.name)

        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        col_line = QColor(*self.color)

        if self.predicted:
            self.setFlag(QGraphicsItem.ItemIsMovable, False)

            pen_width = 1
            if self.node == self._parent_instance.instance.skeleton.nodes[0]:
                pen_width = 3

            if self.color_predicted:
                self.pen_default = QPen(col_line, pen_width)
            else:
                self.pen_default = QPen(QColor(250, 250, 10), pen_width)
            self.pen_default.setCosmetic(True)
            self.pen_missing = self.pen_default
            self.brush = QBrush(QColor(128, 128, 128, 128))
            self.brush_missing = self.brush
        else:
            self.setFlag(QGraphicsItem.ItemIsMovable)

            self.pen_default = QPen(col_line, 1)
            self.pen_default.setCosmetic(
                True
            )  # https://stackoverflow.com/questions/13120486/adjusting-qpen-thickness-when-scaling-qgraphicsview
            self.pen_missing = QPen(col_line, 1)
            self.pen_missing.setCosmetic(True)
            self.brush = QBrush(QColor(*self.color, a=128))
            self.brush_missing = QBrush(QColor(*self.color, a=0))

        self.setPos(self.point.x, self.point.y)
        self.updatePoint(user_change=False)

    def calls(self):
        """ Method to call all callbacks.
        """
        for callback in self.callbacks:
            if callable(callback):
                callback(self)

    def updatePoint(self, user_change: bool = False):
        """
        Method to update data for node/edge when node position is manipulated.

        Args:
            user_change: Whether this being called because of change by user.
        """
        x = self.scenePos().x()
        y = self.scenePos().y()

        context = self._parent_instance.player.context
        if user_change and context:
            context.setPointLocations(
                self._parent_instance.instance, {self.node.name: (x, y)}
            )
        self.show()

        if self.point.visible:
            radius = self.radius
            self.setPen(self.pen_default)
            self.setBrush(self.brush)
        else:
            radius = self.radius / 2.0
            self.setPen(self.pen_missing)
            self.setBrush(self.brush_missing)
            if not self.show_non_visible:
                self.hide()
        self.setRect(-radius, -radius, radius * 2, radius * 2)

        for edge in self.edges:
            edge.updateEdge(self)
            # trigger callbacks for other connected nodes
            edge.connected_to(self).calls()

        # trigger callbacks for this node
        self.calls()

    def toggleVisibility(self):
        context = self._parent_instance.player.context
        visible = not self.point.visible
        if context:
            context.setInstancePointVisibility(
                self._parent_instance.instance, self.node, visible
            )
        else:
            self.point.visible = visible

    def mousePressEvent(self, event):
        """ Custom event handler for mouse press.
        """
        # Do nothing if node is from predicted instance
        if self.parentObject().predicted:
            return

        self.setCursor(Qt.ArrowCursor)

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
                self.parentObject().updatePoints(complete=True, user_change=True)
            else:
                self.dragParent = False
                super(QtNode, self).mousePressEvent(event)
                self.updatePoint()

            self.point.complete = True  # FIXME: move to command
        elif event.button() == Qt.RightButton:
            # Right-click to toggle node as missing from this instance
            self.toggleVisibility()
            self.point.complete = True  # FIXME: move to command
            self.updatePoint(user_change=True)
        elif event.button() == Qt.MidButton:
            pass

    def mouseMoveEvent(self, event):
        """ Custom event handler for mouse move.
        """
        # print(event)
        if self.dragParent:
            self.parentObject().mouseMoveEvent(event)
        else:
            super(QtNode, self).mouseMoveEvent(event)
            self.updatePoint(
                user_change=False
            )  # don't count change until mouse release

    def mouseReleaseEvent(self, event):
        """ Custom event handler for mouse release.
        """
        # print(event)
        self.unsetCursor()
        if self.dragParent:
            self.parentObject().mouseReleaseEvent(event)
            self.parentObject().setSelected(False)
            self.parentObject().setFlag(QGraphicsItem.ItemIsMovable, False)
            self.parentObject().updatePoints(user_change=True)
        else:
            super(QtNode, self).mouseReleaseEvent(event)
            self.updatePoint(user_change=True)
        self.dragParent = False

    def wheelEvent(self, event):
        """Custom event handler for mouse scroll wheel."""
        if self.dragParent:
            angle = event.delta() / 20 + self.parentObject().rotation()
            self.parentObject().setRotation(angle)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Custom event handler to emit signal on event."""
        scene = self.scene()
        if scene is not None:
            view = scene.views()[0]
            view.instanceDoubleClicked.emit(self.parentObject().instance)


class QtEdge(QGraphicsLineItem):
    """
    QGraphicsLineItem to handle display of edge between skeleton instance nodes.

    Args:
        parent: `QGraphicsObject` which will contain this item.
        src: The `QtNode` source node for the edge.
        dst: The `QtNode` destination node for the edge.
        color: Color as (r, g, b) tuple.
        show_non_visible: Whether to show "non-visible" nodes/edges.
    """

    def __init__(
        self,
        parent: QGraphicsObject,
        src: QtNode,
        dst: QtNode,
        color: tuple,
        show_non_visible: bool = True,
        *args,
        **kwargs,
    ):
        self.parent = parent
        self.src = src
        self.dst = dst
        self.show_non_visible = show_non_visible

        super(QtEdge, self).__init__(
            self.src.point.x,
            self.src.point.y,
            self.dst.point.x,
            self.dst.point.y,
            parent=parent,
            *args,
            **kwargs,
        )

        pen = QPen(QColor(*color), 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.full_opacity = 1

    def connected_to(self, node: QtNode):
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

    def angle_to(self, node: QtNode) -> float:
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

    def updateEdge(self, node: QtNode):
        """
        Updates the visual display of node.

        Args:
            node: The node to update.

        Returns:
            None.
        """
        if self.src.point.visible and self.dst.point.visible:
            self.full_opacity = 1
        else:
            self.full_opacity = 0.5 if self.show_non_visible else 0

        if self.parent.edges_shown:
            self.setOpacity(self.full_opacity)

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

    It should be instantiated with an `Instance` and added to the relevant
    `QGraphicsScene`.

    When instantiated, it creates `QtNode`, `QtEdge`, and
    `QtNodeLabel` items as children of itself.

    Args:
        instance: The :class:`Instance` to show.
        predicted: Whether this is a predicted instance.
        color_predicted: Whether to show predicted instance in color.
        color: Color of the visual item.
        markerRadius: Radius of nodes.
        show_non_visible: Whether to show "non-visible" nodes/edges.

    """

    def __init__(
        self,
        instance: Instance = None,
        player: Optional[QtVideoPlayer] = None,
        predicted=False,
        color_predicted=False,
        color=(0, 114, 189),
        markerRadius=4,
        show_non_visible=True,
        *args,
        **kwargs,
    ):
        super(QtInstance, self).__init__(*args, **kwargs)
        self.player = player
        self.skeleton = instance.skeleton
        self.instance = instance
        self.predicted = predicted
        self.color_predicted = color_predicted
        self.show_non_visible = show_non_visible
        self.selectable = not self.predicted or self.color_predicted
        self.color = color
        self.markerRadius = markerRadius

        self.nodes = {}
        self.edges = []
        self.edges_shown = True
        self.labels = {}
        self.labels_shown = True
        self._selected = False
        self._bounding_rect = QRectF()

        if self.predicted:
            self.setZValue(0)
            if not self.color_predicted:
                self.color = (128, 128, 128)
        else:
            self.setZValue(1)

        # Add box to go around instance
        self.box = QGraphicsRectItem(parent=self)
        box_pen = QPen(QColor(*self.color), 1)
        box_pen.setStyle(Qt.DashLine)
        box_pen.setCosmetic(True)
        self.box.setPen(box_pen)

        self.track_label = QtTextWithBackground(parent=self)
        self.track_label.setDefaultTextColor(QColor(*self.color))

        instance_label_text = ""
        if self.instance.track is not None:
            track_name = self.instance.track.name
        else:
            track_name = "[none]"
        instance_label_text += f"<b>Track</b>: {track_name}"
        if hasattr(self.instance, "score"):
            instance_label_text += (
                f"<br /><b>Prediction Score</b>: {round(self.instance.score, 2)}"
            )
        self.track_label.setHtml(instance_label_text)

        # Add nodes
        for (node, point) in self.instance.nodes_points:
            node_item = QtNode(
                parent=self,
                node=node,
                point=point,
                predicted=self.predicted,
                color_predicted=self.color_predicted,
                color=self.color,
                radius=self.markerRadius,
                show_non_visible=self.show_non_visible,
            )

            self.nodes[node.name] = node_item

        # Add edges
        for (src, dst) in self.skeleton.edge_names:
            # Make sure that both nodes are present in this instance before drawing edge
            if src in self.nodes and dst in self.nodes:
                edge_item = QtEdge(
                    parent=self,
                    src=self.nodes[src],
                    dst=self.nodes[dst],
                    color=self.color,
                    show_non_visible=self.show_non_visible,
                )
                self.nodes[src].edges.append(edge_item)
                self.nodes[dst].edges.append(edge_item)
                self.edges.append(edge_item)

        # Add labels to nodes
        # We do this after adding edges so that we can position labels to avoid overlap
        if not self.predicted:
            for node in self.nodes.values():
                node_label = QtNodeLabel(node, predicted=self.predicted, parent=self)
                node_label.adjustPos()

                self.labels[node.name] = node_label
                # add callback to adjust position of label after node has moved
                node.callbacks.append(node_label.adjustPos)
                node.callbacks.append(self.updateBox)

        # Update size of box so it includes all the nodes/edges
        self.updateBox()

    def updatePoints(self, complete: bool = False, user_change: bool = False):
        """
        Updates data and display for all points in skeleton.

        This is called any time the skeleton is manipulated as a whole.

        Args:
            complete: Whether to update all nodes by setting "completed"
                attribute.
            user_change: Whether method is called because of change made by
                user.

        Returns:
            None.
        """

        # Update the position for each node
        context = self.player.context
        if user_change and context:
            new_data = {
                node_item.node.name: (
                    node_item.scenePos().x(),
                    node_item.scenePos().y(),
                )
                for node_item in self.nodes.values()
            }
            context.setPointLocations(self.instance, new_data)

        for node_item in self.nodes.values():
            node_item.setPos(node_item.point.x, node_item.point.y)
            if complete:
                # FIXME: move to command
                node_item.point.complete = True
        # Wait to run callbacks until all nodes are updated
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

    def getPointsBoundingRect(self) -> QRectF:
        """Returns a rect which contains all the nodes in the skeleton."""
        rect = None
        for item in self.edges:
            rect = (
                item.boundingRect()
                if rect is None
                else rect.united(item.boundingRect())
            )
        return rect

    def updateBox(self, *args, **kwargs):
        """
        Updates the box drawn around a selected skeleton.

        This updates both the box attribute stored and the visual box.
        The box attribute is used to determine whether a click should
        select this instance.
        """
        # Only show box if instance is selected
        op = 0.7 if self._selected else 0
        self.box.setOpacity(op)
        # Update the position for the box
        rect = self.getPointsBoundingRect()
        if rect is not None:
            self._bounding_rect = rect
            rect = rect.marginsAdded(QMarginsF(10, 10, 10, 10))
            self.box.setRect(rect)
            self.track_label.setOpacity(op)
            self.track_label.setPos(rect.bottomLeft() + QPointF(0, 5))

    @property
    def selected(self):
        """Whether instance is selected."""
        return self._selected

    @selected.setter
    def selected(self, selected: bool):
        """Sets select-state for instance."""
        self._selected = selected
        # Update the selection box for this skeleton instance
        self.updateBox()

    def showLabels(self, show: bool):
        """
        Draws/hides the labels for this skeleton instance.

        Args:
            show: Show labels if True, hide them otherwise.
        """
        op = 1 if show else 0
        for label in self.labels.values():
            label.setOpacity(op)
        self.labels_shown = show

    def showEdges(self, show: bool):
        """
        Draws/hides the edges for this skeleton instance.

        Args:
            show: Show edges if True, hide them otherwise.
        """
        for edge in self.edges:
            op = edge.full_opacity if show else 0
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


class QtTextWithBackground(QGraphicsTextItem):
    """
    Inherits methods/behavior of `QGraphicsTextItem`, but with background box.

    Color of background box is light or dark depending on the text color.
    """

    def __init__(self, *args, **kwargs):
        super(QtTextWithBackground, self).__init__(*args, **kwargs)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)

    def boundingRect(self):
        """ Method required by Qt.
        """
        return super(QtTextWithBackground, self).boundingRect()

    def paint(self, painter, option, *args, **kwargs):
        """ Method required by Qt.
        """
        text_color = self.defaultTextColor()
        brush = painter.brush()
        background_color = "white" if text_color.lightnessF() < 0.4 else "black"
        background_color = QColor(background_color, a=0.5)
        painter.setBrush(QBrush(background_color))
        painter.drawRect(self.boundingRect())
        painter.setBrush(brush)
        super(QtTextWithBackground, self).paint(painter, option, *args, **kwargs)


def video_demo(labels, standalone=False):
    """Demo function for showing (first) video from dataset."""
    video = labels.videos[0]
    if standalone:
        app = QApplication([])
    window = QtVideoPlayer(video=video)

    window.changedPlot.connect(
        lambda vp, idx, select_idx: plot_instances(vp.view.scene, idx, labels, video)
    )

    window.show()
    window.plot()

    if standalone:
        app.exec_()


def plot_instances(scene, frame_idx, labels, video=None, fixed=True):
    """Demo function for plotting instances."""
    from sleap.gui.overlays.tracks import TrackColorManager

    video = labels.videos[0]
    color_manager = TrackColorManager(labels=labels)
    lfs = labels.find(video, frame_idx)

    if not lfs:
        return

    labeled_frame = lfs[0]

    count_no_track = 0
    for i, instance in enumerate(labeled_frame.instances_to_show):
        if instance.track in labels.tracks:
            pseudo_track = instance.track
        else:
            # Instance without track
            pseudo_track = len(labels.tracks) + count_no_track
            count_no_track += 1

        # Plot instance
        inst = QtInstance(
            instance=instance,
            color=color_manager.get_color(pseudo_track),
            predicted=fixed,
            color_predicted=True,
            show_non_visible=False,
        )
        inst.showLabels(False)
        scene.addItem(inst)
        inst.updatePoints()


if __name__ == "__main__":

    import argparse
    from sleap.io.dataset import Labels

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    args = parser.parse_args()

    labels = Labels.load_json(args.data_path)
    video_demo(labels, standalone=True)

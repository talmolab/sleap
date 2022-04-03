"""
Module for showing and manipulating skeleton instances within a video.

All interactions should go through `QtVideoPlayer`.

Example usage: ::

    >>> my_video = Video(...)
    >>> my_instance = Instance(...)

    >>> vp = QtVideoPlayer(video=my_video)
    >>> vp.addInstance(instance=my_instance, color=(r, g, b))

"""
from collections import deque


# FORCE_REQUESTS controls whether we emit a signal to process frame requests
# if we haven't processed any for a certain amount of time.
# Usually the processing gets triggered by a timer but if the user is (e.g.)
# dragging the mouse, the timer doesn't trigger.
# FORCE_REQUESTS lets us update the frames in real time, assuming the load time
# is short enough to do that.

FORCE_REQUESTS = True


from PySide2 import QtWidgets, QtCore

from PySide2.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
)
from PySide2.QtGui import QImage, QPixmap, QPainter, QPainterPath, QTransform
from PySide2.QtGui import QPen, QBrush, QColor, QFont, QPolygonF
from PySide2.QtGui import QKeyEvent, QMouseEvent, QKeySequence
from PySide2.QtCore import Qt, QRectF, QPointF, QMarginsF, QLineF

import atexit
import math
import time
import numpy as np

from typing import Callable, List, Optional, Union

from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject

from PySide2.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QGraphicsRectItem,
    QGraphicsPolygonItem,
)

import sleap
from sleap.prefs import prefs
from sleap.skeleton import Node
from sleap.instance import Instance, Point
from sleap.io.video import Video
from sleap.gui.widgets.slider import VideoSlider
from sleap.gui.state import GuiState
from sleap.gui.color import ColorManager
from sleap.gui.shortcuts import Shortcuts

import qimage2ndarray


class LoadImageWorker(QtCore.QObject):
    """
    Object to load video frames in background thread.

    Requests to load a frame image are sent by calling the `request` method with
    the frame idx; the video attribute should already be set to the correct
    video.

    These requests are added to a FILO queue polled by the `doProcessing`
    method, called whenever there's time during the Qt event loop.
    (It's also added to the event queue if it hasn't been called for a while
    and we get a request, since the timer doesn't seem to emit events if the
    user has been holding down the mouse for a while.)

    The actual frame loading is wrapped with a mutex lock so that we only load
    a single frame at a time; this helps us not get a bunch of older frame
    requests running concurrently.

    Once the frame loads, the `QImage` is sent via the `result` signal.
    (Qt handles the cross-thread communication if we use signals.)
    """

    result = QtCore.Signal(QImage)
    process = QtCore.Signal()

    load_queue = []
    video = None
    _last_process_time = 0
    _force_request_wait_time = 1
    _recent_load_times = None

    def __init__(self, *args, **kwargs):
        super(LoadImageWorker, self).__init__(*args, **kwargs)

        self._processing_mutex = QtCore.QMutex()
        self._recent_load_times = deque(maxlen=5)

        # Connect signal to processing function so that we can add processing
        # event to event queue from the request handler.
        self.process.connect(self.doProcessing)

        # Start timer which will trigger processing events every 20 ms when we're free
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.doProcessing)
        self.timer.start(20)

    def doProcessing(self):
        self._last_process_time = time.time()

        if not self.load_queue:
            return

        # Use a mutex lock to ensure that we're only loading one frame at a time
        self._processing_mutex.lock()

        # Maybe we had to wait to acquire the lock, so make sure there are still
        # frames to load
        if not self.load_queue:
            return

        # Get the most recent request and clear all the others, since there's no
        # reason to load frames for older requests
        frame_idx = self.load_queue[-1]
        self.load_queue = []

        try:

            t0 = time.time()

            # Get image data
            frame = self.video.get_frame(frame_idx)

            self._recent_load_times.append(time.time() - t0)

            # Set the time to wait before forcing a load request to a little
            # longer than the average time it recently took to load a frame
            avg_load_time = sum(self._recent_load_times) / len(self._recent_load_times)
            self._force_request_wait_time = avg_load_time

        except Exception:
            frame = None

        # Release the lock so other threads can start processing frame requests
        self._processing_mutex.unlock()

        if frame is not None:
            # Convert ndarray to QImage
            qimage = qimage2ndarray.array2qimage(frame)

            # Emit result
            self.result.emit(qimage)

    def request(self, frame_idx):
        # Add request to the queue so that we can just process the most recent.
        self.load_queue.append(frame_idx)

        # If we haven't processed a request for a certain amount of time,
        # then trigger a processing event now. This helps when the user has been
        # continuously changing frames for a while (i.e., dragging on seekbar
        # or holding down arrow key).

        since_last = time.time() - self._last_process_time

        if FORCE_REQUESTS:
            if since_last > self._force_request_wait_time:
                self._last_process_time = time.time()
                self.process.emit()


class QtVideoPlayer(QWidget):
    """
    Main QWidget for displaying video with skeleton instances.

    Signals:
        * changedPlot: Emitted whenever the plot is redrawn

    Attributes:
        video: The :class:`Video` to display
        color_manager: A :class:`ColorManager` object which determines
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

        self.setAcceptDrops(True)

        self._shift_key_down = False

        self.color_manager = color_manager or ColorManager()
        self.state = state or GuiState()
        self.shortcuts = Shortcuts()
        self.context = context
        self.view = GraphicsView(self.state, self)
        self.video = None

        self.seekbar = VideoSlider()
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

        self._register_shortcuts()

        if self.context:
            self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.show_contextual_menu)
            self.is_menu_enabled = True
        else:
            self.is_menu_enabled = False

        self.seekbar.valueChanged.connect(
            lambda e: self.state.set("frame_idx", self.seekbar.value())
        )

        # Make worker thread to load images in the background
        self._loader_thread = QtCore.QThread()
        self._video_image_loader = LoadImageWorker()
        self._video_image_loader.moveToThread(self._loader_thread)
        self._loader_thread.start()

        # Connect signal so that image will be shown after it's loaded
        self._video_image_loader.result.connect(
            lambda qimage: self.view.setImage(qimage)
        )

        def update_selection_state(a, b):
            self.state.set("frame_range", (a, b + 1))
            self.state.set("has_frame_range", (a < b))

        self.seekbar.selectionChanged.connect(update_selection_state)

        self.state.connect("frame_idx", lambda idx: self.plot())
        self.state.connect("frame_idx", lambda idx: self.seekbar.setValue(idx))
        self.state.connect("instance", self.view.selectInstance)

        self.state.connect("show instances", self.plot)
        self.state.connect("show labels", self.plot)
        self.state.connect("show edges", self.plot)
        self.state.connect("video", self.load_video)
        self.state.connect("fit", self.setFitZoom)

        self.view.show()

        # Call cleanup method when application exits to end worker thread
        self.destroyed.connect(self.cleanup)
        atexit.register(self.cleanup)

        if video is not None:
            self.load_video(video)

    def cleanup(self):
        self._loader_thread.quit()
        self._loader_thread.wait()

    def dragEnterEvent(self, event):
        if self.parentWidget():
            self.parentWidget().dragEnterEvent(event)

    def dropEvent(self, event):
        if self.parentWidget():
            self.parentWidget().dropEvent(event)

    def _load_and_show_requested_image(self, frame_idx):
        # Get image data
        try:
            frame = self.video.get_frame(frame_idx)
        except:
            frame = None

        if frame is not None:
            # Convert ndarray to QImage
            qimage = qimage2ndarray.array2qimage(frame)

            # Display image
            self.view.setImage(qimage)

    def _register_shortcuts(self):
        self._shortcut_triggers = dict()

        def frame_step(step, enable_shift_selection):
            if self.video:
                before_frame_idx = self.state["frame_idx"]
                self.state.increment("frame_idx", step=step, mod=self.video.frames)
                # only use shift for selection if not part of shortcut
                if enable_shift_selection and self._shift_key_down:
                    self._select_on_possible_frame_movement(before_frame_idx)

        def add_shortcut(key, step):
            # Register shortcut and have it trigger frame_step action
            shortcut = QtWidgets.QShortcut(self.shortcuts[key], self)
            shortcut.activated.connect(lambda x=step: frame_step(x, False))
            self._shortcut_triggers[key] = shortcut

            # If shift isn't part of shortcut, then we want to allow
            # shift + shortcut for movement + selection.

            # We use hack of convert to/from the string representation of
            # shortcut to determine if shift is in shortcut and to add it.
            no_shift = "Shift" not in shortcut.key().toString()

            if no_shift:
                # Make shortcut + shift key sequence
                shortcut_seq_with_shift = QKeySequence(
                    f"Shift+{shortcut.key().toString()}"
                )

                # Register this new shortcut, enabling shift selection
                shortcut = QtWidgets.QShortcut(shortcut_seq_with_shift, self)
                shortcut.activated.connect(lambda x=step: frame_step(x, True))
                self._shortcut_triggers[key + "_shift_selection"] = shortcut

        add_shortcut("frame next", 1)
        add_shortcut("frame prev", -1)
        add_shortcut("frame next medium step", prefs["medium step size"])
        add_shortcut("frame prev medium step", -prefs["medium step size"])
        add_shortcut("frame next large step", prefs["large step size"])
        add_shortcut("frame prev large step", -prefs["large step size"])

    def setSeekbarSelection(self, a: int, b: int):
        self.seekbar.setSelection(a, b)

    def show_contextual_menu(self, where: QtCore.QPoint):
        if not self.is_menu_enabled:
            return

        scene_pos = self.view.mapToScene(where)
        menu = QtWidgets.QMenu()

        menu.addAction("Add Instance:").setEnabled(False)

        menu.addAction("Default", lambda: self.context.newInstance(init_method="best"))

        menu.addAction(
            "Average",
            lambda: self.context.newInstance(
                init_method="template", location=scene_pos
            ),
        )

        menu.addAction(
            "Force Directed",
            lambda: self.context.newInstance(
                init_method="force_directed", location=scene_pos
            ),
        )

        menu.addAction(
            "Copy Prior Frame",
            lambda: self.context.newInstance(init_method="prior_frame"),
        )

        menu.addAction(
            "Random",
            lambda: self.context.newInstance(init_method="random", location=scene_pos),
        )

        menu.exec_(self.mapToGlobal(where))

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
        self.seekbar.resizeEvent()

        if plot:
            self.plot()

    def reset(self):
        """Reset viewer by removing all video data."""
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
        if instance.instance.n_visible_points > 0:
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

        # Clear exiting objects before drawing instances
        self.view.clear()

        # Emit signal for the instances to be drawn for this frame
        self.changedPlot.emit(self, idx, self.state["instance"])

        # Request for the image to load and be shown for this frame
        # (note that we're calling method directly rather than connecting
        # the method to a signal because Qt was holding onto the signal events
        # for too long before they were received by the loader).
        self._video_image_loader.video = self.video
        self._video_image_loader.request(idx)

    def showInstances(self, show):
        """Show/hide all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.instances:
            inst.showInstances(show)
        for inst in self.predicted_instances:
            inst.showInstances(show)

    def showLabels(self, show):
        """Show/hide node labels for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.selectable_instances:
            inst.showLabels(show)

    def showEdges(self, show):
        """Show/hide node edges for all instances in viewer.

        Args:
            show: Show if True, hide otherwise.
        """
        for inst in self.selectable_instances:
            inst.showEdges(show)

    def highlightPredictions(self, highlight_text: str = ""):
        for inst in self.predicted_instances:
            inst.highlight = True
            inst.highlight_text = highlight_text

    def zoomToFit(self):
        """Zoom view to fit all instances."""
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
            If successful, we call ::

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

        elif event.key() == Qt.Key.Key_Home:
            self.state["frame_idx"] = 0

        elif event.key() == Qt.Key.Key_End and self.video:
            self.state["frame_idx"] = self.video.frames - 1

        elif event.key() == Qt.Key.Key_Escape:
            self.view.click_mode = ""
            self.state["instance"] = None

        elif event.key() == Qt.Key.Key_K:
            self.state["frame_idx"] = self.seekbar.getEndContiguousMark(
                self.state["frame_idx"]
            )
        elif event.key() == Qt.Key.Key_J:
            self.state["frame_idx"] = self.seekbar.getStartContiguousMark(
                self.state["frame_idx"]
            )
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
        if self._shift_key_down:
            self._select_on_possible_frame_movement(frame_t0)

    def _select_on_possible_frame_movement(self, before_frame_idx: int):
        if before_frame_idx != self.state["frame_idx"]:
            # If there's no select, start seekbar selection at frame before action
            start, end = self.seekbar.getSelection()
            if start == end:
                self.seekbar.startSelection(before_frame_idx)
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
    instanceDoubleClicked = QtCore.Signal(Instance, QMouseEvent)
    areaSelected = QtCore.Signal(float, float, float, float)
    pointSelected = QtCore.Signal(float, float)
    leftMouseButtonPressed = QtCore.Signal(float, float)
    rightMouseButtonPressed = QtCore.Signal(float, float)
    leftMouseButtonReleased = QtCore.Signal(float, float)
    rightMouseButtonReleased = QtCore.Signal(float, float)
    leftMouseButtonDoubleClicked = QtCore.Signal(float, float)
    rightMouseButtonDoubleClicked = QtCore.Signal(float, float)

    def __init__(self, state=None, player=None, *args, **kwargs):
        """https://github.com/marcel-goldschen-ohm/PyQtImageViewer/blob/master/QtImageViewer.py"""
        QGraphicsView.__init__(self)
        self.state = state or GuiState()

        self.player = player

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setAcceptDrops(True)

        self.scene.setBackgroundBrush(QBrush(QColor(Qt.black)))

        self._pixmapHandle = None

        self.setRenderHint(QPainter.Antialiasing)

        self.aspectRatioMode = Qt.KeepAspectRatio
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canZoom = True
        self.canPan = True
        self.click_mode = ""
        self.in_zoom = False

        self.zoomFactor = 1
        anchor_mode = QGraphicsView.AnchorUnderMouse
        self.setTransformationAnchor(anchor_mode)

        # Set icon as default background.
        self.setImage(QImage(sleap.util.get_package_file("sleap/gui/background.png")))

    def dragEnterEvent(self, event):
        if self.parentWidget():
            self.parentWidget().dragEnterEvent(event)

    def dropEvent(self, event):
        if self.parentWidget():
            self.parentWidget().dropEvent(event)

    def hasImage(self) -> bool:
        """Returns whether or not the scene contains an image pixmap."""
        return self._pixmapHandle is not None

    def clear(self):
        """Clears the displayed frame from the scene."""

        if self._pixmapHandle:
            # get the pixmap currently shown
            pixmap = self._pixmapHandle.pixmap()

        self.scene.clear()

        if self._pixmapHandle:
            # add the pixmap back
            self._pixmapHandle = self._add_pixmap(pixmap)

    def _add_pixmap(self, pixmap):
        """Adds a pixmap to the scene and transforms it to midpoint coordinates."""
        pixmap_graphics_item = self.scene.addPixmap(pixmap)

        transform = pixmap_graphics_item.transform()
        transform.translate(-0.5, -0.5)
        pixmap_graphics_item.setTransform(transform)

        return pixmap_graphics_item

    def setImage(self, image: Union[QImage, QPixmap, np.ndarray]):
        """
        Set the scene's current image pixmap to the input QImage or QPixmap.

        Args:
            image: The QPixmap or QImage to display.

        Raises:
            RuntimeError: If the input image is not QImage or QPixmap

        Returns:
            None.
        """
        if type(image) is np.ndarray:
            # Convert numpy array of frame image to QImage
            image = qimage2ndarray.array2qimage(image)

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
            self._pixmapHandle = self._add_pixmap(pixmap)

            # Ensure that image is behind everything else
            self._pixmapHandle.setZValue(-1)

        # Set scene size to image size, translated to midpoint coordinates.
        # (If we don't translate the rect, the image will be cut off by
        # 1/2 pixel at the top left and have a 1/2 pixel border at bottom right)
        rect = QRectF(pixmap.rect())
        rect.translate(-0.5, -0.5)
        self.setSceneRect(rect)
        self.updateViewer()

    def updateViewer(self):
        """Apply current zoom."""
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
        return list(filter(lambda x: x.predicted, self.all_instances))

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
        """Returns the index of the currently selected instance.
        If no instance selected, returns None.
        """
        instances = self.all_instances
        if len(instances) == 0:
            return None
        for idx, instance in enumerate(instances):
            if instance.selected:
                return idx

    def getSelectionInstance(self) -> Optional[Instance]:
        """Returns the currently selected instance.
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
        """Maintain current zoom on resize."""
        self.updateViewer()

    def mousePressEvent(self, event):
        """Start mouse pan or zoom mode."""
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

            elif event.modifiers() == Qt.AltModifier:
                if self.canZoom:
                    self.in_zoom = True
                    self.setDragMode(QGraphicsView.RubberBandDrag)

            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        elif event.button() == Qt.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """Stop mouse pan or zoom mode (apply zoom if valid)."""
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())

        # check if mouse moved during click
        has_moved = event.pos() != self._down_pos
        if event.button() == Qt.LeftButton:

            if self.in_zoom:
                self.in_zoom = False
                zoom_rect = self.scene.selectionArea().boundingRect()
                self.scene.setSelectionArea(QPainterPath())  # clear selection
                self.zoomToRect(zoom_rect)

            elif self.click_mode == "":
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

            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseMoveEvent(self, event):
        # re-enable contextual menu if necessary
        if self.player:
            self.player.is_menu_enabled = True
        QGraphicsView.mouseMoveEvent(self, event)

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
        """Clear zoom stack. Doesn't update display."""
        self.zoomFactor = 1

    @staticmethod
    def getInstancesBoundingRect(
        instances: List["QtInstance"], margin: float = 0.0
    ) -> QRectF:
        """Return a rectangle containing all instances.

        Args:
            instances: List of QtInstance objects.
            margin: Margin for padding the rectangle. Padding is applied equally on all
                sides.

        Returns:
            The `QRectF` which contains all of the instances.

        Notes:
            The returned rectangle will be null if the instance list is empty.
        """
        rect = QRectF()
        for item in instances:
            rect = rect.united(item.boundingRect())
        if margin > 0 and not rect.isNull():
            rect = rect.marginsAdded(QMarginsF(margin, margin, margin, margin))
        return rect

    def instancesBoundingRect(self, margin: float = 0) -> QRectF:
        """
        Returns a rect which contains all displayed skeleton instances.

        Args:
            margin: Margin for padding the rect.
        Returns:
            The `QRectF` which contains the skeleton instances.
        """
        return GraphicsView.getInstancesBoundingRect(self.all_instances, margin=margin)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Custom event handler, clears zoom."""
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:

            if event.modifiers() == Qt.AltModifier:
                if self.canZoom:
                    self.clearZoom()
                    self.updateViewer()

            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def wheelEvent(self, event):
        """Custom event handler. Zoom in/out based on scroll wheel change."""
        # zoom on wheel when no mouse buttons are pressed
        if event.buttons() == Qt.NoButton:
            angle = event.angleDelta().y()
            factor = 1.1 if angle > 0 else 0.9

            self.zoomFactor = max(factor * self.zoomFactor, 1)
            self.updateViewer()

        # Trigger wheelEvent for all child elements. This is a bit of a hack.
        # We can't use QGraphicsView.wheelEvent(self, event) since that will scroll
        # view.
        # We want to trigger for all children, since wheelEvent should continue rotating
        # an skeleton even if the skeleton node/node label is no longer under the
        # cursor.
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
        fontSize: Size of the label text.
    """

    def __init__(
        self,
        node: Node,
        parent: QGraphicsObject,
        predicted: bool = False,
        fontSize: float = 12,
        show_non_visible: bool = True,
        *args,
        **kwargs,
    ):
        self.node = node
        self.text = node.name
        self.predicted = predicted
        self.show_non_visible = show_non_visible
        self._parent_instance = parent
        super(QtNodeLabel, self).__init__(self.text, parent=parent, *args, **kwargs)

        self._anchor_x = self.pos().x()
        self._anchor_x = self.pos().y()

        self._base_font = QFont()
        self._base_font.setPixelSize(fontSize)
        self.setFont(self._base_font)

        # set color to match node color
        self.setDefaultTextColor(self.node.pen().color())

        # don't rescale when view is scaled (i.e., zoom)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        self.complete_color = QColor(80, 194, 159)
        self.incomplete_color = QColor(232, 45, 32)
        self.missing_color = QColor(128, 128, 128)
        self.missing_bg_color = QColor(0, 0, 0, a=100)

        self.adjustStyle()

    def adjustPos(self, *args, **kwargs):
        """Update the position of the label based on the position of the node.

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
        """Update visual display of the label and its node."""
        if self.predicted:
            self._base_font.setBold(False)
            self._base_font.setItalic(False)
            self.setFont(self._base_font)
            self.setDefaultTextColor(QColor(128, 128, 128))
        elif not self.node.point.visible:
            self._base_font.setBold(True)
            self._base_font.setItalic(True)
            self.setFont(self._base_font)
            self.setPlainText(self.node.name)
            self.setDefaultTextColor(self.missing_color)
        elif self.node.point.complete:
            self._base_font.setBold(True)
            self._base_font.setItalic(False)
            self.setPlainText(self.node.name)
            self.setFont(self._base_font)
            self.setDefaultTextColor(self.complete_color)  # greenish
            # FIXME: Adjust style of node here as well?
            # self.node.setBrush(complete_color)
        else:
            self._base_font.setBold(False)
            self._base_font.setItalic(False)
            self.setPlainText(self.node.name)
            self.setFont(self._base_font)
            self.setDefaultTextColor(self.incomplete_color)  # redish

    def paint(self, painter, option, widget):
        """Paint overload."""
        if not self.node.point.visible:
            if self.show_non_visible:
                # Add background box for missing nodes
                painter.fillRect(option.rect, self.missing_bg_color)
            else:
                self.hide()
        super(QtNodeLabel, self).paint(painter, option, widget)

    def mousePressEvent(self, event):
        """Pass events along so that clicking label is like clicking node."""
        self.setCursor(Qt.ArrowCursor)
        self.node.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Pass events along so that clicking label is like clicking node."""
        self.node.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Pass events along so that clicking label is like clicking node."""
        self.unsetCursor()
        self.node.mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Pass events along so that clicking label is like clicking node."""
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
        predicted: Whether this point is predicted.
        show_non_visible: Whether to show points where `visible` is False.
        callbacks: List of functions to call after we update to the `Point`.
    """

    def __init__(
        self,
        parent: QGraphicsObject,
        player: QtVideoPlayer,
        node: Node,
        point: Point,
        radius: float,
        predicted=False,
        show_non_visible=True,
        callbacks=None,
        *args,
        **kwargs,
    ):
        self._parent_instance = parent
        self.player = player
        self.point = point
        self.node = node
        self.radius = radius
        self.color_manager = self.player.color_manager
        self.color = self.color_manager.get_item_color(
            self.node, self._parent_instance.instance
        )
        self.edges = []
        self.name = node.name
        self.predicted = predicted
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
            if hasattr(self.point, "score"):
                tt_text = f"{self.name}\n(score: {self.point.score:.2f})"
            else:
                tt_text = self.name
            self.setToolTip(tt_text)

        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        line_color = QColor(*self.color)

        pen_width = self.color_manager.get_item_pen_width(
            self.node, self._parent_instance.instance
        )

        if self.predicted:
            self.setFlag(QGraphicsItem.ItemIsMovable, False)

            self.pen_default = QPen(line_color, pen_width)
            self.pen_default.setCosmetic(True)
            self.pen_missing = self.pen_default

            self.brush = QBrush(QColor(128, 128, 128, 128))
            self.brush_missing = self.brush
        else:
            self.setFlag(QGraphicsItem.ItemIsMovable)

            self.pen_default = QPen(line_color, pen_width)
            self.pen_default.setCosmetic(
                True
            )  # https://stackoverflow.com/questions/13120486/adjusting-qpen-thickness-when-scaling-qgraphicsview
            self.pen_missing = QPen(line_color, 1)  # thin border
            self.pen_missing.setCosmetic(True)
            self.brush = QBrush(QColor(*self.color, a=128))
            self.brush_missing = QBrush(QColor(*self.color, a=0))  # no fill

        self.setPos(self.point.x, self.point.y)
        self.updatePoint(user_change=False)

    def calls(self):
        """Method to call all callbacks."""
        for callback in self.callbacks:
            if callable(callback):
                callback(self)

    @property
    def visible_radius(self):
        if self.point.visible:
            return self.radius / self.player.view.zoomFactor
        else:
            return self.radius / (2.0 * self.player.view.zoomFactor)  # smaller marker

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
            radius = self.radius / 2.0  # smaller marker
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
        """Custom event handler for mouse press."""
        # Do nothing if node is from predicted instance
        if self.parentObject().predicted:
            return

        self.setCursor(Qt.ArrowCursor)

        if event.button() == Qt.LeftButton:
            # Select instance this nodes belong to.
            self.parentObject().player.state["instance"] = self.parentObject().instance

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
            # Select instance this nodes belong to.
            self.parentObject().player.state["instance"] = self.parentObject().instance

            # Right-click to toggle node as missing from this instance
            self.toggleVisibility()
            # Disable contextual menu for right clicks on node
            self.player.is_menu_enabled = False

            self.point.complete = True  # FIXME: move to command
            self.updatePoint(user_change=True)
        elif event.button() == Qt.MidButton:
            pass

    def mouseMoveEvent(self, event):
        """Custom event handler for mouse move."""
        # print(event)
        if self.dragParent:
            self.parentObject().mouseMoveEvent(event)
        else:
            super(QtNode, self).mouseMoveEvent(event)
            self.updatePoint(
                user_change=False
            )  # don't count change until mouse release

    def mouseReleaseEvent(self, event):
        """Custom event handler for mouse release."""
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

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Custom event handler to emit signal on event."""
        scene = self.scene()
        if scene is not None:
            view = scene.views()[0]
            view.instanceDoubleClicked.emit(self.parentObject().instance, event)


class QtEdge(QGraphicsPolygonItem):
    """
    QGraphicsLineItem to handle display of edge between skeleton instance nodes.

    Args:
        parent: `QGraphicsObject` which will contain this item.
        src: The `QtNode` source node for the edge.
        dst: The `QtNode` destination node for the edge.
        show_non_visible: Whether to show "non-visible" nodes/edges.
    """

    def __init__(
        self,
        parent: QGraphicsObject,
        player: QtVideoPlayer,
        src: QtNode,
        dst: QtNode,
        show_non_visible: bool = True,
        *args,
        **kwargs,
    ):
        self.parent = parent
        self.player = player
        self.src = src
        self.dst = dst
        self.show_non_visible = show_non_visible

        super(QtEdge, self).__init__(
            polygon=QPolygonF(),
            parent=parent,
            *args,
            **kwargs,
        )

        self.setLine(
            QLineF(
                self.src.point.x,
                self.src.point.y,
                self.dst.point.x,
                self.dst.point.y,
            )
        )

        edge_pair = (src.node, dst.node)
        color = player.color_manager.get_item_color(edge_pair, parent.instance)
        pen_width = player.color_manager.get_item_pen_width(edge_pair, parent.instance)
        pen = QPen(QColor(*color), pen_width)
        pen.setCosmetic(True)

        brush = QBrush(QColor(*color, a=128))

        self.setPen(pen)
        self.setBrush(brush)
        self.full_opacity = 1

    def line(self):
        return self._line

    def setLine(self, line):
        self._line = line
        polygon = QPolygonF()

        if self.player.state.get("edge style", default="").lower() == "wedge":

            r = self.src.visible_radius / 2.0

            norm_a = line.normalVector()
            norm_a.setLength(r)

            norm_b = line.normalVector()
            norm_b.setLength(-r)

            polygon.append(norm_a.p2())
            polygon.append(line.p2())
            polygon.append(norm_b.p2())
            polygon.append(norm_a.p2())

        else:
            polygon.append(line.p1())
            polygon.append(line.p2())

        self.setPolygon(polygon)

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
        markerRadius: Radius of nodes.
        nodeLabelSize: Font size of node labels.
        show_non_visible: Whether to show "non-visible" nodes/edges.
    """

    def __init__(
        self,
        instance: Instance = None,
        player: Optional[QtVideoPlayer] = None,
        markerRadius=4,
        nodeLabelSize=12,
        show_non_visible=True,
        *args,
        **kwargs,
    ):
        super(QtInstance, self).__init__(*args, **kwargs)
        self.player = player
        self.skeleton = instance.skeleton
        self.instance = instance
        self.predicted = hasattr(instance, "score")

        color_manager = self.player.color_manager
        color = color_manager.get_item_color(self.instance)

        self.show_non_visible = show_non_visible
        self.selectable = not self.predicted or color_manager.color_predicted
        self.markerRadius = markerRadius
        self.nodeLabelSize = nodeLabelSize

        self.nodes = {}
        self.edges = []
        self.edges_shown = True
        self.labels = {}
        self.labels_shown = True
        self._selected = False
        self._bounding_rect = QRectF()

        # Show predicted instances behind non-predicted ones
        self.setZValue(1 if self.predicted else 2)

        if not self.predicted:
            # Initialize missing nodes with random points marked as non-visible.
            self.instance.fill_missing(
                max_x=self.player.video.width, max_y=self.player.video.height
            )

        # Add box to go around instance for selection
        self.box = QGraphicsRectItem(parent=self)
        box_pen_width = color_manager.get_item_pen_width(self.instance)
        box_pen = QPen(QColor(*color), box_pen_width)
        box_pen.setStyle(Qt.DashLine)
        box_pen.setCosmetic(True)
        self.box.setPen(box_pen)

        # Add label for highlighted instance
        self.highlight_label = QtTextWithBackground(parent=self)
        self.highlight_label.setDefaultTextColor(QColor("yellow"))
        font = self.highlight_label.font()
        font.setPointSize(10)
        self.highlight_label.setFont(font)
        self.highlight_label.setOpacity(0.5)
        self.highlight_label.hide()

        # Add box to go around instance for highlight
        self.highlight_box = QGraphicsRectItem(parent=self)
        highlight_pen = QPen(QColor("yellow"), 8)
        highlight_pen.setCosmetic(True)
        self.highlight_box.setPen(highlight_pen)

        self.track_label = QtTextWithBackground(parent=self)
        self.track_label.setDefaultTextColor(QColor(*color))

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
            if point.visible or self.show_non_visible:
                node_item = QtNode(
                    parent=self,
                    player=player,
                    node=node,
                    point=point,
                    predicted=self.predicted,
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
                    player=player,
                    src=self.nodes[src],
                    dst=self.nodes[dst],
                    show_non_visible=self.show_non_visible,
                )
                self.nodes[src].edges.append(edge_item)
                self.nodes[dst].edges.append(edge_item)
                self.edges.append(edge_item)

        # Add labels to nodes
        # We do this after adding edges so that we can position labels to avoid overlap
        if not self.predicted:
            for node in self.nodes.values():
                if node.point.visible or self.show_non_visible:
                    node_label = QtNodeLabel(
                        node,
                        predicted=self.predicted,
                        parent=self,
                        fontSize=self.nodeLabelSize,
                        show_non_visible=self.show_non_visible,
                    )
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
        points = [
            (node.scenePos().x(), node.scenePos().y()) for node in self.nodes.values()
        ]

        if len(points) == 0:
            # Check this condition with rect.isValid()
            top_left, bottom_right = QPointF(np.nan, np.nan), QPointF(np.nan, np.nan)
        else:
            top_left = QPointF(
                min((point[0] for point in points)), min((point[1] for point in points))
            )
            bottom_right = QPointF(
                max((point[0] for point in points)), max((point[1] for point in points))
            )
        rect = QRectF(top_left, bottom_right)
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
    def highlight(self):
        return self.highlight_box.opacity() > 0

    @highlight.setter
    def highlight(self, val):
        op = 0.2 if val else 0
        self.highlight_box.setOpacity(op)
        # Update the position for the box
        rect = self.getPointsBoundingRect()
        if rect is not None:
            self._bounding_rect = rect
            rect = rect.marginsAdded(QMarginsF(10, 10, 10, 10))
            self.highlight_box.setRect(rect)

            if rect.width() > 30:
                # Show label if highlight box isn't too small
                self.highlight_label.setVisible(op > 0)
                self.highlight_label.setPos(rect.topLeft() - QPointF(0, 10))
            else:
                self.highlight_label.hide()

    @property
    def highlight_text(self):
        return ""

    @highlight_text.setter
    def highlight_text(self, val):
        self.highlight_label.setPlainText(val)

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

    def showInstances(self, show: bool):
        """
        Shows/hides skeleton instance.

        Args:
            show: Show skeleton if True, hide otherwise.
        """
        self.setVisible(show)

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
        """Method required Qt to determine bounding rect for item."""
        return self._bounding_rect

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
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
        """Method required by Qt."""
        return super(QtTextWithBackground, self).boundingRect()

    def paint(self, painter, option, *args, **kwargs):
        """Method required by Qt."""
        text_color = self.defaultTextColor()
        brush = painter.brush()
        background_color = "white" if text_color.lightnessF() < 0.4 else "black"
        background_color = QColor(background_color, a=0.5)
        painter.setBrush(QBrush(background_color))
        painter.drawRect(self.boundingRect())
        painter.setBrush(brush)
        super(QtTextWithBackground, self).paint(painter, option, *args, **kwargs)


def video_demo(video=None, labels=None, standalone=False):
    """Demo function for showing video."""

    if not video and not labels:
        return

    if labels and not video:
        video = labels.videos[0]

    if standalone:
        app = QApplication([])
    window = QtVideoPlayer(video=video)

    if labels:
        window.changedPlot.connect(
            lambda vp, idx, select_idx: plot_instances(
                vp.view.scene, idx, labels, video
            )
        )

    window.show()
    window.plot()

    if standalone:
        app.exec_()


def plot_instances(scene, frame_idx, labels, video=None, fixed=True):
    """Demo function for plotting instances."""
    from sleap.gui.color import ColorManager

    video = labels.videos[0]
    color_manager = ColorManager(labels=labels)
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
            color=color_manager.get_track_color(pseudo_track),
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
    video_demo(labels=labels, standalone=True)

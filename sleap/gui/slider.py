"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2 import QtCore, QtWidgets
from PySide2.QtGui import QPen, QBrush, QColor, QKeyEvent, QPolygonF, QPainterPath

from sleap.gui.overlays.tracks import TrackColorManager

import attr
import itertools
import numpy as np
from typing import Dict, Iterable, List, Optional, Union


@attr.s(auto_attribs=True, cmp=False)
class SliderMark:
    """
    Class to hold data for an individual mark on the slider.

    Attributes:
        type: Type of the mark, options are:
            * "simple" (single value)
            * "filled" (single value)
            * "open" (single value)
            * "predicted" (single value)
            * "track" (range of values)
        val: Beginning of mark range
        end_val: End of mark range (for "track" marks)
        row: The row that the mark goes in; used for tracks.
        color: Color of mark, can be string or (r, g, b) tuple.
        filled: Whether the mark is shown filled (solid color).
    """

    type: str
    val: float
    end_val: float = None
    row: int = None
    track: "Track" = None
    _color: Union[tuple, str] = "black"

    @property
    def color(self):
        colors = dict(simple="black", filled="blue", open="blue", predicted="red")

        if self.type in colors:
            return colors[self.type]
        else:
            return self._color

    @color.setter
    def color(self, val):
        self._color = val

    @property
    def QColor(self):
        c = self.color
        if type(c) == str:
            return QColor(c)
        else:
            return QColor(*c)

    @property
    def filled(self):
        if self.type == "open":
            return False
        else:
            return True


class VideoSlider(QtWidgets.QGraphicsView):
    """Drop-in replacement for QSlider with additional features.

    Args:
        orientation: ignored (here for compatibility with QSlider)
        min: initial minimum value
        max: initial maximum value
        val: initial value
        marks: initial set of values to mark on slider
            this can be either
            * list of values to mark
            * list of (track, value)-tuples to mark
        color_manager: A :class:`TrackColorManager` which determines the
            color to use for "track"-type marks

    Signals:
        mousePressed: triggered on Qt event
        mouseMoved: triggered on Qt event
        mouseReleased: triggered on Qt event
        keyPress: triggered on Qt event
        keyReleased: triggered on Qt event
        valueChanged: triggered when value of slider changes
        selectionChanged: triggered when slider range selection changes
        heightUpdated: triggered when the height of slider changes
    """

    mousePressed = QtCore.Signal(float, float)
    mouseMoved = QtCore.Signal(float, float)
    mouseReleased = QtCore.Signal(float, float)
    keyPress = QtCore.Signal(QKeyEvent)
    keyRelease = QtCore.Signal(QKeyEvent)
    valueChanged = QtCore.Signal(int)
    selectionChanged = QtCore.Signal(int, int)
    heightUpdated = QtCore.Signal()

    def __init__(
        self,
        orientation=-1,  # for compatibility with QSlider
        min=0,
        max=100,
        val=0,
        marks=None,
        color_manager: Optional[TrackColorManager] = None,
        *args,
        **kwargs
    ):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )  # ScrollBarAsNeeded

        self._color_manager = color_manager

        self._track_rows = 0
        self._track_height = 3
        self._header_height = 0
        self._min_height = 19 + self._header_height

        # Add border rect
        outline_rect = QtCore.QRect(0, 0, 200, self._min_height - 3)
        self.outlineBox = self.scene.addRect(outline_rect)
        self.outlineBox.setPen(QPen(QColor("black")))

        # Add drag handle rect
        handle_width = 6
        handle_rect = QtCore.QRect(
            0, self._handleTop(), handle_width, self._handleHeight()
        )
        self.setMinimumHeight(self._min_height)
        self.setMaximumHeight(self._min_height)
        self.handle = self.scene.addRect(handle_rect)
        self.handle.setPen(QPen(QColor(80, 80, 80)))
        self.handle.setBrush(QColor(128, 128, 128, 128))

        # Add (hidden) rect to highlight selection
        self.select_box = self.scene.addRect(
            QtCore.QRect(0, 1, 0, outline_rect.height() - 2)
        )
        self.select_box.setPen(QPen(QColor(80, 80, 255)))
        self.select_box.setBrush(QColor(80, 80, 255, 128))
        self.select_box.hide()

        self.scene.setBackgroundBrush(QBrush(QColor(200, 200, 200)))

        self.clearSelection()
        self.setEnabled(True)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.setMarks(marks)

        pen = QPen(QColor(80, 80, 255), 0.5)
        pen.setCosmetic(True)
        self.poly = self.scene.addPath(QPainterPath(), pen, self.select_box.brush())
        self.headerSeries = dict()
        self.drawHeader()

    def _pointsToPath(self, points: List[QtCore.QPointF]) -> QPainterPath:
        """Converts list of `QtCore.QPointF`s to a `QPainterPath`."""
        path = QPainterPath()
        path.addPolygon(QPolygonF(points))
        return path

    def setTracksFromLabels(self, labels: "Labels", video: "Video"):
        """Set slider marks using track information from `Labels` object.

        Note that this is the only method coupled to a SLEAP object.

        Args:
            labels: the dataset with tracks and labeled frames
            video: the video for which to show marks

        Returns:
            None
        """

        if self._color_manager is None:
            self._color_manager = TrackColorManager(labels=labels)

        lfs = labels.find(video)

        slider_marks = []
        track_row = 0

        # Add marks with track
        track_occupancy = labels.get_track_occupany(video)
        for track in labels.tracks:
            if track in track_occupancy and not track_occupancy[track].is_empty:
                for occupancy_range in track_occupancy[track].list:
                    slider_marks.append(
                        SliderMark(
                            "track",
                            val=occupancy_range[0],
                            end_val=occupancy_range[1],
                            row=track_row,
                            color=self._color_manager.get_color(track),
                        )
                    )
                track_row += 1

        # Add marks without track
        if None in track_occupancy:
            for occupancy_range in track_occupancy[None].list:
                for val in range(*occupancy_range):
                    slider_marks.append(SliderMark("simple", val=val))

        # list of frame_idx for simple markers for labeled frames
        labeled_marks = [lf.frame_idx for lf in lfs]
        user_labeled = [lf.frame_idx for lf in lfs if len(lf.user_instances)]

        for frame_idx in labels.get_video_suggestions(video):
            if frame_idx in user_labeled:
                mark_type = "filled"
            elif frame_idx in labeled_marks:
                mark_type = "predicted"
            else:
                mark_type = "open"
            slider_marks.append(SliderMark(mark_type, val=frame_idx))

        self.setTracks(track_row)  # total number of tracks to show
        self.setMarks(slider_marks)

    def setHeaderSeries(self, series: Optional[Dict[int, float]] = None):
        """Show header graph with specified series.

        Args:
            series: {frame number: series value} dict.
        Returns:
            None.
        """
        self.headerSeries = [] if series is None else series
        self._header_height = 30
        self.drawHeader()
        self.updateHeight()

    def clearHeader(self):
        """Remove header graph from slider."""
        self.headerSeries = []
        self._header_height = 0
        self.updateHeight()

    def setTracks(self, track_rows):
        """Set the number of tracks to show in slider.

        Args:
            track_rows: the number of tracks to show
        """
        self._track_rows = track_rows
        self.updateHeight()

    def updateHeight(self):
        """Update the height of the slider."""

        tracks = self._track_rows
        if tracks == 0:
            min_height = self._min_height
            max_height = self._min_height
        else:
            # Start with padding height
            extra_height = 8 + self._header_height
            min_height = extra_height
            max_height = extra_height

            # Add height for tracks
            min_height += self._track_height * min(tracks, 20)
            max_height += self._track_height * tracks

            # Make sure min/max height is at least 19, even if few tracks
            min_height = max(self._min_height, min_height)
            max_height = max(self._min_height, max_height)

        self.setMaximumHeight(max_height)
        self.setMinimumHeight(min_height)

        # Redraw all marks with new height and y position
        marks = self.getMarks()
        self.setMarks(marks)

        self.resizeEvent()
        self.heightUpdated.emit()

    def _toPos(self, val: float, center=False) -> float:
        """
        Converts slider value to x position on slider.

        Args:
            val: The slider value.
            center: Whether to offset by half the width of drag handle,
                so that plotted location will light up with center of handle.

        Returns:
            x position.
        """
        x = val
        x -= self._val_min
        x /= max(1, self._val_max - self._val_min)
        x *= self._sliderWidth()
        if center:
            x += self.handle.rect().width() / 2.0
        return x

    def _toVal(self, x: float, center=False) -> float:
        """Converts x position to slider value."""
        val = x
        val /= self._sliderWidth()
        val *= max(1, self._val_max - self._val_min)
        val += self._val_min
        val = round(val)
        return val

    def _sliderWidth(self) -> float:
        """Returns visual width of slider."""
        return self.outlineBox.rect().width() - self.handle.rect().width()

    def value(self) -> float:
        """Returns value of slider."""
        return self._val_main

    def setValue(self, val: float) -> float:
        """Sets value of slider."""
        self._val_main = val
        x = self._toPos(val)
        self.handle.setPos(x, 0)

    def setMinimum(self, min: float) -> float:
        """Sets minimum value for slider."""
        self._val_min = min

    def setMaximum(self, max: float) -> float:
        """Sets maximum value for slider."""
        self._val_max = max

    def setEnabled(self, val: float) -> float:
        """Set whether the slider is enabled."""
        self._enabled = val

    def enabled(self):
        """Returns whether slider is enabled."""
        return self._enabled

    def clearSelection(self):
        """Clears selection endpoints."""
        self._selection = []
        self.select_box.hide()

    def startSelection(self, val):
        """Adds initial selection endpoint.

        Called when user starts dragging to select range in slider.

        Args:
            val: value of endpoint
        """
        self._selection.append(val)

    def endSelection(self, val, update: bool = False):
        """Add final selection endpoint.

        Called during or after the user is dragging to select range.

        Args:
            val: value of endpoint
            update:
        """
        # If we want to update endpoint and there's already one, remove it
        if update and len(self._selection) % 2 == 0:
            self._selection.pop()
        # Add the selection endpoint
        self._selection.append(val)
        a, b = self._selection[-2:]
        if a == b:
            self.clearSelection()
        else:
            self.drawSelection(a, b)
        # Emit signal (even if user selected same region as before)
        self.selectionChanged.emit(*self.getSelection())

    def hasSelection(self) -> bool:
        """Returns True if a clip is selected, False otherwise."""
        a, b = self.getSelection()
        return a < b

    def getSelection(self):
        """Returns start and end value of current selection endpoints."""
        a, b = 0, 0
        if len(self._selection) % 2 == 0 and len(self._selection) > 0:
            a, b = self._selection[-2:]
        start = min(a, b)
        end = max(a, b)
        return start, end

    def drawSelection(self, a: float, b: float):
        """Draws selection box on slider.

        Args:
            a: one endpoint value
            b: other endpoint value

        Returns:
            None.
        """
        start = min(a, b)
        end = max(a, b)
        start_pos = self._toPos(start, center=True)
        end_pos = self._toPos(end, center=True)
        selection_rect = QtCore.QRect(
            start_pos, 1, end_pos - start_pos, self.outlineBox.rect().height() - 2
        )

        self.select_box.setRect(selection_rect)
        self.select_box.show()

    def moveSelectionAnchor(self, x: float, y: float):
        """
        Moves selection anchor in response to mouse position.

        Args:
            x: x position of mouse
            y: y position of mouse

        Returns:
            None.
        """
        x = max(x, 0)
        x = min(x, self.outlineBox.rect().width())
        anchor_val = self._toVal(x, center=True)

        if len(self._selection) % 2 == 0:
            self.startSelection(anchor_val)

        self.drawSelection(anchor_val, self._selection[-1])

    def releaseSelectionAnchor(self, x, y):
        """
        Finishes selection in response to mouse release.

        Args:
            x: x position of mouse
            y: y position of mouse

        Returns:
            None.
        """
        x = max(x, 0)
        x = min(x, self.outlineBox.rect().width())
        anchor_val = self._toVal(x)
        self.endSelection(anchor_val)

    def clearMarks(self):
        """Clears all marked values for slider."""
        if hasattr(self, "_mark_items"):
            for item in self._mark_items.values():
                self.scene.removeItem(item)
        self._marks = set()  # holds mark position
        self._mark_items = dict()  # holds visual Qt object for plotting mark

    def setMarks(self, marks: Iterable[Union[SliderMark, int]]):
        """Sets all marked values for the slider.

        Args:
            marks: iterable with all values to mark

        Returns:
            None.
        """
        self.clearMarks()
        if marks is not None:
            for mark in marks:
                if not isinstance(mark, SliderMark):
                    mark = SliderMark("simple", mark)
                self.addMark(mark, update=False)
        self.updatePos()

    def getMarks(self):
        """Returns list of marks."""
        return self._marks

    def addMark(self, new_mark: SliderMark, update: bool = True):
        """Adds a marked value to the slider.

        Args:
            new_mark: value to mark
            update: Whether to redraw slider with new mark.

        Returns:
            None.
        """
        # check if mark is within slider range
        if new_mark.val > self._val_max:
            return
        if new_mark.val < self._val_min:
            return

        self._marks.add(new_mark)

        v_top_pad = 3 + self._header_height
        v_bottom_pad = 3

        width = 0

        if new_mark.type == "track":
            v_offset = v_top_pad + (self._track_height * new_mark.row)
            height = 1
        else:
            v_offset = v_top_pad
            height = self.outlineBox.rect().height() - (v_offset + v_bottom_pad)

            width = 2 if new_mark.type in ("open", "filled") else 0

        color = new_mark.QColor
        pen = QPen(color, 0.5)
        pen.setCosmetic(True)
        brush = QBrush(color) if new_mark.filled else QBrush()

        line = self.scene.addRect(-width // 2, v_offset, width, height, pen, brush)
        self._mark_items[new_mark] = line
        if update:
            self.updatePos()

    def updatePos(self):
        """Update the visual x position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)

        for mark in self._mark_items.keys():

            width = 0
            if mark.type == "track":
                width_in_frames = mark.end_val - mark.val
                width = max(2, self._toPos(width_in_frames))

            elif mark.type in ("open", "filled"):
                width = 2

            x = self._toPos(mark.val, center=True)
            self._mark_items[mark].setPos(x, 0)

            rect = self._mark_items[mark].rect()
            rect.setWidth(width)

            self._mark_items[mark].setRect(rect)

    def drawHeader(self):
        """Draw the header graph."""
        if len(self.headerSeries) == 0 or self._header_height == 0:
            self.poly.setPath(QPainterPath())
            return

        step = max(self.headerSeries.keys()) // int(self._sliderWidth())
        step = max(step, 1)
        count = max(self.headerSeries.keys()) // step * step

        sampled = np.full((count), 0.0)
        for key, val in self.headerSeries.items():
            if key < count:
                sampled[key] = val
        sampled = np.max(sampled.reshape(count // step, step), axis=1)
        series = {i * step: sampled[i] for i in range(count // step)}

        series_min = np.min(sampled) - 1
        series_max = np.max(sampled)
        series_scale = (self._header_height - 5) / (series_max - series_min)

        def toYPos(val):
            return self._header_height - ((val - series_min) * series_scale)

        step_chart = False  # use steps rather than smooth line

        points = []
        points.append((self._toPos(0, center=True), toYPos(series_min)))
        for idx, val in series.items():
            points.append((self._toPos(idx, center=True), toYPos(val)))
            if step_chart:
                points.append((self._toPos(idx + step, center=True), toYPos(val)))
        points.append(
            (self._toPos(max(series.keys()) + 1, center=True), toYPos(series_min))
        )

        # Convert to list of QtCore.QPointF objects
        points = list(itertools.starmap(QtCore.QPointF, points))
        self.poly.setPath(self._pointsToPath(points))

    def moveHandle(self, x, y):
        """Move handle in response to mouse position.

        Emits valueChanged signal if value of slider changed.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x -= self.handle.rect().width() / 2.0
        x = max(x, 0)
        x = min(x, self.outlineBox.rect().width() - self.handle.rect().width())

        val = self._toVal(x)

        # snap to nearby mark within handle
        mark_vals = [mark.val for mark in self._marks]
        handle_left = self._toVal(x - self.handle.rect().width() / 2)
        handle_right = self._toVal(x + self.handle.rect().width() / 2)
        marks_in_handle = [
            mark for mark in mark_vals if handle_left < mark < handle_right
        ]
        if marks_in_handle:
            marks_in_handle.sort(key=lambda m: (abs(m - val), m > val))
            val = marks_in_handle[0]

        old = self.value()
        self.setValue(val)

        if old != val:
            self.valueChanged.emit(self._val_main)

    def resizeEvent(self, event=None):
        """Override method to update visual size when necessary.

        Args:
            event
        """
        height = self.size().height()

        outline_rect = self.outlineBox.rect()
        handle_rect = self.handle.rect()
        select_box_rect = self.select_box.rect()

        outline_rect.setHeight(height - 3)
        if event is not None:
            outline_rect.setWidth(event.size().width() - 1)
        self.outlineBox.setRect(outline_rect)

        handle_rect.setTop(self._handleTop())
        handle_rect.setHeight(self._handleHeight())
        self.handle.setRect(handle_rect)

        select_box_rect.setHeight(self._handleHeight())
        self.select_box.setRect(select_box_rect)

        self.updatePos()
        self.drawHeader()
        super(VideoSlider, self).resizeEvent(event)

    def _handleTop(self) -> float:
        """Returns y position of top of handle (i.e., header height)."""
        return 1 + self._header_height

    def _handleHeight(self, outline_rect=None) -> float:
        """
        Returns visual height of handle.

        Args:
            outline_rect: The rect of the outline box for the slider. This
                is only required when calling during initialization (when the
                outline box doesn't yet exist).

        Returns:
            Height of handle in pixels.
        """
        if outline_rect is None:
            outline_rect = self.outlineBox.rect()

        handle_bottom_offset = 1
        handle_height = outline_rect.height() - (
            self._handleTop() + handle_bottom_offset
        )
        return handle_height

    def mousePressEvent(self, event):
        """Override method to move handle for mouse press/drag.

        Args:
            event
        """
        scenePos = self.mapToScene(event.pos())

        # Do nothing if not enabled
        if not self.enabled():
            return
        # Do nothing if click outside slider area
        if not self.outlineBox.rect().contains(scenePos):
            return

        move_function = None
        release_function = None

        if event.modifiers() == QtCore.Qt.ShiftModifier:
            move_function = self.moveSelectionAnchor
            release_function = self.releaseSelectionAnchor

            self.clearSelection()

        elif event.modifiers() == QtCore.Qt.NoModifier:
            move_function = self.moveHandle
            release_function = None

        # Connect to signals
        if move_function is not None:
            self.mouseMoved.connect(move_function)

        def done(x, y):
            if release_function is not None:
                release_function(x, y)
            self.mouseMoved.disconnect(move_function)
            self.mouseReleased.disconnect(done)

        self.mouseReleased.connect(done)

        # Emit signal
        self.mouseMoved.emit(scenePos.x(), scenePos.y())
        self.mousePressed.emit(scenePos.x(), scenePos.y())

    def mouseMoveEvent(self, event):
        """Override method to emid mouseMoved signal on drag."""
        scenePos = self.mapToScene(event.pos())
        self.mouseMoved.emit(scenePos.x(), scenePos.y())

    def mouseReleaseEvent(self, event):
        """Override method to emit mouseReleased signal on release."""
        scenePos = self.mapToScene(event.pos())
        self.mouseReleased.emit(scenePos.x(), scenePos.y())

    def keyPressEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self.keyPress.emit(event)
        event.accept()

    def keyReleaseEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self.keyRelease.emit(event)
        event.accept()

    def boundingRect(self) -> QtCore.QRectF:
        """Method required by Qt."""
        return self.outlineBox.rect()

    def paint(self, *args, **kwargs):
        """Method required by Qt."""
        super(VideoSlider, self).paint(*args, **kwargs)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = VideoSlider(
        min=0,
        max=20,
        val=15,
        marks=(10, 15),  # ((0,10),(0,15),(1,10),(1,11),(2,12)), tracks=3
    )

    window.valueChanged.connect(lambda x: print(x))
    window.show()

    app.exec_()

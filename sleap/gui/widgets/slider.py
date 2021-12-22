"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtGui import QPen, QBrush, QColor, QKeyEvent, QPolygonF, QPainterPath

from sleap.gui.color import ColorManager

import attr
import itertools
import numpy as np
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


# for debug, we can filter out short tracks from slider
SEEKBAR_MIN_TRACK_LEN_TO_SHOW = 0


@attr.s(auto_attribs=True, eq=False)
class SliderMark:
    """
    Class to hold data for an individual mark on the slider.

    Attributes:
        type: Type of the mark, options are:
            * "simple"     (single value)
            * "simple_thin" (    ditto   )
            * "filled"
            * "open"
            * "predicted"
            * "tick"
            * "tick_column"
            * "track"      (range of values)
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
        """Returns color of mark."""
        colors = dict(
            simple="black",
            simple_thin="black",
            filled="blue",
            open="blue",
            predicted=(1, 170, 247),  # light blue
            tick="lightGray",
            tick_column="gray",
        )

        if self.type in colors:
            return colors[self.type]
        else:
            return self._color

    @color.setter
    def color(self, val):
        """Sets color of mark."""
        self._color = val

    @property
    def QColor(self):
        """Returns color of mark as `QColor`."""
        c = self.color
        if type(c) == str:
            return QColor(c)
        else:
            return QColor(*c)

    @property
    def filled(self):
        """Returns whether mark is filled or open."""
        if self.type == "open":
            return False
        else:
            return True

    @property
    def top_pad(self):
        if self.type == "tick_column":
            return 40
        if self.type == "tick":
            return 0
        return 2

    @property
    def bottom_pad(self):
        if self.type == "tick_column":
            return 200
        if self.type == "tick":
            return 0
        return 2

    @property
    def visual_width(self):
        if self.type in ("open", "filled", "tick"):
            return 2
        if self.type in ("tick_column", "simple", "predicted"):
            return 1
        return 0

    def get_height(self, container_height):
        if self.type == "track":
            return 2
        height = container_height
        # if self.padded:
        height -= self.top_pad + self.bottom_pad

        return height


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
        max=1,
        val=0,
        marks=None,
        *args,
        **kwargs,
    ):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setMouseTracking(True)

        self._get_val_tooltip = None

        self.tick_index_offset = 1
        self.zoom_factor = 1

        self._track_rows = 0
        self._track_height = 5
        self._max_tracks_stacked = 120
        self._track_stack_skip_count = 10
        self._header_label_height = 20
        self._header_graph_height = 40
        self._header_height = self._header_label_height  # room for frame labels
        self._min_height = 19 + self._header_height

        self._base_font = QtGui.QFont()
        self._base_font.setPixelSize(10)

        self._tick_marks = []

        # Add border rect
        outline_rect = QtCore.QRectF(0, 0, 200, self._min_height - 3)
        self.box_rect = outline_rect
        # self.outlineBox = self.scene.addRect(outline_rect)
        # self.outlineBox.setPen(QPen(QColor("black", alpha=0)))

        # Add drag handle rect
        self._handle_width = 6
        handle_rect = QtCore.QRect(
            0, self._handle_top, self._handle_width, self._handle_height
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

        self.zoom_box = self.scene.addRect(
            QtCore.QRect(0, 1, 0, outline_rect.height() - 2)
        )
        self.zoom_box.setPen(QPen(QColor(80, 80, 80, 64)))
        self.zoom_box.setBrush(QColor(80, 80, 80, 64))
        self.zoom_box.hide()

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
        self._draw_header()

    # Methods to match API for QSlider

    def value(self) -> float:
        """Returns value of slider."""
        return self._val_main

    def setValue(self, val: float) -> float:
        """Sets value of slider."""
        self._val_main = val
        x = self._toPos(val)
        self.handle.setPos(x, 0)
        self.ensureVisible(x, 0, self._handle_width, 0, 3, 0)

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

    # Methods for working with visual positions (mapping to and from, redrawing)

    def _update_visual_positions(self):
        """Updates the visual x position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)

        for mark in self._mark_items.keys():

            if mark.type == "track":
                width_in_frames = mark.end_val - mark.val
                width = max(2, self._toPos(width_in_frames))

            else:
                width = mark.visual_width

            x = self._toPos(mark.val, center=True)
            self._mark_items[mark].setPos(x, 0)

            if mark in self._mark_labels:
                label_x = max(
                    0, x - self._mark_labels[mark].boundingRect().width() // 2
                )
                self._mark_labels[mark].setPos(label_x, 4)

            rect = self._mark_items[mark].rect()
            rect.setWidth(width)
            rect.setHeight(
                mark.get_height(
                    container_height=self.box_rect.height() - self._header_height
                )
            )

            self._mark_items[mark].setRect(rect)

    def _get_min_max_slider_heights(self):
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
            max_height += self._track_height * min(tracks, self._max_tracks_stacked)

            # Make sure min/max height is at least 19, even if few tracks
            min_height = max(self._min_height, min_height)
            max_height = max(self._min_height, max_height)

        return min_height, max_height

    def _update_slider_height(self):
        """Update the height of the slider."""

        min_height, max_height = self._get_min_max_slider_heights()

        # TODO: find the current height of the scrollbar
        # self.horizontalScrollBar().height() gives the wrong value
        scrollbar_height = 18

        self.setMaximumHeight(max_height + scrollbar_height)
        self.setMinimumHeight(min_height + scrollbar_height)

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
        x *= self._slider_width
        if center:
            x += self.handle.rect().width() / 2.0
        return x

    def _toVal(self, x: float, center=False) -> float:
        """Converts x position to slider value."""
        val = x
        val /= self._slider_width
        val *= max(1, self._val_max - self._val_min)
        val += self._val_min
        val = round(val)
        return val

    @property
    def _slider_width(self) -> float:
        """Returns visual width of slider."""
        return self.box_rect.width() - self.handle.rect().width()

    @property
    def slider_visible_value_range(self) -> float:
        """Value range that's visible given current size and zoom."""
        return self._toVal(self.width() - 1)

    @property
    def _mark_area_height(self) -> float:
        _, max_height = self._get_min_max_slider_heights()
        return max_height - 3 - self._header_height

    @property
    def value_range(self) -> float:
        return self._val_max - self._val_min

    @property
    def box_rect(self) -> QtCore.QRectF:
        return self._box_rect

    @box_rect.setter
    def box_rect(self, rect: QtCore.QRectF):
        self._box_rect = rect

        # Update the scene rect so that it matches how much space we
        # currently want for drawing everything.
        rect.setWidth(rect.width() - 1)
        self.setSceneRect(rect)

    # Methods for range selection and zoom

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
            self._draw_selection(a, b)
        # Emit signal (even if user selected same region as before)
        self.selectionChanged.emit(*self.getSelection())

    def setSelection(self, start_val, end_val):
        """Selects clip from start_val to end_val."""
        self.startSelection(start_val)
        self.endSelection(end_val, update=True)

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

    def _draw_selection(self, a: float, b: float):
        self._update_selection_box_positions(self.select_box, a, b)

    def _draw_zoom_box(self, a: float, b: float):
        self._update_selection_box_positions(self.zoom_box, a, b)

    def _update_selection_box_positions(self, box_object, a: float, b: float):
        """Update box item on slider.

        Args:
            box_object: The box to update
            a: one endpoint value
            b: other endpoint value

        Returns:
            None.
        """
        start = min(a, b)
        end = max(a, b)
        start_pos = self._toPos(start, center=True)
        end_pos = self._toPos(end, center=True)
        box_rect = QtCore.QRect(
            start_pos,
            self._header_height,
            end_pos - start_pos,
            self.box_rect.height(),
        )

        box_object.setRect(box_rect)
        box_object.show()

    def _update_selection_boxes_on_resize(self):
        for box_object in (self.select_box, self.zoom_box):
            rect = box_object.rect()
            rect.setHeight(self._handle_height)
            box_object.setRect(rect)

        if self.select_box.isVisible():
            self._draw_selection(*self.getSelection())

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
        x = min(x, self.box_rect.width())
        anchor_val = self._toVal(x, center=True)

        if len(self._selection) % 2 == 0:
            self.startSelection(anchor_val)

        self._draw_selection(anchor_val, self._selection[-1])

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
        x = min(x, self.box_rect.width())
        anchor_val = self._toVal(x)
        self.endSelection(anchor_val)

    def moveZoomDrag(self, x: float, y: float):
        if getattr(self, "_zoom_start_val", None) is None:
            self._zoom_start_val = self._toVal(x, center=True)

        current_val = self._toVal(x, center=True)

        self._draw_zoom_box(current_val, self._zoom_start_val)

    def releaseZoomDrag(self, x, y):

        self.zoom_box.hide()

        val_a = self._zoom_start_val
        val_b = self._toVal(x, center=True)

        val_start = min(val_a, val_b)
        val_end = max(val_a, val_b)

        # pad the zoom
        val_range = val_end - val_start
        val_start -= val_range * 0.05
        val_end += val_range * 0.05

        self.setZoomRange(val_start, val_end)

        self._zoom_start_val = None

    def setZoomRange(self, start_val: float, end_val: float):

        zoom_val_range = end_val - start_val
        if zoom_val_range > 0:
            self.zoom_factor = self.value_range / zoom_val_range
        else:
            self.zoom_factor = 1

        self.resizeEvent()

        center_val = start_val + zoom_val_range / 2
        center_pos = self._toPos(center_val)

        self.centerOn(center_pos, 0)

    # Methods for modifying marks on slider

    def setNumberOfTracks(self, track_rows):
        """Set the number of tracks to show in slider.

        Args:
            track_rows: the number of tracks to show
        """
        self._track_rows = track_rows
        self._update_slider_height()

    def clearMarks(self):
        """Clears all marked values for slider."""
        if hasattr(self, "_mark_items"):
            for item in self._mark_items.values():
                self.scene.removeItem(item)

        if hasattr(self, "_mark_labels"):
            for item in self._mark_labels.values():
                self.scene.removeItem(item)

        self._marks = set()  # holds mark position
        self._mark_items = dict()  # holds visual Qt object for plotting mark
        self._mark_labels = dict()

    def setMarks(self, marks: Iterable[Union[SliderMark, int]]):
        """Sets all marked values for the slider.

        Args:
            marks: iterable with all values to mark

        Returns:
            None.
        """
        self.clearMarks()

        # Add tick marks first so they're behind other marks
        self._add_tick_marks()

        if marks is not None:
            for mark in marks:
                if not isinstance(mark, SliderMark):
                    mark = SliderMark("simple", mark)
                self.addMark(mark, update=False)

        self._update_visual_positions()

    def setTickMarks(self):
        """Resets which tick marks to show."""
        self._clear_tick_marks()
        self._add_tick_marks()

    def _clear_tick_marks(self):
        if not hasattr(self, "_tick_marks"):
            return

        for mark in self._tick_marks:
            self.removeMark(mark)

    def _add_tick_marks(self):
        val_range = self.slider_visible_value_range

        if val_range < 20:
            val_order = 1
        else:
            val_order = 10
            while val_range // val_order > 24:
                val_order *= 10

        self._tick_marks = []

        for tick_pos in range(
            self._val_min + val_order - 1, self._val_max + 1, val_order
        ):
            self._tick_marks.append(SliderMark("tick", tick_pos))

        for tick_mark in self._tick_marks:
            self.addMark(tick_mark, update=False)

    def removeMark(self, mark: SliderMark):
        """Removes an individual mark."""
        if mark in self._mark_labels:
            self.scene.removeItem(self._mark_labels[mark])
            del self._mark_labels[mark]
        if mark in self._mark_items:
            self.scene.removeItem(self._mark_items[mark])
            del self._mark_items[mark]
        if mark in self._marks:
            self._marks.remove(mark)

    def getMarks(self, type: str = ""):
        """Returns list of marks."""
        if type:
            return [mark for mark in self._marks if mark.type == type]

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

        v_top_pad = self._header_height + 1
        v_bottom_pad = 1
        v_top_pad += new_mark.top_pad
        v_bottom_pad += new_mark.bottom_pad

        width = new_mark.visual_width

        v_offset = v_top_pad
        if new_mark.type == "track":
            v_offset += self._get_track_vertical_pos(
                *self._get_track_column_row(new_mark.row)
            )

        height = new_mark.get_height(
            container_height=self.box_rect.height() - self._header_height
        )

        color = new_mark.QColor
        pen = QPen(color, 0.5)
        pen.setCosmetic(True)
        brush = QBrush(color) if new_mark.filled else QBrush()

        line = self.scene.addRect(-width // 2, v_offset, width, height, pen, brush)
        self._mark_items[new_mark] = line

        if new_mark.type == "tick":
            # Show tick mark behind other slider marks
            self._mark_items[new_mark].setZValue(0)

            # Add a text label to show in header area
            mark_label_text = (
                f"{new_mark.val + self.tick_index_offset:g}"  # sci notation if large
            )
            self._mark_labels[new_mark] = self.scene.addSimpleText(
                mark_label_text, self._base_font
            )
        elif new_mark.type == "track":
            # Show tracks over tick marks
            self._mark_items[new_mark].setZValue(2)
        else:
            # Show in front of tick marks and behind track lines
            self._mark_items[new_mark].setZValue(1)

        if update:
            self._update_visual_positions()

    def _get_track_column_row(self, raw_row: int) -> Tuple[int, int]:
        """
        Returns the column and row for a given track index.

        If there are many tracks we "wrap" around to showing tracks at the top
        of the slider (so that it's not too tall). Each time we "wrap" back to
        the top is a new "column" which starts at "row" 0.
        """
        if raw_row < self._max_tracks_stacked:
            return 0, raw_row

        else:
            rows_after_first_col = raw_row - self._max_tracks_stacked
            rows_per_later_cols = (
                self._max_tracks_stacked - self._track_stack_skip_count
            )

            rows_down = rows_after_first_col % rows_per_later_cols
            col = (rows_after_first_col // rows_per_later_cols) + 1

            return col, rows_down

    def _get_track_vertical_pos(self, col: int, row: int) -> int:
        """
        Returns visible vertical position of track in given column and row.

        The "column" and "row" are given by _get_track_column_row.
        """
        if col == 0:
            return row * self._track_height
        else:
            return (self._track_height * self._track_stack_skip_count) + (
                self._track_height * row
            )

    def _is_track_in_new_column(self, row: int) -> bool:
        """Returns whether this track is at the top of a new column."""
        _, row_down = self._get_track_column_row(row)
        return row_down == 0

    # Methods for header graph

    def setHeaderSeries(self, series: Optional[Dict[int, float]] = None):
        """Show header graph with specified series.

        Args:
            series: {frame number: series value} dict.
        Returns:
            None.
        """
        self.headerSeries = [] if series is None else series
        self._header_height = self._header_label_height + self._header_graph_height
        self._draw_header()
        self._update_slider_height()

    def clearHeader(self):
        """Remove header graph from slider."""
        self.headerSeries = []
        self._header_height = self._header_label_height
        self._update_slider_height()

    def _get_header_series_len(self):
        if hasattr(self.headerSeries, "keys"):
            series_frame_max = max(self.headerSeries.keys())
        else:
            series_frame_max = len(self.headerSeries)
        return series_frame_max

    @property
    def _header_series_items(self):
        """Yields (frame idx, val) for header series items."""
        if hasattr(self.headerSeries, "items"):
            for key, val in self.headerSeries.items():
                yield key, val
        else:
            for key in range(len(self.headerSeries)):
                val = self.headerSeries[key]
                yield key, val

    def _draw_header(self):
        """Draws the header graph."""
        if len(self.headerSeries) == 0 or self._header_height == 0:
            self.poly.setPath(QPainterPath())
            return

        series_frame_max = self._get_header_series_len()

        step = series_frame_max // int(self._slider_width)
        step = max(step, 1)
        count = series_frame_max // step * step

        sampled = np.full((count), 0.0, dtype=float)

        for key, val in self._header_series_items:
            if key < count:
                sampled[key] = val

        sampled = np.max(sampled.reshape(count // step, step), axis=1)
        series = {i * step: sampled[i] for i in range(count // step)}

        series_min = np.min(sampled) - 1
        series_max = np.max(sampled)
        series_scale = (self._header_graph_height) / (series_max - series_min)

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

    def _pointsToPath(self, points: List[QtCore.QPointF]) -> QPainterPath:
        """Converts list of `QtCore.QPointF` objects to a `QPainterPath`."""
        path = QPainterPath()
        path.addPolygon(QPolygonF(points))
        return path

    # Methods for working with slider handle

    def mapMouseXToHandleX(self, x) -> float:
        x -= self.handle.rect().width() / 2.0
        x = max(x, 0)
        x = min(x, self.box_rect.width() - self.handle.rect().width())
        return x

    def moveHandle(self, x, y):
        """Move handle in response to mouse position.

        Emits valueChanged signal if value of slider changed.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x = self.mapMouseXToHandleX(x)

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

    @property
    def _handle_top(self) -> float:
        """Returns y position of top of handle (i.e., header height)."""
        return 1 + self._header_height

    @property
    def _handle_height(self, outline_rect=None) -> float:
        """
        Returns visual height of handle.

        Args:
            outline_rect: The rect of the outline box for the slider. This
                is only required when calling during initialization (when the
                outline box doesn't yet exist).

        Returns:
            Height of handle in pixels.
        """
        return self._mark_area_height

    # Methods for selection of contiguously marked ranges of frames

    def contiguousSelectionMarksAroundVal(self, val):
        """Selects contiguously marked frames around value."""
        if not self.isMarkedVal(val):
            return

        dec_val = self.getStartContiguousMark(val)
        inc_val = self.getEndContiguousMark(val)

        self.setSelection(dec_val, inc_val)

    def getStartContiguousMark(self, val: int) -> int:
        """
        Returns first marked value in contiguously marked region around val.
        """
        last_val = val
        dec_val = self._dec_contiguous_marked_val(last_val)
        while last_val > dec_val > self._val_min:
            last_val = dec_val
            dec_val = self._dec_contiguous_marked_val(last_val)

        return dec_val

    def getEndContiguousMark(self, val: int) -> int:
        """
        Returns last marked value in contiguously marked region around val.
        """
        last_val = val
        inc_val = self._inc_contiguous_marked_val(last_val)
        while last_val < inc_val < self._val_max:
            last_val = inc_val
            inc_val = self._inc_contiguous_marked_val(last_val)

        return inc_val

    def getMarksAtVal(self, val: int) -> List[SliderMark]:
        if val is None:
            return []

        return [
            mark
            for mark in self._marks
            if (mark.val == val and mark.type not in ("tick", "tick_column"))
            or (mark.type == "track" and mark.val <= val < mark.end_val)
        ]

    def isMarkedVal(self, val: int) -> bool:
        """Returns whether value has mark."""
        if self.getMarksAtVal(val):
            return True
        return False

    def _dec_contiguous_marked_val(self, val):
        """Decrements value within contiguously marked range if possible."""
        dec_val = min(
            (
                mark.val
                for mark in self._marks
                if mark.type == "track" and mark.val < val <= mark.end_val
            ),
            default=val,
        )
        if dec_val < val:
            return dec_val

        if val - 1 in [mark.val for mark in self._marks]:
            return val - 1

        # Return original value if we can't decrement it w/in contiguous range
        return val

    def _inc_contiguous_marked_val(self, val):
        """Increments value within contiguously marked range if possible."""
        inc_val = max(
            (
                mark.end_val - 1
                for mark in self._marks
                if mark.type == "track" and mark.val <= val < mark.end_val
            ),
            default=val,
        )
        if inc_val > val:
            return inc_val

        if val + 1 in [mark.val for mark in self._marks]:
            return val + 1

        # Return original value if we can't decrement it w/in contiguous range
        return val

    # Method for cursor

    def setTooltipCallable(self, tooltip_callable: Callable):
        """
        Sets function to get tooltip text for given value in slider.

        Args:
            tooltip_callable: a function which takes the value which the user
                is hovering over and returns the tooltip text to show (if any)
        """
        self._get_val_tooltip = tooltip_callable

    def _update_cursor_for_event(self, event):
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            self.setCursor(QtCore.Qt.CrossCursor)
        elif event.modifiers() == QtCore.Qt.AltModifier:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.unsetCursor()

    # Methods which override QGraphicsView

    def resizeEvent(self, event=None):
        """Override method to update visual size when necessary.

        Args:
            event
        """

        outline_rect = self.box_rect
        handle_rect = self.handle.rect()

        outline_rect.setHeight(self._mark_area_height + self._header_height)

        if event is not None:
            visual_width = event.size().width() - 1
        else:
            visual_width = self.width() - 1

        drawn_width = visual_width * self.zoom_factor

        outline_rect.setWidth(drawn_width)
        self.box_rect = outline_rect

        handle_rect.setTop(self._handle_top)
        handle_rect.setHeight(self._handle_height)
        self.handle.setRect(handle_rect)

        self._update_selection_boxes_on_resize()

        self.setTickMarks()
        self._update_visual_positions()
        self._draw_header()

        super(VideoSlider, self).resizeEvent(event)

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
        if not self.box_rect.contains(scenePos):
            return

        move_function = None
        release_function = None

        self._update_cursor_for_event(event)

        # Shift : selection
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            move_function = self.moveSelectionAnchor
            release_function = self.releaseSelectionAnchor

            self.clearSelection()

        # No modifier : go to frame
        elif event.modifiers() == QtCore.Qt.NoModifier:
            move_function = self.moveHandle
            release_function = None

        # Alt (option) : zoom
        elif event.modifiers() == QtCore.Qt.AltModifier:
            move_function = self.moveZoomDrag
            release_function = self.releaseZoomDrag

        else:
            event.accept()  # mouse events shouldn't be passed to video widgets

        # Connect to signals
        if move_function is not None:
            self.mouseMoved.connect(move_function)

        def done(x, y):
            self.unsetCursor()
            if release_function is not None:
                release_function(x, y)
            if move_function is not None:
                self.mouseMoved.disconnect(move_function)
            self.mouseReleased.disconnect(done)

        self.mouseReleased.connect(done)

        # Emit signal
        self.mouseMoved.emit(scenePos.x(), scenePos.y())
        self.mousePressed.emit(scenePos.x(), scenePos.y())

    def mouseMoveEvent(self, event):
        """Override method to emit mouseMoved signal on drag."""
        scenePos = self.mapToScene(event.pos())

        # Update cursor type based on current modifier key
        self._update_cursor_for_event(event)

        # Show tooltip with information about frame under mouse
        if self._get_val_tooltip:
            hover_frame_idx = self._toVal(self.mapMouseXToHandleX(scenePos.x()))
            tooltip = self._get_val_tooltip(hover_frame_idx)
            QtWidgets.QToolTip.showText(event.globalPos(), tooltip)

        self.mouseMoved.emit(scenePos.x(), scenePos.y())

    def mouseReleaseEvent(self, event):
        """Override method to emit mouseReleased signal on release."""
        scenePos = self.mapToScene(event.pos())

        self.mouseReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """Override method to move handle for mouse double-click.

        Args:
            event
        """
        scenePos = self.mapToScene(event.pos())

        # Do nothing if not enabled
        if not self.enabled():
            return
        # Do nothing if click outside slider area
        if not self.box_rect.contains(scenePos):
            return

        if event.modifiers() == QtCore.Qt.ShiftModifier:
            self.contiguousSelectionMarksAroundVal(self._toVal(scenePos.x()))

    def leaveEvent(self, event):
        self.unsetCursor()

    def keyPressEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self._update_cursor_for_event(event)
        self.keyPress.emit(event)
        event.accept()

    def keyReleaseEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self.unsetCursor()
        self.keyRelease.emit(event)
        event.accept()

    def boundingRect(self) -> QtCore.QRectF:
        """Method required by Qt."""
        return self.box_rect

    def paint(self, *args, **kwargs):
        """Method required by Qt."""
        super(VideoSlider, self).paint(*args, **kwargs)


# Map meaning of mark to the type of mark
class SemanticMarkType(Enum):
    user = "simple"
    predicted_no_track = "simple_thin"
    suggested_with_user = "filled"
    suggested_with_nothing = "open"
    suggested_with_predicted = "predicted"


def set_slider_marks_from_labels(
    slider: VideoSlider,
    labels: "Labels",
    video: "Video",
    color_manager: Optional[ColorManager] = None,
):
    """
    Sets slider marks using track information from `Labels` object.

    Args:
        slider: the slider we're updating
        labels: the dataset with tracks and labeled frames
        video: the video for which to show marks

    Returns:
        None
    """

    if color_manager is None:
        color_manager = ColorManager(labels=labels)

    # Make function which can be used to get tooltip text when hovering
    # over a given value (i.e., frame index) in the slider.
    def get_val_tooltip(idx: int) -> str:
        tooltip = f"Frame {idx+1}"

        frame_mark_types = {mark.type for mark in slider.getMarksAtVal(idx)}

        if SemanticMarkType.user.value in frame_mark_types:
            tooltip += "\nuser labeled"
        elif SemanticMarkType.predicted_no_track.value in frame_mark_types:
            tooltip += "\nprediction without track identity"
        elif SemanticMarkType.suggested_with_user.value in frame_mark_types:
            tooltip += "\nsuggested frame with user labels"
        elif SemanticMarkType.suggested_with_nothing.value in frame_mark_types:
            tooltip += "\nsuggested frame (no labels)"
        elif SemanticMarkType.suggested_with_predicted.value in frame_mark_types:
            tooltip += "\nsuggested frame with prediction"
        elif "track" in frame_mark_types:
            tooltip += "\nprediction with track identity"

        lf = labels.find(video, idx)
        if lf:
            lf = lf[0]
            user_instance_count = len(lf.user_instances)
            pred_instance_count = len(lf.predicted_instances)

            if pred_instance_count:
                tooltip += f"\n{pred_instance_count} predicted instance"
                if pred_instance_count > 1:
                    tooltip += "s"

            if user_instance_count:
                tooltip += f"\n{user_instance_count} user instance"
                if user_instance_count > 1:
                    tooltip += "s"

        return tooltip

    # Set slider to use this function for getting tooltip text
    slider.setTooltipCallable(get_val_tooltip)

    ##########################################
    # Make the slider marks for this dataset #
    ##########################################

    lfs = labels.find(video)

    slider_marks = []
    track_row = 0

    # Add marks with track
    track_occupancy = labels.get_track_occupancy(video)
    for track in labels.tracks:
        if track in track_occupancy and not track_occupancy[track].is_empty:
            if track_row > 0 and slider._is_track_in_new_column(track_row):
                slider_marks.append(
                    SliderMark("tick_column", val=track_occupancy[track].start)
                )

            track_len = track_occupancy[track].end - track_occupancy[track].start

            # for debugging we can only show tracks above certain length
            if track_len > SEEKBAR_MIN_TRACK_LEN_TO_SHOW:
                for occupancy_range in track_occupancy[track].list:
                    slider_marks.append(
                        SliderMark(
                            "track",
                            val=occupancy_range[0],
                            end_val=occupancy_range[1],
                            row=track_row,
                            color=color_manager.get_track_color(track),
                        )
                    )
                track_row += 1

    # Frames with instance without track
    untracked_frames = set()
    if None in track_occupancy:
        for occupancy_range in track_occupancy[None].list:
            untracked_frames.update({val for val in range(*occupancy_range)})

    labeled_marks = {lf.frame_idx for lf in lfs}
    user_labeled = {lf.frame_idx for lf in lfs if len(lf.user_instances)}
    suggested_frames = set(labels.get_video_suggestions(video))

    all_simple_frames = set()
    all_simple_frames.update(untracked_frames)
    all_simple_frames.update(suggested_frames)
    all_simple_frames.update(user_labeled)

    for frame_idx in all_simple_frames:
        if frame_idx in suggested_frames:
            if frame_idx in user_labeled:
                # suggested frame with user labeled instances
                mark_type = SemanticMarkType.suggested_with_user
            elif frame_idx in labeled_marks:
                # suggested frame with only predicted instances
                mark_type = SemanticMarkType.suggested_with_predicted
            else:
                # suggested frame without any instances
                mark_type = SemanticMarkType.suggested_with_nothing
        elif frame_idx in user_labeled:
            # frame with user labeled instances
            mark_type = SemanticMarkType.user
        else:
            # no user instances, predicted instance without track identity
            mark_type = SemanticMarkType.predicted_no_track

        mark_type = mark_type.value

        slider_marks.append(SliderMark(mark_type, val=frame_idx))

    slider.setNumberOfTracks(track_row)  # total number of tracks to show
    slider.setMarks(slider_marks)


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

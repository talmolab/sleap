"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2.QtWidgets import QApplication, QWidget, QLayout, QAbstractSlider
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtWidgets import QSizePolicy, QLabel, QGraphicsRectItem
from PySide2.QtGui import QPainter, QPen, QBrush, QColor, QKeyEvent, QPolygonF, QPainterPath
from PySide2.QtCore import Qt, Signal, QRect, QRectF, QPointF

from sleap.gui.overlays.tracks import TrackColorManager

import attr
import itertools
import numpy as np
from typing import Union

@attr.s(auto_attribs=True, cmp=False)
class SliderMark:
    type: str
    val: float
    end_val: float=None
    row: int=None
    track: 'Track'=None
    _color: Union[tuple,str]="black"

    @property
    def color(self):
        colors = dict(simple="black",
                      filled="blue",
                      open="blue",
                      predicted="red")

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

class VideoSlider(QGraphicsView):
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
    """

    mousePressed = Signal(float, float)
    mouseMoved = Signal(float, float)
    mouseReleased = Signal(float, float)
    keyPress = Signal(QKeyEvent)
    keyRelease = Signal(QKeyEvent)
    valueChanged = Signal(int)
    selectionChanged = Signal(int, int)
    updatedTracks = Signal()

    def __init__(self, orientation=-1, min=0, max=100, val=0,
            marks=None, tracks=0,
            color_manager=None,
            *args, **kwargs):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # ScrollBarAsNeeded

        self._color_manager = color_manager

        self._track_rows = 0
        self._track_height = 3
        self._header_height = 0
        self._min_height = 19 + self._header_height

        # Add border rect
        slider_rect = QRect(0, 0, 200, self._min_height-3)
        self.slider = self.scene.addRect(slider_rect)
        self.slider.setPen(QPen(QColor("black")))

        # Add drag handle rect
        handle_width = 6
        handle_rect = QRect(0, self._handleTop(), handle_width, self._handleHeight())
        self.setMinimumHeight(self._min_height)
        self.setMaximumHeight(self._min_height)
        self.handle = self.scene.addRect(handle_rect)
        self.handle.setPen(QPen(QColor(80, 80, 80)))
        self.handle.setBrush(QColor(128, 128, 128, 128))

        # Add (hidden) rect to highlight selection
        self.select_box = self.scene.addRect(QRect(0, 1, 0, slider_rect.height()-2))
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

        pen = QPen(QColor(80, 80, 255), .5)
        pen.setCosmetic(True)
        self.poly = self.scene.addPath(QPainterPath(), pen, self.select_box.brush())
        self.headerSeries = dict()
        self.drawHeader()

    def _pointsToPath(self, points):
        path = QPainterPath()
        path.addPolygon(QPolygonF(points))
        return path

    def setTracksFromLabels(self, labels, video):
        """Set slider marks using track information from `Labels` object.

        Note that this is the only method coupled to a SLEAP object.

        Args:
            labels: the `labels` with tracks and labeled_frames
            video: the video for which to show marks
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
                    slider_marks.append(SliderMark("track", val=occupancy_range[0], end_val=occupancy_range[1], row=track_row, color=self._color_manager.get_color(track)))
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

        self.setTracks(track_row) # total number of tracks to show
        self.setMarks(slider_marks)

        # self.setHeaderSeries(lfs)

        self.updatedTracks.emit()

    def setHeaderSeries(self, lfs):
        # calculate total point distance for instances from last labeled frame
        def inst_velocity(lf, last_lf):
            val = 0
            for inst in lf:
                if last_lf is not None:
                    last_inst = last_lf.find(track=inst.track)
                    if last_inst:
                        points_a = inst.points_array(invisible_as_nan=True)
                        points_b = last_inst[0].points_array(invisible_as_nan=True)
                        point_dist = np.linalg.norm(points_a - points_b, axis=1)
                        inst_dist = np.sum(point_dist) # np.nanmean(point_dist)
                        val += inst_dist if not np.isnan(inst_dist) else 0
            return val

        series = dict()

        last_lf = None
        for lf in lfs:
            val = inst_velocity(lf, last_lf)
            last_lf = lf
            if not np.isnan(val):
                series[lf.frame_idx] = val #len(lf.instances)

        self.headerSeries = series
        self.drawHeader()

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
        self.resizeEvent()

    def _toPos(self, val, center=False):
        """Convert value to x position on slider."""
        x = val
        x -= self._val_min
        x /= max(1, self._val_max-self._val_min)
        x *= self._sliderWidth()
        if center:
            x  += self.handle.rect().width()/2.
        return x

    def _toVal(self, x, center=False):
        """Convert x position to value."""
        val = x
        val /= self._sliderWidth()
        val *= max(1, self._val_max-self._val_min)
        val += self._val_min
        val = round(val)
        return val

    def _sliderWidth(self):
        return self.slider.rect().width()-self.handle.rect().width()

    def value(self):
        """Get value of slider."""
        return self._val_main

    def setValue(self, val):
        """Set value of slider."""
        self._val_main = val
        x = self._toPos(val)
        self.handle.setPos(x, 0)

    def setMinimum(self, min):
        """Set minimum value for slider."""
        self._val_min = min

    def setMaximum(self, max):
        """Set maximum value for slider."""
        self._val_max = max

    def setEnabled(self, val):
        """Set whether the slider is enabled."""
        self._enabled = val

    def enabled(self):
        """Returns whether slider is enabled."""
        return self._enabled

    def clearSelection(self):
        """Clear selection endpoints."""
        self._selection = []
        self.select_box.hide()

    def startSelection(self, val):
        """Add initial selection endpoint.

        Args:
            val: value of endpoint
        """
        self._selection.append(val)

    def endSelection(self, val, update=False):
        """Add final selection endpoint.

        Args:
            val: value of endpoint
        """
        # If we want to update endpoint and there's already one, remove it
        if update and len(self._selection)%2==0:
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
        """Return True if a clip is selected, False otherwise."""
        a, b = self.getSelection()
        return a < b

    def getSelection(self):
        """Return start and end value of current selection endpoints."""
        a, b = 0, 0
        if len(self._selection)%2 == 0 and len(self._selection) > 0:
            a, b = self._selection[-2:]
        start = min(a, b)
        end = max(a, b)
        return start, end

    def drawSelection(self, a, b):
        """Draw selection box on slider.

        Args:
            a: one endpoint value
            b: other endpoint value
        """
        start = min(a, b)
        end = max(a, b)
        start_pos = self._toPos(start, center=True)
        end_pos = self._toPos(end, center=True)
        selection_rect = QRect(start_pos, 1,
                               end_pos-start_pos, self.slider.rect().height()-2)

        self.select_box.setRect(selection_rect)
        self.select_box.show()

    def moveSelectionAnchor(self, x, y):
        """Move selection anchor in response to mouse position.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x = max(x, 0)
        x = min(x, self.slider.rect().width())
        anchor_val = self._toVal(x, center=True)

        if len(self._selection)%2 == 0:
            self.startSelection(anchor_val)

        self.drawSelection(anchor_val, self._selection[-1])

    def releaseSelectionAnchor(self, x, y):
        """Finish selection in response to mouse release.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x = max(x, 0)
        x = min(x, self.slider.rect().width())
        anchor_val = self._toVal(x)
        self.endSelection(anchor_val)

    def clearMarks(self):
        """Clear all marked values for slider."""
        if hasattr(self, "_mark_items"):
            for item in self._mark_items.values():
                self.scene.removeItem(item)
        self._marks = set() # holds mark position
        self._mark_items = dict() # holds visual Qt object for plotting mark

    def setMarks(self, marks):
        """Set all marked values for the slider.

        Args:
            marks: iterable with all values to mark
        """
        self.clearMarks()
        if marks is not None:
            for mark in marks:
                if not isinstance(mark, SliderMark):
                    mark = SliderMark("simple", mark)
                    print(mark)
                self.addMark(mark, update=False)
        self.updatePos()

    def getMarks(self):
        """Return list of marks.

        Each mark is either val or (track, val)-tuple.
        """
        return self._marks

    def addMark(self, new_mark, update=True):
        """Add a marked value to the slider.

        Args:
            new_mark: value to mark
        """
        # check if mark is within slider range
        if new_mark.val > self._val_max: return
        if new_mark.val < self._val_min: return

        self._marks.add(new_mark)

        v_top_pad = 3 + self._header_height
        v_bottom_pad = 3

        width = 0

        if new_mark.type == "track":
            v_offset = v_top_pad + (self._track_height * new_mark.row)
            height = 1
        else:
            v_offset = v_top_pad
            height = self.slider.rect().height()-(v_offset+v_bottom_pad)

            width = 2 if new_mark.type in ("open", "filled") else 0

        color = new_mark.QColor
        pen = QPen(color, .5)
        pen.setCosmetic(True)
        brush = QBrush(color) if new_mark.filled else QBrush()

        line = self.scene.addRect(-width//2, v_offset, width, height,
                                  pen, brush)
        self._mark_items[new_mark] = line
        if update: self.updatePos()

    def _mark_val(self, mark):
        return mark.val

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
        if len(self.headerSeries) == 0 or self._header_height == 0:
            self.poly.setPath(QPainterPath())
            return

        step = max(self.headerSeries.keys())//int(self._sliderWidth())
        step = max(step, 1)
        count = max(self.headerSeries.keys())//step*step

        sampled = np.full((count), 0.0)
        for key, val in self.headerSeries.items():
            if key < count:
                sampled[key] = val
        sampled = np.max(sampled.reshape(count//step,step), axis=1)
        series = {i*step:sampled[i] for i in range(count//step)}

#         series = {key:self.headerSeries[key] for key in sorted(self.headerSeries.keys())}

        series_min = np.min(sampled) - 1
        series_max = np.max(sampled)
        series_scale = (self._header_height-5)/(series_max - series_min)

        def toYPos(val):
            return self._header_height-((val-series_min)*series_scale)

        step_chart = False # use steps rather than smooth line

        points = []
        points.append((self._toPos(0, center=True), toYPos(series_min)))
        for idx, val in series.items():
            points.append((self._toPos(idx, center=True), toYPos(val)))
            if step_chart:
                points.append((self._toPos(idx+step, center=True), toYPos(val)))
        points.append((self._toPos(max(series.keys()) + 1, center=True), toYPos(series_min)))

        # Convert to list of QPointF objects
        points = list(itertools.starmap(QPointF,points))
        self.poly.setPath(self._pointsToPath(points))

    def moveHandle(self, x, y):
        """Move handle in response to mouse position.

        Emits valueChanged signal if value of slider changed.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x -= self.handle.rect().width()/2.
        x = max(x, 0)
        x = min(x, self.slider.rect().width()-self.handle.rect().width())

        val = self._toVal(x)

        # snap to nearby mark within handle
        mark_vals = [self._mark_val(mark) for mark in self._marks]
        handle_left = self._toVal(x - self.handle.rect().width()/2)
        handle_right = self._toVal(x + self.handle.rect().width()/2)
        marks_in_handle = [mark for mark in mark_vals
                           if handle_left < mark < handle_right]
        if marks_in_handle:
            marks_in_handle.sort(key=lambda m: (abs(m-val), m>val))
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

        slider_rect = self.slider.rect()
        handle_rect = self.handle.rect()
        select_box_rect = self.select_box.rect()

        slider_rect.setHeight(height-3)
        if event is not None: slider_rect.setWidth(event.size().width()-1)
        handle_rect.setHeight(self._handleHeight())
        select_box_rect.setHeight(self._handleHeight())

        self.slider.setRect(slider_rect)
        self.handle.setRect(handle_rect)
        self.select_box.setRect(select_box_rect)

        self.updatePos()
        self.drawHeader()
        super(VideoSlider, self).resizeEvent(event)

    def _handleTop(self):
        return 1 + self._header_height

    def _handleHeight(self, slider_rect=None):
        if slider_rect is None:
            slider_rect = self.slider.rect()

        handle_bottom_offset = 1
        handle_height = slider_rect.height() - (self._handleTop()+handle_bottom_offset)
        return handle_height

    def mousePressEvent(self, event):
        """Override method to move handle for mouse press/drag.

        Args:
            event
        """
        scenePos = self.mapToScene(event.pos())

        # Do nothing if not enabled
        if not self.enabled(): return
        # Do nothing if click outside slider area
        if not self.slider.rect().contains(scenePos): return

        move_function = None
        release_function = None

        if event.modifiers() == Qt.ShiftModifier:
            move_function = self.moveSelectionAnchor
            release_function = self.releaseSelectionAnchor

            self.clearSelection()

        elif event.modifiers() == Qt.NoModifier:
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

    def boundingRect(self) -> QRectF:
        """Method required by Qt."""
        return self.slider.rect()

    def paint(self, *args, **kwargs):
        """Method required by Qt."""
        super(VideoSlider, self).paint(*args, **kwargs)


if __name__ == "__main__":
    app = QApplication([])

    window = VideoSlider(
                min=0, max=20, val=15,
                marks=(10,15)#((0,10),(0,15),(1,10),(1,11),(2,12)), tracks=3
                )

    window.valueChanged.connect(lambda x: print(x))
    window.show()

    app.exec_()

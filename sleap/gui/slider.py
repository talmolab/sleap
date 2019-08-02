"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2.QtWidgets import QApplication, QWidget, QLayout, QAbstractSlider
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtWidgets import QSizePolicy, QLabel, QGraphicsRectItem
from PySide2.QtGui import QPainter, QPen, QBrush, QColor, QKeyEvent
from PySide2.QtCore import Qt, Signal, QRect, QRectF

from sleap.gui.tracks import TrackColorManager

from operator import itemgetter
from itertools import groupby

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

        self._color_manager = color_manager or TrackColorManager()

        self._track_height = 3

        height = 19
        slider_rect = QRect(0, 0, 200, height-3)
        handle_width = 6
        handle_rect = QRect(0, 1, handle_width, slider_rect.height()-2)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)

        self.slider = self.scene.addRect(slider_rect)
        self.slider.setPen(QPen(QColor("black")))

        self.handle = self.scene.addRect(handle_rect)
        self.handle.setPen(QPen(QColor(80, 80, 80)))
        self.handle.setBrush(QColor(128, 128, 128, 128))

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

    def setTracksFromLabels(self, labels, video):
        """Set slider marks using track information from `Labels` object.

        Note that this is the only method coupled to a SLEAP object.

        Args:
            labels: the `labels` with tracks and labeled_frames
            video: the video for which to show marks
        """
        lfs = labels.find(video)

        slider_marks = []
        track_idx = 0

        # Add marks with track
        track_occupancy = labels.get_track_occupany(video)
        for track in labels.tracks:
#             track_idx = labels.tracks.index(track)
            if track in track_occupancy and not track_occupancy[track].is_empty:
                for occupancy_range in track_occupancy[track].list:
                    slider_marks.append((track_idx, *occupancy_range))
                track_idx += 1

        # Add marks without track
        if None in track_occupancy:
            for occupancy_range in track_occupancy[None].list:
                slider_marks.extend(range(*occupancy_range))

        # list of frame_idx for simple markers for labeled frames
        labeled_marks = [lf.frame_idx for lf in lfs]
        user_labeled = [lf.frame_idx for lf in lfs if len(lf.user_instances)]
        # "f" for suggestions with instances and "o" for those without
        # "f" means "filled", "o" means "open"
        # "p" for suggestions with only predicted instances
        def mark_type(frame):
            if frame in user_labeled:
                return "f"
            elif frame in labeled_marks:
                return "p"
            else:
                return "o"
        # list of (type, frame) tuples for suggestions
        suggestion_marks = [(mark_type(frame_idx), frame_idx)
            for frame_idx in labels.get_video_suggestions(video)]
        # combine marks for labeled frame and marks for suggested frames
        slider_marks.extend(suggestion_marks)

        self.setTracks(track_idx)
        self.setMarks(slider_marks)

        self.updatedTracks.emit()

    def setTracks(self, tracks):
        """Set the number of tracks to show in slider.

        Args:
            tracks: the number of tracks to show
        """
        if tracks == 0:
            min_height = max_height = 19
        else:
            min_height = max(19, 8 + (self._track_height * min(tracks, 20)))
            max_height = max(19, 8 + (self._track_height * tracks))

        self.setMaximumHeight(max_height)
        self.setMinimumHeight(min_height)
        self.resizeEvent()

    def _toPos(self, val, center=False):
        x = val
        x -= self._val_min
        x /= max(1, self._val_max-self._val_min)
        x *= self._sliderWidth()
        if center:
            x  += self.handle.rect().width()/2.
        return x

    def _toVal(self, x, center=False):
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
        if self._mark_val(new_mark) > self._val_max: return
        if self._mark_val(new_mark) < self._val_min: return

        self._marks.add(new_mark)

        width = 0
        filled = True
        if type(new_mark) == tuple:
            if type(new_mark[0]) == int:
                # colored track if mark has format: (track_number, start_frame_idx, end_frame_idx)
                track = new_mark[0]
                v_offset = 3 + (self._track_height * track)
                height = 1
                color = QColor(*self._color_manager.get_color(track))
                width = 0
            else:
                # rect (open/filled) if format: ("o", frame_idx) or ("f", frame_idx)
                # ("p", frame_idx) when only predicted instances on frame
                mark_type = new_mark[0]
                v_offset = 3
                height = self.slider.rect().height()-6
                width = 2
                color = QColor("blue")
                if mark_type == "o":
                    filled = False
                if mark_type == "p":
                    color = QColor("red")
        else:
            # line if mark has format: frame_idx
            v_offset = 3
            height = self.slider.rect().height()-6
            color = QColor("black")

        pen = QPen(color, .5)
        pen.setCosmetic(True)
        brush = QBrush(color) if filled else QBrush()

        line = self.scene.addRect(-width//2, v_offset, width, height,
                                  pen, brush)
        self._mark_items[new_mark] = line
        if update: self.updatePos()

    def _mark_val(self, mark):
        return mark[1] if type(mark) == tuple else mark

    def updatePos(self):
        """Update the visual position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)
        for mark in self._mark_items.keys():
            if type(mark) == tuple:
                in_track = True
                v = mark[1]
                if type(mark[0]) == int:
                    width_in_frames = mark[2] - mark[1]
                    width = max(2, self._toPos(width_in_frames))
                else:
                    width = 2
            else:
                in_track = False
                v = mark
                width = 0
            x = self._toPos(v, center=True)
            self._mark_items[mark].setPos(x, 0)

            rect = self._mark_items[mark].rect()
            rect.setWidth(width)

            self._mark_items[mark].setRect(rect)

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
        handle_rect.setHeight(slider_rect.height()-2)
        select_box_rect.setHeight(slider_rect.height()-2)

        self.slider.setRect(slider_rect)
        self.handle.setRect(handle_rect)
        self.select_box.setRect(select_box_rect)

        self.updatePos()
        super(VideoSlider, self).resizeEvent(event)

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
    window.setTracks(5)
#     mark_positions = ((0,10),(0,15),(1,10),(1,11),(2,12),(3,12),(3,13),(3,14),(4,15),(4,16),(4,21))
    mark_positions = [("o",i) for i in range(3,15,4)] + [("f",18)]
    window.setMarks(mark_positions)
    window.valueChanged.connect(lambda x: print(x))
    window.show()

    app.exec_()

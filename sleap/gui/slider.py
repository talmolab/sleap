"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2.QtWidgets import QApplication, QWidget, QLayout, QAbstractSlider
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtWidgets import QSizePolicy, QLabel, QGraphicsRectItem
from PySide2.QtGui import QPainter, QPen, QBrush, QColor, QKeyEvent
from PySide2.QtCore import Qt, Signal, QRect, QRectF

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

    def __init__(self, orientation=-1, min=0, max=100, val=0,
            marks=None, tracks=0,
            *args, **kwargs):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.color_maps = [
            [0,   114,   189],
            [217,  83,    25],
            [237, 177,    32],
            [126,  47,   142],
            [119, 172,    48],
            [77,  190,   238],
            [162,  20,    47],
            ]

        self._track_height = 3
        height = 19 if tracks == 0 else 8 + (self._track_height * tracks)
        if height < 19: height = 19
        slider_rect = QRect(0, 0, 200, height-3)
        handle_width = 6
        handle_rect = QRect(0, 1, handle_width, slider_rect.height()-2)

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

    def setTracksFromLabels(self, labels):
        """Set slider marks using track information from `Labels` object.

        Note that this is the only method coupled to a SLEAP object.

        Args:
            labels: the `labels` with tracks and labeled_frames
        """
        track_count = len(labels.tracks)
        slider_marks = []

        for labeled_frame in labels.labeled_frames:
            for instance in labeled_frame.instances:
                frame_idx = labeled_frame.frame_idx
                if instance.track is not None:
                    # Add mark with track
                    slider_marks.append((labels.tracks.index(instance.track), frame_idx))
                else:
                    # Add mark without track
                    slider_marks.append(frame_idx)

        self.setTracks(track_count)
        self.setMarks(slider_marks)

    def setTracks(self, tracks):
        """Set the number of tracks to show in slider.
        
        Args:
            tracks: the number of tracks to show
        """
        height = 19 if tracks == 0 else 8 + (self._track_height * tracks)
        if height < 19: height = 19
        self._set_height(height)

    def _set_height(self, height):
        slider_rect = self.slider.rect()
        handle_rect = self.handle.rect()
        select_box_rect = self.select_box.rect()

        slider_rect.setHeight(height-3)
        handle_rect.setHeight(slider_rect.height()-2)
        select_box_rect.setHeight(slider_rect.height()-2)

        self.slider.setRect(slider_rect)
        self.handle.setRect(handle_rect)
        self.select_box.setRect(select_box_rect)

        self.setMaximumHeight(height)

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
        start_pos = self._toPos(start)
        end_pos = self._toPos(end)
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
        anchor_val = self._toVal(x)

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
        self._marks.add(new_mark)

        if type(new_mark) == tuple:
            track = new_mark[0]
            v_offset = 3 + (self._track_height * track)
            height = 1
            color = QColor(*self.color_maps[track%len(self.color_maps)])
        else:
            v_offset = 3
            height = self.slider.rect().height()-6
            color = QColor("black")

        line = self.scene.addRect(0, v_offset, 0, height,
                                  QPen(color), QBrush(color))
        self._mark_items[new_mark] = line
        if update: self.updatePos()

    def updatePos(self):
        """Update the visual position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)
        for mark in self._mark_items.keys():
            if type(mark) == tuple:
                in_track = True
                v = mark[1]
            else:
                in_track = False
                v = mark
            x = self._toPos(v, center=True)
            self._mark_items[mark].setPos(x, 0)

            rect = self._mark_items[mark].rect()
            width = 0 if not in_track else self._toPos(1)
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
        old = self.value()
        self.setValue(val)

        if old != val:
            self.valueChanged.emit(self._val_main)

    def resizeEvent(self, event):
        """Override method to update visual size when necessary.

        Args:
            event
        """
        rect = self.slider.rect()
        rect.setWidth(event.size().width()-1)
        self.slider.setRect(rect)
        self.updatePos()


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

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
        pass

if __name__ == "__main__":
    app = QApplication([])

    window = VideoSlider(
                min=0, max=500, val=15,
                marks=(10,15)#((0,10),(0,15),(1,10),(1,11),(2,12)), tracks=3
                )
    window.setTracks(5)
    mark_positions = ((0,10),(0,15),(1,10),(1,11),(2,12),(3,12),(3,13),(3,14),(4,15),(4,16),(4,21))
#     mark_positions = [(0,i) for i in range(200)]
    window.setMarks(mark_positions)
    window.valueChanged.connect(lambda x: print(x))
    window.show()

    app.exec_()

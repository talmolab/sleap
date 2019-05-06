"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2.QtWidgets import QApplication, QWidget, QLayout, QAbstractSlider
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtWidgets import QSizePolicy, QLabel, QGraphicsRectItem
from PySide2.QtGui import QPainter, QPen, QBrush, QColor
from PySide2.QtCore import Qt, Signal, QRect, QRectF

class VideoSlider(QGraphicsView):
    """Drop-in replacement for QSlider with additional features.

    Args:
        orientation: ignored (here for compatibility with QSlider)
        min: initial minimum value
        max: initial maximum value
        val: initial value
        labels: initial set of values to label on slider
    """

    mousePressed = Signal(float, float)
    mouseMoved = Signal(float, float)
    mouseReleased = Signal(float, float)
    valueChanged = Signal(int)
    selectionChanged = Signal(int, int)

    def __init__(self, orientation=-1, min=0, max=100, val=0, labels=None, *args, **kwargs):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        height = 19
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
        self.setLabels(labels)

    def _toPos(self, val, center=False):
        x = val
        x -= self._val_min
        x /= (self._val_max-self._val_min)
        x *= self._sliderWidth()
        if center:
            x  += self.handle.rect().width()/2.
        return x

    def _toVal(self, x, center=False):
        val = x
        val /= self._sliderWidth()
        val *= (self._val_max-self._val_min)
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

    def endSelection(self, val):
        """Add final selection endpoint.

        Args:
            val: value of endpoint
        """
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

    def clearLabels(self):
        """Clear all labeled values for slider."""
        if hasattr(self, "_label_items"):
            for item in self._label_items.values():
                self.scene.removeItem(item)
        self._labels = set()
        self._label_items = dict()

    def setLabels(self, labels):
        """Set all labeled values for the slider.

        Args:
            labels: iterable with all values to label
        """
        self.clearLabels()
        self._labels = set() if labels is None else set(labels)
        for v in self._labels:
            line = self.scene.addLine(0, 3, 0, self.slider.rect().height()-3,
                                      QPen(QColor("black")))
            self._label_items[v] = line
        self.updatePos()

    def addLabel(self, new_label):
        """Add a labeled value to the slider.

        Args:
            new_label: value to label
        """
        self._labels.add(new_label)
        line = self.scene.addLine(0, 3, 0, self.slider.rect().height()-3,
                                  QPen(QColor("black")))
        self._label_items[new_label] = line
        self.updatePos()

    def updatePos(self):
        """Update the visual position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)
        for v in self._label_items.keys():
            x = self._toPos(v, center=True)
            self._label_items[v].setPos(x, 0)

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

        if event.modifiers() == Qt.AltModifier:
            move_function = self.moveSelectionAnchor
            release_function = self.releaseSelectionAnchor

            self.clearSelection()

        elif event.modifiers() == Qt.NoModifier:
            move_function = self.moveHandle
            release_function = None

        # Connect to signals
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

    def boundingRect(self) -> QRectF:
        """Method required by Qt."""
        return self.slider.rect()

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
        pass

if __name__ == "__main__":
    app = QApplication([])

    window = VideoSlider(0, 100, 15, set((10,15)))
    window.valueChanged.connect(lambda x: print(x))
    window.show()

    app.exec_()

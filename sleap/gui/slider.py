"""
Drop-in replacement for QSlider with additional features.
"""

from PySide2.QtWidgets import QApplication, QWidget, QLayout, QAbstractSlider
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide2.QtWidgets import QSizePolicy, QLabel, QGraphicsRectItem
from PySide2.QtGui import QPainter, QPen, QBrush, QColor
from PySide2.QtCore import Qt, Signal, QRect, QRectF

class VideoSlider(QGraphicsView):

    mousePressed = Signal(float, float)
    mouseMoved = Signal(float, float)
    mouseReleased = Signal(float, float)
    valueChanged = Signal(int)

    def __init__(self, min=0, max=100, val=0, labels=None, *args, **kwargs):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        height = 19
        sliderRect = QRect(0, 0, 200, height-3)
        handleWidth = 6
        handleRect = QRect(0, 1, handleWidth, sliderRect.height()-2)

        self.setMaximumHeight(height)
        
        self.slider = self.scene.addRect(sliderRect)
        self.slider.setPen(QPen(QColor("black")))
        
        self.handle = self.scene.addRect(handleRect)
        self.handle.setPen(QPen(QColor(80, 80, 80)))
        self.handle.setBrush(QColor(128, 128, 128, 128))

        self.scene.setBackgroundBrush(QBrush(QColor(200, 200, 200)))

        self.setEnabled(True)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.setLabels(labels)

    def _toPos(self, val):
        x = val
        x -= self._val_min
        x /= (self._val_max-self._val_min)
        x *= self._sliderWidth()
        return x

    def _toVal(self, x):
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
            x = self._toPos(v) + self.handle.rect().width()/2.
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
        if not self.enabled(): return

        scenePos = self.mapToScene(event.pos())
        
        def done(x, y):
            self.mouseMoved.disconnect(self.moveHandle)
            self.mouseReleased.disconnect(done)
        
        # If click within slider, then enable handle dragging
        if self.slider.rect().contains(scenePos):
            self.mouseMoved.connect(self.moveHandle)
            self.moveHandle(scenePos.x(), scenePos.y())
            self.mouseReleased.connect(done)

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

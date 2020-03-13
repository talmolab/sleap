from sleap import Video
from sleap.gui.widgets.video import GraphicsView

from typing import Optional, Text

from PySide2 import QtWidgets, QtGui


class ReceptiveFieldWidget(QtWidgets.QWidget):
    def __init__(self, head_name: Text, *args, **kwargs):
        super(ReceptiveFieldWidget, self).__init__(*args, **kwargs)

        self.layout = QtWidgets.QVBoxLayout()

        self._field_image_widget = ReceptiveFieldImageWidget()

        self._info_text = (
            f"Receptive Field for {head_name}:<br />"
            if head_name
            else "Receptive Field:<br />"
        )
        self._info_widget = QtWidgets.QLabel("")

        self.layout.addWidget(self._field_image_widget)
        self.layout.addWidget(self._info_widget)
        self.layout.addStretch()

        self.setLayout(self.layout)

    def setFieldSize(self, size, scale):
        self._info_widget.setText(self._info_text + f"{size} pixels")
        self._field_image_widget.setFieldSize(size, scale)

    def setImage(self, *args, **kwargs):
        self._field_image_widget.setImage(*args, **kwargs)


class ReceptiveFieldImageWidget(GraphicsView):
    def __init__(self, *args, **kwargs):
        self._widget_size = 200
        self._pen_width = 4
        self._box_size = None
        self._scale = None

        box_pen = QtGui.QPen(QtGui.QColor("blue"), self._pen_width)
        box_pen.setCosmetic(True)

        self.box = QtWidgets.QGraphicsRectItem()
        self.box.setPen(box_pen)

        super(ReceptiveFieldImageWidget, self).__init__(*args, **kwargs)

        self.setFixedSize(self._widget_size, self._widget_size)
        self.scene.addItem(self.box)

        # TODO: zoom around bounding box for labeled instance
        # self.zoomToRect(QtCore.QRectF(0, 0, 1, 1))

    def viewportEvent(self, event):
        # Update the position and visible size of field
        self.setFieldSize()

        # Now draw the viewport
        return super(ReceptiveFieldImageWidget, self).viewportEvent(event)

    def setFieldSize(self, size: Optional[int] = None, scale: int = 1.0):
        if size is not None:
            self._box_size = size
            self._scale = scale

        if self._box_size:
            self.box.show()
        else:
            self.box.hide()
            return

        scaled_box_size = self._box_size // self._scale

        # TODO
        # Calculate offset so that box stays centered in the view
        # visible_box_size = (self._box_size + (self._pen_width * 2)) / self.zoomFactor

        offset = (self._widget_size) // 2
        scene_offset = self.mapToScene(offset, offset)

        self.box.setRect(
            scene_offset.x(), scene_offset.y(), scaled_box_size, scaled_box_size
        )


def demo_receptive_field():
    app = QtWidgets.QApplication([])

    video = Video.from_filename("tests/data/videos/centered_pair_small.mp4")

    win = ReceptiveFieldImageWidget()
    win.setImage(video.get_frame(0))
    win.setFieldSize(50)

    win.show()
    app.exec_()


if __name__ == "__main__":
    demo_receptive_field()

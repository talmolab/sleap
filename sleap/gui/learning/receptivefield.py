from sleap import Video
from sleap.gui.learning import utils
from sleap.nn.config import ModelConfig
from sleap.gui.widgets.video import GraphicsView

from typing import Optional, Text

from PySide2 import QtWidgets, QtGui, QtCore


class ReceptiveFieldWidget(QtWidgets.QWidget):
    def __init__(self, head_name: Text, *args, **kwargs):
        super(ReceptiveFieldWidget, self).__init__(*args, **kwargs)

        self.layout = QtWidgets.QVBoxLayout()

        self._field_image_widget = ReceptiveFieldImageWidget()

        self._info_text_header = (
            f"<p>Receptive Field for {head_name}:</p>"
            if head_name
            else "<p>Receptive Field:</p>"
        )

        self._info_widget = QtWidgets.QLabel("")

        self.layout.addWidget(self._field_image_widget)
        self.layout.addWidget(self._info_widget)
        self.layout.addStretch()

        self.setLayout(self.layout)

    def getInfoText(
        self, size, scale, max_stride, down_blocks, convs_per_block, kernel_size
    ) -> Text:
        result = self._info_text_header
        if size:
            result += f"<p><i>{size} pixels</i></p>"
        else:
            result += f"<p><i>Unable to determine size</i></p>"

        result += f"""
        <p>Receptive field size is a function<br />
        of the number of down blocks ({down_blocks}), the<br />
        number of convolutions per block ({convs_per_block}),<br />
        and the convolution kernel size ({kernel_size}).</p>

        <p>You can control the number of down<br />
        blocks by setting the <b>Max Stride</b> ({max_stride}).</p>

        <p>The number of convolutions per block<br />
        and the kernel size are currently fixed<br />
        by your choice of backbone.</p>

        <p>You can also control the receptive<br />
        field size relative to the original<br />
        image by adjusting the <b>Input Scaling</b> ({scale}).</p>
        """

        return result

    def setModelConfig(self, model_cfg: ModelConfig, scale: float):
        rf_info = utils.receptive_field_info_from_model_cfg(model_cfg)

        self._info_widget.setText(
            self.getInfoText(
                size=rf_info["size"],
                scale=scale,
                max_stride=rf_info["max_stride"],
                down_blocks=rf_info["down_blocks"],
                convs_per_block=rf_info["convs_per_block"],
                kernel_size=rf_info["kernel_size"],
            )
        )

        self._field_image_widget.setFieldSize(rf_info["size"] or 0, scale)

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
        if isinstance(event, QtGui.QPaintEvent):
            self.setFieldSize()

        # Now draw the viewport
        return super(ReceptiveFieldImageWidget, self).viewportEvent(event)

    def setFieldSize(self, size: Optional[int] = None, scale: float = 1.0):
        if size is not None:
            self._box_size = size
            self._scale = scale

        if self._box_size:
            self.box.show()
        else:
            self.box.hide()
            return

        # Adjust box relative to scaling on image that will happen in training
        scaled_box_size = self._box_size // self._scale

        # Calculate offset so that box stays centered in the view
        vis_box_rect = self.mapFromScene(
            0, 0, scaled_box_size, scaled_box_size
        ).boundingRect()
        offset = self._widget_size / 2
        scene_center = self.mapToScene(
            offset - (vis_box_rect.width() / 2), offset - (vis_box_rect.height() / 2)
        )

        self.box.setRect(
            scene_center.x(), scene_center.y(), scaled_box_size, scaled_box_size
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

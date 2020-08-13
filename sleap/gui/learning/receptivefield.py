"""
Widget for previewing receptive field on sample image using model hyperparams.
"""
from sleap import Video
from sleap.nn.config import ModelConfig
from sleap.gui.widgets.video import GraphicsView

import numpy as np

from sleap import Skeleton
from sleap.nn.model import Model

from typing import Optional, Text

from PySide2 import QtWidgets, QtGui, QtCore


def compute_rf(down_blocks: int, convs_per_block: int = 2, kernel_size: int = 3) -> int:
    """
    Computes receptive field for specified model architecture.

    Ref: https://distill.pub/2019/computing-receptive-fields/ (Eq. 2)
    """
    # Define the strides and kernel sizes for a single down block.
    # convs have stride 1, pooling has stride 2:
    block_strides = [1] * convs_per_block + [2]

    # convs have `kernel_size` x `kernel_size` kernels, pooling has 2 x 2 kernels:
    block_kernels = [kernel_size] * convs_per_block + [2]

    # Repeat block parameters by the total number of down blocks.
    strides = np.array(block_strides * down_blocks)
    kernels = np.array(block_kernels * down_blocks)

    # L = Total number of layers
    L = len(strides)

    # Compute the product term of the RF equation.
    rf = 1
    for l in range(L):
        rf += (kernels[l] - 1) * np.prod(strides[:l])

    return int(rf)


def receptive_field_info_from_model_cfg(model_cfg: ModelConfig) -> dict:
    """Gets receptive field information given specific model configuration."""
    rf_info = dict(
        size=None,
        max_stride=None,
        down_blocks=None,
        convs_per_block=None,
        kernel_size=None,
    )

    try:
        model = Model.from_config(model_cfg, Skeleton())
    except ZeroDivisionError:
        # Unable to create model from these config parameters
        return rf_info

    if hasattr(model_cfg.backbone.which_oneof(), "max_stride"):
        rf_info["max_stride"] = model_cfg.backbone.which_oneof().max_stride

    if hasattr(model.backbone, "down_convs_per_block"):
        rf_info["convs_per_block"] = model.backbone.down_convs_per_block
    elif hasattr(model.backbone, "convs_per_block"):
        rf_info["convs_per_block"] = model.backbone.convs_per_block

    if hasattr(model.backbone, "kernel_size"):
        rf_info["kernel_size"] = model.backbone.kernel_size

    rf_info["down_blocks"] = model.backbone.down_blocks

    if rf_info["down_blocks"] and rf_info["convs_per_block"] and rf_info["kernel_size"]:
        rf_info["size"] = compute_rf(
            down_blocks=rf_info["down_blocks"],
            convs_per_block=rf_info["convs_per_block"],
            kernel_size=rf_info["kernel_size"],
        )

    return rf_info


class ReceptiveFieldWidget(QtWidgets.QWidget):
    """
    Widget for previewing receptive field on sample image, with caption.

    Args:
        head_name: If given, then used in caption to show which model the
            preview is for.

    Usage:
        Create, then call `setImage` and `setModelConfig` methods.
    """

    def __init__(self, head_name: Text = "", *args, **kwargs):
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

    def _get_info_text(
        self, size, scale, max_stride, down_blocks, convs_per_block, kernel_size
    ) -> Text:
        """Returns text explaining how receptive field size is determined."""
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
        """Updates receptive field preview from model config."""
        rf_info = receptive_field_info_from_model_cfg(model_cfg)

        self._info_widget.setText(
            self._get_info_text(
                size=rf_info["size"],
                scale=scale,
                max_stride=rf_info["max_stride"],
                down_blocks=rf_info["down_blocks"],
                convs_per_block=rf_info["convs_per_block"],
                kernel_size=rf_info["kernel_size"],
            )
        )

        self._field_image_widget._set_field_size(rf_info["size"] or 0, scale)

    def setImage(self, *args, **kwargs):
        """Sets image on which receptive field box will be drawn."""
        self._field_image_widget.setImage(*args, **kwargs)


class ReceptiveFieldImageWidget(GraphicsView):
    """Widget for showing image with receptive field."""

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
        """
        Re-draw receptive field when needed by overriding QGraphicsView method.
        """
        # Update the position and visible size of field
        if isinstance(event, QtGui.QPaintEvent):
            self._set_field_size()

        # Now draw the viewport
        return super(ReceptiveFieldImageWidget, self).viewportEvent(event)

    def _set_field_size(self, size: Optional[int] = None, scale: float = 1.0):
        """Draws receptive field preview rect, updating size if needed."""
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
    win._set_field_size(50)

    win.show()
    app.exec_()


if __name__ == "__main__":
    demo_receptive_field()

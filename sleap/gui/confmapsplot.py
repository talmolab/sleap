"""
Module for showing confidence maps as an overlay within a QtVideoPlayer.

Example:
    >>> cm = ConfMapsPlot(conf_data.get_frame(0))
    >>> window.view.scene.addItem(cm)
"""

from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
from PySide2.QtWidgets import QGraphicsPixmapItem
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QRectF

import numpy as np
import qimage2ndarray

from sleap.io.video import Video, HDF5Video
from sleap.gui.multicheck import MultiCheckWidget

class ConfMapsPlot(QGraphicsObject):
    """QGraphicsObject to display multiple confidence maps in a QGraphicsView.

    Args:
        frame (numpy.array): Data for one frame of confidence map data.
            Shape of array should be (channels, height, width).
        show (list, optional): List of channels to show. If None, show all channels.

    Returns:
        None.

    When initialized, creates one child ConfMapPlot item for each channel.
    """

    def __init__(self, frame: np.array = None, show=None, *args, **kwargs):
        super(ConfMapsPlot, self).__init__(*args, **kwargs)
        self.frame = frame
        self.conf_maps = []
        self.color_maps = [
            [204, 81, 81],
            [127, 51, 51],
            [81, 204, 204],
            [51, 127, 127],
            [142, 204, 81],
            [89, 127, 51],
            [142, 81, 204],
            [89, 51, 127],
            [204, 173, 81],
            [127, 108, 51],
            [81, 204, 112],
            [51, 127, 70],
            [81, 112, 204],
            [51, 70, 127],
            [204, 81, 173],
            [127, 51, 108],
            [204, 127, 81],
            [127, 79, 51],
            [188, 204, 81],
            [117, 127, 51],
            [96, 204, 81],
            [60, 127, 51],
            [81, 204, 158],
            [51, 127, 98],
            [81, 158, 204],
            [51, 98, 127],
            [96, 81, 204],
            [60, 51, 127],
            [188, 81, 204],
            [117, 51, 127],
            [204, 81, 127],
            [127, 51, 79],
            [204, 104, 81],
            [127, 65, 51],
            [204, 150, 81],
            [127, 94, 51],
            [204, 196, 81],
            [127, 122, 51],
            [165, 204, 81],
            [103, 127, 51],
            [119, 204, 81],
            [74, 127, 51],
            [81, 204, 89],
            [51, 127, 55],
            [81, 204, 135],
            [51, 127, 84],
            [81, 204, 181],
            [51, 127, 113],
            [81, 181, 204],
            [51, 113, 127]
            ]
        for channel in range(self.frame.shape[2]):
            if show is None or channel in show:
                color_map = self.color_maps[channel % len(self.color_maps)]
                conf_map_item = ConfMapPlot(
                    confmap=self.frame[..., channel],
                    color=color_map,
                    parent=self)
                self.conf_maps.append(conf_map_item)

    def boundingRect(self) -> QRectF:
        """Method required by Qt.
        """
        return QRectF()

    def paint(self, painter, option, widget=None):
        """Method required by Qt.
        """
        pass

class ConfMapPlot(QGraphicsPixmapItem):
    """QGraphicsPixmapItem object for drawing single channel of confidence map.

    Args:
        confmap (numpy.array): (h, w) array of one confidence map channel.
        color (list): optional (r, g, b) array for channel color.

    Returns:
        None.

    Note:
        In most cases this should only be called by ConfMapsPlot.
    """

    def __init__(self, confmap: np.array = None, color=[255, 255, 255], *args, **kwargs):
        super(ConfMapPlot, self).__init__(*args, **kwargs)

        self.color_map = color

        if confmap is not None:
            self.confmap = confmap
            image = self.get_conf_image()
            self.setPixmap(QPixmap(image))

    def get_conf_image(self) -> QImage:
        """Converts array data stored in object to QImage.

        Returns:
            QImage.
        """
        if self.confmap is None:
            return

        # Get image data
        frame = self.confmap

        # Colorize single-channel overlap
        frame_a = (frame * 255).astype(np.uint8)
        frame_r = (frame * self.color_map[0]).astype(np.uint8)
        frame_g = (frame * self.color_map[1]).astype(np.uint8)
        frame_b = (frame * self.color_map[2]).astype(np.uint8)

        frame_composite = np.dstack((frame_r, frame_g, frame_b, frame_a))

        # Convert ndarray to QImage
        image = qimage2ndarray.array2qimage(frame_composite)

        return image

if __name__ == "__main__":

    from video import QtVideoPlayer

    #data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    data_path = "training.scale=1.00,sigma=5.h5"
    vid = HDF5Video(data_path, "/box", input_format="channels_first")
    conf_data = HDF5Video(data_path, "/confmaps", input_format="channels_first")

    app = QApplication([])
    window = QtVideoPlayer(video=vid)

    channel_box = MultiCheckWidget(
        count=conf_data.get_frame(0).shape[-1],
        title="Confidence Map Channel",
        default=True
        )
    channel_box.selectionChanged.connect(window.plot)
    window.layout.addWidget(channel_box)

    def plot_confmaps(parent, item_idx):
        selected = channel_box.getSelected()
        conf_maps = ConfMapsPlot(conf_data.get_frame(parent.frame_idx), selected)
        window.view.scene.addItem(conf_maps)

    window.callbacks.append(plot_confmaps)

    window.show()
    window.plot()

    app.exec_()

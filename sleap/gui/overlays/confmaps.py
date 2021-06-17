"""
Overlay for confidence maps.

Currently a `DataOverlay` gets data from a model (i.e., it runs inference on the
current frame) and then uses a `ConfMapsPlot` object to show the resulting
confidence maps.

Example: ::

    >>> cm = ConfMapsPlot(conf_data.get_frame(0))
    >>> window.view.scene.addItem(cm)

"""

from PySide2 import QtWidgets, QtCore, QtGui

import numpy as np
import qimage2ndarray

from sleap.gui.overlays.base import DataOverlay, h5_colors


class ConfMapsPlot(QtWidgets.QGraphicsObject):
    """QGraphicsObject to display multiple confidence maps in a QGraphicsView.

    Args:
        frame (numpy.array): Data for one frame of confidence map data.
            Shape of array should be (channels, height, width).
        show (list, optional): List of channels to show. If None, show all channels.
        show_box (bool, optional): Draw bounding box around confidence maps.

    Returns:
        None.

    When initialized, creates one child ConfMapPlot item for each channel.
    """

    def __init__(
        self, frame: np.array = None, show=None, show_box=True, *args, **kwargs
    ):
        super(ConfMapsPlot, self).__init__(*args, **kwargs)
        self.frame = frame
        self.show_box = show_box

        self.rect = QtCore.QRectF(0, 0, self.frame.shape[1], self.frame.shape[0])

        if self.show_box:
            QtWidgets.QGraphicsRectItem(self.rect, parent=self).setPen(
                QtGui.QPen("yellow")
            )

        for channel in range(self.frame.shape[2]):
            if show is None or channel in show:
                color_map = h5_colors[channel % len(h5_colors)]

                # Add QGraphicsPixmapItem as child object
                ConfMapPlot(
                    confmap=self.frame[..., channel], color=color_map, parent=self
                )

    def boundingRect(self) -> QtCore.QRectF:
        """Method required by Qt."""
        return self.rect

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
        pass


class ConfMapPlot(QtWidgets.QGraphicsPixmapItem):
    """QGraphicsPixmapItem object for drawing single channel of confidence map.

    Args:
        confmap (numpy.array): (h, w) array of one confidence map channel.
        color (list): optional (r, g, b) array for channel color.

    Returns:
        None.

    Note:
        In most cases this should only be called by ConfMapsPlot.
    """

    def __init__(
        self, confmap: np.array = None, color=[255, 255, 255], *args, **kwargs
    ):
        super(ConfMapPlot, self).__init__(*args, **kwargs)

        self.color_map = color

        if confmap is not None:
            self.confmap = confmap
            image = self.get_conf_image()
            self.setPixmap(QtGui.QPixmap(image))

    def get_conf_image(self) -> QtGui.QImage:
        """Converts array data stored in object to QImage.

        Returns:
            QImage.
        """
        if self.confmap is None:
            return

        # Get image data
        frame = self.confmap

        # Colorize single-channel overlap
        if np.ptp(frame) <= 1.0:
            frame_a = (frame * 255).astype(np.uint8)
            frame_r = (frame * self.color_map[0]).astype(np.uint8)
            frame_g = (frame * self.color_map[1]).astype(np.uint8)
            frame_b = (frame * self.color_map[2]).astype(np.uint8)
        else:
            frame_a = (frame).astype(np.uint8)
            frame_r = (frame * (self.color_map[0] / 255.0)).astype(np.uint8)
            frame_g = (frame * (self.color_map[1] / 255.0)).astype(np.uint8)
            frame_b = (frame * (self.color_map[2] / 255.0)).astype(np.uint8)

        frame_composite = np.dstack((frame_r, frame_g, frame_b, frame_a))

        # Convert ndarray to QImage
        image = qimage2ndarray.array2qimage(frame_composite)

        return image


def show_confmaps_from_h5(filename, input_format="channels_last", standalone=False):
    """Demo function."""
    from sleap.io.video import HDF5Video

    video = HDF5Video(filename, "/box", input_format=input_format)
    conf_data = HDF5Video(
        filename, "/confmaps", input_format=input_format, convert_range=False
    )

    confmaps_ = [np.clip(conf_data.get_frame(i), 0, 1) for i in range(conf_data.frames)]
    confmaps = np.stack(confmaps_)

    return demo_confmaps(confmaps=confmaps, video=video, standalone=standalone)


def demo_confmaps(confmaps, video, scale=None, standalone=False, callback=None):
    """Demo function."""
    from PySide2 import QtWidgets
    from sleap.gui.widgets.video import QtVideoPlayer

    if standalone:
        app = QtWidgets.QApplication([])

    win = QtVideoPlayer(video=video)
    win.setWindowTitle("confmaps")
    win.show()

    def plot_confmaps(parent, frame_idx):
        if frame_idx < confmaps.shape[0]:
            frame_conf_map = ConfMapsPlot(confmaps[frame_idx, ...], show_box=not scale)
            if scale:
                frame_conf_map.setScale(scale)
            win.view.scene.addItem(frame_conf_map)

    win.changedPlot.connect(plot_confmaps)
    if callback:
        win.changedPlot.connect(callback)
    win.plot()

    if standalone:
        app.exec_()

    return win


if __name__ == "__main__":

    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    show_confmaps_from_h5(data_path, input_format="channels_first", standalone=True)

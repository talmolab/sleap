"""
Overlay for part affinity fields.

Currently a `DataOverlay` gets data from a model (i.e., it runs inference on the
current frame) and then uses a `MultiQuiverPlot` object to show the resulting
part affinity fields.
"""

from PySide2 import QtWidgets, QtGui, QtCore

import numpy as np
import itertools
import math

from sleap.io.video import HDF5Video

from sleap.gui.overlays.base import DataOverlay, h5_colors


class PafOverlay(DataOverlay):
    """Class show pafs saved in HDF5 (not currently used)."""

    @classmethod
    def from_h5(cls, filename, input_format="channels_last", **kwargs):
        return DataOverlay.from_h5(
            filename, "/pafs", input_format, overlay_class=MultiQuiverPlot, **kwargs
        )


class MultiQuiverPlot(QtWidgets.QGraphicsObject):
    """QtWidgets.QGraphicsObject to display multiple quiver plots in a QtWidgets.QGraphicsView.

    Args:
        frame (numpy.array): Data for one frame of quiver plot data.
            Shape of array should be (channels, height, width).
        show (list, optional): List of channels to show. If None, show all channels.
        decimation (int, optional): Decimation factor. If 1, show every arrow.

    Returns:
        None.

    Note:
        Each channel corresponds to two (h, w) arrays: x and y for the vector.

    When initialized, creates one child QuiverPlot item for each channel.
    """

    def __init__(
        self,
        frame: np.array = None,
        show: list = None,
        decimation: int = 1,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super(MultiQuiverPlot, self).__init__(*args, **kwargs)
        self.frame = frame
        self.affinity_field = []
        self.decimation = decimation
        self.scale = scale

        # if data range is outside [-1, 1], assume it's [-255, 255] and scale
        if np.ptp(self.frame) > 4:
            self.frame = self.frame.astype(np.float64) / 255

        if show is None:
            self.show_list = range(self.frame.shape[2] // 2)
        else:
            self.show_list = show
        for channel in self.show_list:
            if channel < self.frame.shape[-1] // 2:
                color_map = h5_colors[channel % len(h5_colors)]
                aff_field_item = QuiverPlot(
                    field_x=self.frame[..., channel * 2],
                    field_y=self.frame[..., channel * 2 + 1],
                    color=color_map,
                    decimation=self.decimation,
                    scale=self.scale,
                    parent=self,
                )
                self.affinity_field.append(aff_field_item)

    def boundingRect(self) -> QtCore.QRectF:
        """Method required by Qt."""
        return QtCore.QRectF()

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
        pass


class QuiverPlot(QtWidgets.QGraphicsObject):
    """QtWidgets.QGraphicsObject for drawing single quiver plot.

    Args:
        field_x (numpy.array): (h, w) array of x component of vectors.
        field_y (numpy.array): (h, w) array of y component of vectors.
        color (list, optional): Arrow color. Format as (r, g, b) array.
        decimation (int, optional): Decimation factor. If 1, show every arrow.

    Returns:
        None.
    """

    def __init__(
        self,
        field_x: np.array = None,
        field_y: np.array = None,
        color=[255, 255, 255],
        decimation=1,
        scale=1,
        *args,
        **kwargs,
    ):
        super(QuiverPlot, self).__init__(*args, **kwargs)

        self.field_x, self.field_y = None, None
        self.color = color
        self.decimation = decimation
        self.scale = scale
        pen_width = min(4, max(0.1, math.log(self.decimation, 20)))
        self.pen = QtGui.QPen(QtGui.QColor(*self.color), pen_width)
        self.points = []
        self.rect = QtCore.QRectF()

        if field_x is not None and field_y is not None:
            self.field_x, self.field_y = field_x, field_y

            h, w = self.field_x.shape
            h, w = int(h * self.scale), int(w * self.scale)

            self.rect = QtCore.QRectF(0, 0, w, h)

            self._add_arrows()

    def _add_arrows(self, min_length=0.01):
        points = []
        if self.field_x is not None and self.field_y is not None:

            raw_delta_yx = np.stack((self.field_y, self.field_x), axis=-1)

            dim_0 = self.field_x.shape[0] // self.decimation * self.decimation
            dim_1 = self.field_x.shape[1] // self.decimation * self.decimation

            grid = np.mgrid[0 : dim_0 : self.decimation, 0 : dim_1 : self.decimation]
            loc_yx = np.moveaxis(grid, 0, -1)

            # Adjust by scaling factor
            loc_yx = loc_yx * self.scale

            if self.decimation > 1:
                delta_yx = self._decimate(raw_delta_yx, self.decimation)

                # Shift locations to midpoint of decimation square
                loc_yx += self.decimation // 2
            else:
                delta_yx = raw_delta_yx

            delta_yx = delta_yx * self.scale

            # Split into x,y matrices
            loc_y, loc_x = loc_yx[..., 0], loc_yx[..., 1]
            delta_y, delta_x = delta_yx[..., 0], delta_yx[..., 1]

            # Determine vector endpoint
            x2 = delta_x * self.decimation + loc_x
            y2 = delta_y * self.decimation + loc_y
            line_length = (delta_x ** 2 + delta_y ** 2) ** 0.5

            # Determine points for arrow
            arrow_head_size = line_length / 4

            u_dx = np.divide(
                delta_x, line_length, out=np.zeros_like(delta_x), where=line_length != 0
            )
            u_dy = np.divide(
                delta_y, line_length, out=np.zeros_like(delta_y), where=line_length != 0
            )
            p1_x = x2 - u_dx * arrow_head_size - u_dy * arrow_head_size
            p1_y = y2 - u_dy * arrow_head_size + u_dx * arrow_head_size

            p2_x = x2 - u_dx * arrow_head_size + u_dy * arrow_head_size
            p2_y = y2 - u_dy * arrow_head_size - u_dx * arrow_head_size

            # Build list of QPointF objects for faster drawing
            y_x_pairs = itertools.product(
                range(delta_yx.shape[0]), range(delta_yx.shape[1])
            )
            for y, x in y_x_pairs:
                x1, y1 = loc_x[y, x], loc_y[y, x]

                if line_length[y, x] > min_length:
                    points.append((x1, y1))
                    points.append((x2[y, x], y2[y, x]))
                    points.append((p1_x[y, x], p1_y[y, x]))
                    points.append((x2[y, x], y2[y, x]))
                    points.append((p2_x[y, x], p2_y[y, x]))
                    points.append((x2[y, x], y2[y, x]))
            self.points = list(itertools.starmap(QtCore.QPointF, points))

    def _decimate(self, image: np.array, box: int):
        height = width = box
        # Source: https://stackoverflow.com/questions/48482317/slice-an-image-into-tiles-using-numpy
        _nrows, _ncols, depth = image.shape
        _size = image.size
        _strides = image.strides

        nrows, _m = divmod(_nrows, height)
        ncols, _n = divmod(_ncols, width)
        if _m != 0 or _n != 0:
            # if we can't tile whole image, forget about bottom/right edges
            image = image[: (nrows + 1) * box, : (ncols + 1) * box]

        tiles = np.lib.stride_tricks.as_strided(
            np.ravel(image),
            shape=(nrows, ncols, height, width, depth),
            strides=(height * _strides[0], width * _strides[1], *_strides),
            writeable=False,
        )

        # Since strides accesses the ndarray by memory, we need to swap axes if
        # the array is stored column-major (Fortran), which it is from h5py.
        if _strides[0] < _strides[1]:
            tiles = np.swapaxes(tiles, 0, 1)

        return np.mean(tiles, axis=(2, 3))

    def boundingRect(self) -> QtCore.QRectF:
        """Method called by Qt in order to determine whether object is in visible frame."""
        return QtCore.QRectF(self.rect)

    def paint(self, painter, option, widget=None):
        """Method called by Qt to draw object."""
        if self.pen is not None:
            painter.setPen(self.pen)
        painter.drawLines(self.points)
        pass


def show_pafs_from_h5(filename, input_format="channels_last", standalone=False):
    video = HDF5Video(filename, "/box", input_format=input_format)
    paf_data = HDF5Video(
        filename, "/pafs", input_format=input_format, convert_range=False
    )

    pafs_ = [paf_data.get_frame(i) for i in range(paf_data.frames)]
    pafs = np.stack(pafs_)

    return demo_pafs(pafs, video, standalone=standalone)


def demo_pafs(pafs, video, decimation=4, scale=None, standalone=False):
    from sleap.gui.widgets.video import QtVideoPlayer

    if standalone:
        app = QtWidgets.QApplication([])

    win = QtVideoPlayer(video=video)
    win.setWindowTitle("pafs")

    decimation_size_bar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    decimation_size_bar.valueChanged.connect(lambda e: win.plot())
    decimation_size_bar.setValue(decimation)
    decimation_size_bar.setMinimum(1)
    decimation_size_bar.setMaximum(10)
    decimation_size_bar.setEnabled(True)
    win.layout.addWidget(decimation_size_bar)

    win.show()

    def plot_fields(parent, frame_idx):
        if frame_idx < pafs.shape[0]:
            frame_pafs = pafs[frame_idx, ...]
            decimation = decimation_size_bar.value()
            aff_fields_item = MultiQuiverPlot(
                frame_pafs, show=None, decimation=decimation
            )
            if scale:
                aff_fields_item.setScale(scale)
            win.view.scene.addItem(aff_fields_item)

    win.changedPlot.connect(plot_fields)
    win.plot()

    if standalone:
        app.exec_()

    return win


if __name__ == "__main__":

    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    input_format = "channels_first"

    show_pafs_from_h5(data_path, input_format=input_format, standalone=True)

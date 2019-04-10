from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
from PySide2.QtWidgets import QGraphicsPixmapItem, QGraphicsLineItem
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtGui import QPen, QBrush, QColor
from PySide2.QtCore import QRectF

from PySide2.QtWidgets import QGridLayout, QGroupBox, QButtonGroup, QCheckBox

import numpy as np
import itertools
import math

from sleap.io.video import Video, HDF5Video
from sleap.gui.multicheck import MultiCheckWidget

class MultiQuiverPlot(QGraphicsObject):
    """QGraphicsObject to display multiple quiver plots in a QGraphicsView.

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

    def __init__(self, frame: np.array = None, show: list = None, decimation: int = 9, *args, **kwargs):
        super(MultiQuiverPlot, self).__init__(*args, **kwargs)
        self.frame = frame
        self.affinity_field = []
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
        self.decimation = decimation
        if show is None:
            self.show_list = range(self.frame.shape[2]//2)
        else:
            self.show_list = show
        for channel in self.show_list:
            if channel < self.frame.shape[-1]//2:
                color_map = self.color_maps[channel % len(self.color_maps)]
                aff_field_item = QuiverPlot(
                    field_x=self.frame[..., channel*2],
                    field_y=self.frame[..., channel*2+1],
                    color=color_map,
                    decimation=self.decimation,
                    parent=self
                    )
                self.affinity_field.append(aff_field_item)

    def boundingRect(self) -> QRectF:
        """Method required by Qt.
        """
        return QRectF()

    def paint(self, painter, option, widget=None):
        """Method required by Qt.
        """
        pass

class QuiverPlot(QGraphicsObject):
    """QGraphicsPixmapItem for drawing single quiver plot.

    Args:
        field_x (numpy.array): (h, w) array of x component of vectors.
        field_y (numpy.array): (h, w) array of y component of vectors.
        color (list, optional): Arrow color. Format as (r, g, b) array.
        decimation (int, optional): Decimation factor. If 1, show every arrow.

    Returns:
        None.
    """

    def __init__(self, field_x: np.array = None, field_y: np.array = None, color=[255, 255, 255], decimation=1, *args, **kwargs):
        super(QuiverPlot, self).__init__(*args, **kwargs)

        self.field_x, self.field_y = None, None
        self.color = color
        self.decimation = decimation
        pen_width = min(4, max(.1, math.log(self.decimation, 20)))
        self.pen = QPen(QColor(*self.color), pen_width)

        if field_x is not None and field_y is not None:
            self.field_x, self.field_y = field_x, field_y

            self._add_arrows()

    def _add_arrows(self):
        if self.field_x is not None and self.field_y is not None:
            y_x_pairs = itertools.product(
                range(self.field_x.shape[0]),
                range(self.field_y.shape[1])
                )
            for y, x in y_x_pairs:
                # we'll only draw one arrow per decimation box
                if x%self.decimation == 0 and y%self.decimation == 0:
                    # sum all deltas for the vectors in box
                    x_delta = self.field_x[y:y+self.decimation][:, x:x+self.decimation].sum() / self.decimation**2 * self.decimation * .9
                    y_delta = self.field_y[y:y+self.decimation][:, x:x+self.decimation].sum() / self.decimation**2 * self.decimation * .9
                    #x_delta = self.field_x[y, x] * 10
                    #y_delta = self.field_y[y, x] * 10
                    if x_delta != 0 or y_delta != 0:
                        line = QuiverArrow(
                                x+self.decimation//2, y+self.decimation//2,
                                x+self.decimation//2+x_delta, y+self.decimation//2+y_delta,
                                self.pen, parent=self)

    def boundingRect(self) -> QRectF:
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass

class QuiverArrow(QGraphicsItem):

    def __init__(self, x, y, x2, y2, pen=None, *args, **kwargs):
        super(QuiverArrow, self).__init__(*args, **kwargs)

        arrow_points = self._get_arrow_head_points((x, y), (x2, y2))
        shaft_line = QGraphicsLineItem(x, y, x2, y2, *args, **kwargs)
        head_line_1 = QGraphicsLineItem(x2, y2, *arrow_points[0], *args, **kwargs)
        head_line_2 = QGraphicsLineItem(x2, y2, *arrow_points[1], *args, **kwargs)

        if pen is not None:
            self.pen = pen
            shaft_line.setPen(self.pen)
            head_line_1.setPen(self.pen)
            head_line_2.setPen(self.pen)

    def _get_arrow_head_points(self, from_point, to_point):
        x1, y1 = from_point
        x2, y2 = to_point

        dx, dy = x2-x1, y2-y1
        line_length = (dx**2 + dy**2)**.4
        arrow_head_size = line_length / 4
        u_dx, u_dy = dx/line_length, dy/line_length

        p1_x = x2 - u_dx*arrow_head_size - u_dy*arrow_head_size
        p1_y = y2 - u_dy*arrow_head_size + u_dx*arrow_head_size

        p2_x = x2 - u_dx*arrow_head_size + u_dy*arrow_head_size
        p2_y = y2 - u_dy*arrow_head_size - u_dx*arrow_head_size

        return (p1_x, p1_y), (p2_x, p2_y)

    def boundingRect(self) -> QRectF:
        """Method required by Qt.
        """
        return QRectF()

    def paint(self, painter, option, widget=None):
        """Method required by Qt.
        """
        pass


if __name__ == "__main__":

    from video import *

    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    #data_path = "training.scale=1.00,sigma=5.h5"
    vid = HDF5Video(data_path, "/box", input_format="channels_first")
    overlay_data = HDF5Video(data_path, "/pafs", input_format="channels_first")

    app = QApplication([])
    window = QtVideoPlayer(video=vid)

    field_count = overlay_data.get_frame(1).shape[-1]//2 - 1
    # show the first, middle, and last fields
    show_fields = [0, field_count//2, field_count]

    field_check_groupbox = MultiCheckWidget(
        count=field_count,
        selected=show_fields,
        title="Affinity Field Channel"
        )
    field_check_groupbox.selectionChanged.connect(window.plot)
    window.layout.addWidget(field_check_groupbox)

    # show one arrow for each decimation*decimation box
    default_decimation = 9

    decimation_size_bar = QSlider(Qt.Horizontal)
    decimation_size_bar.valueChanged.connect(lambda evt: window.plot())
    decimation_size_bar.setValue(default_decimation)
    decimation_size_bar.setMinimum(1)
    decimation_size_bar.setMaximum(21)
    decimation_size_bar.setEnabled(True)
    window.layout.addWidget(decimation_size_bar)

    def plot_fields(parent,i):
        # build list of checked boxes to determine which affinity fields to show
        selected = field_check_groupbox.getSelected()
        # get decimation size from slider
        decimation = decimation_size_bar.value()
        # show affinity fields
        aff_fields_item = MultiQuiverPlot(overlay_data.get_frame(parent.frame_idx), selected, decimation)

        window.view.scene.addItem(aff_fields_item)

    window.callbacks.append(plot_fields)

    window.show()
    window.plot()

    app.exec_()
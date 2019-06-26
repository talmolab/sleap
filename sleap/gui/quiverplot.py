from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
from PySide2.QtWidgets import QGraphicsPixmapItem, QGraphicsLineItem
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtGui import QPen, QBrush, QColor, QPaintDevice
from PySide2.QtCore import QRectF, QPointF

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
        
        # if data range is outside [-1, 1], assume it's [-255, 255] and scale
        if np.ptp(self.frame) > 2:
            self.frame = self.frame.astype(np.float64)/255
            
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
    """QGraphicsObject for drawing single quiver plot.

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
        self.points = []
        self.rect = QRectF()

        if field_x is not None and field_y is not None:
            self.field_x, self.field_y = field_x, field_y
            self.rect = QRectF(0,0,*self.field_x.shape)

            self._add_arrows()

    def _add_arrows(self, min_length=0.01):
        points = []
        if self.field_x is not None and self.field_y is not None:
            
            dim_0 = self.field_x.shape[0]//self.decimation*self.decimation
            dim_1 = self.field_x.shape[1]//self.decimation*self.decimation
            raw_delta_yx = np.stack((self.field_y,self.field_x),axis=-1)
            grid = np.mgrid[0:dim_0:self.decimation, 0:dim_1:self.decimation]
            loc_yx = np.moveaxis(grid,0,-1)
            if self.decimation > 1:
                delta_yx = self._decimate(raw_delta_yx, self.decimation)
                loc_yx += self.decimation//2
            else:
                delta_yx = raw_delta_yx
            loc_y, loc_x = loc_yx[...,0], loc_yx[...,1]
            delta_y, delta_x = delta_yx[...,0], delta_yx[...,1]
            x2 = delta_x*self.decimation + loc_x
            y2 = delta_y*self.decimation + loc_y
            line_length = (delta_x**2 + delta_y**2)**.5

            arrow_head_size = line_length / 4

            u_dx = np.divide(delta_x, line_length, out=np.zeros_like(delta_x), where=line_length!=0)
            u_dy = np.divide(delta_y, line_length, out=np.zeros_like(delta_y), where=line_length!=0)
            p1_x = x2 - u_dx*arrow_head_size - u_dy*arrow_head_size
            p1_y = y2 - u_dy*arrow_head_size + u_dx*arrow_head_size

            p2_x = x2 - u_dx*arrow_head_size + u_dy*arrow_head_size
            p2_y = y2 - u_dy*arrow_head_size - u_dx*arrow_head_size

            y_x_pairs = itertools.product(
                range(delta_yx.shape[0]),
                range(delta_yx.shape[1])
                )
            for y, x in y_x_pairs:
                x1, y1 = loc_x[y,x], loc_y[y,x]

                if line_length[y,x] > min_length:
                    points.append((x1, y1))
                    points.append((x2[y,x],y2[y,x]))
                    points.append((p1_x[y,x],p1_y[y,x]))
                    points.append((x2[y,x],y2[y,x]))
                    points.append((p2_x[y,x],p2_y[y,x]))
                    points.append((x2[y,x],y2[y,x]))
            self.points = list(itertools.starmap(QPointF,points))

    def _decimate(self, image:np.array, box:int):
        height = width = box
        # Source: https://stackoverflow.com/questions/48482317/slice-an-image-into-tiles-using-numpy
        _nrows, _ncols, depth = image.shape
        _size = image.size
        _strides = image.strides

        nrows, _m = divmod(_nrows, height)
        ncols, _n = divmod(_ncols, width)
        if _m != 0 or _n != 0:
            # if we can't tile whole image, forget about bottom/right edges
            image = image[:(nrows+1)*box,:(ncols+1)*box]

        tiles =  np.lib.stride_tricks.as_strided(
            np.ravel(image),
            shape=(nrows, ncols, height, width, depth),
            strides=(height * _strides[0], width * _strides[1], *_strides),
            writeable=False
        )
        if False: tiles = np.swapaxes(tiles,0,1) # why do we need to swap axes sometimes?
        return np.mean(tiles, axis=(2,3))

    def boundingRect(self) -> QRectF:
        """Method called by Qt in order to determine whether object is in visible frame."""
        return QRectF(self.rect)

    def paint(self, painter, option, widget=None):
        """Method called by Qt to draw object."""
        if self.pen is not None:
            painter.setPen(self.pen)
        painter.drawLines(self.points)
        pass

def show_pafs_from_h5(filename, input_format="channels_last", standalone=False):
    video = HDF5Video(data_path, "/box", input_format=input_format)
    paf_data = HDF5Video(data_path, "/pafs", input_format=input_format, convert_range=False)

    pafs_ = [paf_data.get_frame(i) for i in range(paf_data.frames)]
    pafs = np.stack(pafs_)

    demo_pafs(pafs, video, standalone=True)

def demo_pafs(pafs, video, decimation=1, standalone=False):
    from sleap.gui.video import QtVideoPlayer

    if standalone: app = QApplication([])

    win = QtVideoPlayer(video=video)
    win.setWindowTitle("pafs")
    win.show()

    def plot_fields(parent, i):
        if parent.frame_idx < pafs.shape[0]:
            frame_pafs = pafs[parent.frame_idx, ...]
            aff_fields_item = MultiQuiverPlot(frame_pafs, show=None, decimation=decimation)
            win.view.scene.addItem(aff_fields_item)

    win.changedPlot.connect(plot_fields)
    win.plot()

    if standalone: app.exec_()

if __name__ == "__main__":

    from video import *

    #data_path = "training.scale=1.00,sigma=5.h5"

    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    input_format="channels_first"
    
    data_path = "/Volumes/fileset-mmurthy/nat/nyu-mouse/predict.h5"
    input_format = "channels_last"

    show_pafs_from_h5(data_path, input_format=input_format, standalone=True)

def foo():

    vid = HDF5Video(data_path, "/box", input_format=input_format)
    overlay_data = HDF5Video(data_path, "/pafs", input_format=input_format, convert_range=False)
    print(f"{overlay_data.frames}, {overlay_data.height}, {overlay_data.width}, {overlay_data.channels}")
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
        frame_data = overlay_data.get_frame(parent.frame_idx)
        aff_fields_item = MultiQuiverPlot(frame_data, selected, decimation)

        window.view.scene.addItem(aff_fields_item)

    window.changedPlot.connect(plot_fields)

    window.show()
    window.plot()

    app.exec_()
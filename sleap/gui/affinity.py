from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
from PySide2.QtWidgets import QGraphicsPixmapItem
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtGui import QPen, QBrush, QColor
from PySide2.QtCore import QRectF

import numpy as np
import itertools

from sleap.io.video import Video, HDF5Video

class QtAffinityFields(QGraphicsObject):
    """ Type of QGraphicsObject to display affinity fields in a QGraphicsView.
    
    Initialize with an array of the affinity field data for one frame:
        QGraphicsObject(frame)
    """
    
    def __init__(self, frame: np.array = None, *args, **kwargs):
        """ Initializes the QGraphics Object with a affinity field frame.
    
        This creates a child QtAffinityField item for each channel, so that these will
        all be added to the view along with the parent QtAffinityFields.
    
        Args:
            frame (numpy.array): Formats is (channels * 2, height, width).

        Note:
            Each channel corresponds to two (h,w) arrays: x and y for the vector.
        """
        super(QtAffinityFields, self).__init__(*args, **kwargs)
        self.frame = frame
        self.affinity_field = []
        self.color_maps = [
            [204,81,81],
            [127,51,51],
            [81,204,204],
            [51,127,127],
            [142,204,81],
            [89,127,51],
            [142,81,204],
            [89,51,127],
            [204,173,81],
            [127,108,51],
            [81,204,112],
            [51,127,70],
            [81,112,204],
            [51,70,127],
            [204,81,173],
            [127,51,108],
            [204,127,81],
            [127,79,51],
            [188,204,81],
            [117,127,51],
            [96,204,81],
            [60,127,51],
            [81,204,158],
            [51,127,98],
            [81,158,204],
            [51,98,127],
            [96,81,204],
            [60,51,127],
            [188,81,204],
            [117,51,127],
            [204,81,127],
            [127,51,79],
            [204,104,81],
            [127,65,51],
            [204,150,81],
            [127,94,51],
            [204,196,81],
            [127,122,51],
            [165,204,81],
            [103,127,51],
            [119,204,81],
            [74,127,51],
            [81,204,89],
            [51,127,55],
            [81,204,135],
            [51,127,84],
            [81,204,181],
            [51,127,113],
            [81,181,204],
            [51,113,127]
            ]

        for channel in range(self.frame.shape[2]//2):
            color_map = self.color_maps[channel % len(self.color_maps)]
            aff_field_item = QtAffinityField(
                                field_x=self.frame[...,channel], 
                                field_y=self.frame[...,channel+1], 
                                color=color_map, 
                                parent=self
                                )
            self.affinity_field.append(aff_field_item)
        
    def boundingRect(self) -> QRectF:
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass


class QtAffinityField(QGraphicsObject):
    """ Type of QGraphicsPixmapItem for drawing single field of an affinity field.
    
    Usage:
        Call QtAffinityField(parent=self, field_x, field_y, [color]) from an object
        which can contain child pixmaps items (i.e., a QGraphicsObject).
        
    Args:
        field_x (numpy.array): (h,w) array of x component of field vectors.
        field_y (numpy.array): (h,w) array of y component of field vectors.
        color (list): optional (r,g,b) array for channel color.
    """

    def __init__(self, field_x: np.array = None, field_y: np.array = None, color = [255, 255, 255], *args, **kwargs):
        super(QtAffinityField, self).__init__(*args, **kwargs)
        
        self.color = color
        self.pen = QPen(QColor(*self.color), 1)
        
        if field_x is not None and field_y is not None:
            self.field_x, self.field_y = field_x, field_y
            
            for y,x in itertools.product(range(self.field_x.shape[0]),range(self.field_y.shape[1])):
                if x%9 == 0 and y%9 == 0:
                    x_delta = self.field_x[y,x] * 10 # TO DO: decide how to scale
                    y_delta = self.field_y[y,x] * 10
                    if x_delta != 0 or y_delta != 0:
                        line = QAffinityArrow(x, y, x+x_delta, y+y_delta, self.pen, *args, **kwargs)
            
    def boundingRect(self) -> QRectF:
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass
        

class QAffinityArrow(QGraphicsItem):

    def __init__(self, x, y, x2, y2, pen = None, *args, **kwargs):
        super(QAffinityArrow, self).__init__(*args, **kwargs)
        
        self.line = QGraphicsLineItem(x, y, x2, y2, *args, **kwargs)
        
        arrow_points = self.get_arrow_head_points((x,y), (x2,y2))
        head_line_1 = QGraphicsLineItem(x2, y2, *arrow_points[0], *args, **kwargs)
        head_line_2 = QGraphicsLineItem(x2, y2, *arrow_points[1], *args, **kwargs)
        
        if pen is not None:
            self.pen = pen
            self.line.setPen(self.pen)
            head_line_1.setPen(self.pen)
            head_line_2.setPen(self.pen)
    
    def get_arrow_head_points(self, from_point, to_point):
        arrow_head_size = 3 # TO DO: how should we determine this?
        x1, y1 = from_point
        x2, y2 = to_point

        dx, dy = x2-x1, y2-y1
        line_length = (dx**2 + dy**2)**.5
        u_dx, u_dy = dx/line_length, dy/line_length

        p1_x = x2 - u_dx*arrow_head_size - u_dy*arrow_head_size
        p1_y = y2 - u_dy*arrow_head_size + u_dx*arrow_head_size

        p2_x = x2 - u_dx*arrow_head_size + u_dy*arrow_head_size
        p2_y = y2 - u_dy*arrow_head_size - u_dx*arrow_head_size
        
        return (p1_x, p1_y), (p2_x, p2_y)
    
    def boundingRect(self) -> QRectF:
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass

if __name__ == "__main__":

    from video import *

    data_path = "/Users/nat/tech/sleap/training.scale=1.00,sigma=5.h5"
    vid = HDF5Video(data_path, "/box", input_format="channels_first")
    overlay_data = HDF5Video(data_path, "/pafs", input_format="channels_first")
    
    app = QApplication([])
    window = QtVideoPlayer(video=vid)
    
    def plot_fields(parent,item_idx):
        aff_fields_item = QtAffinityFields(overlay_data.get_frame(parent.frame_idx))
        window.view.scene.addItem(aff_fields_item)
        
    window.callbacks.append(plot_fields)

    window.show()
    window.plot()

    app.exec_()


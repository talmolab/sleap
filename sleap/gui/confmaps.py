from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtWidgets import QGraphicsItem, QGraphicsObject
from PySide2.QtWidgets import QGraphicsPixmapItem
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QRectF

import numpy as np

from sleap.io.video import Video, HDF5Video

class QtConfMaps(QGraphicsObject):
    
    def __init__(self, frame: np.array = None, *args, **kwargs):
        super(QtConfMaps, self).__init__(*args, **kwargs)
        self.frame = frame
        self.conf_maps = []
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
#         self.color_maps = [
#             [248, 240, 124],
#             [248,  48,  60],
#             [168,  56,  44],
#             [242,  50, 138],
#             [30,   82, 182],
#             [4,   150, 100]
#             ]
        for channel in range(self.frame.shape[2]):
            color_map = self.color_maps[channel % len(self.color_maps)]
            conf_map_item = QtConfMap(confmap=self.frame[:,:,channel], color=color_map, parent=self)
            self.conf_maps.append(conf_map_item)
        
    def boundingRect(self):
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass


class QtConfMap(QGraphicsPixmapItem):

    def __init__(self, confmap: np.array = None, color = [255, 255, 255], *args, **kwargs):
        super(QtConfMap, self).__init__(*args, **kwargs)
        
        self.color_map = color
        
        if confmap is not None:
            self.confmap = confmap
            image = self.get_conf_image()
            self.setPixmap(QPixmap(image))
            
    def get_conf_image(self):
    
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

    from video import *

    data_path = "/Users/nat/tech/sleap/training.scale=1.00,sigma=5.h5"
    vid = HDF5Video(data_path, "/box", input_format="channels_first")
    conf_data = HDF5Video(data_path, "/confmaps", input_format="channels_first")
    
    app = QApplication([])
    window = QtVideoPlayer(video=vid)
    
    def plot_confmaps(parent,item_idx):
        conf_maps = QtConfMaps(conf_data.get_frame(parent.frame_idx))
        window.view.scene.addItem(conf_maps)
        
    window.callbacks.append(plot_confmaps)

    window.show()
    window.plot()

    app.exec_()


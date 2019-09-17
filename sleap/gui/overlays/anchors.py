import attr

from PySide2 import QtWidgets, QtGui

from sleap.gui.video import QtVideoPlayer
from sleap.io.dataset import Labels

@attr.s(auto_attribs=True)
class NegativeAnchorOverlay:

    labels: Labels=None
    scene: QtWidgets.QGraphicsScene=None
    pen = QtGui.QPen(QtGui.QColor("red"))
    line_len: int=3

    def add_to_scene(self, video, frame_idx):
        if self.labels is None: return
        if video not in self.labels.negative_anchors: return
        
        anchors = self.labels.negative_anchors[video]
        for idx, x, y in anchors:
            if frame_idx == idx:
                self._add(x,y)

    def _add(self, x, y):
        self.scene.addLine(x-self.line_len, y-self.line_len, x+self.line_len, y+self.line_len, self.pen)
        self.scene.addLine(x+self.line_len, y-self.line_len, x-self.line_len, y+self.line_len, self.pen)
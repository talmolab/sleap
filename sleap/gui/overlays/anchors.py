"""
Overlay for showing negative training sample anchors (currently unused).
"""
import attr

from PySide2 import QtGui

from sleap.gui.overlays.base import BaseOverlay
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.io.dataset import Labels


@attr.s(auto_attribs=True)
class NegativeAnchorOverlay(BaseOverlay):
    """Class to overlay of negative training sample anchors to video frame.

    Attributes:
        labels: The :class:`Labels` dataset from which to get overlay data.
        player: The video player in which to show overlay.
    """

    labels: Labels = None
    _pen = QtGui.QPen(QtGui.QColor("red"))
    _line_len: int = 3

    def add_to_scene(self, video, frame_idx):
        """Adds anchor markers as overlay on frame image."""
        if self.labels is None:
            return
        if video not in self.labels.negative_anchors:
            return

        anchors = self.labels.negative_anchors[video]
        for idx, x, y in anchors:
            if frame_idx == idx:
                self._add(x, y)

    def _add(self, x, y):
        self.player.scene.addLine(
            x - self._line_len,
            y - self._line_len,
            x + self._line_len,
            y + self._line_len,
            self._pen,
        )
        self.player.scene.addLine(
            x + self._line_len,
            y - self._line_len,
            x - self._line_len,
            y + self._line_len,
            self._pen,
        )

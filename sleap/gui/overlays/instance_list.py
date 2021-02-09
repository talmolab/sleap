"""Instance list overlay."""

from sleap.gui.overlays.base import BaseOverlay
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.gui.widgets.video import QtTextWithBackground

import attr
from typing import Iterable, List, Optional
from PySide2 import QtCore, QtGui
from PySide2.QtCore import QPointF


@attr.s(auto_attribs=True)
class InstanceListOverlay(BaseOverlay):
    """Class to show instance list for the current frame as an overlay."""

    text_box: Optional[QtTextWithBackground] = None

    def add_to_scene(self, video: Video, frame_idx: int):
        """Add list of instances as overlay on video."""
        color_manager = self.player.color_manager
        qt_instances = self.player.view.all_instances

        html = f"Instances:"

        for i, qt_instance in enumerate(qt_instances):
            instance = qt_instance.instance
            color = self.player.color_manager.get_item_color(instance)

            if len(html) > 0:
                html += "<br />"

            html_color = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
            txt = (
                f"<b>{i + 1}. {type(instance).__name__}</b> "
                f"({len(instance)}/{len(instance.skeleton)})"
            )
            html += f"<span style='color:{html_color}'>{txt}</span>"

        text_box = QtTextWithBackground()
        text_box.setDefaultTextColor(QtGui.QColor("white"))
        text_box.setHtml(html)
        text_box.setOpacity(0.7)

        self.text_box = text_box

        self.player.scene.addItem(self.text_box)
        # print(f"sceneRect = {self.player.scene.sceneRect()}")
        # pos = self.player.view.mapToScene(10, 300)
        pos = QPointF(10, 300)
        self.text_box.setPos(pos)
        self.text_box.setVisible(False)  # TODO: Remove this
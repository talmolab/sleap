"""Overlay highlighting frames marked as negative (background) frames."""

from __future__ import annotations

import attr
from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import QGraphicsRectItem

from sleap.gui.overlays.base import BaseOverlay
from sleap.gui.widgets.video import QtTextWithBackground

# Dodger blue — distinct from the light blue used for predicted instances.
NEGATIVE_FRAME_COLOR = (30, 144, 255)


@attr.s(auto_attribs=True)
class NegativeFrameOverlay(BaseOverlay):
    """Draws a blue border and caption when the current frame is negative.

    A negative frame is explicitly marked as containing no animals. This overlay
    gives an unmissable on-canvas cue so the state is obvious while labeling,
    mirroring how unconverted predictions are highlighted.

    Attributes:
        labels: The :class:`Labels` dataset from which to get overlay data.
        player: The video player in which to show the overlay.
    """

    def add_to_scene(self, video, frame_idx):
        """Adds the negative-frame border and caption to the player scene."""
        self.items = []

        if self.labels is None or video is None or frame_idx is None:
            return

        lf = self.labels.find(video, frame_idx, return_new=True)[0]
        if not lf.is_negative:
            return

        frame_rect = self.player.scene.sceneRect()
        if frame_rect.isEmpty():
            return

        color = QColor(*NEGATIVE_FRAME_COLOR)

        # Thick border just inside the periphery of the whole frame. The rect is
        # inset slightly so the stroke stays fully visible when the view is fit
        # exactly to the frame; the pen is cosmetic so its width is constant
        # regardless of zoom.
        inset_x = frame_rect.width() * 0.012
        inset_y = frame_rect.height() * 0.012
        border_rect = QRectF(
            frame_rect.left() + inset_x,
            frame_rect.top() + inset_y,
            frame_rect.width() - 2 * inset_x,
            frame_rect.height() - 2 * inset_y,
        )
        border = QGraphicsRectItem(border_rect)
        border_pen = QPen(color, 6)
        border_pen.setCosmetic(True)
        border.setPen(border_pen)
        self.player.scene.addItem(border)
        self.items.append(border)

        # "Negative Frame" caption pinned to the top-left corner. The text item
        # ignores view transformations, so it stays a constant on-screen size.
        caption = QtTextWithBackground()
        caption.setDefaultTextColor(color)
        font = caption.font()
        font.setPointSize(14)
        font.setBold(True)
        caption.setFont(font)
        caption.setPlainText("Negative Frame")
        caption.setPos(border_rect.topLeft())
        self.player.scene.addItem(caption)
        self.items.append(caption)

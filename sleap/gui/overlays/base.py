"""
Base classes for overlays.

Overlays are used for showing additional visuals on top of a video frame (i.e.,
a `QtVideoPlayer` object). Overlay objects are created in the main GUI app,
which then automatically calls the `add_to_scene` for each loaded overlay after
drawing a frame (i.e., when user navigates to a new frame or something changes
so that current frame must be redrawn).
"""

from __future__ import annotations

import abc
import logging

import attr
from qtpy.QtWidgets import QGraphicsItem

from sleap_io import Labels, Video
from sleap.gui.widgets.video import QtVideoPlayer

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class BaseOverlay(abc.ABC):
    """Abstract base class for overlays.

    Most overlays need rely on the `Labels` from which to get data and need the
    `QtVideoPlayer` to which a `QGraphicsItem` will be added, so these
    attributes are included in the base class.

    Args:
        labels: the `Labels` from which to get data
        player: the `QtVideoPlayer` to which a `QGraphicsObject` item will be added
        items: stores all `QGraphicsItem`s currently added to the player from this
            overlay
    """

    labels: Labels | None = None
    player: QtVideoPlayer | None = None
    items: list[QGraphicsItem] | None = None

    @abc.abstractmethod
    def add_to_scene(self, video: Video, frame_idx: int):
        """Add items to scene.

        To use the `remove_from_scene` and `redraw` methods, keep track of a list of
        `QGraphicsItem`s added in this function.
        """
        # Start your method with:
        self.items = []

        # As items are added to the `QtVideoPlayer`, keep track of these items:
        item = self.player.scene.addItem(...)
        self.items.append(item)

    def remove_from_scene(self):
        """Remove all items added to scene by this overlay.

        This method does not need to be called when changing the plot to a new frame.
        """
        if self.items is None:
            return
        for item in self.items:
            try:
                self.player.scene.removeItem(item)

            except RuntimeError as e:
                # Internal C++ object (PySide2.QtWidgets.QGraphicsPathItem) already
                # deleted.
                logger.debug(e)

        # Stop tracking the items after they been removed from the scene
        self.items = []

    def redraw(self, video, frame_idx, *args, **kwargs):
        """Remove all items from the scene before adding new items to the scene.

        This method does not need to be called when changing the plot to a new frame.
        """
        self.remove_from_scene(*args, **kwargs)
        self.add_to_scene(video, frame_idx, *args, **kwargs)


h5_colors = [
    [204, 81, 81],
    [81, 204, 204],
    [51, 127, 127],
    [127, 51, 51],
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
    [51, 113, 127],
]

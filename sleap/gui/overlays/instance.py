"""
Module with overlay for showing instances.
"""
import attr

from sleap.gui.state import GuiState
from sleap.gui.video import QtVideoPlayer
from sleap.io.dataset import Labels


@attr.s(auto_attribs=True)
class InstanceOverlay:
    """Class for adding instances as overlays on video frames.

    Attributes:
        labels: The :class:`Labels` dataset from which to get overlay data.
        player: The video player in which to show overlay.
        color_predicted: Whether to show predicted instances in color (
            rather than all in gray/yellow).
    """

    labels: Labels = None
    player: QtVideoPlayer = None
    color_predicted: bool = False
    state: GuiState = GuiState()

    def add_to_scene(self, video, frame_idx):
        """Adds overlay for frame to player scene."""
        if self.labels is None:
            return

        lf = self.labels.find(video, frame_idx, return_new=True)[0]

        for instance in lf.instances_to_show:
            self.player.addInstance(instance=instance)

        self.player.showLabels(self.state.get("show labels", default=True))
        self.player.showEdges(self.state.get("show edges", default=True))

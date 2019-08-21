import attr

from PySide2 import QtWidgets

from sleap.gui.video import QtVideoPlayer
from sleap.io.dataset import Labels
from sleap.gui.overlays.tracks import TrackColorManager

@attr.s(auto_attribs=True)
class InstanceOverlay:

    labels: Labels=None
    player: QtVideoPlayer=None
    color_manager: TrackColorManager=TrackColorManager(labels)
    color_predicted: bool=False

    def add_to_scene(self, video, frame_idx):
        if self.labels is None: return

        lf = self.labels.find(video, frame_idx, return_new=True)[0]

        count_no_track = 0
        for i, instance in enumerate(lf.instances_to_show):

            if instance.track in self.labels.tracks:
                pseudo_track = instance.track
            else:
                # Instance without track
                pseudo_track = len(self.labels.tracks) + count_no_track
                count_no_track += 1

            is_predicted = hasattr(instance,"score")

            self.player.addInstance(instance=instance,
                    color=self.color_manager.get_color(pseudo_track),
                    predicted=is_predicted,
                    color_predicted=self.color_predicted)
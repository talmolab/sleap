"""
Track trail and track list overlays.
"""
from sleap.gui.overlays.base import BaseOverlay
from sleap.instance import Track
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.prefs import prefs
from sleap.gui.widgets.video import QtTextWithBackground

import attr

from typing import Iterable, List, Optional

from PySide2 import QtCore, QtGui


@attr.s(auto_attribs=True)
class TrackTrailOverlay(BaseOverlay):
    """Class to show track trails as overlay on video frame.

    Initialize this object with both its data source and its visual output
    scene, and it handles both extracting the relevant data for a given
    frame and plotting it in the output.

    Attributes:
        labels: The :class:`Labels` dataset from which to get overlay data.
        player: The video player in which to show overlay.
        trail_length: The maximum number of frames to include in trail.

    Usage:
        After class is instantiated, call :meth:`add_to_scene(frame_idx)`
        to plot the trails in scene.
    """

    trail_length: int = 0
    show: bool = True
    max_node_count: Optional[int] = None

    def get_track_trails(self, frame_selection: Iterable["LabeledFrame"]):
        """Get data needed to draw track trail.

        Args:
            frame_selection: an iterable with the :class:`LabeledFrame`
                objects to include in trail.

        Returns:
            Dictionary keyed by track, value is list of lists of (x, y) tuples
                i.e., for every node in instance, we get a list of positions
        """

        all_track_trails = dict()

        if not frame_selection:
            return

        nodes = self.labels.skeletons[0].nodes
        max_node_count = self.max_node_count or prefs["trail node count"]
        if len(nodes) > max_node_count:
            nodes = nodes[:max_node_count]

        for frame in frame_selection:

            for inst in frame:
                if inst.track is not None:
                    if inst.track not in all_track_trails:
                        all_track_trails[inst.track] = [[] for _ in range(len(nodes))]

                    # loop through all nodes
                    for node_i, node in enumerate(nodes):

                        if node in inst.nodes and inst[node].visible:
                            point = (inst[node].x, inst[node].y)

                        # Add last location of node so that we can easily
                        # calculate trail length (since we adjust opacity).
                        elif len(all_track_trails[inst.track][node_i]):
                            point = all_track_trails[inst.track][node_i][-1]
                        else:
                            point = None

                        if point is not None:
                            all_track_trails[inst.track][node_i].append(point)

        return all_track_trails

    def get_frame_selection(self, video: Video, frame_idx: int):
        """
        Return `LabeledFrame` objects to include in trail for specified frame.
        """

        frame_selection = self.labels.find(video, range(0, frame_idx + 1))
        frame_selection.sort(key=lambda x: x.frame_idx)

        return frame_selection[-self.trail_length :]

    def get_tracks_in_frame(
        self, video: Video, frame_idx: int, include_trails: bool = False
    ) -> List[Track]:
        """
        Returns list of tracks that have instance in specified frame.

        Args:
            video: Video for which we want tracks.
            frame_idx: Frame index for which we want tracks.
            include_trails: Whether to include tracks which aren't in current
                frame but would be included in trail (i.e., previous frames
                within trail_length).
        Returns:
            List of tracks.
        """

        if include_trails:
            lfs = self.get_frame_selection(video, frame_idx)
        else:
            lfs = self.labels.find(video, frame_idx)

        tracks_in_frame = [inst.track for lf in lfs for inst in lf]

        return tracks_in_frame

    def add_to_scene(self, video: Video, frame_idx: int):
        """Plot the trail on a given frame.

        Args:
            video: current video
            frame_idx: index of the frame to which the trail is attached
        """
        if not self.show or self.trail_length == 0:
            return

        frame_selection = self.get_frame_selection(video, frame_idx)

        all_track_trails = self.get_track_trails(frame_selection)

        for track, trails in all_track_trails.items():

            color = QtGui.QColor(*self.player.color_manager.get_track_color(track))
            pen = QtGui.QPen()
            pen.setCosmetic(True)
            pen.setColor(color)

            seg_count = 2 if self.trail_length <= 50 else 3
            seg_len = self.trail_length // seg_count

            for trail in trails:
                if not trail:
                    continue

                # Break list into fixed length segments so that shorter trails
                # will still have the same number of frames in the earlier
                # segments and will just have shorter or missing later segments.

                segments = []
                for seg_idx in range(seg_count):
                    start = max(0, len(trail) - (seg_idx + 1) * seg_len)
                    end = min(len(trail), 1 + len(trail) - seg_idx * seg_len)
                    segments.append(trail[start:end])
                    if start == 0:
                        break

                # Draw each segment, which each later segment (i.e., the part of
                # trail further back from current frame) with a thinner line.

                width = prefs["trail width"]
                for segment in segments:
                    pen.setWidthF(width)
                    path = self.map_to_qt_path(segment)
                    self.player.scene.addPath(path, pen)
                    width /= 2

    @staticmethod
    def map_to_qt_path(point_list):
        """Converts a list of (x, y)-tuples to a `QPainterPath`."""
        if not point_list:
            return QtGui.QPainterPath()

        path = QtGui.QPainterPath(QtCore.QPointF(*point_list[0]))
        for point in point_list:
            path.lineTo(*point)
        return path


@attr.s(auto_attribs=True)
class TrackListOverlay(BaseOverlay):
    """
    Class to show track number and names in overlay.
    """

    text_box: Optional[QtTextWithBackground] = None

    def add_to_scene(self, video: Video, frame_idx: int):
        """Adds track list as overlay on video."""

        html = "Tracks:"
        num_to_show = min(9, len(self.labels.tracks))

        for i, track in enumerate(self.labels.tracks[:num_to_show]):
            idx = i + 1

            if html:
                html += "<br />"
            color = self.player.color_manager.get_track_color(track)
            html_color = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
            track_text = f"<b>{track.name}</b>"
            if str(idx) != track.name:
                track_text += f" ({idx})"
            html += f"<span style='color:{html_color}'>{track_text}</span>"

        text_box = QtTextWithBackground()
        text_box.setDefaultTextColor(QtGui.QColor("white"))
        text_box.setHtml(html)
        text_box.setOpacity(0.7)

        self.text_box = text_box
        self.visible = False

        self.player.scene.addItem(self.text_box)

    @property
    def visible(self):
        """Gets or set whether overlay is visible."""
        if self.text_box is None:
            return False
        return self.text_box.isVisible()

    @visible.setter
    def visible(self, val):
        if self.text_box is None:
            return
        if val:
            pos = self.player.view.mapToScene(10, 10)
            if pos.x() > 0:
                self.text_box.setPos(pos)
            else:
                self.text_box.setPos(10, 10)
        self.text_box.setVisible(val)

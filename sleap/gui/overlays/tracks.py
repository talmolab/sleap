"""Track trail and track list overlays."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import attr
from qtpy import QtCore, QtGui

from sleap.gui.overlays.base import BaseOverlay
from sleap.gui.widgets.video import QtTextWithBackground
from sleap_io import LabeledFrame
from sleap_io.model.instance import Track
from sleap_io import Video
from sleap.prefs import prefs
from sleap.sleap_io_adaptors.lf_labels_utils import get_instances_to_show

# Fixed cap on the number of alpha-graded segments drawn per trail, regardless
# of `trail_length`. Fading needs at least a few segments to look continuous;
# without a cap, a long trail would turn into hundreds of QGraphicsPathItems
# redrawn every frame.
_MAX_FADE_SEGMENTS = 20


@attr.s(auto_attribs=True)
class TrackTrailOverlay(BaseOverlay):
    """Class to show track trails as overlay on video frame.

    Initialize this object with both its data source and its visual output
    scene, and it handles both extracting the relevant data for a given
    frame and plotting it in the output.

    Trails follow the same vocabulary as sleap-io's `render_video`/
    `render_image` trail options: a single named node (or the instance
    centroid), a global opacity, and an oldest-to-newest alpha fade. Trails
    also work without tracks -- when the project has no tracks at all,
    instances are keyed by their position within each frame instead (matching
    sleap-io's index-keyed fallback), so single-instance / untracked data
    still gets a live trail.

    Attributes:
        labels: The :class:`Labels` dataset from which to get overlay data.
        player: The video player in which to show overlay.
        trail_length: The maximum number of frames to include in trail.
        trail_node: Name of the skeleton node the trail follows, or
            `"centroid"` to follow the mean of visible points.
        trail_alpha: Global opacity multiplier for trails (0.0-1.0).
        trail_alpha_fade: If `True`, fade trails from faint (oldest) to
            opaque (newest). If `False`, the whole trail uses `trail_alpha`.

    Usage:
        After class is instantiated, call :meth:`add_to_scene(frame_idx)`
        to plot the trails in scene.
    """

    trail_length: int = 0
    trail_node: str = "centroid"
    trail_alpha: float = 1.0
    trail_alpha_fade: bool = True
    show: bool = True

    @classmethod
    def get_length_options(cls):
        if prefs["trail length"] != 0:
            return (0, 10, 50, 100, 250, 500, prefs["trail length"])
        return (0, 10, 50, 100, 250, 500)

    @classmethod
    def get_node_options(cls, labels) -> List[str]:
        """Return trail-node choices: `"centroid"` + each primary-skeleton node."""
        if not labels.skeletons:
            return ["centroid"]
        return ["centroid"] + list(labels.skeletons[0].node_names)

    def _resolve_node_index(self, trail_node: str) -> Optional[int]:
        """Return the point index for a named node, or `None` for centroid.

        Also falls back to `None` (centroid) for a node name that doesn't
        exist in the primary skeleton -- e.g. stale state after switching to
        a project with a different skeleton.
        """
        if trail_node == "centroid":
            return None
        node_names = self.get_node_options(self.labels)[1:]
        return node_names.index(trail_node) if trail_node in node_names else None

    @staticmethod
    def _get_trail_point(
        inst, node_index: Optional[int]
    ) -> Optional[Tuple[float, float]]:
        """Return the (x, y) trail target for an instance.

        `node_index` of `None` selects the centroid (mean of visible points,
        matching `Instance.centroid_xy`); otherwise the point at that index.
        """
        if node_index is None:
            return inst.centroid_xy
        if node_index >= len(inst.points) or not inst.points["visible"][node_index]:
            return None
        xy = inst.points["xy"][node_index]
        return (float(xy[0]), float(xy[1]))

    def get_track_trails(
        self, frame_selection: Iterable["LabeledFrame"]
    ) -> Optional[Dict[Union[Track, int], List[Tuple[float, float]]]]:
        """Get data needed to draw track trails.

        Args:
            frame_selection: an iterable with the :class:`LabeledFrame`
                objects to include in trail, oldest to newest.

        Returns:
            Dictionary keyed by `Track` (when the project has tracks) or by
            instance position index within each frame (when it does not --
            mirroring sleap-io's trackless trail fallback). Value is a list
            of (x, y) points, oldest to newest.
        """
        all_trails: Dict[Union[Track, int], List[Tuple[float, float]]] = {}

        if not frame_selection:
            return None

        # Project-wide, matching sleap-io: if the project has any tracks at
        # all, untracked instances are dropped (rather than mixing real
        # tracks and positional fallbacks); if it has none, every instance is
        # keyed by its position in the frame instead.
        has_tracks = len(self.labels.tracks) > 0
        node_index = self._resolve_node_index(self.trail_node)

        for frame in frame_selection:
            # Prefer user instances over predicted instances
            for inst_idx, inst in enumerate(get_instances_to_show(frame)):
                if has_tracks:
                    if inst.track is None:
                        continue
                    key = inst.track
                else:
                    key = inst_idx

                point = self._get_trail_point(inst, node_index)

                if key not in all_trails:
                    all_trails[key] = []
                elif point is None and all_trails[key]:
                    # Carry the last known position forward so a missed
                    # detection doesn't break trail length/fade bookkeeping.
                    point = all_trails[key][-1]

                if point is not None:
                    all_trails[key].append(point)

        return all_trails

    def get_frame_selection(self, video: Video, frame_idx: int):
        """Return `LabeledFrame` objects to include in trail for specified frame."""

        frame_selection = self.labels.find(video, range(0, frame_idx + 1))
        frame_selection.sort(key=lambda x: x.frame_idx)

        return frame_selection[-self.trail_length :]

    def get_tracks_in_frame(
        self, video: Video, frame_idx: int, include_trails: bool = False
    ) -> List[Track]:
        """Returns list of tracks that have instance in specified frame.

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
        self.items = []

        if not self.show or self.trail_length == 0:
            return

        frame_selection = self.get_frame_selection(video, frame_idx)

        all_trails = self.get_track_trails(frame_selection)
        if not all_trails:
            return

        width = prefs["trail width"]

        for key, trail in all_trails.items():
            n_points = len(trail)
            if n_points < 2:
                continue

            # `get_track_color` accepts either a `Track` or a plain int, so the
            # untracked (positional-index) case falls out of the same call --
            # no separate color path needed.
            color = self.player.color_manager.get_track_color(key)
            qcolor = QtGui.QColor(*color)
            pen = QtGui.QPen()
            pen.setCosmetic(True)
            pen.setWidthF(width)

            # Segments fade oldest -> newest, matching sleap-io's
            # `draw_trails`, capped at `_MAX_FADE_SEGMENTS` so a long trail
            # doesn't turn into hundreds of scene items.
            n_segments = (
                min(n_points - 1, _MAX_FADE_SEGMENTS) if self.trail_alpha_fade else 1
            )
            boundaries = [
                round(i * (n_points - 1) / n_segments) for i in range(n_segments + 1)
            ]

            for seg_idx in range(n_segments):
                start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
                if end <= start:
                    continue

                if self.trail_alpha_fade:
                    # Newest segment is fully opaque; oldest stays faintly
                    # visible rather than fully transparent.
                    seg_frac = max((seg_idx + 1) / n_segments, 0.05)
                else:
                    seg_frac = 1.0

                qcolor.setAlphaF(max(0.0, min(1.0, seg_frac * self.trail_alpha)))
                pen.setColor(qcolor)
                path = self.map_to_qt_path(trail[start : end + 1])
                item = self.player.scene.addPath(path, pen)
                self.items.append(item)

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
    """Class to show track number and names in overlay."""

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

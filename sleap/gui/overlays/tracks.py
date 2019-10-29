"""
Module that handles track-related overlays (including track color).
"""

from sleap.instance import Track
from sleap.io.dataset import Labels
from sleap.io.video import Video

import attr
import itertools
from typing import Union

from PySide2 import QtCore, QtGui


class TrackColorManager(object):
    """Class to determine color to use for track.

    The color depends on the order of the tracks in `Labels` object,
    so we need to initialize with `Labels`.

    Args:
        labels: The :class:`Labels` dataset which contains the tracks for
            which we want colors.
        palette: String with the color palette name to use.
    """

    def __init__(self, labels: Labels = None, palette: str = "standard"):
        self.labels = labels

        # alphabet
        # "A Colour Alphabet and the Limits of Colour Coding", Paul Green-Armytage
        # https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d

        # twelve
        # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12

        self._palettes = {
            "standard": [
                [0, 114, 189],
                [217, 83, 25],
                [237, 177, 32],
                [126, 47, 142],
                [119, 172, 48],
                [77, 190, 238],
                [162, 20, 47],
            ],
            "five+": [
                [228, 26, 28],
                [55, 126, 184],
                [77, 175, 74],
                [152, 78, 163],
                [255, 127, 0],
            ],
            "solarized": [
                [181, 137, 0],
                [203, 75, 22],
                [220, 50, 47],
                [211, 54, 130],
                [108, 113, 196],
                [38, 139, 210],
                [42, 161, 152],
                [133, 153, 0],
            ],
            "alphabet": [
                [240, 163, 255],
                [0, 117, 220],
                [153, 63, 0],
                [76, 0, 92],
                [25, 25, 25],
                [0, 92, 49],
                [43, 206, 72],
                [255, 204, 153],
                [128, 128, 128],
                [148, 255, 181],
                [143, 124, 0],
                [157, 204, 0],
                [194, 0, 136],
                [0, 51, 128],
                [255, 164, 5],
                [255, 168, 187],
                [66, 102, 0],
                [255, 0, 16],
                [94, 241, 242],
                [0, 153, 143],
                [224, 255, 102],
                [116, 10, 255],
                [153, 0, 0],
                [255, 255, 128],
                [255, 255, 0],
                [255, 80, 5],
            ],
            "twelve": [
                [31, 120, 180],
                [51, 160, 44],
                [227, 26, 28],
                [255, 127, 0],
                [106, 61, 154],
                [177, 89, 40],
                [166, 206, 227],
                [178, 223, 138],
                [251, 154, 153],
                [253, 191, 111],
                [202, 178, 214],
                [255, 255, 153],
            ],
        }

        self.mode = "cycle"
        self._modes = dict(cycle=lambda i, c: i % c, clip=lambda i, c: min(i, c - 1))

        self.set_palette(palette)

    @property
    def labels(self):
        """Gets or sets labels dataset for which we are coloring tracks."""
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val

    @property
    def palette(self):
        """Gets or sets palette (by name)."""
        return self._palette

    @palette.setter
    def palette(self, palette):
        self._palette = palette

        if isinstance(palette, str):
            self.mode = "clip" if palette.endswith("+") else "cycle"

            if palette in self._palettes:
                self._color_map = self._palettes[palette]
            else:
                self._color_map = self._palettes["standard"]
        else:
            self._color_map = palette

    @property
    def palette_names(self):
        """Gets list of palette names."""
        return self._palettes.keys()

    def set_palette(self, palette):
        """Functional alias for palette property setter."""
        self.palette = palette

    def get_color(self, track: Union[Track, int]):
        """Return the color to use for a given track.

        Args:
            track: `Track` object or an int
        Returns:
            (r, g, b)-tuple
        """
        track_idx = (
            self.labels.tracks.index(track) if isinstance(track, Track) else track
        )
        if track_idx is None:
            return (0, 0, 0)
        color_idx = self._modes[self.mode](track_idx, len(self._color_map))
        color = self._color_map[color_idx]
        return color


@attr.s(auto_attribs=True)
class TrackTrailOverlay:
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

    labels: Labels = None
    player: "QtVideoPlayer" = None
    trail_length: int = 4
    show: bool = False

    def get_track_trails(self, frame_selection, track: Track):
        """Get data needed to draw track trail.

        Args:
            frame_selection: an interable with the :class:`LabeledFrame`
                objects to include in trail.
            track: the :class:`Track` for which to get trail

        Returns:
            list of lists of (x, y) tuples
                i.e., for every node in instance, we get a list of positions
        """

        all_trails = [[] for _ in range(len(self.labels.nodes))]

        for frame in frame_selection:
            frame_idx = frame.frame_idx

            inst_on_track = [instance for instance in frame if instance.track == track]
            if inst_on_track:
                # just use the first instance from this track in this frame
                inst = inst_on_track[0]
                # loop through all nodes
                for node_i, node in enumerate(self.labels.nodes):

                    if node in inst.nodes and inst[node].visible:
                        point = (inst[node].x, inst[node].y)
                    elif len(all_trails[node_i]):
                        point = all_trails[node_i][-1]
                    else:
                        point = None

                    if point is not None:
                        all_trails[node_i].append(point)

        return all_trails

    def get_frame_selection(self, video: Video, frame_idx: int):
        """
        Return `LabeledFrame` objects to include in trail for specified frame.
        """

        frame_selection = self.labels.find(video, range(0, frame_idx + 1))
        frame_selection.sort(key=lambda x: x.frame_idx)

        return frame_selection[-self.trail_length :]

    def get_tracks_in_frame(self, video: Video, frame_idx: int):
        """Return list of tracks that have instance in specified frame."""

        tracks_in_frame = [
            inst.track for lf in self.labels.find(video, frame_idx) for inst in lf
        ]
        return tracks_in_frame

    def add_to_scene(self, video: Video, frame_idx: int):
        """Plot the trail on a given frame.

        Args:
            video: current video
            frame_idx: index of the frame to which the trail is attached
        """
        if not self.show:
            return

        frame_selection = self.get_frame_selection(video, frame_idx)
        tracks_in_frame = self.get_tracks_in_frame(video, frame_idx)

        for track in tracks_in_frame:

            trails = self.get_track_trails(frame_selection, track)

            color = QtGui.QColor(*self.player.color_manager.get_color(track))
            pen = QtGui.QPen()
            pen.setCosmetic(True)

            for trail in trails:
                half = len(trail) // 2

                color.setAlphaF(1)
                pen.setColor(color)
                polygon = self.map_to_qt_polygon(trail[:half])
                self.player.scene.addPolygon(polygon, pen)

                color.setAlphaF(0.5)
                pen.setColor(color)
                polygon = self.map_to_qt_polygon(trail[half:])
                self.player.scene.addPolygon(polygon, pen)

    @staticmethod
    def map_to_qt_polygon(point_list):
        """Converts a list of (x, y)-tuples to a `QPolygonF`."""
        return QtGui.QPolygonF(list(itertools.starmap(QtCore.QPointF, point_list)))


@attr.s(auto_attribs=True)
class TrackListOverlay:
    """
    Class to show track number and names in overlay.
    """

    labels: Labels = None
    player: "QtVideoPlayer" = None
    text_box = None

    def add_to_scene(self, video: Video, frame_idx: int):
        """Adds track list as overlay on video."""
        from sleap.gui.video import QtTextWithBackground

        html = ""
        num_to_show = min(9, len(self.labels.tracks))

        for i, track in enumerate(self.labels.tracks[:num_to_show]):
            idx = i + 1

            if html:
                html += "<br />"
            color = self.player.color_manager.get_color(track)
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

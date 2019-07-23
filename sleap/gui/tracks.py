from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, PredictedInstance, Point, LabeledFrame, Track
from sleap.io.dataset import Labels
from sleap.io.video import Video

import itertools
from typing import Union

from PySide2 import QtCore, QtWidgets, QtGui

class TrackColorManager():
    """Class to determine color to use for track. The color depends on the order of
    the tracks in `Labels` object, so we need to initialize with `Labels`.

    Args:
        labels: `Labels` object which contains the tracks for which we want colors
    """

    def __init__(self, labels: Labels=None, palette="standard"):
        self.labels = labels

        # alphabet
        # "A Colour Alphabet and the Limits of Colour Coding", Paul Green-Armytage
        # https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d

        # twelve
        # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12

        self._palettes = dict(
        standard = [
            [0,   114,   189],
            [217,  83,    25],
            [237, 177,    32],
            [126,  47,   142],
            [119, 172,    48],
            [77,  190,   238],
            [162,  20,    47],
        ],
        solarized = [
            [181, 137, 0],
            [203, 75, 22],
            [220, 50, 47],
            [211, 54, 130],
            [108, 113, 196],
            [38, 139, 210],
            [42, 161, 152],
            [133, 153, 0],
        ],
        alphabet = [
            [240,163,255],
            [0,117,220],
            [153,63,0],
            [76,0,92],
            [25,25,25],
            [0,92,49],
            [43,206,72],
            [255,204,153],
            [128,128,128],
            [148,255,181],
            [143,124,0],
            [157,204,0],
            [194,0,136],
            [0,51,128],
            [255,164,5],
            [255,168,187],
            [66,102,0],
            [255,0,16],
            [94,241,242],
            [0,153,143],
            [224,255,102],
            [116,10,255],
            [153,0,0],
            [255,255,128],
            [255,255,0],
            [255,80,5],
        ],
        twelve = [
            [31,120,180],
            [51,160,44],
            [227,26,28],
            [255,127,0],
            [106,61,154],
            [177,89,40],
            [166,206,227],
            [178,223,138],
            [251,154,153],
            [253,191,111],
            [202,178,214],
            [255,255,153],
        ]
        )

        self.set_palette(palette)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val

    @property
    def palette_names(self):
        return self._palettes.keys()

    def set_palette(self, palette):
        if isinstance(palette, str):
            if palette in self._palettes:
                self._color_maps = self._palettes[palette]
            else:
                self._color_maps = self._palettes["standard"]
        else:
            self._color_maps = palette

    def get_color(self, track: Union[Track, int]):
        """Return the color to use for a given track.

        Args:
            track: `Track` object or an int
        Returns:
            (r, g, b)-tuple
        """
        track_i = self.labels.tracks.index(track) if isinstance(track, Track) else track
        color = self._color_maps[track_i%len(self._color_maps)]
        return color


class TrackTrailManager():
    """Class to show track trails. You initialize this object with both its data source
    and its visual output scene, and it handles both extracting the relevant data for a
    given frame and plotting it in the output.

    Args:
        labels: `Labels` object from which to get data
        scene: `QGraphicsScene` in which to plot trails
        trail_length (optional): maximum number of frames to include in trail

    Usage:
        After class is instatiated, call add_trails_to_scene(frame_idx)
        to plot the trails in scene.
    """

    def __init__(self, labels: Labels, scene: QtWidgets.QGraphicsScene, trail_length: int=4):
        self.labels = labels
        self.scene = scene
        self.trail_length = trail_length
        self._color_manager = TrackColorManager(labels)

    def get_track_trails(self, frame_selection, track: Track):
        """Get data needed to draw track trail.

        Args:
            frame_selection: an interable with the `LabeledFrame`s to include in trail
            track: the `Track` for which to get trail

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

    def get_frame_selection(self, frame_idx: int):
        """Return list of `LabeledFrame`s to include in trail for specificed frame."""
        frame_selection = [frame for frame in self.labels.labeled_frames
                           if frame.frame_idx <= frame_idx]
        frame_selection.sort(key=lambda x: x.frame_idx)
        return frame_selection[-self.trail_length:]

    def get_tracks_in_frame(self, frame_idx: int):
        """Return list of tracks that have instance in specified frame."""
        tracks_in_frame = [instance.track for frame in self.labels.labeled_frames for instance in frame
                           if frame.frame_idx == frame_idx]
        return tracks_in_frame

    def add_trails_to_scene(self, frame_idx: int):
        """Plot the trail on a given frame.

        Args:
            frame_idx: index of the frame to which the trail is attached
        """

        frame_selection = self.get_frame_selection(frame_idx)
        tracks_in_frame = self.get_tracks_in_frame(frame_idx)

        for track in tracks_in_frame:

            trails = self.get_track_trails(frame_selection, track)

            color = QtGui.QColor(*self._color_manager.get_color(track))
            pen = QtGui.QPen()
            pen.setCosmetic(True)

            for trail in trails:
                half = len(trail)//2

                color.setAlphaF(1)
                pen.setColor(color)
                polygon = self.map_to_qt_polygon(trail[:half])
                self.scene.addPolygon(polygon, pen)

                color.setAlphaF(.5)
                pen.setColor(color)
                polygon = self.map_to_qt_polygon(trail[half:])
                self.scene.addPolygon(polygon, pen)

    @staticmethod
    def map_to_qt_polygon(point_list):
        """Converts a list of (x, y)-tuples to a `QPolygonF`."""
        return QtGui.QPolygonF(list(itertools.starmap(QtCore.QPointF, point_list)))
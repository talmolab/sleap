from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, PredictedInstance, Point, LabeledFrame
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.gui.video import QtVideoPlayer

import itertools

from PySide2 import QtCore, QtWidgets, QtGui

class TrackColorManager():
    """Class to determine color to use for track. The color depends on the order of
    the tracks in `Labels` object, so we need to initialize with `Labels`.

    Args:
        labels: `Labels` object which contains the tracks for which we want colors
    """

    def __init__(self, labels):
        self._labels = labels

        self._color_maps = [
            [0,   114,   189],
            [217,  83,    25],
            [237, 177,    32],
            [126,  47,   142],
            [119, 172,    48],
            [77,  190,   238],
            [162,  20,    47],
            ]

    def get_color(self, track):
        """Return the color to use for a given track.

        Args:
            track: `Track` object
        Returns:
            (r, g, b)-tuple
        """
        track_i = self._labels.tracks.index(track)
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

    def __init__(self, labels, scene, trail_length=4):
        self.labels = labels
        self.scene = scene
        self.trail_length = trail_length
        self._color_manager = TrackColorManager(labels)

    def get_track_trails(self, frame_selection, track):
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

    def get_frame_selection(self, frame_idx):
        """Return list of `LabeledFrame`s to include in trail for specificed frame."""
        frame_selection = [frame for frame in self.labels.labeled_frames
                           if frame.frame_idx <= frame_idx]
        frame_selection.sort(key=lambda x: x.frame_idx)
        return frame_selection[-self.trail_length:]

    def get_tracks_in_frame(self, frame_idx):
        """Return list of tracks that have instance in specified frame."""
        tracks_in_frame = [instance.track for frame in self.labels.labeled_frames for instance in frame
                           if frame.frame_idx == frame_idx]
        return tracks_in_frame

    def add_trails_to_scene(self, frame_idx):
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


if __name__ == "__main__":

    filename = "tests/data/json_format_v2/centered_pair_predictions.json"
    labels = Labels.load_json(filename)

    data_path = "tests/data/json_format_v2/centered_pair_low_quality.mp4"
    vid = Video.from_filename(data_path)

    app = QtWidgets.QApplication([])
    window = QtVideoPlayer(video=vid)
    scene = window.view.scene

    trail_manager = TrackTrailManager(labels=labels, scene=scene, trail_length = 5)

    window.changedPlot.connect(lambda parent, i:
                                    trail_manager.add_trails_to_scene(parent.frame_idx))
    window.plot()
    window.show()

    app.exec_()
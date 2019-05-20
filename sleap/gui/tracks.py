from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, PredictedInstance, Point, LabeledFrame
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.gui.video import QtVideoPlayer

import numpy as np
import itertools

from PySide2 import QtCore, QtWidgets, QtGui

def get_track_paths(labels, frame_selection, track):
    all_paths = [[] for _ in range(len(labels.nodes))]
    
    for frame in frame_selection:
        frame_idx = frame.frame_idx
        
        inst_on_track = [instance for instance in frame if instance.track == track]
        if inst_on_track:
            # just use the first instance from this track in this frame
            inst = inst_on_track[0]
            # loop through all nodes
            for node_i, node in enumerate(labels.nodes):

                if node in inst.nodes and inst[node].visible:
                    point = (inst[node].x, inst[node].y)
                elif len(all_paths[node_i]):
                    point = all_paths[node_i][-1]
                else:
                    point = None

                if point is not None:
                    all_paths[node_i].append(point)

    return all_paths

def get_track_paths_for_plt(labels, frame_selection, track):
    all_paths = {node: dict(x=[],y=[],z=[]) for node in labels.nodes}
    
    for frame_offset, frame in enumerate(frame_selection):
        frame_idx = frame.frame_idx
        
        inst_on_track = [instance for instance in frame if instance.track == track]
        if inst_on_track:
            # just use the first instance from this track in this frame
            inst = inst_on_track[0]
            # loop through all nodes
            for node in labels.nodes:

                if node in inst.nodes and inst[node].visible:

                    all_paths[node]["x"].append(inst[node].x)
                    all_paths[node]["y"].append(inst[node].y)
                    all_paths[node]["z"].append(frame_offset)

    return all_paths


def map_to_qt_polygon(point_list):
    return QtGui.QPolygonF(list(itertools.starmap(QtCore.QPointF, point_list)))

def add_paths_to_scene(labels, frame_idx, scene, cmap, traj_len=20):
    frame_selection = [frame for frame in labels.labeled_frames
                       if frame.frame_idx in range(frame_idx-traj_len, frame_idx)]
    frame_selection.sort(key=lambda x: x.frame_idx)

    tracks_in_frame = [instance.track for frame in labels.labeled_frames for instance in frame
                       if frame.frame_idx == frame_idx]
    for track_i, track in enumerate(tracks_in_frame):
        paths = get_track_paths(labels, frame_selection, track)

        color = QtGui.QColor(*cmap[track_i%len(cmap)])
        pen = QtGui.QPen()
        pen.setCosmetic(True)

        for path in paths:
            half = len(path)//2
            
            color.setAlphaF(1)
            pen.setColor(color)
            polygon = map_to_qt_polygon(path[:half])
            scene.addPolygon(polygon, pen)
            
            color.setAlphaF(.5)
            pen.setColor(color)
            polygon = map_to_qt_polygon(path[half:])
            scene.addPolygon(polygon, pen)

def matplotlib_plot_paths():

    frame_selection = [frame for frame in labels.labeled_frames
                       if frame.frame_idx in range(0, 100)]
    frame_selection.sort(key=lambda x: x.frame_idx)

    colors = "blue orange red green yellow".split()

    commands = """
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
"""

    for track_i, track in enumerate(labels.tracks):
        all_plt_paths = get_track_paths_for_plt(labels, frame_selection, track)

        for plt_path in all_plt_paths.values():
            if len(plt_path["x"]):
                xs, ys, zs = plt_path["x"], plt_path["y"], plt_path["z"]
                c = colors[track_i%len(colors)]
                commands += f"ax.plot({xs}, {ys}, {zs}, linewidth=.5, c='{c}')\n"

    commands += "plt.show()"

    with open("test_plt.py", "w") as f:
        f.write(commands)

def show_video_with_paths():

    app = QtWidgets.QApplication([])
    window = QtVideoPlayer(video=vid)
    
    def plot_paths(parent, i):
        add_paths_to_scene(labels, parent.frame_idx, window.view.scene, cmap)

    window.changedPlot.connect(plot_paths)
    window.plot()
    window.show()

    app.exec_()
    
if __name__ == "__main__":
    cmap = np.array([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ])

    filename = "tests/data/json_format_v2/centered_pair_predictions.json"
    labels = Labels.load_json(filename)
    
    data_path = "tests/data/json_format_v2/centered_pair_low_quality.mp4"
    vid = Video.from_filename(data_path)

    matplotlib_plot_paths()
    show_video_with_paths()
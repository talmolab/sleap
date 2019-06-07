""" Data generation routines """

import numpy as np
import math
from sleap.io.dataset import Labels

def generate_images(labels:Labels, scale=1.0, output_size=None):

    vid = labels.videos[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    imgs = []
    for labeled_frame in labels:
        img = labeled_frame.video[labeled_frame.frame_idx]
        # TODO: resizing
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)

    # TODO: more options for normalization
    if imgs.dtype == "uint8":
        imgs = imgs.astype("float32") / 255

    return imgs

def generate_points(labels):
    """Generates point data for instances in frames in labels.

    Output is in the format expected by
    * generate_confmaps_from_points()
    * generate_pafs_from_points()
    * augmentation.Augmenter()

    Args:
        labels: the `Labels` object for which we want instance points

    Returns:
        a list (each frame) of lists (each instance) of ndarrays (of points)
            i.e., frames -> instances -> point_array
    """
    return [[inst.points_array(invisible_as_nan=True) for inst in lf.instances] for lf in labels]

def generate_confmaps_from_points(frames_inst_points, skeleton, shape, sigma=5.0, scale=1.0, output_size=None) -> np.ndarray:
    """
    Generates confmaps for set of frames.
    This is used to generate confmaps on the fly during training,
    possibly after augmentation has moved the instance points.

    Args:
        frames_inst_points: a list (each frame) of a list (each inst) of point arrays
        skeleton: skeleton from which to get list of nodes
        shape: shape of frame image, i.e. (h, w)

    Returns:
        confmaps as ndarray with shape (frames, h, w, nodes)
    """
    full_size = shape
    if output_size is None:
        output_size = (shape[0] // (1/scale), shape[1] // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    ball = get_conf_ball(full_size, output_size, sigma)

    num_frames = len(frames_inst_points)
    num_channels = len(skeleton.nodes)
    confmaps = np.zeros((num_frames, output_size[0], output_size[1], num_channels), dtype="float32")

    for frame_idx, points_arrays in enumerate(frames_inst_points):
        for inst_points in points_arrays:
            for node_idx in range(len(skeleton.nodes)):
                if not np.isnan(np.sum(inst_points[node_idx])):
                    x = inst_points[node_idx][0]
                    y = inst_points[node_idx][1]
                    raster_ball(arr=confmaps[frame_idx], ball=ball, c=node_idx, x=x, y=y)

    return confmaps

def generate_pafs_from_points(frames_inst_points, skeleton, shape, sigma=5.0, scale=1.0, output_size=None) -> np.ndarray:
    """
    Generates pafs for set of frames.
    This is used to generate pafs on the fly during training,
    possibly after augmentation has moved the instance points.

    Args:
        frames_inst_points: a list (each frame) of a list (each inst) of point arrays
        skeleton: skeleton from which to get list of nodes
        shape: shape of frame image, i.e. (h, w)

    Returns:
        pafs as ndarray with shape (frames, h, w, nodes)
    """
    full_size = shape
    if output_size is None:
        output_size = (shape[0] // (1/scale), shape[1] // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    ball = get_conf_ball(full_size, output_size, sigma)

    num_frames = len(frames_inst_points)
    num_channels = len(skeleton.edges) * 2

    pafs = np.zeros((num_frames, output_size[0], output_size[1], num_channels), dtype="float32")
    for frame_idx, points_arrays in enumerate(frames_inst_points):
        for inst_points in points_arrays:
            for c, (src_node, dst_node) in enumerate(skeleton.edges):
                src_idx = skeleton.node_to_index(src_node.name)
                dst_idx = skeleton.node_to_index(dst_node.name)
                x0 = inst_points[src_idx][0]
                y0 = inst_points[src_idx][1]
                x1 = inst_points[dst_idx][0]
                y1 = inst_points[dst_idx][1]
                raster_pafs(pafs[frame_idx], c * 2, x0, y0, x1, y1, sigma)

    return pafs

def get_conf_ball(full_size, output_size, sigma):
    # Pre-allocate coordinate grid
    xv = np.linspace(0, full_size[1] - 1, output_size[1], dtype="float32")
    yv = np.linspace(0, full_size[0] - 1, output_size[0], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    x, y = full_size[1]//2, full_size[0]//2
    ball_full = np.exp(-((YY - y) ** 2 + (XX - x) ** 2) / (2 * sigma ** 2))
    window_size = int(sigma*4)
    ball_window = ball_full[y-window_size:y+window_size, x-window_size:x+window_size]

    return ball_window

def raster_ball(arr, ball, c, x, y):
    x, y = int(x), int(y)
    ball_h, ball_w = ball.shape
    out_h, out_w, _ = arr.shape

    ball_slice_y = slice(0, ball_h)
    ball_slice_x = slice(0, ball_w)

    arr_slice_y = slice(y-ball_h//2, y+ball_h//2)
    arr_slice_x = slice(x-ball_w//2, x+ball_w//2)

    # crop ball if it would be out of array bounds
    # i.e., it's close to edge
    if arr_slice_y.start < 0:
        cut = -arr_slice_y.start
        arr_slice_y = slice(0, arr_slice_y.stop)
        ball_slice_y = slice(cut, ball_h)

    if arr_slice_x.start < 0:
        cut = -arr_slice_x.start
        arr_slice_x = slice(0, arr_slice_x.stop)
        ball_slice_x = slice(cut, ball_w)

    if arr_slice_y.stop > out_h:
        cut = arr_slice_y.stop - out_h
        arr_slice_y = slice(arr_slice_y.start, out_h)
        ball_slice_y = slice(cut, ball_h)

    if arr_slice_x.stop > out_w:
        cut = arr_slice_x.stop - out_w
        arr_slice_x = slice(arr_slice_x.start, out_w)
        ball_slice_x = slice(cut, ball_w)

    # impose ball on array
    arr[arr_slice_y, arr_slice_x, c] = np.maximum(
        arr[arr_slice_y, arr_slice_x, c],
        ball[ball_slice_y, ball_slice_x]
        )

def generate_confidence_maps(labels:Labels, sigma=5.0, scale=1.0, output_size=None):
    """Wrapper for generate_confmaps_from_points which takes labels instead of points."""

    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    shape = (vid.height, vid.width)

    points = generate_points(labels)
    confmaps = generate_confmaps_from_points(points, skeleton, shape, sigma=5.0, scale=1.0, output_size=None)

    return confmaps

def raster_pafs(arr, c, x0, y0, x1, y1, sigma=5):
    # skip if any nan
    if np.isnan(np.sum((x0, y0, x1, y1))): return

    delta_x, delta_y = x1 - x0, y1 - y0

    edge_len = (delta_x ** 2 + delta_y ** 2) ** .5
    edge_x = delta_x / edge_len
    edge_y = delta_y / edge_len

    perp_x0 = x0 + (edge_y * sigma)
    perp_y0 = y0 - (edge_x * sigma)
    perp_x1 = x0 - (edge_y * sigma)
    perp_y1 = y0 + (edge_x * sigma)

    # perp 0 -> perp 0 + delta -> perp 1 + delta -> perp 1 -> perp 0
    xx = perp_x0, perp_x0 + delta_x, perp_x1 + delta_x, perp_x1
    yy = perp_y0, perp_y0 + delta_y, perp_y1 + delta_y, perp_y1

    from skimage.draw import polygon, polygon_perimeter

    points_y, points_x = polygon(yy, xx, (arr.shape[0], arr.shape[1]))
    perim_y, perim_x = polygon_perimeter(yy, xx, shape=(arr.shape[0], arr.shape[1]), clip=True)

    # make sure we don't include points more than once
    # otherwise we'll add the edge vector to itself at that point
    all_points = set(zip(points_x, points_y)).union(set(zip(perim_x, perim_y)))

    for x, y in all_points:
        arr[y, x, c] = edge_x
        arr[y, x, c + 1] = edge_y

def get_labels_edge_points_list(labels):
    return [(frame_idx, [instance_edge_points(instance) for instance in labeled_frame.instances]) for
            frame_idx, labeled_frame in enumerate(labels)]

def instance_edge_points(instance):
    skeleton = instance.skeleton
    points = [(edge_idx, (edge_points_tuples(instance, src_node, dst_node)))
              for edge_idx, (src_node, dst_node)
              in enumerate(skeleton.edges)
              if points_are_present(instance, src_node, dst_node)]
    return points

def edge_points_tuples(instance, src_node, dst_node):
    x0, y0 = instance[src_node].x, instance[src_node].y
    x1, y1 = instance[dst_node].x, instance[dst_node].y
    # (instance[src_node].x, instance[src_node].y), (instance[dst_node].x, instance[dst_node].y)
    # return np.array((x0, y0)), np.array((x1, y1))
    return (x0, y0), (x1, y1)

def points_are_present(instance, src_node, dst_node):
    if src_node in instance and dst_node in instance and instance[src_node].visible and instance[dst_node].visible:
        return True
    else:
        return False

def generate_pafs(labels: Labels, sigma=5.0, scale=1.0, output_size=None):
    """Wrapper for generate_pafs_from_points which takes labels instead of points."""

    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    shape = (vid.height, vid.width)

    points = generate_points(labels)
    pafs = generate_pafs_from_points(points, skeleton, shape, sigma=5.0, scale=1.0, output_size=None)

    return pafs

def demo_datagen_time():
    data_path = "tests/data/json_format_v2/centered_pair_predictions.json"

    global labels
    labels = Labels.load_json(data_path)
    labels.labeled_frames = labels.labeled_frames[123:423:10]
    count = len(labels)
    timing_reps = 1

    import timeit
    t = timeit.timeit("generate_confidence_maps(labels)", number=timing_reps, globals=globals())
    t /= timing_reps
    print(f"confmaps time: {t} = {t/count} s/frame for {count} frames")

    t = timeit.timeit("generate_pafs(labels)", number=timing_reps, globals=globals())
    t /= timing_reps
    print(f"pafs time: {t} = {t/count} s/frame for {count} frames")

def demo_datagen():
    import os

    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        # data_path = "tests/data/json_format_v2/centered_pair_predictions.json"
        data_path = "tests/data/json_format_v2/minimal_instance.json"

    labels = Labels.load_json(data_path)
    # labels.labeled_frames = labels.labeled_frames[123:423:10]

    imgs = generate_images(labels)
    print("--imgs--")
    print(imgs.shape)
    print(imgs.dtype)
    print(np.ptp(imgs))

    from PySide2 import QtWidgets
    from sleap.io.video import Video
    from sleap.gui.confmapsplot import demo_confmaps
    from sleap.gui.quiverplot import demo_pafs

    app = QtWidgets.QApplication([])
    vid = Video.from_numpy(imgs * 255)

    confmaps = generate_confidence_maps(labels)
    print("--confmaps--")
    print(confmaps.shape)
    print(confmaps.dtype)
    print(np.ptp(confmaps))

    demo_confmaps(confmaps, vid)

    pafs = generate_pafs(labels)
    print("--pafs--")
    print(pafs.shape)
    print(pafs.dtype)
    print(np.ptp(pafs))

    demo_pafs(pafs, vid)

    app.exec_()

if __name__ == "__main__":
    demo_datagen()
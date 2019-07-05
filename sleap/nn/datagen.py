""" Data generation routines """

import numpy as np
import math
import random

from operator import itemgetter
from typing import Optional

from sleap.io.dataset import Labels

def generate_images(labels:Labels, scale: float=1.0,
                    resize_hack: bool=True, frame_limit: bool=None) -> np.ndarray:
    """
    Generate a ndarray of the image data for any labeled frames

    Args:
        labels: the `Labels` object for which we want images
        scale: the factor to use when rescaling

    Returns:
        ndarray with shape (images, height, width, channels)
    """
    import cv2

    imgs = []
    for labeled_frame in labels.user_labeled_frames[:frame_limit]:
        img = labeled_frame.video[labeled_frame.frame_idx][0]
        # rescale by factor
        y, x, c = img.shape
        if scale != 1.0 or resize_hack:
            y_scaled, x_scaled = int(y//(1/scale)), int(x//(1/scale))

            # FIXME: hack to resize image so dimensions are divisible by 8
            if resize_hack:
                y_scaled, x_scaled = y_scaled//8*8, x_scaled//8*8

            if (x, y) != (x_scaled, y_scaled):
                # resize image
                # note that cv2 wants (x, y) rather than numpy-style (y, x)
                img = cv2.resize(img, (x_scaled, y_scaled))

                # add back singleton channel removed by cv2
                if c == 1:
                    img = img[..., None]
        imgs.append(img)

    imgs = np.stack(imgs, axis=0)

    # TODO: more options for normalization
    if imgs.dtype == "uint8":
        imgs = imgs.astype("float32") / 255

    return imgs

def generate_points(labels:Labels, scale: float=1.0, frame_limit: bool=None) -> list:
    """Generates point data for instances in frames in labels.

    Output is in the format expected by
    * generate_confmaps_from_points()
    * generate_pafs_from_points()
    * augmentation.Augmenter()

    Args:
        labels: the `Labels` object for which we want instance points
        scale: the factor to use when rescaling

    Returns:
        a list (each frame) of lists (each instance) of ndarrays (of points)
            i.e., frames -> instances -> point_array
    """
    return [[inst.points_array(invisible_as_nan=True)*scale
                for inst in lf.user_instances]
                for lf in labels.user_labeled_frames[:frame_limit]]

def generate_confmaps_from_points(frames_inst_points,
                        skeleton: Optional['Skeleton'],
                        shape,
                        node_count: Optional[int] = None,
                        sigma:float=5.0, scale:float=1.0, output_size=None) -> np.ndarray:
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
    if skeleton is None and node_count is None:
        raise ValueError("Either skeleton or node_count must be specified.")

    node_count = len(skeleton.nodes) if skeleton is not None else node_count

    full_size = shape
    if output_size is None:
        output_size = (shape[0] // (1/scale), shape[1] // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    ball = _get_conf_ball(full_size, output_size, sigma*scale)

    num_frames = len(frames_inst_points)
    confmaps = np.zeros((num_frames, output_size[0], output_size[1], node_count),
                        dtype="float32")

    for frame_idx, points_arrays in enumerate(frames_inst_points):
        for inst_points in points_arrays:
            for node_idx in range(node_count):
                if not np.isnan(np.sum(inst_points[node_idx])):
                    x = inst_points[node_idx][0]
                    y = inst_points[node_idx][1]
                    _raster_ball(arr=confmaps[frame_idx], ball=ball, c=node_idx, x=x, y=y)

    return confmaps

def generate_pafs_from_points(frames_inst_points, skeleton, shape,
                        sigma:float=5.0, scale:float=1.0, output_size=None) -> np.ndarray:
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

    num_frames = len(frames_inst_points)
    num_channels = len(skeleton.edges) * 2

    pafs = np.zeros((num_frames, output_size[0], output_size[1], num_channels),
                    dtype="float32")
    for frame_idx, points_arrays in enumerate(frames_inst_points):
        for inst_points in points_arrays:
            for c, (src_node, dst_node) in enumerate(skeleton.edges):
                src_idx = skeleton.node_to_index(src_node.name)
                dst_idx = skeleton.node_to_index(dst_node.name)
                x0 = inst_points[src_idx][0]
                y0 = inst_points[src_idx][1]
                x1 = inst_points[dst_idx][0]
                y1 = inst_points[dst_idx][1]
                _raster_pafs(pafs[frame_idx], c * 2, x0, y0, x1, y1, sigma)

    return pafs

def _get_conf_ball(full_size, output_size, sigma):
    # Pre-allocate coordinate grid
    xv = np.linspace(0, full_size[1] - 1, output_size[1], dtype="float32")
    yv = np.linspace(0, full_size[0] - 1, output_size[0], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    x, y = full_size[1]//2, full_size[0]//2
    ball_full = np.exp(-((YY - y) ** 2 + (XX - x) ** 2) / (2 * sigma ** 2))
    window_size = int(sigma*4)
    ball_window = ball_full[y-window_size:y+window_size, x-window_size:x+window_size]

    return ball_window

def _raster_ball(arr, ball, c, x, y):
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
        ball_slice_y = slice(0, ball_h-cut)

    if arr_slice_x.stop > out_w:
        cut = arr_slice_x.stop - out_w
        arr_slice_x = slice(arr_slice_x.start, out_w)
        ball_slice_x = slice(0, ball_w-cut)

    if ball_slice_x.stop <= ball_slice_x.start \
            or ball_slice_y.stop <= ball_slice_y.start:
        return

    # impose ball on array
    arr[arr_slice_y, arr_slice_x, c] = np.maximum(
        arr[arr_slice_y, arr_slice_x, c],
        ball[ball_slice_y, ball_slice_x]
        )

def generate_confidence_maps(labels:Labels, sigma=5.0, scale=1):
    """Wrapper for generate_confmaps_from_points which takes labels instead of points."""

    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    shape = (vid.height, vid.width)

    points = generate_points(labels, scale=scale)
    confmaps = generate_confmaps_from_points(points, skeleton, shape, sigma=sigma)

    return confmaps

def _raster_pafs(arr, c, x0, y0, x1, y1, sigma):
    # skip if any nan
    if np.isnan(np.sum((x0, y0, x1, y1))): return

    delta_x, delta_y = x1 - x0, y1 - y0

    edge_len = (delta_x ** 2 + delta_y ** 2) ** .5

    # skip if no distance between nodes
    if edge_len == 0.0: return

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
    perim_y, perim_x = polygon_perimeter(yy, xx, shape=(arr.shape[0], arr.shape[1]))

    # make sure we don't include points more than once
    # otherwise we'll add the edge vector to itself at that point
    all_points = set(zip(points_x, points_y)).union(set(zip(perim_x, perim_y)))

    for x, y in all_points:
        arr[y, x, c] = edge_x
        arr[y, x, c + 1] = edge_y

def generate_pafs(labels: Labels, sigma:float=5.0, scale:float=1.0) -> np.ndarray:
    """Wrapper for generate_pafs_from_points which takes labels instead of points."""

    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    shape = (vid.height, vid.width)

    points = generate_points(labels, scale=scale)
    pafs = generate_pafs_from_points(points, skeleton, shape, sigma=sigma)

    return pafs

def point_array_bounding_box(point_array: np.ndarray) -> tuple:
    """Returns (x0, y0, x1, y1) for box that bounds point_array."""
    x0 = np.nanmin(point_array[:, 0])
    y0 = np.nanmin(point_array[:, 1])
    x1 = np.nanmax(point_array[:, 0])
    y1 = np.nanmax(point_array[:, 1])
    return x0, y0, x1, y1

def pad_rect_to(x0: int, y0: int, x1: int, y1: int, pad_to: tuple, within: tuple):
    """Grow (x0, y0, x1, y1) so it's as large as pad_to but stays inside within.

    Args:
        x0: point for starting rect
        y0: point for starting rect
        x1: point for starting rect
        y1: point for starting rect
        pad_to: (h, w) for size of rect that we want to return.
        within: (h, w) that we want rect to stay inside.

    Returns:
        (x0, y0, x1, y1) for rect such that
        * (y1-y0), (x1-x0) = pad_to
        * 0 <= (y1-y0) <= within h
        * 0 <= (x1-x0) <= within w
    """
    pad_to_y, pad_to_x = pad_to
    x_margin = pad_to_x - (x1-x0)
    y_margin = pad_to_y - (y1-y0)

    # initial values
    x0 -= x_margin//2
    x1 += x_margin-x_margin//2
    y0 -= y_margin//2
    y1 += y_margin-y_margin//2

    # adjust to stay inside within
    within_y, within_x = within
    if x0 < 0:
        x0 = 0
        x1 = pad_to_x
    if x1 > within_x:
        x1 = within_x
        x0 = within_x-pad_to_x
    if y0 < 0:
        y0 = 0
        y1 = pad_to_y
    if y1 > within_y:
        y1 = within_y
        y0 = within_y-pad_to_y

    return x0, y0, x1, y1

def generate_centroid_points(points: list) -> list:
    """Takes the points for each instance and replaces it with a single centroid point."""

    centroids = [[_centroid(*point_array_bounding_box(point_array))
                    for point_array in frame] for frame in points]

    return centroids

def _centroid(x0, y0, x1, y1) -> np.ndarray:
    a = np.array((x0+(x1-x0)/2, y0+(y1-y0)/2))
    return np.expand_dims(a, axis=0)

def instance_crops(imgs: np.ndarray, points: list,
                    min_crop_size: int=0, negative_samples: int=0) -> np.ndarray:
    """
    Take imgs, points and return imgs, points cropped around instances.

    Note that if there are multiple instances in a image, this will result in more
    (but smaller) instances than we started with.

    Args:
        imgs: output from generate_images()
        points: output from generate_points()
        min_crop_size: int, the minimum crop square size
    Returns:
        imgs, points (matching format of input)
    """
    frame_count = imgs.shape[0]
    img_shape = imgs.shape[1], imgs.shape[2]

    # List of bounding box for every instance
    bbs = [point_array_bounding_box(point_array) for frame in points for point_array in frame]
    bbs = [(int(x0), int(y0), int(x1), int(y1)) for (x0, y0, x1, y1) in bbs]

    # List to map bb to its img frame idx
    img_idxs = [i for i, frame in enumerate(points) for _ in frame]

    # Find least power of 2 that's large enough to bound all the instances
    max_height = max((y1 - y0 for (x0, y0, x1, y1) in bbs))
    max_width = max((x1 - x0 for (x0, y0, x1, y1) in bbs))
    max_dim = max(max_height, max_width)
    max_dim = max(max_dim, min_crop_size)
    # TODO: add extra margin?
    box_side = min((2**i for i in range(4, 10) if 2**i >= max_dim), default=512) # 16 to 512

    # TODO: make sure we have valid box_size

    # Grow all bounding boxes to the same size
    box_shape = min(box_side, img_shape[0]), min(box_side, img_shape[1])
    bbs = list(map(lambda bb: pad_rect_to(*bb, box_shape, img_shape), bbs))

    # Add bounding boxes for negative samples
    neg_sample_list = []

    # Collect negative samples (and some extras)
    for _ in range(max(int(negative_samples*1.5), negative_samples+10)):
        # find negative sample
        # pick a random image
        sample_img_idx = random.randrange(frame_count)
        # pick a random box within image
        x, y = random.randrange(img_shape[1] - box_side), random.randrange(img_shape[0] - box_side)
        sample_bb = (x, y, x+box_side, y+box_side)

        frame_bbs = [bbs[i] for i, frame in enumerate(img_idxs) if frame == sample_img_idx]
        area_covered = sum(map(lambda bb: _overlap_area(sample_bb, bb), frame_bbs))/(box_side**2)

        # append negative sample to lists
        neg_sample_list.append((area_covered, sample_img_idx, sample_bb))

    # Pick the best samples
    neg_sample_list.sort(key=itemgetter(0))
    _, neg_img_idxs, neg_bbs = zip(*neg_sample_list)

    # Include the negative samples with the positive samples
    img_idxs.extend(neg_img_idxs[:negative_samples])
    bbs.extend(neg_bbs[:negative_samples])

    # Crop images

    imgs = [imgs[img_idxs[i], bb[1]:bb[3], bb[0]:bb[2]] for i, bb in enumerate(bbs)] # imgs[frame_idx, y0:y1, x0:x1]
    imgs = np.stack(imgs, axis=0)

    # Make point arrays for each image (instead of each frame as before)

    # Note that we want all points from the frame, not just the points for the instance
    # around which we're cropping (i.e., point_array in frame_points).
    points = list(map(lambda i: points[i], img_idxs))

    # translate points to location w/in cropped image
    points = [point_array - np.asarray([bbs[i][0], bbs[i][1]]) for i, point_array in enumerate(points)]

    return imgs, points

def _overlap_area(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return inter_area

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
        data_path = "tests/data/json_format_v1/centered_pair.json"
        # data_path = "tests/data/json_format_v2/minimal_instance.json"

    labels = Labels.load_json(data_path)
    # labels.labeled_frames = labels.labeled_frames[123:423:10]

    scale = .5

    imgs = generate_images(labels, scale)
    print("--imgs--")
    print(imgs.shape)
    print(imgs.dtype)
    print(np.ptp(imgs))

    points = generate_points(labels, scale)

    # Prepare GUI demo

    from PySide2 import QtWidgets
    from sleap.io.video import Video
    from sleap.gui.confmapsplot import demo_confmaps
    from sleap.gui.quiverplot import demo_pafs

    app = QtWidgets.QApplication([])

    # Centroids

    # generate centoid data on full frames before instance cropping
    centroid_imgs = generate_images(labels, scale=.25)
    centroid_vid = Video.from_numpy(centroid_imgs * 255)
    centroid_points = generate_centroid_points(generate_points(labels, scale=.25))
    centroid_confmaps = generate_confmaps_from_points(centroid_points, None,
                                (centroid_imgs.shape[1], centroid_imgs.shape[2]),
                                node_count=1, sigma=5.0)

    demo_confmaps(centroid_confmaps, centroid_vid)

    # Instance cropping

    imgs, points = instance_crops(imgs, points, min_crop_size=0, negative_samples=15)

    vid = Video.from_numpy(imgs * 255)

    skeleton = labels.skeletons[0]
    img_shape = (imgs.shape[1], imgs.shape[2])

    confmaps = generate_confmaps_from_points(points, skeleton, img_shape, sigma=5.0*scale)
    print("--confmaps--")
    print(confmaps.shape)
    print(confmaps.dtype)
    print(np.ptp(confmaps))

    demo_confmaps(confmaps, vid)

    pafs = generate_pafs_from_points(points, skeleton, img_shape, sigma=5.0*scale)
    print("--pafs--")
    print(pafs.shape)
    print(pafs.dtype)
    print(np.ptp(pafs))

    demo_pafs(pafs, vid)

    app.exec_()

if __name__ == "__main__":
    demo_datagen()
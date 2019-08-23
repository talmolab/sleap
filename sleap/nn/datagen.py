""" Data generation routines """

import cv2
import numpy as np
from math import ceil
import random

from operator import itemgetter
from typing import Dict, List, Optional, Tuple

from sleap.io.dataset import Labels

def generate_training_data(labels, params):
    """
    Generate imgs (ndarray) and points (list) to use for training.

    Encapsulates:
        generating images for relevant frames (generate_images)
        generating points for relevant frames (generate_points)
        cropping images (instance_crops)
        adding random negative samples (also in instance_crops)
        adding specific negative samples (negative_anchor_crops)

    Args:
        labels: the `Labels` object for which we want images
        params: dict with any parameters we need for encapsulated functions
    Returns:
        (imgs, points)-tuple:
        * imgs = ndarray with shape (images, height, width, channels)
        * points = list (each frame) of lists (each instance) of ndarrays (of points)
            i.e., frames -> instances -> point_array
    """

    resize_hack = not params["instance_crop"]

    imgs = generate_images(labels, params["scale"],
                frame_limit=params.get("frame_limit", None),
                resize_hack=resize_hack)

    points = generate_points(labels, params["scale"],
                frame_limit=params.get("frame_limit", None))

    if params["instance_crop"]:
        # Crop and include any *random* negative samples
        imgs, points = instance_crops(
                            imgs, points,
                            min_crop_size = params["min_crop_size"],
                            negative_samples = params["negative_samples"])

        # Include any *specific* negative samples
        imgs, points = add_negative_anchor_crops(
                            labels,
                            imgs, points,
                            scale=params["scale"])

    return imgs, points

def generate_images(labels:Labels, scale: float=1.0,
                    resize_hack: bool=True, frame_limit: int=None) -> np.ndarray:
    """
    Generate a ndarray of the image data for any user labeled frames.

    Wrapper that calls generate_images_from_list() with list of all frames
    that were labeled by user.
    """
    frame_list = [(lf.video, lf.frame_idx)
                  for lf in labels.user_labeled_frames[:frame_limit]]
    return generate_images_from_list(labels, frame_list, scale, resize_hack)

def generate_points(labels:Labels, scale: float=1.0, frame_limit: int=None) -> list:
    """Generates point data for instances for any user labeled frames.

    Wrapper that calls generate_points_from_list() with list of all frames
    that were labeled by user.
    """
    frame_list = [(lf.video, lf.frame_idx)
                  for lf in labels.user_labeled_frames[:frame_limit]]
    return generate_points_from_list(labels, frame_list, scale)

def generate_images_from_list(
                    labels:Labels, frame_list: List[Tuple],
                    scale: float=1.0, resize_hack: bool=True) -> np.ndarray:
    """
    Generate a ndarray of the image data for given list of frames

    Args:
        labels: the `Labels` object for which we want images
        frame_list: list of (video, frame_idx) tuples
        scale: the factor to use when rescaling
        resize_hack: if True, scale img so both dimensions are divisible by 8

    Returns:
        ndarray with shape (images, height, width, channels)
    """
    imgs = []
    for video, frame_idx in frame_list:
        img = video[frame_idx][0]
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

def generate_points_from_list(labels:Labels, frame_list: List[Tuple], scale: float=1.0) -> list:
    """Generates point data for instances in specified frames.

    Output is in the format expected by
    * generate_confmaps_from_points()
    * generate_pafs_from_points()
    * augmentation.Augmenter()

    Args:
        labels: the `Labels` object for which we want instance points
        frame_list: list of (video, frame_idx) tuples
        scale: the factor to use when rescaling

    Returns:
        a list (each frame) of lists (each instance) of ndarrays (of points)
            i.e., frames -> instances -> point_array
    """
    def lf_points_from_singleton(lf_singleton):
        if len(lf_singleton) == 0: return []
        lf = lf_singleton[0]
        points = [inst.points_array(invisible_as_nan=True)*scale
                  for inst in lf.user_instances]
        return points

    lfs = [labels.find(video, frame_idx) for (video, frame_idx) in frame_list]

    return list(map(lf_points_from_singleton, lfs))

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

    output_size = tuple(map(int, output_size))

    ball = _get_conf_ball(output_size, sigma*scale)

    num_frames = len(frames_inst_points)
    confmaps = np.zeros((num_frames, output_size[0], output_size[1], node_count),
                        dtype="float32")

    for frame_idx, points_arrays in enumerate(frames_inst_points):
        for inst_points in points_arrays:
            for node_idx in range(node_count):
                if not np.isnan(np.sum(inst_points[node_idx])):
                    x = inst_points[node_idx][0] * scale
                    y = inst_points[node_idx][1] * scale
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
                x0 = inst_points[src_idx][0] * scale
                y0 = inst_points[src_idx][1] * scale
                x1 = inst_points[dst_idx][0] * scale
                y1 = inst_points[dst_idx][1] * scale
                _raster_pafs(pafs[frame_idx], c * 2, x0, y0, x1, y1, sigma)

    return pafs

def _get_conf_ball(output_size, sigma):
    # Pre-allocate coordinate grid
    xv = np.linspace(0, output_size[1] - 1, output_size[1], dtype="float32")
    yv = np.linspace(0, output_size[0] - 1, output_size[0], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    x, y = output_size[1]//2, output_size[0]//2
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

def _to_np_point(x, y) -> np.ndarray:
    a = np.array((x, y))
    return np.expand_dims(a, axis=0)

def _centroid(x0, y0, x1, y1) -> np.ndarray:
    return _to_np_point(x = x0+(x1-x0)/2, y = y0+(y1-y0)/2)

def instance_crops(
            imgs: np.ndarray,
            points: list,
            min_crop_size: int=0,
            negative_samples: int=0) -> Tuple[np.ndarray, List]:
    """
    Take imgs, points and return imgs, points cropped around instances.

    Note that if there are multiple instances in a image, this will result in more
    (but smaller) images than we started with.

    Includes crop for each instance, and optionally will also include random
    "negative sample" crops that aren't centered on an instance.

    Args:
        imgs: output from generate_images()
        points: output from generate_points()
        min_crop_size: int, the minimum crop square size
        negative_samples: int, number of *random* negative samples to include
    Returns:
        imgs, points (matching format of input)
    """
    img_shape = imgs.shape[1], imgs.shape[2]

    # List of bounding box for every instance, map from list idx -> frame idx
    bbs, img_idxs = _bbs_from_points(points)
    bbs = _pad_bbs_to_min(bbs, min_crop_size, img_shape)

    # Crop images and combine/translate points
    crop_imgs, crop_points = _crop_and_transform(imgs, points, img_idxs, bbs)

    # Add bounding boxes for *random* negative samples
    if negative_samples > 0:
        neg_img_idxs, neg_bbs = get_random_negative_samples(img_idxs, bbs, img_shape, negative_samples)
        neg_imgs, neg_points = _crop_and_transform(imgs, points, neg_img_idxs, neg_bbs)
        crop_imgs, crop_points = _extend_imgs_points(crop_imgs, crop_points, neg_imgs, neg_points)

    return crop_imgs, crop_points

def _crop_and_transform(imgs, points, img_idxs, bbs):
    crop_imgs = _crop(imgs, img_idxs, bbs)
    crop_points = _transform_crop_points(points, img_idxs, bbs)
    return crop_imgs, crop_points

def _extend_imgs_points(imgs_a, points_a, imgs_b, points_b):
    imgs = np.concatenate((imgs_a, imgs_b))
    points = points_a + points_b
    return imgs, points

def _pad_bbs_to_min(bbs, min_crop_size, img_shape):
    padded_bbs = _pad_bbs(
                    bbs = bbs,
                    box_shape = _bb_pad_shape(bbs, min_crop_size, img_shape),
                    img_shape = img_shape)
    return padded_bbs

def _bb_pad_shape(bbs, min_crop_size, img_shape):
    """
    Given a list of bounding boxes, finds the square size which will be:

        1. large enough to contain every bounding box
        2. no larger than the image
        3. at least as large as the minimum crop size

    TODO: what should we do when these can't all be satisfied?

    Returns:
        (size, size) tuple
    """
    # Find a nicely sized box that's large enough to bound all instances
    max_height = max((y1 - y0 for (x0, y0, x1, y1) in bbs))
    max_width = max((x1 - x0 for (x0, y0, x1, y1) in bbs))
    max_dim = max(max_height, max_width)
    max_dim = max(max_dim, min_crop_size)
    max_dim += 20 # pad
    box_side = ceil(max_dim/64)*64 # round up to nearest multiple of 64

    # TODO: make sure we have valid box_size

    # Grow all bounding boxes to the same size
    box_shape = min(box_side, img_shape[0]), min(box_side, img_shape[1])

    return box_shape

def _transform_crop_points(points, img_idxs, bbs):
    """Takes points on the original images and returns points in bounding boxes.

    The input points will be per frame, the output points will be per box;
    we use img_idxs to figure out which bounding box goes with which frame.

    The points are translated from original img coordinates to bb coordinates.

    Args:
        points: list (per img) of list (per inst) of points_array matrices
            i.e., output from generate_point()
        img_idxs: list mapping bb idx -> original img idx
        bbs: list of (x0, y0, x1, y1) tuples
            img_idxs, bbs = output from _bbs_from_points()
    Returns:
        new points list, per bounding box instead of per original img
    """
    # Make point arrays for each image (instead of each frame as before)

    # Note that we want all points from the frame, not just the points for the instance
    # around which we're cropping (i.e., point_array in frame_points).
    crop_points = list(map(lambda i: points[i], img_idxs))

    # translate points to location w/in cropped image
    crop_points = [_translate_points_array(points_array, bbs[i][0], bbs[i][1])
                    for i, points_array in enumerate(crop_points)]

    return crop_points

def _translate_points_array(points_array, x, y):
    if len(points_array) == 0: return points_array
    return points_array - np.asarray([x,y])

def merge_boxes(box_a, box_b):
    """Return a box that contains both boxes."""

    a_x1, a_y1, a_x2, a_y2 = box_a
    b_x1, b_y1, b_x2, b_y2 = box_b

    c_x1, c_y1 = min(a_x1, b_x1), min(a_y1, b_y1)
    c_x2, c_y2 = max(a_x2, b_x2), max(a_y2, b_y2)

    return (c_x1, c_y1, c_x2, c_y2)

def merge_boxes_with_overlap(boxes):
    """Return a list of boxes after merging any overlapping boxes."""

    if len(boxes) < 2: return boxes

    first_box = boxes[0]
    other_boxes = boxes[1:]

    keep_boxes = []

    # Check first box against each of the other boxes
    for box in other_boxes:
        if box_overlap_area(first_box, box) > 0:
            first_box = merge_boxes(first_box, box)
        else:
            keep_boxes.append(box)

    # Now filter the other boxes for overlap amongst themselves
    other_boxes = merge_boxes_with_overlap(keep_boxes)

    return [first_box] + other_boxes

def merge_boxes_with_overlap_and_padding(boxes, pad_factor_box, within):
    """
    Returns a list of boxes after merging any overlapping boxes
    and padding each box to a size multiple of pad_factor_box.
    """

    # Pad then merge
    padded_boxes = [pad_box_to_multiple(box, pad_factor_box, within) for box in boxes]
    merged_boxes = merge_boxes_with_overlap(padded_boxes)

    # If we merged any boxes, then do this again until we don't merge
    if len(merged_boxes) == len(boxes):
        return merged_boxes
    else:
        return merge_boxes_with_overlap_and_padding(merged_boxes, pad_factor_box, within)

def pad_box_to_multiple(box, pad_factor_box, within):

    box_h = box[3] - box[1] # difference in y
    box_w = box[2] - box[0] # difference in x

    pad_h, pad_w = pad_factor_box

    # Find multiple of pad_factor_box that's large enough to hold box
    multiple_h, multiple_w = ceil(box_h / pad_h), ceil(box_w / pad_w)

    # Maintain aspect ratio
    multiple = max(multiple_h, multiple_w)

    # Return padded box
    return pad_rect_to(*box, (pad_h*multiple, pad_w*multiple), within)

def bounding_box_nms(boxes, scores, iou_threshold):
    """
    Suppress bounding boxes which overlap more than iou_threshold.

    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        boxes: (boxes * 4)-size ndarray matrix
        scores: (boxes)-length vector, uses to determine which overlapping
            box to suppress
        iou_threshold: intersection over union value to use as threshold
    Returns:
        list of (x0, y0, x1, y1) ndarrays for non-suppressed boxes
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if type(boxes) == list:
        boxes = np.stack(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes, sort by scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_threshold)[0])))

    # return the list of picked boxes
    return pick

def negative_anchor_crops(
            labels: Labels,
            negative_anchors: Dict['Video', Dict[int, Tuple]],
            scale, crop_size) -> Tuple[np.ndarray, List]:
    """
    Returns crops around *specific* negative samples from Labels object.

    Args:
        labels: the `Labels` object
        scale: scale, should match scale given to generate_images()
        crop_size: the size of the crops returned by instance_crops()
    Returns:
        imgs, points
            These match output from instance_crops(),
            and can be combined using _extend_crop_data().
    """

    # negative_anchors[video]: (frame_idx, x, y) for center of crop

    neg_anchor_tuples = [(video, frame_idx, x, y)
                    for video in negative_anchors
                    for (frame_idx, x, y) in negative_anchors[video]]

    if len(neg_anchor_tuples) == 0: return None, None

    frame_list = [(video, frame_idx)
                    for (video, frame_idx, x, y) in neg_anchor_tuples]
    anchors = [[_to_np_point(x,y)]
                    for (video, frame_idx, x, y) in neg_anchor_tuples]

    imgs = generate_images_from_list(labels, frame_list, scale)
    points = generate_points_from_list(labels, frame_list, scale)

    # List of bounding box for every instance, map from list idx -> frame idx
    bbs, img_idxs = _bbs_from_points(anchors)

    # Grow all bounding boxes to the same size
    bbs = _pad_bbs(bbs, (crop_size, crop_size), (imgs.shape[1], imgs.shape[2]))

    # Crop images and combine/translate points
    crop_imgs, crop_points = _crop_and_transform(imgs, points, img_idxs, bbs)

    return crop_imgs, crop_points

def add_negative_anchor_crops(labels: Labels, imgs: np.ndarray, points: list, scale: float) -> Tuple[np.ndarray, List]:
    """Wrapper to build and append negative anchor crops."""
    # Include any *specific* negative samples
    neg_imgs, neg_points = negative_anchor_crops(
                                labels,
                                labels.negative_anchors,
                                scale=scale,
                                crop_size=imgs.shape[1])

    if neg_imgs is not None:
        imgs, points = _extend_imgs_points(imgs, points, neg_imgs, neg_points)

    return imgs, points

def get_random_negative_samples(img_idxs, bbs, img_shape, negative_samples):
    if len(bbs) == 0: return

    frame_count = len({frame for frame in img_idxs})
    box_side = bbs[0][2] - bbs[0][0] # x1 - x0 for the first bb

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
        area_covered = sum(map(lambda bb: box_overlap_area(sample_bb, bb), frame_bbs))/(box_side**2)

        # append negative sample to lists
        neg_sample_list.append((area_covered, sample_img_idx, sample_bb))

    # Pick the best samples
    neg_sample_list.sort(key=itemgetter(0))
    _, neg_img_idxs, neg_bbs = zip(*neg_sample_list)

    return neg_img_idxs[:negative_samples], neg_bbs[:negative_samples]

def _bbs_from_points(points):
    # List of bounding box for every instance
    bbs = [point_array_bounding_box(point_array) for frame in points for point_array in frame]
    bbs = [(int(x0), int(y0), int(x1), int(y1)) for (x0, y0, x1, y1) in bbs]

    # List to map bb to its img frame idx
    img_idxs = [i for i, frame in enumerate(points) for _ in frame]

    return bbs, img_idxs

def box_overlap_area(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return inter_area

def _pad_bbs(bbs, box_shape, img_shape):
    return list(map(lambda bb: pad_rect_to(*bb, box_shape, img_shape), bbs))

def _crop(imgs, img_idxs, bbs) -> np.ndarray:
    imgs = [imgs[img_idxs[i], bb[1]:bb[3], bb[0]:bb[2]] for i, bb in enumerate(bbs)] # imgs[frame_idx, y0:y1, x0:x1]
    imgs = np.stack(imgs, axis=0)
    return imgs

def fullsize_points_from_crop(idx: int, point_array: np.ndarray,
                              bbs: list, img_idxs: list):
    """
    Map point within crop back to original image frames.

    Args:
        idx: index in imgs stack
        point_array: (x, y) for each node, ndarray with shape (nodes, 2)
        bbs: list idx -> bounding box
        img_idxs: list idx -> frame_idx
    Returns:
        frame_idx, point_array
    """
    bb = bbs[idx]

    top_left_point = ((bb[0], bb[1]),) # for (x, y) column vector
    point_array += np.array(top_left_point)

    frame_idx = img_idxs[idx]

    return frame_idx, point_array

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
    # testing
    labels.negative_anchors = {labels.videos[0]: [(0, 125, 125), (0, 150, 150)]}
    # labels.labeled_frames = labels.labeled_frames[123:423:10]

    scale = 1

    imgs, points = generate_training_data(
                        labels = labels,
                        params = dict(
                                    scale = scale,
                                    instance_crop = True,
                                    min_crop_size = 0,
                                    negative_samples = 0))

    print("--imgs--")
    print(imgs.shape)
    print(imgs.dtype)
    print(np.ptp(imgs))

    # Prepare GUI demo

    from PySide2 import QtWidgets
    from sleap.io.video import Video
    from sleap.gui.overlays.confmaps import demo_confmaps
    from sleap.gui.overlays.pafs import demo_pafs

    app = QtWidgets.QApplication([])

    # Centroids

    # generate centoid data on full frames before instance cropping
    # centroid_imgs = generate_images(labels, scale=.25)
    # centroid_vid = Video.from_numpy(centroid_imgs * 255)
    # centroid_points = generate_centroid_points(generate_points(labels, scale=.25))
    # centroid_confmaps = generate_confmaps_from_points(centroid_points, None,
    #                             (centroid_imgs.shape[1], centroid_imgs.shape[2]),
    #                             node_count=1, sigma=5.0)

    # demo_confmaps(centroid_confmaps, centroid_vid)

    vid = Video.from_numpy(imgs * 255)

    skeleton = labels.skeletons[0]
    img_shape = (imgs.shape[1], imgs.shape[2])

    confmaps = generate_confmaps_from_points(points, skeleton, img_shape, scale=.5, sigma=5.0*scale)
    print("--confmaps--")
    print(confmaps.shape)
    print(confmaps.dtype)
    print(np.ptp(confmaps))

    demo_confmaps(confmaps, vid)

    pafs = generate_pafs_from_points(points, skeleton, img_shape, scale=.5, sigma=5.0*scale)
    print("--pafs--")
    print(pafs.shape)
    print(pafs.dtype)
    print(np.ptp(pafs))

    demo_pafs(pafs, vid)

    app.exec_()

if __name__ == "__main__":
    demo_datagen()
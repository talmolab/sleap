"""This module contains generic utilities used for training and inference."""

import tensorflow as tf
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment


def group_array(
    X: np.ndarray, groups: np.ndarray, axis: int = 0
) -> Dict[np.ndarray, np.ndarray]:
    """Groups an array into a dictionary keyed by a grouping vector.

    Args:
        X: Numpy array with length n along the specified axis.
        groups: Vector of n values denoting the group that each slice of X should be
            assigned to. This is also referred to as an indicator, indexing, class,
            or labels vector.
        axis: Dimension of X to group on. The length of this axis in X must correspond
            to the length of groups.

    Returns:
        A dictionary with keys mapping each unique value in groups to a subset of X.

    References:
        See this `blog post
        <https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/>`
        for performance comparisons of different approaches.

    Example: ::

        >>> group_array(np.arange(5), np.array([1, 5, 2, 1, 5]))
        {1: array([0, 3]), 5: array([1, 4]), 2: array([2])}

    """

    group_inds = defaultdict(list)
    for ind, key in enumerate(groups):
        group_inds[key].append(ind)

    return {key: np.take(X, inds, axis=axis) for key, inds in group_inds.items()}


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Computes the intersection over union for a pair of bounding boxes.

    Args:
        bbox1: Bounding box specified by corner coordinates [y1, x1, y2, x2].
        bbox2: Bounding box specified by corner coordinates [y1, x1, y2, x2].

    Returns:
        A float scalar calculated as the ratio between the areas of the intersection
        and the union of the two bounding boxes.
    """

    bbox1_y1, bbox1_x1, bbox1_y2, bbox1_x2 = bbox1
    bbox2_y1, bbox2_x1, bbox2_y2, bbox2_x2 = bbox2

    intersection_y1 = max(bbox1_y1, bbox2_y1)
    intersection_x1 = max(bbox1_x1, bbox2_x1)
    intersection_y2 = min(bbox1_y2, bbox2_y2)
    intersection_x2 = min(bbox1_x2, bbox2_x2)

    intersection_area = max(intersection_x2 - intersection_x1 + 1, 0) * max(
        intersection_y2 - intersection_y1 + 1, 0
    )

    bbox1_area = (bbox1_x2 - bbox1_x1 + 1) * (bbox1_y2 - bbox1_y1 + 1)
    bbox2_area = (bbox2_x2 - bbox2_x1 + 1) * (bbox2_y2 - bbox2_y1 + 1)

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area

    return iou


@tf.function
def tf_linear_sum_assignment(cost_matrix: tf.Tensor) -> tf.Tensor:
    """Run `linear_sum_assignment` as a TensorFlow function.

    Args:
        cost_matrix: Cost matrix of shape `(n_src, n_dst)`. Make sure to replace `NaN`s
            with `np.inf`.

    Returns:
        A tuple of `(row, col)` with the indices of the optimal assignments.
    """

    def _linear_sum_assignment(cost_matrix):
        """Wrap `linear_sum_assignment` for type safety."""
        row, col = linear_sum_assignment(cost_matrix)
        return row.astype("int32"), col.astype("int32")

    return tf.numpy_function(
        func=_linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int32, tf.int32]
    )


def match_points(points1: tf.Tensor, points2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Match closest points across two sets.

    Args:
        points1: A set of reference points of shape `(N, D)`.
        points2: A set of unmatched points of shape `(M, D)`.

    Returns:
        A tuple of `(inds1, inds2)` with the indices of matching points across the two
        lists of points.

        `inds1` is a `tf.int32` tensor of shape `(n,)` where `n <= N` that indexes into
        `points1`.

        `inds2` is a `tf.int32` tensor of shape `(m,)` where `m <= M` that indexes into
        `points2`.

    Notes:
        This uses Euclidean distance and the Hungarian algorithm for matching.

    See also: tf_linear_sum_assignment
    """
    dists = tf.linalg.norm(
        tf.expand_dims(tf.cast(points1, tf.float32), axis=1)
        - tf.expand_dims(tf.cast(points2, tf.float32), axis=0),
        axis=-1,
    )
    return tf_linear_sum_assignment(dists)

from numpy.testing import assert_array_equal, assert_allclose
import numpy as np
import tensorflow as tf

import sleap
from sleap.nn.identity import (
    group_class_peaks,
    classify_peaks_from_maps,
)


sleap.use_cpu_only()


def test_group_class_peaks():
    peak_class_probs = np.array(
        [
            [0.1, 0.9],
            [0.9, 0.1],
            [0.95, 0.05],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.1, 0.9],
        ]
    )
    peak_sample_inds = np.array([0, 0, 0, 0, 1, 1, 1])
    peak_channel_inds = np.array([0, 0, 1, 1, 0, 0, 0])
    peak_inds, class_inds = group_class_peaks(
        peak_class_probs,
        peak_sample_inds,
        peak_channel_inds,
        n_samples=2,
        n_channels=2,
    )

    assert_array_equal(peak_inds, [0, 1, 2, 4, 6])
    assert_array_equal(class_inds, [1, 0, 0, 0, 1])


def test_classify_peaks_from_maps():
    peak_class_probs = np.array(
        [
            [0.1, 0.9],
            [0.91, 0.09],
            [0.95, 0.05],
            [0.8, 0.2],
            [0.92, 0.08],
            [0.85, 0.15],
            [0.07, 0.93],
        ]
    )
    peak_sample_inds = np.array([0, 0, 0, 0, 1, 1, 1])
    peak_channel_inds = np.array([0, 0, 1, 1, 0, 0, 0])
    peak_points = tf.reshape(tf.range(7 * 2, dtype=tf.float32), [7, 2])
    peak_vals = tf.ones([7], tf.float32)

    class_maps = np.zeros([2, 14, 14, 2], dtype="float32")
    for s, (x, y), pr in zip(peak_sample_inds, peak_points, peak_class_probs):
        class_maps[s, int(y), int(x), :] = pr
    class_maps = tf.cast(class_maps, tf.float32)

    points, point_vals, class_probs = classify_peaks_from_maps(
        class_maps,
        peak_points,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
        n_channels=2,
    )

    assert_array_equal(points[0][0], peak_points.numpy()[[1, 2]])
    assert_array_equal(points[0][1], [peak_points.numpy()[0], [np.nan, np.nan]])
    assert_array_equal(points[1][0], [peak_points.numpy()[4], [np.nan, np.nan]])
    assert_array_equal(points[1][1], [peak_points.numpy()[6], [np.nan, np.nan]])

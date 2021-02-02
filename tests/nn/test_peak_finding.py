import pytest
import numpy as np
import tensorflow as tf
import sleap
from numpy.testing import assert_array_equal, assert_allclose

from sleap.nn.peak_finding import (
    find_offsets_local_direction,
    find_global_peaks_rough,
    find_local_peaks_rough,
    find_global_peaks,
    find_local_peaks,
    find_global_peaks_integral,
    find_local_peaks_integral,
    find_global_peaks_with_offsets,
    find_local_peaks_with_offsets,
)
from sleap.nn.data.confidence_maps import (
    make_confmaps,
    make_grid_vectors,
    make_multi_confmaps,
)


sleap.nn.system.use_cpu_only()


def test_find_local_offsets():
    offsets = find_offsets_local_direction(
        np.array([[0.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 1.0, 0.0]]).reshape(
            1, 3, 3, 1
        ),
        0.25,
    )
    assert tuple(offsets.shape) == (1, 2)
    assert offsets[0][0] == 0.25
    assert offsets[0][1] == 0.0

    offsets = find_offsets_local_direction(
        np.array([[0.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 0.0]]).reshape(
            1, 3, 3, 1
        ),
        0.25,
    )
    assert offsets[0][0] == 0.0
    assert offsets[0][1] == 0.0


def test_find_global_peaks_rough():
    xv, yv = make_grid_vectors(image_height=8, image_width=8, output_stride=1)
    points = tf.cast([[1, 2], [3, 4], [5, 6]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)
    points2 = points + 1
    cms = tf.stack([cm, make_confmaps(points2, xv, yv, sigma=1.0)])

    peaks, peak_vals = find_global_peaks(cms, threshold=0.1, refinement=None)

    assert peaks.shape == (2, 3, 2)
    assert peak_vals.shape == (2, 3)
    assert_array_equal(peaks[0], points)
    assert_array_equal(peak_vals[0], [1, 1, 1])
    assert_array_equal(peaks[1], points2)
    assert_array_equal(peak_vals[1], [1, 1, 1])

    peaks, peak_vals = find_global_peaks_rough(
        tf.zeros((1, 8, 8, 3), dtype=tf.float32), threshold=0.1
    )
    assert peaks.shape == (1, 3, 2)
    assert peak_vals.shape == (1, 3)
    assert tf.reduce_all(tf.math.is_nan(peaks))
    assert_array_equal(peak_vals, [[0, 0, 0]])


def test_find_global_peaks_integral():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)

    peaks, peak_vals = find_global_peaks(
        tf.expand_dims(cm, axis=0),
        threshold=0.1,
        refinement="integral",
        integral_patch_size=5,
    )

    assert peaks.shape == (1, 3, 2)
    assert peak_vals.shape == (1, 3)
    assert_allclose(peaks[0].numpy(), points.numpy(), atol=0.1)
    assert_allclose(peak_vals[0].numpy(), [1, 1, 1], atol=0.3)

    peaks, peak_vals = find_global_peaks(
        tf.zeros((1, 8, 8, 3), dtype=tf.float32),
        threshold=0.1,
        refinement="integral",
        integral_patch_size=5,
    )
    assert peaks.shape == (1, 3, 2)
    assert peak_vals.shape == (1, 3)
    assert tf.reduce_all(tf.math.is_nan(peaks))
    assert_array_equal(peak_vals, [[0, 0, 0]])

    peaks, peak_vals = find_global_peaks(
        tf.stack([tf.zeros([12, 12, 3], dtype=tf.float32), cm], axis=0),
        threshold=0.1,
        refinement="integral",
        integral_patch_size=5,
    )
    assert peaks.shape == (2, 3, 2)
    assert tf.reduce_all(tf.math.is_nan(peaks[0]))
    assert_allclose(peaks[1].numpy(), points.numpy(), atol=0.1)

    peaks, peak_vals = find_global_peaks_integral(
        tf.stack([tf.zeros([12, 12, 3], dtype=tf.float32), cm], axis=0),
        threshold=0.1,
        crop_size=5,
    )
    assert peaks.shape == (2, 3, 2)
    assert tf.reduce_all(tf.math.is_nan(peaks[0]))
    assert_allclose(peaks[1].numpy(), points.numpy(), atol=0.1)


def test_find_global_peaks_local():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[1.6, 2.6], [3.6, 4.6], [5.6, 6.6]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)

    peaks, peak_vals = find_global_peaks(
        tf.expand_dims(cm, axis=0), threshold=0.1, refinement="local"
    )

    assert peaks.shape == (1, 3, 2)
    assert peak_vals.shape == (1, 3)
    assert_allclose(
        peaks[0].numpy(), np.array([[1.75, 2.75], [3.75, 4.75], [5.75, 6.75]])
    )
    assert_allclose(peak_vals[0].numpy(), [1, 1, 1], atol=0.3)


def test_find_local_peaks_rough():
    xv, yv = make_grid_vectors(image_height=16, image_width=16, output_stride=1)
    instances = tf.cast(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[np.nan, np.nan], [11, 12]],
        ],
        tf.float32,
    )
    cms = make_multi_confmaps(instances, xv=xv, yv=yv, sigma=1.0)
    instances2 = tf.cast([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], tf.float32)
    cms = tf.stack(
        [cms, make_multi_confmaps(instances2, xv=xv, yv=yv, sigma=1.0)], axis=0
    )

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        cms, threshold=0.1, refinement=None
    )

    assert peak_points.shape == (9, 2)
    assert peak_vals.shape == (9,)
    assert peak_sample_inds.shape == (9,)
    assert peak_channel_inds.shape == (9,)

    assert_array_equal(
        peak_points.numpy(),
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [11, 12],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ],
    )
    assert_array_equal(peak_vals, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_array_equal(peak_sample_inds, [0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(peak_channel_inds, [0, 1, 0, 1, 1, 0, 1, 0, 1])

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        tf.zeros([1, 4, 4, 3], tf.float32), threshold=0.1, refinement=None
    )
    assert peak_points.shape == (0, 2)
    assert peak_vals.shape == (0,)
    assert peak_sample_inds.shape == (0,)
    assert peak_channel_inds.shape == (0,)


def test_find_local_peaks_integral():
    xv, yv = make_grid_vectors(image_height=32, image_width=32, output_stride=1)
    instances = (
        tf.cast(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[np.nan, np.nan], [11, 12]],
            ],
            tf.float32,
        )
        * 2
        + 0.3
    )
    cms = make_multi_confmaps(instances, xv=xv, yv=yv, sigma=1.0)
    instances2 = tf.cast([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], tf.float32) * 2 + 0.3
    cms = tf.stack(
        [cms, make_multi_confmaps(instances2, xv=xv, yv=yv, sigma=1.0)], axis=0
    )

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        cms, threshold=0.1, refinement="integral", integral_patch_size=5
    )

    assert peak_points.shape == (9, 2)
    assert peak_vals.shape == (9,)
    assert peak_sample_inds.shape == (9,)
    assert peak_channel_inds.shape == (9,)

    assert_allclose(
        peak_points.numpy(),
        np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [11, 12],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ]
        )
        * 2
        + 0.3,
        atol=0.2,
    )
    assert_allclose(peak_vals, [1, 1, 1, 1, 1, 1, 1, 1, 1], atol=0.1)
    assert_array_equal(peak_sample_inds, [0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(peak_channel_inds, [0, 1, 0, 1, 1, 0, 1, 0, 1])

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        tf.zeros([1, 4, 4, 3], tf.float32), refinement="integral", integral_patch_size=5
    )
    assert peak_points.shape == (0, 2)
    assert peak_vals.shape == (0,)
    assert peak_sample_inds.shape == (0,)
    assert peak_channel_inds.shape == (0,)

    (
        peak_points,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_integral(tf.zeros([1, 4, 4, 3], tf.float32), crop_size=5)
    assert peak_points.shape == (0, 2)
    assert peak_vals.shape == (0,)
    assert peak_sample_inds.shape == (0,)
    assert peak_channel_inds.shape == (0,)


def test_find_local_peaks_local():
    xv, yv = make_grid_vectors(image_height=32, image_width=32, output_stride=1)
    instances = (
        tf.cast(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[np.nan, np.nan], [11, 12]],
            ],
            tf.float32,
        )
        * 2
        + 0.25
    )
    cms = make_multi_confmaps(instances, xv=xv, yv=yv, sigma=1.0)
    instances2 = tf.cast([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], tf.float32) * 2 + 0.25
    cms = tf.stack(
        [cms, make_multi_confmaps(instances2, xv=xv, yv=yv, sigma=1.0)], axis=0
    )

    peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(
        cms, threshold=0.1, refinement="local"
    )

    assert peak_points.shape == (9, 2)
    assert peak_vals.shape == (9,)
    assert peak_sample_inds.shape == (9,)
    assert peak_channel_inds.shape == (9,)

    assert_allclose(
        peak_points.numpy(),
        np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [11, 12],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ]
        )
        * 2
        + 0.25,
    )
    assert_allclose(peak_vals, [1, 1, 1, 1, 1, 1, 1, 1, 1], atol=0.1)
    assert_array_equal(peak_sample_inds, [0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(peak_channel_inds, [0, 1, 0, 1, 1, 0, 1, 0, 1])


@pytest.mark.parametrize("output_stride", [1, 2])
def test_find_global_peaks_with_offsets(output_stride, min_labels):
    p = min_labels.to_pipeline()
    p += sleap.pipelines.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=min_labels.skeletons,
    )
    p += sleap.pipelines.InstanceCropper(crop_width=192, crop_height=192)
    p += sleap.pipelines.InstanceConfidenceMapGenerator(
        sigma=1.5, output_stride=output_stride, all_instances=False, with_offsets=True
    )
    p += sleap.pipelines.Batcher(
        batch_size=2,
    )

    ex = p.peek()
    cms = ex["instance_confidence_maps"]
    offs = ex["offsets"]

    refined_peaks, peak_vals = find_global_peaks_with_offsets(cms, offs)
    refined_peaks *= output_stride

    assert_allclose(ex["center_instance"], refined_peaks, atol=1e-3)


@pytest.mark.parametrize("output_stride", [1, 2])
def test_find_local_peaks_with_offsets(output_stride, min_labels):
    p = min_labels.to_pipeline()
    p += sleap.pipelines.MultiConfidenceMapGenerator(
        sigma=1.5, output_stride=output_stride, centroids=False, with_offsets=True
    )
    p += sleap.pipelines.Batcher(batch_size=2)

    ex = p.peek()
    cms = ex["confidence_maps"]
    offs = ex["offsets"]

    (
        refined_peaks,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_with_offsets(cms, offs)
    refined_peaks *= output_stride

    peaks_gt = tf.reshape(ex["instances"], [-1, 2])
    inds1, inds2 = sleap.nn.utils.match_points(peaks_gt, refined_peaks)
    peaks_gt = tf.gather(peaks_gt, inds1)
    refined_peaks = tf.gather(refined_peaks, inds2)

    assert_allclose(peaks_gt, refined_peaks, atol=1e-3)

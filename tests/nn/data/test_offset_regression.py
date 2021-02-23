import numpy as np
import tensorflow as tf
import sleap
from sleap.nn.data import offset_regression


sleap.use_cpu_only()  # hide GPUs for test


def test_make_offsets():
    points = np.array([[1.8, 2.3], [0.4, 3.1], [np.nan, np.nan]], "float32")
    xv, yv = sleap.nn.data.confidence_maps.make_grid_vectors(4, 4, output_stride=1)
    off = offset_regression.make_offsets(points, xv, yv, stride=1)

    assert off.shape == (4, 4, 3, 2)

    XX, YY = tf.meshgrid(xv, yv)
    np.testing.assert_allclose(XX + off[..., 0, 0], 1.8, atol=1e-6)
    np.testing.assert_allclose(YY + off[..., 0, 1], 2.3, atol=1e-6)
    np.testing.assert_allclose(XX + off[..., 1, 0], 0.4, atol=1e-6)
    np.testing.assert_allclose(YY + off[..., 1, 1], 3.1, atol=1e-6)
    np.testing.assert_array_equal(off[..., 2, :], 0)


def test_mask_offsets():
    points = np.array([[1.0, 1.0]], "float32")
    xv, yv = sleap.nn.data.confidence_maps.make_grid_vectors(4, 4, output_stride=1)
    off = offset_regression.make_offsets(points, xv, yv, stride=1)
    cm = sleap.nn.data.confidence_maps.make_confmaps(points, xv, yv, sigma=1)
    off_mask = offset_regression.mask_offsets(off, cm, threshold=0.2)

    np.testing.assert_array_equal(off_mask[:3, :3], off[:3, :3])
    np.testing.assert_array_equal(off_mask[3:, :], 0.0)
    np.testing.assert_array_equal(off_mask[:, 3:], 0.0)

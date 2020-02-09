import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import utils


def test_ensure_list():
    assert utils.ensure_list([0, 1, 2]) == [0, 1, 2]
    assert utils.ensure_list(0) == [0]
    assert utils.ensure_list([0]) == [0]


def test_expand_to_rank():
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=2, prepend=True), [[0, 1, 2]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=3, prepend=True), [[[0, 1, 2]]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=2, prepend=False), [[0], [1], [2]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(
            tf.reshape(tf.range(3), [1, 3]), target_rank=2, prepend=True
        ),
        [[0, 1, 2]],
    )



def test_make_grid_vector():
    xv, yv = utils.make_grid_vectors(image_height=4, image_width=3, output_stride=1)

    assert xv.dtype == tf.float32
    assert xv.shape == (3,)
    assert yv.dtype == tf.float32
    assert yv.shape == (4,)

    np.testing.assert_allclose(xv, [0, 1, 2])
    np.testing.assert_allclose(yv, [0, 1, 2, 3])

    xv, yv = utils.make_grid_vectors(image_height=4, image_width=3, output_stride=2)
    np.testing.assert_allclose(xv, [0, 2])
    np.testing.assert_allclose(yv, [0, 2])

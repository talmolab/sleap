import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import resizing


def test_find_padding_for_stride():
    assert resizing.find_padding_for_stride(
        image_height=127, image_width=129, max_stride=32) == (1, 31)
    assert resizing.find_padding_for_stride(
        image_height=128, image_width=128, max_stride=32) == (0, 0)


def test_pad_to_stride():
    np.testing.assert_array_equal(
        resizing.pad_to_stride(tf.ones([3, 5, 1]), max_stride=2),
        tf.expand_dims([
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]], axis=-1)
        )
    assert resizing.pad_to_stride(
        tf.ones([3, 5, 1], dtype=tf.uint8), max_stride=2).dtype == tf.uint8
    assert resizing.pad_to_stride(
        tf.ones([3, 5, 1], dtype=tf.float32), max_stride=2).dtype == tf.float32
    assert resizing.pad_to_stride(
        tf.ones([4, 4, 1]), max_stride=2).shape == (4, 4, 1)


def test_resize_image():
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.uint8), scale=[0.25, 3]).shape == (12, 2, 1)
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.uint8), scale=0.5).shape == (2, 4, 1)
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.uint8), scale=0.5).dtype == tf.uint8
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.float32), scale=0.5).dtype == tf.float32

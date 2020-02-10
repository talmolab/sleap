import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import resizing
from sleap.nn.data import providers


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


def test_resizer(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    ds_data = labels_reader.make_dataset()

    resizer = resizing.Resizer(image_key="image", scale=0.25)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (96, 96, 1)

    resizer = resizing.Resizer(image_key="image", pad_to_stride=100)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (400, 400, 1)

    resizer = resizing.Resizer(image_key="image", scale=0.25, pad_to_stride=100)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (100, 100, 1)

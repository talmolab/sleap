import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import normalization
from sleap.nn.data import providers


def test_ensure_min_image_rank():
    assert normalization.ensure_min_image_rank(tf.zeros([2, 2])).shape == (2, 2, 1)
    assert normalization.ensure_min_image_rank(tf.zeros([2, 2, 1])).shape == (2, 2, 1)


def test_ensure_float():
    assert normalization.ensure_float(tf.zeros([2, 2], tf.uint8)).dtype == tf.float32
    assert normalization.ensure_float(tf.zeros([2, 2], tf.float32)).dtype == tf.float32


def test_ensure_int():
    np.testing.assert_array_equal(
        normalization.ensure_int(tf.constant([0.0, 0.5, 1.0])), np.array([0, 127, 255])
    )

    np.testing.assert_array_equal(
        normalization.ensure_int(tf.constant([0.0, 127.0, 255.0])),
        np.array([0, 127, 255]),
    )

    np.testing.assert_array_equal(
        normalization.ensure_int(tf.constant([0, 127, 255])), np.array([0, 127, 255])
    )


def test_ensure_grayscale():
    np.testing.assert_array_equal(
        normalization.ensure_grayscale(tf.ones([2, 2, 3], tf.uint8) * 255),
        tf.ones([2, 2, 1], tf.uint8) * 255,
    )
    np.testing.assert_array_equal(
        normalization.ensure_grayscale(tf.ones([2, 2, 1], tf.uint8) * 255),
        tf.ones([2, 2, 1], tf.uint8) * 255,
    )
    np.testing.assert_allclose(
        normalization.ensure_grayscale(tf.ones([2, 2, 3], tf.float32)),
        tf.ones([2, 2, 1], tf.float32),
        atol=1e-4,
    )


def test_ensure_rgb():
    np.testing.assert_array_equal(
        normalization.ensure_rgb(tf.ones([2, 2, 3], tf.uint8) * 255),
        tf.ones([2, 2, 3], tf.uint8) * 255,
    )
    np.testing.assert_array_equal(
        normalization.ensure_rgb(tf.ones([2, 2, 1], tf.uint8) * 255),
        tf.ones([2, 2, 3], tf.uint8) * 255,
    )


def test_convert_rgb_to_bgr():
    img_rgb = tf.stack(
        [
            tf.ones([2, 2], dtype=tf.uint8) * 1,
            tf.ones([2, 2], dtype=tf.uint8) * 2,
            tf.ones([2, 2], dtype=tf.uint8) * 3,
        ],
        axis=-1,
    )
    img_bgr = tf.stack(
        [
            tf.ones([2, 2], dtype=tf.uint8) * 3,
            tf.ones([2, 2], dtype=tf.uint8) * 2,
            tf.ones([2, 2], dtype=tf.uint8) * 1,
        ],
        axis=-1,
    )

    np.testing.assert_array_equal(normalization.convert_rgb_to_bgr(img_rgb), img_bgr)


def test_scale_image_range():
    np.testing.assert_array_equal(
        normalization.scale_image_range(
            tf.cast([0, 0.5, 1.0], tf.float32), min_val=-1.0, max_val=1.0
        ),
        [-1, 0, 1],
    )


def test_normalizer(min_labels):
    # tf.executing_eagerly()

    labels_reader = providers.LabelsReader(min_labels)
    ds_img = labels_reader.make_dataset()

    normalizer = normalization.Normalizer(ensure_grayscale=True)
    ds = normalizer.transform_dataset(ds_img)
    example = next(iter(ds))
    assert example["image"].shape[-1] == 1

    normalizer = normalization.Normalizer(ensure_float=True, ensure_grayscale=True)
    ds = normalizer.transform_dataset(ds_img)
    example = next(iter(ds))
    assert example["image"].dtype == tf.float32
    assert example["image"].shape[-1] == 1

    normalizer = normalization.Normalizer(ensure_float=True, ensure_rgb=True)
    ds = normalizer.transform_dataset(ds_img)
    example = next(iter(ds))
    assert example["image"].dtype == tf.float32
    assert example["image"].shape[-1] == 3

    normalizer = normalization.Normalizer(ensure_grayscale=True, ensure_rgb=True)
    ds = normalizer.transform_dataset(ds_img)
    example = next(iter(ds))
    assert example["image"].shape[-1] == 1


def test_normalizer_from_config():
    normalizer = normalization.Normalizer.from_config(
        config=normalization.PreprocessingConfig(
            ensure_rgb=False, ensure_grayscale=False, imagenet_mode=None
        )
    )
    assert normalizer.image_key == "image"
    assert normalizer.ensure_float == True
    assert normalizer.ensure_rgb == False
    assert normalizer.ensure_grayscale == False
    assert normalizer.imagenet_mode is None

    normalizer = normalization.Normalizer.from_config(
        config=normalization.PreprocessingConfig(
            ensure_rgb=False, ensure_grayscale=False, imagenet_mode="tf"
        )
    )
    assert normalizer.image_key == "image"
    assert normalizer.ensure_float == True
    assert normalizer.ensure_rgb == False
    assert normalizer.ensure_grayscale == False
    assert normalizer.imagenet_mode is "tf"


def test_ensure_grayscale_from_provider(small_robot_mp4_vid):
    video = providers.VideoReader(
        video=small_robot_mp4_vid,
        example_indices=[0],
    )

    normalizer = normalization.Normalizer(image_key="image", ensure_grayscale=True)

    ds = video.make_dataset()
    ds = normalizer.transform_dataset(ds)
    example = next(iter(ds))

    assert example["image"].shape[-1] == 1


def test_ensure_rgb_from_provider(centered_pair_vid):
    video = providers.VideoReader(
        video=centered_pair_vid,
        example_indices=[0],
    )

    normalizer = normalization.Normalizer(image_key="image", ensure_rgb=True)

    ds = video.make_dataset()
    ds = normalizer.transform_dataset(ds)
    example = next(iter(ds))

    assert example["image"].shape[-1] == 3

import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

import sleap
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test
from sleap.nn.data import resizing
from sleap.nn.data import providers
from sleap.nn.data.resizing import SizeMatcher

from tests.fixtures.videos import TEST_H5_FILE, TEST_SMALL_ROBOT_MP4_FILE


def test_find_padding_for_stride():
    assert resizing.find_padding_for_stride(
        image_height=127, image_width=129, max_stride=32
    ) == (1, 31)
    assert resizing.find_padding_for_stride(
        image_height=128, image_width=128, max_stride=32
    ) == (0, 0)


def test_pad_to_stride():
    np.testing.assert_array_equal(
        resizing.pad_to_stride(tf.ones([3, 5, 1]), max_stride=2),
        tf.expand_dims(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            axis=-1,
        ),
    )
    assert (
        resizing.pad_to_stride(tf.ones([3, 5, 1], dtype=tf.uint8), max_stride=2).dtype
        == tf.uint8
    )
    assert (
        resizing.pad_to_stride(tf.ones([3, 5, 1], dtype=tf.float32), max_stride=2).dtype
        == tf.float32
    )
    assert resizing.pad_to_stride(tf.ones([4, 4, 1]), max_stride=2).shape == (4, 4, 1)


def test_resize_image():
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.uint8), scale=[0.25, 3]
    ).shape == (12, 2, 1)
    assert resizing.resize_image(
        tf.ones([4, 8, 1], dtype=tf.uint8), scale=0.5
    ).shape == (2, 4, 1)
    assert (
        resizing.resize_image(tf.ones([4, 8, 1], dtype=tf.uint8), scale=0.5).dtype
        == tf.uint8
    )
    assert (
        resizing.resize_image(tf.ones([4, 8, 1], dtype=tf.float32), scale=0.5).dtype
        == tf.float32
    )


def test_resizer(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    ds_data = labels_reader.make_dataset()
    data_example = next(iter(ds_data))

    resizer = resizing.Resizer(image_key="image", scale=0.25)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (96, 96, 1)
    np.testing.assert_array_equal(example["scale"], (0.25, 0.25))
    np.testing.assert_allclose(example["instances"], data_example["instances"] * 0.25)

    resizer = resizing.Resizer(image_key="image", pad_to_stride=100)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (400, 400, 1)
    np.testing.assert_array_equal(example["scale"], (1.0, 1.0))
    np.testing.assert_allclose(example["instances"], data_example["instances"])

    resizer = resizing.Resizer(image_key="image", scale=0.25, pad_to_stride=100)
    ds = resizer.transform_dataset(ds_data)
    example = next(iter(ds))
    assert example["image"].shape == (100, 100, 1)
    np.testing.assert_array_equal(example["scale"], (0.25, 0.25))
    np.testing.assert_allclose(example["instances"], data_example["instances"] * 0.25)


def test_resizer_from_config():
    resizer = resizing.Resizer.from_config(
        config=resizing.PreprocessingConfig(input_scaling=0.5, pad_to_stride=32)
    )
    assert resizer.image_key == "image"
    assert resizer.points_key == "instances"
    assert resizer.scale == 0.5
    assert resizer.pad_to_stride == 32

    resizer = resizing.Resizer.from_config(
        config=resizing.PreprocessingConfig(input_scaling=0.5, pad_to_stride=32),
        pad_to_stride=16,
    )
    assert resizer.image_key == "image"
    assert resizer.points_key == "instances"
    assert resizer.scale == 0.5
    assert resizer.pad_to_stride == 32

    resizer = resizing.Resizer.from_config(
        config=resizing.PreprocessingConfig(input_scaling=0.5, pad_to_stride=None),
        pad_to_stride=32,
    )
    assert resizer.image_key == "image"
    assert resizer.points_key == "instances"
    assert resizer.scale == 0.5
    assert resizer.pad_to_stride == 32

    with pytest.raises(ValueError):
        resizer = resizing.Resizer.from_config(
            config=resizing.PreprocessingConfig(input_scaling=0.5, pad_to_stride=None)
        )


def test_size_matcher():
    # Create some fake data using two different size videos.
    skeleton = sleap.Skeleton.from_names_and_edge_inds(["A"])
    labels = sleap.Labels(
        [
            sleap.LabeledFrame(
                frame_idx=0,
                video=sleap.Video.from_filename(
                    TEST_SMALL_ROBOT_MP4_FILE, grayscale=True
                ),
                instances=[
                    sleap.Instance.from_pointsarray(
                        np.array([[128, 128]]), skeleton=skeleton
                    )
                ],
            ),
            sleap.LabeledFrame(
                frame_idx=0,
                video=sleap.Video.from_filename(
                    TEST_H5_FILE, dataset="/box", input_format="channels_first"
                ),
                instances=[
                    sleap.Instance.from_pointsarray(
                        np.array([[128, 128]]), skeleton=skeleton
                    )
                ],
            ),
        ]
    )

    # Create a loader for those labels.
    labels_reader = providers.LabelsReader(labels)
    ds = labels_reader.make_dataset()
    ds_iter = iter(ds)
    assert next(ds_iter)["image"].shape == (320, 560, 1)
    assert next(ds_iter)["image"].shape == (512, 512, 1)

    def check_padding(image, from_y, to_y, from_x, to_x):
        assert (image.numpy()[from_y:to_y, from_x:to_x] == 0).all()

    # Check SizeMatcher when target dims is not strictly larger than actual image dims
    size_matcher = SizeMatcher(max_image_height=560, max_image_width=560)
    transform_iter = iter(size_matcher.transform_dataset(ds))
    im1 = next(transform_iter)["image"]
    assert im1.shape == (560, 560, 1)
    # padding should be on the bottom
    check_padding(im1, 321, 560, 0, 560)
    im2 = next(transform_iter)["image"]
    assert im2.shape == (560, 560, 1)

    # Variant 2
    size_matcher = SizeMatcher(max_image_height=320, max_image_width=560)
    transform_iter = iter(size_matcher.transform_dataset(ds))
    im1 = next(transform_iter)["image"]
    assert im1.shape == (320, 560, 1)
    im2 = next(transform_iter)["image"]
    assert im2.shape == (320, 560, 1)
    # padding should be on the right
    check_padding(im2, 0, 320, 321, 560)

    # Check SizeMatcher when target is 'max' in both dimensions
    size_matcher = SizeMatcher(max_image_height=512, max_image_width=560)
    transform_iter = iter(size_matcher.transform_dataset(ds))
    im1 = next(transform_iter)["image"]
    assert im1.shape == (512, 560, 1)
    # Check padding is on the bottom
    check_padding(im1, 320, 512, 0, 560)
    im2 = next(transform_iter)["image"]
    assert im2.shape == (512, 560, 1)
    # Check padding is on the right
    check_padding(im2, 0, 512, 512, 560)

    # Check SizeMatcher when target is larger in both dimensions
    size_matcher = SizeMatcher(max_image_height=750, max_image_width=750)
    transform_iter = iter(size_matcher.transform_dataset(ds))
    im1 = next(transform_iter)["image"]
    assert im1.shape == (750, 750, 1)
    # Check padding is on the bottom
    check_padding(im1, 700, 750, 0, 750)
    im2 = next(transform_iter)["image"]
    assert im2.shape == (750, 750, 1)

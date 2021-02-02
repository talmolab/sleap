import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import instance_centroids
from sleap.nn.data import instance_cropping
from sleap.nn.config import InstanceCroppingConfig


def test_normalize_bboxes():
    bbox = tf.convert_to_tensor([[0, 0, 3, 3]], tf.float32)
    norm_bbox = instance_cropping.normalize_bboxes(bbox, 9, 9)
    np.testing.assert_array_equal(norm_bbox, [[0, 0, 0.375, 0.375]])

    unnorm_bbox = instance_cropping.unnormalize_bboxes(norm_bbox, 9, 9)
    np.testing.assert_array_equal(unnorm_bbox, bbox)


def test_make_centered_bboxes():
    bbox = instance_cropping.make_centered_bboxes(
        tf.convert_to_tensor([[1, 1]], tf.float32), box_height=3, box_width=3
    )
    np.testing.assert_array_equal(bbox, [[0, 0, 2, 2]])

    bbox = instance_cropping.make_centered_bboxes(
        tf.convert_to_tensor([[2, 2]], tf.float32), box_height=4, box_width=4
    )
    np.testing.assert_array_equal(bbox, [[0.5, 0.5, 3.5, 3.5]])


def test_crop_bboxes():
    xv = tf.cast(tf.range(4), tf.uint8)
    yv = tf.cast(tf.range(5), tf.uint8)
    XX, YY = tf.meshgrid(xv, yv)
    img = tf.stack([XX, YY], axis=-1)

    centroids = tf.convert_to_tensor([[1, 1]], tf.float32)
    bboxes = instance_cropping.make_centered_bboxes(
        centroids, box_height=3, box_width=3
    )
    crops = instance_cropping.crop_bboxes(img, bboxes)

    patch_xx = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    patch_yy = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    expected_patch = np.expand_dims(np.stack([patch_xx, patch_yy], axis=-1), axis=0)
    np.testing.assert_array_equal(crops, expected_patch)
    np.testing.assert_array_equal(crops, np.expand_dims(img.numpy()[:3, :3, :], axis=0))
    assert crops.dtype == img.dtype


def test_crop_bboxes_rounding():
    # Test for rounding truncation bug when computing bounding box size for cropping.
    bboxes = instance_cropping.make_centered_bboxes(
        tf.cast([[464.42838, 550.14276]], tf.float32), box_height=100, box_width=100
    )
    crops = instance_cropping.crop_bboxes(
        tf.zeros([16, 16, 1], tf.float32), bboxes=bboxes
    )
    assert crops.shape == (1, 100, 100, 1)


def test_instance_cropper(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons,
    )
    instance_cropper = instance_cropping.InstanceCropper(
        crop_width=160, crop_height=160, keep_full_image=False
    )

    ds = instance_centroid_finder.transform_dataset(labels_reader.make_dataset())
    ds = instance_cropper.transform_dataset(ds)

    example = next(iter(ds))

    assert example["instance_image"].shape == (160, 160, 1)
    assert example["instance_image"].dtype == tf.uint8

    assert example["bbox"].shape == (4,)
    assert example["bbox"].dtype == tf.float32

    assert example["center_instance"].shape == (2, 2)
    assert example["center_instance"].dtype == tf.float32

    assert example["center_instance_ind"] == 0
    assert example["center_instance_ind"].dtype == tf.int32

    assert example["all_instances"].shape == (2, 2, 2)
    assert example["all_instances"].dtype == tf.float32

    assert example["centroid"].shape == (2,)
    assert example["centroid"].dtype == tf.float32

    assert example["full_image_height"] == 384
    assert example["full_image_height"].dtype == tf.int32

    assert example["full_image_width"] == 384
    assert example["full_image_width"].dtype == tf.int32

    assert example["video_ind"] == 0
    assert example["video_ind"].dtype == tf.int32

    assert example["frame_ind"] == 0
    assert example["frame_ind"].dtype == tf.int64

    np.testing.assert_array_equal(example["scale"], (1.0, 1.0))
    assert example["scale"].dtype == tf.float32

    assert example["skeleton_inds"].shape == (2,)
    assert example["skeleton_inds"].dtype == tf.int32

    assert "image" not in example


def test_instance_cropper_keeping_full_image(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons,
    )
    instance_cropper = instance_cropping.InstanceCropper(
        crop_width=160, crop_height=160, keep_full_image=True
    )

    ds = instance_centroid_finder.transform_dataset(labels_reader.make_dataset())
    ds = instance_cropper.transform_dataset(ds)

    example = next(iter(ds))

    assert example["instance_image"].shape == (160, 160, 1)
    assert example["instance_image"].dtype == tf.uint8

    assert example["full_image_height"] == 384
    assert example["full_image_height"].dtype == tf.int32

    assert example["full_image_width"] == 384
    assert example["full_image_width"].dtype == tf.int32

    assert example["image"].shape == (384, 384, 1)
    assert example["image"].dtype == tf.uint8


def test_instance_cropper_from_config():
    cropper = instance_cropping.InstanceCropper.from_config(
        config=InstanceCroppingConfig(crop_size=16),
        crop_size=None,
    )
    assert cropper.crop_width == 16
    assert cropper.crop_height == 16

    cropper = instance_cropping.InstanceCropper.from_config(
        config=InstanceCroppingConfig(crop_size=16),
        crop_size=24,
    )
    assert cropper.crop_width == 16
    assert cropper.crop_height == 16

    with pytest.raises(ValueError):
        cropper = instance_cropping.InstanceCropper.from_config(
            config=InstanceCroppingConfig(crop_size=None),
            crop_size=None,
        )

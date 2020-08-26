import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only; use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import augmentation


def test_augmentation(min_labels):
    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))

    augmenter = augmentation.ImgaugAugmenter.from_config(
        augmentation.AugmentationConfig(
            rotate=True, rotation_min_angle=-90, rotation_max_angle=-90
        )
    )
    ds = augmenter.transform_dataset(ds)

    example = next(iter(ds))

    assert example["image"].shape == (384, 384, 1)
    assert example["image"].dtype == tf.uint8

    np.testing.assert_allclose(
        tf.image.rot90(example_preaug["image"]), example["image"]
    )

    assert example["instances"].shape == (2, 2, 2)
    assert example["instances"].dtype == tf.float32
    # TODO: check for correctness
    assert tf.reduce_all(example["instances"] != example_preaug["instances"])


def test_random_cropper(min_labels):
    cropper = augmentation.RandomCropper(crop_height=64, crop_width=32)
    assert "image" in cropper.input_keys
    assert "instances" in cropper.input_keys
    assert "crop_bbox" in cropper.output_keys

    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))
    ds = cropper.transform_dataset(ds)
    example = next(iter(ds))

    assert example["image"].shape == (64, 32, 1)
    assert "crop_bbox" in example
    offset = tf.stack([example["crop_bbox"][0, 1], example["crop_bbox"][0, 0]], axis=-1)
    assert tf.reduce_all(
        example["instances"] == (
            example_preaug["instances"] - tf.expand_dims(offset, axis=0)))

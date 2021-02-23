import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import instance_centroids
from sleap.nn.data import instance_cropping
from sleap.nn.data.utils import make_grid_vectors
from sleap.nn.data.confidence_maps import (
    make_confmaps,
    make_multi_confmaps,
    make_multi_confmaps_with_offsets,
    SingleInstanceConfidenceMapGenerator,
    MultiConfidenceMapGenerator,
    InstanceConfidenceMapGenerator,
)


def test_make_confmaps():
    xv, yv = make_grid_vectors(image_height=4, image_width=5, output_stride=1)
    points = tf.cast([[0.5, 1.0], [3, 3.5], [2.0, 2.0]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)

    assert cm.dtype == tf.float32
    assert cm.shape == (4, 5, 3)
    np.testing.assert_allclose(
        cm,
        [
            [
                [0.535, 0.0, 0.018],
                [0.535, 0.0, 0.082],
                [0.197, 0.001, 0.135],
                [0.027, 0.002, 0.082],
                [0.001, 0.001, 0.018],
            ],
            [
                [0.882, 0.0, 0.082],
                [0.882, 0.006, 0.368],
                [0.325, 0.027, 0.607],
                [0.044, 0.044, 0.368],
                [0.002, 0.027, 0.082],
            ],
            [
                [0.535, 0.004, 0.135],
                [0.535, 0.044, 0.607],
                [0.197, 0.197, 1.0],
                [0.027, 0.325, 0.607],
                [0.001, 0.197, 0.135],
            ],
            [
                [0.119, 0.01, 0.082],
                [0.119, 0.119, 0.368],
                [0.044, 0.535, 0.607],
                [0.006, 0.882, 0.368],
                [0.0, 0.535, 0.082],
            ],
        ],
        atol=1e-3,
    )

    # Grid aligned peak
    points = tf.cast([[2, 3]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)
    assert cm.shape == (4, 5, 1)
    assert cm[3, 2] == 1.0

    # Output stride
    xv, yv = make_grid_vectors(image_height=8, image_width=8, output_stride=2)
    points = tf.cast([[2, 4]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)
    assert cm.shape == (4, 4, 1)
    assert cm[2, 1] == 1.0

    # Missing points
    xv, yv = make_grid_vectors(image_height=8, image_width=8, output_stride=2)
    points = tf.cast([[2, 4]], tf.float32)
    cm = make_confmaps(points, xv, yv, sigma=1.0)
    points_with_nan = tf.cast([[2, 4], [np.nan, np.nan]], tf.float32)
    cm_with_nan = make_confmaps(points_with_nan, xv, yv, sigma=1.0)
    assert cm_with_nan.shape == (4, 4, 2)
    assert cm_with_nan.dtype == tf.float32
    np.testing.assert_array_equal(cm_with_nan[:, :, 0], cm[:, :, 0])
    assert (cm_with_nan[:, :, 1].numpy() == 0).all()


def test_make_multi_confmaps():
    xv, yv = make_grid_vectors(image_height=4, image_width=5, output_stride=1)
    instances = tf.cast(
        [
            [[0.5, 1.0], [2.0, 2.0]],
            [[1.5, 1.0], [2.0, 3.0]],
            [[np.nan, np.nan], [-1.0, 5.0]],
        ],
        tf.float32,
    )
    cms = make_multi_confmaps(instances, xv=xv, yv=yv, sigma=1.0)
    assert cms.shape == (4, 5, 2)
    assert cms.dtype == tf.float32

    cm0 = make_confmaps(instances[0], xv=xv, yv=yv, sigma=1.0)
    cm1 = make_confmaps(instances[1], xv=xv, yv=yv, sigma=1.0)
    cm2 = make_confmaps(instances[2], xv=xv, yv=yv, sigma=1.0)

    np.testing.assert_array_equal(
        cms, tf.reduce_max(tf.stack([cm0, cm1, cm2], axis=-1), axis=-1)
    )


def test_make_multi_confmaps_with_offsets():
    xv, yv = make_grid_vectors(image_height=4, image_width=5, output_stride=1)
    instances = tf.cast(
        [
            [[0.5, 1.0], [2.0, 2.0]],
            [[1.5, 1.0], [2.0, 3.0]],
            [[np.nan, np.nan], [-1.0, 5.0]],
        ],
        tf.float32,
    )
    cms, offsets = make_multi_confmaps_with_offsets(
        instances, xv, yv, stride=1, sigma=1.0, offsets_threshold=0.2
    )
    assert offsets.shape == (4, 5, 2, 2)


def test_single_instance_confidence_map_generator(min_labels_robot):
    labels_reader = providers.LabelsReader(min_labels_robot)
    confmap_generator = SingleInstanceConfidenceMapGenerator(
        sigma=5, output_stride=2, with_offsets=False
    )
    ds = labels_reader.make_dataset()
    ds = confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert example["confidence_maps"].shape == (320 // 2, 560 // 2, 2)
    assert example["confidence_maps"].dtype == tf.float32

    confmap_generator = SingleInstanceConfidenceMapGenerator(
        sigma=5, output_stride=2, with_offsets=True
    )
    ds = labels_reader.make_dataset()
    ds = confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert example["confidence_maps"].shape == (320 // 2, 560 // 2, 2)
    assert example["confidence_maps"].dtype == tf.float32
    assert example["offsets"].shape == (320 // 2, 560 // 2, 4)
    assert example["offsets"].dtype == tf.float32


def test_multi_confidence_map_generator(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    multi_confmap_generator = MultiConfidenceMapGenerator(
        sigma=3 / 2, output_stride=2, centroids=False
    )
    ds = labels_reader.make_dataset()
    ds = multi_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))

    assert example["confidence_maps"].shape == (192, 192, 2)
    assert example["confidence_maps"].dtype == tf.float32

    instances = example["instances"].numpy() / multi_confmap_generator.output_stride
    cms = example["confidence_maps"].numpy()

    np.testing.assert_allclose(
        cms[int(instances[0, 0, 1]), int(instances[0, 0, 0]), :], [0.948463, 0.0]
    )
    np.testing.assert_allclose(
        cms[int(instances[1, 0, 1]), int(instances[1, 0, 0]), :], [0.66676116, 0.0]
    )

    np.testing.assert_allclose(
        cms[int(instances[0, 1, 1]), int(instances[0, 1, 0]), :], [0.0, 0.9836702]
    )
    np.testing.assert_allclose(
        cms[int(instances[1, 1, 1]), int(instances[1, 1, 0]), :], [0.0, 0.8815618]
    )

    multi_confmap_generator.with_offsets = True
    ds = labels_reader.make_dataset()
    ds = multi_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert "offsets" in example


def test_multi_confidence_map_generator_centroids(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons,
    )
    multi_confmap_generator = MultiConfidenceMapGenerator(
        sigma=5 / 2, output_stride=2, centroids=True
    )
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = multi_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))

    assert example["centroid_confidence_maps"].shape == (192, 192, 1)
    assert example["centroid_confidence_maps"].dtype == tf.float32

    centroids = example["centroids"].numpy() / multi_confmap_generator.output_stride
    centroid_cms = example["centroid_confidence_maps"].numpy()

    np.testing.assert_allclose(
        centroid_cms[int(centroids[0, 1]), int(centroids[0, 0]), :], [0.9811318]
    )
    np.testing.assert_allclose(
        centroid_cms[int(centroids[1, 1]), int(centroids[1, 0]), :], [0.8642299]
    )

    multi_confmap_generator.with_offsets = True
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = multi_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert "offsets" in example


def test_instance_confidence_map_generator(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons,
    )
    instance_cropper = instance_cropping.InstanceCropper(
        crop_width=160, crop_height=160
    )
    instance_confmap_generator = InstanceConfidenceMapGenerator(
        sigma=5 / 2, output_stride=2, all_instances=False
    )
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = instance_cropper.transform_dataset(ds)
    ds = instance_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))

    assert example["instance_confidence_maps"].shape == (80, 80, 2)
    assert example["instance_confidence_maps"].dtype == tf.float32
    assert "all_instance_confidence_maps" not in example

    points = (
        example["center_instance"].numpy() / instance_confmap_generator.output_stride
    )
    cms = example["instance_confidence_maps"].numpy()

    np.testing.assert_allclose(
        cms[(points[:, 1]).astype(int), (points[:, 0]).astype(int), :],
        [[0.9139312, 0.0], [0.0, 0.94459903]],
    )

    instance_confmap_generator.with_offsets = True
    instance_confmap_generator.flatten_offsets = False
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = instance_cropper.transform_dataset(ds)
    ds = instance_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert "offsets" in example
    assert example["offsets"].shape == (80, 80, 2, 2)

    instance_confmap_generator.flatten_offsets = True
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = instance_cropper.transform_dataset(ds)
    ds = instance_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert example["offsets"].shape == (80, 80, 4)


def test_instance_confidence_map_generator_with_all_instances(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons,
    )
    instance_cropper = instance_cropping.InstanceCropper(
        crop_width=160, crop_height=160
    )
    instance_confmap_generator = InstanceConfidenceMapGenerator(
        sigma=5 / 2, output_stride=2, all_instances=True
    )
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = instance_cropper.transform_dataset(ds)
    ds = instance_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))

    assert "instance_confidence_maps" in example
    assert example["all_instance_confidence_maps"].shape == (80, 80, 2)
    assert example["all_instance_confidence_maps"].dtype == tf.float32

    instances = (
        example["all_instances"].numpy() / instance_confmap_generator.output_stride
    )
    all_cms = example["all_instance_confidence_maps"].numpy()

    x = (instances[:, :, 0]).astype(int)
    y = (instances[:, :, 1]).astype(int)
    x[(x < 0) | (x >= all_cms.shape[1])] = 0
    y[(y < 0) | (y >= all_cms.shape[0])] = 0

    np.testing.assert_allclose(
        all_cms[y, x, :],
        [[[0.91393119, 0.0], [0.0, 0.94459903]], [[0.0, 0.0], [0.0, 0.0]]],
        atol=1e-6,
    )

    instance_confmap_generator.with_offsets = True
    ds = labels_reader.make_dataset()
    ds = instance_centroid_finder.transform_dataset(ds)
    ds = instance_cropper.transform_dataset(ds)
    ds = instance_confmap_generator.transform_dataset(ds)
    example = next(iter(ds))
    assert "offsets" in example
    assert "all_instance_offsets" in example

import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import instance_centroids


def test_find_points_bbox_midpoint():
    pts = tf.convert_to_tensor([
        [1, 2],
        [2, 3]], dtype=tf.float32)
    mid_pt = instance_centroids.find_points_bbox_midpoint(pts)
    np.testing.assert_array_equal(mid_pt, [1.5, 2.5])

    pts = tf.convert_to_tensor([
        [1, 2],
        [np.nan, np.nan],
        [2, 3]], dtype=tf.float32)
    mid_pt = instance_centroids.find_points_bbox_midpoint(pts)
    np.testing.assert_array_equal(mid_pt, [1.5, 2.5])


def test_get_instance_anchors():
    instances = tf.convert_to_tensor([
        [[0, 1],
         [2, 3],
         [4, 5]],
        [[6, 7],
         [8, 9],
         [10, 11]]
    ])
    anchor_inds = tf.convert_to_tensor([0, 1], tf.int32)
    anchors = instance_centroids.get_instance_anchors(instances, anchor_inds)
    np.testing.assert_array_equal(anchors, [[0, 1], [8, 9]])


def test_instance_centroid_finder(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    labels_ds = labels_reader.make_dataset()

    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=False)
    ds = instance_centroid_finder.transform_dataset(labels_ds)

    example = next(iter(ds))

    assert example["centroids"].dtype == tf.float32
    np.testing.assert_allclose(example["centroids"],
        [[122.49705, 180.57481],
         [242.28264, 195.62775]])

def test_instance_centroid_finder_anchored(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    labels_ds = labels_reader.make_dataset()

    instance_centroid_finder = instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True, anchor_part_names="A",
        skeletons=labels_reader.labels.skeletons)
    ds = instance_centroid_finder.transform_dataset(labels_ds)

    example = next(iter(ds))

    assert example["centroids"].dtype == tf.float32
    np.testing.assert_allclose(example["centroids"],
        [[92.65221, 202.72598],
         [205.93005, 187.88963]])

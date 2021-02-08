import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import edge_maps
from sleap.nn.data.utils import make_grid_vectors


def test_distance_to_edge():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    sampling_grid = tf.stack(tf.meshgrid(xv, yv), axis=-1)  # (height, width, 2)
    distances = edge_maps.distance_to_edge(
        sampling_grid, edge_source=edge_source, edge_destination=edge_destination
    )

    np.testing.assert_allclose(
        distances,
        [
            [[1.25, 0.0], [0.25, 0.5], [1.25, 2.0]],
            [[1.0, 0.5], [0.0, 0.0], [1.0, 0.5]],
            [[1.25, 2.0], [0.25, 0.5], [1.25, 0.0]],
        ],
        atol=1e-3,
    )


def test_edge_confidence_map():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    edge_confidence_map = edge_maps.make_edge_maps(
        xv=xv,
        yv=yv,
        edge_source=edge_source,
        edge_destination=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        edge_confidence_map,
        [
            [[0.458, 1.000], [0.969, 0.882], [0.458, 0.135]],
            [[0.607, 0.882], [1.000, 1.000], [0.607, 0.882]],
            [[0.458, 0.135], [0.969, 0.882], [0.458, 1.000]],
        ],
        atol=1e-3,
    )


def test_make_pafs():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    pafs = edge_maps.make_pafs(
        xv=xv,
        yv=yv,
        edge_source=edge_source,
        edge_destination=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        pafs,
        [
            [
                [[0.0, 0.458], [0.707, 0.707]],
                [[0.0, 0.969], [0.624, 0.624]],
                [[0.0, 0.458], [0.096, 0.096]],
            ],
            [
                [[0.0, 0.607], [0.624, 0.624]],
                [[0.0, 1.0], [0.707, 0.707]],
                [[0.0, 0.607], [0.624, 0.624]],
            ],
            [
                [[0.0, 0.458], [0.096, 0.096]],
                [[0.0, 0.969], [0.624, 0.624]],
                [[0.0, 0.458], [0.707, 0.707]],
            ],
        ],
        atol=1e-3,
    )


def test_make_multi_pafs():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast(
        [
            [[1, 0.5], [0, 0]],
            [[1, 0.5], [0, 0]],
        ],
        tf.float32,
    )
    edge_destination = tf.cast(
        [
            [[1, 1.5], [2, 2]],
            [[1, 1.5], [2, 2]],
        ],
        tf.float32,
    )
    sigma = 1.0

    pafs = edge_maps.make_multi_pafs(
        xv=xv,
        yv=yv,
        edge_sources=edge_source,
        edge_destinations=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        pafs,
        [
            [
                [[0.0, 0.916], [1.414, 1.414]],
                [[0.0, 1.938], [1.248, 1.248]],
                [[0.0, 0.916], [0.191, 0.191]],
            ],
            [
                [[0.0, 1.213], [1.248, 1.248]],
                [[0.0, 2.0], [1.414, 1.414]],
                [[0.0, 1.213], [1.248, 1.248]],
            ],
            [
                [[0.0, 0.916], [0.191, 0.191]],
                [[0.0, 1.938], [1.248, 1.248]],
                [[0.0, 0.916], [1.414, 1.414]],
            ],
        ],
        atol=1e-3,
    )


def test_get_edge_points():
    instances = tf.reshape(tf.range(4 * 3 * 2), [4, 3, 2])
    edge_inds = tf.cast([[0, 1], [1, 2], [0, 2]], tf.int32)
    edge_sources, edge_destinations = edge_maps.get_edge_points(instances, edge_inds)
    np.testing.assert_array_equal(
        edge_sources,
        [
            [[0, 1], [2, 3], [0, 1]],
            [[6, 7], [8, 9], [6, 7]],
            [[12, 13], [14, 15], [12, 13]],
            [[18, 19], [20, 21], [18, 19]],
        ],
    )
    np.testing.assert_array_equal(
        edge_destinations,
        [
            [[2, 3], [4, 5], [4, 5]],
            [[8, 9], [10, 11], [10, 11]],
            [[14, 15], [16, 17], [16, 17]],
            [[20, 21], [22, 23], [22, 23]],
        ],
    )


def test_part_affinity_fields_generator(min_labels):
    labels_reader = providers.LabelsReader(min_labels)
    paf_generator = edge_maps.PartAffinityFieldsGenerator(
        sigma=8, output_stride=2, skeletons=labels_reader.labels.skeletons
    )

    ds = labels_reader.make_dataset()
    ds = paf_generator.transform_dataset(ds)

    example = next(iter(ds))
    assert example["part_affinity_fields"].shape == (192, 192, 1, 2)
    assert example["part_affinity_fields"].dtype == tf.float32

    np.testing.assert_allclose(
        example["part_affinity_fields"][196 // 2, 250 // 2, :, :],
        [[0.9600351, 0.20435576]],
    )

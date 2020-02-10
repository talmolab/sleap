import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import edge_maps
from sleap.nn.data.utils import make_grid_vectors


def test_distance_to_edge():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    sampling_grid = tf.stack(tf.meshgrid(xv, yv), axis=-1)  # (height, width, 2)
    distances = edge_maps.distance_to_edge(sampling_grid,
        edge_source=edge_source, edge_destination=edge_destination)

    np.testing.assert_allclose(
        distances,
        [
         [[1.25, 0.  ],
          [0.25, 0.5 ],
          [1.25, 2.  ]],
         [[1.  , 0.5 ],
          [0.  , 0.  ],
          [1.  , 0.5 ]],
         [[1.25, 2.  ],
          [0.25, 0.5 ],
          [1.25, 0.  ]]
        ],
        atol=1e-3
    )


def test_edge_confidence_map():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    edge_confidence_map = edge_maps.make_edge_maps(xv=xv, yv=yv,
        edge_source=edge_source, edge_destination=edge_destination, sigma=sigma)

    np.testing.assert_allclose(
        edge_confidence_map,
        [
         [[0.458, 1.000],
          [0.969, 0.882],
          [0.458, 0.135]],
         [[0.607, 0.882],
          [1.000, 1.000],
          [0.607, 0.882]],
         [[0.458, 0.135],
          [0.969, 0.882],
          [0.458, 1.000]]
        ],
        atol=1e-3
    )


def test_make_pafs():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = tf.cast([[1, 0.5], [0, 0]], tf.float32)
    edge_destination = tf.cast([[1, 1.5], [2, 2]], tf.float32)
    sigma = 1.0

    pafs = edge_maps.make_pafs(xv=xv, yv=yv,
        edge_source=edge_source, edge_destination=edge_destination, sigma=sigma)

    np.testing.assert_allclose(
        pafs,
        [[[[0.   , 0.458],
           [0.707, 0.707]],
          [[0.   , 0.969],
           [0.624, 0.624]],
          [[0.   , 0.458],
           [0.096, 0.096]]],
         [[[0.   , 0.607],
           [0.624, 0.624]],
          [[0.   , 1.   ],
           [0.707, 0.707]],
          [[0.   , 0.607],
           [0.624, 0.624]]],
         [[[0.   , 0.458],
           [0.096, 0.096]],
          [[0.   , 0.969],
           [0.624, 0.624]],
          [[0.   , 0.458],
           [0.707, 0.707]]]],
        atol=1e-3)

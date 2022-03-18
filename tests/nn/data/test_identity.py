import tensorflow as tf
import numpy as np
import sleap

from sleap.nn.data.identity import (
    make_class_vectors,
    make_class_maps,
    ClassVectorGenerator,
    ClassMapGenerator,
)


sleap.use_cpu_only()


def test_make_class_vectors():
    vecs = make_class_vectors(tf.cast([-1, 0, 1, 2], tf.int32), 3)
    np.testing.assert_array_equal(vecs, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_make_class_maps():
    xv, yv = sleap.nn.data.confidence_maps.make_grid_vectors(32, 32, output_stride=1)
    pts = tf.cast([[4, 6], [18, 24]], tf.float32)
    cms = sleap.nn.data.confidence_maps.make_confmaps(pts, xv, yv, sigma=2)
    class_maps = make_class_maps(cms, class_inds=[1, 0], n_classes=2, threshold=0.2)
    np.testing.assert_array_equal(
        tf.gather_nd(class_maps, [[6, 4], [24, 18]]), [[0, 1], [1, 0]]
    )


def test_class_vector_generator(min_tracks_2node_labels):
    labels = min_tracks_2node_labels

    gen = ClassVectorGenerator()

    p = labels.to_pipeline()
    ds = p.make_dataset()
    ds = gen.transform_dataset(ds)
    ex = next(iter(ds))

    np.testing.assert_array_equal(ex["class_vectors"], [[0, 1], [1, 0]])
    assert ex["class_vectors"].dtype == tf.float32

    p = labels.to_pipeline()
    p += gen
    p += sleap.pipelines.InstanceCentroidFinder()
    p += sleap.pipelines.InstanceCropper(32, 32)
    ex = p.peek()

    np.testing.assert_array_equal(ex["class_vectors"], [0, 1])
    assert ex["class_vectors"].dtype == tf.float32


def test_class_map_generator(min_tracks_2node_labels):
    labels = min_tracks_2node_labels

    gen = ClassMapGenerator(
        sigma=4.0, output_stride=4, centroids=False, class_map_threshold=0.2
    )

    p = labels.to_pipeline()
    ds = p.make_dataset()
    ds = gen.transform_dataset(ds)
    ex = next(iter(ds))

    subs = (
        tf.cast(
            [[444.75, 335.25], [457.75, 301.75], [415.75, 435.25], [422.75, 396.25]],
            tf.int32,
        )
        // 4
    )
    np.testing.assert_allclose(
        tf.gather_nd(ex["class_maps"], subs),
        [[0, 1], [0, 1], [1, 0], [1, 0]],
        atol=1e-2,
    )

    gen = ClassMapGenerator(
        sigma=4.0, output_stride=4, centroids=True, class_map_threshold=0.2
    )

    p = labels.to_pipeline()
    p += sleap.nn.data.instance_centroids.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="thorax",
        skeletons=labels.skeletons,
    )
    ds = p.make_dataset()
    ds = gen.transform_dataset(ds)
    ex = next(iter(ds))

    subs = (
        tf.cast(
            [[457.75, 301.75], [422.75, 396.25]],
            tf.int32,
        )
        // 4
    )
    np.testing.assert_allclose(
        tf.gather_nd(ex["class_maps"], subs), [[0, 1], [1, 0]], atol=1e-2
    )

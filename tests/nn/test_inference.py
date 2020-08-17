import pytest
import numpy as np
import tensorflow as tf
import sleap
from numpy.testing import assert_array_equal, assert_allclose

from sleap.nn.data.confidence_maps import (
    make_confmaps,
    make_grid_vectors,
    make_multi_confmaps,
)

from sleap.nn.inference import (
    CentroidCropGroundTruth,
    FindInstancePeaksGroundTruth,
    FindInstancePeaks,
    TopDownModel,
)

sleap.nn.system.use_cpu_only()


@pytest.fixture
def test_labels():
    skel = sleap.Skeleton()
    skel.add_node("a")
    skel.add_node("b")

    vid = sleap.Video.from_numpy(np.zeros((8, 12, 12, 1), dtype="uint8"))

    labels = sleap.Labels()
    for fidx in range(len(vid)):
        insts = []
        insts.append(
            sleap.Instance.from_pointsarray(
                points=np.array([[1, 2], [3, 4]]) + fidx, skeleton=skel
            )
        )
        if fidx >= 3:
            insts.append(
                sleap.Instance.from_pointsarray(
                    points=np.array([[5, 6], [7, 8]]) + fidx, skeleton=skel
                )
            )

        lf = sleap.LabeledFrame(video=vid, frame_idx=fidx, instances=insts)
        labels.append(lf)

    return labels


@pytest.fixture
def test_pipeline(test_labels):
    p = sleap.nn.data.pipelines.Pipeline(
        sleap.nn.data.pipelines.LabelsReader(labels=test_labels)
    )
    p += sleap.nn.data.pipelines.InstanceCentroidFinder(
        center_on_anchor_part=True,
        anchor_part_names="a",
        skeletons=test_labels.skeleton,
    )
    p += sleap.nn.data.pipelines.Batcher(batch_size=4, unrag=False)
    return p


def test_centroid_crop_gt_layer(test_labels, test_pipeline):
    ex = test_pipeline.peek()

    crop_layer = CentroidCropGroundTruth(crop_size=6)
    out = crop_layer(ex)

    assert tuple(out["crops"].shape) == (4, None, 6, 6, 1)
    assert tuple(out["crop_offsets"].shape) == (4, None, 2)
    assert tuple(out["centroids"].shape) == (4, None, 2)
    assert tuple(out["centroid_vals"].shape) == (4, None)

    assert tuple(out["crops"].bounding_shape()) == (4, 2, 6, 6, 1)
    assert tuple(out["crop_offsets"].bounding_shape()) == (4, 2, 2)
    assert tuple(out["centroids"].bounding_shape()) == (4, 2, 2)
    assert tuple(out["centroid_vals"].bounding_shape()) == (4, 2)

    assert out["crops"].dtype == tf.uint8
    assert out["crop_offsets"].dtype == tf.float32
    assert out["centroids"].dtype == tf.float32
    assert out["centroid_vals"].dtype == tf.float32

    assert (out["centroids"][0][0].numpy() == test_labels[0][0].numpy()[0]).all()
    assert (out["centroids"][1][0].numpy() == test_labels[1][0].numpy()[0]).all()


def test_instance_peaks_gt_layer(test_labels, test_pipeline):
    crop_layer = CentroidCropGroundTruth(crop_size=6)
    instance_peaks_layer = FindInstancePeaksGroundTruth()

    ex = test_pipeline.peek()
    crop_output = crop_layer(ex)
    out = instance_peaks_layer(ex, crop_output)

    assert tuple(out["centroids"].shape) == (4, None, 2)
    assert tuple(out["centroid_vals"].shape) == (4, None)
    assert tuple(out["instance_peaks"].shape) == (4, None, None, 2)
    assert tuple(out["instance_peak_vals"].shape) == (4, None, None)

    assert out["centroids"][0].shape == (1, 2)
    assert out["centroids"][1].shape == (1, 2)
    assert out["centroids"][2].shape == (1, 2)
    assert out["centroids"][3].shape == (2, 2)

    assert tuple(out["centroids"].bounding_shape()) == (4, 2, 2)
    assert tuple(out["centroid_vals"].bounding_shape()) == (4, 2)
    assert tuple(out["instance_peaks"].bounding_shape()) == (4, 2, 2, 2)
    assert tuple(out["instance_peak_vals"].bounding_shape()) == (4, 2, 2)

    assert out["centroids"].dtype == tf.float32
    assert out["centroid_vals"].dtype == tf.float32
    assert out["instance_peaks"].dtype == tf.float32
    assert out["instance_peak_vals"].dtype == tf.float32

    assert (out["instance_peaks"][0][0].numpy() == test_labels[0][0].numpy()).all()
    assert (out["instance_peaks"][1][0].numpy() == test_labels[1][0].numpy()).all()


def test_instance_peaks_layer():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], tf.float32)
    cms = tf.stack(
        [
            make_confmaps(points, xv, yv, sigma=1.0),
            make_confmaps(points + 1, xv, yv, sigma=1.0),
        ],
        axis=0,
    )

    x_in = tf.keras.layers.Input([12, 12, 3])
    x_out = tf.keras.layers.Lambda(lambda x: x)(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    instance_peaks_layer = FindInstancePeaks(
        keras_model=model, input_scale=1.0, peak_threshold=0.2, keep_confmaps=False
    )

    # Raw tensor
    out = instance_peaks_layer(cms)
    assert tuple(out["instance_peaks"].shape) == (2, None, 3, 2)
    assert tuple(out["instance_peak_vals"].shape) == (2, None, 3)
    assert tuple(out["instance_peaks"].bounding_shape()) == (2, 1, 3, 2)
    assert tuple(out["instance_peak_vals"].bounding_shape()) == (2, 1, 3)
    assert_allclose(out["instance_peaks"][0][0].numpy(), points.numpy(), atol=0.1)
    assert_allclose(out["instance_peak_vals"][0][0].numpy(), [1, 1, 1], atol=0.3)
    assert_allclose(out["instance_peaks"][1][0].numpy(), points.numpy() + 1, atol=0.1)
    assert_allclose(out["instance_peak_vals"][1][0].numpy(), [1, 1, 1], atol=0.3)

    # Batched example
    crops = tf.RaggedTensor.from_tensor(tf.expand_dims(cms, axis=1), lengths=[1, 1])
    out = instance_peaks_layer({"crops": crops})
    assert tuple(out["instance_peaks"].shape) == (2, None, 3, 2)
    assert tuple(out["instance_peak_vals"].shape) == (2, None, 3)
    assert tuple(out["instance_peaks"].bounding_shape()) == (2, 1, 3, 2)
    assert tuple(out["instance_peak_vals"].bounding_shape()) == (2, 1, 3)
    assert_allclose(out["instance_peaks"][0][0].numpy(), points.numpy(), atol=0.1)
    assert_allclose(out["instance_peak_vals"][0][0].numpy(), [1, 1, 1], atol=0.3)
    assert_allclose(out["instance_peaks"][1][0].numpy(), points.numpy() + 1, atol=0.1)
    assert_allclose(out["instance_peak_vals"][1][0].numpy(), [1, 1, 1], atol=0.3)

    # Batch size = 1, multi-instance example
    crops = tf.RaggedTensor.from_tensor(tf.expand_dims(cms, axis=0), lengths=[2])
    out = instance_peaks_layer({"crops": crops})
    assert tuple(out["instance_peaks"].shape) == (1, None, 3, 2)
    assert tuple(out["instance_peak_vals"].shape) == (1, None, 3)
    assert tuple(out["instance_peaks"].bounding_shape()) == (1, 2, 3, 2)
    assert tuple(out["instance_peak_vals"].bounding_shape()) == (1, 2, 3)
    assert_allclose(out["instance_peaks"][0][0].numpy(), points.numpy(), atol=0.1)
    assert_allclose(out["instance_peak_vals"][0][0].numpy(), [1, 1, 1], atol=0.3)
    assert_allclose(out["instance_peaks"][0][1].numpy(), points.numpy() + 1, atol=0.1)
    assert_allclose(out["instance_peak_vals"][0][1].numpy(), [1, 1, 1], atol=0.3)

    # Offset adjustment and pass through centroids
    instance_peaks_layer = FindInstancePeaks(
        keras_model=model, input_scale=1.0, peak_threshold=0.2, keep_confmaps=True
    )
    # (samples, h, w, c) -> (samples, ?, h, w, c)
    crops = tf.RaggedTensor.from_tensor(tf.expand_dims(cms, axis=1), lengths=[1, 1])
    # (samples, centroids, 2) -> (samples, ?, 2)
    crop_offsets = tf.RaggedTensor.from_tensor(
        tf.reshape(tf.cast([1, 2, 3, 4], tf.float32), [2, 1, 2]), lengths=[1, 1]
    )
    out = instance_peaks_layer(
        {
            "crops": crops,
            "centroids": tf.zeros([]),
            "centroid_vals": tf.zeros([]),
            "crop_offsets": crop_offsets,
        }
    )
    assert "centroids" in out
    assert "centroid_vals" in out
    assert_allclose(
        out["instance_peaks"][0][0].numpy(),
        points.numpy() + np.array([[1, 2]]),
        atol=0.1,
    )
    assert_allclose(
        out["instance_peaks"][1][0].numpy(),
        points.numpy() + 1 + np.array([[3, 4]]),
        atol=0.1,
    )

    # Input scaling
    scale = 0.5
    instance_peaks_layer = FindInstancePeaks(
        keras_model=model, input_scale=scale, peak_threshold=0.2, keep_confmaps=False
    )
    xv, yv = make_grid_vectors(
        image_height=12 / scale, image_width=12 / scale, output_stride=1
    )
    points = tf.cast([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], tf.float32)
    cms = tf.stack(
        [
            make_confmaps(points / scale, xv, yv, sigma=1.0 / scale),
            make_confmaps((points + 1) / scale, xv, yv, sigma=1.0 / scale),
        ],
        axis=0,
    )
    out = instance_peaks_layer(cms)

    assert_allclose(
        out["instance_peaks"][0][0].numpy(), points.numpy() / scale, atol=0.15
    )
    assert_allclose(
        out["instance_peaks"][1][0].numpy(), (points.numpy() + 1) / scale, atol=0.15
    )


def test_topdown_model(test_pipeline):
    model = TopDownModel(
        centroid_crop=CentroidCropGroundTruth(crop_size=4),
        instance_peaks=FindInstancePeaksGroundTruth(),
        max_instances=2,
    )

    out = model.predict(test_pipeline.make_dataset())

    assert tuple(out["centroids"].shape) == (8, 2, 2)
    assert tuple(out["centroid_vals"].shape) == (8, 2)
    assert tuple(out["instance_peaks"].shape) == (8, 2, 2, 2)
    assert tuple(out["instance_peak_vals"].shape) == (8, 2, 2)
    assert tuple(out["n_valid"].shape) == (8,)

    assert (out["n_valid"] == [1, 1, 1, 2, 2, 2, 2, 2]).all()

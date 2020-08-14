import pytest
import numpy as np
import tensorflow as tf
import sleap

from sleap.nn.inference import (
    CentroidCropGroundTruth, FindInstancePeaksGroundTruth, TopDownModel
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


def test_topdown_model(test_pipeline):
    model = TopDownModel(
        centroid_crop=CentroidCropGroundTruth(crop_size=4),
        instance_peaks=FindInstancePeaksGroundTruth(),
        max_instances=2
    )

    out = model.predict(test_pipeline.make_dataset())

    assert tuple(out["centroids"].shape) == (8, 2, 2)
    assert tuple(out["centroid_vals"].shape) == (8, 2)
    assert tuple(out["instance_peaks"].shape) == (8, 2, 2, 2)
    assert tuple(out["instance_peak_vals"].shape) == (8, 2, 2)
    assert tuple(out["n_valid"].shape) == (8,)

    assert (out["n_valid"] == [1, 1, 1, 2, 2, 2, 2, 2]).all()

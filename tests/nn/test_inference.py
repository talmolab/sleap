import ast
from typing import cast
import pytest
import numpy as np
import json
from sleap.io.dataset import Labels
from sleap.nn.tracking import FlowCandidateMaker, Tracker
import tensorflow as tf
import sleap
from numpy.testing import assert_array_equal, assert_allclose
from pathlib import Path

from sleap.nn.data.confidence_maps import (
    make_confmaps,
    make_grid_vectors,
    make_multi_confmaps,
)

from sleap.nn.inference import (
    InferenceLayer,
    InferenceModel,
    Predictor,
    get_model_output_stride,
    find_head,
    SingleInstanceInferenceLayer,
    SingleInstanceInferenceModel,
    SingleInstancePredictor,
    CentroidCropGroundTruth,
    CentroidCrop,
    CentroidInferenceModel,
    FindInstancePeaksGroundTruth,
    FindInstancePeaks,
    TopDownMultiClassFindPeaks,
    TopDownInferenceModel,
    TopDownMultiClassInferenceModel,
    TopDownPredictor,
    BottomUpPredictor,
    BottomUpMultiClassPredictor,
    TopDownMultiClassPredictor,
    load_model,
    export_model,
    _make_cli_parser,
    _make_tracker_from_cli,
    main as sleap_track,
    export_cli as sleap_export,
)

from sleap.gui.learning import runners

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


def test_instance_peaks_gt_layer_nans():
    # Covers nasty edge case when evaluating centroid models and
    # GT instances have NaNs
    flat_values = tf.cast(
        [
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
            [np.nan, np.nan],
            [1, 1],
            [np.nan, np.nan],
        ],
        tf.float32,
    )
    nested_value_rowids = (
        tf.cast([0, 0, 1, 1], tf.int64),
        tf.cast([0, 0, 1, 1, 2, 2, 3, 3], tf.int64),
    )
    instances = tf.RaggedTensor.from_nested_value_rowids(
        flat_values, nested_value_rowids
    )

    flat_values = tf.cast([[0, 0], [1, 1], [0, 0], [1, 1]], tf.float32)
    nested_value_rowids = (tf.cast([0, 0, 1, 1], tf.int32),)
    centroids = tf.RaggedTensor.from_nested_value_rowids(
        flat_values, nested_value_rowids
    )

    flat_values = tf.cast([1, 1, 1, 1], tf.float32)
    nested_value_rowids = (tf.cast([0, 0, 1, 1], tf.int32),)
    centroid_vals = tf.RaggedTensor.from_nested_value_rowids(
        flat_values, nested_value_rowids
    )

    example_gt = {"instances": instances}
    crop_output = {"centroids": centroids, "centroid_vals": centroid_vals}

    layer = FindInstancePeaksGroundTruth()
    peaks_gt = layer(example_gt, crop_output)
    assert tuple(peaks_gt["instance_peaks"].bounding_shape()) == (2, 2, 2, 2)


def test_centroid_crop_layer():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[[1.75, 2.75]], [[3.75, 4.75]], [[5.75, 6.75]]], tf.float32)
    cms = tf.expand_dims(make_multi_confmaps(points, xv, yv, sigma=1.5), axis=0)

    x_in = tf.keras.layers.Input([12, 12, 1])
    x_out = tf.keras.layers.Lambda(lambda x: x, name="CentroidConfmapsHead")(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    layer = CentroidCrop(
        keras_model=model,
        input_scale=1.0,
        crop_size=3,
        pad_to_stride=1,
        output_stride=None,
        refinement="local",
        integral_patch_size=5,
        peak_threshold=0.2,
        return_confmaps=False,
    )

    # For Codecov to realize the wrapped CentroidCrop.call is tested/covered,
    # we need to unbind CentroidCrop.call from its bind with TfMethodTarget object
    # and then rebind the standalone function with the CentroidCrop object
    TfMethodTarget_object = layer.call.__wrapped__.__self__  # Get the bound object
    original_func = TfMethodTarget_object.weakrefself_func__()  # Get unbound function
    layer.call = original_func.__get__(layer, layer.__class__)  # Bind function

    out = layer(cms)
    assert tuple(out["centroids"].shape) == (1, None, 2)
    assert tuple(out["centroid_vals"].shape) == (1, None)
    assert tuple(out["crops"].shape) == (1, None, 3, 3, 1)
    assert tuple(out["crop_offsets"].shape) == (1, None, 2)

    assert tuple(out["centroids"].bounding_shape()) == (1, 3, 2)
    assert tuple(out["centroid_vals"].bounding_shape()) == (1, 3)
    assert tuple(out["crops"].bounding_shape()) == (1, 3, 3, 3, 1)
    assert tuple(out["crop_offsets"].bounding_shape()) == (1, 3, 2)

    assert_allclose(out["centroids"][0].numpy(), points.numpy().squeeze(axis=1))
    assert_allclose(out["centroid_vals"][0].numpy(), [1, 1, 1], atol=0.1)


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
    x_out = tf.keras.layers.Lambda(lambda x: x, name="CenteredInstanceConfmapsHead")(
        x_in
    )
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    instance_peaks_layer = FindInstancePeaks(
        keras_model=model,
        input_scale=1.0,
        peak_threshold=0.2,
        return_confmaps=False,
        refinement="integral",
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
        keras_model=model,
        input_scale=1.0,
        peak_threshold=0.2,
        return_confmaps=True,
        refinement="integral",
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
        keras_model=model,
        input_scale=scale,
        peak_threshold=0.2,
        return_confmaps=False,
        refinement="integral",
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
    model = TopDownInferenceModel(
        centroid_crop=CentroidCropGroundTruth(crop_size=4),
        instance_peaks=FindInstancePeaksGroundTruth(),
    )

    out = model.predict(test_pipeline.make_dataset())

    assert tuple(out["centroids"].shape) == (8, 2, 2)
    assert tuple(out["centroid_vals"].shape) == (8, 2)
    assert tuple(out["instance_peaks"].shape) == (8, 2, 2, 2)
    assert tuple(out["instance_peak_vals"].shape) == (8, 2, 2)
    assert tuple(out["n_valid"].shape) == (8,)

    assert (out["n_valid"] == [1, 1, 1, 2, 2, 2, 2, 2]).all()


def test_inference_layer():
    # Convert to float
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=1.0, pad_to_stride=1, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 4, 4, 1], 255), tf.uint8)
    out = layer(data)
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 1)
    assert tf.reduce_all(out == 1.0)

    # Convert from rgb to grayscale, infer ensure grayscale
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=1.0, pad_to_stride=1, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 4, 4, 3], 255), tf.uint8)
    out = layer(data)
    assert layer.ensure_grayscale
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 1)
    assert tf.reduce_all(out == 1.0)

    # Infer ensure rgb, convert from grayscale
    x_in = tf.keras.layers.Input([4, 4, 3])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=1.0, pad_to_stride=1, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 4, 4, 1], 255), tf.uint8)
    out = layer(data)
    assert not layer.ensure_grayscale
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 3)
    assert tf.reduce_all(out == 1.0)

    # Input scaling
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=0.5, pad_to_stride=1, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 8, 8, 1], 255), tf.uint8)
    out = layer(data)
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 1)
    assert tf.reduce_all(out == 1.0)

    # Stride padding
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=1, pad_to_stride=2, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 3, 3, 1], 255), tf.uint8)
    out = layer(data)
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 1)

    # Scaling and stride padding
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)
    layer = sleap.nn.inference.InferenceLayer(
        keras_model=keras_model, input_scale=0.5, pad_to_stride=2, ensure_grayscale=None
    )
    data = tf.cast(tf.fill([1, 6, 6, 1], 255), tf.uint8)
    out = layer(data)
    assert out.dtype == tf.float32
    assert tuple(out.shape) == (1, 4, 4, 1)


def test_get_model_output_stride():
    # Single input/output
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    model = tf.keras.Model(x_in, x)
    assert get_model_output_stride(model) == 1

    # Single input/output, downsampled
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.MaxPool2D(strides=2, padding="same")(x_in)
    model = tf.keras.Model(x_in, x)
    assert get_model_output_stride(model) == 2

    # Single input/output, downsampled, uneven
    x_in = tf.keras.layers.Input([5, 5, 1])
    x = tf.keras.layers.MaxPool2D(strides=2, padding="same")(x_in)
    model = tf.keras.Model(x_in, x)
    assert model.output.shape[1] == 3
    with pytest.warns(UserWarning):
        stride = get_model_output_stride(model)
    assert stride == 1

    # Multi input/output
    x_in = [tf.keras.layers.Input([4, 4, 1]), tf.keras.layers.Input([8, 8, 1])]
    x = [tf.keras.layers.MaxPool2D(strides=2, padding="same")(x) for x in x_in]
    model = tf.keras.Model(x_in, x)
    assert get_model_output_stride(model) == 1
    assert get_model_output_stride(model, input_ind=0, output_ind=0) == 2
    assert get_model_output_stride(model, input_ind=0, output_ind=1) == 1
    assert get_model_output_stride(model, input_ind=1, output_ind=0) == 4
    assert get_model_output_stride(model, input_ind=1, output_ind=1) == 2


def test_find_head():
    x_in = tf.keras.layers.Input([4, 4, 1])
    x = tf.keras.layers.Lambda(lambda x: x, name="A_0")(x_in)
    model = tf.keras.Model(x_in, x)

    assert find_head(model, "A") == 0
    assert find_head(model, "B") is None


def test_single_instance_inference():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[1.75, 2.75], [3.75, 4.75], [5.75, 6.75]], tf.float32)
    points = np.stack([points, points + 1], axis=0)
    cms = tf.stack(
        [
            make_confmaps(points[0], xv, yv, sigma=1.0),
            make_confmaps(points[1], xv, yv, sigma=1.0),
        ],
        axis=0,
    )

    x_in = tf.keras.layers.Input([12, 12, 3])
    x = tf.keras.layers.Lambda(lambda x: x, name="SingleInstanceConfmapsHead")(x_in)
    keras_model = tf.keras.Model(x_in, x)

    layer = SingleInstanceInferenceLayer(keras_model=keras_model, refinement="local")
    assert layer.output_stride == 1

    out = layer(cms)

    assert tuple(out["instance_peaks"].shape) == (2, 1, 3, 2)
    out["instance_peaks"] = tf.squeeze(out["instance_peaks"], axis=1)
    assert tuple(out["instance_peak_vals"].shape) == (2, 1, 3)
    out["instance_peak_vals"] = tf.squeeze(out["instance_peak_vals"], axis=1)
    assert_array_equal(out["instance_peaks"], points)
    assert_allclose(out["instance_peak_vals"], 1.0, atol=0.1)
    assert "confmaps" not in out

    out = layer({"image": cms})
    assert tuple(out["instance_peaks"].shape) == (2, 1, 3, 2)
    out["instance_peaks"] = tf.squeeze(out["instance_peaks"], axis=1)
    assert_array_equal(out["instance_peaks"], points)

    layer = SingleInstanceInferenceLayer(
        keras_model=keras_model, refinement="local", return_confmaps=True
    )
    out = layer(cms)
    assert "confmaps" in out
    assert_array_equal(out["confmaps"], cms)

    model = SingleInstanceInferenceModel(layer)
    preds = model.predict(cms)
    assert preds["instance_peaks"].shape == (2, 1, 3, 2)
    preds["instance_peaks"] = preds["instance_peaks"].squeeze(axis=1)
    assert_array_equal(preds["instance_peaks"], points)
    assert "instance_peak_vals" in preds
    assert "confmaps" in preds


def test_single_instance_predictor(
    min_labels_robot, min_single_instance_robot_model_path
):
    predictor = SingleInstancePredictor.from_trained_models(
        min_single_instance_robot_model_path
    )
    predictor.verbosity = "none"
    assert predictor.is_grayscale == False
    labels_pr = predictor.predict(min_labels_robot)
    assert len(labels_pr) == 2
    assert len(labels_pr[0].instances) == 1

    points_gt = np.concatenate(
        [min_labels_robot[0][0].numpy(), min_labels_robot[1][0].numpy()], axis=0
    )
    points_pr = np.concatenate(
        [labels_pr[0][0].numpy(), labels_pr[1][0].numpy()], axis=0
    )
    assert_allclose(points_gt, points_pr, atol=10.0)


def test_single_instance_predictor_high_peak_thresh(
    min_labels_robot, min_single_instance_robot_model_path
):
    predictor = SingleInstancePredictor.from_trained_models(
        min_single_instance_robot_model_path, peak_threshold=0
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels_robot)
    assert len(labels_pr) == 2
    assert len(labels_pr[0]) == 1
    assert labels_pr[0][0].n_visible_points == 2
    assert len(labels_pr[1]) == 1
    assert labels_pr[1][0].n_visible_points == 2

    predictor = SingleInstancePredictor.from_trained_models(
        min_single_instance_robot_model_path, peak_threshold=1.5
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels_robot)
    assert len(labels_pr) == 2
    assert len(labels_pr[0]) == 0
    assert len(labels_pr[1]) == 0


def test_topdown_predictor_centroid(min_labels, min_centroid_model_path):
    predictor = TopDownPredictor.from_trained_models(
        centroid_model_path=min_centroid_model_path
    )

    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2

    assert predictor.is_grayscale == True

    points_gt = np.concatenate(
        [min_labels[0][0].numpy(), min_labels[0][1].numpy()], axis=0
    )
    points_pr = np.concatenate(
        [labels_pr[0][0].numpy(), labels_pr[0][1].numpy()], axis=0
    )
    inds1, inds2 = sleap.nn.utils.match_points(points_gt, points_pr)
    assert_allclose(points_gt[inds1.numpy()], points_pr[inds2.numpy()], atol=1.5)

    # test max_instances (>2 will fail)
    predictor.inference_model.centroid_crop.max_instances = 2
    labels_pr = predictor.predict(min_labels)

    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2


def test_topdown_predictor_centroid_high_threshold(min_labels, min_centroid_model_path):
    predictor = TopDownPredictor.from_trained_models(
        centroid_model_path=min_centroid_model_path, peak_threshold=1.5
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 0


def test_topdown_predictor_centered_instance(
    min_labels, min_centered_instance_model_path
):
    predictor = TopDownPredictor.from_trained_models(
        confmap_model_path=min_centered_instance_model_path
    )

    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2

    assert predictor.is_grayscale == True

    points_gt = np.concatenate(
        [min_labels[0][0].numpy(), min_labels[0][1].numpy()], axis=0
    )
    points_pr = np.concatenate(
        [labels_pr[0][0].numpy(), labels_pr[0][1].numpy()], axis=0
    )
    inds1, inds2 = sleap.nn.utils.match_points(points_gt, points_pr)
    assert_allclose(points_gt[inds1.numpy()], points_pr[inds2.numpy()], atol=1.5)


def test_topdown_predictor_centered_instance_high_threshold(
    min_labels, min_centered_instance_model_path
):
    predictor = TopDownPredictor.from_trained_models(
        confmap_model_path=min_centered_instance_model_path, peak_threshold=1.5
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 0


def test_bottomup_predictor(min_labels, min_bottomup_model_path):
    predictor = BottomUpPredictor.from_trained_models(
        model_path=min_bottomup_model_path
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2

    assert predictor.is_grayscale == True

    points_gt = np.concatenate(
        [min_labels[0][0].numpy(), min_labels[0][1].numpy()], axis=0
    )
    points_pr = np.concatenate(
        [labels_pr[0][0].numpy(), labels_pr[0][1].numpy()], axis=0
    )
    inds1, inds2 = sleap.nn.utils.match_points(points_gt, points_pr)
    assert_allclose(points_gt[inds1.numpy()], points_pr[inds2.numpy()], atol=1.75)

    # Test inference with score threshold too high
    predictor = BottomUpPredictor.from_trained_models(
        model_path=min_bottomup_model_path,
        min_line_scores=1.1,
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr[0]) == 0


def test_bottomup_predictor_high_peak_thresh(min_labels, min_bottomup_model_path):
    predictor = BottomUpPredictor.from_trained_models(
        model_path=min_bottomup_model_path, peak_threshold=1.5
    )
    predictor.verbosity = "none"
    labels_pr = predictor.predict(min_labels)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 0


def test_bottomup_multiclass_predictor(
    min_tracks_2node_labels, min_bottomup_multiclass_model_path
):
    labels_gt = sleap.Labels(min_tracks_2node_labels[[0]])
    predictor = BottomUpMultiClassPredictor.from_trained_models(
        model_path=min_bottomup_multiclass_model_path,
        peak_threshold=0.7,
        integral_refinement=False,
    )
    labels_pr = predictor.predict(labels_gt)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2

    inds1 = np.argsort([x.track.name for x in labels_gt[0]])
    inds2 = np.argsort([x.track.name for x in labels_pr[0]])
    assert labels_gt[0][inds1[0]].track == labels_pr[0][inds2[0]].track
    assert labels_gt[0][inds1[1]].track == labels_pr[0][inds2[1]].track

    assert_allclose(
        labels_gt[0][inds1[0]].numpy(), labels_pr[0][inds2[0]].numpy(), rtol=0.02
    )
    assert_allclose(
        labels_gt[0][inds1[1]].numpy(), labels_pr[0][inds2[1]].numpy(), rtol=0.02
    )

    labels_pr = predictor.predict(
        sleap.nn.data.pipelines.VideoReader(labels_gt.video, example_indices=[0])
    )
    labels_pr[0][0].track.name == "female"
    labels_pr[0][1].track.name == "male"


def test_bottomup_multiclass_predictor_high_threshold(
    min_tracks_2node_labels, min_bottomup_multiclass_model_path
):
    labels_gt = sleap.Labels(min_tracks_2node_labels[[0]])
    predictor = BottomUpMultiClassPredictor.from_trained_models(
        model_path=min_bottomup_multiclass_model_path,
        peak_threshold=1.5,
        integral_refinement=False,
    )
    labels_pr = predictor.predict(labels_gt)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 0


def test_topdown_multiclass_predictor(
    min_tracks_2node_labels, min_topdown_multiclass_model_path
):
    labels_gt = sleap.Labels(min_tracks_2node_labels[[0]])
    predictor = TopDownMultiClassPredictor.from_trained_models(
        confmap_model_path=min_topdown_multiclass_model_path,
        peak_threshold=0.7,
        integral_refinement=False,
    )
    labels_pr = predictor.predict(labels_gt)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 2

    inds1 = np.argsort([x.track.name for x in labels_gt[0]])
    inds2 = np.argsort([x.track.name for x in labels_pr[0]])
    assert labels_gt[0][inds1[0]].track == labels_pr[0][inds2[0]].track
    assert labels_gt[0][inds1[1]].track == labels_pr[0][inds2[1]].track

    assert_allclose(
        labels_gt[0][inds1[0]].numpy(), labels_pr[0][inds2[0]].numpy(), rtol=0.02
    )
    assert_allclose(
        labels_gt[0][inds1[1]].numpy(), labels_pr[0][inds2[1]].numpy(), rtol=0.02
    )


def test_topdown_multiclass_predictor_high_threshold(
    min_tracks_2node_labels, min_topdown_multiclass_model_path
):
    labels_gt = sleap.Labels(min_tracks_2node_labels[[0]])
    predictor = TopDownMultiClassPredictor.from_trained_models(
        confmap_model_path=min_topdown_multiclass_model_path,
        peak_threshold=1.5,
        integral_refinement=False,
    )
    labels_pr = predictor.predict(labels_gt)
    assert len(labels_pr) == 1
    assert len(labels_pr[0].instances) == 0


@pytest.mark.parametrize("resize_input_shape", [True, False])
@pytest.mark.parametrize(
    "model_fixture_name",
    [
        "min_centroid_model_path",
        "min_centered_instance_model_path",
        "min_bottomup_model_path",
        "min_single_instance_robot_model_path",
        "min_bottomup_multiclass_model_path",
        "min_topdown_multiclass_model_path",
    ],
)
def test_load_model(resize_input_shape, model_fixture_name, request):
    model_path = request.getfixturevalue(model_fixture_name)
    fname_mname_ptype_ishape = [
        ("centroid", "centroid_model", TopDownPredictor, (None, 384, 384, 1)),
        ("centered_instance", "confmap_model", TopDownPredictor, (None, 96, 96, 1)),
        ("bottomup_model", "bottomup_model", BottomUpPredictor, (None, 384, 384, 1)),
        (
            "single_instance",
            "confmap_model",
            SingleInstancePredictor,
            (None, 160, 280, 3),
        ),
        (
            "bottomup_multiclass",
            "model",
            BottomUpMultiClassPredictor,
            (None, 512, 512, 1),
        ),
        (
            "topdown_multiclass",
            "confmap_model",
            TopDownMultiClassPredictor,
            (None, 128, 128, 1),
        ),
    ]
    expected_model_name = None
    expected_predictor_type = None
    input_shape = None

    # Create predictor
    predictor = load_model(model_path, resize_input_layer=resize_input_shape)

    # Determine predictor type
    for (fname, mname, ptype, ishape) in fname_mname_ptype_ishape:
        if fname in model_fixture_name:
            expected_model_name = mname
            expected_predictor_type = ptype
            input_shape = ishape
            break

    # Assert predictor type and model input shape are correct
    assert isinstance(predictor, expected_predictor_type)
    keras_model = getattr(predictor, expected_model_name).keras_model
    if resize_input_shape:
        assert keras_model.input_shape == (None, None, None, input_shape[-1])
    else:
        assert keras_model.input_shape == input_shape


def test_topdown_multi_size_inference(
    min_centroid_model_path,
    min_centered_instance_model_path,
    centered_pair_labels,
    min_tracks_2node_labels,
):
    imgs = centered_pair_labels.video[:2]
    assert imgs.shape == (2, 384, 384, 1)

    predictor = load_model(
        [min_centroid_model_path, min_centered_instance_model_path],
        resize_input_layer=True,
    )
    preds = predictor.predict(imgs)
    assert len(preds) == 2

    imgs = min_tracks_2node_labels.video[:2]
    assert imgs.shape == (2, 1024, 1024, 1)
    preds = predictor.predict(imgs)
    assert len(preds) == 2


def test_ensure_numpy(
    min_centroid_model_path, min_centered_instance_model_path, min_labels_slp
):

    model = load_model([min_centroid_model_path, min_centered_instance_model_path])

    # each frame has same number of instances
    same_shape = min_labels_slp.video[:4]

    out = model.inference_model.predict(same_shape, numpy=False)

    assert type(out["instance_peaks"]) == tf.RaggedTensor
    assert type(out["instance_peak_vals"]) == tf.RaggedTensor
    assert type(out["centroids"]) == tf.RaggedTensor
    assert type(out["centroid_vals"]) == tf.RaggedTensor

    out = model.inference_model.predict(same_shape, numpy=True)

    assert type(out["instance_peaks"]) == np.ndarray
    assert type(out["instance_peak_vals"]) == np.ndarray
    assert type(out["centroids"]) == np.ndarray
    assert type(out["centroid_vals"]) == np.ndarray
    assert type(out["n_valid"]) == np.ndarray

    out = model.inference_model.predict_on_batch(same_shape, numpy=False)

    assert type(out["instance_peaks"]) == tf.RaggedTensor
    assert type(out["instance_peak_vals"]) == tf.RaggedTensor
    assert type(out["centroids"]) == tf.RaggedTensor
    assert type(out["centroid_vals"]) == tf.RaggedTensor

    out = model.inference_model.predict_on_batch(same_shape, numpy=True)

    assert type(out["instance_peaks"]) == np.ndarray
    assert type(out["instance_peak_vals"]) == np.ndarray
    assert type(out["centroids"]) == np.ndarray
    assert type(out["centroid_vals"]) == np.ndarray
    assert type(out["n_valid"]) == np.ndarray

    # variable number of instances
    diff_shape = min_labels_slp.video[4:8]

    out = model.inference_model.predict(diff_shape, numpy=False)

    assert type(out["instance_peaks"]) == tf.RaggedTensor
    assert type(out["instance_peak_vals"]) == tf.RaggedTensor
    assert type(out["centroids"]) == tf.RaggedTensor
    assert type(out["centroid_vals"]) == tf.RaggedTensor

    out = model.inference_model.predict(diff_shape, numpy=True)

    assert type(out["instance_peaks"]) == np.ndarray
    assert type(out["instance_peak_vals"]) == np.ndarray
    assert type(out["centroids"]) == np.ndarray
    assert type(out["centroid_vals"]) == np.ndarray
    assert type(out["n_valid"]) == np.ndarray

    out = model.inference_model.predict_on_batch(diff_shape, numpy=False)

    assert type(out["instance_peaks"]) == tf.RaggedTensor
    assert type(out["instance_peak_vals"]) == tf.RaggedTensor
    assert type(out["centroids"]) == tf.RaggedTensor
    assert type(out["centroid_vals"]) == tf.RaggedTensor

    out = model.inference_model.predict_on_batch(diff_shape, numpy=True)

    assert type(out["instance_peaks"]) == np.ndarray
    assert type(out["instance_peak_vals"]) == np.ndarray
    assert type(out["centroids"]) == np.ndarray
    assert type(out["centroid_vals"]) == np.ndarray
    assert type(out["n_valid"]) == np.ndarray


def test_centroid_inference():

    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[[1.75, 2.75]], [[3.75, 4.75]], [[5.75, 6.75]]], tf.float32)
    cms = tf.expand_dims(make_multi_confmaps(points, xv, yv, sigma=1.5), axis=0)

    x_in = tf.keras.layers.Input([12, 12, 1])
    x_out = tf.keras.layers.Lambda(lambda x: x, name="CentroidConfmapsHead")(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    layer = CentroidCrop(
        keras_model=model,
        input_scale=1.0,
        crop_size=3,
        pad_to_stride=1,
        output_stride=None,
        refinement="local",
        integral_patch_size=5,
        peak_threshold=0.2,
        return_confmaps=False,
        return_crops=False,
    )

    # For Codecov to realize the wrapped CentroidCrop.call is tested/covered,
    # we need to unbind CentroidCrop.call from its bind with TfMethodTarget object
    # and then rebind the standalone function with the CentroidCrop object
    TfMethodTarget_object = layer.call.__wrapped__.__self__  # Get the bound object
    original_func = TfMethodTarget_object.weakrefself_func__()  # Get unbound function
    layer.call = original_func.__get__(layer, layer.__class__)  # Bind function

    out = layer(cms)
    assert tuple(out["centroids"].shape) == (1, None, 2)
    assert tuple(out["centroid_vals"].shape) == (1, None)

    assert tuple(out["centroids"].bounding_shape()) == (1, 3, 2)
    assert tuple(out["centroid_vals"].bounding_shape()) == (1, 3)

    assert_allclose(out["centroids"][0].numpy(), points.numpy().squeeze(axis=1))
    assert_allclose(out["centroid_vals"][0].numpy(), [1, 1, 1], atol=0.1)

    model = CentroidInferenceModel(layer)

    preds = model.predict(cms)

    assert preds["centroids"].shape == (1, 3, 2)
    assert preds["centroid_vals"].shape == (1, 3)

    # test max instances (>3 will fail)
    layer.max_instances = 3
    out = layer(cms)

    model = CentroidInferenceModel(layer)

    preds = model.predict(cms)


def export_frozen_graph(model, preds, output_path):

    tensors = {}

    for key, val in preds.items():
        dtype = str(val.dtype) if isinstance(val.dtype, np.dtype) else repr(val.dtype)
        tensors[key] = {
            "type": f"{type(val).__name__}",
            "shape": f"{val.shape}",
            "dtype": dtype,
            "device": f"{val.device if hasattr(val, 'device') else 'N/A'}",
        }

    with output_path as d:
        model.export_model(d.as_posix(), tensors=tensors)

        tf.compat.v1.reset_default_graph()
        with tf.compat.v2.io.gfile.GFile(f"{d}/frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        with open(f"{d}/info.json") as json_file:
            info = json.load(json_file)

        for tensor_info in info["frozen_model_inputs"] + info["frozen_model_outputs"]:

            saved_name = (
                tensor_info.split("Tensor(")[1].split(", shape")[0].replace('"', "")
            )
            saved_shape = ast.literal_eval(
                tensor_info.split("shape=", 1)[1].split("), ")[0] + ")"
            )
            saved_dtype = tensor_info.split("dtype=")[1].split(")")[0]

            loaded_shape = tuple(graph.get_tensor_by_name(f"import/{saved_name}").shape)
            loaded_dtype = graph.get_tensor_by_name(f"import/{saved_name}").dtype.name

            assert saved_shape == loaded_shape
            assert saved_dtype == loaded_dtype


def test_single_instance_save(min_single_instance_robot_model_path, tmp_path):

    single_instance_model = tf.keras.models.load_model(
        min_single_instance_robot_model_path + "/best_model.h5", compile=False
    )

    model = SingleInstanceInferenceModel(
        SingleInstanceInferenceLayer(keras_model=single_instance_model)
    )

    preds = model.predict(np.zeros((4, 160, 280, 3), dtype="uint8"))

    export_frozen_graph(model, preds, tmp_path)


def test_centroid_save(min_centroid_model_path, tmp_path):

    centroid_model = tf.keras.models.load_model(
        min_centroid_model_path + "/best_model.h5", compile=False
    )

    centroid = CentroidCrop(
        keras_model=centroid_model, crop_size=160, return_crops=False
    )

    model = CentroidInferenceModel(centroid)

    preds = model.predict(np.zeros((4, 384, 384, 1), dtype="uint8"))

    export_frozen_graph(model, preds, tmp_path)


def test_topdown_save(
    min_centroid_model_path, min_centered_instance_model_path, min_labels_slp, tmp_path
):

    centroid_model = tf.keras.models.load_model(
        min_centroid_model_path + "/best_model.h5", compile=False
    )

    top_down_model = tf.keras.models.load_model(
        min_centered_instance_model_path + "/best_model.h5", compile=False
    )

    centroid = CentroidCrop(keras_model=centroid_model, crop_size=96)

    instance_peaks = FindInstancePeaks(keras_model=top_down_model)

    model = TopDownInferenceModel(centroid, instance_peaks)

    imgs = min_labels_slp.video[:4]
    preds = model.predict(imgs)

    export_frozen_graph(model, preds, tmp_path)


def test_topdown_id_save(
    min_centroid_model_path, min_topdown_multiclass_model_path, min_labels_slp, tmp_path
):

    centroid_model = tf.keras.models.load_model(
        min_centroid_model_path + "/best_model.h5", compile=False
    )

    top_down_id_model = tf.keras.models.load_model(
        min_topdown_multiclass_model_path + "/best_model.h5", compile=False
    )

    centroid = CentroidCrop(keras_model=centroid_model, crop_size=128)

    instance_peaks = TopDownMultiClassFindPeaks(keras_model=top_down_id_model)

    model = TopDownMultiClassInferenceModel(centroid, instance_peaks)

    imgs = min_labels_slp.video[:4]
    preds = model.predict(imgs)

    export_frozen_graph(model, preds, tmp_path)


def test_single_instance_predictor_save(min_single_instance_robot_model_path, tmp_path):

    # directly initialize predictor
    predictor = SingleInstancePredictor.from_trained_models(
        min_single_instance_robot_model_path, resize_input_layer=False
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level load to predictor
    predictor = load_model(
        min_single_instance_robot_model_path, resize_input_layer=False
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level export (with unragging)
    export_model(min_single_instance_robot_model_path, save_path=tmp_path.as_posix())
    cmd = f"-m {min_single_instance_robot_model_path} -e {tmp_path.as_posix()}"
    sleap_export(cmd.split())

    # high level export (without unragging)
    export_model(
        min_single_instance_robot_model_path,
        save_path=tmp_path.as_posix(),
        unrag_outputs=False,
    )

    # max_instances should raise an exception for single instance
    with pytest.raises(Exception):
        export_model(
            min_single_instance_robot_model_path,
            save_path=tmp_path.as_posix(),
            unrag_outputs=False,
            max_instances=1,
        )


def test_topdown_predictor_save(
    min_centroid_model_path, min_centered_instance_model_path, tmp_path
):

    # directly initialize predictor
    predictor = TopDownPredictor.from_trained_models(
        centroid_model_path=min_centroid_model_path,
        confmap_model_path=min_centered_instance_model_path,
        resize_input_layer=False,
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level load to predictor
    predictor = load_model(
        [min_centroid_model_path, min_centered_instance_model_path],
        resize_input_layer=False,
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level export (with unragging)
    export_model(
        [min_centroid_model_path, min_centered_instance_model_path],
        save_path=tmp_path.as_posix(),
    )

    # high level export (without unragging)
    export_model(
        [min_centroid_model_path, min_centered_instance_model_path],
        save_path=tmp_path.as_posix(),
        unrag_outputs=False,
    )

    # test max instances
    export_model(
        [min_centroid_model_path, min_centered_instance_model_path],
        save_path=tmp_path.as_posix(),
        unrag_outputs=False,
        max_instances=4,
    )


def test_topdown_id_predictor_save(
    min_centroid_model_path, min_topdown_multiclass_model_path, tmp_path
):

    # directly initialize predictor
    predictor = TopDownMultiClassPredictor.from_trained_models(
        centroid_model_path=min_centroid_model_path,
        confmap_model_path=min_topdown_multiclass_model_path,
        resize_input_layer=False,
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level load to predictor
    predictor = load_model(
        [min_centroid_model_path, min_topdown_multiclass_model_path],
        resize_input_layer=False,
    )

    predictor.export_model(save_path=tmp_path.as_posix())

    # high level export (with unragging)
    export_model(
        [min_centroid_model_path, min_topdown_multiclass_model_path],
        save_path=tmp_path.as_posix(),
    )

    # high level export (without unragging)
    export_model(
        [min_centroid_model_path, min_topdown_multiclass_model_path],
        save_path=tmp_path.as_posix(),
        unrag_outputs=False,
    )

    # test max instances
    export_model(
        [min_centroid_model_path, min_topdown_multiclass_model_path],
        save_path=tmp_path.as_posix(),
        unrag_outputs=False,
        max_instances=4,
    )


@pytest.mark.parametrize(
    "output_path,tracker_method", [("not_default", "flow"), (None, "simple")]
)
def test_retracking(
    centered_pair_predictions: Labels, tmpdir, output_path, tracker_method
):
    slp_path = Path(tmpdir, "old_slp.slp")
    labels: Labels = Labels.save(centered_pair_predictions, slp_path)

    # Create sleap-track command
    cmd = (
        f"{slp_path} --tracking.tracker {tracker_method} --video.index 0 --frames 1-3 "
        "--cpu"
    )
    if tracker_method == "flow":
        cmd += " --tracking.save_shifted_instances 1"
    if output_path == "not_default":
        output_path = Path(tmpdir, "tracked_slp.slp")
        cmd += f" --output {output_path}"
    args = f"{cmd}".split()

    # Track predictions
    sleap_track(args=args)

    # Get expected output name
    if output_path is None:
        parser = _make_cli_parser()
        args, _ = parser.parse_known_args(args=args)
        tracker = _make_tracker_from_cli(args)
        output_path = f"{slp_path}.{tracker.get_name()}.slp"

    # Assert tracked predictions file exists
    assert Path(output_path).exists()

    # Assert tracking has changed
    def load_instance(labels_in: Labels):
        lf = labels_in.get(0)
        return lf.instances[0]

    new_labels = Labels.load_file(str(output_path))
    new_inst = load_instance(new_labels)
    old_inst = load_instance(centered_pair_predictions)
    assert new_inst.track != old_inst.track


def test_sleap_track(
    centered_pair_predictions: Labels,
    min_centroid_model_path: str,
    min_centered_instance_model_path: str,
    tmpdir,
):
    slp_path = str(Path(tmpdir, "old_slp.slp"))
    labels: Labels = Labels.save(centered_pair_predictions, slp_path)

    # Create sleap-track command
    args = f"{slp_path} --model {min_centered_instance_model_path} --frames 1-3 --cpu".split()
    args = (
        f"{slp_path} --model {min_centroid_model_path} "
        f"--model {min_centered_instance_model_path} --video.index 0 --frames 1-3 --cpu"
    ).split()

    # Run inference
    sleap_track(args=args)

    # Assert predictions file exists
    output_path = f"{slp_path}.predictions.slp"
    assert Path(output_path).exists()

    # Create invalid sleap-track command
    args = [slp_path, "--cpu"]
    with pytest.raises(ValueError):
        sleap_track(args=args)


def test_flow_tracker(centered_pair_predictions: Labels, tmpdir):
    """Test flow tracker instances are pruned."""
    labels: Labels = centered_pair_predictions
    track_window = 5

    # Setup tracker
    tracker: Tracker = Tracker.make_tracker_by_name(
        tracker="flow", track_window=track_window, save_shifted_instances=True
    )
    tracker.candidate_maker = cast(FlowCandidateMaker, tracker.candidate_maker)

    # Run tracking
    frames = sorted(labels.labeled_frames, key=lambda lf: lf.frame_idx)

    # Run tracking on subset of frames using psuedo-implementation of
    # sleap.nn.tracking.run_tracker
    for lf in frames[:20]:

        # Clear the tracks
        for inst in lf.instances:
            inst.track = None

        track_args = dict(untracked_instances=lf.instances, img=lf.video[lf.frame_idx])
        tracker.track(**track_args)

        # Check that saved instances are pruned to track window
        for key in tracker.candidate_maker.shifted_instances.keys():
            assert lf.frame_idx - key[0] <= track_window  # Keys are pruned
            assert abs(key[0] - key[1]) <= track_window  # References within window


def test_top_down_model(
    centered_pair_predictions: Labels, min_centroid_model_path: str
):
    labels = centered_pair_predictions
    video = sleap.load_video(labels.videos[0].backend.filename)
    predictor = sleap.load_model(min_centroid_model_path, batch_size=16)

    # Preload images
    imgs = video[:1024]

    with pytest.raises(TypeError):
        predictor.predict(imgs[:16])  # warmup

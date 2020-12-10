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
    InferenceLayer,
    InferenceModel,
    CentroidCrop,
    get_model_output_stride,
    SingleInstanceInferenceLayer,
    SingleInstanceInferenceModel
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


def test_centroid_crop_layer():
    xv, yv = make_grid_vectors(image_height=12, image_width=12, output_stride=1)
    points = tf.cast([[[1.75, 2.75]], [[3.75, 4.75]], [[5.75, 6.75]]], tf.float32)
    cms = tf.expand_dims(make_multi_confmaps(points, xv, yv, sigma=1.5), axis=0)

    x_in = tf.keras.layers.Input([12, 12, 1])
    x_out = tf.keras.layers.Lambda(lambda x: x)(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    layer = CentroidCrop(
        keras_model=model, input_scale=1.0, crop_size=3, pad_to_stride=1,
        output_stride=None, refinement="local", integral_patch_size=5,
        peak_threshold=0.2, return_confmaps=False
    )

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
    x_out = tf.keras.layers.Lambda(lambda x: x)(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    instance_peaks_layer = FindInstancePeaks(
        keras_model=model, input_scale=1.0, peak_threshold=0.2, return_confmaps=False,
        refinement="integral"
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
        keras_model=model, input_scale=1.0, peak_threshold=0.2, return_confmaps=True,
        refinement="integral"
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
        keras_model=model, input_scale=scale, peak_threshold=0.2, return_confmaps=False,
        refinement="integral"
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
        keras_model=keras_model,
        input_scale=1.0,
        pad_to_stride=1,
        ensure_grayscale=None
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
        keras_model=keras_model,
        input_scale=1.0,
        pad_to_stride=1,
        ensure_grayscale=None
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
        keras_model=keras_model,
        input_scale=1.0,
        pad_to_stride=1,
        ensure_grayscale=None
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
        keras_model=keras_model,
        input_scale=0.5,
        pad_to_stride=1,
        ensure_grayscale=None
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
        keras_model=keras_model,
        input_scale=1,
        pad_to_stride=2,
        ensure_grayscale=None
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
        keras_model=keras_model,
        input_scale=0.5,
        pad_to_stride=2,
        ensure_grayscale=None
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
    x = tf.keras.layers.Lambda(lambda x: x)(x_in)
    keras_model = tf.keras.Model(x_in, x)

    layer = SingleInstanceInferenceLayer(keras_model=keras_model, refinement="local")
    assert layer.output_stride == 1

    out = layer(cms)
    assert_array_equal(out["peaks"], points)
    assert_allclose(out["peak_vals"], 1.0, atol=0.1)
    assert "confmaps" not in out

    out = layer({"image": cms})
    assert_array_equal(out["peaks"], points)

    layer = SingleInstanceInferenceLayer(keras_model=keras_model, refinement="local",
        return_confmaps=True)
    out = layer(cms)
    assert "confmaps" in out
    assert_array_equal(out["confmaps"], cms)

    model = SingleInstanceInferenceModel(layer)
    preds = model.predict(cms)
    assert_array_equal(preds["peaks"], points)
    assert "peak_vals" in preds
    assert "confmaps" in preds

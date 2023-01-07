import sleap
import tensorflow as tf
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from sleap.nn.inference import TopDownPredictor
from sleap.nn.utils import tf_linear_sum_assignment, match_points, reset_input_layer

sleap.use_cpu_only()


def test_tf_linear_sum_assignment():
    r, c = tf_linear_sum_assignment(tf.cast([[-1, 0], [0, -1]], tf.float32))
    assert_array_equal(r, [0, 1])
    assert_array_equal(c, [0, 1])
    assert r.dtype == tf.int32
    assert c.dtype == tf.int32


def test_match_points():
    inds1, inds2 = match_points([[0, 0], [1, 2]], [[1, 2], [0, 0]])

    assert_array_equal(inds1, [0, 1])
    assert_array_equal(inds2, [1, 0])


def test_reset_input_layer(min_centroid_model_path):
    """Verify that input layer size is reset."""

    predictor = TopDownPredictor.from_trained_models(
        centroid_model_path=min_centroid_model_path, resize_input_layer=False
    )
    og_keras_model: tf.keras.Model = predictor.centroid_model.keras_model
    og_weights = [layer.get_weights() for layer in og_keras_model.layers[1:]]
    assert og_keras_model.input_shape == (None, 384, 384, 1)

    keras_model = reset_input_layer(keras_model=og_keras_model)
    new_weights = [layer.get_weights() for layer in keras_model.layers[1:]]
    assert keras_model.input_shape == (None, None, None, 1)
    assert len(keras_model.layers) == len(og_keras_model.layers)
    for og_weight, new_weight in zip(og_weights, new_weights):
        for ogw, nw in zip(og_weight, new_weight):
            assert ogw.shape == nw.shape
            np.testing.assert_array_equal(ogw.flatten(), nw.flatten())

    new_shape = (None, 384, 384, 1)
    keras_model = reset_input_layer(keras_model=keras_model, new_shape=new_shape)
    assert keras_model.input_shape == new_shape

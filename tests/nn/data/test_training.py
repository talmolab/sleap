import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.data import training


def test_keymapper():
    ds = tf.data.Dataset.from_tensors({"a": 0, "b": 1})
    mapper = training.KeyMapper(key_maps={"a": "x", "b": "y"})
    ds = mapper.transform_dataset(ds)
    np.testing.assert_array_equal(next(iter(ds)), {"x": 0, "y": 1})
    assert mapper.input_keys == ["a", "b"]
    assert mapper.output_keys == ["x", "y"]

    ds = tf.data.Dataset.from_tensors({"a": 0, "b": 1})
    ds = training.KeyMapper(key_maps=[{"a": "x"}, {"b": "y"}]).transform_dataset(ds)
    np.testing.assert_array_equal(next(iter(ds)), ({"x": 0}, {"y": 1}))
    assert mapper.input_keys == ["a", "b"]
    assert mapper.output_keys == ["x", "y"]

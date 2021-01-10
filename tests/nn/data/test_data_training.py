import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only; use_cpu_only()  # hide GPUs for test

import sleap
from sleap.nn.data.providers import LabelsReader
from sleap.nn.data import training


def test_split_labels_reader(min_labels):
    labels = sleap.Labels([min_labels[0], min_labels[0], min_labels[0], min_labels[0]])
    labels_reader = LabelsReader(labels)
    reader1, reader2 = training.split_labels_reader(labels_reader, [0.5, 0.5])
    assert len(reader1) == 2
    assert len(reader2) == 2
    assert len(set(reader1.example_indices).intersection(set(reader2.example_indices))) == 0

    reader1, reader2 = training.split_labels_reader(labels_reader, [0.1, 0.5])
    assert len(reader1) == 1
    assert len(reader2) == 2
    assert len(set(reader1.example_indices).intersection(set(reader2.example_indices))) == 0

    reader1, reader2 = training.split_labels_reader(labels_reader, [0.1, -1])
    assert len(reader1) == 1
    assert len(reader2) == 3
    assert len(set(reader1.example_indices).intersection(set(reader2.example_indices))) == 0

    labels = sleap.Labels([min_labels[0], min_labels[0], min_labels[0], min_labels[0]])
    labels_reader = LabelsReader(labels, example_indices=[1, 2, 3])
    reader1, reader2 = training.split_labels_reader(labels_reader, [0.1, -1])
    assert len(reader1) == 1
    assert len(reader2) == 2
    assert len(set(reader1.example_indices).intersection(set(reader2.example_indices))) == 0
    assert 0 not in reader1.example_indices
    assert 0 not in reader2.example_indices


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

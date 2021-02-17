"""Transformers and utilities for training-related operations."""

import numpy as np
import tensorflow as tf
import sleap
from sleap.nn.data.providers import LabelsReader
from sleap.nn.data.utils import expand_to_rank, ensure_list
import attr
from typing import List, Text, Optional, Any, Union, Dict, Tuple, Sequence
from sklearn.model_selection import train_test_split


def split_labels_train_val(
    labels: sleap.Labels, validation_fraction: float
) -> Tuple[sleap.Labels, List[int], sleap.Labels, List[int]]:
    """Make a train/validation split from a labels dataset.

    Args:
        labels: A `sleap.Labels` dataset with labeled frames.
        validation_fraction: Fraction of frames to use for validation.

    Returns:
        A tuple of `(labels_train, idx_train, labels_val, idx_val)`.

        `labels_train` and `labels_val` are `sleap.Label` objects containing the
        selected frames for each split. Their `videos`, `tracks` and `provenance`
        attributes are identical to `labels` even if the split does not contain
        instances with a particular video or track.

        `idx_train` and `idx_val` are list indices of the labeled frames within the
        input labels that were assigned to each split, i.e.:

            `labels[idx_train] == labels_train[:]`

        If there is only one labeled frame in `labels`, both of the labels will contain
        the same frame.

        If `validation_fraction` would result in fewer than one label for either split,
        it will be rounded to ensure there is at least one label in each.
    """
    if len(labels) == 1:
        return labels, [0], labels, [0]

    # Split indices.
    n_val = round(len(labels) * validation_fraction)
    n_val = max(min(n_val, len(labels) - 1), 1)

    idx_train, idx_val = train_test_split(list(range(len(labels))), test_size=n_val)

    # Create labels and keep original metadata.
    labels_train = sleap.Labels(labels[idx_train])
    labels_train.videos = labels.videos
    labels_train.tracks = labels.tracks
    labels_train.provenance = labels.provenance

    labels_val = sleap.Labels(labels[idx_val])
    labels_val.videos = labels.videos
    labels_val.tracks = labels.tracks
    labels_val.provenance = labels.provenance

    return labels_train, idx_train, labels_val, idx_val


def split_labels(
    labels: sleap.Labels, split_fractions: Sequence[float]
) -> Tuple[sleap.Labels]:
    """Split a `sleap.Labels` into multiple new ones with random subsets of the data.

    Args:
        labels: An instance of `sleap.Labels`.
        split_fractions: One or more floats between 0 and 1 that specify the fraction of
            examples that should be in each dataset. These should add up to <= 1.0.
            Fractions of less than 1 element will be rounded up to ensure that is at
            least 1 element in each split. One of the fractions may be -1 to indicate
            that it should contain all elements left over from the other splits.

    Returns:
        A tuple of new `sleap.Labels` instances of the same length as `split_fractions`.

    Raises:
        ValueError: If more than one split fraction is specified as -1.
        ValueError: If the splits add up to more than the total available examples.

    Note:
        Sampling is done without replacement.
    """
    # Get indices for labeled frames.
    labels_indices = np.arange(len(labels)).astype("int64")

    # Compute split sizes.
    n_examples = len(labels_indices)
    n_examples_per_split = np.array(split_fractions).astype("float64")
    if (n_examples_per_split == -1).sum() > 1:
        raise ValueError("Only one split fraction can be specified as -1.")
    n_examples_per_split[n_examples_per_split == -1] = np.NaN
    n_examples_per_split = np.ceil(n_examples_per_split * n_examples)
    n_examples_per_split[np.isnan(n_examples_per_split)] = np.maximum(
        n_examples - np.nansum(n_examples_per_split), 1
    )
    n_examples_per_split = n_examples_per_split.astype("int64")
    if n_examples_per_split.sum() > n_examples:
        raise ValueError("Splits cannot sum to more than the total input labels.")

    # Sample and create new Labels instances.
    split_labels = []
    for n_samples in n_examples_per_split:
        # Sample.
        sampled_indices = np.random.default_rng().choice(
            labels_indices, size=n_samples, replace=False
        )

        # Create new instance.
        split_labels.append(sleap.Labels([labels[int(ind)] for ind in sampled_indices]))

        # Exclude the sampled indices from the available indices.
        labels_indices = np.setdiff1d(labels_indices, sampled_indices)

    return tuple(split_labels)


def split_labels_reader(
    labels_reader: LabelsReader, split_fractions: Sequence[float]
) -> Tuple[LabelsReader]:
    """Split a `LabelsReader` into multiple new ones with random subsets of the data.

    Args:
        labels_reader: An instance of `sleap.nn.data.providers.LabelsReader`. This is a
            provider that generates datasets that contain elements read from a
            `sleap.Labels` instance.
        split_fractions: One or more floats between 0 and 1 that specify the fraction of
            examples that should be in each dataset. These should add up to <= 1.0.
            Fractions of less than 1 element will be rounded up to ensure that is at
            least 1 element in each split. One of the fractions may be -1 to indicate
            that it should contain all elements left over from the other splits.

    Returns:
        A tuple of `LabelsReader` instances of the same length as `split_fractions`. The
        indices will be stored in the `example_indices` in each `LabelsReader` instance.

        The actual `sleap.Labels` instance will be the same for each instance, only the
        `example_indices` that are iterated over will change across splits.

        If the input `labels_reader` already has `example_indices`, a subset of these
        will be sampled to generate the splits.

    Raises:
        ValueError: If more than one split fraction is specified as -1.
        ValueError: If the splits add up to more than the total available examples.

    Note:
        Sampling is done without replacement.
    """
    # Get available indices.
    labels_indices = labels_reader.example_indices
    if labels_indices is None:
        labels_indices = np.arange(len(labels_reader))
    labels_indices = np.array(labels_indices).astype("int64")

    # Compute split sizes.
    n_examples = len(labels_indices)
    n_examples_per_split = np.array(split_fractions).astype("float64")
    if (n_examples_per_split == -1).sum() > 1:
        raise ValueError("Only one split fraction can be specified as -1.")
    n_examples_per_split[n_examples_per_split == -1] = np.NaN
    n_examples_per_split = np.ceil(n_examples_per_split * n_examples)
    n_examples_per_split[np.isnan(n_examples_per_split)] = np.maximum(
        n_examples - np.nansum(n_examples_per_split), 1
    )
    n_examples_per_split = n_examples_per_split.astype("int64")
    if n_examples_per_split.sum() > n_examples:
        raise ValueError("Splits cannot sum to more than the total input labels.")

    # Sample and create new LabelsReader instances.
    split_readers = []
    for n_samples in n_examples_per_split:
        # Sample.
        sampled_indices = np.random.default_rng().choice(
            labels_indices, size=n_samples, replace=False
        )

        # Create new instance.
        split_readers.append(
            LabelsReader(labels_reader.labels, example_indices=sampled_indices)
        )

        # Exclude the sampled indices from the available indices.
        labels_indices = np.setdiff1d(labels_indices, sampled_indices)

    return tuple(split_readers)


@attr.s(auto_attribs=True)
class KeyMapper:
    """Maps example keys to specified outputs.

    This is useful for transforming examples into tuples that map onto specific layer
    names for training.

    Attributes:
        key_maps: Dictionary or list of dictionaries with string keys and values of
            the form: {input_key: output_key}. If a list, the examples will be in tuples
            in the same order.
    """

    key_maps: List[Dict[Text, Text]] = attr.ib(
        converter=attr.converters.optional(ensure_list)
    )

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        input_keys = []
        for key_map in self.key_maps:
            input_keys.extend(list(key_map.keys()))
        return input_keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have. These may be nested."""
        output_keys = []
        for key_map in self.key_maps:
            output_keys.extend(list(key_map.values()))
        return output_keys

    def transform_dataset(self, ds_input: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset with input keys mapped to new key names.

        Args:
            ds_input: Any `tf.data.Dataset` that generates examples as a dictionary of
                tensors with the keys in `input_keys`.

        Return:
            A dataset that generates examples with the tensors in `input_keys` mapped to
            keys in `output_keys` according to the structure in `key_maps`.
        """

        def map_keys(example):
            """Local processing function for dataset mapping."""
            output_keys = []
            for key_map in self.key_maps:
                output_keys.append(
                    {key_out: example[key_in] for key_in, key_out in key_map.items()}
                )
            return tuple(output_keys)

        ds_output = ds_input.map(map_keys)
        return ds_output

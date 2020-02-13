"""Transformers and utilities for training-related operations."""

import numpy as np
import tensorflow as tf
from sleap.nn.data.utils import expand_to_rank, ensure_list
import attr
from typing import List, Text, Optional, Any, Union, Dict




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

    key_maps: List[Dict[Text, Text]] = attr.ib(converter=attr.converters.optional(ensure_list))

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
                output_keys.append({key_out: example[key_in] for key_in, key_out in key_map.items()})
            return tuple(output_keys)
        ds_output = ds_input.map(map_keys)
        return ds_output


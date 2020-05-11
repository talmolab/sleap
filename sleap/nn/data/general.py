"""General purpose transformers for common pipeline processing tasks."""

import tensorflow as tf
import attr
from typing import List, Text


@attr.s(auto_attribs=True)
class KeyRenamer:
    """Transformer for renaming example keys."""

    old_key_names: List[Text] = attr.ib(factory=list)
    new_key_names: List[Text] = attr.ib(factory=list)
    drop_old: bool = True

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return self.old_key_names

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        if self.drop_old:
            return self.new_key_names
        else:
            return self.old_key_names + self.new_key_names

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains filtered data."""

        def rename_keys(example):
            """Local processing function for dataset mapping."""
            for old_key, new_key in zip(self.old_key_names, self.new_key_names):
                example[new_key] = example[old_key]
            if self.drop_old:
                for old_key in self.old_key_names:
                    example.pop(old_key)
            return example

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            rename_keys, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds


@attr.s(auto_attribs=True)
class KeyFilter:
    """Transformer for filtering example keys."""

    keep_keys: List[Text] = attr.ib(factory=list)

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return self.keep_keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.keep_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains filtered data."""

        def filter_keys(example):
            """Local processing function for dataset mapping."""
            return {key: example[key] for key in self.keep_keys}

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            filter_keys, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds


@attr.s(auto_attribs=True)
class KeyDeviceMover:
    """Transformer for moving example keys to cpu."""

    keys: List[Text] = attr.ib(factory=list)

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return self.keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains data but moved to cpu."""

        def move_keys(example):
            """Local processing function for dataset mapping."""
            with tf.device("/cpu:0"):
                for key in self.keys:
                    if key in example:
                        example[key] = tf.identity(example[key])
            for key in self.keys:
                print(f"{key} on {example[key].device}")
            return example

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            move_keys, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds

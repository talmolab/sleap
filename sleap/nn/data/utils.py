"""Miscellaneous utility functions for data processing."""

import tensorflow as tf
from typing import Any, List


def ensure_list(x: Any) -> List[Any]:
    """Convert the input into a list if it is not already."""
    if not isinstance(x, list):
        return [x]
    return x


def expand_to_rank(x: tf.Tensor, target_rank: int, prepend: bool = True) -> tf.Tensor:
    """Expand a tensor to a target rank by adding singleton dimensions.

    Args:
        x: Any `tf.Tensor` with rank <= `target_rank`.
        target_rank: Rank to expand the input to.
        prepend: If True, singleton dimensions are added before the first axis of the
            data. If False, singleton dimensions are added after the last axis.

    Returns:
        The expanded tensor of the same dtype as the input, but with rank `target_rank`.

        The output has the same exact data as the input tensor and will be identical if
        they are both flattened.
    """
    singleton_dims = tf.ones([target_rank - tf.rank(x)], tf.int32)
    if prepend:
        new_shape = tf.concat([singleton_dims, tf.shape(x)], axis=0)
    else:
        new_shape = tf.concat([tf.shape(x), singleton_dims], axis=0)
    return tf.reshape(x, shape=new_shape)

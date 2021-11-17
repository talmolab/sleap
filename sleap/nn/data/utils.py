"""Miscellaneous utility functions for data processing."""

import tensorflow as tf
from keras.utils import tf_utils
import numpy as np
from typing import Any, List, Tuple, Dict, Text, Optional


def ensure_list(x: Any) -> List[Any]:
    """Convert the input into a list if it is not already."""
    if not isinstance(x, list):
        return [x]
    return x


def expand_to_rank(x: tf.Tensor, target_rank: int, prepend: bool = True) -> tf.Tensor:
    """Expand a tensor to a target rank by adding singleton dimensions.

    Args:
        x: Any `tf.Tensor` with rank <= `target_rank`. If the rank is higher than
            `target_rank`, the tensor will be returned with the same shape.
        target_rank: Rank to expand the input to.
        prepend: If True, singleton dimensions are added before the first axis of the
            data. If False, singleton dimensions are added after the last axis.

    Returns:
        The expanded tensor of the same dtype as the input, but with rank `target_rank`.

        The output has the same exact data as the input tensor and will be identical if
        they are both flattened.
    """
    n_singleton_dims = tf.maximum(target_rank - tf.rank(x), 0)
    singleton_dims = tf.ones([n_singleton_dims], tf.int32)
    if prepend:
        new_shape = tf.concat([singleton_dims, tf.shape(x)], axis=0)
    else:
        new_shape = tf.concat([tf.shape(x), singleton_dims], axis=0)
    return tf.reshape(x, shape=new_shape)


def make_grid_vectors(
    image_height: int, image_width: int, output_stride: int = 1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Make sampling grid vectors from image dimensions.

    This is a useful function for creating the x- and y-vectors that define a sampling
    grid over an image space. These vectors can be used to generate a full meshgrid or
    for equivalent broadcasting operations.

    Args:
        image_height: Height of the image grid that will be sampled, specified as a
            scalar integer.
        image_width: width of the image grid that will be sampled, specified as a
            scalar integer.
        output_stride: Sampling step size, specified as a scalar integer. This can be
            used to specify a sampling grid that has a smaller shape than the image
            grid but with values span the same range. This can be thought of as the
            reciprocal of the output scale, i.e., it will induce subsampling when set to
            values greater than 1.

    Returns:
        Tuple of grid vectors (xv, yv). These are tensors of dtype tf.float32 with
        shapes (grid_width,) and (grid_height,) respectively.

        The grid dimensions are calculated as:
            grid_width = image_width // output_stride
            grid_height = image_height // output_stride
    """
    xv = tf.cast(tf.range(0, image_width, delta=output_stride), tf.float32)
    yv = tf.cast(tf.range(0, image_height, delta=output_stride), tf.float32)
    return xv, yv


def gaussian_pdf(x: tf.Tensor, sigma: float) -> tf.Tensor:
    """Compute the PDF of an unnormalized 0-centered Gaussian distribution.

    Args:
        x: Any tensor of dtype tf.float32 with values to compute the PDF for.

    Returns:
        A tensor of the same shape as `x`, but with values of a PDF of an unnormalized
        Gaussian distribution. Values of 0 have an unnormalized PDF value of 1.0.
    """
    return tf.exp(-(tf.square(x)) / (2 * tf.square(sigma)))


def describe_tensors(
    example: Dict[Text, tf.Tensor], return_description: bool = False
) -> Optional[str]:
    """Print the keys in a example.

    Args:
        example: Dictionary keyed by strings with tensors as values.
        return_description: If `True`, returns the string description instead of
            printing it.

    Returns:
        String description if `return_description` is `True`, otherwise `None`.
    """
    desc = []
    key_length = max(len(k) for k in example.keys())
    for key, val in example.items():
        dtype = str(val.dtype) if isinstance(val.dtype, np.dtype) else repr(val.dtype)
        desc.append(
            f"{key.rjust(key_length)}: type={type(val).__name__}, "
            f"shape={val.shape}, "
            f"dtype={dtype}, "
            f"device={val.device if hasattr(val, 'device') else 'N/A'}"
        )
    desc = "\n".join(desc)

    if return_description:
        return desc
    else:
        print(desc)


def unrag_example(
    example: Dict[str, tf.Tensor], numpy: bool = False
) -> Dict[str, tf.Tensor]:
    """Convert ragged tensors in an example into normal tensors with NaN padding.

    Args:
        example: Dictionary keyed by strings with tensors as values.
        numpy: If `True`, convert values to numpy arrays or Python primitives.

    Returns:
        The same dictionary, but values of type `tf.RaggedTensor` will be converted to
        tensors of type `tf.Tensor` with NaN padding if the ragged dimensions are of
        variable length.

        The output shapes will be the bounding shape of the ragged tensors.

        If `numpy` is `True`, the values will be `numpy.ndarray`s or Python primitives
        depending on their data type and shape.

    See also: keras.utils.sync_to_numpy_or_python_type
    """
    for key in example:
        if isinstance(example[key], tf.RaggedTensor):
            example[key] = example[key].to_tensor(
                default_value=tf.cast(np.nan, example[key].dtype)
            )
    if numpy:
        example = tf_utils.sync_to_numpy_or_python_type(example)
    return example


def unrag_tensor(x: tf.RaggedTensor, max_size: int, axis: int) -> tf.Tensor:
    """Converts a ragged tensor to a full tensor by padding to a maximum size.

    This function is useful for converting ragged tensors to a fixed size when one or
    more of the dimensions are of variable length.

    Args:
        x: Ragged tensor to convert.
        max_size: Maximum size of the axis to pad.
        axis: Axis of `x` to pad to `max_size`. This must specify ragged dimensions.
            If more than one axis is specified, `max_size` must be of the same length as
            `axis`.

    Returns:
        A padded version of `x`. Padding will use the equivalent of NaNs in the tensor's
        native dtype.

        This will replace the shape of the specified `axis` with `max_size`, leaving the
        remaining dimensions set to the bounding shape of the ragged tensor.
    """
    bounding_shape = x.bounding_shape()
    axis = tf.cast(axis, tf.int64)
    axis = axis % len(x.shape)  # Handle negative indices.
    axis = tf.reshape(axis, [-1, 1])  # Ensure (n, 1) shape for indexing.
    max_size = tf.cast(max_size, bounding_shape.dtype)
    max_size = tf.reshape(max_size, [-1])  # Ensure (n,) shape for indexing.
    shape = tf.tensor_scatter_nd_update(bounding_shape, axis, max_size)
    return x.to_tensor(default_value=tf.cast(np.NaN, x.dtype), shape=shape)

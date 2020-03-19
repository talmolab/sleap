"""Miscellaneous utility functions for data processing."""

import tensorflow as tf
from typing import Any, List, Tuple


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

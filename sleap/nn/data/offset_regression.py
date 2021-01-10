"""Utilities for creating offset regression maps."""

import tensorflow as tf


def make_offsets(points: tf.Tensor, xv: tf.Tensor, yv: tf.Tensor, stride: int = 1) -> tf.Tensor:
    """Make point offset maps on a grid.

    Args:
        points: Point locations as a `tf.Tensor` of shape `(n_points, 2)` and dtype
            `tf.float32` where each row specifies the x- and y-coordinates of the map
            centers. Each point will generate a different map.
        xv: Sampling grid vector for x-coordinates of shape `(grid_width,)` and dtype
            tf.float32. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape `(grid_height,)` and dtype
            tf.float32. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        stride: Scaling factor for offset coordinates. The individual offset vectors
            will be divided by this value. Useful for adjusting for strided sampling
            grids so that the offsets point to the smaller grid coordinates.

    Returns:
        The offset maps as a `tf.Tensor` of shape
        `(grid_height, grid_width, n_points, 2)` and dtype `tf.float32`. The last axis
        corresponds to the x- and y-offsets at each grid point for each input point.

    See also:
        sleap.nn.data.utils.make_grid_vectors
    """

    # Vectorize for broadcasting.
    xv = tf.reshape(xv, [1, -1, 1, 1])
    yv = tf.reshape(yv, [-1, 1, 1, 1])
    x = tf.reshape(tf.gather(points, 0, axis=1), [1, 1, -1, 1])
    y = tf.reshape(tf.gather(points, 1, axis=1), [1, 1, -1, 1])

    # Compute offsets.
    dx = x - xv
    dy = y - yv

    # Broadcast and concatenate into a single tensor.
    shape = tf.broadcast_dynamic_shape(tf.shape(dx), tf.shape(dy))
    offsets = tf.concat(
        [tf.broadcast_to(dx, shape), tf.broadcast_to(dy, shape)], axis=-1
    )

    # Adjust for stride.
    offsets /= tf.cast(stride, tf.float32)

    # Replace NaNs with 0.
    offsets = tf.where(tf.math.is_finite(offsets), offsets, 0.0)

    return offsets


def mask_offsets(
    offsets: tf.Tensor, confmaps: tf.Tensor, threshold: float = 0.2
) -> tf.Tensor:
    """Mask offset maps using a confidence map threshold.

    This is useful for restricting offset maps to local neighborhoods around the peaks.

    Args:
        offsets: A set of offset maps as a `tf.Tensor` of shape
            `(grid_height, grid_width, n_points, 2)` and dtype `tf.float32`. This can be
            generated by `make_offsets`.
        confmaps: Confidence maps for the same points as the offset maps as a
            `tf.Tensor` of shape `(grid_height, grid_width, n_points)` and dtype
            `tf.float32`. This can be generated by
            `sleap.nn.data.confidence_maps.make_confmaps`.
        threshold: Minimum confidence map value below which offsets will be replaced
            with zeros.

    Returns:
        The offset maps with the same shape as the inputs but with zeros where the
        confidence maps are below the specified threshold.

    See also: make_offsets, sleap.nn.data.confidence_maps.make_confmaps
    """
    mask = tf.expand_dims(confmaps > threshold, axis=-1)
    masked = tf.where(mask, offsets, 0.0)
    return masked

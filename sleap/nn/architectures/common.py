import numpy as np
import collections
from functools import wraps

from tensorflow.keras.layers import Conv2D, BatchNormalization, Add


def expand_to_n(x, n):
    """Expands an object `x` to `n` elements if scalar.

    This is a utility function that wraps np.tile functionality.

    Args:
        x: Scalar of any type
        n: Number of repetitions

    Returns:
        Tiled version of `x` with __len__ == `n`.

    """
    if not isinstance(x, (collections.Sequence, np.ndarray)):
        x = [x]

    if np.size(x) == 1:
        x = np.tile(x, n)
    elif np.size(x) != n:
        raise ValueError("Variable to expand must be scalar.")

    return x


def conv(num_filters, kernel_size=(3, 3), activation="relu", **kwargs):
    """Convenience presets for Conv2D.

    Args:
        num_filters: Number of output filters (channels)
        kernel_size: Size of convolution kernel
        activation: Activation function applied to output
        **kwargs: Arbitrary keyword arguments passed on to keras.layers.Conv2D

    Returns:
        keras.layers.Conv2D instance built with presets
    """
    return Conv2D(
        num_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding="same",
        **kwargs
    )


def conv1(num_filters, **kwargs):
    """Convenience presets for 1x1 Conv2D.

    Args:
        num_filters: Number of output filters (channels)
        **kwargs: Arbitrary keyword arguments passed on to keras.layers.Conv2D

    Returns:
        keras.layers.Conv2D instance built with presets
    """
    return conv(num_filters, kernel_size=(1, 1), **kwargs)


def conv3(num_filters, **kwargs):
    """Convenience presets for 3x3 Conv2D.

    Args:
        num_filters: Number of output filters (channels)
        **kwargs: Arbitrary keyword arguments passed on to keras.layers.Conv2D

    Returns:
        keras.layers.Conv2D instance built with presets
    """
    return conv(num_filters, kernel_size=(3, 3), **kwargs)


def residual_block(x_in, num_filters=None, batch_norm=True):
    """Residual bottleneck block.

    This function builds a residual block that is used at every step of stacked
    hourglass construction. Note that the layers are actually instantiated and
    connected.

    The bottleneck is constructed by applying a 1x1 conv with `num_filters / 2`
    channels, a 3x3 conv with `num_filters / 2` channels, and a 1x1 conv with
    `num_filters`. The output of this last conv is skip-connected with the input
    via an Add layer (the residual).

    If the input `x_in` has a different number of channels as `num_filters`, an
    additional 1x1 conv is applied to the input whose output will be used for the
    skip connection.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer
        num_filters: The number output channels of the block. If not specified,
            defaults to the same number of channels as the input tensor. Must be
            divisible by 2 since the bottleneck halves the number of filters in
            the intermediate convs.
        batch_norm: Apply batch normalization after each convolution

    Returns:
        x_out: tf.Tensor of the output of the block of the same width and height
            as the input with `num_filters` channels.
    """

    # Default to output the same number of channels as input
    if num_filters is None:
        num_filters = x_in.shape[-1]

    # Number of output channels must be divisible by 2
    if num_filters % 2 != 0:
        raise ValueError(
            "Number of output filters must be divisible by 2 in residual blocks."
        )

    # If number of input and output channels are different, add a 1x1 conv to use as the
    # identity tensor to which we add the residual at the end
    x_identity = x_in
    if x_in.shape[-1] != num_filters:
        x_identity = conv1(num_filters)(x_in)
        if batch_norm:
            x_identity = BatchNormalization()(x_identity)

    # Bottleneck: 1x1 -> 3x3 -> 1x1 -> Add residual to identity
    x = conv1(num_filters // 2)(x_in)
    if batch_norm:
        x = BatchNormalization()(x)
    x = conv3(num_filters // 2)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = conv1(num_filters)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x_out = Add()([x_identity, x])

    return x_out

import numpy as np
import collections
import tensorflow as tf
import keras
from keras.layers import Conv2D, BatchNormalization, Add, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose

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
    return Conv2D(num_filters, kernel_size=kernel_size, activation=activation, padding="same", **kwargs)

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
        raise ValueError("Number of output filters must be divisible by 2 in residual blocks.")
    
    # If number of input and output channels are different, add a 1x1 conv to use as the
    # identity tensor to which we add the residual at the end
    x_identity = x_in
    if x_in.shape[-1] != num_filters:
        x_identity = conv1(num_filters)(x_in)
        if batch_norm: x_identity = BatchNormalization()(x_identity)
    
    # Bottleneck: 1x1 -> 3x3 -> 1x1 -> Add residual to identity
    x = conv1(num_filters // 2)(x_in)
    if batch_norm: x = BatchNormalization()(x)
    x = conv3(num_filters // 2)(x)
    if batch_norm: x = BatchNormalization()(x)
    x = conv1(num_filters)(x)
    if batch_norm: x = BatchNormalization()(x)
    x_out = Add()([x_identity, x])

    return x_out

def hourglass_block(x_in, num_output_channels, num_filters, depth=3, batch_norm=True, upsampling_layers=True, interp="bilinear"):
    """Creates a single hourglass block.

    This function builds an hourglass block from residual blocks and max pooling.

    The hourglass is defined as a set of `depth` residual blocks followed by 2-strided
    max pooling for downsampling, then an intermediate residual block, followed by
    `depth` blocks of upsampling -> skip Add -> residual blocks.

    The output tensors are then produced by linear activation with 1x1 convs.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have `num_filters` 
            channels since the hourglass adds a residual to this input.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        num_filters: The number feature channels of the block. These features are
            used throughout the hourglass and will be passed on to the next block
            and need not match the `num_output_channels`. Must be divisible by 2.
        depth: The number of pooling steps applied to the input. The input must
            be a tensor with `2^depth` height and width to allow for symmetric
            pooling and upsampling with skip connections.
        batch_norm: Apply batch normalization after each convolution
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x: tf.Tensor of the features output by the block with `num_filters`
            channels. This tensor can be passed on to the next hourglass or
            ignored if this is the last hourglass.
        x_out: tf.Tensor of the output of the block of the same width and height
            as the input with `num_output_channels` channels.
    """
    
    # Check if input tensor has the right number of channels
    if x_in.shape[-1] != num_filters:
        raise ValueError("Input tensor must have the same number of channels as the intermediate output of the hourglass (%d)." % num_filters)
    
    # Check if input tensor has the right height/width for pooling given depth
    if x_in.shape[-2] % (2**depth) != 0 or x_in.shape[-2] % (2**depth) != 0:
        raise ValueError("Input tensor must have width and height dimensions divisible by %d." % (2**depth))
    
    # Down
    x = x_in
    blocks_down = []
    for i in range(depth):
        x = residual_block(x, num_filters, batch_norm)
        blocks_down.append(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        
    x = residual_block(x, num_filters, batch_norm)
    
    # Middle
    x_identity = residual_block(x, num_filters, batch_norm)
    x = residual_block(x, num_filters, batch_norm)
    x = residual_block(x, num_filters, batch_norm)
    x = residual_block(x, num_filters, batch_norm)
    x = Add()([x_identity, x])
    
    # Up
    for x_down in blocks_down[::-1]:
        x_down = residual_block(x_down, num_filters, batch_norm)
        if upsampling_layers:
            x = UpSampling2D(size=(2,2), interpolation=interp)(x)
        else:
            x = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x)
        x = Add()([x_down, x])
        x = residual_block(x, num_filters, batch_norm)
        
    # Head
    x = conv1(num_filters)(x)
    if batch_norm: x = BatchNormalization()(x)
    
    x_out = conv1(num_output_channels, activation="linear")(x)
    
    x = conv1(num_filters, activation="linear")(x)
    x_ = conv1(num_filters, activation="linear")(x_out)
    x = Add()([x_in, x, x_])
    
    return x, x_out


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
        x = [x,]
        
    if np.size(x) == 1:
        x = np.tile(x, n)
    elif np.size(x) != n:
        raise ValueError("Variable to expand must be scalar.")
        
    return x

def stacked_hourglass(x_in, num_output_channels, num_hourglass_blocks=3, num_filters=32, depth=3, batch_norm=True, intermediate_inputs=True, upsampling_layers=True, interp="bilinear"):
    """Stacked hourglass block.

    This function builds and connects multiple hourglass blocks. See `hourglass` for
    more specifics on the implementation.

    Individual hourglasses can be customized by providing an iterable of hyperparameters
    for each of the arguments of the function (except `num_output_channels`). If scalars
    are provided, all hourglasses will share the same hyperparameters.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. If the number of channels
            are not the same as `num_filters`, an additional residual block is
            applied to this input.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        num_filters: The number feature channels of the block. These features are
            used throughout the hourglass and will be passed on to the next block
            and need not match the `num_output_channels`. Must be divisible by 2.
        depth: The number of pooling steps applied to the input. The input must
            be a tensor with `2^depth` height and width to allow for symmetric
            pooling and upsampling with skip connections.
        batch_norm: Apply batch normalization after each convolution
        intermediate_inputs: Re-introduce the input tensor `x_in` after each hourglass
            by concatenating with intermediate outputs
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x_outs: List of tf.Tensors of the output of the block of the same width and height
            as the input with `num_output_channels` channels.
    """
    
    # Expand block-specific parameters if scalars provided
    num_filters = expand_to_n(num_filters, num_hourglass_blocks)
    depth = expand_to_n(depth, num_hourglass_blocks)
    batch_norm = expand_to_n(batch_norm, num_hourglass_blocks)
    upsampling_layers = expand_to_n(upsampling_layers, num_hourglass_blocks)
    interp = expand_to_n(interp, num_hourglass_blocks)
    
    # Make sure first block gets the right number of channels
    x = x_in
    if x.shape[-1] != num_filters[0]:
        x = residual_block(x, num_filters[0], batch_norm[0])
    
    # Create individual hourglasses and collect intermediate outputs
    x_outs = []
    for i in range(num_hourglass_blocks):
        if i > 0 and intermediate_inputs:
            x = Concatenate()([x, x_in])
            x = residual_block(x, num_filters[i], batch_norm[i])

        x, x_out = hourglass_block(x, num_output_channels, num_filters[i], depth=depth[i], batch_norm=batch_norm[i], upsampling_layers=upsampling_layers[i], interp=interp[i])
        x_outs.append(x_out)
        
    return x_outs


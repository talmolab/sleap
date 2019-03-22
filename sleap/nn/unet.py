import keras
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPool2D, UpSampling2D

def conv(num_filters, kernel_size=(5, 5), activation="relu", initializer="glorot_normal", **kwargs):
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


def unet(x_in, num_output_channels, depth=3, convs_per_depth=2, num_filters=16, upsampling_layers=True, interp="bilinear"):
    """U-net block.

    Implementation based off of `CARE
    <https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/internals/nets.py>`_.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
            that are divisible by `2^depth`.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        depth: The number of pooling steps applied to the input. The input must
            be a tensor with `2^depth` height and width to allow for symmetric
            pooling and upsampling with skip connections.
        convs_per_depth: The number of convolutions applied before pooling or
            after upsampling.
        num_filters: The base number feature channels of the block. The number of
            filters is doubled at each pooling step.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x_out: tf.Tensor of the output of the block of the same width and height
            as the input with `num_output_channels` channels.

    """

    # Check if input tensor has the right height/width for pooling given depth
    if x_in.shape[-2] % (2**depth) != 0 or x_in.shape[-2] % (2**depth) != 0:
        raise ValueError("Input tensor must have width and height dimensions divisible by %d." % (2**depth))

    x = x_in
    
    # Downsampling
    skip_layers = []
    for n in range(depth):
        for i in range(convs_per_depth):
            x = conv(num_filters * 2 ** n)(x)
        skip_layers.append(x)
        x = MaxPool2D(pool_size=(2,2))(x)

    # Middle
    for i in range(convs_per_depth - 1):
        x = conv(num_filters * 2 ** depth)(x)
    x = conv(num_filters * 2 ** max(0, depth-1))(x)

    # Upsampling (with skips)
    for n in reversed(range(depth)):
        if upsampling_layers:
            x = UpSampling2D(size=(2,2), interpolation=interp)(x)
        else:
            x = Conv2DTranspose(num_filters * 2 ** n, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x)

        x = Concatenate(axis=-1)([x, skip_layers[n]])
        
        for i in range(convs_per_depth - 1):
            x = conv(num_filters * 2 ** n)(x)

        x = conv(num_filters * 2 ** max(0, n-1))(x)
    
    # Final layer
    x_out = conv(num_output_channels, activation="linear")(x)

    return x_out

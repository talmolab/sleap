import attr

from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D


@attr.s(auto_attribs=True)
class LeapCNN:
    """LEAP CNN block.

    Implementation generalized from original paper (`Pereira et al., 2019
    <https://www.nature.com/articles/s41592-018-0234-5>`_).

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
            that are divisible by `2^down_blocks`.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        down_blocks: The number of pooling steps applied to the input. The input
            must be a tensor with `2^down_blocks` height and width.
        up_blocks: The number of upsampling steps applied after downsampling.
        upsampling_layers: If True, use upsampling instead of transposed convs.
        num_filters: The base number feature channels of the block. The number of
            filters is doubled at each pooling step.
        interp: Method to use for interpolation when upsampling smaller features.

    """

    down_blocks: int = 3
    up_blocks: int = 3
    upsampling_layers: int = True
    num_filters: int = 64
    interp: str = "bilinear"

    def output(self, x_in, num_output_channels):
        """
        Generate a tensorflow graph for the backbone and return the output tensor.

        Args:
            x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
            that are divisible by `2^down_blocks.
            num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        Returns:
            x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.
        """
        return leap_cnn(x_in, num_output_channels, **attr.asdict(self))


def leap_cnn(x_in, num_output_channels, down_blocks=3, up_blocks=3, upsampling_layers=True, num_filters=64, interp="bilinear"):
    """LEAP CNN block.

    Implementation generalized from original paper (`Pereira et al., 2019
    <https://www.nature.com/articles/s41592-018-0234-5>`_).

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
            that are divisible by `2^down_blocks`.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        down_blocks: The number of pooling steps applied to the input. The input
            must be a tensor with `2^down_blocks` height and width.
        up_blocks: The number of upsampling steps applied after downsampling.
        upsampling_layers: If True, use upsampling instead of transposed convs.
        num_filters: The base number feature channels of the block. The number of
            filters is doubled at each pooling step.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.

    """

    # Check if input tensor has the right height/width for pooling given depth
    if x_in.shape[-2] % (2**down_blocks) != 0 or x_in.shape[-2] % (2**down_blocks) != 0:
        raise ValueError("Input tensor must have width and height dimensions divisible by %d." % (2**down_blocks))

    x = x_in

    for i in range(down_blocks):
        x = Conv2D(num_filters * (2 ** i), kernel_size=3, padding="same", activation="relu")(x)
        x = Conv2D(num_filters * (2 ** i), kernel_size=3, padding="same", activation="relu")(x)
        x = Conv2D(num_filters * (2 ** i), kernel_size=3, padding="same", activation="relu")(x)
        x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)

    for i in range(up_blocks, 0, -1):
        if upsampling_layers:
            x = UpSampling2D(interpolation=interp)(x)
        else:
            x = Conv2DTranspose(num_filters * (2 ** i), kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x)
        x = Conv2D(num_filters * (2 ** i), kernel_size=3, padding="same", activation="relu")(x)
        x = Conv2D(num_filters * (2 ** i), kernel_size=3, padding="same", activation="relu")(x)

    x = Conv2D(num_output_channels, kernel_size=3, padding="same", activation="linear")(x)

    return x

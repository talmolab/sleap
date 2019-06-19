import attr
from sleap.nn.architectures.common import conv, expand_to_n
from keras.layers import Conv2DTranspose, Concatenate, MaxPool2D, UpSampling2D


@attr.s(auto_attribs=True)
class UNet:
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
        kernel_size: Size of the convolutional kernels for each filter.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    """
    depth: int = 3
    convs_per_depth: int = 2
    num_filters: int = 16
    kernel_size: int = 5
    upsampling_layers: bool = True
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
        return unet(x_in, num_output_channels, **attr.asdict(self))


@attr.s(auto_attribs=True)
class StackedUNet:
    """Stacked U-net block.

    See `unet` for more specifics on the implementation.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
           that are divisible by `2^depth`.
        num_output_channels: The number of output channels of the block. These
           are the final output tensors on which intermediate supervision may be
           applied.
        num_stacks: The number of blocks to stack on top of each other.
        depth: The number of pooling steps applied to the input. The input must
           be a tensor with `2^depth` height and width to allow for symmetric
           pooling and upsampling with skip connections.
        convs_per_depth: The number of convolutions applied before pooling or
           after upsampling.
        num_filters: The base number feature channels of the block. The number of
           filters is doubled at each pooling step.
        kernel_size: Size of the convolutional kernels for each filter.
        intermediate_inputs: Re-introduce the input tensor `x_in` after each block
           by concatenating with intermediate outputs
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
    """

    num_stacks: int = 3
    depth: int = 3
    convs_per_depth: int = 2
    num_filters: int = 16
    kernel_size: int = 5
    upsampling_layers: bool = True
    intermediate_inputs: bool = True
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
        return stacked_unet(x_in, num_output_channels, **attr.asdict(self))


def unet(x_in, num_output_channels, depth=3, convs_per_depth=2, num_filters=16,
         kernel_size=5, upsampling_layers=True, interp="bilinear"):
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
        kernel_size: Size of the convolutional kernels for each filter.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x_out: tf.Tensor of the output of the block of the same width and height
            as the input with `num_output_channels` channels.

    """

    # Check if input tensor has the right height/width for pooling given depth
    if x_in.shape[-2] % (2**depth) != 0 or x_in.shape[-2] % (2**depth) != 0:
        raise ValueError("Input tensor must have width and height dimensions divisible by %d." % (2**depth))

    # Ensure we have a tuple in case scalar provided
    kernel_size = expand_to_n(kernel_size, 2)

    # Input tensor
    x = x_in
    
    # Downsampling
    skip_layers = []
    for n in range(depth):
        for i in range(convs_per_depth):
            x = conv(num_filters * 2 ** n, kernel_size=kernel_size)(x)
        skip_layers.append(x)
        x = MaxPool2D(pool_size=(2,2))(x)

    # Middle
    for i in range(convs_per_depth - 1):
        x = conv(num_filters * 2 ** depth, kernel_size=kernel_size)(x)
    x = conv(num_filters * 2 ** max(0, depth-1), kernel_size=kernel_size)(x)

    # Upsampling (with skips)
    for n in reversed(range(depth)):
        if upsampling_layers:
            x = UpSampling2D(size=(2,2), interpolation=interp)(x)
        else:
            x = Conv2DTranspose(num_filters * 2 ** n, kernel_size=kernel_size, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x)

        x = Concatenate(axis=-1)([x, skip_layers[n]])
        
        for i in range(convs_per_depth - 1):
            x = conv(num_filters * 2 ** n, kernel_size=kernel_size)(x)

        x = conv(num_filters * 2 ** max(0, n-1), kernel_size=kernel_size)(x)
    
    # Final layer
    x_out = conv(num_output_channels, activation="linear")(x)

    return x_out


def stacked_unet(x_in, num_output_channels, num_stacks=3, depth=3, convs_per_depth=2, num_filters=16, kernel_size=5,
                 upsampling_layers=True, intermediate_inputs=True, interp="bilinear"):
    """Stacked U-net block.

    See `unet` for more specifics on the implementation.

    Args:
        x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
            that are divisible by `2^depth`.
        num_output_channels: The number of output channels of the block. These
            are the final output tensors on which intermediate supervision may be
            applied.
        num_stacks: The number of blocks to stack on top of each other.
        depth: The number of pooling steps applied to the input. The input must
            be a tensor with `2^depth` height and width to allow for symmetric
            pooling and upsampling with skip connections.
        convs_per_depth: The number of convolutions applied before pooling or
            after upsampling.
        num_filters: The base number feature channels of the block. The number of
            filters is doubled at each pooling step.
        kernel_size: Size of the convolutional kernels for each filter.
        intermediate_inputs: Re-introduce the input tensor `x_in` after each block
            by concatenating with intermediate outputs
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.

    Returns:
        x_outs: tf.Tensor of the output of the block of the same width and height
            as the input with `num_output_channels` channels.

    """

    # Expand block-specific parameters if scalars provided
    depth = expand_to_n(depth, num_stacks)
    convs_per_depth = expand_to_n(convs_per_depth, num_stacks)
    num_filters = expand_to_n(num_filters, num_stacks)
    kernel_size = expand_to_n(kernel_size, num_stacks)
    upsampling_layers = expand_to_n(upsampling_layers, num_stacks)
    interp = expand_to_n(interp, num_stacks)

    # Create individual blocks and collect intermediate outputs
    x = x_in
    x_outs = []
    for i in range(num_stacks):
        if i > 0 and intermediate_inputs:
            x = Concatenate()([x, x_in])

        x_out = unet(x, num_output_channels, depth=depth[i], convs_per_depth=convs_per_depth[i], 
            num_filters=num_filters[i], kernel_size=kernel_size[i],
            upsampling_layers=upsampling_layers[i], interp=interp[i])
        x_outs.append(x_out)
        x = x_out
        
    return x_outs


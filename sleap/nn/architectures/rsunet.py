"""This module provides a generalized implementatio of ResUNet. 

See the 'ResUNet' class docustring for more information. 
"""

import tensorflow as tf
import numpy as np
import attr
from typing import Tuple, List

from sleap.nn.architectures.common import IntermediateFeature
from sleap.nn.config.model import RSUNetConfig


def conv(x, channels, kernel_size=3, stride=1, bias=False, **kwargs):
    """Basic convolutional layer.

    Args:
        x: Input tensor.
        channels: Specifies the number of channels in the block
        kernel_size: Specifies kernel size
        stride: Specifies the stride of the block, default = 1.
        activation: Specify type of activation for example "relu", "tanh", etc.
        prefix: Prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    return tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=bias,
        **kwargs,
    )(x)


def bn_act(x, activation="relu", prefix=""):
    """BatchNorm and activation.

    Args:
        x: Input tensor.
        activation: Specify type of activation for example "relu", "tanh", etc.
        prefix: Prefix to name of the block
    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}/bn")(x)
    x = tf.keras.layers.Activation(activation, name=f"{prefix}/{activation}")(x)
    return x


def bn_act_conv(x, channels, kernel_size=3, activation="relu", prefix=""):
    """BatchNorm -> activation -> convolution.

    Args:
        x: Input tensor.
        channels: Specifies the number of channels in the block
        kernel_size: Specifies kernel size
        activation: Specify type of activation for example "relu", "tanh", etc.
        prefix: Prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = bn_act(x, activation=activation, prefix=prefix)
    x = conv(x, channels, kernel_size=kernel_size, name=f"{prefix}/conv")
    return x


def res_block(x, channels, activation="relu", prefix=""):
    """Residual block without channel expansion.

    Args:
        x: input tensor.
        channels: int, specifies the number of channels in the block
        activation: string, specify type of activation for example "relu", "tanh", etc.
        prefix: string, prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x_in = x
    x = bn_act_conv(x, channels, activation=activation, prefix=f"{prefix}/conv1")
    x = bn_act_conv(x, channels, activation=activation, prefix=f"{prefix}/conv2")
    x = tf.keras.layers.Add(name=f"{prefix}/add_res")([x, x_in])
    return x


def conv_block(x, channels, activation="relu", prefix=""):
    """BN-Act-Conv -> Res -> BN-Act-Conv.

    Args:
        x: input tensor.
        channels: Specifies the number of channels in the block
        activation: Specify type of activation for example "relu", "tanh", etc.
        prefix: Prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = bn_act_conv(x, channels, activation=activation, prefix=f"{prefix}/pre")
    x = res_block(x, channels, activation=activation, prefix=f"{prefix}/res")
    x = bn_act_conv(x, channels, activation=activation, prefix=f"{prefix}/post")
    return x


def down_conv_block(x, channels, stride=2, activation="relu", prefix=""):
    """Downsampling convolutional block.

    Args:
        x: input tensor.
        channels: Specifies the number of channels in the block
        stride: Specifies the stride of the block, default = 2.
        activation: Specifies type of activation for example "relu", "tanh", etc.
        prefix: Prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = tf.keras.layers.MaxPool2D(strides=stride, name=f"{prefix}/pool")(x)
    x = conv_block(x, channels, activation=activation, prefix=f"{prefix}/conv_block")
    return x


def up_block(x, skip_x, channels, stride=2, interpolation="bilinear", prefix=""):
    """Upsampling -> conv -> skip addition.

    Args:
        x: Input tensor.
        skip_x: Source tensor that will be fused with the input tensor. Note: skip_x
            must have the same shape as x.
        channels: Specifies the number of channels in the block.
        stride: Specifies the stride of the block, default = 2.
        interpolation: specifies the type of interpolation used where the default value
            is "bilinear".
        prefix: Prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = tf.keras.layers.UpSampling2D(
        size=stride, interpolation=interpolation, name=f"{prefix}/upsamp"
    )(x)
    x = conv(x, channels, kernel_size=1, name=f"{prefix}/conv")
    x = tf.keras.layers.Add(name=f"{prefix}/add_skip")([x, skip_x])
    return x


def up_conv_block(
    x,
    skip_x,
    channels,
    stride=2,
    interpolation="bilinear",
    activation="relu",
    prefix="",
):
    """Upsampling block -> conv block.

    Args:
        x: input tensor.
        skip_x: Source tensor that will be fused with the input tensor. Note: skip_x
            must have the same shape as x.
        channels: int, specifies the number of channels in the block
        stride: Specifies the stride of the block, default = 2.
        interpolation: specifies the type of interpolation used where the default value
            is "bilinear".
        prefix: string, prefix to name of the block

    Returns:
        Output tensor for the stacked batch norm and activation blocks.
    """
    x = up_block(
        x,
        skip_x,
        channels,
        stride=stride,
        interpolation=interpolation,
        prefix=f"{prefix}/up_block",
    )
    x = conv_block(x, channels, activation=activation, prefix=f"{prefix}/conv_block")
    return x


def rs_unet(
    x,
    down_widths,
    up_widths,
    activation="relu",
    interpolation="bilinear",
    prefix="rsunet",
):
    """RS-UNet

    Args:
        x: input tensor.
        down_widths: list of widths of each layer of the downsampling block
        up_widths: list of widths of each layer of the upsampling block
        activation: Specifies type of activation for example "relu", "tanh", etc.
        interpolation: "bilinear"
        prefix: string, prefix to name of the block

    Returns:
        A tuple of (`x`, `intermediate_activations`).

        `x` is the output tensor from the last upsampling block.

        `intermediate_activations` is a list of `IntermediateActivation`s containing
        tensors with the outputs from each block of the decoder for use in building
        multi-output models at different feature strides.
    """
    x = conv_block(
        x, channels=down_widths[0], activation=activation, prefix=f"{prefix}/iconv"
    )

    down_depth = len(down_widths) - 1
    up_depth = len(up_widths)
    skip_sources = []
    for d in range(down_depth):
        skip_sources.append(x)
        x = down_conv_block(
            x,
            channels=down_widths[d + 1],
            stride=2,
            activation=activation,
            prefix=f"{prefix}/dconv{d+1}",
        )
        print(f"down_block = {d} | tensor shape = {x.shape}")

    print("up_widths =", up_widths)
    print("down_widths =", down_widths)

    intermediate_outputs = []

    for d in range(up_depth):
        x = up_conv_block(
            x,
            skip_sources.pop(),
            channels=up_widths[d],
            stride=2,
            interpolation=interpolation,
            activation=activation,
            prefix=f"{prefix}/uconv{d}",
        )

        stride = 2 ** (down_depth - d - 1)
        print(
            f"down_depth = {down_depth} | up_block = {d} | stride = {stride} | tensor shape = {x.shape}"
        )
        intermediate_outputs.append(IntermediateFeature(x, stride=stride))

    x = bn_act(x, activation=activation, prefix=f"{prefix}/final")

    return x, intermediate_outputs


@attr.s(auto_attribs=True)
class RSUNet:
    """RSUNet encoder-decoder architecture for residual unet.

    This is a tensorflow version of the architecture implemented in
    https://github.com/seung-lab/pytorch-emvision/blob/a930f5d30c7b791f37df8c3ee0dc3b23d
    aca62cb/emvision/models/dynamic_rsunet.py

    The default configuration with 4 down blocks and 2 up blocks and 64 base filters has
    2,864,271 million parameters.

    Attributes:
        filters: Base number of filters in the first encoder block. More filters will
            increase the representational capacity of the network at the cost of memory
            and runtime.
        filters_rate: Factor to increase the number of filters by in each block.
        down_blocks: Number of blocks with pooling in the encoder.
        up_blocks: Number of blocks with pooling in the decoder.
        maximum_stride: Determines the number of downsampling blocks in the network,
            increasing receptive field size at the cost of network size.
        output_stride: The stride of the output confidence maps relative to the input
            image. This is the reciprocal of the resolution, e.g., an output stride of 2
            results in confidence maps that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
    Note:
        This bears some differences with other implementations, particularly with
        respect to the skip connection being additive rather than concenating.
    """

    filters: int = 64
    filters_rate: float = 2
    down_blocks: int = 4
    up_blocks: int = 2

    @classmethod
    def from_config(cls, config: RSUNetConfig) -> "RSUNet":
        """Create the backbone from a configuration.

        Args:
            config: A `RSUNetConfig` instance specifying the configuration of the
                backbone.

        Returns:
            An instantiated `RSUNet`.
        """

        down_blocks = int(np.log2(config.maximum_stride)) + 1
        up_blocks = int(down_blocks - np.log2(config.output_stride)) - 1
        # up_blocks = int(down_blocks - np.log2(config.output_stride))

        return cls(
            filters=config.filters,
            filters_rate=config.filters_rate,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
        )

    @property
    def maximum_stride(self) -> int:
        """Return the maximum encoder stride relative to the input."""
        return int(2 ** (self.down_blocks - 1))  ##CHANGE

    @property
    def output_stride(self) -> int:
        """Return the stride of the output of the decoder."""
        return int(2 ** (self.down_blocks - self.up_blocks - 2))
        # return int(2 ** (self.down_blocks - self.up_blocks))

    def make_backbone(
        self, x_in: tf.Tensor
    ) -> Tuple[tf.Tensor, List[IntermediateFeature]]:
        """Create the backbone and return the output tensors for building a model.

        Args:
            x_in: A `tf.Tensor` representing the input to this backbone. This is
                typically an instance of `tf.keras.layers.Input()` but can also be any
                rank-4 tensor. Can be grayscale or RGB.

        Returns:
            A tuple of (`x_main`, `intermediate_activations`).

            `x_main` is the output tensor from the last upsampling block.

            `intermediate_activations` is a list of `IntermediateActivation`s containing
            tensors with the outputs from each block of the decoder for use in building
            multi-output models at different feature strides.
        """
        img_shape = x_in.shape[1:]

        # Calculate the widths numerically
        down_widths = [
            self.filters * (self.filters_rate ** d) for d in range(self.down_blocks)
        ]
        up_widths = [
            self.filters * (self.filters_rate ** d)
            for d in range(
                self.down_blocks - 2, (self.down_blocks - self.up_blocks) - 2, -1
            )
        ]
        output, intermediate_features = rs_unet(x_in, down_widths, up_widths)

        return output, intermediate_features

"""This module provides a generalized implementation of (stacked) hourglass.

See the `Hourglass` class docstring for more information.
"""

import tensorflow as tf
import numpy as np

import attr
from typing import Text, Optional, List

from sleap.nn.architectures import encoder_decoder
from sleap.nn.architectures.common import IntermediateFeature
from sleap.nn.config import HourglassConfig


def conv(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    prefix: Text = "conv",
) -> tf.Tensor:
    """Apply basic convolution with ReLU and batch normalization.

    Args:
        x: Input tensor.
        filters: Number of convolutional filters (output channels).
        kernel_size: Size (height == width) of convolutional kernel.
        stride: Striding of convolution. If >1, the output is smaller than the input.
        prefix: String to prepend to the sublayers of this convolution.

    Returns:
        The output tensor after applying convolution and batch normalization.
    """
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        strides=stride,
        activation="relu",
        name=prefix + "_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + "_bn")(x)
    return x


@attr.s(auto_attribs=True)
class StemBlock(encoder_decoder.EncoderBlock):
    """Stem layers of the hourglass. These are not repeated with multiple stacks.

    The default structure of this block is:
        Conv(7 x 7 x filters, stride 2) -> Conv(3 x 3 x 2*filters) -> MaxPool(stride 2)
        -> Conv(3 x 3 x output_filters)

    Attributes:
        pool: If True, pooling is applied. See `pooling_stride`.
        pooling_stride: Determines how much pooling is applied within the stem block. If
            set to 1, no pooling is applied. If set to 2, the max pooling layer will
            have a stride of 2. If set to 4, the first convolution and the max pooling
            layer will both have a stride of 2.
        filters: Base number of convolutional filters.
        output_filters: Number of filters to output at the end of this block.
        first_conv_stride: Stride of the first convolutional layer. Set to 1 to increase
            the effective spatial resolution of the initial activations.
    """

    pool: bool = True
    pooling_stride: int = 4
    filters: int = 128
    output_filters: int = 256

    def make_block(self, x_in: tf.Tensor, prefix: Text = "stem") -> tf.Tensor:
        """Create the block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        x = conv(
            x_in,
            filters=self.filters,
            kernel_size=7,
            stride=2 if (self.pool and self.pooling_stride == 4) else 1,
            prefix=prefix + "_conv7x7",
        )
        x = conv(x, filters=2 * self.filters, prefix=prefix + "_conv3x3")

        x = tf.keras.layers.MaxPool2D(
            strides=2 if (self.pool and self.pooling_stride > 1) else 1,
            padding="same",
            name=prefix + "_pool",
        )(x)
        x = conv(x, filters=self.output_filters, prefix=prefix + "_conv3x3_out")
        return x


@attr.s(auto_attribs=True)
class DownsamplingBlock(encoder_decoder.EncoderBlock):
    """Convolutional downsampling block of the hourglass.

    This block is the simplified convolution-only block described in the `Associative
    Embedding paper <https://arxiv.org/abs/1611.05424>`_, not the original residual
    blocks used in the `original hourglass paper <https://arxiv.org/abs/1603.06937>`_.
    This block is simpler and demonstrated similar performance to the residual block.

    The structure of this block is simply:
        MaxPool(stride 2) -> Conv(3 x 3 x filters)

    Attributes:
        filters: Number of filters in the convolutional layer of the block.
    """

    filters: int = 256

    def make_block(self, x_in: tf.Tensor, prefix: Text = "downsample") -> tf.Tensor:
        """Create the block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        x = tf.keras.layers.MaxPool2D(strides=2, padding="same", name=prefix + "_pool")(
            x_in
        )
        x = conv(x, filters=self.filters, prefix=prefix + "_conv")
        return x


@attr.s(auto_attribs=True)
class UpsamplingBlock(encoder_decoder.DecoderBlock):
    """Upsampling block that integrates skip connections with refinement.

    This block implements both the intermediate block after the skip connection from the
    downsampling path, as well as the upsampling block from the main network backbone
    path.

    The structure of this block is:
        x_in -> Conv(3 x 3 x filters) -> Upsample -> x_up
        skip_in -> Conv(3 x 3 x filters) -> x_middle
        x_up + x_middle -> x_out

    Attributes:
        filters: Number of filters in the output tensor.
        interp_method: Interpolation method for the upsampling step. In the original
            implementation, nearest neighbor interpolation was used. Valid values are
            "nearest" or "bilinear".
    """

    filters: int = 256
    interp_method: Text = "bilinear"

    def make_block(
        self,
        x: tf.Tensor,
        current_stride: Optional[int] = None,
        skip_source: Optional[IntermediateFeature] = None,
        prefix: Text = "upsample",
    ) -> tf.Tensor:
        """Instantiate the upsampling block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            current_stride: The stride of input tensor.
            skip_source: A tensor that will be used to form a skip connection if
                the block is configured to use it.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        x = conv(x, filters=self.filters, prefix=prefix + "_conv")
        x = tf.keras.layers.UpSampling2D(
            interpolation=self.interp_method, name=prefix + "_" + self.interp_method
        )(x)

        x_skip = conv(skip_source, filters=self.filters, prefix=prefix + "_skip")
        x = tf.keras.layers.Add(name=prefix + "_skip_add")([x, x_skip])
        return x


@attr.s(auto_attribs=True)
class Hourglass(encoder_decoder.EncoderDecoder):
    """Encoder-decoder definition of the (stacked) hourglass network backbone.

    This implements the architecture of the `Associative Embedding paper
    <https://arxiv.org/abs/1611.05424>`_, which improves upon the architecture in the
    `original hourglass paper <https://arxiv.org/abs/1603.06937>`_. The primary changes
    are to replace the residual block with simple convolutions and modify the filter
    sizes.

    The basic structure of this backbone is:
        x_in -> stem -> {encoder_stack -> decoder_stack} * stacks -> x_out

    Attributes:
        down_blocks: Number of downsampling blocks. The original implementation has 4
            downsampling blocks.
        up_blocks: Number of upsampling blocks. The original implementation is symmetric
            and has 4 upsampling blocks. If specifying more than 1 stack, this should be
            equal to down_blocks.
        stem_filters: Number of filters to output from the stem block. This block of
            convolutions will not be repeated across stacks, so it serves as a
            convenient way to reduce the input image size while extracting fine-scale
            image features. In the original implementation this is 128.
        stem_stride: Stride of the stem block. This can be set to 1, 2 or 4. If >1, this
            increases the spatial receptive field at the cost of losing fine details at
            higher resolution. In the original implementation this is 4.
        filters: Base number of filters. This will be the number of filters in the first
            block, where subsequent blocks will have an linearly increasing number of
            filters (see `filter_increase`). In the original implementation this is 256.
        filter_increase: Number to increment the number of filters in each subsequent
            block by. This number is added, not multiplied, at each block. In the
            original implementation this is 128.
        interp_method: Method for interpolation in the upsampling blocks. In the
            original implementation this is nearest neighbor interpolation. Valid values
            are "nearest" or "bilinear".
        stacks: Number of repeated stacks of symmetric downsampling -> upsampling
            stacks. Intermediate outputs are returned which can be used to apply
            intermediate supervision.
    """

    down_blocks: int = 4
    up_blocks: int = 4
    stem_filters: int = 128
    stem_stride: int = 4
    filters: int = 256
    filter_increase: int = 128
    interp_method: Text = "nearest"
    stacks: int = 3

    @property
    def stem_stack(self) -> List[encoder_decoder.EncoderBlock]:
        """Define stem stack configuration."""
        return [
            StemBlock(
                filters=self.stem_filters,
                output_filters=self.filters,
                pool=True,
                pooling_stride=self.stem_stride,
            )
        ]

    @property
    def encoder_stack(self) -> List[encoder_decoder.EncoderBlock]:
        """Define encoder stack configuration."""
        encoder_blocks = []

        # Downsampling path
        for i in range(self.down_blocks):
            encoder_blocks.append(
                DownsamplingBlock(filters=self.filters + (i * self.filter_increase))
            )

        return encoder_blocks

    @property
    def decoder_stack(self) -> List[encoder_decoder.DecoderBlock]:
        """Define decoder stack configuration."""
        # Upsampling path
        decoder_blocks = []
        for i in range(self.up_blocks):
            decoder_blocks.append(
                UpsamplingBlock(
                    filters=self.filters
                    + ((self.down_blocks - i - 1) * self.filter_increase),
                    interp_method=self.interp_method,
                )
            )
        return decoder_blocks

    @classmethod
    def from_config(cls, config: HourglassConfig) -> "Hourglass":
        """Create a model from a set of configuration parameters.

        Args:
            config: An `HourglassConfig` instance with the desired parameters.

        Returns:
            An instance of this class with the specified configuration.
        """
        stem_blocks = np.log2(config.stem_stride).astype(int)
        down_blocks = np.log2(config.max_stride).astype(int) - stem_blocks
        up_blocks = np.log2(config.max_stride / config.output_stride).astype(int)
        return cls(
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            stem_filters=config.stem_filters,
            stem_stride=config.stem_stride,
            filters=config.filters,
            filter_increase=config.filter_increase,
            interp_method="nearest",
            stacks=config.stacks,
        )

"""This module provides a generalized implementation of UNet.

See the `UNet` class docstring for more information.
"""

import attr
from typing import List, Optional, Text
from sleap.nn.architectures import encoder_decoder
from sleap.nn.config import UNetConfig
import numpy as np
import tensorflow as tf


@attr.s(auto_attribs=True)
class PoolingBlock(encoder_decoder.EncoderBlock):
    """Pooling-only encoder block.

    Used to compensate for UNet having a skip source before the pooling, so the blocks
    need to end with a conv, not the pooling layer. This is added to the end of the
    encoder stack to ensure that the number of down blocks is equal to the number of
    pooling steps.

    Attributes:
        pool: If True, applies max pooling at the end of the block.
        pooling_stride: Stride of the max pooling operation. If 1, the output of this
            block will be at the same stride (== 1/scale) as the input.
    """

    pool: bool = True
    pooling_stride: int = 2

    def make_block(self, x_in: tf.Tensor, prefix: Text = "conv_block") -> tf.Tensor:
        """Instantiate the encoder block from an input tensor."""
        x = x_in
        if self.pool:
            x = tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=self.pooling_stride,
                padding="same",
                name=f"{prefix}_last_pool",
            )(x)
        return x


@attr.s(auto_attribs=True)
class UNet(encoder_decoder.EncoderDecoder):
    """UNet encoder-decoder architecture for fully convolutional networks.

    This is the canonical architecture described in `Ronneberger et al., 2015
    <https://arxiv.org/abs/1505.04597>`_.

    The default configuration with 4 down/up blocks and 64 base filters has ~34.5M
    parameters.

    Attributes:
        filters: Base number of filters in the first encoder block. More filters will
            increase the representational capacity of the network at the cost of memory
            and runtime.
        filters_rate: Factor to increase the number of filters by in each block.
        kernel_size: Size of convolutional kernels (== height == width).
        stem_kernel_size: Size of convolutional kernels in stem blocks.
        stem_blocks: If >0, will create additional "down" blocks for initial
            downsampling. These will be configured identically to the down blocks below.
        down_blocks: Number of blocks with pooling in the encoder. More down blocks will
        convs_per_block: Number of convolutions in each block. More convolutions per
            block will increase the representational capacity of the network at the cost
            of memory and runtime.
            increase the effective maximum receptive field.
        up_blocks: Number of blocks with upsampling in the decoder. If this is equal to
            `down_blocks`, the output of this network will be at the same stride (scale)
            as the input.
        middle_block: If True, add an additional block at the end of the encoder.
        up_interpolate: If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. If using transposed convolutions, the
            number of filters are determined by `filters` and `filters_rate` to
            progressively decrease the number of filters at each step.
        block_contraction: If True, reduces the number of filters at the end of middle
            and decoder blocks. This has the effect of introducing an additional
            bottleneck before each upsampling step. The original implementation does not
            do this, but the CARE implementation does.

    Note:
        This bears some differences with other implementations, particularly with
        respect to the skip connection source tensors in the encoder. In the original,
        the skip connection is formed from the output of the convolutions in each
        encoder block, not the pooling step. This results in skip connections starting
        at the first stride level as well as subsequent ones.
    """

    filters: int = 64
    filters_rate: float = 2
    kernel_size: int = 3
    stem_kernel_size: int = 3
    convs_per_block: int = 2
    stem_blocks: int = 0
    down_blocks: int = 4
    middle_block: bool = True
    up_blocks: int = 4
    up_interpolate: bool = False
    block_contraction: bool = False

    @property
    def stem_stack(self) -> Optional[List[encoder_decoder.SimpleConvBlock]]:
        """Define the downsampling stem."""
        if self.stem_blocks == 0:
            return None

        blocks = []
        for block in range(self.stem_blocks):
            block_filters = int(self.filters * (self.filters_rate ** block))
            blocks.append(
                encoder_decoder.SimpleConvBlock(
                    pool=(block > 0),
                    pool_before_convs=True,
                    pooling_stride=2,
                    num_convs=self.convs_per_block,
                    filters=block_filters,
                    kernel_size=self.stem_kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                )
            )

        # Always finish with a pooling block to account for pooling before convs.
        blocks.append(PoolingBlock(pool=True, pooling_stride=2))

        return blocks

    @property
    def encoder_stack(self) -> List[encoder_decoder.SimpleConvBlock]:
        """Define the encoder stack."""
        blocks = []
        for block in range(self.down_blocks):
            block_filters = int(
                self.filters * (self.filters_rate ** (block + self.stem_blocks))
            )
            blocks.append(
                encoder_decoder.SimpleConvBlock(
                    pool=(block > 0),
                    pool_before_convs=True,
                    pooling_stride=2,
                    num_convs=self.convs_per_block,
                    filters=block_filters,
                    kernel_size=self.kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                )
            )

        # Always finish with a pooling block to account for pooling before convs.
        blocks.append(PoolingBlock(pool=True, pooling_stride=2))

        # Create a middle block (like the CARE implementation).
        if self.middle_block:
            if self.convs_per_block > 1:
                # First convs are one exponent higher than the last encoder block.
                block_filters = int(
                    self.filters
                    * (self.filters_rate ** (self.down_blocks + self.stem_blocks))
                )
                blocks.append(
                    encoder_decoder.SimpleConvBlock(
                        pool=False,
                        pool_before_convs=False,
                        pooling_stride=2,
                        num_convs=self.convs_per_block - 1,
                        filters=block_filters,
                        kernel_size=self.kernel_size,
                        use_bias=True,
                        batch_norm=False,
                        activation="relu",
                        block_prefix="_middle_expand",
                    )
                )

            if self.block_contraction:
                # Contract the channels with an exponent lower than the last encoder block.
                block_filters = int(
                    self.filters
                    * (self.filters_rate ** (self.down_blocks + self.stem_blocks - 1))
                )
            else:
                # Keep the block output filters the same.
                block_filters = int(
                    self.filters
                    * (self.filters_rate ** (self.down_blocks + self.stem_blocks))
                )
            blocks.append(
                encoder_decoder.SimpleConvBlock(
                    pool=False,
                    pool_before_convs=False,
                    pooling_stride=2,
                    num_convs=1,
                    filters=block_filters,
                    kernel_size=self.kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                    block_prefix="_middle_contract",
                )
            )

        return blocks

    @property
    def decoder_stack(self) -> List[encoder_decoder.SimpleUpsamplingBlock]:
        """Define the decoder stack."""
        blocks = []
        for block in range(self.up_blocks):
            block_filters_in = int(
                self.filters
                * (
                    self.filters_rate
                    ** (self.down_blocks + self.stem_blocks - 1 - block)
                )
            )
            if self.block_contraction:
                block_filters_out = int(
                    self.filters
                    * (
                        self.filters_rate
                        ** (self.down_blocks + self.stem_blocks - 2 - block)
                    )
                )
            else:
                block_filters_out = block_filters_in
            blocks.append(
                encoder_decoder.SimpleUpsamplingBlock(
                    upsampling_stride=2,
                    transposed_conv=(not self.up_interpolate),
                    transposed_conv_filters=block_filters_in,
                    transposed_conv_kernel_size=self.kernel_size,
                    transposed_conv_batch_norm=False,
                    interp_method="bilinear",
                    skip_connection=True,
                    skip_add=False,
                    refine_convs=self.convs_per_block,
                    refine_convs_first_filters=block_filters_in,
                    refine_convs_filters=block_filters_out,
                    refine_convs_kernel_size=self.kernel_size,
                    refine_convs_batch_norm=False,
                )
            )
        return blocks

    @classmethod
    def from_config(cls, config: UNetConfig) -> "UNet":
        """Create a model from a set of configuration parameters.

        Args:
            config: An `UNetConfig` instance with the desired parameters.

        Returns:
            An instance of this class with the specified configuration.
        """
        stem_blocks = 0
        if config.stem_stride is not None:
            stem_blocks = np.log2(config.stem_stride).astype(int)
        down_blocks = np.log2(config.max_stride).astype(int) - stem_blocks
        up_blocks = np.log2(config.max_stride / config.output_stride).astype(int)

        return cls(
            filters=config.filters,
            filters_rate=config.filters_rate,
            kernel_size=3,
            stem_kernel_size=7,
            convs_per_block=2,
            stem_blocks=stem_blocks,
            down_blocks=down_blocks,
            middle_block=config.middle_block,
            up_blocks=up_blocks,
            up_interpolate=config.up_interpolate,
            stacks=config.stacks,
        )

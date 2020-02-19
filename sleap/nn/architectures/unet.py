"""This module provides a generalized implementation of UNet.

See the `UNet` class docstring for more information.
"""

import attr
from typing import List, Optional
from sleap.nn.architectures import encoder_decoder
from sleap.nn.config import UNetConfig
import numpy as np


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

    @property
    def stem_stack(self) -> Optional[List[encoder_decoder.SimpleConvBlock]]:
        """Define the downsampling stem."""
        if self.stem_blocks == 0:
            return None

        blocks = []
        for block in range(self.stem_blocks + 1):
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

        return blocks

    @property
    def encoder_stack(self) -> List[encoder_decoder.SimpleConvBlock]:
        """Define the encoder stack."""
        blocks = []
        for block in range(self.down_blocks + 1):
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
        return blocks

    @property
    def decoder_stack(self) -> List[encoder_decoder.SimpleUpsamplingBlock]:
        """Define the decoder stack."""
        blocks = []
        for block in range(self.up_blocks):
            block_filters = int(
                self.filters
                * (
                    self.filters_rate
                    ** (self.stem_blocks + self.down_blocks - block - 1)
                )
            )
            blocks.append(
                encoder_decoder.SimpleUpsamplingBlock(
                    upsampling_stride=2,
                    transposed_conv=(not self.up_interpolate),
                    transposed_conv_filters=block_filters,
                    transposed_conv_kernel_size=self.kernel_size,
                    transposed_conv_batch_norm=False,
                    interp_method="bilinear",
                    skip_connection=True,
                    skip_add=False,
                    refine_convs=self.convs_per_block,
                    refine_convs_filters=block_filters,
                    refine_convs_kernel_size=self.kernel_size,
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

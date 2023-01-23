"""This module provides a generalized implementatio of ConvNeXT. 

See the 'ConvNeXT' class docustring for more information. 
"""

import tensorflow as tf
import numpy as np
import attr
from typing import Tuple, List
import sys

import sleap
from sleap.nn.architectures.common import IntermediateFeature
from sleap.nn.config.model import ConvNeXTConfig


def convnext_encoder(x, crop_size):
    """ """

    encoder_backbone = tf.keras.applications.convnext.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=False,
        weights=None,
        # input_shape=(crop_size, crop_size, 1),
        input_tensor=x,
    )

    # x_in_dec = encoder_backbone.output
    # encoder = tf.keras.Model(x, x_in_dec, name="encoder_backbone")
    x_in_dec = encoder_backbone(x)

    encoder_backbone.summary()

    intermediate_block_names = [
        "convnext_tiny_stem",  ###
        "convnext_tiny_downsampling_block_0",
        "convnext_tiny_downsampling_block_1",
        "convnext_tiny_downsampling_block_2",
    ]

    # print(["encoder intermediate blocks: " + str(np.size(intermediate_block_names))])

    intermediate_features = []
    for block_name in intermediate_block_names:
        for i, layer in enumerate(encoder_backbone.layers):
            if layer.name == block_name:
                x_i = encoder_backbone.layers[i - 1].output
                intermediate_features.append(
                    sleap.nn.architectures.common.IntermediateFeature(
                        tensor=x_i, stride=crop_size // x_i.shape[1]
                    )
                )
                break
    return x_in_dec, intermediate_features


def convnext_decoder(x_in_dec, intermediate_features, crop_size):
    """ """
    upsampling_stack = sleap.nn.architectures.upsampling.UpsamplingStack(
        output_stride=2,  ## this was two
        transposed_conv=False,
        refine_convs_filters=32,
        refine_convs_filters_rate=1.5,
    )
    x, decoder_intermediate_features = upsampling_stack.make_stack(
        x_in_dec,
        current_stride=crop_size // x_in_dec.shape[1],
        skip_sources=intermediate_features,
    )
    print(["decoder intermediate blocks: " + str(np.size(intermediate_features))])
    return x, decoder_intermediate_features


def convnext(x_in, crop_size):
    """ """

    x_in_dec, intermediate_features = convnext_encoder(x_in, crop_size)
    x, decoder_intermediate_features = convnext_decoder(
        x_in_dec, intermediate_features, crop_size
    )

    return x, decoder_intermediate_features


@attr.s(auto_attribs=True)
class ConvNeXT:
    """ """

    down_blocks: int = 4
    up_blocks: int = 4
    crop_size: int = 512

    @classmethod
    def from_config(cls, config: ConvNeXTConfig) -> "ConvNeXT":
        """Create the backbone from a configuration.

        Args:
            config: A `ConvNeXTConfig` instance specifying the configuration of the
                backbone.

        Returns:
            An instantiated `ConvNeXT`.
        """
        down_blocks = 4
        up_blocks = 4

        # up_blocks = int(down_blocks - np.log2(config.output_stride))

        return cls(
            crop_size=config.crop_size,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
        )

    @property
    def maximum_stride(self) -> int:
        """Return the maximum encoder stride relative to the input."""
        return int(2 ** (self.down_blocks))  ##CHANGE

    @property
    def output_stride(self) -> int:
        """Return the stride of the output of the decoder."""
        return int(2 ** (self.down_blocks - self.up_blocks))
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
        # self.crop_size = x_in.shape[1:]
        output, intermediate_features = convnext(x_in, self.crop_size)
        print(intermediate_features)

        return output, intermediate_features

"""This module defines common upsampling layer stack configurations.

The generic upsampling stack consists of:
    - transposed convolution or bilinear upsampling with stride > 1
    - skip connections
    - 0 or more 3x3 convolutions for refinement

Configuring these components suffices to define the "decoder" portion of most canonical
"encoder-decoder"-like architectures (e.g., LEAP CNN, UNet, Hourglass, etc.), as well as
simpler patterns like shallow or direct upsampling (e.g., DLC).
"""

import numpy as np
import tensorflow as tf
import attr
from typing import Union, Sequence, List, Tuple, Optional

from sleap.nn.architectures.common import IntermediateFeature
from sleap.nn.config import UpsamplingConfig


@attr.s(auto_attribs=True)
class UpsamplingStack:
    """Standard stack of upsampling layers with refinement and skip connections.

    Attributes:
        output_stride: The desired final stride of the output tensor of the stack.
        upsampling_stride: The striding of the upsampling *layer* (not tensor). This is
            typically set to 2, such that the tensor doubles in size with each
            upsampling step, but can be set higher to upsample to the desired
            `output_stride` directly in fewer steps. See the notes in the `make_stack`
            method for examples.
        transposed_conv: If True, use a strided transposed convolution to perform
            learnable upsampling. If False, bilinear upsampling will be used instead.
        transposed_conv_filters: Integer that specifies the base number of filters in
            each transposed convolution layer. This will be scaled by the
            `transposed_conv_filters_rate` at every upsampling step. No effect if
            bilinear upsampling is used.
        transposed_conv_filters_rate: Factor to scale the number of filters in the
            transposed convolution layer after each upsampling step. If set to 1, the
            number of filters won't change. No effect if bilinear upsampling is used.
        transposed_conv_kernel_size: Size of the kernel for the transposed convolution.
            No effect if bilinear upsampling is used.
        transposed_conv_batchnorm: Specifies whether batch norm should be applied after
            the transposed convolution (and before the ReLU activation). No effect if
            bilinear upsampling is used.
        make_skip_connection: If True, incoming feature tensors form skip connection
            with upsampled features. If False, no skip connection will be formed.
        skip_add: If True, incoming feature tensors form skip connection with upsampled
            features via element-wise addition. Height/width are matched via stride and
            a 1x1 linear conv is applied if the channel counts do no match up. If False,
            the skip connection is formed via channel-wise concatenation.
        refine_convs: If greater than 0, specifies the number of 3x3 convolutions that
            will be applied after the upsampling step for refinement. These layers can
            serve the purpose of "mixing" the skip connection fused features, or to
            refine the current feature map after upsampling, which can help to prevent
            aliasing and checkerboard effects. If 0, no additional convolutions will be
            applied.
        refine_convs_filters: Similar to `transposed_conv_filters`, specifies the number
            of filters to use for the refinement convolutions in each upsampling step.
            No effect if `refine_convs` is 0.
        refine_convs_filters_rate: Factor to scale the number of filters in the refine
            conv layers after each upsampling step. The same number of filters are used
            for all convs within the same upsampling step. If set to 1, the number of
            filters won't change. No effect if `refine_convs` is 0.
        refine_convs_batchnorm: Specifies whether batch norm should be applied after
            each 3x3 convolution and before the ReLU activation. No effect if
            `refine_convs` is 0.
    """

    output_stride: int
    upsampling_stride: int = 2

    transposed_conv: bool = True
    transposed_conv_filters: int = 64
    transposed_conv_filters_rate: float = 1
    transposed_conv_kernel_size: int = 4
    transposed_conv_batchnorm: bool = True

    make_skip_connection: bool = True
    skip_add: bool = False

    refine_convs: int = 2
    refine_convs_filters: int = 64
    refine_convs_filters_rate: float = 1
    refine_convs_batchnorm: bool = True

    @classmethod
    def from_config(
        cls, config: UpsamplingConfig, output_stride: int
    ) -> "UpsamplingStack":
        """Create a model from a set of configuration parameters.

        Args:
            config: An `UpsamplingConfig` instance with the desired parameters.
            output_stride: The desired final stride of the output tensor of the stack.

        Returns:
            An instance of this class with the specified configuration.
        """
        return cls(
            output_stride=output_stride,
            upsampling_stride=config.block_stride,
            transposed_conv=config.method == "transposed_conv",
            transposed_conv_filters=config.filters,
            transposed_conv_filters_rate=config.filters_rate,
            transposed_conv_kernel_size=config.transposed_conv_kernel_size,
            transposed_conv_batchnorm=config.batch_norm,
            make_skip_connection=config.skip_connections is not None,
            skip_add=config.skip_connections is not None
            and config.skip_connections == "add",
            refine_convs=config.refine_convs,
            refine_convs_filters=config.filters,
            refine_convs_filters_rate=config.filters_rate,
            refine_convs_batchnorm=config.batch_norm,
        )

    def make_stack(
        self,
        x: tf.Tensor,
        current_stride: int,
        skip_sources: Optional[Sequence[IntermediateFeature]] = None,
    ) -> Tuple[tf.Tensor, List[IntermediateFeature]]:
        """Create the stack of upsampling layers.

        Args:
            x: Feature tensor at low resolution, typically from encoder/backbone.
            current_stride: Stride of the feature tensor relative to the network input.
            skip_sources: If a list of `IntermediateFeature`s are provided, they will be
                searched to find source tensors with matching stride after each
                upsampling step. The first element of this list with a matching stride
                will be selected as the source at each level. Skip connection will be a
                concatenation or addition, depending on the `skip_add` class attribute.

        Returns:
            A tuple of the resulting upsampled tensor at the stride specified by the
            `output_stride` class attribute, and a list of intermediate tensors after
            each upsampling step.

            The intermediate features are useful when creating multi-head architectures
            with different output strides for the heads.

        Note:
            The computed number of upsampling steps will be determined by the
            `current_stride` argument, and `output_stride` and `upsampling_stride` class
            attributes.

            Specifically, the number of upsampling steps is equal to:
                `log(current_stride) - log(output_stride)`
            where the log base is equal to the `upsampling_stride`.

            These can be used to control the number of upsampling steps indirectly, for
            example:
                Start with `current_stride = 16` and want to get to `output_stride = 4`;
                with `upsampling_stride = 2` this will take 2 upsampling steps, and
                with `upsampling_stride = 4` this will take 1 upsampling step.
        """
        # Calculate the number of upsampling steps.
        num_blocks = int(
            (np.log(current_stride) - np.log(self.output_stride))
            / np.log(self.upsampling_stride)
        )

        # Initialize intermediate features with the output of the encoder stack.
        intermediate_feats = [IntermediateFeature(tensor=x, stride=current_stride)]

        # Create each upsampling block.
        for block in range(num_blocks):

            # Update stride level.
            new_stride = current_stride // self.upsampling_stride
            block_prefix = f"upsample_s{current_stride}_to_s{new_stride}"

            if self.transposed_conv:

                # Upsample via strided transposed convolution.
                block_trans_conv_filters = int(
                    self.transposed_conv_filters
                    * self.transposed_conv_filters_rate ** block
                )
                x = tf.keras.layers.Conv2DTranspose(
                    filters=block_trans_conv_filters,
                    kernel_size=self.transposed_conv_kernel_size,
                    strides=self.upsampling_stride,
                    padding="same",
                    name=block_prefix + "_trans_conv",
                )(x)

                if self.transposed_conv_batchnorm:
                    x = tf.keras.layers.BatchNormalization(name=block_prefix + "_bn")(x)
                x = tf.keras.layers.Activation("relu", name=block_prefix + "_relu")(x)

            else:
                # Upsample via bilinear interpolation.
                x = tf.keras.layers.UpSampling2D(
                    size=self.upsampling_stride,
                    interpolation="bilinear",
                    name=block_prefix + "_interp",
                )(x)

            # Tensor is now upsampled to the updated stride.
            current_stride = new_stride

            # Form skip connection if there are any available at this stride level.
            if skip_sources is not None and self.make_skip_connection:
                added_skip = False
                for skip_source in skip_sources:
                    if not added_skip and skip_source.stride == current_stride:
                        if self.skip_add:
                            source_x = skip_source.tensor
                            if source_x.shape[-1] != x.shape[-1]:
                                # Adjust channel count via 1x1 linear conv if not
                                # matching.
                                source_x = tf.keras.layers.Conv2D(
                                    filters=x.shape[-1],
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    name=block_prefix + "_skip_conv1x1",
                                )(source_x)

                            # Concatenate via addition.
                            x = tf.keras.layers.Add(name=block_prefix + "_skip_add")(
                                [source_x, x]
                            )

                        else:
                            # Simple concatenation.
                            x = tf.keras.layers.Concatenate(
                                name=block_prefix + "_skip_concat"
                            )([skip_source.tensor, x])

                        added_skip = True

            # Add further convolutions to refine after upsampling and/or skip.
            block_refine_filters = int(
                self.refine_convs_filters * self.refine_convs_filters_rate ** block
            )
            for i in range(self.refine_convs):
                x = tf.keras.layers.Conv2D(
                    filters=block_refine_filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    name=block_prefix + f"_refine{i}_conv",
                )(x)
                if self.refine_convs_batchnorm:
                    x = tf.keras.layers.BatchNormalization(
                        name=block_prefix + f"_refine{i}_bn"
                    )(x)
                x = tf.keras.layers.Activation(
                    "relu", name=block_prefix + f"_refine{i}_relu"
                )(x)

            intermediate_feats.append(
                IntermediateFeature(tensor=x, stride=current_stride)
            )

        return x, intermediate_feats

"""Implements wrappers for constructing (optionally pretrained) NASNets.

See original paper:
https://arxiv.org/abs/1707.07012
"""

import tensorflow as tf
import keras_applications as applications
import attr
from sleap.nn.architectures import common


@attr.s(auto_attribs=True)
class NASNetMobile:
    """NASNet Mobile backbone.

    This backbone has ~4.3M params.

    Attributes:
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
        pretrained: Load pretrained ImageNet weights for transfer learning. If
            False, random weights are used for initialization.
    """

    upsampling_layers: bool = True
    interp: str = "bilinear"
    up_blocks: int = 5
    refine_conv_up: bool = False
    pretrained: bool = True

    def output(self, x_in, num_output_channels):
        """Builds the layers for this backbone and return the output tensor.

        Args:
            x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
                that are divisible by `2^down_blocks.
            num_output_channels: The number of output channels of the block. These
                are the final output tensors on which intermediate supervision may be
                applied.

        Returns:
            x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.
        """

        x = x_in

        if self.pretrained:
            # Input should be rescaled from [0, 1] to [-1, 1] and needs to be 3 channels (RGB)
            x = tf.keras.layers.Lambda(common.scale_input)(x)

            if x_in.shape[-1] == 1:
                x = tf.keras.layers.Lambda(common.tile_channels)(x)

        # Automatically downloads weights
        backbone_model = applications.nasnet.NASNetMobile(
            include_top=False,
            input_shape=(int(x_in.shape[-3]), int(x_in.shape[-2]), 3),
            weights="imagenet" if self.pretrained else None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up, interp=self.interp)

        x = tf.keras.layers.Conv2D(num_output_channels, (3, 3), padding="same")(x)

        return x

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        # This is a fixed constant for this backbone.
        return 5

    @property
    def output_scale(self):
        """Returns relative scaling factor of this backbone."""

        return 1 / (2 ** (self.down_blocks - self.up_blocks))


@attr.s(auto_attribs=True)
class NASNetLarge:
    """NASNet Large backbone.

    This backbone has ~84.9M params.

    Attributes:
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
        pretrained: Load pretrained ImageNet weights for transfer learning. If
            False, random weights are used for initialization.
    """

    upsampling_layers: bool = True
    interp: str = "bilinear"
    up_blocks: int = 5
    refine_conv_up: bool = False
    pretrained: bool = True

    def output(self, x_in, num_output_channels):
        """Builds the layers for this backbone and return the output tensor.

        Args:
            x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
                that are divisible by `2^down_blocks.
            num_output_channels: The number of output channels of the block. These
                are the final output tensors on which intermediate supervision may be
                applied.

        Returns:
            x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.
        """

        x = x_in

        if self.pretrained:
            # Input should be rescaled from [0, 1] to [-1, 1] and needs to be 3 channels (RGB)
            x = tf.keras.layers.Lambda(common.scale_input)(x)

            if x_in.shape[-1] == 1:
                x = tf.keras.layers.Lambda(common.tile_channels)(x)

        # Automatically downloads weights
        backbone_model = applications.nasnet.NASNetLarge(
            include_top=False,
            input_shape=(int(x_in.shape[-3]), int(x_in.shape[-2]), 3),
            weights="imagenet" if self.pretrained else None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up, interp=self.interp)

        x = tf.keras.layers.Conv2D(num_output_channels, (3, 3), padding="same")(x)

        return x

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        # This is a fixed constant for this backbone.
        return 5

    @property
    def output_scale(self):
        """Returns relative scaling factor of this backbone."""

        return 1 / (2 ** (self.down_blocks - self.up_blocks))


@attr.s(auto_attribs=True)
class GeneralizedNASNet:
    """Generalized version of the NASNet backbone.

    This allows for selecting the architectural hyperparameters, but cannot use
    pretrained weights since the configuration may not have been previously used.

    Attributes:
        stem_block_filters: Number of initial filters on the stem of the backbone.
        num_blocks: Number of repeat NAS blocks. This is the N of the N @ P NASNet
            notation. It is set to 4 or 6 for the mobile or large variants respectively.
        filter_multiplier: Width or base filter multiplier. This also constrains the
            valid number of filters in the penultimate layer (see below).
        penultimate_filter_multiplier: The factor by which to scale the number of
            filters in the penultimate layer. The total number of filters in the
            penultimate layer is the P of the N @ P NASNet notation. This number is
            calculated as 24 * (filter_multiplier ** 2) * penultimate_filter_multiplier.
        skip_reductions: If True, skips the reduction cell outputs.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
    """

    stem_block_filters: int = 96
    num_blocks: int = 4
    filter_multiplier: int = 2
    penultimate_filter_multiplier: int = 1
    skip_reductions: bool = False
    upsampling_layers: bool = True
    interp: str = "bilinear"
    up_blocks: int = 5
    refine_conv_up: bool = False

    def output(self, x_in, num_output_channels):
        """Builds the layers for this backbone and return the output tensor.

        Args:
            x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
                that are divisible by `2^down_blocks.
            num_output_channels: The number of output channels of the block. These
                are the final output tensors on which intermediate supervision may be
                applied.

        Returns:
            x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.
        """

        x = x_in

        backbone_model = applications.nasnet.NASNet(
            input_shape=x.shape[1:],
            num_blocks=self.num_blocks,
            filter_multiplier=self.filter_multiplier,
            penultimate_filters=self.penultimate_filters,
            skip_reduction=self.skip_reductions,
            include_top=False,
            weights=None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up, interp=self.interp)

        x = tf.keras.layers.Conv2D(num_output_channels, (3, 3), padding="same")(x)

        return x

    @property
    def penultimate_filters(self):
        return 24 * (self.filter_multiplier ** 2) * self.penultimate_filter_multiplier
    

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        # This is a fixed constant for this backbone.
        return 5

    @property
    def output_scale(self):
        """Returns relative scaling factor of this backbone."""

        return 1 / (2 ** (self.down_blocks - self.up_blocks))

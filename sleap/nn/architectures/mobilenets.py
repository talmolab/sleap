"""Implements wrappers for constructing (optionally pretrained) MobileNets.

See original papers:
- MobileNetV1: https://arxiv.org/abs/1704.04861
- MobileNetV2: https://arxiv.org/abs/1801.04381
"""

import tensorflow as tf
import keras_applications as applications
import attr
from sleap.nn.architectures import common


@attr.s(auto_attribs=True)
class MobileNetV1:
    """MobileNetV1 backbone.

    At depth_multiplier = 1, this backbone has:
        ~0.2M params @ alpha = 0.25
        ~0.8M params @ alpha = 0.5
        ~1.8M params @ alpha = 0.75
        ~3.2M params @ alpha = 1.0

    Parameter counts scale proportionally with the depth_multiplier, e.g., at
    depth_multiplier = 2, the backbone has ~6.4M params @ alpha = 1.0.

    Attributes:
        depth_multiplier: Factor to scale the number of depthwise convolution filters
            by. Called resolution multiplier in the original paper. Must be an integer.
            Pretrained weights are only available for depth_multiplier = 1.
        alpha: Factor to scale the base number of filters by. Pretrained weights are
            only available for alpha = 0.25, 0.5, 0.75 or 1.0.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
        pretrained: Load pretrained ImageNet weights for transfer learning. If
            False, random weights are used for initialization.
    """
    depth_multiplier: int = 1
    alpha: float = 0.5
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
        backbone_model = applications.mobilenet.MobileNet(
            include_top=False,
            input_shape=x.shape[1:],
            weights="imagenet" if self.pretrained else None,
            depth_multiplier=self.depth_multiplier,
            alpha=self.alpha,
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
class MobileNetV2:
    """MobileNetV2 backbone.

    This backbone has:
        ~0.4M params @ alpha = 0.35
        ~0.7M params @ alpha = 0.5
        ~1.4M params @ alpha = 0.75
        ~2.3M params @ alpha = 1.0
        ~3.8M params @ alpha = 1.3
        ~4.4M params @ alpha = 1.4

    Attributes:
        alpha: Factor to scale the base number of filters by. Pretrained weights are
            only available for alpha = 0.35, 0.5, 0.75, 1.0, 1.3 or 1.4.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
        pretrained: Load pretrained ImageNet weights for transfer learning. If
            False, random weights are used for initialization.
    """
    alpha: float = 0.5
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
        backbone_model = applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            input_shape=x.shape[1:],
            weights="imagenet" if self.pretrained else None,
            alpha=self.alpha,
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

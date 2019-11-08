"""Implements wrappers for constructing (optionally pretrained) DenseNets.

See original paper:
https://arxiv.org/abs/1608.06993
"""

import tensorflow as tf
import keras_applications as applications
import attr
from sleap.nn.architectures import common


@attr.s(auto_attribs=True)
class DenseNet121:
    """DenseNet121 backbone.

    This backbone has ~7M params.

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
        backbone_model = applications.densenet.DenseNet121(
            include_top=False,
            input_shape=x.shape[1:],
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
class DenseNet169:
    """DenseNet169 backbone.

    This backbone has ~12.6M params.

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
        backbone_model = applications.densenet.DenseNet169(
            include_top=False,
            input_shape=x.shape[1:],
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
class DenseNet201:
    """DenseNet201 backbone.

    This backbone has ~18.3M params.

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
        backbone_model = applications.densenet.DenseNet201(
            include_top=False,
            input_shape=x.shape[1:],
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
class GeneralizedDenseNet:
    """Generalized version of the 4-block DenseNet backbone.

    This allows for selecting the number of blocks in each dense layer, but cannot use
    pretrained weights since the configuration may not have been previously used.

    Attributes:
        n_dense_blocks_1: Number of blocks in dense layer 1.
        n_dense_blocks_2: Number of blocks in dense layer 2.
        n_dense_blocks_3: Number of blocks in dense layer 3.
        n_dense_blocks_4: Number of blocks in dense layer 4.
        upsampling_layers: Use upsampling instead of transposed convolutions.
        interp: Method to use for interpolation when upsampling smaller features.
        up_blocks: Number of upsampling steps to perform. The backbone reduces
            the output scale by 1/32. If set to 5, outputs will be upsampled to the
            input resolution.
        refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
    """
    
    n_dense_blocks_1: int = 3
    n_dense_blocks_2: int = 6
    n_dense_blocks_3: int = 12
    n_dense_blocks_4: int = 8
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

        # Automatically downloads weights
        backbone_model = applications.densenet.DenseNet(
            blocks=self.dense_blocks,
            include_top=False,
            input_shape=x.shape[1:],
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
    def dense_blocks(self):
        return [
        self.n_dense_blocks_1,
        self.n_dense_blocks_2,
        self.n_dense_blocks_3,
        self.n_dense_blocks_4
        ]

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        # This is a fixed constant for this backbone.
        return 5

    @property
    def output_scale(self):
        """Returns relative scaling factor of this backbone."""

        return 1 / (2 ** (self.down_blocks - self.up_blocks))

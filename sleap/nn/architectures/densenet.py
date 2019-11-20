"""Implements wrappers for constructing (optionally pretrained) DenseNets.

See original paper:
https://arxiv.org/abs/1608.06993
"""

import attr
from typing import List, Tuple, Union
from sleap.nn.architectures import common
import tensorflow as tf
import numpy as np
from keras_applications import densenet as bb

layers = tf.keras.layers
bb.backend = tf.keras.backend
bb.layers = tf.keras.layers
bb.models = tf.keras.models
bb.keras_utils = tf.keras.utils


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
        backbone_model = bb.DenseNet121(
            include_top=False,
            input_shape=x.shape[1:],
            weights="imagenet" if self.pretrained else None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(
            x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up,
            interp=self.interp,
        )

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
        backbone_model = bb.DenseNet169(
            include_top=False,
            input_shape=x.shape[1:],
            weights="imagenet" if self.pretrained else None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(
            x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up,
            interp=self.interp,
        )

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
        backbone_model = bb.DenseNet201(
            include_top=False,
            input_shape=x.shape[1:],
            weights="imagenet" if self.pretrained else None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(
            x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up,
            interp=self.interp,
        )

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
        backbone_model = bb.DenseNet(
            blocks=self.dense_blocks,
            include_top=False,
            input_shape=x.shape[1:],
            weights=None,
            pooling=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )

        # Output size is reduced by factor of 32 (2 ** 5)
        x = backbone_model(x)

        # Upsampling blocks.
        x = common.upsampling_blocks(
            x,
            up_blocks=self.up_blocks,
            upsampling_layers=self.upsampling_layers,
            refine_conv_up=self.refine_conv_up,
            interp=self.interp,
        )

        x = tf.keras.layers.Conv2D(num_output_channels, (3, 3), padding="same")(x)

        return x

    @property
    def dense_blocks(self):
        return [
            self.n_dense_blocks_1,
            self.n_dense_blocks_2,
            self.n_dense_blocks_3,
            self.n_dense_blocks_4,
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


def make_backbone(x_in, blocks, stem_stride=1, stem_filters=64, return_mid_feats=False):

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x_in)
    x = tf.keras.layers.Conv2D(
        stem_filters, 7, strides=1, use_bias=False, name="conv1/conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="conv1/bn")(x)
    x = tf.keras.layers.Activation("relu", name="conv1/relu")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=stem_stride, name="pool1")(x)

    x_mids = []
    for i, block in enumerate(blocks):
        x = bb.transition_block(x, 0.5, name=f"pool{i + 1}")
        x = bb.dense_block(x, block, name=f"conv{i + 2}")
        x_mids.append(x)

    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="bn")(x)
    x = tf.keras.layers.Activation("relu", name="relu")(x)

    model = tf.keras.Model(x_in, x, name="UDenseNet_backbone")

    if return_mid_feats:
        return model, x_mids
    else:
        return model


def make_head(feats, output_channels, up_blocks, filters=64, mid_feats=None, name=""):
    filters = np.array(filters)
    if filters.size == 1:
        filters = np.repeat(filters, up_blocks)

    x = feats
    x_mid_out = []
    for i in range(up_blocks):
        x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)

        if mid_feats is not None:
            for j, mid_feat in enumerate(mid_feats):
                if mid_feat.shape[1] == x.shape[1]:
                    x = tf.keras.layers.Concatenate(axis=-1)([x, mid_feat])

        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters[i] // 2,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
        )(x)
        x_mid_out.append(x)

    x = tf.keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation=None,
        name=f"{name}_out",
    )(x)
    return x, x_mid_out


@attr.s(auto_attribs=True)
class UDenseNet:
    """UDenseNet backbone, a UNet-like architecture with skip connections to heads.

    Attributes:
        stem_stride: Initial downsampling stride in the stem block.
        stem_filters: Initial number of conv filters in the stem block.
        dense_blocks: List of integers defining the size of each dense block. Can be of
            any length > 0.
        output_scale: Scale of the output tensor relative to the input.
        n_heads: Number of heads to produce. Intermediate heads will pass features from
            every scale to the next head, starting from the first backbone with dense
            blocks and transitions.
        head_filters: Filters to use in each head block after concatenation with
            previous filters.
    """

    stem_stride: int = 1
    stem_filters: int = 64
    dense_blocks: List[int] = [2, 4, 6, 8]
    output_scale: Union[float, List[float]] = 1.0
    n_heads: int = 1
    head_filters: Union[int, List[int]] = 64

    def output(self, x_in, n_output_channels):
        """Builds the layers for this backbone and return the output tensor.

        Args:
            x_in: Input 4-D tf.Tensor.
            n_output_channels: Number of output channels.

        Returns:
            A tf.keras.Model with as many outputs as the n_heads attribute for this
                model.
        """

        backbone, backbone_mid_feats = make_backbone(
            x_in,
            self.dense_blocks,
            stem_stride=self.stem_stride,
            stem_filters=self.stem_filters,
            return_mid_feats=True,
        )

        output_scales = self.output_scale
        if not isinstance(output_scales, list):
            output_scales = [output_scales] * self.n_heads

        heads_filters = self.head_filters
        if not isinstance(heads_filters, list):
            heads_filters = [heads_filters] * self.n_heads

        head_input = backbone.output
        mid_feats = backbone_mid_feats[::-1]
        outputs = []
        for head, (output_scale, head_filters) in enumerate(
            zip(output_scales, heads_filters)
        ):
            output_size = int(x_in.shape[1] * output_scale)
            up_blocks = int(np.log(output_size / backbone.output_shape[1]) / np.log(2))

            x, head_mid_feats = make_head(
                head_input,
                n_output_channels,
                up_blocks,
                filters=head_filters,
                mid_feats=mid_feats,
                name=f"head{head + 1}",
            )
            outputs.append(x)

            # Update middle features by scale.
            old_mid_feat_sizes = [f.shape[1] for f in mid_feats]
            new_mid_feat_sizes = [f.shape[1] for f in head_mid_feats]
            all_mid_feat_sizes = np.unique(old_mid_feat_sizes + new_mid_feat_sizes)
            next_mid_feats = []
            for feat_size in all_mid_feat_sizes:
                if feat_size in new_mid_feat_sizes:
                    next_mid_feats.append(
                        head_mid_feats[new_mid_feat_sizes.index(feat_size)]
                    )
                else:
                    next_mid_feats.append(
                        mid_feats[old_mid_feat_sizes.index(feat_size)]
                    )
            mid_feats = next_mid_feats

        return tf.keras.Model(x_in, outputs, name="UDenseNet")

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        return int(np.log(self.stem_stride) / np.log(2)) + len(self.dense_blocks)

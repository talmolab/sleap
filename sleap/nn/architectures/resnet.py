"""ResNet-based backbones.

This module primarily generalizes the ResNet architectures for configurable output
stride based on atrous convolutions, DeepLabv2-style (https://arxiv.org/abs/1606.00915).

ResNet variants that have pretrained weights can be loaded for transfer learning.

Based on the tf.keras.applications implementation:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py
"""

import os
import tensorflow as tf
import numpy as np
import attr

from typing import Tuple, Optional, Text, Callable, Mapping, Sequence, Any, List
from sleap.nn.architectures.upsampling import IntermediateFeature, UpsamplingStack
from sleap.nn.config import ResNetConfig


BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
)
WEIGHTS_HASHES = {
    "resnet50": "4d473c1dd8becc155b73f8504c6f6626",
    "resnet101": "88cf7a10940856eca736dc7b7e228a21",
    "resnet152": "ee4c566cf9a93f14d82f913c2dc6dd0c",
    "resnet50v2": "fac2f116257151a9d068a22e544a4917",
    "resnet101v2": "c0ed64b8031c3730f411d2eb4eea35b5",
    "resnet152v2": "ed17cf2e0169df9d443503ef94b23b33",
    "resnext50": "62527c363bdd9ec598bed41947b379fc",
    "resnext101": "0f678c91647380debd923963594981b3",
}
WEIGHTS_FILENAMES = {
    model_name: model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
    for model_name in WEIGHTS_HASHES.keys()
}
WEIGHTS_URIS = {
    model_name: BASE_WEIGHTS_PATH + filename
    for model_name, filename in WEIGHTS_FILENAMES.items()
}


def make_resnet_model(
    backbone_fn: Callable[[tf.Tensor, int], tf.Tensor],
    preact: bool = False,
    use_bias: bool = True,
    model_name: Text = "resnet",
    weights: Text = "imagenet",
    input_tensor: Optional[tf.Tensor] = None,
    input_shape: Optional[Tuple[int]] = None,
    stem_filters: int = 64,
    stem_stride1: int = 2,
    stem_stride2: int = 2,
) -> Tuple[tf.keras.Model, List[IntermediateFeature]]:
    """Instantiate the ResNet, ResNetV2 (TODO), and ResNeXt (TODO) architecture.

    Optionally loads weights pre-trained on ImageNet.

    Args:
        backbone_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
           (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `tf.keras.layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.

    Returns:
        A tuple of the `tf.keras.Model` mapping input to final feature outputs, and a
        list of `IntermediateFeature`s from every block in the backbone.

    Raises:
        ValueError: in case of invalid argument for `weights`.
    """

    if not (weights in {"imagenet", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if input_tensor is None:
        # Create input layer if tensor was not provided.
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    intermediate_feats = []

    # First stem block.
    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(
        img_input
    )
    x = tf.keras.layers.Conv2D(
        stem_filters, 7, strides=stem_stride1, use_bias=use_bias, name="conv1_conv"
    )(x)

    if not preact:
        x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(x)
        x = tf.keras.layers.Activation("relu", name="conv1_relu")(x)

    intermediate_feats.append(IntermediateFeature(tensor=x, stride=stem_stride1))

    # Second stem block.
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=stem_stride2, name="pool1_pool")(x)
    intermediate_feats.append(
        IntermediateFeature(tensor=x, stride=stem_stride1 * stem_stride2)
    )

    # Main backbone stack.
    x, backbone_intermediate_feats = backbone_fn(
        x, current_stride=stem_stride1 * stem_stride2
    )
    intermediate_feats.extend(backbone_intermediate_feats)

    if preact:
        x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="post_bn")(x)
        x = tf.keras.layers.Activation("relu", name="post_relu")(x)

    # Ensure that the model takes into account any potential predecessors of
    # `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    if weights == "imagenet" and model_name in WEIGHTS_HASHES:
        # Download and load pretrained ImageNet weights.
        weights_path = tf.keras.utils.get_file(
            fname=WEIGHTS_FILENAMES[model_name],
            origin=WEIGHTS_URIS[model_name],
            cache_subdir="models",
            file_hash=WEIGHTS_HASHES[model_name],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        # Load custom weights.
        model.load_weights(weights)

    return model, intermediate_feats


def block_v1(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation_rate: int = 1,
    conv_shortcut: bool = True,
    name: Optional[Text] = None,
) -> tf.Tensor:
    """Create a ResNetv1 residual block.

    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        dilation_rate: default 1, atrous convolution dilation rate of first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    Returns:
        Output tensor for the residual block.
    """

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            dilation_rate=dilation_rate,
            name=name + "_0_conv",
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(
            epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(
        filters, 1, strides=stride, dilation_rate=dilation_rate, name=name + "_1_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + "_3_bn")(x)

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def stack_v1(
    x: tf.Tensor,
    filters: int,
    blocks: int,
    stride1: int = 2,
    dilation_rate: int = 1,
    name: Optional[Text] = None,
) -> tf.Tensor:
    """Create a set of stacked ResNetv1 residual blocks.

    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        dilation_rate: default 1, atrous convolution dilation rate of first layer in the first block.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block_v1(
        x, filters, stride=stride1, dilation_rate=dilation_rate, name=name + "_block1"
    )
    for i in range(2, blocks + 1):
        x = block_v1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def make_backbone_fn(
    stack_fn: Callable[[tf.Tensor, Any], Tuple[tf.Tensor, List[IntermediateFeature]]],
    stack_configs: Sequence[Mapping[Text, Any]],
    output_stride: int,
) -> Callable[[tf.Tensor, int], tf.Tensor]:
    """Return a function that creates a block stack with output stride adjustments.

    Args:
        stack_fn: Function that takes a tensor as the first positional argument,
            followed by any number of keyword arguments. This function will construct
            each stack of blocks in the backbone.
        stack_configs: List of dictionaries containing the keyword arguments for each
            stack. The stack_fn will be called consecutively with each element of
            stack_configs expanded as keyword arguments. Each element must contain the
            "stride1" key specifying the stride of the first layer of the stack. This
            may be adjusted to achieve the desired target output stride by converting
            strided convs into dilated convs.
        output_stride: The desired target output stride. The final output of the
            returned backbone creation function will be at this stride relative to the
            input stride.

    Returns:
        Function that creates the backbone stacks based on the stack_configs.

        This function will have the signature:
            x_out, intermediate_feats = backbone_fn(x_in, current_stride)

        The current stride describes the stride of the x_in input tensor.

    Raises:
        ValueError: If the desired output stride cannot be achieved.
    """

    def backbone_fn(x: tf.Tensor, current_stride: int) -> tf.Tensor:
        """Construct backbone from partial configuration."""
        dilation_rate = 1
        intermediate_feats = []

        for stack_config in stack_configs:

            # Adjust stride or dilation rate.
            if current_stride < output_stride:
                current_stride *= stack_config["stride1"]
                stride1 = stack_config["stride1"]
            elif current_stride == output_stride:
                stride1 = 1
                if stack_config["stride1"] > 1:
                    dilation_rate *= 2
            else:
                raise ValueError(
                    f"Could not adjust output stride. Current: {current_stride}, "
                    f"desired: {output_stride}"
                )

            stack_config.update(dilation_rate=dilation_rate, stride1=stride1)

            # Create a stack block.
            x = stack_fn(x, **stack_config)

            # Save intermediate feature with stride metadata. This can serve as the
            # source tensor for a skip connection.
            intermediate_feats.append(
                IntermediateFeature(tensor=x, stride=current_stride)
            )

        return x, intermediate_feats

    return backbone_fn


def tile_channels(X: tf.Tensor) -> tf.Tensor:
    """Tile single channel to 3 channel tensor.

    This functon is useful to replicate grayscale single-channel images into 3-channel
    monochrome RGB images.

    Args:
        X: Tensor of shape (samples, height, width, 1).

    Returns:
        Tensor of shape (samples, height, width, 3) where the channels are identical.
    """
    return tf.tile(X, [1, 1, 1, 3])


def imagenet_preproc_v1(X: tf.Tensor) -> tf.Tensor:
    """Preprocess images according to ImageNet/caffe/channels_last.

    Args:
        X: Tensor of shape (samples, height, width, 3) of dtype float32 with values in
            the range [0, 1]. The channels axis is in RGB ordering.

    Returns:
        Tensor of the same shape and dtype with channels reversed to BGR ordering and
        values scaled to [0, 255] and subtracted by the ImageNet/caffe pretrained model
        channel means (103.939, 116.779, 123.68) for BGR respectively. The effective
        range of values will then be around ~[-128, 127].
    """
    X = X * 255
    X = X[..., ::-1]
    X = X - tf.constant(
        [[[[103.939, 116.779, 123.68]]]], dtype=tf.float32, shape=[1, 1, 1, 3]
    )

    return X


@attr.s(auto_attribs=True)
class ResNetv1:
    """ResNetv1 backbone with configurable output stride and pretrained weights.

    Attributes:
        model_name: Backbone name. Must be one of "resnet50", "resnet101", or
            "resnet152" if using pretrained weights.
        stack_configs: List of dictionaries containing the keyword arguments for each
            stack. The stack_fn will be called consecutively with each element of
            stack_configs expanded as keyword arguments. Each element must contain the
            "stride1" key specifying the stride of the first layer of the stack. This
            may be adjusted to achieve the desired target output stride by converting
            strided convs into dilated convs.
        upsampling_stack: Definition of the upsampling layers that convert the ResNet
            backbone features into the output features with the desired stride. See
            the `UpsamplingStack` documentation for more.  If not provided, the
            activations from the last backbone block will be the output.
        features_output_stride: Output stride of the standard ResNet backbone.
            Canonically, ResNets have 5 layers with 2-stride, resulting in a final
            feature output layer with stride of 32. If a lower value is specified, the
            strided convolution layers will be adjusted to have a stride of 1, but the
            receptive field is maintained by compensating with dilated (atrous)
            convolution kernel expansion, in the same style as DeepLabv2. Valid values
            are 1, 2, 4, 8, 16 or 32.
        pretrained: If True, initialize with weights pretrained on ImageNet. If False,
            random weights will be used.
        frozen: If True, the backbone weights will be not be trainable. This is useful
            for fast fine-tuning of ResNet features, but relies on having an upsampling
            stack with sufficient representational capacity to adapt the fixed features.
        skip_connections: If True, form skip connections between outputs of each block
            in the ResNet backbone and the upsampling stack.

    Note:
        This defines the ResNetv1 architecture, not v2.
    """

    model_name: Text
    stack_configs: Sequence[Mapping[Text, Any]]
    upsampling_stack: Optional[UpsamplingStack] = None
    features_output_stride: int = 16
    pretrained: bool = True
    frozen: bool = False
    skip_connections: bool = False

    @classmethod
    def from_config(cls, config: ResNetConfig) -> "ResNetv1":
        """Create a model from a set of configuration parameters.

        Args:
            config: An `ResNetConfig` instance with the desired parameters.

        Returns:
            An instance of this class with the specified configuration.
        """
        if config.version == "ResNet50":
            new_cls = ResNet50
        elif config.version == "ResNet101":
            new_cls = ResNet101
        elif config.version == "ResNet152":
            new_cls = ResNet152
        else:
            raise ValueError(
                f"Invalid ResNet version in the configuration: {config.version}"
            )

        upsampling_stack = None
        skip_connections = False
        if config.upsampling is not None:
            upsampling_stack = UpsamplingStack.from_config(
                config=config.upsampling, output_stride=config.output_stride
            )
            skip_connections = config.upsampling.skip_connections is not None

        return new_cls(
            upsampling_stack=upsampling_stack,
            features_output_stride=config.max_stride,
            pretrained=config.weights != "random",
            frozen=config.weights == "frozen",
            skip_connections=skip_connections,
        )

    @property
    def down_blocks(self) -> int:
        """Return the number of downsampling steps in the model."""
        return int(np.log2(self.features_output_stride))

    @property
    def maximum_stride(self) -> int:
        """Return the maximum stride that the input must be divisible by."""
        return self.features_output_stride

    @property
    def output_stride(self) -> int:
        """Return stride of the output of the backbone."""
        if self.upsampling_stack is not None:
            return self.upsampling_stack.output_stride
        else:
            return self.features_output_stride

    @property
    def output_scale(self) -> float:
        """Return relative scaling factor of this backbone."""
        return 1 / float(self.output_stride)

    def make_backbone(
        self, x_in: tf.Tensor
    ) -> Tuple[tf.Tensor, List[IntermediateFeature]]:
        """Create the full backbone starting with the specified input tensor.

        Args:
            x_in: Input tensor of shape (samples, height, width, channels).

        Returns:
            A tuple of the final output tensor at the stride specified by the
            `upsampling_stack.features_output_stride` class attribute, and a list of
            intermediate tensors after each upsampling step.

            The intermediate features are useful when creating multi-head architectures
            with different output strides for the heads.
        """
        # Apply expected preprocessing to the inputs if using pretrained weights.
        if self.pretrained:
            if x_in.shape[-1] == 1:
                x_in = tf.keras.layers.Lambda(tile_channels, name="tile_channels")(x_in)

            x_in = tf.keras.layers.Lambda(
                imagenet_preproc_v1, name="imagenet_preproc_v1"
            )(x_in)

        # Adjust stem strides if necessary.
        stem_stride1 = 2
        stem_stride2 = 2
        if self.features_output_stride <= 2:
            stem_stride2 = 1
        if self.features_output_stride == 1:
            stem_stride1 = 1

        # Configure the backbone instantiation function.
        backbone_fn = make_backbone_fn(
            stack_fn=stack_v1,
            stack_configs=self.stack_configs,
            output_stride=self.features_output_stride,
        )

        # Create the backbone and return intermediate features.
        backbone, intermediate_feats = make_resnet_model(
            backbone_fn=backbone_fn,
            preact=False,
            use_bias=True,
            model_name=self.model_name,
            weights="imagenet" if self.pretrained else None,
            input_tensor=x_in,
            stem_stride1=stem_stride1,
            stem_stride2=stem_stride2,
        )

        # Freeze pretrained weights so they are not updated in training.
        if self.frozen:
            for layer in backbone.layers:
                layer.trainable = False

        if self.upsampling_stack is not None:
            # Use post-stem intermediate activations for skip connections if specified.
            skip_sources = None
            if self.skip_connections:
                skip_sources = intermediate_feats[2:]

            # Return the result of the upsampling stack, starting with the ResNet
            # features as input.
            return self.upsampling_stack.make_stack(
                backbone.output,
                current_stride=self.features_output_stride,
                skip_sources=skip_sources,
            )

        else:
            # Just return the final activation layer and backbone intermediate features.
            return backbone.output, intermediate_feats


@attr.s
class ResNet50(ResNetv1):
    """ResNet50 backbone.

    This model has a stack of 3, 4, 6 and 3 residual blocks.

    Attributes:
        upsampling_stack: Definition of the upsampling layers that convert the ResNet
            backbone features into the output features with the desired stride. See
            the `UpsamplingStack` documentation for more.  If not provided, the
            activations from the last backbone block will be the output.
        features_output_stride: Output stride of the standard ResNet backbone.
            Canonically, ResNets have 5 layers with 2-stride, resulting in a final
            feature output layer with stride of 32. If a lower value is specified, the
            strided convolution layers will be adjusted to have a stride of 1, but the
            receptive field is maintained by compensating with dilated (atrous)
            convolution kernel expansion, in the same style as DeepLabv2. Valid values
            are 1, 2, 4, 8, 16 or 32.
        pretrained: If True, initialize with weights pretrained on ImageNet. If False,
            random weights will be used.
        frozen: If True, the backbone weights will be not be trainable. This is useful
            for fast fine-tuning of ResNet features, but relies on having an upsampling
            stack with sufficient representational capacity to adapt the fixed features.
        skip_connections: If True, form skip connections between outputs of each block
            in the ResNet backbone and the upsampling stack.

    Note:
        This defines the ResNetv1 architecture, not v2.
    """

    model_name = attr.ib()
    stack_configs = attr.ib()

    @model_name.default
    def _fixed_model_name(self) -> Text:
        """ResNet50 model name."""
        return "resnet50"

    @stack_configs.default
    def _fixed_stack_configs(self) -> Sequence[Mapping[Text, Any]]:
        """ResNet50 layer stack configuration."""
        return [
            dict(filters=64, blocks=3, stride1=1, name="conv2"),
            dict(filters=128, blocks=4, stride1=2, name="conv3"),
            dict(filters=256, blocks=6, stride1=2, name="conv4"),
            dict(filters=512, blocks=3, stride1=2, name="conv5"),
        ]

    def __attrs_post_init__(self):
        """Enforce fixed attributes."""
        self.model_name = self._fixed_model_name()
        self.stack_configs = self._fixed_stack_configs()


@attr.s
class ResNet101(ResNetv1):
    """ResNet101 backbone.

    This model has a stack of 3, 4, 23 and 3 residual blocks.

    Attributes:
        upsampling_stack: Definition of the upsampling layers that convert the ResNet
            backbone features into the output features with the desired stride. See
            the `UpsamplingStack` documentation for more.  If not provided, the
            activations from the last backbone block will be the output.
        features_output_stride: Output stride of the standard ResNet backbone.
            Canonically, ResNets have 5 layers with 2-stride, resulting in a final
            feature output layer with stride of 32. If a lower value is specified, the
            strided convolution layers will be adjusted to have a stride of 1, but the
            receptive field is maintained by compensating with dilated (atrous)
            convolution kernel expansion, in the same style as DeepLabv2. Valid values
            are 1, 2, 4, 8, 16 or 32.
        pretrained: If True, initialize with weights pretrained on ImageNet. If False,
            random weights will be used.
        frozen: If True, the backbone weights will be not be trainable. This is useful
            for fast fine-tuning of ResNet features, but relies on having an upsampling
            stack with sufficient representational capacity to adapt the fixed features.
        skip_connections: If True, form skip connections between outputs of each block
            in the ResNet backbone and the upsampling stack.

    Note:
        This defines the ResNetv1 architecture, not v2.
    """

    model_name = attr.ib()
    stack_configs = attr.ib()

    @model_name.default
    def _fixed_model_name(self) -> Text:
        """ResNet101 model name."""
        return "resnet101"

    @stack_configs.default
    def _fixed_stack_configs(self) -> Sequence[Mapping[Text, Any]]:
        """ResNet101 layer stack configuration."""
        return [
            dict(filters=64, blocks=3, stride1=1, name="conv2"),
            dict(filters=128, blocks=4, stride1=2, name="conv3"),
            dict(filters=256, blocks=23, stride1=2, name="conv4"),
            dict(filters=512, blocks=3, stride1=2, name="conv5"),
        ]

    def __attrs_post_init__(self):
        """Enforce fixed attributes."""
        self.model_name = self._fixed_model_name()
        self.stack_configs = self._fixed_stack_configs()


@attr.s
class ResNet152(ResNetv1):
    """ResNet152 backbone.

    This model has a stack of 3, 4, 23 and 3 residual blocks.

    Attributes:
        upsampling_stack: Definition of the upsampling layers that convert the ResNet
            backbone features into the output features with the desired stride. See
            the `UpsamplingStack` documentation for more. If not provided, the
            activations from the last backbone block will be the output.
        features_output_stride: Output stride of the standard ResNet backbone.
            Canonically, ResNets have 5 layers with 2-stride, resulting in a final
            feature output layer with stride of 32. If a lower value is specified, the
            strided convolution layers will be adjusted to have a stride of 1, but the
            receptive field is maintained by compensating with dilated (atrous)
            convolution kernel expansion, in the same style as DeepLabv2. Valid values
            are 1, 2, 4, 8, 16 or 32.
        pretrained: If True, initialize with weights pretrained on ImageNet. If False,
            random weights will be used.
        frozen: If True, the backbone weights will be not be trainable. This is useful
            for fast fine-tuning of ResNet features, but relies on having an upsampling
            stack with sufficient representational capacity to adapt the fixed features.
        skip_connections: If True, form skip connections between outputs of each block
            in the ResNet backbone and the upsampling stack.

    Note:
        This defines the ResNetv1 architecture, not v2.
    """

    model_name = attr.ib()
    stack_configs = attr.ib()

    @model_name.default
    def _fixed_model_name(self) -> Text:
        """ResNet152 model name."""
        return "resnet152"

    @stack_configs.default
    def _fixed_stack_configs(self) -> Sequence[Mapping[Text, Any]]:
        """ResNet152 layer stack configuration."""
        return [
            dict(filters=64, blocks=3, stride1=1, name="conv2"),
            dict(filters=128, blocks=8, stride1=2, name="conv3"),
            dict(filters=256, blocks=36, stride1=2, name="conv4"),
            dict(filters=512, blocks=3, stride1=2, name="conv5"),
        ]

    def __attrs_post_init__(self):
        """Enforce fixed attributes."""
        self.model_name = self._fixed_model_name()
        self.stack_configs = self._fixed_stack_configs()

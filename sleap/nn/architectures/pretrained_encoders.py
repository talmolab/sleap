"""Encoder-decoder backbones with pretrained encoder models.

This module enables encoder-decoder style architectures using a wide range of
state-of-the-art architectures as the encoder, with a UNet-like upsampling stack that
also takes advantage of skip connections from the encoder blocks. Additionally, the
encoder models can use ImageNet-pretrained weights for initialization.

This module is made possible by the work by Pavel Yakubovskiy who graciously put
together a library implementing common pretrained fully-convolutional architectures and
their weights.

For more info, see the source repositories:
    - https://github.com/qubvel/segmentation_models
    - https://github.com/qubvel/classification_models

License:

The MIT License

Copyright (c) 2018, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import attr
from typing import Tuple, List

from sleap.nn.architectures.upsampling import IntermediateFeature
from sleap.nn.data.normalization import ensure_rgb
from sleap.nn.config import PretrainedEncoderConfig

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import io
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    # Import segmentation_models suppressing output to stdout about backend.
    import segmentation_models as sm


AVAILABLE_ENCODERS = [
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50",
    "resnext101",
    "inceptionv3",
    "inceptionresnetv2",
    "densenet121",
    "densenet169",
    "densenet201",
    "seresnet18",
    "seresnet34",
    "seresnet50",
    "seresnet101",
    "seresnet152",
    "seresnext50",
    "seresnext101",
    "senet154",
    "mobilenet",
    "mobilenetv2",
    "efficientnetb0",
    "efficientnetb1",
    "efficientnetb2",
    "efficientnetb3",
    "efficientnetb4",
    "efficientnetb5",
    "efficientnetb6",
    "efficientnetb7",
]


@attr.s(auto_attribs=True)
class UnetPretrainedEncoder:
    """UNet with an (optionally) pretrained encoder model.

    This backbone enables the use of a variety of popular neural network architectures
    for feature extraction in the backbone. These can be used with ImageNet-pretrained
    weights for initialization. The decoder (upsampling stack) receives skip connections
    from intermediate activations in the encoder blocks.

    All of the encoder models have a maximum stride of 32 and the input does not need to
    be preprocessed in any special way. Grayscale images will be tiled to have 3
    channels automatically.

    See https://github.com/qubvel/classification_models#specification for more
    information on the individual backbones.

    Attributes:
        encoder: Name of the model to use as the encoder. Valid encoder names are:
            - `"vgg16", "vgg19",`
            - `"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"`
            - `"resnext50", "resnext101"`
            - `"inceptionv3", "inceptionresnetv2"`
            - `"densenet121", "densenet169", "densenet201"`
            - `"seresnet18", "seresnet34", "seresnet50", "seresnet101", "seresnet152",`
              `"seresnext50", "seresnext101", "senet154"`
            - `"mobilenet", "mobilenetv2"`
            - `"efficientnetb0", "efficientnetb1", "efficientnetb2", "efficientnetb3",`
              `"efficientnetb4", "efficientnetb5", "efficientnetb6", "efficientnetb7"`
            Defaults to `"mobilenetv2"`
        decoder_filters: A tuple of integers denoting the number of filters to use in
            the upsampling blocks of the decoder, starting from the lowest resolution
            block. The length of this attribute also specifies the number of upsampling
            steps and therefore the output stride of the backbone. Specify 5 filter
            numbers to get an output stride of 1 (same size as the input). Defaults to
        pretrained: If `True` (the default), load pretrained weights for the encoder. If
            `False`, the same model architecture will be used for the encoder but the
            weights will be randomly initialized.
    """

    encoder: str = attr.ib(
        default="efficientnetb0", validator=attr.validators.in_(AVAILABLE_ENCODERS)
    )
    decoder_filters: Tuple[int] = (256, 256, 128, 128)
    pretrained: bool = True

    @classmethod
    def from_config(cls, config: PretrainedEncoderConfig) -> "UnetPretrainedEncoder":
        """Create the backbone from a configuration.

        Args:
            config: A `PretrainedEncoderConfig` instance specifying the
                configuration of the backbone.

        Returns:
            An instantiated `UnetPretrainedEncoder`.
        """
        up_blocks = int(np.log2(32 // config.output_stride))
        decoder_filters = [
            int(config.decoder_filters * (config.decoder_filters_rate ** i))
            for i in range(up_blocks)
        ]

        return cls(
            encoder=config.encoder,
            pretrained=config.pretrained,
            decoder_filters=tuple(decoder_filters),
        )

    @property
    def down_blocks(self) -> int:
        """Return the number of downsampling blocks in the encoder."""
        return 5

    @property
    def up_blocks(self) -> int:
        """Return the number of upsampling blocks in the decoder."""
        return len(self.decoder_filters)

    @property
    def maximum_stride(self) -> int:
        """Return the maximum encoder stride relative to the input."""
        return 32

    @property
    def output_stride(self) -> int:
        """Return the stride of the output of the decoder."""
        return int(2 ** (self.down_blocks - self.up_blocks))

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
        img_shape = x_in.shape[1:]

        if self.pretrained and img_shape[-1] == 1:
            # Add preprocessing layer if needed.
            x_in = tf.keras.layers.Lambda(lambda x: ensure_rgb(x), name="ensure_rgb")(
                x_in
            )
            img_shape = (img_shape[0], img_shape[1], 3)

        # Create base model.
        base_model = sm.models.unet.Unet(
            backbone_name=self.encoder,
            input_shape=img_shape,
            classes=1,
            activation="linear",
            encoder_weights="imagenet" if self.pretrained else None,
            decoder_block_type="upsampling",
            decoder_filters=self.decoder_filters,
            decoder_use_batchnorm=True,
            layers=tf.keras.layers,
            models=tf.keras.models,
            backend=tf.keras.backend,
            utils=tf.keras.utils,
        )

        # Collect intermediate features from the decoder.
        x_outs = []
        for i in range(self.up_blocks):
            x_outs.append(base_model.get_layer(f"decoder_stage{i}b_relu").output)

        # Build a model that outputs all decoder block activations.
        features_model = tf.keras.Model(inputs=base_model.inputs, outputs=x_outs)

        # Connect the inputs to the model graph.
        backbone_model = tf.keras.Model(
            tf.keras.utils.get_source_inputs(x_in)[0], features_model(x_in)
        )

        # Collect output tensors.
        intermediate_features = []
        for i in range(self.up_blocks):
            intermediate_features.append(
                IntermediateFeature(
                    tensor=backbone_model.outputs[i],
                    stride=2 ** (4 - i),
                )
            )
        output = intermediate_features[-1].tensor

        return output, intermediate_features

import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only; use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import UnetPretrainedEncoder
from sleap.nn.config import UnetPretrainedEncoderConfig


def test_unet_pretrained_backbone():
    backbone = UnetPretrainedEncoder(
        encoder="mobilenetv2",
        decoder_filters=(32, 32, 32, 32),
        pretrained=True,
    )
    assert backbone.pretrained
    assert backbone.down_blocks == 5
    assert backbone.up_blocks == 4
    assert backbone.maximum_stride == 32
    assert backbone.output_stride == 2

    x_in = tf.keras.layers.Input([256, 256, 1])
    x, intermediate_feats = backbone.make_backbone(x_in)

    assert tuple(x.shape) == (None, 128, 128, 32)
    assert len(intermediate_feats) == 4


def test_unet_pretrained_backbone_from_config():
    backbone = UnetPretrainedEncoder.from_config(
        UnetPretrainedEncoderConfig(
            encoder="resnet18",
            pretrained=False,
            decoder_filters=256,
            decoder_filters_rate=0.5,
            output_stride=1,
        )
    )

    assert backbone.decoder_filters == (256, 128, 64, 32, 16)
    assert not backbone.pretrained
    assert backbone.encoder == "resnet18"
    assert backbone.output_stride == 1

import numpy as np
import tensorflow as tf
import pytest
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import UnetPretrainedEncoder
from sleap.nn.config import PretrainedEncoderConfig


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("input_channels", [1, 3])
def test_unet_pretrained_backbone(pretrained, input_channels):
    backbone = UnetPretrainedEncoder(
        encoder="resnet18",
        decoder_filters=(64, 32, 16, 8),
        pretrained=pretrained,
    )
    assert backbone.pretrained == pretrained
    assert backbone.down_blocks == 5
    assert backbone.up_blocks == 4
    assert backbone.maximum_stride == 32
    assert backbone.output_stride == 2

    x_in = tf.keras.layers.Input([256, 256, input_channels])
    x, intermediate_feats = backbone.make_backbone(x_in)

    assert tuple(x.shape) == (None, 128, 128, 8)
    assert len(intermediate_feats) == 4
    assert tuple(intermediate_feats[0].tensor.shape) == (None, 16, 16, 64)
    assert tuple(intermediate_feats[1].tensor.shape) == (None, 32, 32, 32)
    assert tuple(intermediate_feats[2].tensor.shape) == (None, 64, 64, 16)
    assert tuple(intermediate_feats[3].tensor.shape) == (None, 128, 128, 8)

    model = tf.keras.Model(x_in, x)
    preds = model.predict(np.zeros([1, 256, 256, input_channels], dtype="float32"))
    assert preds.shape == (1, 128, 128, 8)


def test_unet_pretrained_backbone_from_config():
    backbone = UnetPretrainedEncoder.from_config(
        PretrainedEncoderConfig(
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

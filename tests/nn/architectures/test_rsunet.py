import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()

from sleap.nn.architectures import RSUNet
from sleap.nn.config import RSUNetConfig


def test_rsunet_backbone():
    backbone = RSUNet(
        filters=32,
        filters_rate=2,
    )

    assert backbone.down_blocks == 4
    assert backbone.up_blocks == 2
    assert backbone.maximum_stride == 8
    assert backbone.output_stride == 1

    x_in = tf.keras.layers.Input([256, 256, 1])
    x, intermediate_feats = backbone.make_backbone(x_in)

    assert tuple(x.shape) == (None, 128, 128, 64)
    assert len(intermediate_feats) == 2
    assert tuple(intermediate_feats[0].tensor.shape) == (None, 64, 64, 128)
    assert tuple(intermediate_feats[1].tensor.shape) == (None, 128, 128, 64)

    model = tf.keras.Model(x_in, x)
    preds = model.predict(np.zeros([1, 256, 256, 1], dtype="float32"))
    assert preds.shape == (1, 128, 128, 64)

    def test_usunet_backbone_from_config():
        backbone = RSUNet.from_config(
            RSUNetConfig(
                filters=32,
                filters_rate=2,
                down_blocks=4,
                upblocks=2,
            )
        )
        assert backbone.encoder == "rsunet"
        assert backbone.output_stride == 1

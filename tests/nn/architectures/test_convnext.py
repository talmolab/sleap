import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()

from sleap.nn.architectures import ConvNeXT
from sleap.nn.config import ConvNeXTConfig


def test_convnext_backbone():
    backbone = ConvNeXT(crop_size=160)

    assert backbone.crop_size == 160

    x_in = tf.keras.layers.Input([160, 160, 1])
    x, intermediate_feats = backbone.make_backbone(x_in)

    assert tuple(x.shape) == (None, 160, 160, 162)
    assert len(intermediate_feats) == 6
    assert tuple(intermediate_feats[0].tensor.shape) == (None, 5, 5, 768)
    assert tuple(intermediate_feats[1].tensor.shape) == (None, 10, 10, 32)
    assert tuple(intermediate_feats[2].tensor.shape) == (None, 20, 20, 48)
    assert tuple(intermediate_feats[3].tensor.shape) == (None, 40, 40, 72)
    assert tuple(intermediate_feats[4].tensor.shape) == (None, 80, 80, 108)

    model = tf.keras.Model(x_in, x)
    preds = model.predict(np.zeros([1, 256, 256, 1], dtype="float32"))
    assert preds.shape == (1, 128, 128, 64)

    def test_usunet_backbone_from_config():
        backbone = ConvNeXT.from_config(ConvNeXTConfig())
        assert backbone.encoder == "convnext"
        assert backbone.output_stride == 1


import pytest
import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

import sleap
from sleap.nn.data import pipelines


def test_pipeline_concatenation():
    A = pipelines.Pipeline.from_sequence([
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
        ])
    B = pipelines.Pipeline.from_sequence([
        pipelines.InstanceCropper(crop_width=16, crop_height=16)
        ])

    C = A + B
    assert len(C.transformers) == 2
    assert isinstance(C.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(C.transformers[1], pipelines.InstanceCropper)

    C = A | B
    assert len(C.transformers) == 2
    assert isinstance(C.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(C.transformers[1], pipelines.InstanceCropper)

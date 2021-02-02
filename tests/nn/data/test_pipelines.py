import pytest
import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

import sleap
from sleap.nn.data import pipelines


def test_pipeline_concatenation():
    A = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
    )
    B = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCropper(crop_width=16, crop_height=16)
    )

    C = A + B
    assert len(C.transformers) == 2
    assert isinstance(C.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(C.transformers[1], pipelines.InstanceCropper)

    C = A | B
    assert len(C.transformers) == 2
    assert isinstance(C.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(C.transformers[1], pipelines.InstanceCropper)

    D = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
    )
    D += pipelines.Pipeline.from_blocks(
        pipelines.InstanceCropper(crop_width=16, crop_height=16)
    )
    assert len(D.transformers) == 2
    assert isinstance(D.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(D.transformers[1], pipelines.InstanceCropper)

    E = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
    )
    E |= pipelines.Pipeline.from_blocks(
        pipelines.InstanceCropper(crop_width=16, crop_height=16)
    )
    assert len(E.transformers) == 2
    assert isinstance(E.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(E.transformers[1], pipelines.InstanceCropper)

    F = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
    )
    F += pipelines.InstanceCropper(crop_width=16, crop_height=16)
    assert len(F.transformers) == 2
    assert isinstance(F.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(F.transformers[1], pipelines.InstanceCropper)

    G = pipelines.Pipeline.from_blocks(
        pipelines.InstanceCentroidFinder(center_on_anchor_part=False)
    )
    G |= pipelines.InstanceCropper(crop_width=16, crop_height=16)
    assert len(G.transformers) == 2
    assert isinstance(G.transformers[0], pipelines.InstanceCentroidFinder)
    assert isinstance(G.transformers[1], pipelines.InstanceCropper)

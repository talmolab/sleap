import pytest
import sleap

from sleap.nn.model import Model
from sleap.nn.config import (
    SingleInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
    ClassMapsHeadConfig,
    HeadsConfig,
    UNetConfig,
    BackboneConfig,
    ModelConfig,
)

sleap.use_cpu_only()


def test_model_from_config():
    skel = sleap.Skeleton()
    skel.add_node("a")
    cfg = ModelConfig()
    cfg.backbone.unet = UNetConfig(filters=4, max_stride=4, output_stride=2)
    with pytest.raises(ValueError):
        Model.from_config(cfg, skeleton=skel)
    cfg.heads.single_instance = SingleInstanceConfmapsHeadConfig(
        part_names=None,
        sigma=1.5,
        output_stride=2,
        loss_weight=2.0,
        offset_refinement=True,
    )
    model = Model.from_config(cfg, skeleton=skel)

    assert isinstance(model.heads[0], sleap.nn.heads.SingleInstanceConfmapsHead)
    assert isinstance(model.heads[1], sleap.nn.heads.OffsetRefinementHead)

    keras_model = model.make_model(input_shape=(16, 16, 1))
    assert keras_model.output_names == [
        "SingleInstanceConfmapsHead",
        "OffsetRefinementHead",
    ]
    assert tuple(keras_model.outputs[0].shape) == (None, 8, 8, 1)
    assert tuple(keras_model.outputs[1].shape) == (None, 8, 8, 2)

    cfg.heads.single_instance = None
    with pytest.raises(ValueError):
        Model.from_config(cfg, skeleton=skel)

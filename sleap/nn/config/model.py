import attr
from typing import Optional, Text, List, Sequence, Tuple


@attr.s(auto_attribs=True)
class CentroidsHeadConfig:
    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsHeadConfig:
    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1


@attr.s(auto_attribs=True)
class CenteredInstanceConfmapsHeadConfig:
    anchor_part: Optional[Text] = None
    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1


@attr.s(auto_attribs=True)
class MultiInstanceConfmapsHeadConfig:
    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1


@attr.s(auto_attribs=True)
class PartAffinityFieldsHeadConfig:
    edges: Optional[Sequence[Tuple[Text, Text]]] = None
    sigma: float = 15.0
    output_stride: int = 1


@attr.s(auto_attribs=True)
class HeadsConfig:
    single_instance: Optional[SingleInstanceConfmapsHeadConfig] = None
    centroid: Optional[CentroidsHeadConfig] = None
    centered_instance: Optional[CenteredInstanceConfmapsHeadConfig] = None
    multi_instance: Optional[MultiInstanceConfmapsHeadConfig] = None
    pafs: Optional[PartAffinityFieldsHeadConfig] = None
    # TODO: implement mutual exclusivity in validators


@attr.s(auto_attribs=True)
class LEAPConfig:
    max_stride: int = 8  # determines down blocks
    output_stride: int = 1  # determines up blocks
    filters: int = 64
    filters_rate: float = 2
    up_interpolate: bool = False
    stacks: int = 1


@attr.s(auto_attribs=True)
class UNetConfig:
    stem_stride: Optional[int] = None  # if not None, use stem
    max_stride: int = 16  # determines down blocks
    output_stride: int = 1  # determines up blocks
    filters: int = 64
    filters_rate: float = 2
    middle_block: bool = True
    up_interpolate: bool = False
    stacks: int = 1


@attr.s(auto_attribs=True)
class HourglassConfig:
    stem_stride: int = 4  # if not None, use stem
    max_stride: int = 64  # determines down blocks
    output_stride: int = 4  # determines up blocks
    stem_filters: int = 128
    filters: int = 256
    filter_increase: int = 128
    stacks: int = 3


@attr.s(auto_attribs=True)
class UpsamplingConfig:
    method: Text = attr.ib(
        default="interpolation",
        validator=attr.validators.in_(["interpolation", "transposed_conv"]),
    )
    skip_connections: Optional[Text] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.in_(["add", "concatenate"])),
    )
    block_stride: int = 2
    filters: int = 64
    filters_rate: float = 1
    refine_convs: int = 2
    batch_norm: bool = True
    transposed_conv_kernel_size: int = 4


@attr.s(auto_attribs=True)
class ResNetConfig:
    version: Text = attr.ib(
        default="ResNet50",
        validator=attr.validators.in_(["ResNet50", "ResNet101", "ResNet152"]),
    )
    weights: Text = attr.ib(
        default="frozen", validator=attr.validators.in_(["random", "frozen", "tunable"])
    )
    upsampling: UpsamplingConfig = attr.ib(factory=UpsamplingConfig)
    output_stride: int = 4


@attr.s(auto_attribs=True)
class BackboneConfig:
    leap: Optional[LEAPConfig] = None
    unet: Optional[UNetConfig] = None
    hourglass: Optional[HourglassConfig] = None
    resnet: Optional[ResNetConfig] = None
    # TODO: implement mutual exclusivity in validators?


@attr.s(auto_attribs=True)
class ModelConfig:
    backbone: BackboneConfig = attr.ib(factory=BackboneConfig)
    heads: HeadsConfig = attr.ib(factory=HeadsConfig)

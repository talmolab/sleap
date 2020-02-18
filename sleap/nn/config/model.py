import attr
from typing import Optional, Text, List, Sequence, Tuple
from sleap.nn.config.utils import oneof


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
class MultiInstanceConfig:
    multi_instance: MultiInstanceConfmapsHeadConfig = attr.ib(factory=MultiInstanceConfmapsHeadConfig)
    pafs: PartAffinityFieldsHeadConfig = attr.ib(factory=PartAffinityFieldsHeadConfig)


@oneof
@attr.s(auto_attribs=True)
class HeadsConfig:
    single_instance: Optional[SingleInstanceConfmapsHeadConfig] = None
    centroid: Optional[CentroidsHeadConfig] = None
    centered_instance: Optional[CenteredInstanceConfmapsHeadConfig] = None
    multi_instance: Optional[MultiInstanceConfig] = None


@attr.s(auto_attribs=True)
class LEAPConfig:
    """LEAP backbone configuration.

    Attributes:
        max_stride: Determines the number of downsampling blocks in the network,
            increasing receptive field size at the cost of network size.
        output_stride: Determines the number of upsampling blocks in the network.
        filters: Base number of filters in the network.
        filters_rate: Factor to scale the number of filters by at each block.
        up_interpolate: If True, use bilinear upsampling instead of transposed
            convolutions for upsampling. This can save computations but may lower
            overall accuracy.
        stacks: Number of repeated stacks of the network (excluding the stem).
    """

    max_stride: int = 8  # determines down blocks
    output_stride: int = 1  # determines up blocks
    filters: int = 64
    filters_rate: float = 2
    up_interpolate: bool = False
    stacks: int = 1


@attr.s(auto_attribs=True)
class UNetConfig:
    """UNet backbone configuration.

    Attributes:
        stem_stride: If not None, controls how many stem blocks to use for initial
            downsampling. These are useful for learned downsampling that is able to
            retain spatial information while reducing large input image sizes.
        max_stride: Determines the number of downsampling blocks in the network,
            increasing receptive field size at the cost of network size.
        output_stride: Determines the number of upsampling blocks in the network.
        filters: Base number of filters in the network.
        filters_rate: Factor to scale the number of filters by at each block.
        middle_block: If True, add an intermediate block between the downsampling and
            upsampling branch for additional processing for features at the largest
            receptive field size.
        up_interpolate: If True, use bilinear upsampling instead of transposed
            convolutions for upsampling. This can save computations but may lower
            overall accuracy.
        stacks: Number of repeated stacks of the network (excluding the stem).
    """

    stem_stride: Optional[int] = None
    max_stride: int = 16
    output_stride: int = 1
    filters: int = 64
    filters_rate: float = 2
    middle_block: bool = True
    up_interpolate: bool = False
    stacks: int = 1


@attr.s(auto_attribs=True)
class HourglassConfig:
    """Hourglass backbone configuration.

    Attributes:
        stem_stride: Controls how many stem blocks to use for initial downsampling.
            These are useful for learned downsampling that is able to retain spatial
            information while reducing large input image sizes.
        max_stride: Determines the number of downsampling blocks in the network,
            increasing receptive field size at the cost of network size.
        output_stride: Determines the number of upsampling blocks in the network.
        filters: Base number of filters in the network.
        filters_increase: Constant to increase the number of filters by at each block.
        stacks: Number of repeated stacks of the network (excluding the stem).
    """

    stem_stride: int = 4
    max_stride: int = 64
    output_stride: int = 4
    stem_filters: int = 128
    filters: int = 256
    filter_increase: int = 128
    stacks: int = 3


@attr.s(auto_attribs=True)
class UpsamplingConfig:
    """Upsampling stack configuration.

    Attributes:
        method: If "transposed_conv", use a strided transposed convolution to perform
            learnable upsampling. If "interpolation", bilinear upsampling will be used
            instead.
        skip_connections: If "add", incoming feature tensors form skip connection with
            upsampled features via element-wise addition. Height/width are matched via
            stride and a 1x1 linear conv is applied if the channel counts do no match
            up. If "concatenate", the skip connection is formed via channel-wise
            concatenation. If None, skip connections will not be formed.
        block_stride: The striding of the upsampling *layer* (not tensor). This is
            typically set to 2, such that the tensor doubles in size with each
            upsampling step, but can be set higher to upsample to the desired
            `output_stride` directly in fewer steps.
        filters: Integer that specifies the base number of filters in each convolution
            layer. This will be scaled by the `filters_rate` at every upsampling step.
        filters_rate: Factor to scale the number of filters in the convolution layers
            after each upsampling step. If set to 1, the number of filters won't change.
        refine_convs: If greater than 0, specifies the number of 3x3 convolutions that
            will be applied after the upsampling step for refinement. These layers can
            serve the purpose of "mixing" the skip connection fused features, or to
            refine the current feature map after upsampling, which can help to prevent
            aliasing and checkerboard effects. If 0, no additional convolutions will be
            applied.
        conv_batchnorm: Specifies whether batch norm should be applied after each
            convolution (and before the ReLU activation).
        transposed_conv_kernel_size: Size of the kernel for the transposed convolution.
            No effect if bilinear upsampling is used.
    """

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


@oneof
@attr.s(auto_attribs=True)
class BackboneConfig:
    leap: Optional[LEAPConfig] = None
    unet: Optional[UNetConfig] = None
    hourglass: Optional[HourglassConfig] = None
    resnet: Optional[ResNetConfig] = None


@attr.s(auto_attribs=True)
class ModelConfig:
    backbone: BackboneConfig = attr.ib(factory=BackboneConfig)
    heads: HeadsConfig = attr.ib(factory=HeadsConfig)

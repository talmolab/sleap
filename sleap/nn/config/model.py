import attr
from typing import Optional, Text, List, Sequence, Tuple
from sleap.nn.config.utils import oneof


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsHeadConfig:
    """Configurations for single instance confidence map heads.

    These heads are used in single instance models that make the assumption that only
    one of each body part is present in the image. These heads produce confidence maps
    with a single peak for each part type which can be detected via global peak finding.

    Do not use this head if there is more than one animal present in the image.

    Attributes:
        part_names: Text name of the body parts (nodes) that the head will be configured
            to produce. The number of parts determines the number of channels in the
            output. If not specified, all body parts in the skeleton will be used.
        sigma: Spread of the Gaussian distribution of the confidence maps as a scalar
            float. Smaller values are more precise but may be difficult to learn as they
            have a lower density within the image space. Larger values are easier to
            learn but are less precise with respect to the peak coordinate. This spread
            is in units of pixels of the model input image, i.e., the image resolution
            after any input scaling is applied.
        output_stride: The stride of the output confidence maps relative to the input
            image. This is the reciprocal of the resolution, e.g., an output stride of 2
            results in confidence maps that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
        offset_refinement: If `True`, model will also output an offset refinement map
            used to achieve subpixel localization of peaks during inference. This can
            improve the localization accuracy of the model at the cost of additional
            memory and training and inference time. If `False` (the default), subpixel
            localization can be achieved post-hoc with deterministic refinement, which
            does not require additional resources or training, but may not achieve the
            same accuracy as learned refinement.
    """

    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0
    offset_refinement: bool = False


@attr.s(auto_attribs=True)
class CentroidsHeadConfig:
    """Configurations for centroid confidence map heads.

    These heads are used in topdown models that rely on centroid detection to detect
    instances for cropping before predicting the remaining body parts.

    Multiple centroids can be present (one per instance), so their coordinates can be
    recovered in inference via local peak finding.

    Attributes:
        anchor_part: Text name of a body part (node) to use as the anchor point. If
            None, the midpoint of the bounding box of all visible instance points will
            be used as the anchor. The bounding box midpoint will also be used if the
            anchor part is specified but not visible in the instance. Setting a reliable
            anchor point can significantly improve topdown model accuracy as they
            benefit from a consistent geometry of the body parts relative to the center
            of the image.
        sigma: Spread of the Gaussian distribution of the confidence maps as a scalar
            float. Smaller values are more precise but may be difficult to learn as they
            have a lower density within the image space. Larger values are easier to
            learn but are less precise with respect to the peak coordinate. This spread
            is in units of pixels of the model input image, i.e., the image resolution
            after any input scaling is applied.
        output_stride: The stride of the output confidence maps relative to the input
            image. This is the reciprocal of the resolution, e.g., an output stride of 2
            results in confidence maps that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
        offset_refinement: If `True`, model will also output an offset refinement map
            used to achieve subpixel localization of peaks during inference. This can
            improve the localization accuracy of the model at the cost of additional
            memory and training and inference time. If `False` (the default), subpixel
            localization can be achieved post-hoc with deterministic refinement, which
            does not require additional resources or training, but may not achieve the
            same accuracy as learned refinement.
    """

    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0
    offset_refinement: bool = False


@attr.s(auto_attribs=True)
class CenteredInstanceConfmapsHeadConfig:
    """Configurations for centered instance confidence map heads.

    These heads are used in topdown multi-instance models that make the assumption that
    there is an instance reliably centered in the cropped input image. These heads are
    useful when centroids are easy to detect as they learn complex relationships between
    the geometry of body parts, even when animals are occluded.

    This comes at the cost of a strong reliance on the accuracy of the instance-centered
    cropping, i.e., it is heavily limited by the accuracy of the centroid model.

    Additionally, since one image crop is evaluated per instance, topdown models scale
    linearly with the number of animals in the frame, which can result in poor
    performance when many instances are present.

    Use this head when centroids are easy to detect, preferably using a consistent body
    part as an anchor, and when there are few animals that cover a small region of the
    full frame.

    Attributes:
        anchor_part: Text name of a body part (node) to use as the anchor point. If
            None, the midpoint of the bounding box of all visible instance points will
            be used as the anchor. The bounding box midpoint will also be used if the
            anchor part is specified but not visible in the instance. Setting a reliable
            anchor point can significantly improve topdown model accuracy as they
            benefit from a consistent geometry of the body parts relative to the center
            of the image.
        part_names: Text name of the body parts (nodes) that the head will be configured
            to produce. The number of parts determines the number of channels in the
            output. If not specified, all body parts in the skeleton will be used.
        sigma: Spread of the Gaussian distribution of the confidence maps as a scalar
            float. Smaller values are more precise but may be difficult to learn as they
            have a lower density within the image space. Larger values are easier to
            learn but are less precise with respect to the peak coordinate. This spread
            is in units of pixels of the model input image, i.e., the image resolution
            after any input scaling is applied.
        output_stride: The stride of the output confidence maps relative to the input
            image. This is the reciprocal of the resolution, e.g., an output stride of 2
            results in confidence maps that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
        offset_refinement: If `True`, model will also output an offset refinement map
            used to achieve subpixel localization of peaks during inference. This can
            improve the localization accuracy of the model at the cost of additional
            memory and training and inference time. If `False` (the default), subpixel
            localization can be achieved post-hoc with deterministic refinement, which
            does not require additional resources or training, but may not achieve the
            same accuracy as learned refinement.
    """

    anchor_part: Optional[Text] = None
    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0
    offset_refinement: bool = False


@attr.s(auto_attribs=True)
class MultiInstanceConfmapsHeadConfig:
    """Configurations for multi-instance confidence map heads.

    These heads are used in bottom-up multi-instance models that do not make any
    assumption about the connectivity of the body parts. These heads will generate
    multiple local peaks for each body part type and must be detected using local peak
    finding.

    Although this head alone is sufficient to detect multiple copies of each body part
    type, it provides no information as to which sets of points should be grouped
    together to the same instance. If this is required, a head that provides
    connectivity or grouping information is required, e.g., part affinity fields.

    Use this head when multiple instances of each body part are present and do not need
    to be grouped or will be grouped using additional information.

    This head type has the advantage that it only needs to evaluate each frame once to
    find all peaks, in contrast to topdown models that must be evaluated for each crop.
    This constant scaling with the number of instances can be especially beneficial when
    there are many animals present in the frame.

    Attributes:
        part_names: Text name of the body parts (nodes) that the head will be configured
            to produce. The number of parts determines the number of channels in the
            output. If not specified, all body parts in the skeleton will be used.
        sigma: Spread of the Gaussian distribution of the confidence maps as a scalar
            float. Smaller values are more precise but may be difficult to learn as they
            have a lower density within the image space. Larger values are easier to
            learn but are less precise with respect to the peak coordinate. This spread
            is in units of pixels of the model input image, i.e., the image resolution
            after any input scaling is applied.
        output_stride: The stride of the output confidence maps relative to the input
            image. This is the reciprocal of the resolution, e.g., an output stride of 2
            results in confidence maps that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
        offset_refinement: If `True`, model will also output an offset refinement map
            used to achieve subpixel localization of peaks during inference. This can
            improve the localization accuracy of the model at the cost of additional
            memory and training and inference time. If `False` (the default), subpixel
            localization can be achieved post-hoc with deterministic refinement, which
            does not require additional resources or training, but may not achieve the
            same accuracy as learned refinement.
    """

    part_names: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0
    offset_refinement: bool = False


@attr.s(auto_attribs=True)
class PartAffinityFieldsHeadConfig:
    """Configurations for multi-instance part affinity field heads.

    These heads are used in bottom-up multi-instance models that require information
    about body part connectivity in order to group multiple detections of each body part
    type into distinct instances.

    Part affinity fields are an image-space representation of the directed graph that
    defines the skeleton. Pixels that are close to the line (directed edge) formed
    between pairs of nodes of the same instance will contain unit vectors pointing along
    the direction of the the connection. The similarity between this line and the
    average of the unit vectors at the pixels underneath the line can be used as a
    matching score to associate candidate pairs of body part detections.

    Use this head when multiple instances of each body part are present and need to be
    grouped to coherent instances.

    This head type has the advantage that it only needs to evaluate each frame once to
    find all peaks, in contrast to topdown models that must be evaluated for each crop.
    This constant scaling with the number of instances can be especially beneficial when
    there are many animals present in the frame.

    Attributes:
        edges: List of 2-tuples of the form `(source_node, destination_node)` that
            define pairs of text names of the directed edges of the graph. If not set,
            all edges in the skeleton will be used.
        sigma: Spread of the Gaussian distribution that weigh the part affinity fields
            as a function of their distance from the edge they represent. Smaller values
            are more precise but may be difficult to learn as they have a lower density
            within the image space. Larger values are easier to learn but are less
            precise with respect to the edge distance, so can be less useful in
            disambiguating between edges that are nearby and parallel in direction. This
            spread is in units of pixels of the model input image, i.e., the image
            resolution after any input scaling is applied.
        output_stride: The stride of the output part affinity fields relative to the
            input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in PAFs that are 0.5x the size of the input. Increasing
            this value can considerably speed up model performance and decrease memory
            requirements, at the cost of decreased spatial resolution.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
    """

    edges: Optional[Sequence[Tuple[Text, Text]]] = None
    sigma: float = 15.0
    output_stride: int = 1
    loss_weight: float = 1.0


@attr.s(auto_attribs=True)
class MultiInstanceConfig:
    """Configuration for combined multi-instance confidence map and PAF model heads.

    This configuration specifies a multi-head model that outputs both multi-instance
    confidence maps and part affinity fields, which together enable multi-instance pose
    estimation in a bottom-up fashion, i.e., no instance cropping or centroids are
    required.

    Attributes:
        confmaps: Part confidence map configuration (see the description in
            `MultiInstanceConfmapsHeadConfig`).
        pafs: Part affinity fields configuration (see the description in
            `PartAffinityFieldsHeadConfig`).
    """

    confmaps: MultiInstanceConfmapsHeadConfig = attr.ib(
        factory=MultiInstanceConfmapsHeadConfig
    )
    pafs: PartAffinityFieldsHeadConfig = attr.ib(factory=PartAffinityFieldsHeadConfig)


@attr.s(auto_attribs=True)
class ClassMapsHeadConfig:
    """Configurations for class map heads.

    These heads are used in bottom-up multi-instance models that classify detected
    points using a fixed set of learned classes (e.g., animal identities).

    Class maps are an image-space representation of the probability of that each class
    occupies a given pixel. This is similar to semantic segmentation, however only the
    pixels in the neighborhood of the landmarks have a class assignment.

    Attributes:
        classes: List of string names of the classes that this head will predict.
        sigma: Spread of the Gaussian distribution that determines the neighborhood
            that the class maps will be nonzero around each landmark.
        output_stride: The stride of the output class maps relative to the input image.
            This is the reciprocal of the resolution, e.g., an output stride of 2
            results in maps that are 0.5x the size of the input. This should be the same
            size as the confidence maps they are associated with.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
    """

    classes: Optional[List[Text]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0


@attr.s(auto_attribs=True)
class MultiClassBottomUpConfig:
    """Configuration for multi-instance confidence map and class map models.

    This configuration specifies a multi-head model that outputs both multi-instance
    confidence maps and class maps, which together enable multi-instance pose tracking
    in a bottom-up fashion, i.e., no instance cropping, centroids or PAFs are required.
    The limitation with this approach is that the classes, e.g., animal identities, must
    be labeled in the training data and cannot be generalized beyond those classes. This
    is still useful for applications in which the animals are uniquely identifiable and
    tracking their identities at inference time is critical, e.g., for closed loop
    experiments.

    Attributes:
        confmaps: Part confidence map configuration (see the description in
            `MultiInstanceConfmapsHeadConfig`).
        class_maps: Class map configuration (see the description in
            `ClassMapsHeadConfig`).
    """

    confmaps: MultiInstanceConfmapsHeadConfig = attr.ib(
        factory=MultiInstanceConfmapsHeadConfig
    )
    class_maps: ClassMapsHeadConfig = attr.ib(factory=ClassMapsHeadConfig)


@attr.s(auto_attribs=True)
class ClassVectorsHeadConfig:
    """Configurations for class vectors heads.

    These heads are used in top-down multi-instance models that classify detected
    points using a fixed set of learned classes (e.g., animal identities).

    Class vectors represent the probability that the image is associated with each of
    the specified classes. This is similar to a standard classification task.

    Attributes:
        classes: List of string names of the classes that this head will predict.
        num_fc_layers: Number of fully-connected layers before the classification output
            layer. These can help in transforming general image features into
            classification-specific features.
        num_fc_units: Number of units (dimensions) in the fully-connected layers before
            classification. Increasing this can improve the representational capacity in
            the pre-classification layers.
        output_stride: The stride of the output class maps relative to the input image.
            This is the reciprocal of the resolution, e.g., an output stride of 2
            results in maps that are 0.5x the size of the input. This should be the same
            size as the confidence maps they are associated with.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
    """

    classes: Optional[List[Text]] = None
    num_fc_layers: int = 1
    num_fc_units: int = 64
    global_pool: bool = True
    output_stride: int = 1
    loss_weight: float = 1.0


@attr.s(auto_attribs=True)
class MultiClassTopDownConfig:
    """Configuration for centered-instance confidence map and class map models.

    This configuration specifies a multi-head model that outputs both centered-instance
    confidence maps and class vectors, which together enable multi-instance pose
    tracking in a top-down fashion, i.e., instance-centered crops followed by pose
    estimation and classification.

    The limitation with this approach is that the classes, e.g., animal identities, must
    be labeled in the training data and cannot be generalized beyond those classes. This
    is still useful for applications in which the animals are uniquely identifiable and
    tracking their identities at inference time is critical, e.g., for closed loop
    experiments.

    Attributes:
        confmaps: Part confidence map configuration (see the description in
            `CenteredInstanceConfmapsHeadConfig`).
        class_vectors: Class map configuration (see the description in
            `ClassVectorsHeadConfig`).
    """

    confmaps: CenteredInstanceConfmapsHeadConfig = attr.ib(
        factory=CenteredInstanceConfmapsHeadConfig
    )
    class_vectors: ClassVectorsHeadConfig = attr.ib(factory=ClassVectorsHeadConfig)


@oneof
@attr.s(auto_attribs=True)
class HeadsConfig:
    """Configurations related to the model output head type.

    Only one attribute of this class can be set, which defines the model output type.

    Attributes:
        single_instance: An instance of `SingleInstanceConfmapsHeadConfig`.
        centroid: An instance of `CentroidsHeadConfig`.
        centered_instance: An instance of `CenteredInstanceConfmapsHeadConfig`.
        multi_instance: An instance of `MultiInstanceConfig`.
        multi_class_bottomup: An instance of `MultiClassBottomUpConfig`.
        multi_class_topdown: An instance of `MultiClassTopDownConfig`.
    """

    single_instance: Optional[SingleInstanceConfmapsHeadConfig] = None
    centroid: Optional[CentroidsHeadConfig] = None
    centered_instance: Optional[CenteredInstanceConfmapsHeadConfig] = None
    multi_instance: Optional[MultiInstanceConfig] = None
    multi_class_bottomup: Optional[MultiClassBottomUpConfig] = None
    multi_class_topdown: Optional[MultiClassTopDownConfig] = None


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
            receptive field size. This will not introduce an extra pooling step.
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
    """ResNet backbone configuration.

    Attributes:
        version: Name of the ResNetV1 variant. Can be one of: "ResNet50", "ResNet101",
            or "ResNet152".
        weights: Controls how the network weights are initialized. If "random", the
            network is not pretrained. If "frozen", the network uses pretrained weights
            and keeps them fixed. If "tunable", the network uses pretrained weights and
            allows them to be trainable.
        upsampling: A `UpsamplingConfig` that defines an upsampling branch if not None.
        max_stride: Stride of the backbone feature activations. These should be <= 32.
        output_stride: Stride of the final output. If the upsampling branch is not
            defined, the output stride is controlled via dilated convolutions or reduced
            pooling in the backbone.
    """

    version: Text = attr.ib(
        default="ResNet50",
        validator=attr.validators.in_(["ResNet50", "ResNet101", "ResNet152"]),
    )
    weights: Text = attr.ib(
        default="frozen", validator=attr.validators.in_(["random", "frozen", "tunable"])
    )
    upsampling: Optional[UpsamplingConfig] = None
    max_stride: int = 32
    output_stride: int = 4


@attr.s(auto_attribs=True)
class PretrainedEncoderConfig:
    """Configuration for UNet backbone with pretrained encoder.

    Attributes:
        encoder: Name of the network architecture to use as the encoder. Valid encoder
            names are:
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
            Defaults to `"efficientnetb0"`.
        pretrained: If `True`, use initialized with weights pretrained on ImageNet.
        decoder_filters: Base number of filters for the upsampling blocks in the
            decoder.
        decoder_filters_rate: Factor to scale the number of filters by at each
            consecutive upsampling block in the decoder.
        output_stride: Stride of the final output.
        decoder_batchnorm: If `True` (the default), use batch normalization in the
            decoder layers.
    """

    encoder: Text = attr.ib(default="efficientnetb0")
    pretrained: bool = True
    decoder_filters: int = 256
    decoder_filters_rate: float = 1.0
    output_stride: int = 2
    decoder_batchnorm: bool = True


@oneof
@attr.s(auto_attribs=True)
class BackboneConfig:
    """Configurations related to the model backbone.

    Only one field can be set and will determine which backbone architecture to use.

    Attributes:
        leap: A `LEAPConfig` instance.
        unet: A `UNetConfig` instance.
        hourglass: A `HourglassConfig` instance.
        resnet: A `ResNetConfig` instance.
    """

    leap: Optional[LEAPConfig] = None
    unet: Optional[UNetConfig] = None
    hourglass: Optional[HourglassConfig] = None
    resnet: Optional[ResNetConfig] = None
    pretrained_encoder: Optional[PretrainedEncoderConfig] = None


@attr.s(auto_attribs=True)
class ModelConfig:
    """Configurations related to model architecture.

    Attributes:
        backbone: Configurations related to the main network architecture.
        heads: Configurations related to the output heads.
    """

    backbone: BackboneConfig = attr.ib(factory=BackboneConfig)
    heads: HeadsConfig = attr.ib(factory=HeadsConfig)

"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `tf.keras.Model` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""
import tensorflow as tf

import attr
from typing import List, TypeVar, Optional, Text, Tuple

import sleap
from sleap.nn.architectures import (
    LeapCNN,
    UNet,
    Hourglass,
    ResNetv1,
    ResNet50,
    ResNet101,
    ResNet152,
    UnetPretrainedEncoder,
    IntermediateFeature,
)
from sleap.nn.heads import (
    Head,
    CentroidConfmapsHead,
    SingleInstanceConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    ClassVectorsHead,
    OffsetRefinementHead,
)
from sleap.nn.config import (
    LEAPConfig,
    UNetConfig,
    HourglassConfig,
    ResNetConfig,
    PretrainedEncoderConfig,
    SingleInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
    MultiClassBottomUpConfig,
    MultiClassTopDownConfig,
    BackboneConfig,
    HeadsConfig,
    ModelConfig,
)
from sleap.nn.data.utils import ensure_list


ARCHITECTURES = [
    LeapCNN,
    UNet,
    Hourglass,
    ResNetv1,
    ResNet50,
    ResNet101,
    ResNet152,
    UnetPretrainedEncoder,
]
ARCHITECTURE_NAMES = [cls.__name__ for cls in ARCHITECTURES]
Architecture = TypeVar("Architecture", *ARCHITECTURES)

BACKBONE_CONFIG_TO_CLS = {
    LEAPConfig: LeapCNN,
    UNetConfig: UNet,
    HourglassConfig: Hourglass,
    ResNetConfig: ResNetv1,
    PretrainedEncoderConfig: UnetPretrainedEncoder,
}

HEADS = [
    Head,
    CentroidConfmapsHead,
    SingleInstanceConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    ClassVectorsHead,
    OffsetRefinementHead,
]
Head = TypeVar("Head", *HEADS)


@attr.s(auto_attribs=True)
class Model:
    """SLEAP model that describes an architecture and output types.

    Attributes:
        backbone: An `Architecture` class that provides methods for building a
            tf.keras.Model given an input.
        heads: List of `Head`s that define the outputs of the network.
        keras_model: The current `tf.keras.Model` instance if one has been created.
    """

    backbone: Architecture
    heads: List[Head] = attr.ib(converter=ensure_list)
    keras_model: Optional[tf.keras.Model] = None

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        skeleton: Optional[sleap.Skeleton] = None,
        tracks: Optional[List[sleap.Track]] = None,
        update_config: bool = False,
    ) -> "Model":
        """Create a SLEAP model from configurations.

        Arguments:
            config: The configurations as a `ModelConfig` instance.
            skeleton: A `sleap.Skeleton` to use if not provided in the config.
            update_config: If `True`, the input model configuration will be updated with
                values inferred from other fields.

        Returns:
            An instance of `Model` built with the specified configurations.
        """
        # Figure out which backbone class to use.
        backbone_config = config.backbone.which_oneof()
        backbone_cls = BACKBONE_CONFIG_TO_CLS.get(type(backbone_config), None)
        if backbone_cls is None:
            raise ValueError(
                "Backbone architecture (config.model.backbone) was not specified."
            )

        # Figure out which head class to use.
        head_config = config.heads.which_oneof()
        if isinstance(head_config, SingleInstanceConfmapsHeadConfig):
            part_names = head_config.part_names
            if part_names is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                part_names = skeleton.node_names
                if update_config:
                    head_config.part_names = part_names
            heads = [
                SingleInstanceConfmapsHead.from_config(
                    head_config, part_names=part_names
                )
            ]
            output_stride = heads[0].output_stride
            if head_config.offset_refinement:
                heads.append(
                    OffsetRefinementHead.from_config(head_config, part_names=part_names)
                )

        elif isinstance(head_config, CentroidsHeadConfig):
            heads = [CentroidConfmapsHead.from_config(head_config)]
            output_stride = heads[0].output_stride
            if head_config.offset_refinement:
                heads.append(OffsetRefinementHead.from_config(head_config))

        elif isinstance(head_config, CenteredInstanceConfmapsHeadConfig):
            part_names = head_config.part_names
            if part_names is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                part_names = skeleton.node_names
                if update_config:
                    head_config.part_names = part_names
            heads = [
                CenteredInstanceConfmapsHead.from_config(
                    head_config, part_names=part_names
                )
            ]
            output_stride = heads[0].output_stride
            if head_config.offset_refinement:
                heads.append(
                    OffsetRefinementHead.from_config(head_config, part_names=part_names)
                )

        elif isinstance(head_config, MultiInstanceConfig):
            part_names = head_config.confmaps.part_names
            if part_names is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                part_names = skeleton.node_names
                if update_config:
                    head_config.confmaps.part_names = part_names

            edges = head_config.pafs.edges
            if edges is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                edges = skeleton.edge_names
                if update_config:
                    head_config.pafs.edges = edges

            heads = [
                MultiInstanceConfmapsHead.from_config(
                    head_config.confmaps, part_names=part_names
                ),
                PartAffinityFieldsHead.from_config(head_config.pafs, edges=edges),
            ]
            output_stride = min(heads[0].output_stride, heads[1].output_stride)
            output_stride = heads[0].output_stride
            if head_config.confmaps.offset_refinement:
                heads.append(
                    OffsetRefinementHead.from_config(
                        head_config.confmaps, part_names=part_names
                    )
                )

        elif isinstance(head_config, MultiClassBottomUpConfig):
            part_names = head_config.confmaps.part_names
            if part_names is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                part_names = skeleton.node_names
                if update_config:
                    head_config.confmaps.part_names = part_names

            classes = head_config.class_maps.classes
            if classes is None:
                if tracks is None:
                    raise ValueError(
                        "Classes must be provided when the head configuration is "
                        "incomplete."
                    )
                classes = [t.name for t in tracks]
            if update_config:
                head_config.class_maps.classes = classes

            heads = [
                MultiInstanceConfmapsHead.from_config(
                    head_config.confmaps, part_names=part_names
                ),
                ClassMapsHead.from_config(head_config.class_maps, classes=classes),
            ]
            output_stride = min(heads[0].output_stride, heads[1].output_stride)
            output_stride = heads[0].output_stride
            if head_config.confmaps.offset_refinement:
                heads.append(
                    OffsetRefinementHead.from_config(
                        head_config.confmaps, part_names=part_names
                    )
                )

        elif isinstance(head_config, MultiClassTopDownConfig):
            part_names = head_config.confmaps.part_names
            if part_names is None:
                if skeleton is None:
                    raise ValueError(
                        "Skeleton must be provided when the head configuration is "
                        "incomplete."
                    )
                part_names = skeleton.node_names
                if update_config:
                    head_config.confmaps.part_names = part_names

            classes = head_config.class_vectors.classes
            if classes is None:
                if tracks is None:
                    raise ValueError(
                        "Classes must be provided when the head configuration is "
                        "incomplete."
                    )
                classes = [t.name for t in tracks]
            if update_config:
                head_config.class_vectors.classes = classes

            heads = [
                CenteredInstanceConfmapsHead.from_config(
                    head_config.confmaps, part_names=part_names
                ),
                ClassVectorsHead.from_config(
                    head_config.class_vectors, classes=classes
                ),
            ]
            output_stride = min(heads[0].output_stride, heads[1].output_stride)
            output_stride = heads[0].output_stride
            if head_config.confmaps.offset_refinement:
                heads.append(
                    OffsetRefinementHead.from_config(
                        head_config.confmaps, part_names=part_names
                    )
                )
        else:
            raise ValueError(
                "Head configuration (config.model.heads) was not specified."
            )

        backbone_config.output_stride = output_stride

        return cls(backbone=backbone_cls.from_config(backbone_config), heads=heads)

    @property
    def maximum_stride(self) -> int:
        """Return the maximum stride of the model backbone."""
        return self.backbone.maximum_stride

    def make_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create a trainable model by connecting the backbone with the heads.

        Args:
            input_shape: Tuple of (height, width, channels) specifying the shape of the
                inputs before preprocessing.

        Returns:
            An instantiated `tf.keras.Model`.
        """
        # Create input layer.
        x_in = tf.keras.layers.Input(input_shape, name="input")

        # Create backbone.
        x_main, x_mid = self.backbone.make_backbone(x_in=x_in)

        # Make sure main and intermediate feature outputs are lists.
        if type(x_main) != list:
            x_main = [x_main]
        if len(x_mid) > 0 and isinstance(x_mid[0], IntermediateFeature):
            x_mid = [x_mid]

        # Build output layers for each head.
        x_outs = []
        for output in self.heads:
            x_head = []
            if output.output_stride == self.backbone.output_stride:
                # The main output has the same stride as the head, so build output layer
                # from that tensor.
                for i, x in enumerate(x_main):
                    x_head.append(output.make_head(x))

            else:
                # Look for an intermediate activation that has the correct stride.
                for feats in zip(*x_mid):
                    # TODO: Test for this assumption?
                    assert all([feat.stride == feats[0].stride for feat in feats])
                    if feats[0].stride == output.output_stride:
                        for i, feat in enumerate(feats):
                            x_head.append(output.make_head(feat.tensor))
                        break

            if len(x_head) == 0:
                raise ValueError(
                    f"Could not find a feature activation for output at stride "
                    f"{output.output_stride}."
                )
            x_outs.extend(x_head)
        # TODO: Warn/error if x_main was not connected to any heads?

        # Create model.
        self.keras_model = tf.keras.Model(inputs=x_in, outputs=x_outs)
        return self.keras_model

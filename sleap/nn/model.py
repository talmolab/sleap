"""SLEAP model configuration and initialization.

This module provides classes used in model configuration, creation and loading.
"""
import tensorflow as tf

import attr
import cattr
import json
from jsmin import jsmin
from typing import TypeVar, Text, Any, Dict, Sequence, List, Optional, Tuple, Union

import sleap
from sleap.nn.architectures.common import IntermediateFeature
from sleap.nn.architectures.leap import LeapCNN
from sleap.nn.architectures.unet import Unet
from sleap.nn.architectures.hourglass import Hourglass
from sleap.nn.architectures.resnet import ResNet50, ResNet101, ResNet152
from sleap.nn import heads


# TODO: Define interface required for architectures (maximum_stride, output_stride,
# make_backbone).
ARCHITECTURES = [LeapCNN, Unet, Hourglass, ResNet50, ResNet101, ResNet152]
ARCHITECTURE_NAMES = [cls.__name__ for cls in ARCHITECTURES]
Architecture = TypeVar("Architecture", *ARCHITECTURES)



@attr.s(auto_attribs=True)
class ModelConfig:
    """SLEAP model configuration.
    
    Attributes:
        architecture: Name of the backbone architecture.
        backbone: Architecture instance specifying the backbone network architecture.
        outputs: List of `sleap.nn.heads.OutputHead` specifying the configuration of all
            of the outputs of the network.
    """

    architecture: Text = attr.ib(validator=attr.validators.in_(ARCHITECTURE_NAMES))
    backbone: Architecture = attr.ib()
    outputs: Sequence[heads.OutputHead]

    @backbone.validator
    def _check_backbone(self, attribute, value):
        backbone_class_name = type(value).__name__
        if backbone_class_name != self.architecture:
            raise ValueError(
                f"Backbone ({backbone_class_name}) and architecture name "
                f"({self.architecture}) must be the same."
            )

    @classmethod
    def from_config(
        cls,
        backbone: Architecture,
        outputs: Sequence[heads.OutputHead],
    ) -> "ModelConfig":
        """Create a model config from component configurations.

        This method is a convenient way to initialize this clsas without having to
        specify the `architecture` class name as a string as well.

        Args:
            backbone: Architecture instance specifying the backbone network
                architecture.
            outputs: List of `sleap.nn.heads.OutputHead` specifying the configuration of
                all of the outputs of the network.

        Returns:
            The initialized `ModelConfig` instance.
        """
        return cls(
            architecture=type(backbone).__name__,
            backbone=backbone,
            outputs=outputs,
        )

    @classmethod
    def from_cattr(cls, data_dicts: Dict[Text, Any]) -> "ModelConfig":
        """Structure a backbone from decoded JSON dictionaries."""
        for arch_cls in ARCHITECTURES:
            if arch_cls.__name__ == data_dicts["architecture"]:
                backbone_dict = data_dicts.get("backbone", {})
                outputs = [
                    heads.OutputHead.from_cattr(data_dict)
                    for data_dict in data_dicts["outputs"]
                ]
                return cls(
                    architecture=data_dicts["architecture"],
                    backbone=arch_cls(**backbone_dict),
                    outputs=outputs,
                )

        raise ValueError(
            "Could not find architecture with name: '%s'" % data_dicts["type"]
        )

    @classmethod
    def from_legacy_model_cattr(
        cls,
        data_dicts: Dict[Text, Any],
        scale: float = 1.0,
        sigma: float = 5.0,
        skeletons: Optional[Sequence[sleap.Skeleton]] = None,
        instance_crop_use_ctr_node: bool = False,
        instance_crop_ctr_node_ind: int = 0,
    ) -> "ModelConfig":
        """Structure a model from a legacy `Model` configuration format.

        This method is typically called by `from_legacy_job_cattr` which will
        automatically fill in the additional attributes from `TrainerConfig`.

        Args:
            data_dicts: Data decoded from JSON representation of a legacy `Model`.
            scale: Attribute from legacy `TrainerConfig`.
            sigma: Attribute from legacy `TrainerConfig`.
            skeletons: Attribute from legacy `Model`. If not provided, these will be
                pulled out of `data_dicts` if available. Use this to manually override
                the skeletons to use when structuring.
            instance_crop_use_ctr_node: Attribute from legacy `TrainerConfig`.
            instance_crop_ctr_node_ind: Attribute from legacy `TrainerConfig`.

        Returns:
            An initialized instance of `ModelConfig` with configuration inferred from
            the legacy format decoded JSON data.

            A best effort will be made to map the relevant fields to the new format, but
            some parameters may not have a 1-to-1 matching.
        """
        # Setup backbone.
        arch_dict = data_dicts["backbone"]
        if data_dicts["backbone_name"] == "UNet":
            architecture = "Unet"
            backbone = Unet(
                stacks=1,
                filters=arch_dict.get("num_filters", 16),
                filters_rate=2,
                kernel_size=arch_dict.get("kernel_size", 5),
                convs_per_block=arch_dict.get("convs_per_depth", 2),
                stem_blocks=0,
                down_blocks=arch_dict.get("down_blocks", 3),
                middle_block=True,  # TODO: Check this for correctness in legacy code.
                up_blocks=arch_dict.get("up_blocks", 3),
                up_interpolate=arch_dict.get("upsampling_layers", True),
            )

        elif data_dicts["backbone_name"] == "LeapCNN":
            architecture = "LeapCNN"
            backbone = LeapCNN(
                stacks=1,
                filters=arch_dict.get("num_filters", 64),
                filters_rate=2,
                down_blocks=arch_dict.get("down_blocks", 3),
                up_blocks=arch_dict.get("up_blocks", 3),
                up_interpolate=arch_dict.get("upsampling_layers", True),
                up_convs_per_block=2,
            )

        else:
            raise ValueError(
                f"Legacy architecture not implemented: {data_dicts['backbone_name']}"
            )

        if skeletons is None:
            # Structure legacy skeleton format.
            skeleton_converter = sleap.Skeleton.make_cattr()

            # Structure skeletons.
            skeletons = skeleton_converter.structure(
                data_dicts.get("skeletons", []), List[sleap.Skeleton]
            )

        # Pull out first skeleton data if available.
        node_names = None
        edge_names = None
        anchor_part_name = None
        if len(skeletons) > 0:
            skeleton = skeletons[0]
            node_names = skeleton.node_names
            edge_names = skeleton.edge_names
            anchor_part_name = node_names[instance_crop_ctr_node_ind]

        # Infer output scale (single output head).
        output_scale = scale / float(backbone.output_stride)
        output_stride = int(1 / output_scale)

        # Setup outputs.
        if data_dicts["output_type"] == 0:
            # CONFIDENCE_MAP = 0
            outputs = [
                heads.OutputHead(
                    type="MultiPartConfmaps",
                    config=heads.MultiPartConfmaps(part_names=node_names, sigma=sigma),
                    stride=output_stride,
                )
            ]

        elif data_dicts["output_type"] == 1:
            # PART_AFFINITY_FIELD = 1
            outputs = [
                heads.OutputHead(
                    type="PartAffinityFields",
                    config=heads.PartAffinityFields(
                        edges=edge_names, max_distance=sigma
                    ),
                    stride=output_stride,
                )
            ]

        elif data_dicts["output_type"] == 2:
            # CENTROIDS = 2
            outputs = [
                heads.OutputHead(
                    type="CentroidConfmap",
                    config=heads.CentroidConfmap(
                        use_anchor_part=instance_crop_use_ctr_node,
                        anchor_part_name=anchor_part_name,
                        sigma=sigma,
                    ),
                    stride=output_stride,
                )
            ]

        elif data_dicts["output_type"] == 3:
            # TOPDOWN_CONFIDENCE_MAP = 3
            outputs = [
                heads.OutputHead(
                    type="SinglePartConfmaps",
                    config=heads.SinglePartConfmaps(
                        part_names=node_names,
                        sigma=sigma,
                        centered=True,
                        center_on_anchor_part=instance_crop_use_ctr_node,
                        anchor_part_name=anchor_part_name,
                    ),
                    stride=output_stride,
                )
            ]

        else:
            raise ValueError(
                f"Unrecognized legacy output type: {data_dicts['output_type']}"
            )

        return cls(
            architecture=architecture,
            backbone=backbone,
            outputs=outputs,
        )

    @classmethod
    def from_legacy_job_cattr(
        cls,
        data_dicts: Dict[Text, Any],
        skeletons: Optional[Sequence[sleap.Skeleton]] = None,
    ) -> "ModelConfig":
        """Structure a model from a legacy `TrainingJob` configuration format.

        This will extract fields from decoded JSON data of a legacy `TrainingJob`. These
        contained two top-level keys: "model" (corresponding to an unstructured `Model`)
        and "trainer" (corresponding to an unstructured `TrainerConfig`).

        Args:
            data_dicts: Data decoded from JSON representation of a legacy `TrainingJob`.
            skeletons: Attribute from legacy `Model`. If not provided, these will be
                pulled out of `data_dicts` if available. Use this to manually override
                the skeletons to use when structuring.

        Returns:
            An initialized instance of `ModelConfig` with configuration inferred from
            the legacy format decoded JSON data.

            A best effort will be made to map the relevant fields to the new format, but
            some parameters may not have a 1-to-1 matching.

        See also: `ModelConfig.load_legacy_job`, `ModelConfig.from_legacy_model_cattr`
        """
        trainer_dict = data_dicts.get("trainer", {})
        return cls.from_legacy_model_cattr(
            data_dicts=data_dicts.get("model", {}),
            scale=trainer_dict.get("scale", 1.0),
            sigma=trainer_dict.get("sigma", 5.0),
            skeletons=skeletons,
            instance_crop_use_ctr_node=trainer_dict.get(
                "instance_crop_use_ctr_node", False
            ),
            instance_crop_ctr_node_ind=trainer_dict.get(
                "instance_crop_ctr_node_ind", 0
            ),
        )

    @classmethod
    def load_legacy_job(
        cls, filepath: Text, skeletons: Optional[Sequence[sleap.Skeleton]] = None
    ) -> "ModelConfig":
        """Load and structure a model from a legacy `TrainingJob` format JSON file.

        This will load and extract fields from decoded JSON data of a legacy
        `TrainingJob`. These contained two top-level keys: "model" (corresponding to an
        unstructured `Model`) and "trainer" (corresponding to an unstructured
        `TrainerConfig`).

        Args:
            filepath: Path to a JSON file containing representation of a legacy
                `TrainingJob`.
            skeletons: Attribute from legacy `Model`. If not provided, these will be
                pulled out of `data_dicts` if available. Use this to manually override
                the skeletons to use when structuring.

        Returns:
            An initialized instance of `ModelConfig` with configuration inferred from
            the legacy format decoded JSON data.

            A best effort will be made to map the relevant fields to the new format, but
            some parameters may not have a 1-to-1 matching.

        See also:
            `ModelConfig.from_legacy_job_cattr`, `ModelConfig.from_legacy_model_cattr`
        """
        data_dicts = json.loads(jsmin(read(open(filepath, "r"))))
        return cls.from_legacy_job_cattr(data_dicts, skeletons=skeletons)

    def make_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create a trainable model from the configuration.

        Args:
            input_shape: Tuple of (height, width, channels) specifying the shape of the
                inputs before preprocessing.
            
        Returns:
            An instantiated `tf.keras.Model`.
        """
        # Create input layer.
        x_in = tf.keras.layers.Input(input_shape, name="input")

        # TODO: Preprocessing

        # Create backbone.
        x_main, x_mid = self.backbone.make_backbone(x_in=x_in)

        # Make sure main and intermediate feature outputs are lists.
        if isinstance(x_main, tf.Tensor):
            x_main = [x_main]
        if isinstance(x_mid[0], IntermediateFeature):
            x_mid = [x_mid]

        # Build output layers for each head.
        x_outs = []
        for output in self.outputs:
            x_head = []
            if output.stride == self.backbone.output_stride:
                # The main output has the same stride as the head, so build output layer
                # from that tensor.
                for i, x in enumerate(x_main):
                    x_head.append(output.make_head(x, name=f"{output.type}_{i}"))

            else:
                # Look for an intermediate activation that has the correct stride.
                for feats in zip(*x_mid):
                    # TODO: Test for this assumption?
                    assert all([feat.stride == feats[0].stride for feat in feats])
                    if feats[0].stride == output.stride:
                        for i, feat in enumerate(feats):
                            x_head.append(
                                output.make_head(feat.tensor, name=f"{output.type}_{i}")
                            )
                        break

            if len(x_head) == 0:
                raise ValueError(
                    f"Could not find a feature activation for output at stride {output.stride}."
                )
            x_outs.append(x_head)

        # Create model.
        keras_model = tf.keras.Model(inputs=x_in, outputs=x_outs)
        return keras_model


# Register global cattr structuring hook.
cattr.register_structure_hook(ModelConfig, lambda d, t: ModelConfig.from_cattr(d))

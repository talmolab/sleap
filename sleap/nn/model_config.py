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


def scalar_to_2_tuple(x: Any) -> Tuple[Any, Any]:
    """Convert scalars to 2-tuples.

    Args:
        x: Any scalar or possible tuple or list.

    Returns:
        A tuple of length 2 if `x` is a scalar or a tuple/list of length 1.
        Returns `x` if it already is a 2-tuple/list.

    Raises:
        ValueError: If `x` is a tuple/list of length > 2.
    """
    if not isinstance(x, (tuple, list)):
        return (x, x)
    elif len(x) == 1:
        return (x[0], x[0])
    elif len(x) == 2:
        return x
    else:
        raise ValueError("Input is a tuple of length > 2.")


@attr.s(auto_attribs=True)
class PreprocessingConfig:
    """Parameters for preprocessing input data.
    
    This set of attributes describe the type of preprocessing that models expect to be
    applied to the data.
    
    Attributes:
        center_crop_instances: If True, instances within each image will be cropped and
            centered. Each cropped instance will be treated as an independent sample for
            training. For inference, setting this to True indicates that a centroid must
            be provided to the model to detection of each instance (e.g., for top-down
            models). Cropping is done before any other preprocessing (e.g., scaling). If
            False, the entire image is used.
        center_on_anchor_part: If True, specifies that centering should be done relative
            to a body part rather than the center of the instance bounding box. This has
            no effect if `center_crop_instances` is False.
        anchor_part_name: String specifying the body part name to use as an anchor for
            centering. If `center_crop_instances` or `center_on_anchor_part` are False,
            this has no effect and does not need to be specified.
        center_crop_bounding_box_size: Specifies the (height, width) raw image pixels of
            the bounding box to crop. Can also be specified as a scalar integer to crop
            a square. If `center_crop_instances` or `center_on_anchor_part` are False,
            this has no effect and does not need to be specified. If center cropping is
            enabled but this parameter is not specified (is set to `None`), the bounding
            box size will be inferred from the data if training, but is required for
            inference.
        bounding_box_padding: Specifies the padding in raw image pixels to apply if
            performing automatic calculation of `center_crop_bounding_box_size` during
            training.
        force_grayscale: If True, converts input images to grayscale if they are RGB via
            `tf.image.rgb_to_grayscale`.
        force_rgb: If True, converts input images to RGB if they are grayscale by tiling
            the channels.
        scale_inputs: If not None, inputs will be resized by a relative scale factor to
            this (scale_height, scale_width). Can also be specified as a scalar float to
            scale both dimensions by the same factor. This is not mutually exclusive
            `resize_to_shape`, but will be applied first. This is useful to adjust the
            input scale of the images regardless of the original image size, e.g., for
            adjusting the physical length scale of the features in variable size images.
        resize_to_shape: If not None, inputs will be resized to this (height, width).
            Can also be specified as a scalar integer to resize the input to a square.
            This is not mutually exclusive with `scale_inputs`, but will be applied
            afterwards. This is useful for rescaling large images to an explicit size
            rather by a fixed scale factor.
        normalize_from_range: If not None, specifies the (min_val, max_val) range of
            input data values to scale from. Data will first be cast to `float32` before
            scaling. If not specified, defaults to (0, 255) for uint8 data, or the
            limits of the data otherwise. If `normalize_to_range` is set to None, this
            has no effect.
        normalize_to_range: If not None, normalize the range of input values to this
            (min_val, max_val) after casting to float32. To specify the input range, set
            `normalize_from_range` explicitly.
        pad_to_stride_constraints: If True, padding will be applied to the bottom-right
            of the inputs after resizing in order to meet the shape constraints required
            by the backbone architecture. If False, shape constraints will be met via
            cropping from the bottom-right, which may result in data loss at the
            extremes of the image.
    """

    center_crop_instances: bool = False
    center_on_anchor_part: bool = False
    anchor_part_name: Optional[Text] = None
    center_crop_bounding_box_size: Optional[Union[int, Tuple[int, int]]] = attr.ib(
        default=None, converter=attr.converters.optional(scalar_to_2_tuple)
    )
    bounding_box_padding: int = 16
    force_grayscale: bool = False
    force_rgb: bool = False
    scale_inputs: Optional[Union[float, Tuple[float, float]]] = attr.ib(
        default=None, converter=attr.converters.optional(scalar_to_2_tuple)
    )
    resize_to_shape: Optional[Union[int, Tuple[int, int]]] = attr.ib(
        default=None, converter=attr.converters.optional(scalar_to_2_tuple)
    )
    interpolation_method: Text = attr.ib(
        default="bilinear", validator=attr.validators.in_(["nearest", "bilinear"])
    )
    normalize_from_range: Optional[Tuple[float, float]] = None
    normalize_to_range: Optional[Tuple[float, float]] = (0.0, 1.0)
    pad_to_stride_constraints: bool = True

    @property
    def is_complete_for_training(self) -> bool:
        """Return True if the configuration is fully specified for training.
        
        If not center cropping, nothing needs to be specified.

        If center cropping is enabled but not centering on an explicit anchor part,
        nothing else needs to be specified.

        If center cropping on an anchor part, then the anchor part name must be
        specified.
        """
        if not self.center_crop_instances:
            return True
        if not self.center_on_anchor_part:
            return True
        if self.anchor_part_name is not None:
            return True
        return False

    @property
    def is_complete_for_inference(self) -> bool:
        """Return True if the configuration is fully specified for inference.

        If center cropping, in addition to the configuration required for training, the
        bounding box size must also be specified. This is typically computed
        automatically from the ground truth data during training.
        """
        return self.is_complete_for_training and (
            not self.center_crop_instances
            or self.center_crop_bounding_box_size is not None
        )


@attr.s(auto_attribs=True)
class ModelConfig:
    """SLEAP model configuration.
    
    Attributes:
        preprocessing: Instance of `PreprocessingConfig` that specifies how the input
            data should be processed before being provided to the network.
        architecture: Name of the backbone architecture.
        backbone: Architecture instance specifying the backbone network architecture.
        outputs: List of `sleap.nn.heads.OutputHead` specifying the configuration of all
            of the outputs of the network.
    """

    preprocessing: PreprocessingConfig
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
        preprocessing: PreprocessingConfig,
        backbone: Architecture,
        outputs: Sequence[heads.OutputHead],
    ) -> "ModelConfig":
        """Create a model config from component configurations.

        This method is a convenient way to initialize this clsas without having to
        specify the `architecture` clsas name as a string as well.

        Args:
            preprocessing: Instance of `PreprocessingConfig` that specifies how the
                input data should be processed before being provided to the network.
            backbone: Architecture instance specifying the backbone network
                architecture.
            outputs: List of `sleap.nn.heads.OutputHead` specifying the configuration of
                all of the outputs of the network.

        Returns:
            The initialized `ModelConfig` instance.
        """
        return cls(
            preprocessing=preprocessing,
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
                    preprocessing=cattr.structure(
                        data_dicts["preprocessing"], PreprocessingConfig
                    ),
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
        instance_crop_padding: int = 16,
        bounding_box_size: Optional[int] = None,
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
            instance_crop_padding: Attribute from legacy `TrainerConfig`.
            bounding_box_size: Attribute from legacy `TrainerConfig`.

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

        # Use default preprocessing settings.
        preprocessing = PreprocessingConfig(
            center_crop_instances=data_dicts["output_type"] == 3,  # topdown
            center_on_anchor_part=instance_crop_use_ctr_node,
            anchor_part_name=anchor_part_name,
            center_crop_bounding_box_size=bounding_box_size,
            bounding_box_padding=instance_crop_padding,
        )
        if scale != 1.0:
            # Add input scaling.
            preprocessing.scale_inputs = (scale, scale)

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
            preprocessing=preprocessing,
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
            instance_crop_padding=trainer_dict.get("instance_crop_padding", 16),
            bounding_box_size=trainer_dict.get("bounding_box_size", None),
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

    @property
    def is_complete_for_training(self) -> bool:
        """Return True if the configuration is fully specified for training.

        This requires that the `PreprocessingConfig` is complete for training and that
        all output heads are complete.
        """
        return self.preprocessing.is_complete_for_training and all(
            [head.is_complete for head in self.outputs]
        )

    @property
    def is_complete_for_inference(self) -> bool:
        """Return True if the configuration is fully specified for inference.

        This requires that the `PreprocessingConfig` is complete for inference and that
        all output heads are complete.
        """
        return self.preprocessing.is_complete_for_inference and all(
            [head.is_complete for head in self.outputs]
        )

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

"""SLEAP model configuration and initialization.

This module provides classes used in model configuration, creation and loading.
"""
import tensorflow as tf

import attr
import cattr
from typing import TypeVar, Text, Any, Dict, Sequence, List, Optional, Tuple

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
class PreprocessingConfig:
    """Parameters for preprocessing input data.
    
    This set of attributes describe the type of preprocessing that models expect to be
    applied to the data.
    
    Attributes:
        scale_inputs: If not None, inputs will be resized by a relative scale factor to
            this (scale_height, scale_width) via bilinear interpolation.
        resize_to_shape: If not None, inputs will be resized to this (height, width)
            before any other preprocessing via bilinear interpolation. This is useful
            for rescaling large images, resizing variable size inputs to a fixed shape,
            or for resizing without preserving aspect ratio.
        normalize_to_range: Normalize range of input values to this (min_val, max_val)
            after casting to float32. For uint8 inputs this will normalize the full
            range of values (0, 255) to (min_val, max_val), but for other dtypes this
            normalize inputs to the data range.
        pad_to_stride_constraints: If True, padding will be applied to the bottom-right
            of the inputs after resizing in order to meet the shape constraints required
            by the backbone architecture. If False, shape constraints will be met via
            cropping from the bottom-right, which may result in data loss at the
            extremes of the image.
    """

    # TODO: Center cropping
    scale_inputs: Optional[Tuple[float, float]] = None
    resize_to_shape: Optional[Tuple[int, int]] = None
    normalize_to_range: Tuple[float, float] = (0.0, 1.0)
    pad_to_stride_constraints: bool = True


@attr.s(auto_attribs=True)
class ModelConfig:
    """SLEAP model configuration.
    
    Attributes:
        preprocessing: Instance of `PreprocessingConfig` that specifies how the input
            data should be processed before being provided to the network.
        architecture: Name of the backbone architecture.
        backbone: Architecture instance specifying the backbone network architecture.
        outputs: List of `sleap.nn.heads.OutputHead` specifying the configuration of all
            the outputs of the network.
    """

    preprocessing: PreprocessingConfig
    architecture: Text = attr.ib(validator=attr.validators.in_(ARCHITECTURE_NAMES))
    backbone: Architecture
    outputs: Sequence[heads.OutputHead]

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
    def from_legacy_cattr(
        cls,
        data_dicts: [Text, Any],
        scale: float = 1.0,
        sigma: float = 5.0,
        skeletons: Optional[Sequence[sleap.Skeleton]] = None,
        instance_crop_use_ctr_node: bool = False,
        instance_crop_ctr_node_ind: int = 0,
    ) -> "ModelConfig":
        """Create a model from legacy configuration format."""
        # Use default preprocessing settings.
        preprocessing = PreprocessingConfig()
        if scale != 1.0:
            # Add input scaling.
            preprocessing.scale_inputs = (scale, scale)

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

        if len(skeletons) == 0:
            raise ValueError(
                "No skeletons were specified in the data. "
                "Provide these manually to create a Model from legacy format."
            )
        skeleton = skeletons[0]

        # Infer output scale.
        output_scale = scale / float(backbone.output_stride)
        output_stride = int(1 / output_scale)

        # Setup outputs.
        if data_dicts["output_type"] == 0:
            # CONFIDENCE_MAP = 0
            outputs = [
                heads.OutputHead(
                    type="MultiPartConfmaps",
                    config=heads.MultiPartConfmaps(
                        part_names=skeleton.node_names, sigma=sigma
                    ),
                    stride=output_stride,
                )
            ]

        elif data_dicts["output_type"] == 1:
            # PART_AFFINITY_FIELD = 1
            outputs = [
                heads.OutputHead(
                    type="PartAffinityFields",
                    config=heads.PartAffinityFields(
                        edges=skeleton.edge_names, max_distance=sigma
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
                        user_part_anchor=instance_crop_use_ctr_node,
                        anchor_part_name=skeleton.node_names[
                            instance_crop_ctr_node_ind
                        ],
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
                        part_names=skeleton.node_names, sigma=sigma, centered=True
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

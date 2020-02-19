"""Head definition and configuration for different SLEAP output types.

This module provides classes for defining outputs that different backbones can be
trained to generate.
"""

import tensorflow as tf
import attr
import cattr

from typing import Union, TypeVar, Text, Sequence, Optional, Tuple, Dict, Any


@attr.s(auto_attribs=True)
class CentroidConfmap:
    """Single-landmark confidence map for locating the instance centroid.

    Attributes:
        use_anchor_part: If True, attempt to compute centroid from the specified part.
            This is useful when the body morphology is such that the bounding box
            centroid often falls on very different parts of the body or the background.
            When the body part is not visible, the bounding box centroid is used
            instead. If False, the bounding box centroid is always used.
        anchor_part_name: String specifying the body part name within the skeleton.
        sigma: Spread of the confidence map around the centroid location.
    """

    use_anchor_part: bool = False
    anchor_part_name: Optional[Text] = None
    sigma: float = 5.0

    @property
    def is_complete(self):
        """Return True if the configuration is fully specified."""
        return not self.use_anchor_part or self.anchor_part_name is not None

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
        if not self.is_complete:
            raise ValueError("Output head configuration is not complete.")
        return 1


@attr.s(auto_attribs=True)
class SinglePartConfmaps:
    """Single-instance landmark part confidence maps for locating landmarks.

    This output type differs from `MultiPartConfmaps` in that it indicates that only a
    single peak of each type should be generated, even if multiple instances are
    present. This is used in top-down multi-instance mode, or in single-instance mode.

    Attributes:
        part_names: Names of the parts corresponding to each channel in the output. Each
            of these will produce a different confidence map centered at the part
            location.
        sigma: Spread of the confidence map around each part location.
        centered: If True, indicates that this head expects a centered input image and
            will produce confidence maps corresponding to the centered instance. This
            implies that region proposals must first be generated to do the alignment
            (e.g., from `CentroidConfmap` peaks).
        center_on_anchor_part: If True, specifies that centering should be done relative
            to a body part rather than the center of the instance bounding box.
        anchor_part_name: String specifying the body part name to use as an anchor for
            centering. If `center_on_anchor_part` is False, this has no effect and does
            not need to be specified.
    """

    part_names: Optional[Sequence[Text]] = None
    sigma: float = 5.0
    centered: bool = False
    center_on_anchor_part: bool = False
    anchor_part_name: Optional[Text] = None

    @property
    def is_complete(self) -> bool:
        """Return True if the configuration is fully specified."""
        part_names_specified = self.part_names is not None and len(self.part_names) > 0
        anchor_specified = (
            not self.center_on_anchor_part or self.anchor_part_name is not None
        )
        centering_specified = not self.centered or anchor_specified
        return part_names_specified and centering_specified

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
        if not self.is_complete:
            raise ValueError("Output head configuration is not complete.")
        return len(self.part_names)


@attr.s(auto_attribs=True)
class MultiPartConfmaps:
    """Multi-instance landmark part confidence maps for locating landmarks.

    This output type differs from `SinglePartConfmaps` in that it indicates that all
    visible peaks of each type from all instances should be generated. This is used in
    bottom-up multi-instance mode, typically in conjunction with part affinity fields to
    enable grouping of sets of part types.

    Attributes:
        part_names: Names of the parts corresponding to each channel in the output. Each
            of these will produce a different confidence map centered at the part
            location.
        sigma: Spread of the confidence map around each part location.
    """

    part_names: Optional[Sequence[Text]] = None
    sigma: float = 5.0

    @property
    def is_complete(self) -> bool:
        """Return True if the configuration is fully specified."""
        return self.part_names is not None and len(self.part_names) > 0

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
        if not self.is_complete:
            raise ValueError("Output head configuration is not complete.")
        return len(self.part_names)


@attr.s(auto_attribs=True)
class PartAffinityFields:
    """Unit vector fields defined within the directed graph on images.

    This class specifies an output type corresponding to the part affinity field
    representation described in `Cao et al., 2016 <https://arxiv.org/abs/1611.08050>`_.

    In this representation, the body plan is described through a directed graph. Each
    edge of the graph is represented in image space by computing a unit vector at each
    pixel near the edge line segment that points along the direction of the edge (i.e.,
    from source to node locations).

    The unit vectors are represented by two channels each, resulting in a final tensor
    with twice the number of channels as the input.

    Attributes:
        max_distance: The maximum distance orthogonal to the source-destination line
            segment at which the part affinity field is non-zero. Points further than
            this distance (in full input scale pixels) will have 0 for both components.
        edges: List of (src_part, dst_part) tuples specifying the names of the source
            and destination parts.
    """

    edges: Optional[Sequence[Tuple[Text, Text]]] = None
    max_distance: float = 5.0

    @property
    def is_complete(self) -> bool:
        """Return True if the configuration is fully specified."""
        return self.edges is not None

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
        if not self.is_complete:
            raise ValueError("Output head configuration is not complete.")
        return len(self.edges) * 2


OUTPUT_TYPES = [
    CentroidConfmap,
    SinglePartConfmaps,
    MultiPartConfmaps,
    PartAffinityFields,
]
OUTPUT_TYPE_NAMES = [cls.__name__ for cls in OUTPUT_TYPES]
OutputConfig = TypeVar("OutputConfig", *OUTPUT_TYPES)


@attr.s(auto_attribs=True)
class OutputHead:
    """Output head definition used to configure the output layer of a model.

    Attributes:
        type: String name of the output head. This must be a name of a valid output
            type ("CentroidConfmap", "SinglePartConfmaps", "MultiPartConfmaps",
            "PartAffinityFields").
        config: Head type-specific configuration. This is an instance of one of the
            valid head types with the attributes specific to that type. Valid types
            include `CentroidConfmap`, `SinglePartConfmaps`, `MultiPartConfmaps`, and
            `PartAffinityFields`.
        stride: The expected output stride of the tensor resulting from this head. This
            is effectively (1 / scale) of the output relative to the input and used to
            appropriately connect the backbone to this head. Set to 1 to specify that a
            tensor of the same size as the input should be output. If >1, the output
            will be smaller than the input.
    """

    type: Text = attr.ib(validator=attr.validators.in_(OUTPUT_TYPE_NAMES))
    config: OutputConfig = attr.ib()
    stride: int

    @config.validator
    def _check_config(self, attribute, value):
        config_class_name = type(value).__name__
        if config_class_name != self.type:
            raise ValueError(
                f"Output head config ({config_class_name}) and type ({self.type}) "
                "must be the same."
            )

    @classmethod
    def from_config(cls, config: OutputConfig, stride: int) -> "OutputHead":
        """Create an output head from an output head configuration.

        This method is a convenient way to initialize this class without having to
        specify the `OutputConfig` class name as a string as well.

        Args:
            config: Head type-specific configuration. This is an instance of one of the
                valid head types with the attributes specific to that type. Valid types
                include `CentroidConfmap`, `SinglePartConfmaps`, `MultiPartConfmaps`,
                and `PartAffinityFields`.
            stride: The expected output stride of the tensor resulting from this head.
                This is effectively (1 / scale) of the output relative to the input and
                used to appropriately connect the backbone to this head. Set to 1 to
                specify that a tensor of the same size as the input should be output. If
                >1, the output will be smaller than the input.

        Returns:
            The initialized `OutputHead` instance.
        """
        return cls(
            type=type(config).__name__,
            config=config,
            stride=stride
            )

    @classmethod
    def from_cattr(cls, data_dicts: Dict[Text, Any]) -> "OutputHead":
        """Structure an output head from decoded JSON dictionaries."""
        for output_type_cls in OUTPUT_TYPES:
            if output_type_cls.__name__ == data_dicts["type"]:
                config_dict = data_dicts.get("config", {})
                return cls(
                    type=data_dicts["type"],
                    config=output_type_cls(**config_dict),
                    stride=data_dicts["stride"],
                )

        raise ValueError(
            "Could not find output type with name: '%s'" % data_dicts["type"]
        )

    @property
    def is_complete(self) -> bool:
        """Return True if the configuration is fully specified."""
        return self.config.is_complete

    @property
    def num_channels(self) -> int:
        """Return the number of channels in the output tensor for this head."""
        return self.config.num_channels

    def make_head(self, x: tf.Tensor, name: Optional[Text] = None) -> tf.Tensor:
        """Generate a layer that maps the input tensor to the channels of the head.

        Args:
            x: The input tensor to the head. This is typically the output of a backbone.
            name: Name of the layer. If not specified, defaults to the `type` of this
                output head.

        Returns:
            The output tensor mapped to the configured number of channels for this
            output type through a 1x1 linear convolution.
        """
        if not self.is_complete:
            raise ValueError("Output head configuration is not complete.")
        if name is None:
            name = self.type
        return tf.keras.layers.Conv2D(
            filters=self.config.num_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            name=name,
        )(x)


# Register global cattr structuring hook.
cattr.register_structure_hook(OutputHead, lambda d, t: OutputHead.from_cattr(d))

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
        use_part_anchor: If True, attempt to compute centroid from the specified part.
            This is useful when the body morphology is such that the bounding box
            centroid often falls on very different parts of the body or the background.
            When the body part is not visible, the bounding box centroid is used
            instead. If False, the bounding box centroid is always used.
        anchor_part_name: String specifying the body part name within the skeleton. This
            takes precedence over `anchor_part_ind` if both are specified.
        anchor_part_ind: Index of the body part within the skeleton. If both this and
            `anchor_part_name` are specified, the latter will take precedence.
        sigma: Spread of the confidence map around the centroid location.
    """

    use_part_anchor: bool = False
    anchor_part_name: Optional[Text] = attr.ib(default=None)
    anchor_part_ind: Optional[int] = attr.ib(default=None)
    sigma: float = 5.0

    @anchor_part_name.validator
    def _check_anchor_part_name(self, attribute, value):
        if self.use_part_anchor:
            if value is None and self.anchor_part_ind is None:
                raise ValueError(
                    "If using a part anchor, either the anchor_part_name or "
                    "anchor_part_ind must be set."
                )

    @anchor_part_ind.validator
    def _check_anchor_part_ind(self, attribute, value):
        if self.use_part_anchor:
            if value is None and self.anchor_part_name is None:
                raise ValueError(
                    "If using a part anchor, either the anchor_part_name or "
                    "anchor_part_ind must be set."
                )

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
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
    """

    part_names: Sequence[Text]
    sigma: float = 5.0

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
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

    part_names: Sequence[Text]
    sigma: float = 5.0

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
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
        edge_inds: List of (src_ind, dst_ind) tuples specifying the index of the source
            and destination nodes (parts) that form the directed graph.
        part_names: List of names of the nodes (parts) that correspond to the indices in
            `edge_inds`.
    """

    edge_inds: Sequence[Tuple[int, int]] = attr.ib()
    part_names: Sequence[Text]
    max_distance: float = 5.0

    @edge_inds.validator
    def _check_edge_inds(self, attribute, value):
        min_parts = max([max(src, dst) for (src, dst) in value]) + 1
        if len(self.part_names) < min_parts:
            raise ValueError(
                f"Fewer part names specified ({len(self.part_names)}) than expected "
                f"from edge indices ({min_parts}). Check the edge indices for the "
                "part affinity field output head."
            )

    @property
    def num_channels(self) -> int:
        """Return number of channels in the output tensor."""
        return len(self.edge_inds) * 2


OUTPUT_TYPES = [
    CentroidConfmap,
    SinglePartConfmaps,
    MultiPartConfmaps,
    PartAffinityFields,
]
OutputConfig = TypeVar("OutputConfig", *OUTPUT_TYPES)


@attr.s(auto_attribs=True)
class OutputHead:
    """Output head definition used to configure the output layer of a model.

    Attributes:
        type: String name of the output head. This must be a name of a valid output
            type ("CentroidConfmap", "SinglePartConfmaps", "MultiPartConfmaps",
            "PartAffinityFields").
        config: Head type-specific configuration. This is an instance of one of the
            valid head types with the attributes specific to that type.
        stride: The expected output stride of the tensor resulting from this head. This
            is effectively (1 / scale) of the output relative to the input and used to
            appropriately connect the backbone to this head. Set to 1 to specify that a
            tensor of the same size as the input should be output. If >1, the output
            will be smaller than the input.
    """

    type: Text = attr.ib(
        validator=attr.validators.in_([cls.__name__ for cls in OUTPUT_TYPES])
    )
    config: OutputConfig
    stride: int

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

"""Model head definitions for defining model output types."""

import attr
from typing import Optional, Text, List, Sequence, Tuple, Union

from sleap.nn.config import (
    CentroidsHeadConfig,
    SingleInstanceConfmapsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
)


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsHead:
    """Head for specifying single instance confidence maps."""

    part_names: List[Text]
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: SingleInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "SingleInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `SingleInstanceConfmapsHeadConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class CentroidConfmapsHead:
    """Head for specifying instance centroid confidence maps."""

    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return 1

    @classmethod
    def from_config(cls, config: CentroidsHeadConfig) -> "CentroidConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `CentroidsHeadConfig` instance specifying the head parameters.

        Returns:
            The instantiated head with the specified configuration options.
        """
        return cls(
            anchor_part=config.anchor_part,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class CenteredInstanceConfmapsHead:
    """Head for specifying centered instance confidence maps."""

    part_names: List[Text]
    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: CenteredInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "CenteredInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `CenteredInstanceConfmapsHeadConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        return cls(
            part_names=part_names,
            anchor_part=config.anchor_part,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class MultiInstanceConfmapsHead:
    """Head for specifying multi-instance confidence maps."""

    part_names: List[Text]
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: MultiInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "MultiInstanceConfmapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `MultiInstanceConfmapsHeadConfig` instance specifying the head
                parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.part_names is not None:
            part_names = config.part_names
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


@attr.s(auto_attribs=True)
class PartAffinityFieldsHead:
    """Head for specifying multi-instance part affinity fields."""

    edges: Sequence[Tuple[Text, Text]]
    sigma: float = 15.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return int(len(self.edges) * 2)

    @classmethod
    def from_config(
        cls,
        config: PartAffinityFieldsHeadConfig,
        edges: Optional[Sequence[Tuple[Text, Text]]] = None,
    ) -> "PartAffinityFieldsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `PartAffinityFieldsHeadConfig` instance specifying the head
                parameters.
            edges: List of 2-tuples of the form `(source_node, destination_node)` that
                define pairs of text names of the directed edges of the graph. This must
                be set if the `edges` attribute of the configuration is not set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.edges is not None:
            edges = config.edges
        return cls(
            edges=edges,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


ConfmapConfig = Union[
    CentroidsHeadConfig,
    SingleInstanceConfmapsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
]


@attr.s(auto_attribs=True)
class OffsetRefinementHead:
    """Head for specifying offset refinement maps."""

    part_names: List[Text]
    output_stride: int = 1
    sigma_threshold: float = 0.2
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return int(len(self.part_names) * 2)

    @classmethod
    def from_config(
        cls,
        config: ConfmapConfig,
        part_names: Optional[List[Text]] = None,
        sigma_threshold: float = 0.2,
        loss_weight: float = 1.0,
    ) -> "OffsetRefinementHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `ConfmapConfig` instance specifying the head parameters.
            part_names: Text name of the body parts (nodes) that the head will be
                configured to produce. The number of parts determines the number of
                channels in the output. This must be provided if the `part_names`
                attribute of the configuration is not set.
            sigma_threshold: Minimum confidence map value below which offsets will be
                replaced with zeros.
            loss_weight: Weight of the loss associated with this head.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if hasattr(config, "part_names"):
            if config.part_names is not None:
                part_names = config.part_names
        elif hasattr(config, "anchor_part"):
            part_names = [config.anchor_part]
        return cls(
            part_names=part_names,
            output_stride=config.output_stride,
            sigma_threshold=sigma_threshold,
            loss_weight=loss_weight,
        )

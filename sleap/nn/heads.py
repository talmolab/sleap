import attr
from typing import Optional, Text, List, Sequence, Tuple

from sleap.nn.config import (
    CentroidsHeadConfig,
    SingleInstanceConfmapsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
)


@attr.s(auto_attribs=True)
class CentroidConfmapsHead:
    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1

    @property
    def channels(self):
        return 1

    @classmethod
    def from_config(cls, config: CentroidsHeadConfig) -> "CentroidConfmapsHead":
        return cls(
            anchor_part=config.anchor_part,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsHead:
    part_names: List[Text]
    sigma: float = 5.0
    output_stride: int = 1

    @property
    def channels(self):
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: SingleInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "SingleInstanceConfmapsHead":
        if config.part_names is not None:
            part_names = config.part_names
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class CenteredInstanceConfmapsHead:
    part_names: List[Text]
    anchor_part: Optional[Text] = None
    sigma: float = 5.0
    output_stride: int = 1

    @property
    def channels(self):
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: CenteredInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "CenteredInstanceConfmapsHead":
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
    part_names: List[Text]
    sigma: float = 5.0
    output_stride: int = 1

    @property
    def channels(self):
        return len(self.part_names)

    @classmethod
    def from_config(
        cls,
        config: MultiInstanceConfmapsHeadConfig,
        part_names: Optional[List[Text]] = None,
    ) -> "MultiInstanceConfmapsHead":
        if config.part_names is not None:
            part_names = config.part_names
        return cls(
            part_names=part_names,
            sigma=config.sigma,
            output_stride=config.output_stride,
        )


@attr.s(auto_attribs=True)
class PartAffinityFieldsHead:
    edges: Sequence[Tuple[Text, Text]]
    sigma: float = 15.0
    output_stride: int = 1

    @property
    def channels(self):
        return int(len(self.edges) * 2)

    @classmethod
    def from_config(
        cls,
        config: PartAffinityFieldsHeadConfig,
        edges: Optional[Sequence[Tuple[Text, Text]]] = None,
    ) -> "PartAffinityFieldsHead":
        if config.edges is not None:
            edges = config.edges
        return cls(edges=edges, sigma=config.sigma, output_stride=config.output_stride)

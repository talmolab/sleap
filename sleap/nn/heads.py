"""Model head definitions for defining model output types."""

import tensorflow as tf
import attr
from typing import Optional, Text, List, Sequence, Tuple, Union
from abc import ABC, abstractmethod

from sleap.nn.config import (
    CentroidsHeadConfig,
    SingleInstanceConfmapsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
    ClassMapsHeadConfig,
    ClassVectorsHeadConfig,
)


@attr.s(auto_attribs=True)
class Head(ABC):
    """Base class for model output heads."""

    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    @abstractmethod
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        pass

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "linear"

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "mse"

    def make_head(self, x_in: tf.Tensor, name: Optional[Text] = None) -> tf.Tensor:
        """Make head output tensor from input feature tensor.

        Args:
            x_in: An input `tf.Tensor`.
            name: If provided, specifies the name of the output layer. If not (the
                default), uses the name of the head as the layer name.

        Returns:
            A `tf.Tensor` with the correct shape for the head.
        """
        if name is None:
            name = f"{type(self).__name__}"
        return tf.keras.layers.Conv2D(
            filters=self.channels,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=self.activation,
            name=name,
        )(x_in)


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsHead(Head):
    """Head for specifying single instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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
            loss_weight=config.loss_weight,
        )


@attr.s(auto_attribs=True)
class CentroidConfmapsHead(Head):
    """Head for specifying instance centroid confidence maps.

    Attributes:
        anchor_part: Name of the part to use as an anchor node. If not specified, the
            bounding box centroid will be used.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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
            loss_weight=config.loss_weight,
        )


@attr.s(auto_attribs=True)
class CenteredInstanceConfmapsHead(Head):
    """Head for specifying centered instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        anchor_part: Name of the part to use as an anchor node. If not specified, the
            bounding box centroid will be used.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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
            loss_weight=config.loss_weight,
        )


@attr.s(auto_attribs=True)
class MultiInstanceConfmapsHead(Head):
    """Head for specifying multi-instance confidence maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma: Spread of the confidence maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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
class PartAffinityFieldsHead(Head):
    """Head for specifying multi-instance part affinity fields.

    Attributes:
        edges: List of tuples of `(source, destination)` node names.
        sigma: Spread of the part affinity fields.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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


@attr.s(auto_attribs=True)
class ClassMapsHead(Head):
    """Head for specifying class identity maps.

    Attributes:
        classes: List of string names of the classes.
        sigma: Spread of the class maps around each node.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    classes: List[Text]
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.classes)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "sigmoid"

    @classmethod
    def from_config(
        cls,
        config: ClassMapsHeadConfig,
        classes: Optional[List[Text]] = None,
    ) -> "ClassMapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `ClassMapsHeadConfig` instance specifying the head parameters.
            classes: List of string names of the classes that this head will predict.
                This must be set if the `classes` attribute of the configuration is not
                set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.classes is not None:
            classes = config.classes
        return cls(
            classes=classes,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )


@attr.s(auto_attribs=True)
class ClassVectorsHead(Head):
    """Head for specifying classification heads.

    Attributes:
        classes: List of string names of the classes.
        num_fc_layers: Number of fully connected layers after flattening input features.
        num_fc_units: Number of units (dimensions) in fully connected layers prior to
            classification output.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    classes: List[Text]
    num_fc_layers: int = 1
    num_fc_units: int = 64
    global_pool: bool = True
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.classes)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "softmax"

    @property
    def loss_function(self) -> str:
        """Return the name of the loss function to use for this head."""
        return "categorical_crossentropy"

    @classmethod
    def from_config(
        cls,
        config: ClassVectorsHeadConfig,
        classes: Optional[List[Text]] = None,
    ) -> "ClassVectorsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `ClassVectorsHeadConfig` instance specifying the head parameters.
            classes: List of string names of the classes that this head will predict.
                This must be set if the `classes` attribute of the configuration is not
                set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.classes is not None:
            classes = config.classes
        return cls(
            classes=classes,
            num_fc_layers=config.num_fc_layers,
            num_fc_units=config.num_fc_units,
            global_pool=config.global_pool,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )

    def make_head(self, x_in: tf.Tensor, name: Optional[Text] = None) -> tf.Tensor:
        """Make head output tensor from input feature tensor.

        Args:
            x_in: An input `tf.Tensor`.
            name: If provided, specifies the name of the output layer. If not (the
                default), uses the name of the head as the layer name.

        Returns:
            A `tf.Tensor` with the correct shape for the head.
        """
        if name is None:
            name = f"{type(self).__name__}"
        x = x_in
        if self.global_pool:
            x = tf.keras.layers.GlobalMaxPool2D(name="pre_classification_global_pool")(
                x
            )
        x = tf.keras.layers.Flatten(name="pre_classification_flatten")(x)
        for i in range(self.num_fc_layers):
            x = tf.keras.layers.Dense(
                self.num_fc_units, name=f"pre_classification{i}_fc"
            )(x)
            x = tf.keras.layers.Activation("relu", name=f"pre_classification{i}_relu")(
                x
            )
        x = tf.keras.layers.Dense(self.channels, activation=self.activation, name=name)(
            x
        )
        return x


ConfmapConfig = Union[
    CentroidsHeadConfig,
    SingleInstanceConfmapsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
]


@attr.s(auto_attribs=True)
class OffsetRefinementHead(Head):
    """Head for specifying offset refinement maps.

    Attributes:
        part_names: List of strings specifying the part names associated with channels.
        sigma_threshold: Threshold of confidence map values to use for defining the
            boundary of the offset maps.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

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

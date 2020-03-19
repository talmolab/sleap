"""Common utilities for architecture and model building."""

import attr
import tensorflow as tf


@attr.s(auto_attribs=True)
class IntermediateFeature:
    """Intermediate feature tensor for use in skip connections.

    This class is effectively a named tuple to store the stride (resolution) metadata.

    Attributes:
        tensor: The tensor output from an intermediate layer.
        stride: Stride of the tensor relative to the input.
    """

    tensor: tf.Tensor
    stride: int

    @property
    def scale(self) -> float:
        """Return the absolute scale of the tensor relative to the input.

        This is equivalent to the reciprocal of the stride, e.g., stride 2 => scale 0.5.
        """
        return 1.0 / float(self.stride)

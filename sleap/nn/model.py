import os
import attr
import cattr
import tensorflow as tf
import numpy as np
from enum import Enum
from typing import List, Text, Callable, Tuple, Dict, Union
import logging

from sleap import Skeleton
from sleap.nn.architectures import *
from sleap.nn import utils

logger = logging.getLogger(__name__)


class ModelOutputType(Enum):
    """
    Supported output type for SLEAP models. Currently supported
    output modes are:

    CONFIDENCE_MAPS: Scalar fields representing the probability of finding a
    skeleton node within an image. Models with this type will output a tensor
    that contains N channels, where N is the number of unique nodes across all
    skeletons for the model.
    PART_AFFINITY_FIELDS: A nonparametric representation made up from a set of
    "2D vector fields that encode the location and orientation of limbs over
    the image domain". Models with this type will output a tensor that contains
    2*E channels where E is the number of unique edges across all skeletons for
    the model.
    See "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"
    by Cao et al.

    """

    CONFIDENCE_MAP = 0
    PART_AFFINITY_FIELD = 1
    CENTROIDS = 2
    TOPDOWN_CONFIDENCE_MAP = 3

    def __str__(self):
        if self == ModelOutputType.CONFIDENCE_MAP:
            return "confmaps"
        elif self == ModelOutputType.PART_AFFINITY_FIELD:
            return "pafs"
        elif self == ModelOutputType.CENTROIDS:
            return "centroids"
        elif self == ModelOutputType.TOPDOWN_CONFIDENCE_MAP:
            return "topdown_confidence_maps"
        else:
            # This shouldn't ever happen I don't think.
            raise NotImplementedError(
                f"__str__ not implemented for ModelOutputType={self}"
            )


@attr.s(auto_attribs=True)
class Model:
    """
    The Model class is a wrapper class that specifies an interface to pose
    estimation models supported by sLEAP. It is fairly straighforward class
    that allows specification of the underlying architecture and the
    instantiation of a predictor object for inference.

        Args:
            output_type: The output type of this model.
            skeletons:
            backbone: A class with an output method that returns a
            tf.Tensor of the output of the backbone block. This tensor
            will be set as the outputs of the keras.Model that is constructed.
            See sleap.nn.architectures for example backbone block classes.
            backbone_name: The name of the backbone architecture, this defaults
            to self.backbone.__name__ when set to None. In general, the user should
            not set this value.

    """

    output_type: ModelOutputType
    backbone: BackboneType
    skeletons: Union[None, List[Skeleton]] = None
    backbone_name: str = None

    def __attrs_post_init__(self):

        if not isinstance(self.backbone, tuple(available_archs)):
            raise ValueError(
                f"backbone ({self.backbone}) is not "
                f"in available architectures ({available_archs})"
            )

        if not hasattr(self.backbone, "output"):
            raise ValueError(
                f"backbone ({self.backbone}) has now output method! "
                f"Not a valid backbone architecture!"
            )

        if self.backbone_name is None:
            self.backbone_name = self.backbone.__class__.__name__

    def output(self, input_tensor, num_output_channels=None):
        """
        Invoke the backbone function with current backbone_args and backbone_kwargs
        to produce the model backbone block. This is a convenience property for
        self.backbone.output(input_tensor, num_ouput_channels)

        Args:
            input_tensor: An input layer to feed into the backbone.
            num_output_channels: The number of output channels for the network.

        Returns:
            The return value of backbone output method, should be a tf.Tensor that is
            the output of the backbone block.
        """

        # TODO: Add support for multiple skeletons
        # If we need to, figure out how many output channels we will have
        if num_output_channels is None:
            if self.skeletons is not None:
                if (
                    self.output_type == ModelOutputType.CONFIDENCE_MAP or
                    self.output_type == ModelOutputType.TOPDOWN_CONFIDENCE_MAP
                ):
                    num_outputs_channels = len(self.skeletons[0].nodes)
                elif self.output_type == ModelOutputType.PART_AFFINITY_FIELD:
                    num_outputs_channels = len(self.skeleton[0].edges) * 2
            else:
                raise ValueError(
                    "Model.skeletons has not been set. "
                    "Cannot infer num output channels."
                )

        return self.backbone.output(input_tensor, num_output_channels)

    @property
    def name(self):
        """
        Get the name of the backbone function. This is a convenience method for:
        self.backbone.__name__

        Returns:
            A string representation of the backbone's name.
        """
        return self.backbone_name

    @property
    def down_blocks(self):
        """Returns the number of pooling or striding blocks in the backbone.

        This is useful when computing valid dimensions of the input data.

        If the backbone does not provide enough information to infer this,
        this is set to 0.
        """

        if hasattr(self.backbone, "down_blocks"):
            return self.backbone.down_blocks

        else:
            return 0

    @property
    def output_scale(self):
        """Calculates output scale relative to input."""

        if hasattr(self.backbone, "output_scale"):
            return self.backbone.output_scale

        elif hasattr(self.backbone, "down_blocks") and hasattr(
            self.backbone, "up_blocks"
        ):
            asym = self.backbone.down_blocks - self.backbone.up_blocks
            return 1 / (2 ** asym)

        elif hasattr(self.backbone, "initial_stride"):
            return 1 / self.backbone.initial_stride

        else:
            return 1

    @staticmethod
    def _structure_model(model_dict, cls):
        """Structuring hook for instantiating Model via cattrs.

        This function should be used directly with cattrs as a
        structuring hook. It serves the purpose of instantiating
        the appropriate backbone class from the string name.

        This is required when backbone classes do not have a
        unique attribute name from which to infer the appropriate
        class to use.

        Args:
            model_dict: Dictionaries containing deserialized Model.
            cls: Class to return (not used).

        Returns:
            An instantiated Model class with the correct backbone.

        Example:
            >> cattr.register_structure_hook(Model, Model.structure_model)
        """

        arch_idx = available_arch_names.index(model_dict["backbone_name"])
        backbone_cls = available_archs[arch_idx]

        return Model(
            backbone=backbone_cls(**model_dict["backbone"]),
            output_type=ModelOutputType(model_dict["output_type"]),
            skeletons=model_dict["skeletons"],
        )


@attr.s(auto_attribs=True)
class InferenceModel:
    """This class provides convenience metadata and methods for running inference from
    a trained model."""

    skeleton: Skeleton
    input_scale: float = 1.0
    output_scale: float = 1.0
    input_tensor_ind: int = 0
    output_tensor_ind: int = -1
    down_blocks: int = 5
    model_path: Text = None
    keras_model: keras.Model = None

    @classmethod
    def from_training_job(cls, training_job: Union["sleap.nn.job.TrainingJob", Text]):
        """Create an InferenceModel from a TrainingJob or path to json file."""

        if isinstance(training_job, str):
            from sleap.nn.job import TrainingJob

            training_job = TrainingJob.load_json(training_job)

        return cls(
            skeleton=training_job.model.skeletons[0],
            input_scale=training_job.trainer.scale,
            output_scale=training_job.trainer.scale * training_job.model.output_scale,
            input_tensor_ind=0,
            output_tensor_ind=-1,
            down_blocks=training_job.model.down_blocks,
            model_path=training_job.model_path,
        )

    def __attrs_post_init__(self):

        # Load model if needed.
        if self.keras_model is None:
            self.load_model()

        self._setup_input_output_tensors()

    def load_model(self, model_path: Text = None):
        """Loads a saved model with specified settings."""

        # Use attribute-stored model path if a new one was not specified.
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("Model path was not specified.")

        # Load from disk.
        self.keras_model = tf.keras.models.load_model(
            model_path, custom_objects={"tf": tf}
        )

        # Store the path to the current model.
        self.model_path = model_path

    def _setup_input_output_tensors(self):
        """Create model with the specified input/output tensors."""

        self.keras_model = keras.Model(
            self.keras_model.inputs[self.input_tensor_ind],
            self.keras_model.outputs[self.output_tensor_ind],
        )

    @property
    def input_tensor(self) -> tf.Tensor:
        """Returns the input tensor to the model."""
        return self.keras_model.input

    @property
    def output_tensor(self) -> tf.Tensor:
        """Returns the output tensor from the model."""
        return self.keras_model.output

    @property
    def output_relative_scale(self) -> float:
        """Returns the scale of the model outputs relative to the inputs."""
        return self.output_scale / self.input_scale

    @property
    def trained_input_shape(self) -> Tuple[int]:
        """Returns the shape of the input tensor."""
        return tuple(self.input_tensor.shape)

    def predict(self, X: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """Runs inference on input data."""

        return utils.batched_call_slices(
            self.keras_model, X, batch_size=batch_size, return_numpy=True
        )

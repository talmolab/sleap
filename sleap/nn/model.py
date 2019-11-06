import os
import attr
import cattr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from enum import Enum
from typing import List, Text, Callable, Tuple, Dict, Union
import logging

# from sleap.nn.training import TrainingJob
from sleap.skeleton import Skeleton
from sleap.nn.augmentation import Augmenter
from sleap.nn.architectures import *

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

    def __str__(self):
        if self == ModelOutputType.CONFIDENCE_MAP:
            return "confmaps"
        elif self == ModelOutputType.PART_AFFINITY_FIELD:
            return "pafs"
        elif self == ModelOutputType.CENTROIDS:
            return "centroids"
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
                if self.output_type == ModelOutputType.CONFIDENCE_MAP:
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
    a TrainingJob."""

    job: "sleap.nn.training.TrainingJob"
    _keras_model: keras.Model = None
    _model_path: Text = None
    _trained_input_shape: Tuple[int] = None
    _output_channels: int = None

    @property
    def skeleton(self) -> Skeleton:
        """Returns the skeleton associated with this model."""

        return self.job.model.skeletons[0]

    @property
    def output_type(self) -> ModelOutputType:
        """Returns the output type of this model."""

        return self.job.model.output_type

    @property
    def input_scale(self) -> float:
        """Returns the scale of the images that the model was trained on."""

        return self.job.trainer.scale

    @property
    def output_scale(self) -> float:
        """Returns the scale of the outputs of the model relative to the original data.

        For a model trained on inputs with scale = 0.5 that outputs predictions that
        are half of the size of the inputs, the output scale is 0.25.
        """
        return self.input_scale * self.job.model.output_scale

    @property
    def output_relative_scale(self) -> float:
        """Returns the scale of the outputs relative to the scaled inputs.

        This differs from output_scale in that it is the scaling factor after
        applying the input scaling.
        """

        return self.job.model.output_scale

    def compute_output_shape(
        self, input_shape: Tuple[int], relative=True
    ) -> Tuple[int]:
        """Returns the output tensor shape for a given input shape.

        Args:
            input_shape: Shape of input images in the form (height, width).
            relative: If True, input_shape specifies the shape after input scaling.

        Returns:
            A tuple of (height, width, channels) of the output of the model.
        """

        # TODO: Support multi-input/multi-output models.

        scaling_factor = self.output_scale
        if relative:
            scaling_factor = self.output_relative_scale

        output_shape = (
            int(input_shape[0] * scaling_factor),
            int(input_shape[1] * scaling_factor),
            self.output_channels,
        )

        return output_shape

    def load_model(self, model_path: Text = None) -> keras.Model:
        """Loads a saved model from disk and caches it.

        Args:
            model_path: If not provided, uses the model
                paths in the training job.

        Returns:
            The loaded Keras model. This model can accept any size
            of inputs that are valid.
        """

        if not model_path:
            # Try the best model first.
            model_path = os.path.join(self.job.save_dir, self.job.best_model_filename)

            # Try the final model if that didn't exist.
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    self.job.save_dir, self.job.final_model_filename
                )

        # Load from disk.
        keras_model = keras.models.load_model(model_path, custom_objects={"tf": tf})
        logger.info("Loaded model: " + model_path)

        # Store the loaded model path for reference.
        self._model_path = model_path

        # TODO: Multi-input/output support
        # Find the original data shape from the input shape of the first input node.
        self._trained_input_shape = keras_model.get_input_shape_at(0)

        # Save output channels since that should be static.
        self._output_channels = keras_model.get_output_shape_at(0)[-1]

        # Create input node with undetermined height/width.
        input_tensor = keras.layers.Input((None, None, self.input_channels))
        keras_model = keras.Model(
            inputs=input_tensor, outputs=keras_model(input_tensor)
        )

        # Save the modified and loaded model.
        self._keras_model = keras_model

        return self.keras_model

    @property
    def keras_model(self) -> keras.Model:
        """Returns the underlying Keras model, loading it if necessary."""

        if self._keras_model is None:
            self.load_model()

        return self._keras_model

    @property
    def model_path(self) -> Text:
        """Returns the path to the loaded model."""

        if not self._model_path:
            raise AttributeError(
                "No model loaded. Call inference_model.load_model() first."
            )

        return self._model_path

    @property
    def trained_input_shape(self) -> Tuple[int]:
        """Returns the shape of the model when it was loaded."""

        if not self._trained_input_shape:
            raise AttributeError(
                "No model loaded. Call inference_model.load_model() first."
            )

        return self._trained_input_shape

    @property
    def output_channels(self) -> int:
        """Returns the number of output channels of the model."""
        if not self._trained_input_shape:
            raise AttributeError(
                "No model loaded. Call inference_model.load_model() first."
            )

        return self._output_channels

    @property
    def input_channels(self) -> int:
        """Returns the number of channels expected for the input data."""

        # TODO: Multi-output support
        return self.trained_input_shape[-1]

    @property
    def is_grayscale(self) -> bool:
        """Returns True if the model expects grayscale images."""

        return self.input_channels == 1

    @property
    def down_blocks(self):
        """Returns the number of pooling steps applied during the model.

        Data needs to be of a shape divisible by the number of pooling steps.
        """

        # TODO: Replace this with an explicit calculation that takes stride sizes into account.
        return self.job.model.down_blocks

    def predict(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Runs inference on the input data.

        This is a simple wrapper around the keras model predict function.

        Args:
            X: The inputs to provide to the model. Can be different height/width as
                the data it was trained on.
            batch_size: Batch size to perform inference on at a time.
            normalize: Applies normalization to the input data if needed
                (e.g., if casting or range normalization is required).

        Returns:
            The outputs of the model.
        """

        if normalize:
            # TODO: Store normalization scheme in the model metadata.
            if isinstance(X, np.ndarray):
                if X.dtype == np.dtype("uint8"):
                    X = X.astype("float32") / 255.0
            elif isinstance(X, list):
                for i in range(len(X)):
                    if X[i].dtype == np.dtype("uint8"):
                        X[i] = X[i].astype("float32") / 255.0

        return self.keras_model.predict(X, batch_size=batch_size)

"""Inference pipelines and utilities.

This module contains the classes and high level APIs for predicting instances on new
data using trained models.

The inference logic is implemented at two levels:

- Low-level `InferenceModel`s which subclass `tf.keras.Model` and implement the core
  TensorFlow operations surrounding inference. These should only be used when
  implementing custom inference routines, such as real-time or performance-critical
  applications. They do not implement tracking (identity association).

- High-level `Predictor`s which handle data loading, preprocessing, inference, tracking
  and postprocessing, including converting raw array results into SLEAP-specific data
  structures. These should be used for general-purpose prediction, including interactive
  inference and applications that require tracking (identity association).

For more information on tracking, see the `sleap.nn.tracking` module.

The recommended high-level API for loading saved models is the `sleap.load_models`
function which provides a simplified interface for creating `Predictor`s.
"""

import attr
import logging
import warnings
import os
import time
from abc import ABC, abstractmethod
from typing import Text, Optional, List, Dict, Union, Iterator

import tensorflow as tf
import numpy as np

import sleap
from sleap import util
from sleap.nn.config import TrainingJobConfig
from sleap.nn.model import Model
from sleap.nn.tracking import Tracker, run_tracker
from sleap.nn.paf_grouping import PAFScorer
from sleap.nn.data.grouping import group_examples_iter
from sleap.nn.data.pipelines import (
    Provider,
    Pipeline,
    LabelsReader,
    VideoReader,
    Normalizer,
    Resizer,
    Prefetcher,
    LambdaFilter,
    KerasModelPredictor,
    LocalPeakFinder,
    PredictedInstanceCropper,
    InstanceCentroidFinder,
    InstanceCropper,
    GlobalPeakFinder,
    MockGlobalPeakFinder,
    KeyFilter,
    KeyRenamer,
    KeyDeviceMover,
    PredictedCenterInstanceNormalizer,
    PointsRescaler,
)

logger = logging.getLogger(__name__)


def safely_generate(ds: tf.data.Dataset, progress: bool = True):
    """Yields examples from dataset, catching and logging exceptions."""
    # Unsafe generating:
    # for example in ds:
    #     yield example

    ds_iter = iter(ds)

    i = 0
    wall_t0 = time.time()
    done = False
    while not done:
        try:
            next_val = next(ds_iter)
            yield next_val
        except StopIteration:
            done = True
        except Exception as e:
            logger.info(f"ERROR in sample index {i}")
            logger.info(e)
            logger.info("")
        finally:
            if not done:
                i += 1

            # Show the current progress (frames, time, fps)
            if progress:
                if (i and i % 1000 == 0) or done:
                    elapsed_time = time.time() - wall_t0
                    logger.info(
                        f"Finished {i} examples in {elapsed_time:.2f} seconds "
                        "(inference + postprocessing)"
                    )
                    if elapsed_time:
                        logger.info(f"examples/s = {i/elapsed_time}")


def get_keras_model_path(path: Text) -> Text:
    if path.endswith(".json"):
        path = os.path.dirname(path)
    return os.path.join(path, "best_model.h5")


@attr.s(auto_attribs=True)
class Predictor(ABC):
    """Base interface class for predictors."""

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        pass

    @abstractmethod
    def make_pipeline(self):
        pass

    @abstractmethod
    def predict(self, data_provider: Provider):
        pass


@attr.s(auto_attribs=True)
class MockPredictor(Predictor):
    labels: sleap.Labels

    @classmethod
    def from_trained_models(cls, labels_path: Text):
        labels = sleap.Labels.load_file(labels_path)
        return cls(labels=labels)

    def make_pipeline(self):
        pass

    def predict(self, data_provider: Provider):

        prediction_video = None

        # Try to match specified video by its full path
        prediction_video_path = os.path.abspath(data_provider.video.filename)
        for video in self.labels.videos:
            if os.path.abspath(video.filename) == prediction_video_path:
                prediction_video = video
                break

        if prediction_video is None:
            # Try to match on filename (without path)
            prediction_video_path = os.path.basename(data_provider.video.filename)
            for video in self.labels.videos:
                if os.path.basename(video.filename) == prediction_video_path:
                    prediction_video = video
                    break

        if prediction_video is None:
            # Default to first video in labels file
            prediction_video = self.labels.videos[0]

        # Get specified frames from labels file (or use None for all frames)
        frame_idx_list = (
            list(data_provider.example_indices)
            if data_provider.example_indices
            else None
        )

        frames = self.labels.find(video=prediction_video, frame_idx=frame_idx_list)

        # Run tracker as specified
        if self.tracker:
            frames = run_tracker(tracker=self.tracker, frames=frames)
            self.tracker.final_pass(frames)

        # Return frames (there are no "raw" predictions we could return)
        return frames


@attr.s(auto_attribs=True)
class VisualPredictor(Predictor):
    """Predictor class for generating the visual output of model."""

    config: TrainingJobConfig
    model: Model
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)

    @classmethod
    def from_trained_models(cls, model_path: Text) -> "VisualPredictor":
        cfg = TrainingJobConfig.load_json(model_path)
        keras_model_path = get_keras_model_path(model_path)
        model = Model.from_config(cfg.model)
        model.keras_model = tf.keras.models.load_model(keras_model_path, compile=False)

        return cls(config=cfg, model=model)

    def head_specific_output_keys(self) -> List[Text]:
        keys = []

        key = self.confidence_maps_key_name
        if key:
            keys.append(key)

        key = self.part_affinity_fields_key_name
        if key:
            keys.append(key)

        return keys

    @property
    def confidence_maps_key_name(self) -> Optional[Text]:
        head_key = self.config.model.heads.which_oneof_attrib_name()

        if head_key in ("multi_instance", "single_instance"):
            return "predicted_confidence_maps"

        if head_key == "centroid":
            return "predicted_centroid_confidence_maps"

        # todo: centered_instance

        return None

    @property
    def part_affinity_fields_key_name(self) -> Optional[Text]:
        head_key = self.config.model.heads.which_oneof_attrib_name()

        if head_key == "multi_instance":
            return "predicted_part_affinity_fields"

        return None

    def make_pipeline(self):
        pipeline = Pipeline()

        pipeline += Normalizer.from_config(self.config.data.preprocessing)
        pipeline += Resizer.from_config(
            self.config.data.preprocessing, keep_full_image=False, points_key=None
        )

        pipeline += KerasModelPredictor(
            keras_model=self.model.keras_model,
            model_input_keys="image",
            model_output_keys=self.head_specific_output_keys(),
        )

        self.pipeline = pipeline

    def predict_generator(self, data_provider: Provider):
        if self.pipeline is None:
            # Pass in data provider when mocking one of the models.
            self.make_pipeline()

        self.pipeline.providers = [data_provider]

        # Yield each example from dataset, catching and logging exceptions
        return safely_generate(self.pipeline.make_dataset())

    def predict(self, data_provider: Provider):
        generator = self.predict_generator(data_provider)
        examples = list(generator)

        return examples


class CentroidCropGroundTruth(tf.keras.layers.Layer):
    """Keras layer that simulates a centroid cropping model using ground truth.

    This layer is useful for testing and evaluating centered instance models.

    Attributes:
        crop_size: The length of the square box to extract around each centroid.
    """

    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def call(self, example_gt: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Return the ground truth instance crops.

        Args:
            example_gt: Dictionary generated from a labels pipeline with the keys:
                `"image": (batch_size, height, width, channels)`
                `"centroids": (batch_size, n_centroids, 2)`: The input centroids.
                    Axis 1 is expected to be ragged.
                These can be generated by the `InstanceCentroidFinder` transformer.

        Returns:
            Dictionary containing the output of the instance cropping layer with keys:
            `"crops": (batch_size, n_centroids, crop_size, crop_size, channels)`
            `"crop_offsets": (batch_size, n_centroids, crop_size, crop_size, channels)`
                These contain the top-left coordinates of each crop in the full images.
            `"centroids": (batch_size, n_centroids, 2)`
            `"centroid_vals": (batch_size, n_centroids)`

            Axis 1 of all keys are expected to be ragged.

            `"centroids"` are from the input example and `"centroid_vals"` will be
            filled with ones.
        """
        # Pull out data from example.
        full_imgs = example_gt["image"]
        crop_sample_inds = example_gt["centroids"].value_rowids()  # (n_peaks,)
        n_peaks = tf.shape(crop_sample_inds)[0]  # total number of peaks in the batch
        centroid_points = example_gt["centroids"].flat_values  # (n_peaks, 2)
        centroid_vals = tf.ones(tf.shape(centroid_points)[0])  # (n_peaks,)

        # Store crop offsets.
        crop_offsets = centroid_points - (self.crop_size / 2)

        # Crop instances around centroids.
        bboxes = sleap.nn.data.instance_cropping.make_centered_bboxes(
            centroid_points, self.crop_size, self.crop_size
        )
        crops = sleap.nn.peak_finding.crop_bboxes(full_imgs, bboxes, crop_sample_inds)

        # Reshape to (n_peaks, crop_height, crop_width, channels)
        img_channels = tf.shape(full_imgs)[3]
        crops = tf.reshape(
            crops, [n_peaks, self.crop_size, self.crop_size, img_channels]
        )

        # Group crops by sample.
        samples = tf.shape(full_imgs, out_type=tf.int64)[0]

        crops = tf.RaggedTensor.from_value_rowids(
            crops, crop_sample_inds, nrows=samples
        )
        crop_offsets = tf.RaggedTensor.from_value_rowids(
            crop_offsets, crop_sample_inds, nrows=samples
        )
        centroid_vals = tf.RaggedTensor.from_value_rowids(
            centroid_vals, crop_sample_inds, nrows=samples
        )

        return dict(
            crops=crops,
            crop_offsets=crop_offsets,
            centroids=example_gt["centroids"],
            centroid_vals=centroid_vals,
        )


class FindInstancePeaksGroundTruth(tf.keras.layers.Layer):
    """Keras layer that simulates a centered instance peaks model.

    This layer is useful for testing and evaluating centroid models."""

    def __init__(self):
        super().__init__()

    def call(
        self, example_gt: Dict[str, tf.Tensor], crop_output: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Return the ground truth instance peaks given a set of crops.

        Args:
            example_gt: Dictionary generated from a labels pipeline with the key:
                `"instances": (batch_size, n_instances_gt, n_nodes, 2)`
                    Axes 1 and 2 are expected to be ragged dimensions.
            crop_output: Dictionary containing the output of the instance cropping layer
                with keys:
                `"centroids": (batch_size, n_centroids, 2)`,
                `"centroid_vals": (batch_size, n_centroids)`
                    Axis 1 of both keys are expected to be ragged.

        Returns:
            A dictionary with the instance peaks for each frame. The peaks are just the
            ground truth instances matched to the crop output centroids via greedy
            matching of the closest node point to each centroid.

            The output will have keys:
                `"centroids": (batch_size, n_centroids, 2)`: The input centroids.
                `"centroid_vals": (batch_size, n_centroids)`: The input centroid
                    confidence values.
                `"instance_peaks": (batch_size, n_centroids, n_nodes, 2)`: The matched
                    instances.
                `"instance_peak_vals": (batch_size, n_centroids, n_nodes)`: Peak
                    confidence values (all 1.0).
        """
        # Compute pairwise distances between centroids and all instance points within
        # each sample.
        a = tf.expand_dims(
            example_gt["instances"].with_row_splits_dtype(tf.int64), axis=1
        )  # (batch_size, 1, n_insts, n_nodes, 2)
        a = a.to_tensor(default_value=tf.cast(np.NaN, tf.float32))
        b = tf.expand_dims(
            tf.expand_dims(crop_output["centroids"], axis=2), axis=2
        ).with_row_splits_dtype(
            tf.int64
        )  # (batch_size, n_centroids, 1, 1, 2)
        dists = a - b  # (batch_size, n_centroids, n_insts, n_nodes, 2)
        dists = tf.sqrt(tf.reduce_sum(dists ** 2, axis=-1))  # reduce over xy
        dists = tf.reduce_min(dists, axis=-1)  # reduce over nodes
        dists = dists.to_tensor(
            tf.cast(np.NaN, tf.float32)
        )  # (batch_size, n_centroids, n_insts)

        # Find nearest GT instance to each centroid.
        matches = tf.argmin(dists, axis=2)  # (batch_size, n_centroids)

        # Argmin will return indices for NaNs as well, so we must filter the matches.
        subs = tf.where(~tf.reduce_all(tf.math.is_nan(dists), axis=2))
        valid_matches = tf.gather_nd(matches, subs)
        match_sample_inds = tf.gather(subs, 0, axis=1)

        # Get the matched instances.
        instance_peaks = tf.gather_nd(
            example_gt["instances"],
            tf.stack([match_sample_inds, valid_matches], axis=1),
        )
        instance_peaks = tf.RaggedTensor.from_value_rowids(
            instance_peaks, match_sample_inds
        )  # (batch_size, n_centroids, n_nodes, 2)

        # Set all peak values to 1.
        instance_peak_vals = tf.gather(
            tf.ones_like(instance_peaks, dtype=tf.float32), 0, axis=-1
        )  # (batch_size, n_centroids, n_nodes)

        return dict(
            centroids=crop_output["centroids"],
            centroid_vals=crop_output["centroid_vals"],
            instance_peaks=instance_peaks,
            instance_peak_vals=instance_peak_vals,
        )


class InferenceLayer(tf.keras.layers.Layer):
    """Base layer for wrapping a Keras model into a layer with preprocessing.

    This layer is useful for wrapping input preprocessing operations that would
    otherwise be handled by a separate pipeline.

    This layer expects the same input as the model (rank-4 image) and automatically
    converts the input to a float if it is in integer form. This can help improve
    performance by enabling inference directly on `uint8` inputs.

    The `call()` method can be overloaded to create custom inference routines that
    take advantage of the `preprocess()` method.

    Attributes:
        keras_model: A `tf.keras.Model` that will be called on the input to this layer.
        input_scale: If not 1.0, input image will be resized by this factor.
        pad_to_stride: If not 1, input image will be paded to ensure that it is
            divisible by this value (after scaling).
        ensure_grayscale: If `True`, converts inputs to grayscale if not already. If
            `False`, converts inputs to RGB if not already. If `None` (default), infer
            from the shape of the input layer of the model.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        input_scale: float = 1.0,
        pad_to_stride: int = 1,
        ensure_grayscale: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keras_model = keras_model
        self.input_scale = input_scale
        self.pad_to_stride = pad_to_stride
        if ensure_grayscale is None:
            ensure_grayscale = self.keras_model.inputs[0].shape[-1] == 1
        self.ensure_grayscale = ensure_grayscale

    def preprocess(self, imgs: tf.Tensor) -> tf.Tensor:
        """Apply all preprocessing operations configured for this layer.

        Args:
            imgs: A batch of images as a tensor.

        Returns:
            The input tensor after applying preprocessing operations. The tensor will
            always be a `tf.float32`, which will be adjusted to the range `[0, 1]` if it
            was previously an integer.
        """
        if self.ensure_grayscale:
            imgs = sleap.nn.data.normalization.ensure_grayscale(imgs)
        else:
            imgs = sleap.nn.data.normalization.ensure_rgb(imgs)

        imgs = sleap.nn.data.normalization.ensure_float(imgs)

        if self.input_scale != 1.0:
            imgs = sleap.nn.data.resizing.resize_image(imgs, self.input_scale)

        if self.pad_to_stride > 1:
            imgs = sleap.nn.data.resizing.pad_to_stride(imgs, self.pad_to_stride)

        return imgs

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """Call the model with preprocessed data.

        Args:
            data: Inputs to the model.

        Returns:
            Output of the model after being called with preprocessing.
        """
        return self.keras_model(self.preprocess(data))


class InferenceModel(tf.keras.Model):
    """SLEAP inference model base class.

    This class wraps the `tf.keras.Model` class to provide SLEAP-specific inference
    utilities such as handling different input data types, preprocessing and variable
    output shapes.
    """

    def predict(
        self,
        data: Union[
            np.ndarray,
            tf.Tensor,
            Dict[str, tf.Tensor],
            tf.data.Dataset,
            Pipeline,
            sleap.Video,
        ],
        numpy: bool = True,
        batch_size: int = 4,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
        """Predict instances in the data.

        Args:
            data: Input data in any form. Possible types:
                - `np.ndarray`, `tf.Tensor`: Images of shape
                    `(samples, height, width, channels)`
                - `dict` with key `"image"` as a tensor
                - `tf.data.Dataset` that generates examples in one of the above formats.
                - `sleap.Pipeline` that generates examples in one of the above formats.
                - `sleap.Video` which will be converted into a pipeline that generates
                    batches of `batch_size` frames.
            numpy: If `True` (default), returned values will be converted to
                `np.ndarray`s or Python primitives if scalars.
            batch_size: Batch size to use for inference. No effect if using a dataset or
                pipeline as input since those are expected to generate batches.

        Returns:
            The model outputs as a dictionary of (potentially ragged) tensors or numpy
            arrays if `numpy` is `True`.

            If `numpy` is `False`, values of the dictionary may be `tf.RaggedTensor`s
            with the same length for axis 0 (samples), but variable length axis 1
            (instances).

            If `numpy` is `True` and the output contained ragged tensors, they will be
            NaN-padded to the bounding shape and an additional key `"n_valid"` will be
            included to indicate the number of valid elements (before padding) in axis
            1 of the tensors.
        """
        if isinstance(data, (sleap.Video, sleap.Labels)):
            data = data.to_pipeline(batch_size=batch_size)
        if isinstance(data, Pipeline):
            data = data.make_dataset()

        outs = super().predict(data, batch_size=batch_size, **kwargs)

        if numpy:
            for v in outs.values():
                if isinstance(v, tf.RaggedTensor):
                    outs["n_valid"] = v.row_lengths()
                    break
            outs = sleap.nn.data.utils.unrag_example(outs, numpy=True)
        return outs


def get_model_output_stride(
    model: tf.keras.Model, input_ind: int = 0, output_ind: int = -1
) -> int:
    """Return the stride (1/scale) of the model outputs relative to the input.

    Args:
        model: A `tf.keras.Model`.
        input_ind: The index of the input to use as reference. Defaults to 0, indicating
            the first input for multi-output models.
        output_ind: The index of the output to compute the stride for. Defaults to -1,
            indicating the last output for multi-output models.

    Returns:
        The output stride of the model computed as the integer ratio of the input's
        height relative to the output's height, e.g., for a single input/output model:

            `model.input.shape[1] // model.output.shape[1]`

        Raises a warning if the shapes do not divide evenly.
    """
    size_in = model.inputs[input_ind].shape[1]
    size_out = model.outputs[output_ind].shape[1]
    if size_in % size_out != 0:
        warnings.warn(
            f"Model input of shape {model.inputs[input_ind].shape} does not divide "
            f"evenly with output of shape {model.outputs[output_ind].shape}."
        )
    return size_in // size_out


def find_head(model: tf.keras.Model, name: str) -> Optional[int]:
    """Return the index of a head in a model's outputs.

    Args:
        model: A `tf.keras.Model` trained by SLEAP.
        name: A string that is contained in the model output tensor name.

    Returns:
        The index of the first output with a matched name or `None` if none were found.

    Notes:
        SLEAP model heads are named:
        - `"SingleInstanceConfmapsHead"`
        - `"CentroidConfmapsHead"`
        - `"CenteredInstanceConfmapsHead"`
        - `"MultiInstanceConfmapsHead"`
        - `"PartAffinityFieldsHead"`
        - `"OffsetRefinementHead"`
    """
    for i, head_name in enumerate(model.output_names):
        if name in head_name:
            return i
    return None


class SingleInstanceInferenceLayer(InferenceLayer):
    """Inference layer for applying single instance models.

    This layer encapsulates all of the inference operations requires for generating
    predictions from a single instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            single instance confidence maps.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        pad_to_stride: If not 1, input image will be paded to ensure that it is
            divisible by this value (after scaling). This should be set to the max
            stride of the model.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since the
            data must be copied off of the GPU, but is useful for visualizing the raw
            output of the model.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"SingleInstanceConfmapsHead"` in its name.
        offsets_ind: Index of the output tensor of the model corresponding to
            offset regression maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"OffsetRefinementHead"` in its name. If the head is not present, the method
            specified in the `refinement` attribute will be used.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        input_scale: float = 1.0,
        pad_to_stride: int = 1,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        confmaps_ind: Optional[int] = None,
        offsets_ind: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            keras_model=keras_model,
            input_scale=input_scale,
            pad_to_stride=pad_to_stride,
            **kwargs,
        )
        self.confmaps_ind = confmaps_ind
        self.offsets_ind = offsets_ind
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.output_stride = output_stride
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(
                self.keras_model, "SingleInstanceConfmapsHead"
            )

        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'SingleInstanceConfmapsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if self.output_stride is None:
            # Attempt to automatically infer the output stride.
            self.output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.confmaps_ind
            )

    def call(self, data):
        """Predict instance confidence maps and find peaks.

        Args:
            inputs: Full frame images as a `tf.Tensor` of shape
                `(samples, height, width, channels)` or a dictionary with key:
                `"image"`: Full frame images in the same format as above.

        Returns:
            A dictionary of outputs grouped by sample with keys:

            `"peaks"`: The predicted peaks of shape `(samples, nodes, 2)`.
            `"peak_vals": The peak confidence values of shape `(samples, nodes)`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"confmaps"` containing a `tf.Tensor` of shape
            `(samples, output_height, output_width, 1)` containing the confidence maps
            predicted by the model.
        """
        if isinstance(data, dict):
            imgs = data["image"]
        else:
            imgs = data
        imgs = self.preprocess(imgs)
        preds = self.keras_model(imgs)
        offsets = None
        if isinstance(preds, list):
            cms = preds[self.confmaps_ind]
            if self.offsets_ind is not None:
                offsets = preds[self.offsets_ind]
        else:
            cms = preds
        if self.offsets_ind is None:
            peaks, peak_vals = sleap.nn.peak_finding.find_global_peaks(
                cms,
                threshold=self.peak_threshold,
                refinement=self.refinement,
                integral_patch_size=self.integral_patch_size,
            )
        else:
            peaks, peak_vals = sleap.nn.peak_finding.find_global_peaks_with_offsets(
                cms,
                offsets,
                threshold=self.peak_threshold,
            )

        # Adjust for stride and scale.
        peaks = peaks * self.output_stride
        if self.input_scale != 1.0:
            # Note: We add 0.5 here to offset TensorFlow's weird image resizing. This
            # may not always(?) be the most correct approach.
            # See: https://github.com/tensorflow/tensorflow/issues/6720
            peaks = (peaks / self.input_scale) + 0.5

        out = {"peaks": peaks, "peak_vals": peak_vals}
        if self.return_confmaps:
            out["confmaps"] = cms
        return out


class SingleInstanceInferenceModel(InferenceModel):
    """Single instance prediction model.

    This model encapsulates the basic single instance approach where it is assumed that
    there is only one instance in the frame. The images are passed to a peak detector
    which is trained to detect all body parts for the instance assuming a single peak
    per body part.

    Attributes:
        single_instance_layer: A single instance instance peak detection layer. This
            layer takes as input full images and outputs the detected peaks.
    """

    def __init__(self, single_instance_layer, **kwargs):
        super().__init__(**kwargs)
        self.single_instance_layer = single_instance_layer

    def call(self, example):
        """Predict instances for one batch of images.

        Args:
            example: This may be either a single batch of images as a 4-D tensor of
                shape `(batch_size, height, width, channels)`, or a dictionary
                containing the image batch in the `"images"` key.

        Returns:
            The predicted instances as a dictionary of tensors with keys:

            `"peaks": (batch_size, n_nodes, 2)`: Instance skeleton points.
            `"peak_vals": (batch_size, n_instances, n_nodes)`: Confidence values for the
                instance skeleton points.
        """
        return self.single_instance_layer(example)


@attr.s(auto_attribs=True)
class SingleInstancePredictor(Predictor):
    """Single instance predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained single instance SLEAP model.

    This should be initialized using the `from_trained_models()` constructor or the
    high-level API (`sleap.load_model`).

    Attributes:
        confmap_config: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained model.
        confmap_model: A `sleap.nn.model.Model` instance created from the trained model.
        inference_model: A `sleap.nn.inference.SingleInstanceInferenceModel` that wraps
            a trained `tf.keras.Model` to implement preprocessing and peak finding.
        pipeline: A `sleap.nn.data.Pipeline` that loads the data and batches input data.
            This will be updated dynamically if new data sources are used.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        integral_refinement: If `True`, peaks will be refined with integral regression.
            If `False`, `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        batch_size: The default batch size to use when loading data for inference.
            Higher values increase inference speed at the cost of higher memory usage.
    """

    confmap_config: TrainingJobConfig
    confmap_model: Model
    inference_model: Optional[SingleInstanceInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2
    integral_refinement: bool = True
    integral_patch_size: int = 5
    batch_size: int = 4

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained model and configuration."""
        self.inference_model = SingleInstanceInferenceModel(
            SingleInstanceInferenceLayer(
                keras_model=self.confmap_model.keras_model,
                input_scale=self.confmap_config.data.preprocessing.input_scaling,
                pad_to_stride=self.confmap_model.maximum_stride,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
            )
        )

    @classmethod
    def from_trained_models(
        cls,
        model_path: Text,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
        batch_size: int = 4,
    ) -> "SingleInstancePredictor":
        """Create the predictor from a saved model.

        Args:
            model_path: Path to a model folder or training job JSON file inside a model
                folder. This folder should contain `training_config.json` and
                `best_model.h5` files for a trained model.
            peak_threshold: Minimum confidence map value to consider a global peak as
                valid.
            integral_refinement: If `True`, peaks will be refined with integral
                regression. If `False`, `"local"`, peaks will be refined with quarter
                pixel local gradient offset. This has no effect if the model has an
                offset regression head.
            integral_patch_size: Size of patches to crop around each rough peak for
                integral refinement as an integer scalar.
            batch_size: The default batch size to use when loading data for inference.
                Higher values increase inference speed at the cost of higher memory
                usage.

        Returns:
            An instance of`SingleInstancePredictor` with the models loaded.
        """
        # Load confmap model.
        confmap_config = TrainingJobConfig.load_json(model_path)
        confmap_keras_model_path = get_keras_model_path(model_path)
        confmap_model = Model.from_config(confmap_config.model)
        confmap_model.keras_model = tf.keras.models.load_model(
            confmap_keras_model_path, compile=False
        )
        obj = cls(
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
        )
        obj._initialize_inference_model()
        return obj

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:
        """Make a data loading pipeline.

        Args:
            data_provider: If not `None`, the pipeline will be created with an instance
                of a `sleap.pipelines.Provider`.

        Returns:
            The created `sleap.pipelines.Pipeline` with batching and prefetching.

        Notes:
            This method also updates the class attribute for the pipeline and will be
            called automatically when predicting on data from a new source.
        """
        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        pipeline += sleap.nn.data.pipelines.Batcher(
            batch_size=self.batch_size, drop_remainder=False, unrag=False
        )

        pipeline += Prefetcher()

        self.pipeline = pipeline

        return pipeline

    def _predict_generator(
        self, data_provider: Provider
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating or updating the input `sleap.pipelines.Pipeline`
        for loading the data, as well as looping over the batches and running inference.

        Args:
            data_provider: The `sleap.pipelines.Provider` that contains data that should
                be used for inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        if self.pipeline is None:
            self.make_pipeline()
        if self.inference_model is None:
            self._initialize_inference_model()

        # Update the data provider source.
        self.pipeline.providers = [data_provider]

        # Loop over data batches.
        for ex in self.pipeline.make_dataset():
            # Run inference on current batch.
            preds = self.inference_model.predict(ex)

            ex["peaks"] = preds["peaks"]
            ex["peak_vals"] = preds["peak_vals"]

            # Convert to numpy arrays if not already.
            if isinstance(ex["video_ind"], tf.Tensor):
                ex["video_ind"] = ex["video_ind"].numpy().flatten()
            if isinstance(ex["frame_ind"], tf.Tensor):
                ex["frame_ind"] = ex["frame_ind"].numpy().flatten()
            
            yield ex

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
    ) -> List[sleap.LabeledFrame]:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"video_ind"`, `"frame_ind"`,
                `"peaks"`, and `"peak_vals"`. This can be created using the
                `_predict_generator()` method.
            data_provider: The `sleap.pipelines.Provider` that the predictions are being
                created from. This is used to retrieve the `sleap.Video` instance
                associated with each inference result.

        Returns:
            A list of `sleap.LabeledFrame`s with `sleap.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        skeleton = self.confmap_config.data.labels.skeletons[0]

        # Loop over batches.
        predicted_frames = []
        for ex in generator:

            # Loop over frames.
            for video_ind, frame_ind, points, confidences in zip(
                ex["video_ind"], ex["frame_ind"], ex["peaks"], ex["peak_vals"]
            ):
                # Loop over instances.
                predicted_instances = [
                    sleap.instance.PredictedInstance.from_arrays(
                        points=points,
                        point_confidences=confidences,
                        instance_score=np.nansum(confidences),
                        skeleton=skeleton,
                    )
                ]

                predicted_frames.append(
                    sleap.LabeledFrame(
                        video=data_provider.videos[video_ind],
                        frame_idx=frame_ind,
                        instances=predicted_instances,
                    )
                )

        return predicted_frames

    def predict(
        self,
        data: Union[Provider, sleap.Labels, sleap.Video],
        make_labels: bool = True,
    ) -> Union[List[Dict[str, np.ndarray]], sleap.Labels]:
        """Run inference on a data source.

        Args:
            data: A `sleap.pipelines.Provider`, `sleap.Labels` or `sleap.Video` to
                run inference over.
            make_labels: If `True` (the default), returns a `sleap.Labels` instance with
                `sleap.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.

        Returns:
            A `sleap.Labels` with `sleap.PredictedInstance`s if `make_labels` is `True`,
            otherwise a list of dictionaries containing batches of numpy arrays with the
            raw results.
        """
        # Create provider if necessary.
        if isinstance(data, sleap.Labels):
            data = LabelsReader(data)
        elif isinstance(data, sleap.Video):
            data = VideoReader(data)

        # Initialize inference loop generator.
        generator = self._predict_generator(data)

        if make_labels:
            # Create SLEAP data structures while consuming results.
            return sleap.Labels(
                self._make_labeled_frames_from_generator(generator, data)
            )
        else:
            # Just return the raw results.
            return list(generator)


class CentroidCrop(InferenceLayer):
    """Inference layer for applying centroid crop-based models.

    This layer encapsulates all of the inference operations requires for generating
    predictions from a centroid confidence map model. This includes preprocessing,
    model forward pass, peak finding, coordinate adjustment and cropping.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centroid/anchor confidence maps.
        crop_size: Integer scalar specifying the height/width of the centered crops.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        pad_to_stride: If not 1, input image will be paded to ensure that it is
            divisible by this value (after scaling). This should be set to the max
            stride of the model.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since the
            data must be copied off of the GPU, but is useful for visualizing the raw
            output of the model.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"CentroidConfmapsHead"` in its name.
        offsets_ind: Index of the output tensor of the model corresponding to
            offset regression maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"OffsetRefinementHead"` in its name. If the head is not present, the method
            specified in the `refinement` attribute will be used.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        crop_size: int,
        input_scale: float = 1.0,
        pad_to_stride: int = 1,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        confmaps_ind: Optional[int] = None,
        offsets_ind: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            keras_model=keras_model,
            input_scale=input_scale,
            pad_to_stride=pad_to_stride,
            **kwargs,
        )

        self.crop_size = crop_size

        self.confmaps_ind = confmaps_ind
        self.offsets_ind = offsets_ind

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(self.keras_model, "CentroidConfmapsHead")

        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'CentroidConfmapsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if output_stride is None:
            # Attempt to automatically infer the output stride.
            output_stride = get_model_output_stride(
                self.keras_model, 0, self.confmaps_ind
            )
        self.output_stride = output_stride
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps

    def call(self, inputs):
        """Predict centroid confidence maps and crop around peaks.

        This layer can be chained with a `FindInstancePeaks` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Full frame images as a `tf.Tensor` of shape
                `(samples, height, width, channels)` or a dictionary with key:
                `"image"`: Full frame images in the same format as above.

        Returns:
            A dictionary of outputs grouped by sample with keys:

            `"crops"`: Cropped images of shape
                `(samples, ?, crop_size, crop_size, channels)`.
            `"crop_offsets"`: Coordinates of the top-left of the crops as `(x, y)`
                offsets of shape `(samples, ?, 2)` for adjusting the predicted peak
                coordinates.
            `"centroids"`: The predicted centroids of shape `(samples, ?, 2)`.
            `"centroid_vals": The centroid confidence values of shape `(samples, ?)`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"centroid_confmaps"` containing a `tf.RaggedTensor` of
            shape `(samples, ?, output_height, output_width, 1)` containing the
            confidence maps predicted by the model.
        """
        if isinstance(inputs, dict):
            # Pull out image from example dictionary.
            imgs = inputs["image"]
        else:
            # Assume inputs are image tensors.
            imgs = inputs

        # Store full images for cropping.
        full_imgs = imgs

        # Preprocess inputs (scaling, padding, colorspace, int to float).
        imgs = self.preprocess(imgs)

        # Predict confidence maps.
        out = self.keras_model(imgs)
        offsets = None
        if isinstance(out, list):
            cms = out[self.confmaps_ind]
            if self.offsets_ind is not None:
                offsets = out[self.offsets_ind]
        else:
            cms = out

        # Find centroids peaks.
        if self.offsets_ind is None:
            # Use deterministic refinement.
            (
                centroid_points,
                centroid_vals,
                crop_sample_inds,
                _,
            ) = sleap.nn.peak_finding.find_local_peaks(
                cms,
                threshold=self.peak_threshold,
                refinement=self.refinement,
                integral_patch_size=self.integral_patch_size,
            )
        else:
            # Use learned offsets.
            (
                centroid_points,
                centroid_vals,
                crop_sample_inds,
                _,
            ) = sleap.nn.peak_finding.find_local_peaks_with_offsets(
                cms,
                offsets,
                threshold=self.peak_threshold,
            )

        # Adjust for stride and scale.
        centroid_points = centroid_points * self.output_stride
        if self.input_scale != 1.0:
            # Note: We add 0.5 here to offset TensorFlow's weird image resizing. This
            # may not always(?) be the most correct approach.
            # See: https://github.com/tensorflow/tensorflow/issues/6720
            centroid_points = (centroid_points / self.input_scale) + 0.5

        # Store crop offsets.
        crop_offsets = centroid_points - (self.crop_size / 2)

        n_peaks = tf.shape(centroid_points)[0]
        if n_peaks > 0:
            # Crop instances around centroids.
            bboxes = sleap.nn.data.instance_cropping.make_centered_bboxes(
                centroid_points, self.crop_size, self.crop_size
            )
            crops = sleap.nn.peak_finding.crop_bboxes(
                full_imgs, bboxes, crop_sample_inds
            )

            # Reshape to (n_peaks, crop_height, crop_width, channels)
            crops = tf.reshape(
                crops, [n_peaks, self.crop_size, self.crop_size, full_imgs.shape[3]]
            )
        else:
            # No peaks found, so just create a placeholder stack.
            crops = tf.zeros(
                [n_peaks, self.crop_size, self.crop_size, full_imgs.shape[3]],
                dtype=full_imgs.dtype,
            )

        # Group crops by sample (samples, ?, ...).
        samples = tf.shape(imgs)[0]
        centroids = tf.RaggedTensor.from_value_rowids(
            centroid_points, crop_sample_inds, nrows=samples
        )
        crops = tf.RaggedTensor.from_value_rowids(
            crops, crop_sample_inds, nrows=samples
        )
        crop_offsets = tf.RaggedTensor.from_value_rowids(
            crop_offsets, crop_sample_inds, nrows=samples
        )
        centroid_vals = tf.RaggedTensor.from_value_rowids(
            centroid_vals, crop_sample_inds, nrows=samples
        )

        outputs = dict(
            centroids=centroids,
            centroid_vals=centroid_vals,
            crops=crops,
            crop_offsets=crop_offsets,
        )
        if self.return_confmaps:
            # Return confidence maps with outputs.
            cms = tf.RaggedTensor.from_value_rowids(
                cms, crop_sample_inds, nrows=samples
            )
            outputs["centroid_confmaps"] = cms
        return outputs


class FindInstancePeaks(InferenceLayer):
    """Keras layer that predicts instance peaks from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted peaks. This will result in slower inference times since the
            data must be copied off of the GPU, but is useful for visualizing the raw
            output of the model.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"CenteredInstanceConfmapsHead"` in its name.
        offsets_ind: Index of the output tensor of the model corresponding to
            offset regression maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"OffsetRefinementHead"` in its name. If the head is not present, the method
            specified in the `refinement` attribute will be used.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        input_scale: float = 1.0,
        output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        confmaps_ind: Optional[int] = None,
        offsets_ind: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            keras_model=keras_model, input_scale=input_scale, pad_to_stride=1, **kwargs
        )
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.confmaps_ind = confmaps_ind
        self.offsets_ind = offsets_ind

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(
                self.keras_model, "CenteredInstanceConfmapsHead"
            )

        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'CenteredInstanceConfmapsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if output_stride is None:
            # Attempt to automatically infer the output stride.
            output_stride = get_model_output_stride(
                self.keras_model, 0, self.confmaps_ind
            )
        self.output_stride = output_stride

    def call(
        self, inputs: Union[Dict[str, tf.Tensor], tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Predict confidence maps and infer peak coordinates.

        This layer can be chained with a `CentroidCrop` layer to create a top-down
        inference function from full images.

        Args:
            inputs: Instance-centered images as a `tf.Tensor` of shape
                `(samples, height, width, channels)` or `tf.RaggedTensor` of shape
                `(samples, ?, height, width, channels)` where images are grouped by
                sample and may contain a variable number of crops, or a dictionary with
                keys:
                `"crops"`: Cropped images in either format above.
                `"crop_offsets"`: (Optional) Coordinates of the top-left of the crops as
                    `(x, y)` offsets of shape `(samples, ?, 2)` for adjusting the
                    predicted peak coordinates. No adjustment is performed if not
                    provided.
                `"centroids"`: (Optional) If provided, will be passed through to the
                    output.
                `"centroid_vals"`: (Optional) If provided, will be passed through to the
                    output.

        Returns:
            A dictionary of outputs with keys:

            `"instance_peaks"`: The predicted peaks for each instance in the batch as a
                `tf.RaggedTensor` of shape `(samples, ?, nodes, 2)`.
            `"instance_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `tf.RaggedTensor` of shape
                `(samples, ?, nodes)`.

            If provided (e.g., from an input `CentroidCrop` layer), the centroids that
            generated the crops will also be included in the keys `"centroids"` and
            `"centroid_vals"`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"instance_confmaps"` containing a `tf.RaggedTensor` of
            shape `(samples, ?, output_height, output_width, nodes)` containing the
            confidence maps predicted by the model.
        """
        if isinstance(inputs, dict):
            crops = inputs["crops"]
        else:
            # Tensor input provided. We'll infer the extra fields in the expected input
            # dictionary.
            crops = inputs
            inputs = {}

        if isinstance(crops, tf.RaggedTensor):
            crops = inputs["crops"]  # (samples, ?, height, width, channels)

            # Flatten crops into (n_peaks, height, width, channels)
            crop_sample_inds = crops.value_rowids()  # (n_peaks,)
            samples = crops.nrows()
            crops = crops.merge_dims(0, 1)

        else:
            if "crop_sample_inds" in inputs:
                # Crops provided as a regular tensor, use the metadata are in the input.
                samples = inputs["samples"]
                crop_sample_inds = inputs["crop_sample_inds"]
            else:
                # Assuming crops is (samples, height, width, channels).
                samples = tf.shape(crops)[0]
                crop_sample_inds = tf.range(samples, dtype=tf.int32)

        # Preprocess inputs (scaling, padding, colorspace, int to float).
        crops = self.preprocess(crops)

        # Network forward pass.
        out = self.keras_model(crops)

        # Sort outputs.
        offsets = None
        if isinstance(out, list):
            cms = out[self.confmaps_ind]
            if self.offsets_ind is not None:
                offsets = out[self.offsets_ind]
        else:
            # Assume confidence maps if single output.
            cms = out

        # Find peaks.
        if self.offsets_ind is None:
            # Use deterministic refinement.
            peak_points, peak_vals = sleap.nn.peak_finding.find_global_peaks(
                cms,
                threshold=self.peak_threshold,
                refinement=self.refinement,
                integral_patch_size=self.integral_patch_size,
            )
        else:
            # Use learned offsets.
            (
                peak_points,
                peak_vals,
            ) = sleap.nn.peak_finding.find_global_peaks_with_offsets(
                cms,
                offsets,
                threshold=self.peak_threshold,
            )

        # Adjust for stride and scale.
        peak_points = peak_points * self.output_stride
        if self.input_scale != 1.0:
            # Note: We add 0.5 here to offset TensorFlow's weird image resizing. This
            # may not always(?) be the most correct approach.
            # See: https://github.com/tensorflow/tensorflow/issues/6720
            peak_points = (peak_points / self.input_scale) + 0.5

        # Adjust for crop offsets if provided.
        if "crop_offsets" in inputs:
            # Flatten (samples, ?, 2) -> (n_peaks, 2).
            crop_offsets = inputs["crop_offsets"].merge_dims(0, 1)
            peak_points = peak_points + tf.expand_dims(crop_offsets, axis=1)

        # Group peaks by sample (samples, ?, nodes, 2).
        peaks = tf.RaggedTensor.from_value_rowids(
            peak_points, crop_sample_inds, nrows=samples
        )
        peak_vals = tf.RaggedTensor.from_value_rowids(
            peak_vals, crop_sample_inds, nrows=samples
        )

        # Build outputs.
        outputs = {"instance_peaks": peaks, "instance_peak_vals": peak_vals}
        if "centroids" in inputs:
            outputs["centroids"] = inputs["centroids"]
        if "centroids" in inputs:
            outputs["centroid_vals"] = inputs["centroid_vals"]
        if self.return_confmaps:
            cms = tf.RaggedTensor.from_value_rowids(
                cms, crop_sample_inds, nrows=samples
            )
            outputs["instance_confmaps"] = cms
        return outputs


class TopDownInferenceModel(InferenceModel):
    """Top-down instance prediction model.

    This model encapsulates the top-down approach where instances are first detected by
    local peak detection of an anchor point and then cropped. These instance-centered
    crops are then passed to an instance peak detector which is trained to detect all
    remaining body parts for the instance that is centered within the crop.

    Attributes:
        centroid_crop: A centroid cropping layer. This can be either `CentroidCrop` or
            `CentroidCropGroundTruth`. This layer takes the full image as input and
            outputs a set of centroids and cropped boxes.
        instance_peaks: A instance peak detection layer. This can be either
            `FindInstancePeaks` or `FindInstancePeaksGroundTruth`. This layer takes as
            input the output of the centroid cropper and outputs the detected peaks for
            the instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, CentroidCropGroundTruth],
        instance_peaks: Union[FindInstancePeaks, FindInstancePeaksGroundTruth],
    ):
        super().__init__()
        self.centroid_crop = centroid_crop
        self.instance_peaks = instance_peaks

    def call(
        self, example: Union[Dict[str, tf.Tensor], tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Predict instances for one batch of images.

        Args:
            example: This may be either a single batch of images as a 4-D tensor of
                shape `(batch_size, height, width, channels)`, or a dictionary
                containing the image batch in the `"images"` key. If using a ground
                truth model for either centroid cropping or instance peaks, the full
                example from a `Pipeline` is required for providing the metadata.

        Returns:
            The predicted instances as a dictionary of tensors with keys:

            `"centroids": (batch_size, n_instances, 2)`: Instance centroids.
            `"centroid_vals": (batch_size, n_instances)`: Instance centroid confidence
                values.
            `"instance_peaks": (batch_size, n_instances, n_nodes, 2)`: Instance skeleton
                points.
            `"instance_peak_vals": (batch_size, n_instances, n_nodes)`: Confidence
                values for the instance skeleton points.
        """
        if isinstance(example, tf.Tensor):
            example = dict(image=example)

        crop_output = self.centroid_crop(example)

        if isinstance(self.instance_peaks, FindInstancePeaksGroundTruth):
            peaks_output = self.instance_peaks(example, crop_output)
        else:
            peaks_output = self.instance_peaks(crop_output)
        return peaks_output


@attr.s(auto_attribs=True)
class TopdownPredictor(Predictor):
    """Top-down multi-instance predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained top-down multi-instance SLEAP model.

    This should be initialized using the `from_trained_models()` constructor or the
    high-level API (`sleap.load_model`).

    Attributes:
        centroid_config: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained centroid model. If `None`, ground truth centroids will be
            used if available from the data source.
        centroid_model: A `sleap.nn.model.Model` instance created from the trained
            centroid model. If `None`, ground truth centroids will be used if available
            from the data source.
        confmap_config: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained centered instance model. If `None`, ground truth instances
            will be used if available from the data source.
        confmap_model: A `sleap.nn.model.Model` instance created from the trained
            centered-instance model. If `None`, ground truth instances will be used if
            available from the data source.
        inference_model: A `sleap.nn.inference.TopDownInferenceModel` that wraps a
            trained `tf.keras.Model` to implement preprocessing, centroid detection,
            cropping and peak finding.
        pipeline: A `sleap.nn.data.Pipeline` that loads the data and batches input data.
            This will be updated dynamically if new data sources are used.
        tracker: A `sleap.nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.
        batch_size: The default batch size to use when loading data for inference.
            Higher values increase inference speed at the cost of higher memory usage.
        peak_threshold: Minimum confidence map value to consider a local peak as valid.
        integral_refinement: If `True`, peaks will be refined with integral regression.
            If `False`, `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
    """

    centroid_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    centroid_model: Optional[Model] = attr.ib(default=None)
    confmap_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    confmap_model: Optional[Model] = attr.ib(default=None)
    inference_model: Optional[TopDownInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    tracker: Optional[Tracker] = attr.ib(default=None, init=False)
    batch_size: int = 4
    peak_threshold: float = 0.2
    integral_refinement: bool = True
    integral_patch_size: int = 5

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        use_gt_centroid = self.centroid_config is None
        use_gt_confmap = self.confmap_config is None

        if use_gt_centroid:
            centroid_crop_layer = CentroidCropGroundTruth(
                crop_size=self.confmap_config.data.instance_cropping.crop_size
            )
        else:
            if use_gt_confmap:
                crop_size = 1
            else:
                crop_size = self.confmap_config.data.instance_cropping.crop_size
            centroid_crop_layer = CentroidCrop(
                keras_model=self.centroid_model.keras_model,
                crop_size=crop_size,
                input_scale=self.centroid_config.data.preprocessing.input_scaling,
                pad_to_stride=self.centroid_config.data.preprocessing.pad_to_stride,
                output_stride=self.centroid_config.model.heads.centroid.output_stride,
                peak_threshold=self.peak_threshold,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
                return_confmaps=False,
            )

        if use_gt_confmap:
            instance_peaks_layer = FindInstancePeaksGroundTruth()
        else:
            cfg = self.confmap_config
            instance_peaks_layer = FindInstancePeaks(
                keras_model=self.confmap_model.keras_model,
                input_scale=cfg.data.preprocessing.input_scaling,
                peak_threshold=self.peak_threshold,
                output_stride=cfg.model.heads.centered_instance.output_stride,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
                return_confmaps=False,
            )

        self.inference_model = TopDownInferenceModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer
        )

    @classmethod
    def from_trained_models(
        cls,
        centroid_model_path: Optional[Text] = None,
        confmap_model_path: Optional[Text] = None,
        batch_size: int = 4,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
    ) -> "TopdownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_model_path: Path to a centroid model folder or training job JSON
                file inside a model folder. This folder should contain
                `training_config.json` and `best_model.h5` files for a trained model.
            confmap_model_path: Path to a centered instance model folder or training job
                JSON file inside a model folder. This folder should contain
                `training_config.json` and `best_model.h5` files for a trained model.
            batch_size: The default batch size to use when loading data for inference.
                Higher values increase inference speed at the cost of higher memory
                usage.
            peak_threshold: Minimum confidence map value to consider a local peak as
                valid.
            integral_refinement: If `True`, peaks will be refined with integral
                regression. If `False`, `"local"`, peaks will be refined with quarter
                pixel local gradient offset. This has no effect if the model has an
                offset regression head.
            integral_patch_size: Size of patches to crop around each rough peak for
                integral refinement as an integer scalar.

        Returns:
            An instance of `TopdownPredictor` with the loaded models.

            One of the two models can be left as `None` to perform inference with ground
            truth data. This will only work with `LabelsReader` as the provider.
        """
        if centroid_model_path is None and confmap_model_path is None:
            raise ValueError(
                "Either the centroid or topdown confidence map model must be provided."
            )

        if centroid_model_path is not None:
            # Load centroid model.
            centroid_config = TrainingJobConfig.load_json(centroid_model_path)
            centroid_keras_model_path = get_keras_model_path(centroid_model_path)
            centroid_model = Model.from_config(centroid_config.model)
            centroid_model.keras_model = tf.keras.models.load_model(
                centroid_keras_model_path, compile=False
            )
        else:
            centroid_config = None
            centroid_model = None

        if confmap_model_path is not None:
            # Load confmap model.
            confmap_config = TrainingJobConfig.load_json(confmap_model_path)
            confmap_keras_model_path = get_keras_model_path(confmap_model_path)
            confmap_model = Model.from_config(confmap_config.model)
            confmap_model.keras_model = tf.keras.models.load_model(
                confmap_keras_model_path, compile=False
            )
        else:
            confmap_config = None
            confmap_model = None

        obj = cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            batch_size=batch_size,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
        )
        obj._initialize_inference_model()
        return obj

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:
        """Make a data loading pipeline.

        Args:
            data_provider: If not `None`, the pipeline will be created with an instance
                of a `sleap.pipelines.Provider`.

        Returns:
            The created `sleap.pipelines.Pipeline` with batching and prefetching.

        Notes:
            This method also updates the class attribute for the pipeline and will be
            called automatically when predicting on data from a new source.
        """
        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        if self.centroid_model is None:
            anchor_part = self.confmap_config.data.instance_cropping.center_on_part
            pipeline += sleap.nn.data.pipelines.InstanceCentroidFinder(
                center_on_anchor_part=anchor_part is not None,
                anchor_part_names=anchor_part,
                skeletons=self.confmap_config.data.labels.skeletons,
            )

        pipeline += sleap.nn.data.pipelines.Batcher(
            batch_size=self.batch_size, drop_remainder=False, unrag=False
        )

        pipeline += Prefetcher()

        self.pipeline = pipeline

        return pipeline

    def _predict_generator(
        self, data_provider: Provider
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating or updating the input `sleap.pipelines.Pipeline`
        for loading the data, as well as looping over the batches and running inference.

        Args:
            data_provider: The `sleap.pipelines.Provider` that contains data that should
                be used for inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        if self.pipeline is None:
            if self.centroid_config is not None and self.confmap_config is not None:
                self.make_pipeline()
            else:
                # Pass in data provider when mocking one of the models.
                self.make_pipeline(data_provider=data_provider)
        if self.inference_model is None:
            self._initialize_inference_model()

        # Update the data provider source.
        self.pipeline.providers = [data_provider]

        # Loop over data batches.
        for ex in self.pipeline.make_dataset():
            # Run inference on current batch.
            preds = self.inference_model.predict(ex)

            # Crop possibly variable length results.
            ex["instance_peaks"] = [
                x[:n] for x, n in zip(preds["instance_peaks"], preds["n_valid"])
            ]
            ex["instance_peak_vals"] = [
                x[:n] for x, n in zip(preds["instance_peak_vals"], preds["n_valid"])
            ]
            ex["centroids"] = [
                x[:n] for x, n in zip(preds["centroids"], preds["n_valid"])
            ]
            ex["centroid_vals"] = [
                x[:n] for x, n in zip(preds["centroid_vals"], preds["n_valid"])
            ]

            # Convert to numpy arrays if not already.
            if isinstance(ex["video_ind"], tf.Tensor):
                ex["video_ind"] = ex["video_ind"].numpy().flatten()
            if isinstance(ex["frame_ind"], tf.Tensor):
                ex["frame_ind"] = ex["frame_ind"].numpy().flatten()

            yield ex

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
    ) -> List[sleap.LabeledFrame]:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and runs
        them through the tracker if it is specified.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"image"`, `"video_ind"`,
                `"frame_ind"`, `"instance_peaks"`, `"instance_peak_vals"`, and
                `"centroid_vals"`. This can be created using the `_predict_generator()`
                method.
            data_provider: The `sleap.pipelines.Provider` that the predictions are being
                created from. This is used to retrieve the `sleap.Video` instance
                associated with each inference result.

        Returns:
            A list of `sleap.LabeledFrame`s with `sleap.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        if self.confmap_config is not None:
            skeleton = self.confmap_config.data.labels.skeletons[0]
        else:
            skeleton = self.centroid_config.data.labels.skeletons[0]

        # Loop over batches.
        predicted_frames = []
        for ex in generator:

            # Loop over frames.
            for image, video_ind, frame_ind, points, confidences, scores in zip(
                ex["image"],
                ex["video_ind"],
                ex["frame_ind"],
                ex["instance_peaks"],
                ex["instance_peak_vals"],
                ex["centroid_vals"],
            ):

                # Loop over instances.
                predicted_instances = []
                for pts, confs, score in zip(points, confidences, scores):
                    predicted_instances.append(
                        sleap.instance.PredictedInstance.from_arrays(
                            points=pts,
                            point_confidences=confs,
                            instance_score=score,
                            skeleton=skeleton,
                        )
                    )

                if self.tracker:
                    # Set tracks for predicted instances in this frame.
                    predicted_instances = self.tracker.track(
                        untracked_instances=predicted_instances, img=image, t=frame_ind
                    )

                predicted_frames.append(
                    sleap.LabeledFrame(
                        video=data_provider.videos[video_ind],
                        frame_idx=frame_ind,
                        instances=predicted_instances,
                    )
                )

        if self.tracker:
            self.tracker.final_pass(predicted_frames)

        return predicted_frames

    def predict(
        self,
        data: Union[Provider, sleap.Labels, sleap.Video],
        make_labels: bool = True,
    ) -> Union[List[Dict[str, np.ndarray]], sleap.Labels]:
        """Run inference and tracking on a data source.

        Args:
            data: A `sleap.pipelines.Provider`, `sleap.Labels` or `sleap.Video` to
                run inference over.
            make_labels: If `True` (the default), returns a `sleap.Labels` instance with
                `sleap.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.

        Returns:
            A `sleap.Labels` with `sleap.PredictedInstance`s if `make_labels` is `True`,
            otherwise a list of dictionaries containing batches of numpy arrays with the
            raw results.
        """
        # Create provider if necessary.
        if isinstance(data, sleap.Labels):
            data = LabelsReader(data)
        elif isinstance(data, sleap.Video):
            data = VideoReader(data)

        # Initialize inference loop generator.
        generator = self._predict_generator(data)

        if make_labels:
            # Create SLEAP data structures while consuming results.
            return sleap.Labels(
                self._make_labeled_frames_from_generator(generator, data)
            )
        else:
            # Just return the raw results.
            return list(generator)


class BottomUpInferenceLayer(InferenceLayer):
    """Keras layer that predicts instances from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps and part affinity fields as output.
        paf_scorer: A `sleap.nn.paf_grouping.PAFScorer` instance configured to group
            instances based on peaks and PAFs produced by the model.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        cm_output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        paf_output_stride: Output stride of the model, denoting the scale of the output
            part affinity fields relative to the images (after input scaling). This is
            used for adjusting the peak coordinates to the PAF grid. This will be
            inferred from the model shapes if not provided.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset. This has no
            effect if the model has an offset regression head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        return_confmaps: If `True`, the confidence maps will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        return_pafs: If `True`, the part affinity fields will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"MultiInstanceConfmapsHead"` in its name.
        pafs_ind: Index of the output tensor of the model corresponding to part affinity
            fields. If `None` (the default), this will be detected automatically by
            searching for the first tensor that contains `"PartAffinityFieldsHead"` in
            its name.
        offsets_ind: Index of the output tensor of the model corresponding to
            offset regression maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"OffsetRefinementHead"` in its name. If the head is not present, the method
            specified in the `refinement` attribute will be used.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        paf_scorer: PAFScorer,
        input_scale: float = 1.0,
        pad_to_stride: int = 1,
        cm_output_stride: Optional[int] = None,
        paf_output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        return_pafs: bool = False,
        confmaps_ind: Optional[int] = None,
        pafs_ind: Optional[int] = None,
        offsets_ind: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            keras_model=keras_model,
            input_scale=input_scale,
            pad_to_stride=pad_to_stride,
            **kwargs,
        )
        self.paf_scorer = paf_scorer
        self.confmaps_ind = confmaps_ind
        self.pafs_ind = pafs_ind
        self.offsets_ind = offsets_ind

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(self.keras_model, "MultiInstanceConfmapsHead")

        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'MultiInstanceConfmapsHead'."
            )

        if self.pafs_ind is None:
            self.pafs_ind = find_head(self.keras_model, "PartAffinityFieldsHead")

        if self.pafs_ind is None:
            raise ValueError(
                "Index of the part affinity fields output tensor must be specified if "
                "not named 'PartAffinityFieldsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if cm_output_stride is None:
            # Attempt to automatically infer the output stride.
            cm_output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.confmaps_ind
            )
        self.cm_output_stride = cm_output_stride
        if paf_output_stride is None:
            # Attempt to automatically infer the output stride.
            paf_output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.pafs_ind
            )
        self.paf_output_stride = paf_output_stride

        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.return_pafs = return_pafs

    def forward_pass(self, data):
        """Run preprocessing and model inference on a batch."""
        if isinstance(data, dict):
            imgs = data["image"]
        else:
            imgs = data

        # Preprocess full images.
        imgs = self.preprocess(imgs)

        # Model forward pass.
        preds = self.keras_model(imgs)
        if self.offsets_ind is None:
            cms = preds[self.confmaps_ind]
            pafs = preds[self.pafs_ind]
            offsets = None
        else:
            cms = preds[self.confmaps_ind]
            pafs = preds[self.pafs_ind]
            offsets = preds[self.offsets_ind]

        if isinstance(cms, list):
            cms = cms[-1]
        if isinstance(pafs, list):
            pafs = pafs[-1]

        return cms, pafs, offsets

    def find_peaks(self, cms, offsets):
        """Run peak finding on predicted confidence maps."""
        # Find local peaks.
        if self.offsets_ind is None:
            # Use deterministic refinement.
            (
                peaks,
                peak_vals,
                peak_sample_inds,
                peak_channel_inds,
            ) = sleap.nn.peak_finding.find_local_peaks(
                cms,
                threshold=self.peak_threshold,
                refinement=self.refinement,
                integral_patch_size=self.integral_patch_size,
            )
        else:
            # Use learned offsets.
            (
                peaks,
                peak_vals,
                peak_sample_inds,
                peak_channel_inds,
            ) = sleap.nn.peak_finding.find_local_peaks_with_offsets(
                cms,
                offsets,
                threshold=self.peak_threshold,
            )

        # Adjust for confidence map output stride.
        peaks = peaks * tf.cast(self.cm_output_stride, tf.float32)

        # Group peaks by sample.
        n_samples = tf.shape(cms)[0]
        peaks = tf.RaggedTensor.from_value_rowids(
            peaks, peak_sample_inds, nrows=n_samples
        )
        peak_vals = tf.RaggedTensor.from_value_rowids(
            peak_vals, peak_sample_inds, nrows=n_samples
        )
        peak_channel_inds = tf.RaggedTensor.from_value_rowids(
            peak_channel_inds, peak_sample_inds, nrows=n_samples
        )

        return peaks, peak_vals, peak_channel_inds

    def call(self, data):
        """Predict instances for one batch of images.

        Args:
            data: This may be either a single batch of images as a 4-D tensor of shape
            `(batch_size, height, width, channels)`, or a dictionary containing the
            image batch in the `"images"` key.

        Returns:
            The predicted instances as a dictionary of tensors with keys:

            `"instance_peaks": (batch_size, n_instances, n_nodes, 2)`: Instance skeleton
            points.

            `"instance_peak_vals": (batch_size, n_instances, n_nodes)`: Confidence
            values for the instance skeleton points.

            `"instance_scores": (batch_size, n_instances)`: PAF matching score for each
            instance.

            If `BottomUpInferenceLayer.return_confmaps` is `True`, the predicted
            confidence maps will be returned in the `"confmaps"` key.

            If `BottomUpInferenceLayer.return_pafs` is `True`, the predicted PAFs will
            be returned in the `"part_affinity_fields"` key.
        """
        cms, pafs, offsets = self.forward_pass(data)
        peaks, peak_vals, peak_channel_inds = self.find_peaks(cms, offsets)
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
        ) = self.paf_scorer.predict(pafs, peaks, peak_vals, peak_channel_inds)

        # Adjust for input scaling.
        if self.input_scale != 1.0:
            # Note: We add 0.5 here to offset TensorFlow's weird image resizing. This
            # may not always(?) be the most correct approach.
            # See: https://github.com/tensorflow/tensorflow/issues/6720
            predicted_instances = (predicted_instances / self.input_scale) + 0.5

        # Build outputs and return.
        out = {
            "instance_peaks": predicted_instances,
            "instance_peak_vals": predicted_peak_scores,
            "instance_scores": predicted_instance_scores,
        }
        if self.return_confmaps:
            out["confmaps"] = cms
        if self.return_pafs:
            out["part_affinity_fields"] = pafs
        return out


class BottomUpInferenceModel(InferenceModel):
    """Bottom-up instance prediction model.

    This model encapsulates the bottom-up approach where points are first detected by
    local peak detection and then grouped into instances by connectivity scoring using
    part affinity fields.

    Attributes:
        bottomup_layer: A `BottomUpInferenceLayer`. This layer takes as input a full
            image and outputs the predicted instances.
    """

    def __init__(self, bottomup_layer, **kwargs):
        super().__init__(**kwargs)
        self.bottomup_layer = bottomup_layer

    @property
    def inference_layer(self):
        return self.bottomup_layer

    def call(self, example):
        """Predict instances for one batch of images.

        Args:
            example: This may be either a single batch of images as a 4-D tensor of
                shape `(batch_size, height, width, channels)`, or a dictionary
                containing the image batch in the `"images"` key.

        Returns:
            The predicted instances as a dictionary of tensors with keys:

            `"instance_peaks": (batch_size, n_instances, n_nodes, 2)`: Instance skeleton
                points.
            `"instance_peak_vals": (batch_size, n_instances, n_nodes)`: Confidence
                values for the instance skeleton points.
            `"instance_scores": (batch_size, n_instances)`: PAF matching score for each
                instance.

            If `BottomUpInferenceModel.bottomup_layer.return_confmaps` is `True`, the
            predicted confidence maps will be returned in the `"confmaps"` key.

            If `BottomUpInferenceModel.bottomup_layer.return_pafs` is `True`, the
            predicted PAFs will be returned in the `"part_affinity_fields"` key.
        """
        if isinstance(example, tf.Tensor):
            example = dict(image=example)
        return self.bottomup_layer(example)


@attr.s(auto_attribs=True)
class BottomupPredictor(Predictor):
    """Bottom-up multi-instance predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained bottom-up multi-instance SLEAP model.

    This should be initialized using the `from_trained_models()` constructor or the
    high-level API (`sleap.load_model`).

    Attributes:
        bottomup_config: The `sleap.nn.config.TrainingJobConfig` containing the metadata
            for the trained bottomup model.
        bottomup_model: A `sleap.nn.model.Model` instance created from the trained
            bottomup model. If `None`, ground truth centroids will be used if available
            from the data source.
        inference_model: A `sleap.nn.inference.BottomUpInferenceModel` that wraps a
            trained `tf.keras.Model` to implement preprocessing, centroid detection,
            cropping and peak finding.
        pipeline: A `sleap.nn.data.Pipeline` that loads the data and batches input data.
            This will be updated dynamically if new data sources are used.
        tracker: A `sleap.nn.tracking.Tracker` that will be called to associate
            detections over time. Predicted instances will not be assigned to tracks if
            if this is `None`.
        batch_size: The default batch size to use when loading data for inference.
            Higher values increase inference speed at the cost of higher memory usage.
        peak_threshold: Minimum confidence map value to consider a local peak as valid.
        integral_refinement: If `True`, peaks will be refined with integral regression.
            If `False`, `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        max_edge_length_ratio: The maximum expected length of a connected pair of points
            in relative image units. Candidate connections above this length will be
            penalized during matching.
        paf_line_points: Number of points to sample along the line integral.
    """

    bottomup_config: TrainingJobConfig
    bottomup_model: Model
    inference_model: Optional[BottomUpInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    tracker: Optional[Tracker] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2
    batch_size: int = 4
    integral_refinement: bool = True
    integral_patch_size: int = 5
    max_edge_length_ratio: float = 0.5
    paf_line_points: int = 10

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained model and configuration."""
        self.inference_model = BottomUpInferenceModel(
            BottomUpInferenceLayer(
                keras_model=self.bottomup_model.keras_model,
                paf_scorer=PAFScorer.from_config(
                    self.bottomup_config.model.heads.multi_instance,
                    max_edge_length_ratio=self.max_edge_length_ratio,
                    n_points=self.paf_line_points,
                    min_instance_peaks=0,
                ),
                input_scale=self.bottomup_config.data.preprocessing.input_scaling,
                pad_to_stride=self.bottomup_model.maximum_stride,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
            )
        )

    @classmethod
    def from_trained_models(
        cls,
        model_path: Text,
        batch_size: int = 4,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
    ) -> "BottomupPredictor":
        """Create predictor from a saved model.

        Args:
            model_path: Path to a bottom-up model folder or training job JSON file
                inside a model folder. This folder should contain `training_config.json`
                and `best_model.h5` files for a trained model.
            batch_size: The default batch size to use when loading data for inference.
                Higher values increase inference speed at the cost of higher memory
                usage.
            peak_threshold: Minimum confidence map value to consider a local peak as
                valid.
            integral_refinement: If `True`, peaks will be refined with integral
                regression. If `False`, `"local"`, peaks will be refined with quarter
                pixel local gradient offset. This has no effect if the model has an
                offset regression head.
            integral_patch_size: Size of patches to crop around each rough peak for
                integral refinement as an integer scalar.

        Returns:
            An instance of `BottomupPredictor` with the loaded model.
        """
        # Load bottomup model.
        bottomup_config = TrainingJobConfig.load_json(model_path)
        bottomup_keras_model_path = get_keras_model_path(model_path)
        bottomup_model = Model.from_config(bottomup_config.model)
        bottomup_model.keras_model = tf.keras.models.load_model(
            bottomup_keras_model_path, compile=False
        )
        obj = cls(
            bottomup_config=bottomup_config,
            bottomup_model=bottomup_model,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
        )
        obj._initialize_inference_model()
        return obj

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:
        """Make a data loading pipeline.

        Args:
            data_provider: If not `None`, the pipeline will be created with an instance
                of a `sleap.pipelines.Provider`.

        Returns:
            The created `sleap.pipelines.Pipeline` with batching and prefetching.

        Notes:
            This method also updates the class attribute for the pipeline and will be
            called automatically when predicting on data from a new source.
        """
        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        pipeline += sleap.nn.data.pipelines.Batcher(
            batch_size=self.batch_size, drop_remainder=False, unrag=False
        )

        pipeline += Prefetcher()

        self.pipeline = pipeline

        return pipeline

    def _predict_generator(
        self, data_provider: Provider
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Create a generator that yields batches of inference results.

        This method handles creating or updating the input `sleap.pipelines.Pipeline`
        for loading the data, as well as looping over the batches and running inference.

        Args:
            data_provider: The `sleap.pipelines.Provider` that contains data that should
                be used for inference.

        Returns:
            A generator yielding batches predicted results as dictionaries of numpy
            arrays.
        """
        # Initialize data pipeline and inference model if needed.
        if self.pipeline is None:
            self.make_pipeline()
        if self.inference_model is None:
            self._initialize_inference_model()

        # Update the data provider source.
        self.pipeline.providers = [data_provider]

        # Loop over data batches.
        for ex in self.pipeline.make_dataset():
            # Run inference on current batch.
            preds = self.inference_model.predict(ex)

            # Crop possibly variable length results.
            ex["instance_peaks"] = [
                x[:n] for x, n in zip(preds["instance_peaks"], preds["n_valid"])
            ]
            ex["instance_peak_vals"] = [
                x[:n] for x, n in zip(preds["instance_peak_vals"], preds["n_valid"])
            ]
            ex["instance_scores"] = [
                x[:n] for x, n in zip(preds["instance_scores"], preds["n_valid"])
            ]

            # Convert to numpy arrays if not already.
            if isinstance(ex["video_ind"], tf.Tensor):
                ex["video_ind"] = ex["video_ind"].numpy().flatten()
            if isinstance(ex["frame_ind"], tf.Tensor):
                ex["frame_ind"] = ex["frame_ind"].numpy().flatten()

            yield ex

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
    ) -> List[sleap.LabeledFrame]:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures and runs
        them through the tracker if it is specified.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"image"`, `"video_ind"`,
                `"frame_ind"`, `"instance_peaks"`, `"instance_peak_vals"`, and
                `"instance_scores"`. This can be created using the
                `_predict_generator()` method.
            data_provider: The `sleap.pipelines.Provider` that the predictions are being
                created from. This is used to retrieve the `sleap.Video` instance
                associated with each inference result.

        Returns:
            A list of `sleap.LabeledFrame`s with `sleap.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        skeleton = self.bottomup_config.data.labels.skeletons[0]

        # Loop over batches.
        predicted_frames = []
        for ex in generator:

            # Loop over frames.
            for image, video_ind, frame_ind, points, confidences, scores in zip(
                ex["image"],
                ex["video_ind"],
                ex["frame_ind"],
                ex["instance_peaks"],
                ex["instance_peak_vals"],
                ex["instance_scores"],
            ):

                # Loop over instances.
                predicted_instances = []
                for pts, confs, score in zip(points, confidences, scores):
                    predicted_instances.append(
                        sleap.instance.PredictedInstance.from_arrays(
                            points=pts,
                            point_confidences=confs,
                            instance_score=score,
                            skeleton=skeleton,
                        )
                    )

                if self.tracker:
                    # Set tracks for predicted instances in this frame.
                    predicted_instances = self.tracker.track(
                        untracked_instances=predicted_instances, img=image, t=frame_ind
                    )

                predicted_frames.append(
                    sleap.LabeledFrame(
                        video=data_provider.videos[video_ind],
                        frame_idx=frame_ind,
                        instances=predicted_instances,
                    )
                )

        if self.tracker:
            self.tracker.final_pass(predicted_frames)

        return predicted_frames

    def predict(
        self,
        data: Union[Provider, sleap.Labels, sleap.Video],
        make_labels: bool = True,
    ) -> Union[List[Dict[str, np.ndarray]], sleap.Labels]:
        """Run inference and tracking on a data source.

        Args:
            data: A `sleap.pipelines.Provider`, `sleap.Labels` or `sleap.Video` to
                run inference over.
            make_labels: If `True` (the default), returns a `sleap.Labels` instance with
                `sleap.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.

        Returns:
            A `sleap.Labels` with `sleap.PredictedInstance`s if `make_labels` is `True`,
            otherwise a list of dictionaries containing batches of numpy arrays with the
            raw results.
        """
        # Create provider if necessary.
        if isinstance(data, sleap.Labels):
            data = LabelsReader(data)
        elif isinstance(data, sleap.Video):
            data = VideoReader(data)

        # Initialize inference loop generator.
        generator = self._predict_generator(data)

        if make_labels:
            # Create SLEAP data structures while consuming results.
            return sleap.Labels(
                self._make_labeled_frames_from_generator(generator, data)
            )
        else:
            # Just return the raw results.
            return list(generator)


CLI_PREDICTORS = {
    "topdown": TopdownPredictor,
    "bottomup": BottomupPredictor,
    "single": SingleInstancePredictor,
}


def make_cli_parser():
    import argparse
    from sleap.util import frame_list

    parser = argparse.ArgumentParser()

    # Add args for entire pipeline
    parser.add_argument(
        "video_path", type=str, nargs="?", default="", help="Path to video file"
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        help="Path to trained model directory (with training_config.json). "
        "Multiple models can be specified, each preceded by --model.",
    )

    parser.add_argument(
        "--frames",
        type=frame_list,
        default="",
        help="List of frames to predict. Either comma separated list (e.g. 1,2,3) or "
        "a range separated by hyphen (e.g. 1-3, for 1,2,3). (default is entire video)",
    )
    parser.add_argument(
        "--only-labeled-frames",
        action="store_true",
        default=False,
        help="Only run inference on labeled frames (when running on labels dataset file).",
    )
    parser.add_argument(
        "--only-suggested-frames",
        action="store_true",
        default=False,
        help="Only run inference on suggested frames (when running on labels dataset file).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="The output filename to use for the predicted data.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels dataset file (for inference on multiple videos or for re-tracking pre-existing predictions).",
    )

    # TODO: better video parameters

    parser.add_argument(
        "--video.dataset", type=str, default="", help="The dataset for HDF5 videos."
    )

    parser.add_argument(
        "--video.input_format",
        type=str,
        default="",
        help="The input_format for HDF5 videos.",
    )

    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference only on CPU. If not specified, will use available GPU.",
    )
    device_group.add_argument(
        "--first-gpu",
        action="store_true",
        help="Run inference on the first GPU, if available.",
    )
    device_group.add_argument(
        "--last-gpu",
        action="store_true",
        help="Run inference on the last GPU, if available.",
    )
    device_group.add_argument(
        "--gpu", type=int, default=0, help="Run inference on the i-th GPU specified."
    )

    # Add args for each predictor class
    for predictor_name, predictor_class in CLI_PREDICTORS.items():
        if "peak_threshold" in attr.fields_dict(predictor_class):
            # get the default value to show in help string, although we'll
            # use None as default so that unspecified vals won't be passed to
            # builder.
            default_val = attr.fields_dict(predictor_class)["peak_threshold"].default

            parser.add_argument(
                f"--{predictor_name}.peak_threshold",
                type=float,
                default=None,
                help=f"Threshold to use when finding peaks in {predictor_class.__name__} (default: {default_val}).",
            )

        if "batch_size" in attr.fields_dict(predictor_class):
            default_val = attr.fields_dict(predictor_class)["batch_size"].default
            parser.add_argument(
                f"--{predictor_name}.batch_size",
                type=int,
                default=4,
                help=f"Batch size to use for model inference in {predictor_class.__name__} (default: {default_val}).",
            )

    # Add args for tracking
    Tracker.add_cli_parser_args(parser, arg_scope="tracking")

    parser.add_argument(
        "--test-pipeline",
        default=False,
        action="store_true",
        help="Test pipeline construction without running anything.",
    )

    return parser


def make_video_readers_from_cli(args) -> List[VideoReader]:
    if args.video_path:
        # TODO: better support for video params
        video_kwargs = dict(
            dataset=vars(args).get("video.dataset"),
            input_format=vars(args).get("video.input_format"),
        )

        video_reader = VideoReader.from_filepath(
            filename=args.video_path, example_indices=args.frames, **video_kwargs
        )

        return [video_reader]

    if args.labels:
        # TODO: Replace with LabelsReader.
        labels = sleap.Labels.load_file(args.labels)

        readers = []

        if args.only_labeled_frames:
            user_labeled_frames = labels.user_labeled_frames
        else:
            user_labeled_frames = []

        for video in labels.videos:
            if args.only_labeled_frames:
                frame_indices = [
                    lf.frame_idx for lf in user_labeled_frames if lf.video == video
                ]
                readers.append(VideoReader(video=video, example_indices=frame_indices))
            elif args.only_suggested_frames:
                readers.append(
                    VideoReader(
                        video=video, example_indices=labels.get_video_suggestions(video)
                    )
                )
            else:
                readers.append(VideoReader(video=video))

        return readers

    raise ValueError("You must specify either video_path or labels dataset path.")


def make_predictor_from_paths(paths, **kwargs) -> Predictor:
    """Build predictor object from a list of model paths."""
    return make_predictor_from_models(find_heads_for_model_paths(paths), **kwargs)


def find_heads_for_model_paths(paths) -> Dict[str, str]:
    """Given list of models paths, returns dict with path keyed by head name."""
    trained_model_paths = dict()

    if paths is None:
        return trained_model_paths

    for model_path in paths:
        # Load the model config
        cfg = TrainingJobConfig.load_json(model_path)

        # Get the head from the model (i.e., what the model will predict)
        key = cfg.model.heads.which_oneof_attrib_name()

        # If path is to config file json, then get the path to parent dir
        if model_path.endswith(".json"):
            model_path = os.path.dirname(model_path)

        trained_model_paths[key] = model_path

    return trained_model_paths


def make_predictor_from_models(
    trained_model_paths: Dict[str, str],
    labels_path: Optional[str] = None,
    policy_args: Optional[dict] = None,
    **kwargs,
) -> Predictor:
    """Given dict of paths keyed by head name, returns appropriate predictor."""

    def get_relevant_args(key):
        if policy_args is not None and key in policy_args:
            return policy_args[key]
        return dict()

    if "multi_instance" in trained_model_paths:
        predictor = BottomupPredictor.from_trained_models(
            trained_model_paths["multi_instance"],
            **get_relevant_args("bottomup"),
            **kwargs,
        )
    elif "single_instance" in trained_model_paths:
        predictor = SingleInstancePredictor.from_trained_models(
            trained_model_paths["single_instance"],
            **get_relevant_args("single"),
            **kwargs,
        )
    elif (
        "centroid" in trained_model_paths and "centered_instance" in trained_model_paths
    ):
        predictor = TopdownPredictor.from_trained_models(
            centroid_model_path=trained_model_paths["centroid"],
            confmap_model_path=trained_model_paths["centered_instance"],
            **get_relevant_args("topdown"),
            **kwargs,
        )
    elif len(trained_model_paths) == 0 and labels_path:
        predictor = MockPredictor.from_trained_models(labels_path=labels_path)
    else:
        raise ValueError(
            f"Unable to run inference with {list(trained_model_paths.keys())} heads."
        )

    return predictor


def make_tracker_from_cli(policy_args):
    if "tracking" in policy_args:
        tracker = Tracker.make_tracker_by_name(**policy_args["tracking"])
        return tracker

    return None


def save_predictions_from_cli(args, predicted_frames, prediction_metadata=None):
    from sleap import Labels

    if args.output:
        output_path = args.output
    elif args.video_path:
        out_dir = os.path.dirname(args.video_path)
        out_name = os.path.basename(args.video_path) + ".predictions.slp"
        output_path = os.path.join(out_dir, out_name)
    elif args.labels:
        out_dir = os.path.dirname(args.labels)
        out_name = os.path.basename(args.labels) + ".predictions.slp"
        output_path = os.path.join(out_dir, out_name)
    else:
        # We shouldn't ever get here but if we do, just save in working dir.
        output_path = "predictions.slp"

    labels = Labels(labeled_frames=predicted_frames, provenance=prediction_metadata)

    print(f"Saving: {output_path}")
    Labels.save_file(labels, output_path)


def load_model(
    model_path: Union[str, List[str]],
    batch_size: int = 4,
    refinement: str = "integral",
    tracker: Optional[str] = None,
    tracker_window: int = 5,
    tracker_max_instances: Optional[int] = None,
) -> Predictor:
    """Load a trained SLEAP model.

    Args:
        model_path: Path to model or list of path to models that were trained by SLEAP.
            These should be the directories that contain `training_job.json` and
            `best_model.h5`.
        batch_size: Number of frames to predict at a time. Larger values result in
            faster inference speeds, but require more memory.
        refinement: If `"integral"`, peak locations will be refined with integral
            regression. If `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        tracker: Name of the tracker to use with the inference model. Must be one of
            `"simple"` or `"flow"`. If `None`, no identity tracking across frames will
            be performed.
        tracker_window: Number of frames of history to use when tracking. No effect when
            `tracker` is `None`.
        tracker_max_instances: If not `None`, discard instances beyond this count when
            tracking. No effect when `tracker` is `None`.

    Returns:
        An instance of a `Predictor` based on which model type was detected.

        If this is a top-down model, paths to the centroids model as well as the
        centered instance model must be provided. A `TopdownPredictor` instance will be
        returned.

        If this is a bottom-up model, a `BottomupPredictor` will be returned.

        If this is a single-instance model, a `SingleInstancePredictor` will be
        returned.

        If a `tracker` is specified, the predictor will also run identity tracking over
        time.

    See also: TopdownPredictor, BottomupPredictor, SingleInstancePredictor
    """
    if isinstance(model_path, str):
        model_path = [model_path]
    predictor = make_predictor_from_paths(
        model_path, batch_size=batch_size, integral_refinement=refinement == "integral"
    )
    if tracker is not None:
        predictor.tracker = Tracker.make_tracker_by_name(
            tracker=tracker,
            track_window=tracker_window,
            post_connect_single_breaks=True,
            clean_instance_count=tracker_max_instances,
        )
    return predictor


def main():
    """CLI for running inference."""
    parser = make_cli_parser()
    args, _ = parser.parse_known_args()
    print(args)

    if args.cpu or not sleap.nn.system.is_gpu_system():
        sleap.nn.system.use_cpu_only()
    else:
        if args.first_gpu:
            sleap.nn.system.use_first_gpu()
        elif args.last_gpu:
            sleap.nn.system.use_last_gpu()
        else:
            sleap.nn.system.use_gpu(args.gpu)
    sleap.nn.system.disable_preallocation()

    print("System:")
    sleap.nn.system.summary()

    video_readers = make_video_readers_from_cli(args)

    # Find the specified models
    model_paths_by_head = find_heads_for_model_paths(args.models)

    # Make a scoped dictionary with args specified from cli
    policy_args = util.make_scoped_dictionary(vars(args), exclude_nones=True)

    # Create appropriate predictor given these models
    predictor = make_predictor_from_models(
        model_paths_by_head, labels_path=args.labels, policy_args=policy_args
    )

    # Make the tracker
    tracker = make_tracker_from_cli(policy_args)
    predictor.tracker = tracker

    if args.test_pipeline:
        print()

        print(policy_args)
        print()

        print(predictor)
        print()

        predictor.make_pipeline()
        print("===pipeline transformers===")
        print()
        for transformer in predictor.pipeline.transformers:
            print(transformer.__class__.__name__)
            print(f"\t-> {transformer.input_keys}")
            print(f"\t   {transformer.output_keys} ->")
            print()

        print("--test-pipeline arg set so stopping here.")
        return

    # Run inference!
    t0 = time.time()
    predicted_frames = []

    for video_reader in video_readers:
        video_predicted_frames = predictor.predict(video_reader).labeled_frames
        predicted_frames.extend(video_predicted_frames)

    # Create dictionary of metadata we want to save with predictions
    prediction_metadata = dict()
    for head, path in model_paths_by_head.items():
        prediction_metadata[f"model.{head}.path"] = os.path.abspath(path)
    for scope in policy_args.keys():
        for key, val in policy_args[scope].items():
            prediction_metadata[f"{scope}.{key}"] = val
    prediction_metadata["video.path"] = args.video_path
    prediction_metadata["sleap.version"] = sleap.__version__

    save_predictions_from_cli(args, predicted_frames, prediction_metadata)
    print(f"Total Time: {time.time() - t0}")


if __name__ == "__main__":
    main()

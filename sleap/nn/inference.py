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
import argparse
import logging
import warnings
import os
import sys
import tempfile
import platform
import shutil
import atexit
import subprocess
import rich.progress
import pandas as pd
from rich.pretty import pprint
from collections import deque
import json
from time import time
from datetime import datetime
from pathlib import Path
import tensorflow_hub as hub
from abc import ABC, abstractmethod
from typing import Text, Optional, List, Dict, Union, Iterator, Tuple
from threading import Thread
from queue import Queue

import tensorflow as tf
import numpy as np

import sleap

from sleap.nn.config import TrainingJobConfig, DataConfig
from sleap.nn.data.resizing import SizeMatcher
from sleap.nn.model import Model
from sleap.nn.tracking import Tracker, run_tracker
from sleap.nn.paf_grouping import PAFScorer
from sleap.nn.data.pipelines import (
    Provider,
    Pipeline,
    LabelsReader,
    VideoReader,
    Normalizer,
    Resizer,
    Prefetcher,
    InstanceCentroidFinder,
    KerasModelPredictor,
)
from sleap.nn.utils import reset_input_layer
from sleap.io.dataset import Labels
from sleap.util import frame_list, make_scoped_dictionary
from sleap.instance import PredictedInstance, LabeledFrame

from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

MOVENET_MODELS = {
    "lightning": {
        "model_path": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "image_size": 192,
    },
    "thunder": {
        "model_path": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "image_size": 256,
    },
}

MOVENET_SKELETON = sleap.Skeleton.from_names_and_edge_inds(
    [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    [
        (10, 8),
        (8, 6),
        (6, 5),
        (5, 7),
        (7, 9),
        (6, 12),
        (5, 11),
        (12, 14),
        (14, 16),
        (11, 13),
        (13, 15),
        (4, 2),
        (2, 0),
        (0, 1),
        (1, 3),
    ],
)

logger = logging.getLogger(__name__)


def get_keras_model_path(path: Text) -> str:
    """Utility method for finding the path to a saved Keras model.

    Args:
        path: Path to a model run folder or job file.

    Returns:
        Path to `best_model.h5` in the run folder.
    """
    # TODO: Move this to TrainingJobConfig or Model?
    if path.endswith(".json"):
        path = os.path.dirname(path)
    return os.path.join(path, "best_model.h5")


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.1f} FPS", style="progress.data.speed")


@attr.s(auto_attribs=True)
class Predictor(ABC):
    """Base interface class for predictors."""

    verbosity: str = attr.ib(
        validator=attr.validators.in_(["none", "rich", "json"]),
        default="rich",
        kw_only=True,
    )
    report_rate: float = attr.ib(default=2.0, kw_only=True)
    model_paths: List[str] = attr.ib(factory=list, kw_only=True)

    @property
    def report_period(self) -> float:
        """Time between progress reports in seconds."""
        return 1.0 / self.report_rate

    @classmethod
    def from_model_paths(
        cls,
        model_paths: Union[str, List[str]],
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        resize_input_layer: bool = True,
        max_instances: Optional[int] = None,
    ) -> "Predictor":
        """Create the appropriate `Predictor` subclass from a list of model paths.

        Args:
            model_paths: A single or list of trained model paths. Special cases of
                non-SLEAP models include "movenet-thunder" and "movenet-lightning".
            peak_threshold: Minimum confidence map value to consider a peak as valid.
            integral_refinement: If `True`, peaks will be refined with integral
                regression. If `False`, `"local"`, peaks will be refined with quarter
                pixel local gradient offset. This has no effect if the model has an
                offset regression head.
            integral_patch_size: Size of patches to crop around each rough peak for
                integral refinement as an integer scalar.
            batch_size: The default batch size to use when loading data for inference.
                Higher values increase inference speed at the cost of higher memory
                usage.
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).
            max_instances: If not `None`, discard instances beyond this count when
                predicting, regardless of whether filtering is done at the tracking
                stage. This is useful for preventing extraneous instances from being
                created when tracking is not being applied.

        Returns:
            A subclass of `Predictor`.

        See also: `SingleInstancePredictor`, `TopDownPredictor`, `BottomUpPredictor`,
            `MoveNetPredictor`, `TopDownMultiClassPredictor`,
            `BottomUpMultiClassPredictor`.
        """
        # Read configs and find model types.
        if isinstance(model_paths, str):
            # Special case for handling movenet models
            if "movenet" in model_paths:
                model_name = model_paths.split("-")[-1]  # Expect movenet-<model_name>
                predictor = MoveNetPredictor.from_trained_models(
                    model_name=model_name, peak_threshold=peak_threshold
                )
                predictor.model_paths = MOVENET_MODELS[model_name]["model_path"]
                return predictor

            model_paths = [model_paths]

        model_configs = [sleap.load_config(model_path) for model_path in model_paths]
        model_paths = [cfg.filename for cfg in model_configs]
        model_types = [
            cfg.model.heads.which_oneof_attrib_name() for cfg in model_configs
        ]

        if "single_instance" in model_types:
            predictor = SingleInstancePredictor.from_trained_models(
                model_path=model_paths[model_types.index("single_instance")],
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                resize_input_layer=resize_input_layer,
            )

        elif (
            "centroid" in model_types
            or "centered_instance" in model_types
            or "multi_class_topdown" in model_types
        ):
            centroid_model_path = None
            if "centroid" in model_types:
                centroid_model_path = model_paths[model_types.index("centroid")]

            confmap_model_path = None
            if "centered_instance" in model_types:
                confmap_model_path = model_paths[model_types.index("centered_instance")]

            td_multiclass_model_path = None
            if "multi_class_topdown" in model_types:
                td_multiclass_model_path = model_paths[
                    model_types.index("multi_class_topdown")
                ]

            if td_multiclass_model_path is not None:
                predictor = TopDownMultiClassPredictor.from_trained_models(
                    centroid_model_path=centroid_model_path,
                    confmap_model_path=td_multiclass_model_path,
                    batch_size=batch_size,
                    peak_threshold=peak_threshold,
                    integral_refinement=integral_refinement,
                    integral_patch_size=integral_patch_size,
                    resize_input_layer=resize_input_layer,
                )
            else:
                predictor = TopDownPredictor.from_trained_models(
                    centroid_model_path=centroid_model_path,
                    confmap_model_path=confmap_model_path,
                    batch_size=batch_size,
                    peak_threshold=peak_threshold,
                    integral_refinement=integral_refinement,
                    integral_patch_size=integral_patch_size,
                    resize_input_layer=resize_input_layer,
                    max_instances=max_instances,
                )

        elif "multi_instance" in model_types:
            predictor = BottomUpPredictor.from_trained_models(
                model_path=model_paths[model_types.index("multi_instance")],
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                resize_input_layer=resize_input_layer,
                max_instances=max_instances,
            )

        elif "multi_class_bottomup" in model_types:
            predictor = BottomUpMultiClassPredictor.from_trained_models(
                model_path=model_paths[model_types.index("multi_class_bottomup")],
                peak_threshold=peak_threshold,
                integral_refinement=integral_refinement,
                integral_patch_size=integral_patch_size,
                batch_size=batch_size,
                resize_input_layer=resize_input_layer,
            )

        else:
            raise ValueError(
                "Could not create predictor from model paths:" + "\n".join(model_paths)
            )
        predictor.model_paths = model_paths
        return predictor

    @classmethod
    @abstractmethod
    def from_trained_models(cls, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def data_config(self) -> DataConfig:
        pass

    @property
    @abstractmethod
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        pass

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

        if self.data_config.preprocessing.resize_and_pad_to_target:
            points_key = None
            if data_provider is not None and "instances" in data_provider.output_keys:
                points_key = "instances"
            pipeline += SizeMatcher.from_config(
                config=self.data_config.preprocessing,
                provider=data_provider,
                points_key=points_key,
            )

        pipeline += Normalizer(
            ensure_float=False,
            ensure_grayscale=self.is_grayscale,
            ensure_rgb=(not self.is_grayscale),
        )

        pipeline += sleap.nn.data.pipelines.Batcher(
            batch_size=self.batch_size, drop_remainder=False, unrag=False
        )

        pipeline += Prefetcher()

        self.pipeline = pipeline

        return pipeline

    @abstractmethod
    def _initialize_inference_model(self):
        pass

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
        self.make_pipeline(data_provider)
        if self.inference_model is None:
            self._initialize_inference_model()

        def process_batch(ex):
            # Run inference on current batch.
            preds = self.inference_model.predict_on_batch(ex, numpy=True)

            # Add model outputs to the input data example.
            ex.update(preds)

            # Convert to numpy arrays if not already.
            if isinstance(ex["video_ind"], tf.Tensor):
                ex["video_ind"] = ex["video_ind"].numpy().flatten()
            if isinstance(ex["frame_ind"], tf.Tensor):
                ex["frame_ind"] = ex["frame_ind"].numpy().flatten()

            # Adjust for potential SizeMatcher scaling.
            offset_x = ex.get("offset_x", 0)
            offset_y = ex.get("offset_y", 0)
            ex["instance_peaks"] -= np.reshape([offset_x, offset_y], [-1, 1, 1, 2])
            ex["instance_peaks"] /= np.expand_dims(
                np.expand_dims(ex["scale"], axis=1), axis=1
            )

            return ex

        # Loop over data batches with optional progress reporting.
        if self.verbosity == "rich":
            with rich.progress.Progress(
                "{task.description}",
                rich.progress.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "ETA:",
                rich.progress.TimeRemainingColumn(),
                RateColumn(),
                auto_refresh=False,
                refresh_per_second=self.report_rate,
                speed_estimate_period=5,
            ) as progress:
                task = progress.add_task("Predicting...", total=len(data_provider))
                last_report = time()
                for ex in self.pipeline.make_dataset():
                    ex = process_batch(ex)
                    progress.update(task, advance=len(ex["frame_ind"]))

                    # Handle refreshing manually to support notebooks.
                    elapsed_since_last_report = time() - last_report
                    if elapsed_since_last_report > self.report_period:
                        progress.refresh()

                    # Return results.
                    yield ex

        elif self.verbosity == "json":
            n_processed = 0
            n_total = len(data_provider)
            n_recent = deque(maxlen=30)
            elapsed_recent = deque(maxlen=30)
            last_report = time()
            t0_all = time()
            t0_batch = time()
            for ex in self.pipeline.make_dataset():
                # Process batch of examples.
                ex = process_batch(ex)

                # Track timing and progress.
                elapsed_batch = time() - t0_batch
                t0_batch = time()
                n_batch = len(ex["frame_ind"])
                n_processed += n_batch
                elapsed_all = time() - t0_all

                # Compute recent rate.
                n_recent.append(n_batch)
                elapsed_recent.append(elapsed_batch)
                rate = sum(n_recent) / sum(elapsed_recent)
                eta = (n_total - n_processed) / rate

                # Report.
                elapsed_since_last_report = time() - last_report
                if elapsed_since_last_report > self.report_period:
                    print(
                        json.dumps(
                            {
                                "n_processed": n_processed,
                                "n_total": n_total,
                                "elapsed": elapsed_all,
                                "rate": rate,
                                "eta": eta,
                            }
                        ),
                        flush=True,
                    )
                    last_report = time()

                # Return results.
                yield ex
        else:
            for ex in self.pipeline.make_dataset():
                yield process_batch(ex)

    def predict(
        self, data: Union[Provider, sleap.Labels, sleap.Video], make_labels: bool = True
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
        if isinstance(data, np.ndarray):
            data = sleap.Video(backend=sleap.io.video.NumpyVideo(data))
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

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
        max_instances: Optional[int] = None,
    ):
        """Export a trained SLEAP model as a frozen graph. Initializes model,
        creates a dummy tracing batch and passes it through the model. The
        frozen graph is saved along with training meta info.

        Args:
            save_path: Path to output directory to store the frozen graph
            signatures: String defining the input and output types for
                computation.
            save_traces: If `True` (default) the SavedModel will store the
                function traces for each layer
            model_name: (Optional) Name to give the model. If given, will be
                added to the output json file containing meta information about the
                model
            tensors: (Optional) Dictionary describing the predicted tensors (see
                sleap.nn.data.utils.describe_tensors as an example)
            unrag_outputs: If `True` (default), any ragged tensors will be
                converted to normal tensors and padded with NaNs
            max_instances: If set, determines the max number of instances that a
                multi-instance model returns. This is enforced during centroid
                cropping and therefore only compatible with TopDown models.
        """

        self._initialize_inference_model()
        predictor_name = type(self).__name__

        if max_instances is not None:
            if "TopDown" in predictor_name:
                print(f"\n max instances set, limiting instances to {max_instances} \n")
                self.inference_model.centroid_crop.max_instances = max_instances
            else:
                raise Exception(f"{predictor_name} does not support max instance limit")

        first_inference_layer = self.inference_model.layers[0]
        keras_model_shape = first_inference_layer.keras_model.input.shape

        sample_shape = tuple(
            (
                np.array(keras_model_shape[1:3]) / first_inference_layer.input_scale
            ).astype(int)
        ) + (keras_model_shape[3],)

        tracing_batch = np.zeros((1,) + sample_shape, dtype="uint8")
        outputs = self.inference_model.predict(tracing_batch)

        self.inference_model.export_model(
            save_path, signatures, save_traces, model_name, tensors, unrag_outputs
        )


# TODO: Rewrite this class.
@attr.s(auto_attribs=True)
class VisualPredictor(Predictor):
    """Predictor class for generating the visual output of model."""

    config: TrainingJobConfig
    model: Model
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)

    @property
    def data_config(self) -> DataConfig:
        return self.config.data

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
        if self.data_config.preprocessing.resize_and_pad_to_target:
            pipeline += SizeMatcher.from_config(
                config=self.data_config.preprocessing,
                points_key=None,
            )
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

    def safely_generate(self, ds: tf.data.Dataset, progress: bool = True):
        """Yields examples from dataset, catching and logging exceptions."""
        # Unsafe generating:
        # for example in ds:
        #     yield example

        ds_iter = iter(ds)

        i = 0
        wall_t0 = time()
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
                        elapsed_time = time() - wall_t0
                        logger.info(
                            f"Finished {i} examples in {elapsed_time:.2f} seconds "
                            "(inference + postprocessing)"
                        )
                        if elapsed_time:
                            logger.info(f"examples/s = {i/elapsed_time}")

    def predict_generator(self, data_provider: Provider):
        if self.pipeline is None:
            # Pass in data provider when mocking one of the models.
            self.make_pipeline()

        self.pipeline.providers = [data_provider]

        # Yield each example from dataset, catching and logging exceptions
        return self.safely_generate(self.pipeline.make_dataset())

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
        dists = tf.sqrt(tf.reduce_sum(tf.math.square(dists), axis=-1))  # reduce over xy
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
            instance_peaks, match_sample_inds, nrows=example_gt["instances"].nrows()
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
        pad_to_stride: If not 1, input image will be padded to ensure that it is
            divisible by this value (after scaling).
        ensure_grayscale: If `True`, converts inputs to grayscale if not already. If
            `False`, converts inputs to RGB if not already. If `None` (default), infer
            from the shape of the input layer of the model.
        ensure_float: If `True`, converts inputs to `float32` and scales the values to
            be between 0 and 1.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        input_scale: float = 1.0,
        pad_to_stride: int = 1,
        ensure_grayscale: Optional[bool] = None,
        ensure_float: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keras_model = keras_model
        self.input_scale = input_scale
        self.pad_to_stride = pad_to_stride
        if ensure_grayscale is None:
            ensure_grayscale = self.keras_model.inputs[0].shape[-1] == 1
        self.ensure_grayscale = ensure_grayscale
        self.ensure_float = ensure_float

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
            # TODO: Find out why this does not work on the GPU (but does on CPU).
            imgs = sleap.nn.data.normalization.ensure_grayscale(imgs)
        else:
            imgs = sleap.nn.data.normalization.ensure_rgb(imgs)

        if self.ensure_float:
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

    def predict_on_batch(
        self,
        data: Union[
            np.ndarray,
            tf.Tensor,
            Dict[str, tf.Tensor],
        ],
        numpy: bool = False,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
        """Predict a single batch of samples.

        Args:
            data: Input data in any form. Possible types:
                - `np.ndarray`, `tf.Tensor`: Images of shape
                    `(samples, height, width, channels)`
                - `dict` with key `"image"` as a tensor
            numpy: If `True` (default), returned values will be converted to
                `np.ndarray`s or Python primitives if scalars.

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

        outs = super().predict_on_batch(data, **kwargs)

        if numpy:
            for v in outs.values():
                if isinstance(v, tf.RaggedTensor):
                    outs["n_valid"] = v.row_lengths()
                    break
            outs = sleap.nn.data.utils.unrag_example(outs, numpy=True)

        return outs

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
    ):
        """Save the frozen graph of a model.

        Args:
            save_path: Path to output directory to store the frozen graph
            signatures: String defining the input and output types for
                computation.
            save_traces: If `True` (default) the SavedModel will store the
                function traces for each layer
            model_name: (Optional) Name to give the model. If given, will be
                added to the output json file containing meta information about the
                model
            tensors: (Optional) Dictionary describing the predicted tensors (see
                sleap.nn.data.utils.describe_tensors as an example)
            unrag_outputs: If `True` (default), any ragged tensors will be
                converted to normal tensors and padded with NaNs


        Notes:
            This function call writes relevant meta data to an `info.json` file
            in the given save_path in addition to the frozen_graph.pb file

        """
        os.makedirs(save_path, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save(tmp_dir, save_format="tf", save_traces=save_traces)

            imported = tf.saved_model.load(tmp_dir)

        model = imported.signatures[signatures]

        info = {
            "model_structured_input_signature": model.structured_input_signature,
            "model_structured_outputs_signature": model.structured_outputs,
        }

        if model_name:
            info["model_name"] = model_name
        if tensors:
            info["predicted_tensors"] = tensors

        full_model = tf.function(
            lambda x: sleap.nn.data.utils.unrag_example(model(x), numpy=False)
            if unrag_outputs
            else model(x)
        )

        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        info["frozen_model_inputs"] = frozen_func.inputs
        info["frozen_model_outputs"] = frozen_func.outputs
        info["unragged_outputs"] = unrag_outputs

        with (Path(save_path) / "info.json").open("w") as fp:
            json.dump(
                info, fp, indent=4, sort_keys=True, separators=(",", ": "), default=str
            )

        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=save_path,
            name="frozen_graph.pb",
            as_text=False,
        )


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

            `"instance_peaks"`: The predicted peaks of shape `(samples, 1, nodes, 2)`.
            `"instance_peak_vals": The peak confidence values of shape
            `(samples, 1, nodes)`.

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

        out = {
            "instance_peaks": tf.expand_dims(peaks, axis=1),
            "instance_peak_vals": tf.expand_dims(peak_vals, axis=1),
        }
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

            `"instance_peaks": (batch_size, 1, n_nodes, 2)`: Instance skeleton points.
            `"instance_peak_vals": (batch_size, 1, n_nodes)`: Confidence
                values for the instance skeleton points.
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
                peak_threshold=self.peak_threshold,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
                output_stride=self.confmap_config.model.heads.single_instance.output_stride,
            )
        )

    @property
    def data_config(self) -> DataConfig:
        return self.confmap_config.data

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        return self.confmap_model.keras_model.input.shape[-1] == 1

    @classmethod
    def from_trained_models(
        cls,
        model_path: Text,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
        batch_size: int = 4,
        resize_input_layer: bool = True,
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
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).

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
        if resize_input_layer:
            confmap_model.keras_model = reset_input_layer(
                keras_model=confmap_model.keras_model, new_shape=None
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

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
    ) -> List[sleap.LabeledFrame]:
        """Create labeled frames from a generator that yields inference results.

        This method converts pure arrays into SLEAP-specific data structures.

        Args:
            generator: A generator that returns dictionaries with inference results.
                This should return dictionaries with keys `"video_ind"`, `"frame_ind"`,
                `"instance_peaks"`, and `"instance_peak_vals"`. This can be created
                using the `_predict_generator()` method.
            data_provider: The `sleap.pipelines.Provider` that the predictions are being
                created from. This is used to retrieve the `sleap.Video` instance
                associated with each inference result.

        Returns:
            A list of `sleap.LabeledFrame`s with `sleap.PredictedInstance`s created from
            arrays returned from the inference result generator.
        """
        skeleton = self.confmap_config.data.labels.skeletons[0]
        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    break

                # Loop over frames.
                for video_ind, frame_ind, points, confidences in zip(
                    ex["video_ind"],
                    ex["frame_ind"],
                    ex["instance_peaks"],
                    ex["instance_peak_vals"],
                ):
                    # Loop over instances.
                    if np.isnan(points[0]).all():
                        predicted_instances = []
                    else:
                        predicted_instances = [
                            PredictedInstance.from_numpy(
                                points=points[0],
                                point_confidences=confidences[0],
                                instance_score=np.nansum(confidences[0]),
                                skeleton=skeleton,
                            )
                        ]

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        return predicted_frames

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
        max_instances: Optional[int] = None,
    ):
        super().export_model(
            save_path,
            signatures,
            save_traces,
            model_name,
            tensors,
            unrag_outputs,
            max_instances,
        )

        self.confmap_config.save_json(os.path.join(save_path, "confmap_config.json"))


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
        return_crops: If `True`, the crops and offsets will be returned together with
            the predicted peaks. This is true by default since crops are used
            for finding instance peaks in a top down model. If using a centroid
            only inference model, this should be set to `False`.
        max_instances: If set, determines the max number of instances that a
            multi-instance model returns.
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
        return_crops: bool = True,
        max_instances: Optional[int] = None,
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
                self.keras_model, output_ind=self.confmaps_ind
            )
        self.output_stride = output_stride
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.return_crops = return_crops
        self.max_instances = max_instances

    @tf.function
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

            `"centroids"`: The predicted centroids of shape `(samples, ?, 2)`.
            `"centroid_vals": The centroid confidence values of shape `(samples, ?)`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"centroid_confmaps"` containing a `tf.RaggedTensor` of
            shape `(samples, ?, output_height, output_width, 1)` containing the
            confidence maps predicted by the model.

            If the `return_crops` attribute is set to `True`, the output will
            also contain keys named `crops` and `crop_offsets`. The former is a
            `tf.RaggedTensor` of cropped images of shape `(samples, ?,
            crop_size, crop_size, channels)`. The latter is a `tf.RaggedTensor`
            of Coordinates of the top-left of the crops as `(x, y)` offsets of
            shape `(samples, ?, 2)` for adjusting the predicted peak
            coordinates.
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

        samples = tf.shape(imgs)[0]

        n_peaks = tf.shape(centroid_points)[0]

        if n_peaks > 0:
            if self.max_instances is not None:
                centroid_points = tf.RaggedTensor.from_value_rowids(
                    centroid_points, crop_sample_inds, nrows=samples
                )
                centroid_vals = tf.RaggedTensor.from_value_rowids(
                    centroid_vals, crop_sample_inds, nrows=samples
                )

                _centroid_vals = tf.TensorArray(
                    size=samples,
                    dtype=tf.float32,
                    infer_shape=False,
                    element_shape=[None],
                )

                _centroid_points = tf.TensorArray(
                    size=samples,
                    dtype=tf.float32,
                    infer_shape=False,
                    element_shape=[None, 2],
                )

                _row_ids = tf.TensorArray(
                    size=samples,
                    dtype=tf.int32,
                    infer_shape=False,
                    element_shape=[None],
                )

                for sample in range(samples):
                    n_centroids = tf.shape(centroid_vals[sample])[0]
                    if self.max_instances < n_centroids:
                        top_points = tf.math.top_k(
                            centroid_vals[sample],
                            k=self.max_instances,
                        )
                        top_inds = top_points.indices

                        _centroid_vals = _centroid_vals.write(
                            sample, tf.gather(centroid_vals[sample], top_inds)
                        )

                        _centroid_points = _centroid_points.write(
                            sample, tf.gather(centroid_points[sample], top_inds)
                        )

                        _row_ids = _row_ids.write(
                            sample, tf.fill([len(top_inds)], sample)
                        )
                    else:
                        _centroid_vals = _centroid_vals.write(
                            sample, centroid_vals[sample]
                        )
                        _centroid_points = _centroid_points.write(
                            sample, centroid_points[sample]
                        )
                        _row_ids = _row_ids.write(
                            sample, tf.fill([n_centroids], sample)
                        )

                centroid_vals = _centroid_vals.concat()
                centroid_points = _centroid_points.concat()
                crop_sample_inds = _row_ids.concat()

                n_peaks = tf.shape(crop_sample_inds)[0]

                crop_offsets = centroid_points - (self.crop_size / 2)

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
        centroids = tf.RaggedTensor.from_value_rowids(
            centroid_points, crop_sample_inds, nrows=samples
        )
        centroid_vals = tf.RaggedTensor.from_value_rowids(
            centroid_vals, crop_sample_inds, nrows=samples
        )

        outputs = dict(centroids=centroids, centroid_vals=centroid_vals)
        if self.return_confmaps:
            # Return confidence maps with outputs.
            cms = tf.RaggedTensor.from_value_rowids(
                cms, crop_sample_inds, nrows=samples
            )
            outputs["centroid_confmaps"] = cms

        if self.return_crops:
            # return crops and offsets
            crops = tf.RaggedTensor.from_value_rowids(
                crops, crop_sample_inds, nrows=samples
            )
            crop_offsets = tf.RaggedTensor.from_value_rowids(
                crop_offsets, crop_sample_inds, nrows=samples
            )

            outputs["crops"] = crops
            outputs["crop_offsets"] = crop_offsets

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
                self.keras_model, output_ind=self.confmaps_ind
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
        if "centroid_vals" in inputs:
            outputs["centroid_vals"] = inputs["centroid_vals"]
        if "centroid_confmaps" in inputs:
            outputs["centroid_confmaps"] = inputs["centroid_confmaps"]
        if self.return_confmaps:
            cms = tf.RaggedTensor.from_value_rowids(
                cms, crop_sample_inds, nrows=samples
            )
            outputs["instance_confmaps"] = cms
        return outputs


class CentroidInferenceModel(InferenceModel):
    """Centroid only instance prediction model.

    This model encapsulates the first step in a top-down approach where instances are detected by
    local peak detection of an anchor point and then cropped.

    Attributes:
        centroid_crop: A centroid cropping layer. This can be either `CentroidCrop` or
            `CentroidCropGroundTruth`. This layer takes the full image as input and
            outputs a set of centroids and cropped boxes.
    """

    def __init__(self, centroid_crop: Union[CentroidCrop, CentroidCropGroundTruth]):
        super().__init__()
        self.centroid_crop = centroid_crop

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
        """
        if isinstance(example, tf.Tensor):
            example = dict(image=example)

        crop_output = self.centroid_crop(example)

        return crop_output


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
            if "instances" in example:
                peaks_output = self.instance_peaks(example, crop_output)
            else:
                raise ValueError(
                    "Ground truth data was not detected... "
                    "Please load both models when predicting on non-ground-truth data."
                )
        else:
            peaks_output = self.instance_peaks(crop_output)
        return peaks_output


@attr.s(auto_attribs=True)
class TopDownPredictor(Predictor):
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
        max_instances: If not `None`, discard instances beyond this count when
            predicting, regardless of whether filtering is done at the tracking stage.
            This is useful for preventing extraneous instances from being created when
            tracking is not being applied.
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
    max_instances: Optional[int] = None

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
                max_instances=self.max_instances,
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

    @property
    def data_config(self) -> DataConfig:
        return (
            self.centroid_config.data
            if self.centroid_config
            else self.confmap_config.data
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
        resize_input_layer: bool = True,
        max_instances: Optional[int] = None,
    ) -> "TopDownPredictor":
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
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).
            max_instances: If not `None`, discard instances beyond this count when
                predicting, regardless of whether filtering is done at the tracking
                stage. This is useful for preventing extraneous instances from being
                created when tracking is not being applied.

        Returns:
            An instance of `TopDownPredictor` with the loaded models.

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
            if resize_input_layer:
                # Reset input layer dimensions to be more flexible
                centroid_model.keras_model = reset_input_layer(
                    keras_model=centroid_model.keras_model, new_shape=None
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
            if resize_input_layer:
                # Reset input layer dimensions to be more flexible
                confmap_model.keras_model = reset_input_layer(
                    keras_model=confmap_model.keras_model, new_shape=None
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
            max_instances=max_instances,
        )
        obj._initialize_inference_model()
        return obj

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        is_gray = False
        if self.centroid_model is not None:
            is_gray = self.centroid_model.keras_model.input.shape[-1] == 1
        else:
            is_gray = self.confmap_model.keras_model.input.shape[-1] == 1
        return is_gray

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
        if self.data_config.preprocessing.resize_and_pad_to_target:
            pipeline += SizeMatcher.from_config(
                config=self.data_config.preprocessing,
                provider=data_provider,
                points_key=None,
            )

        pipeline += Normalizer(
            ensure_float=False,
            ensure_grayscale=self.is_grayscale,
            ensure_rgb=(not self.is_grayscale),
        )

        if self.centroid_model is None:
            anchor_part = self.confmap_config.data.instance_cropping.center_on_part
            pipeline += InstanceCentroidFinder(
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

        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    break

                if "n_valid" in ex:
                    ex["instance_peaks"] = [
                        x[:n] for x, n in zip(ex["instance_peaks"], ex["n_valid"])
                    ]
                    ex["instance_peak_vals"] = [
                        x[:n] for x, n in zip(ex["instance_peak_vals"], ex["n_valid"])
                    ]
                    ex["centroids"] = [
                        x[:n] for x, n in zip(ex["centroids"], ex["n_valid"])
                    ]
                    ex["centroid_vals"] = [
                        x[:n] for x, n in zip(ex["centroid_vals"], ex["n_valid"])
                    ]

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
                        if np.isnan(pts).all():
                            continue

                        predicted_instances.append(
                            PredictedInstance.from_numpy(
                                points=pts,
                                point_confidences=confs,
                                instance_score=score,
                                skeleton=skeleton,
                            )
                        )

                    if self.tracker:
                        # Set tracks for predicted instances in this frame.
                        predicted_instances = self.tracker.track(
                            untracked_instances=predicted_instances,
                            img_hw=ex["image"].shape[-3:-1],
                            img=image,
                            t=frame_ind,
                        )

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        if self.tracker:
            self.tracker.final_pass(predicted_frames)

        return predicted_frames

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
        max_instances: Optional[int] = None,
    ):
        super().export_model(
            save_path,
            signatures,
            save_traces,
            model_name,
            tensors,
            unrag_outputs,
            max_instances,
        )

        if self.confmap_config is not None:
            self.confmap_config.save_json(
                os.path.join(save_path, "confmap_config.json")
            )
        if self.centroid_config is not None:
            self.centroid_config.save_json(
                os.path.join(save_path, "centroid_config.json")
            )


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
        return_paf_graph: If `True`, the part affinity field graph will be returned
            together with the predicted instances. The graph is obtained by parsing the
            part affinity fields with the `paf_scorer` instance and is an intermediate
            representation used during instance grouping.
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
        return_paf_graph: bool = False,
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
        self.return_paf_graph = return_paf_graph

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

            If `BottomUpInferenceLayer.return_paf_graph` is `True`, the predicted PAF
            graph will be returned in the `"peaks"`, `"peak_vals"`, `"peak_channel_inds"`,
            `"edge_inds"`, `"edge_peak_inds"` and `"line_scores"` keys.
        """
        cms, pafs, offsets = self.forward_pass(data)
        peaks, peak_vals, peak_channel_inds = self.find_peaks(cms, offsets)
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
            edge_inds,
            edge_peak_inds,
            line_scores,
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
        if self.return_paf_graph:
            out["peaks"] = peaks
            out["peak_vals"] = peak_vals
            out["peak_channel_inds"] = peak_channel_inds
            out["edge_inds"] = edge_inds
            out["edge_peak_inds"] = edge_peak_inds
            out["line_scores"] = line_scores
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
class BottomUpPredictor(Predictor):
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
            as a fraction of the image size. Candidate connections longer than this
            length will be penalized during matching.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.
        paf_line_points: Number of points to sample along the line integral.
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs. Useful for rejecting spurious detections when
            there are no better ones.
        max_instances: If not `None`, discard instances beyond this count when
            predicting, regardless of whether filtering is done at the tracking stage.
            This is useful for preventing extraneous instances from being created when
            tracking is not being applied.
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
    max_edge_length_ratio: float = 0.25
    dist_penalty_weight: float = 1.0
    paf_line_points: int = 10
    min_line_scores: float = 0.25
    max_instances: Optional[int] = None

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained model and configuration."""
        self.inference_model = BottomUpInferenceModel(
            BottomUpInferenceLayer(
                keras_model=self.bottomup_model.keras_model,
                paf_scorer=PAFScorer.from_config(
                    self.bottomup_config.model.heads.multi_instance,
                    max_edge_length_ratio=self.max_edge_length_ratio,
                    dist_penalty_weight=self.dist_penalty_weight,
                    n_points=self.paf_line_points,
                    min_instance_peaks=0,
                    min_line_scores=self.min_line_scores,
                ),
                input_scale=self.bottomup_config.data.preprocessing.input_scaling,
                pad_to_stride=self.bottomup_model.maximum_stride,
                peak_threshold=self.peak_threshold,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
                cm_output_stride=self.bottomup_config.model.heads.multi_instance.confmaps.output_stride,
                paf_output_stride=self.bottomup_config.model.heads.multi_instance.pafs.output_stride,
            )
        )

    @property
    def data_config(self) -> DataConfig:
        return self.bottomup_config.data

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        return self.bottomup_model.keras_model.input.shape[-1] == 1

    @classmethod
    def from_trained_models(
        cls,
        model_path: Text,
        batch_size: int = 4,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
        max_edge_length_ratio: float = 0.25,
        dist_penalty_weight: float = 1.0,
        paf_line_points: int = 10,
        min_line_scores: float = 0.25,
        resize_input_layer: bool = True,
        max_instances: Optional[int] = None,
    ) -> "BottomUpPredictor":
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
            max_edge_length_ratio: The maximum expected length of a connected pair of
                points as a fraction of the image size. Candidate connections longer
                than this length will be penalized during matching.
            dist_penalty_weight: A coefficient to scale weight of the distance penalty
                as a scalar float. Set to values greater than 1.0 to enforce the
                distance penalty more strictly.
            paf_line_points: Number of points to sample along the line integral.
            min_line_scores: Minimum line score (between -1 and 1) required to form a
                match between candidate point pairs. Useful for rejecting spurious
                detections when there are no better ones.
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).
            max_instances: If not `None`, discard instances beyond this count when
                predicting, regardless of whether filtering is done at the tracking
                stage. This is useful for preventing extraneous instances from being
                created when tracking is not being applied.

        Returns:
            An instance of `BottomUpPredictor` with the loaded model.
        """
        # Load bottomup model.
        bottomup_config = TrainingJobConfig.load_json(model_path)
        bottomup_keras_model_path = get_keras_model_path(model_path)
        bottomup_model = Model.from_config(bottomup_config.model)
        bottomup_model.keras_model = tf.keras.models.load_model(
            bottomup_keras_model_path, compile=False
        )
        if resize_input_layer:
            bottomup_model.keras_model = reset_input_layer(
                keras_model=bottomup_model.keras_model, new_shape=None
            )
        obj = cls(
            bottomup_config=bottomup_config,
            bottomup_model=bottomup_model,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            paf_line_points=paf_line_points,
            min_line_scores=min_line_scores,
            max_instances=max_instances,
        )
        obj._initialize_inference_model()
        return obj

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
        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    break

                if "n_valid" in ex:
                    # Crop possibly variable length results.
                    ex["instance_peaks"] = [
                        x[:n] for x, n in zip(ex["instance_peaks"], ex["n_valid"])
                    ]
                    ex["instance_peak_vals"] = [
                        x[:n] for x, n in zip(ex["instance_peak_vals"], ex["n_valid"])
                    ]
                    ex["instance_scores"] = [
                        x[:n] for x, n in zip(ex["instance_scores"], ex["n_valid"])
                    ]

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
                        if np.isnan(pts).all():
                            continue

                        predicted_instances.append(
                            PredictedInstance.from_numpy(
                                points=pts,
                                point_confidences=confs,
                                instance_score=score,
                                skeleton=skeleton,
                            )
                        )

                    if self.max_instances is not None:
                        # Filter by score.
                        predicted_instances = sorted(
                            predicted_instances, key=lambda x: x.score, reverse=True
                        )
                        predicted_instances = predicted_instances[
                            : min(self.max_instances, len(predicted_instances))
                        ]

                    if self.tracker:
                        # Set tracks for predicted instances in this frame.
                        predicted_instances = self.tracker.track(
                            untracked_instances=predicted_instances,
                            img_hw=ex["image"].shape[-3:-1],
                            img=image,
                            t=frame_ind,
                        )

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        if self.tracker:
            self.tracker.final_pass(predicted_frames)

        return predicted_frames


class BottomUpMultiClassInferenceLayer(InferenceLayer):
    """Keras layer that predicts instances from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps and class maps as output.
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        cm_output_stride: Output stride of the model, denoting the scale of the output
            confidence maps relative to the images (after input scaling). This is used
            for adjusting the peak coordinates to the image grid. This will be inferred
            from the model shapes if not provided.
        class_maps_output_stride: Output stride of the model, denoting the scale of the
            output class maps relative to the images (after input scaling). This is
            used for adjusting the peak coordinates to the class maps grid. This will be
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
        return_class_maps: If `True`, the class maps will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"MultiInstanceConfmapsHead"` in its name.
        class_maps_ind: Index of the output tensor of the model corresponding to class
            maps. If `None` (the default), this will be detected automatically by
            searching for the first tensor that contains `"ClassMapsHead"` in its name.
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
        cm_output_stride: Optional[int] = None,
        class_maps_output_stride: Optional[int] = None,
        peak_threshold: float = 0.2,
        refinement: Optional[str] = "local",
        integral_patch_size: int = 5,
        return_confmaps: bool = False,
        return_class_maps: bool = False,
        confmaps_ind: Optional[int] = None,
        class_maps_ind: Optional[int] = None,
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
        self.class_maps_ind = class_maps_ind
        self.offsets_ind = offsets_ind

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(self.keras_model, "MultiInstanceConfmapsHead")

        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'MultiInstanceConfmapsHead'."
            )

        if self.class_maps_ind is None:
            self.class_maps_ind = find_head(self.keras_model, "ClassMapsHead")

        if self.class_maps_ind is None:
            raise ValueError(
                "Index of the part affinity fields output tensor must be specified if "
                "not named 'ClassMapsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if cm_output_stride is None:
            # Attempt to automatically infer the output stride.
            cm_output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.confmaps_ind
            )
        self.cm_output_stride = cm_output_stride
        if class_maps_output_stride is None:
            # Attempt to automatically infer the output stride.
            class_maps_output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.class_maps_ind
            )
        self.class_maps_output_stride = class_maps_output_stride

        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.return_class_maps = return_class_maps

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
        cms = preds[self.confmaps_ind]
        class_maps = preds[self.class_maps_ind]
        if self.offsets_ind is not None:
            offsets = preds[self.offsets_ind]
        else:
            offsets = None

        if isinstance(cms, list):
            cms = cms[-1]
        if isinstance(class_maps, list):
            class_maps = class_maps[-1]

        return cms, class_maps, offsets

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
                cms, offsets, threshold=self.peak_threshold
            )

        # Adjust for confidence map output stride.
        peaks = peaks * tf.cast(self.cm_output_stride, tf.float32)

        return peaks, peak_vals, peak_sample_inds, peak_channel_inds

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

            If `inference_layer.return_confmaps` is `True`, the predicted confidence
            maps will be returned in the `"confmaps"` key.

            If `inference_layer.return_class_maps` is `True`, the predicted class maps
            will be returned in the `"class_maps"` key.
        """
        cms, class_maps, offsets = self.forward_pass(data)
        peaks, peak_vals, peak_sample_inds, peak_channel_inds = self.find_peaks(
            cms, offsets
        )
        peaks = peaks / tf.cast(self.class_maps_output_stride, tf.float32)
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
        ) = sleap.nn.identity.classify_peaks_from_maps(
            class_maps,
            peaks,
            peak_vals,
            peak_sample_inds,
            peak_channel_inds,
            n_channels=tf.shape(cms)[3],
        )
        predicted_instances = predicted_instances * tf.cast(
            self.class_maps_output_stride, tf.float32
        )

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
        if self.return_class_maps:
            out["class_maps"] = class_maps
        return out


class BottomUpMultiClassInferenceModel(InferenceModel):
    """Bottom-up multi-class instance prediction model.

    This model encapsulates the bottom-up multi-class approach where points are first
    detected by local peak finding and then grouped into instances by their identity
    classifications.

    Attributes:
        inference_layer: A `BottomUpMultiClassInferenceLayer`. This layer takes as input
            a full image and outputs the predicted instances.
    """

    def __init__(self, inference_layer, **kwargs):
        super().__init__(**kwargs)
        self.inference_layer = inference_layer

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

            If `inference_layer.return_confmaps` is `True`, the predicted confidence
            maps will be returned in the `"confmaps"` key.

            If `inference_layer.return_class_maps` is `True`, the predicted class maps
            will be returned in the `"class_maps"` key.
        """
        if isinstance(example, tf.Tensor):
            example = dict(image=example)
        return self.inference_layer(example)


@attr.s(auto_attribs=True)
class BottomUpMultiClassPredictor(Predictor):
    """Bottom-up multi-instance predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained bottom-up multi-instance SLEAP model.

    This should be initialized using the `from_trained_models()` constructor or the
    high-level API (`sleap.load_model`).

    Attributes:
        config: The `sleap.nn.config.TrainingJobConfig` containing the metadata for the
            trained model.
        model: A `sleap.nn.model.Model` instance created from the trained model. If
            `None`, ground truth centroids will be used if available from the data
            source.
        inference_model: A `sleap.nn.inference.BottomUpMultiClassInferenceModel` that
            wraps a trained `tf.keras.Model` to implement preprocessing, peak finding
            and classification.
        pipeline: A `sleap.nn.data.Pipeline` that loads the data and batches input data.
            This will be updated dynamically if new data sources are used.
        batch_size: The default batch size to use when loading data for inference.
            Higher values increase inference speed at the cost of higher memory usage.
        peak_threshold: Minimum confidence map value to consider a local peak as valid.
        integral_refinement: If `True`, peaks will be refined with integral regression.
            If `False`, `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        integral_patch_size: Size of patches to crop around each rough peak for integral
            refinement as an integer scalar.
        tracks: If provided, instances will be created using these track instances. If
            not, instances will be assigned tracks from the provider if possible.
    """

    config: TrainingJobConfig
    model: Model
    inference_model: Optional[BottomUpMultiClassInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2
    batch_size: int = 4
    integral_refinement: bool = True
    integral_patch_size: int = 5
    tracks: Optional[List[sleap.Track]] = None

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained model and configuration."""
        self.inference_model = BottomUpMultiClassInferenceModel(
            BottomUpMultiClassInferenceLayer(
                keras_model=self.model.keras_model,
                input_scale=self.config.data.preprocessing.input_scaling,
                pad_to_stride=self.model.maximum_stride,
                peak_threshold=self.peak_threshold,
                refinement="integral" if self.integral_refinement else "local",
                integral_patch_size=self.integral_patch_size,
                cm_output_stride=self.config.model.heads.multi_class_bottomup.confmaps.output_stride,
                class_maps_output_stride=self.config.model.heads.multi_class_bottomup.class_maps.output_stride,
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
        resize_input_layer: bool = True,
    ) -> "BottomUpMultiClassPredictor":
        """Create predictor from a saved model.

        Args:
            model_path: Path to a model folder or training job JSON file inside a model
                folder. This folder should contain `training_config.json` and
                `best_model.h5` files for a trained model.
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
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).

        Returns:
            An instance of `BottomUpPredictor` with the loaded model.
        """
        # Load bottomup model.
        config = TrainingJobConfig.load_json(model_path)
        keras_model_path = get_keras_model_path(model_path)
        model = Model.from_config(config.model)
        model.keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
        if resize_input_layer:
            model.keras_model = reset_input_layer(
                keras_model=model.keras_model, new_shape=None
            )
        obj = cls(
            config=config,
            model=model,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
        )
        obj._initialize_inference_model()
        return obj

    @property
    def data_config(self) -> DataConfig:
        return self.config.data

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        return self.model.keras_model.input.shape[-1] == 1

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
        skeleton = self.config.data.labels.skeletons[0]
        tracks = self.tracks
        if tracks is None:
            if hasattr(data_provider, "tracks"):
                tracks = data_provider.tracks
            elif (
                self.config.model.heads.multi_class_bottomup.class_maps.classes
                is not None
            ):
                names = self.config.model.heads.multi_class_bottomup.class_maps.classes
                tracks = [sleap.Track(name=n, spawned_on=0) for n in names]

        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    return

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
                    for i, (pts, confs, score) in enumerate(
                        zip(points, confidences, scores)
                    ):
                        if np.isnan(pts).all():
                            continue
                        track = None
                        if tracks is not None and len(tracks) >= (i - 1):
                            track = tracks[i]
                        predicted_instances.append(
                            PredictedInstance.from_numpy(
                                points=pts,
                                point_confidences=confs,
                                instance_score=np.nanmean(confs),
                                skeleton=skeleton,
                                track=track,
                                tracking_score=np.nanmean(score),
                            )
                        )

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        return predicted_frames


class TopDownMultiClassFindPeaks(InferenceLayer):
    """Keras layer that predicts and classifies peaks from images using a trained model.

    This layer encapsulates all of the inference operations required for generating
    predictions from a centered instance confidence map and multi-class model. This
    includes preprocessing, model forward pass, peak finding, coordinate adjustment, and
    classification.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            centered instance confidence maps and classification.
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
        return_class_vectors: If `True`, the classification probabilities will be
            returned together with the predicted peaks. This will not line up with the
            grouped instances, for which the associtated class probabilities will always
            be returned in `"instance_scores"`.
        confmaps_ind: Index of the output tensor of the model corresponding to
            confidence maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"CenteredInstanceConfmapsHead"` in its name.
        offsets_ind: Index of the output tensor of the model corresponding to
            offset regression maps. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"OffsetRefinementHead"` in its name. If the head is not present, the method
            specified in the `refinement` attribute will be used.
        class_vectors_ind: Index of the output tensor of the model corresponding to the
            classification vectors. If `None` (the default), this will be detected
            automatically by searching for the first tensor that contains
            `"ClassVectorsHead"` in its name.
        optimal_grouping: If `True` (the default), group peaks from classification
            probabilities. If saving a frozen graph of the model, this will be
            overridden to `False`.
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
        return_class_vectors: bool = False,
        confmaps_ind: Optional[int] = None,
        offsets_ind: Optional[int] = None,
        class_vectors_ind: Optional[int] = None,
        optimal_grouping: bool = True,
        **kwargs,
    ):
        super().__init__(
            keras_model=keras_model, input_scale=input_scale, pad_to_stride=1, **kwargs
        )
        self.peak_threshold = peak_threshold
        self.refinement = refinement
        self.integral_patch_size = integral_patch_size
        self.return_confmaps = return_confmaps
        self.return_class_vectors = return_class_vectors
        self.confmaps_ind = confmaps_ind
        self.class_vectors_ind = class_vectors_ind
        self.offsets_ind = offsets_ind
        self.optimal_grouping = optimal_grouping

        if self.confmaps_ind is None:
            self.confmaps_ind = find_head(
                self.keras_model, "CenteredInstanceConfmapsHead"
            )
        if self.confmaps_ind is None:
            raise ValueError(
                "Index of the confidence maps output tensor must be specified if not "
                "named 'CenteredInstanceConfmapsHead'."
            )

        if self.class_vectors_ind is None:
            self.class_vectors_ind = find_head(self.keras_model, "ClassVectorsHead")
        if self.class_vectors_ind is None:
            raise ValueError(
                "Index of the classifications output tensor must be specified if not "
                "named 'ClassVectorsHead'."
            )

        if self.offsets_ind is None:
            self.offsets_ind = find_head(self.keras_model, "OffsetRefinementHead")

        if output_stride is None:
            # Attempt to automatically infer the output stride.
            output_stride = get_model_output_stride(
                self.keras_model, output_ind=self.confmaps_ind
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
                `tf.Tensor` of shape `(samples, n_classes, nodes, 2)`. Instances will
                be ordered by class and will be filled with `NaN` where not found.
            `"instance_peak_vals"`: The value of the confidence maps at the predicted
                peaks for each instance in the batch as a `tf.Tensor` of shape
                `(samples, n_classes, nodes)`.

            If provided (e.g., from an input `CentroidCrop` layer), the centroids that
            generated the crops will also be included in the keys `"centroids"` and
            `"centroid_vals"`.

            If the `return_confmaps` attribute is set to `True`, the output will also
            contain a key named `"instance_confmaps"` containing a `tf.RaggedTensor` of
            shape `(samples, ?, output_height, output_width, nodes)` containing the
            confidence maps predicted by the model.

            If the `return_class_vectors` attribute is set to `True`, the output will also
            contain a key named `"class_vectors"` containing the full classification
            probabilities for all crops.

            If the `optimal_grouping` attribute is set to `True`, peaks are
            grouped from classification properties. This is overridden to False
            if exporting a frozen graph to allow for tracing. Note: If set to False
            this will change the output dict keys and shapes.
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
        cms = out[self.confmaps_ind]
        peak_class_probs = out[self.class_vectors_ind]
        offsets = None
        if self.offsets_ind is not None:
            offsets = out[self.offsets_ind]

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
                cms, offsets, threshold=self.peak_threshold
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

        if self.optimal_grouping:
            # Group peaks from classification probabilities.
            (
                points,
                point_vals,
                class_probs,
            ) = sleap.nn.identity.classify_peaks_from_vectors(
                peak_points, peak_vals, peak_class_probs, crop_sample_inds, samples
            )

            # Build outputs.
            outputs = {
                "instance_peaks": points,
                "instance_peak_vals": point_vals,
                "instance_scores": class_probs,
            }

        else:
            outputs = {
                "instance_peaks": peak_points,
                "instance_peak_vals": peak_vals,
                "instance_scores": peak_class_probs,
            }

        if "centroids" in inputs:
            outputs["centroids"] = inputs["centroids"]
        if "centroids" in inputs:
            outputs["centroid_vals"] = inputs["centroid_vals"]
        if self.return_confmaps:
            cms = tf.RaggedTensor.from_value_rowids(
                cms, crop_sample_inds, nrows=samples
            )
            outputs["instance_confmaps"] = cms
        if self.return_class_vectors and self.optimal_grouping:
            outputs["class_vectors"] = peak_class_probs
        return outputs


class TopDownMultiClassInferenceModel(InferenceModel):
    """Top-down instance prediction model.

    This model encapsulates the top-down approach where instances are first detected by
    local peak detection of an anchor point and then cropped. These instance-centered
    crops are then passed to an instance peak detector which is trained to detect all
    remaining body parts for the instance that is centered within the crop.

    Attributes:
        centroid_crop: A centroid cropping layer. This can be either `CentroidCrop` or
            `CentroidCropGroundTruth`. This layer takes the full image as input and
            outputs a set of centroids and cropped boxes.
        instance_peaks: A instance peak detection and classification layer, an instance
            of `TopDownMultiClassFindPeaks`. This layer takes as input the output of the
            centroid cropper and outputs the detected peaks and classes for the
            instances within each crop.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, CentroidCropGroundTruth],
        instance_peaks: TopDownMultiClassFindPeaks,
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
        peaks_output = self.instance_peaks(crop_output)
        return peaks_output

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
    ):
        self.instance_peaks.optimal_grouping = False

        super().export_model(
            save_path, signatures, save_traces, model_name, tensors, unrag_outputs
        )


@attr.s(auto_attribs=True)
class TopDownMultiClassPredictor(Predictor):
    """Top-down multi-instance predictor with classification.

    This high-level class handles initialization, preprocessing and tracking using a
    trained top-down multi-instance classification SLEAP model.

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
        inference_model: A `TopDownMultiClassInferenceModel` that wraps a trained
            `tf.keras.Model` to implement preprocessing, centroid detection, cropping,
            peak finding and classification.
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
        tracks: If provided, instances will be created using these track instances. If
            not, instances will be assigned tracks from the provider if possible.
    """

    centroid_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    centroid_model: Optional[Model] = attr.ib(default=None)
    confmap_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    confmap_model: Optional[Model] = attr.ib(default=None)
    inference_model: Optional[TopDownMultiClassInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    tracker: Optional[Tracker] = attr.ib(default=None, init=False)
    batch_size: int = 4
    peak_threshold: float = 0.2
    integral_refinement: bool = True
    integral_patch_size: int = 5
    tracks: Optional[List[sleap.Track]] = None

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained models and configuration."""
        use_gt_centroid = self.centroid_config is None
        use_gt_confmap = self.confmap_config is None
        if use_gt_confmap:
            raise ValueError(
                "Both a centroid and a confidence map model must be provided to "
                "initialize a TopDownMultiClassPredictor."
            )

        if use_gt_centroid:
            centroid_crop_layer = CentroidCropGroundTruth(
                crop_size=self.confmap_config.data.instance_cropping.crop_size
            )
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

        cfg = self.confmap_config
        instance_peaks_layer = TopDownMultiClassFindPeaks(
            keras_model=self.confmap_model.keras_model,
            input_scale=cfg.data.preprocessing.input_scaling,
            peak_threshold=self.peak_threshold,
            output_stride=cfg.model.heads.multi_class_topdown.confmaps.output_stride,
            refinement="integral" if self.integral_refinement else "local",
            integral_patch_size=self.integral_patch_size,
            return_confmaps=False,
        )

        self.inference_model = TopDownMultiClassInferenceModel(
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
        resize_input_layer: bool = True,
    ) -> "TopDownMultiClassPredictor":
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
            resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
                resized to (None, None, None, num_color_channels).

        Returns:
            An instance of `TopDownMultiClassPredictor` with the loaded models.

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
            if resize_input_layer:
                centroid_model.keras_model = reset_input_layer(
                    keras_model=centroid_model.keras_model, new_shape=None
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
            if resize_input_layer:
                confmap_model.keras_model = reset_input_layer(
                    keras_model=confmap_model.keras_model, new_shape=None
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

    @property
    def data_config(self) -> DataConfig:
        return (
            self.centroid_config.data
            if self.centroid_config
            else self.confmap_config.data
        )

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        is_gray = False
        if self.centroid_model is not None:
            is_gray = self.centroid_model.keras_model.input.shape[-1] == 1
        else:
            is_gray = self.confmap_model.keras_model.input.shape[-1] == 1
        return is_gray

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

        tracks = self.tracks
        if tracks is None:
            if hasattr(data_provider, "tracks"):
                tracks = data_provider.tracks
            elif (
                self.confmap_config.model.heads.multi_class_topdown.class_vectors.classes
                is not None
            ):
                names = (
                    self.confmap_config.model.heads.multi_class_topdown.class_vectors.classes
                )
                tracks = [sleap.Track(name=n, spawned_on=0) for n in names]

        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    break

                # Loop over frames.
                for (
                    image,
                    video_ind,
                    frame_ind,
                    centroid_vals,
                    points,
                    confidences,
                    scores,
                ) in zip(
                    ex["image"],
                    ex["video_ind"],
                    ex["frame_ind"],
                    ex["centroid_vals"],
                    ex["instance_peaks"],
                    ex["instance_peak_vals"],
                    ex["instance_scores"],
                ):
                    # Loop over instances.
                    predicted_instances = []
                    for i, (pts, centroid_val, confs, score) in enumerate(
                        zip(points, centroid_vals, confidences, scores)
                    ):
                        if np.isnan(pts).all():
                            continue
                        track = None
                        if tracks is not None and len(tracks) >= (i - 1):
                            track = tracks[i]
                        predicted_instances.append(
                            PredictedInstance.from_numpy(
                                points=pts,
                                point_confidences=confs,
                                instance_score=centroid_val,
                                skeleton=skeleton,
                                track=track,
                                tracking_score=score,
                            )
                        )

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        return predicted_frames

    def export_model(
        self,
        save_path: str,
        signatures: str = "serving_default",
        save_traces: bool = True,
        model_name: Optional[str] = None,
        tensors: Optional[Dict[str, str]] = None,
        unrag_outputs: bool = True,
        max_instances: Optional[int] = None,
    ):
        super().export_model(
            save_path,
            signatures,
            save_traces,
            model_name,
            tensors,
            unrag_outputs,
            max_instances,
        )

        if self.confmap_config is not None:
            self.confmap_config.save_json(
                os.path.join(save_path, "confmap_config.json")
            )
        if self.centroid_config is not None:
            self.centroid_config.save_json(
                os.path.join(save_path, "centroid_config.json")
            )


def make_model_movenet(model_name: str) -> tf.keras.Model:
    """Load a MoveNet model by name.

    Args:
        model_name: Name of the model ("lightning" or "thunder")

    Returns:
        A tf.keras.Model ready for inference.
    """
    model_path = MOVENET_MODELS[model_name]["model_path"]
    image_size = MOVENET_MODELS[model_name]["image_size"]

    x_in = tf.keras.layers.Input([image_size, image_size, 3], name="image")

    x = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, dtype=tf.int32), name="cast_to_int32"
    )(x_in)
    layer = hub.KerasLayer(
        model_path,
        signature="serving_default",
        output_key="output_0",
        name="movenet_layer",
    )
    x = layer(x)

    def split_outputs(x):
        x_ = tf.reshape(x, [-1, 17, 3])
        keypoints = tf.gather(x_, [1, 0], axis=-1)
        keypoints *= image_size
        scores = tf.squeeze(tf.gather(x_, [2], axis=-1), axis=-1)
        return keypoints, scores

    x = tf.keras.layers.Lambda(split_outputs, name="keypoints_and_scores")(x)
    model = tf.keras.Model(x_in, x)
    return model


class MoveNetInferenceLayer(InferenceLayer):
    """Inference layer for applying single instance models.

    This layer encapsulates all of the inference operations requires for generating
    predictions from a single instance confidence map model. This includes
    preprocessing, model forward pass, peak finding and coordinate adjustment.

    Attributes:
        keras_model: A `tf.keras.Model` that accepts rank-4 images as input and predicts
            rank-4 confidence maps as output. This should be a model that is trained on
            single instance confidence maps.
        model_name: Variable indicating which model of MoveNet to use, either "lightning"
            or "thunder."
        input_scale: Float indicating if the images should be resized before being
            passed to the model.
        pad_to_stride: If not 1, input image will be paded to ensure that it is
            divisible by this value (after scaling). This should be set to the max
            stride of the model.
        ensure_grayscale: Boolean indicating whether the type of input data is grayscale
            or not.
        ensure_float: Boolean indicating whether the type of data is float or not.
    """

    def __init__(self, model_name="lightning"):
        self.keras_model = make_model_movenet(model_name)
        self.model_name = model_name
        self.image_size = MOVENET_MODELS[model_name]["image_size"]
        super().__init__(
            keras_model=self.keras_model,
            input_scale=1.0,
            pad_to_stride=1,
            ensure_grayscale=False,
            ensure_float=False,
        )

    def call(self, ex):
        if type(ex) == dict:
            img = ex["image"]

        else:
            img = ex

        points, confidences = super().call(img)
        points = tf.expand_dims(points, axis=1)  # (batch, 1, nodes, 2)
        confidences = tf.expand_dims(confidences, axis=1)  # (batch, 1, nodes)
        return {"instance_peaks": points, "confidences": confidences}


class MoveNetInferenceModel(InferenceModel):
    """MoveNet prediction model.

    This model encapsulates the basic MoveNet approach. The images are passed to a model
    which is trained to detect all body parts (17 joints in total).

    Attributes:
        inference_layer: A MoveNet layer. This layer takes as input full images/videos and
        outputs the detected peaks.
    """

    def __init__(self, inference_layer, **kwargs):
        super().__init__(**kwargs)
        self.inference_layer = inference_layer

    @property
    def model_name(self):
        return self.inference_layer.model_name

    @property
    def image_size(self):
        return self.inference_layer.image_size

    def call(self, x):
        return self.inference_layer(x)


@attr.s(auto_attribs=True)
class MoveNetPredictor(Predictor):
    """MoveNet predictor.

    This high-level class handles initialization, preprocessing and tracking using a
    trained MoveNet model.
    This should be initialized using the `from_trained_models()` constructor or the
    high-level API (`sleap.load_model`).

    Attributes:
        inference_model: A `sleap.nn.inference.MoveNetInferenceModel` that wraps
            a trained `tf.keras.Model` to implement preprocessing and peak finding.
        pipeline: A `sleap.nn.data.Pipeline` that loads the data and batches input data.
            This will be updated dynamically if new data sources are used.
        peak_threshold: Minimum confidence map value to consider a global peak as valid.
        batch_size: The default batch size to use when loading data for inference.
            Higher values increase inference speed at the cost of higher memory usage.
        model_name: Variable indicating which model of MoveNet to use, either "lightning"
            or "thunder."
    """

    inference_model: Optional[MoveNetInferenceModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2
    batch_size: int = 1
    model_name: str = "lightning"

    def _initialize_inference_model(self):
        """Initialize the inference model from the trained model and configuration."""
        # Force batch size to be 1 since that's what the underlying model expects.
        self.batch_size = 1
        self.inference_model = MoveNetInferenceModel(
            MoveNetInferenceLayer(
                model_name=self.model_name,
            )
        )

    @property
    def data_config(self) -> DataConfig:
        if self.inference_model is None:
            self._initialize_inference_model()

        data_config = DataConfig()
        data_config.preprocessing.resize_and_pad_to_target = True
        data_config.preprocessing.target_height = self.inference_model.image_size
        data_config.preprocessing.target_width = self.inference_model.image_size
        return data_config

    @property
    def is_grayscale(self) -> bool:
        """Return whether the model expects grayscale inputs."""
        return False

    @classmethod
    def from_trained_models(
        cls, model_name: Text, peak_threshold: float = 0.2
    ) -> "MoveNetPredictor":
        """Create the predictor from a saved model.

        Args:
            model_name: Variable indicating which model of MoveNet to use, either "lightning"
                or "thunder."
            peak_threshold: Minimum confidence map value to consider a global peak as
                valid.

        Returns:
            An instance of`MoveNetPredictor` with the models loaded.
        """

        obj = cls(
            model_name=model_name,
            peak_threshold=peak_threshold,
            batch_size=1,
        )
        obj._initialize_inference_model()
        return obj

    def _make_labeled_frames_from_generator(
        self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
    ) -> List[sleap.LabeledFrame]:
        skeleton = MOVENET_SKELETON
        predicted_frames = []

        def _object_builder():
            while True:
                ex = prediction_queue.get()
                if ex is None:
                    break

                # Loop over frames.
                for video_ind, frame_ind, points, confidences in zip(
                    ex["video_ind"],
                    ex["frame_ind"],
                    ex["instance_peaks"],
                    ex["confidences"],
                ):
                    # Filter out points with low confidences
                    points[confidences < self.peak_threshold] = np.nan

                    # Create predicted instances from MoveNet predictions
                    if np.isnan(points).all():
                        predicted_instances = []
                    else:
                        predicted_instances = [
                            PredictedInstance.from_numpy(
                                points=points[0],  # (nodes, 2)
                                point_confidences=confidences[0],  # (nodes,)
                                instance_score=np.nansum(confidences[0]),  # ()
                                skeleton=skeleton,
                            )
                        ]

                    predicted_frames.append(
                        LabeledFrame(
                            video=data_provider.videos[video_ind],
                            frame_idx=frame_ind,
                            instances=predicted_instances,
                        )
                    )

        # Set up threaded object builder.
        prediction_queue = Queue()
        object_builder = Thread(target=_object_builder)
        object_builder.start()

        # Loop over batches.
        try:
            for ex in generator:
                prediction_queue.put(ex)

        except KeyError as e:
            # Gracefully handle seeking errors by early termination.
            if "Unable to load frame" in str(e):
                pass  # TODO: Print warning obeying verbosity? (This code path is also
                # called for interactive prediction where we don't want any spam.)
            else:
                raise

        finally:
            prediction_queue.put(None)
            object_builder.join()

        return predicted_frames


def load_model(
    model_path: Union[str, List[str]],
    batch_size: int = 4,
    peak_threshold: float = 0.2,
    refinement: str = "integral",
    tracker: Optional[str] = None,
    tracker_window: int = 5,
    tracker_max_instances: Optional[int] = None,
    disable_gpu_preallocation: bool = True,
    progress_reporting: str = "rich",
    resize_input_layer: bool = True,
    max_instances: Optional[int] = None,
) -> Predictor:
    """Load a trained SLEAP model.

    Args:
        model_path: Path to model or list of path to models that were trained by SLEAP.
            These should be the directories that contain `training_job.json` and
            `best_model.h5`. Special cases of non-SLEAP models include "movenet-thunder"
            and "movenet-lightning".
        batch_size: Number of frames to predict at a time. Larger values result in
            faster inference speeds, but require more memory.
        peak_threshold: Minimum confidence map value to consider a peak as valid.
        refinement: If `"integral"`, peak locations will be refined with integral
            regression. If `"local"`, peaks will be refined with quarter pixel local
            gradient offset. This has no effect if the model has an offset regression
            head.
        tracker: Name of the tracker to use with the inference model. Must be one of
            `"simple"` or `"flow"`. If `None`, no identity tracking across frames will
            be performed.
        tracker_window: Number of frames of history to use when tracking. No effect when
            `tracker` is `None`.
        tracker_max_instances: If not `None`, create at most this many tracks.
        disable_gpu_preallocation: If `True` (the default), initialize the GPU and
            disable preallocation of memory. This is necessary to prevent freezing on
            some systems with low GPU memory and has negligible impact on performance.
            If `False`, no GPU initialization is performed. No effect if running in
            CPU-only mode.
        progress_reporting: Mode of inference progress reporting. If `"rich"` (the
            default), an updating progress bar is displayed in the console or notebook.
            If `"json"`, a JSON-serialized message is printed out which can be captured
            for programmatic progress monitoring. If `"none"`, nothing is displayed
            during inference -- this is recommended when running on clusters or headless
            machines where the output is captured to a log file.
        resize_input_layer: If True, the the input layer of the `tf.Keras.model` is
            resized to (None, None, None, num_color_channels).
        max_instances: If not `None`, discard instances beyond this count when
            predicting, regardless of whether filtering is done at the tracking stage.
            This is useful for preventing extraneous instances from being created when
            tracking is not being applied.

    Returns:
        An instance of a `Predictor` based on which model type was detected.

        If this is a top-down model, paths to the centroids model as well as the
        centered instance model must be provided. A `TopDownPredictor` instance will be
        returned.

        If this is a bottom-up model, a `BottomUpPredictor` will be returned.

        If this is a single-instance model, a `SingleInstancePredictor` will be
        returned.

        If a `tracker` is specified, the predictor will also run identity tracking over
        time.

    See also: TopDownPredictor, BottomUpPredictor, SingleInstancePredictor
    """

    def unpack_sleap_model(model_path):
        """Returns uncompressed ZIP packaged model path.

        Also returns `TemporaryDirectory` where model was extracted to.
        """
        if isinstance(model_path, str):
            model_paths = [model_path]
        else:
            model_paths = model_path

        # Uncompress ZIP packaged models.
        tmp_dirs = []
        for i, model_path in enumerate(model_paths):
            mp = Path(model_path)
            if model_path.endswith(".zip"):
                # Create temp dir on demand.
                tmp_dir = tempfile.TemporaryDirectory()
                tmp_dirs.append(tmp_dir)

                # Remove the temp dir when program exits in case something goes wrong.
                atexit.register(shutil.rmtree, tmp_dir.name, ignore_errors=True)

                # Extract and replace in the list.
                shutil.unpack_archive(model_path, extract_dir=tmp_dir.name)
                unzipped_mp = Path(tmp_dir.name, mp.name).with_suffix("")
                if Path(unzipped_mp, "best_model.h5").exists():
                    unzipped_model_path = str(unzipped_mp)
                else:
                    unzipped_model_path = str(unzipped_mp.parent)
                model_paths[i] = unzipped_model_path

        return model_paths, tmp_dirs

    tmp_dirs = []
    if "movenet" in model_path:
        model_paths = model_path
    else:
        model_paths, tmp_dirs = unpack_sleap_model(model_path)

    if disable_gpu_preallocation:
        sleap.disable_preallocation()

    predictor: Predictor = Predictor.from_model_paths(
        model_paths,
        peak_threshold=peak_threshold,
        integral_refinement=refinement == "integral",
        batch_size=batch_size,
        resize_input_layer=resize_input_layer,
        max_instances=max_instances,
    )
    predictor.verbosity = progress_reporting
    if tracker is not None:
        use_max_tracker = tracker_max_instances is not None
        if use_max_tracker and not tracker.endswith("maxtracks"):
            # Append maxtracks to the tracker name to use the right tracker variants.
            tracker += "maxtracks"

        predictor.tracker = Tracker.make_tracker_by_name(
            tracker=tracker,
            track_window=tracker_window,
            post_connect_single_breaks=True,
            max_tracking=use_max_tracker,
            max_tracks=tracker_max_instances,
            # clean_instance_count=tracker_max_instances,
        )

    # Remove temp dirs.
    for tmp_dir in tmp_dirs:
        tmp_dir.cleanup()

    return predictor


def export_model(
    model_path: Union[str, List[str]],
    save_path: str = "exported_model",
    signatures: str = "serving_default",
    save_traces: bool = True,
    model_name: Optional[str] = None,
    tensors: Optional[Dict[str, str]] = None,
    unrag_outputs: bool = True,
    max_instances: Optional[int] = None,
):
    """High level export of a trained SLEAP model as a frozen graph.

    Args:
        model_path: Path to model or list of path to models that were trained by SLEAP.
            These should be the directories that contain `training_job.json` and
            `best_model.h5`.
        save_path: Path to output directory to store the frozen graph.
        signatures: String defining the input and output types for computation.
        save_traces: If `True` (default) the SavedModel will store the function traces
            for each layer.
        model_name: (Optional) Name to give the model. If given, will be added to the
            output json file containing meta information about the model.
        tensors: (Optional) Dictionary describing the predicted tensors (see
            sleap.nn.data.utils.describe_tensors as an example).
        unrag_outputs: If `True` (default), any ragged tensors will be
            converted to normal tensors and padded with NaNs
        max_instances: If set, determines the max number of instances that a
            multi-instance model returns. This is enforced during centroid
            cropping and therefore only compatible with TopDown models.
    """
    predictor = load_model(model_path, resize_input_layer=False)

    predictor.export_model(
        save_path,
        signatures,
        save_traces,
        model_name,
        tensors,
        unrag_outputs,
        max_instances,
    )


def export_cli(args: Optional[list] = None):
    """CLI for sleap-export."""

    parser = _make_export_cli_parser()

    args, _ = parser.parse_known_args(args=args)
    print("Args:")
    pprint(vars(args))
    print()

    export_model(
        args.models,
        args.export_path,
        unrag_outputs=(not args.ragged),
        max_instances=args.max_instances,
    )


def _make_export_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for sleap-export CLI."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        help=(
            "Path to trained model directory (with training_config.json). "
            "Multiple models can be specified, each preceded by --model."
        ),
    )
    parser.add_argument(
        "-e",
        "--export_path",
        type=str,
        nargs="?",
        default="exported_model",
        help=(
            "Path to output directory where the frozen model will be exported to. "
            "Defaults to a folder named 'exported_model'."
        ),
    )
    parser.add_argument(
        "-r",
        "--ragged",
        action="store_true",
        default=False,
        help=(
            "Keep tensors ragged if present. If ommited, convert ragged tensors"
            " into regular tensors with NaN padding."
        ),
    )
    parser.add_argument(
        "-n",
        "--max_instances",
        type=int,
        help=(
            "Limit maximum number of instances in multi-instance models. "
            "Not available for ID models. Defaults to None."
        ),
    )

    return parser


def _make_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        The `argparse.ArgumentParser` that defines the CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default="",
        help=(
            "Path to data to predict on. This can be a labels (.slp) file or any "
            "supported video format."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        help=(
            "Path to trained model directory (with training_config.json). "
            "Multiple models can be specified, each preceded by --model."
        ),
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help=(
            "List of frames to predict when running on a video. Can be specified as a "
            "comma separated list (e.g. 1,2,3) or a range separated by hyphen (e.g., "
            "1-3, for 1,2,3). If not provided, defaults to predicting on the entire "
            "video."
        ),
    )
    parser.add_argument(
        "--only-labeled-frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on user labeled frames when running on labels dataset. "
            "This is useful for generating predictions to compare against ground truth."
        ),
    )
    parser.add_argument(
        "--only-suggested-frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on unlabeled suggested frames when running on labels "
            "dataset. This is useful for generating predictions for initialization "
            "during labeling."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "The output filename to use for the predicted data. If not provided, "
            "defaults to '[data_path].predictions.slp'."
        ),
    )
    parser.add_argument(
        "--no-empty-frames",
        action="store_true",
        default=False,
        help=(
            "Clear any empty frames that did not have any detected instances before "
            "saving to output."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["none", "rich", "json"],
        default="rich",
        help=(
            "Verbosity of inference progress reporting. 'none' does not output "
            "anything during inference, 'rich' displays an updating progress bar, "
            "and 'json' outputs the progress as a JSON encoded response to the "
            "console."
        ),
    )
    parser.add_argument(
        "--video.dataset", type=str, default=None, help="The dataset for HDF5 videos."
    )
    parser.add_argument(
        "--video.input_format",
        type=str,
        default="channels_last",
        help="The input_format for HDF5 videos.",
    )
    parser.add_argument(
        "--video.index",
        type=str,
        default="",
        help=(
            "Integer index of video in .slp file to predict on. To be used with an .slp"
            " path as an alternative to specifying the video path."
        ),
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
        "--gpu",
        type=str,
        default="auto",
        help=(
            "Run training on the i-th GPU on the system. If 'auto', run on the GPU with"
            " the highest percentage of available memory."
        ),
    )
    parser.add_argument(
        "--max_edge_length_ratio",
        type=float,
        default=0.25,
        help="The maximum expected length of a connected pair of points "
        "as a fraction of the image size. Candidate connections longer "
        "than this length will be penalized during matching. "
        "Only applies to bottom-up (PAF) models.",
    )
    parser.add_argument(
        "--dist_penalty_weight",
        type=float,
        default=1.0,
        help="A coefficient to scale weight of the distance penalty. Set "
        "to values greater than 1.0 to enforce the distance penalty more strictly. "
        "Only applies to bottom-up (PAF) models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Number of frames to predict at a time. Larger values result in faster "
            "inference speeds, but require more memory."
        ),
    )
    parser.add_argument(
        "--open-in-gui",
        action="store_true",
        help="Open the resulting predictions in the GUI when finished.",
    )
    parser.add_argument(
        "--peak_threshold",
        type=float,
        default=0.2,
        help="Minimum confidence map value to consider a peak as valid.",
    )
    parser.add_argument(
        "-n",
        "--max_instances",
        type=int,
        help=(
            "Limit maximum number of instances in multi-instance models. "
            "Not available for ID models. Defaults to None."
        ),
    )

    # Deprecated legacy args. These will still be parsed for backward compatibility but
    # are hidden from the CLI help.
    parser.add_argument(
        "--labels",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--topdown.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bottomup.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--topdown.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bottomup.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    # Add tracker args.
    Tracker.add_cli_parser_args(parser, arg_scope="tracking")

    return parser


def _make_provider_from_cli(args: argparse.Namespace) -> Tuple[Provider, str]:
    """Make data provider from parsed CLI args.

    Args:
        args: Parsed CLI namespace.

    Returns:
        `(provider_list, data_path_list, output_path_list)` where `provider_list` contains the data providers,
        `data_path_list` contains the paths to the specified data, and the `output_path_list` contains the list
        of output paths if a CSV file with a column of output paths was provided; otherwise, `output_path_list`
        defaults to None
    """

    # Figure out which input path to use.
    data_path = args.data_path

    if data_path is None or data_path == "":
        raise ValueError(
            "You must specify a path to a video or a labels dataset. "
            "Run 'sleap-track -h' to see full command documentation."
        )

    data_path_obj = Path(data_path)

    # Set output_path_list to None as a default to return later
    output_path_list = None

    # Check that input value is valid
    if not data_path_obj.exists():
        raise ValueError("Path to data_path does not exist")

    elif data_path_obj.is_file():
        # If the file is a CSV file, check for data_paths and output_paths
        if data_path_obj.suffix.lower() == ".csv":
            try:
                data_path_column = None
                # Read the CSV file
                df = pd.read_csv(data_path)

                # collect data_paths from column
                for col_index in range(df.shape[1]):
                    path_str = df.iloc[0, col_index]
                    if Path(path_str).exists():
                        data_path_column = df.columns[col_index]
                        break
                if data_path_column is None:
                    raise ValueError(
                        f"Column containing valid data_paths does not exist in the CSV file: {data_path}"
                    )
                raw_data_path_list = df[data_path_column].tolist()

                # optional output_path column to specify multiple output_paths
                output_path_column_index = df.columns.get_loc(data_path_column) + 1
                if (
                    output_path_column_index < df.shape[1]
                    and df.iloc[:, output_path_column_index].dtype == object
                ):
                    # Ensure the next column exists
                    output_path_list = df.iloc[:, output_path_column_index].tolist()
                else:
                    output_path_list = None

            except pd.errors.EmptyDataError as e:
                raise ValueError(f"CSV file is empty: {data_path}. Error: {e}") from e

        # If the file is a text file, collect data_paths
        elif data_path_obj.suffix.lower() == ".txt":
            try:
                with open(data_path_obj, "r") as file:
                    raw_data_path_list = [line.strip() for line in file.readlines()]
            except Exception as e:
                raise ValueError(
                    f"Error reading text file: {data_path}. Error: {e}"
                ) from e
        else:
            raw_data_path_list = [data_path_obj.as_posix()]

        raw_data_path_list = [Path(p) for p in raw_data_path_list]

    # Check for multiple video inputs
    # Compile file(s) into a list for later iteration
    elif data_path_obj.is_dir():
        raw_data_path_list = [
            file_path for file_path in data_path_obj.iterdir() if file_path.is_file()
        ]

    # Provider list to accomodate multiple video inputs
    provider_list = []
    data_path_list = []
    for file_path in raw_data_path_list:
        # Create a provider for each file
        if file_path.as_posix().endswith(".slp") and len(raw_data_path_list) > 1:
            print(f"slp file skipped: {file_path.as_posix()}")

        elif file_path.as_posix().endswith(".slp"):
            labels = sleap.load_file(file_path.as_posix())

            if args.only_labeled_frames:
                provider_list.append(LabelsReader.from_user_labeled_frames(labels))
            elif args.only_suggested_frames:
                provider_list.append(LabelsReader.from_unlabeled_suggestions(labels))
            elif getattr(args, "video.index") != "":
                provider_list.append(
                    VideoReader(
                        video=labels.videos[int(getattr(args, "video.index"))],
                        example_indices=frame_list(args.frames),
                    )
                )
            else:
                provider_list.append(LabelsReader(labels))

            data_path_list.append(file_path)

        else:
            try:
                video_kwargs = dict(
                    dataset=vars(args).get("video.dataset"),
                    input_format=vars(args).get("video.input_format"),
                )
                provider_list.append(
                    VideoReader.from_filepath(
                        filename=file_path.as_posix(),
                        example_indices=frame_list(args.frames),
                        **video_kwargs,
                    )
                )
                print(f"Video: {file_path.as_posix()}")
                data_path_list.append(file_path)
                # TODO: Clean this up.
            except Exception:
                print(f"Error reading file: {file_path.as_posix()}")

    return provider_list, data_path_list, output_path_list


def _make_predictor_from_cli(args: argparse.Namespace) -> Predictor:
    """Make predictor from parsed CLI args.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The `Predictor` created from loaded models.
    """
    peak_threshold = None
    for deprecated_arg in [
        "single.peak_threshold",
        "topdown.peak_threshold",
        "bottomup.peak_threshold",
    ]:
        val = getattr(args, deprecated_arg, None)
        if val is not None:
            peak_threshold = val
    if peak_threshold is None:
        peak_threshold = args.peak_threshold

    batch_size = None
    for deprecated_arg in [
        "single.batch_size",
        "topdown.batch_size",
        "bottomup.batch_size",
    ]:
        val = getattr(args, deprecated_arg, None)
        if val is not None:
            batch_size = val
    if batch_size is None:
        batch_size = args.batch_size

    if args.models is not None:
        predictor: Predictor = load_model(
            args.models,
            peak_threshold=peak_threshold,
            batch_size=batch_size,
            refinement="integral",
            progress_reporting=args.verbosity,
            max_instances=args.max_instances,
        )

        if type(predictor) == BottomUpPredictor:
            predictor.inference_model.bottomup_layer.paf_scorer.max_edge_length_ratio = (
                args.max_edge_length_ratio
            )
            predictor.inference_model.bottomup_layer.paf_scorer.dist_penalty_weight = (
                args.dist_penalty_weight
            )
    else:
        # TODO(LM): create predictor that only uses tracker - based off load_model
        pass

    return predictor


def _make_tracker_from_cli(args: argparse.Namespace) -> Optional[Tracker]:
    """Make tracker from parsed CLI arguments.

    Args:
        args: Parsed CLI namespace.

    Returns:
        An instance of `Tracker` or `None` if tracking method was not specified.
    """
    policy_args = make_scoped_dictionary(vars(args), exclude_nones=True)
    if "tracking" in policy_args:
        tracker = Tracker.make_tracker_by_name(**policy_args["tracking"])
        return tracker
    return None


def main(args: Optional[list] = None):
    """Entrypoint for `sleap-track` CLI for running inference.

    Args:
        args: A list of arguments to be passed into sleap-track.
    """
    t0 = time()
    start_timestamp = str(datetime.now())
    print("Started inference at:", start_timestamp)

    # Setup CLI.
    parser = _make_cli_parser()

    # Parse inputs.
    args, _ = parser.parse_known_args(args)
    print("Args:")
    pprint(vars(args))
    print()

    # Setup devices.
    if args.cpu or not sleap.nn.system.is_gpu_system():
        sleap.nn.system.use_cpu_only()
    else:
        if args.first_gpu:
            sleap.nn.system.use_first_gpu()
        elif args.last_gpu:
            sleap.nn.system.use_last_gpu()
        else:
            if args.gpu == "auto":
                free_gpu_memory = sleap.nn.system.get_gpu_memory()
                if len(free_gpu_memory) > 0:
                    gpu_ind = np.argmax(free_gpu_memory)
                    mem = free_gpu_memory[gpu_ind]
                    logger.info(
                        f"Auto-selected GPU {gpu_ind} with {mem} MiB of free memory."
                    )
                else:
                    logger.info(
                        "Failed to query GPU memory from nvidia-smi. Defaulting to "
                        "first GPU."
                    )
                    gpu_ind = 0
            else:
                gpu_ind = int(args.gpu)
            sleap.nn.system.use_gpu(gpu_ind)
    sleap.disable_preallocation()

    print("Versions:")
    sleap.versions()
    print()

    print("System:")
    sleap.nn.system.summary()
    print()

    # Setup data loader.
    provider_list, data_path_list, output_path_list = _make_provider_from_cli(args)

    output_path = None

    # if output_path has not been extracted from a csv file yet
    if output_path_list is None and args.output is not None:
        output_path = args.output
        output_path_obj = Path(output_path)

        # check if output_path is valid before running inference
        if Path(output_path).is_file() and len(data_path_list) > 1:
            raise ValueError(
                "output_path argument must be a directory if multiple video inputs are given"
            )

    # Setup tracker.
    tracker = _make_tracker_from_cli(args)

    if args.models is not None and "movenet" in args.models[0]:
        args.models = args.models[0]

    # Either run inference (and tracking) or just run tracking (if using an existing prediction where inference has already been run)
    if args.models is not None:

        # Run inference on all files inputed
        for i, (data_path, provider) in enumerate(zip(data_path_list, provider_list)):
            # Setup models.
            data_path_obj = Path(data_path)
            predictor = _make_predictor_from_cli(args)
            predictor.tracker = tracker

            # Run inference!
            labels_pr = predictor.predict(provider)

            # if output path was not provided, create an output path
            if output_path is None:
                # if output path was not provided, create an output path
                if output_path_list:
                    output_path = output_path_list[i]

                else:
                    output_path = data_path_obj.with_suffix(".predictions.slp")

                output_path_obj = Path(output_path)

            # if output_path was provided and multiple inputs were provided, create a directory to store outputs
            elif len(data_path_list) > 1:
                output_path_obj = Path(output_path)
                output_path = (
                    output_path_obj
                    / (data_path_obj.with_suffix(".predictions.slp")).name
                )
                output_path_obj = Path(output_path)
                # Create the containing directory if needed.
                output_path_obj.parent.mkdir(exist_ok=True, parents=True)

            labels_pr.provenance["model_paths"] = predictor.model_paths
            labels_pr.provenance["predictor"] = type(predictor).__name__

            if args.no_empty_frames:
                # Clear empty frames if specified.
                labels_pr.remove_empty_frames()

            finish_timestamp = str(datetime.now())
            total_elapsed = time() - t0
            print("Finished inference at:", finish_timestamp)
            print(f"Total runtime: {total_elapsed} secs")
            print(f"Predicted frames: {len(labels_pr)}/{len(provider)}")

            # Add provenance metadata to predictions.
            labels_pr.provenance["sleap_version"] = sleap.__version__
            labels_pr.provenance["platform"] = platform.platform()
            labels_pr.provenance["command"] = " ".join(sys.argv)
            labels_pr.provenance["data_path"] = data_path_obj.as_posix()
            labels_pr.provenance["output_path"] = output_path_obj.as_posix()
            labels_pr.provenance["total_elapsed"] = total_elapsed
            labels_pr.provenance["start_timestamp"] = start_timestamp
            labels_pr.provenance["finish_timestamp"] = finish_timestamp

            print("Provenance:")
            pprint(labels_pr.provenance)
            print()

            labels_pr.provenance["args"] = vars(args)

            # Save results.
            try:
                labels_pr.save(output_path)
            except Exception:
                print("WARNING: Provided output path invalid.")
                fallback_path = data_path_obj.with_suffix(".predictions.slp")
                labels_pr.save(fallback_path)
            print("Saved output:", output_path)

            if args.open_in_gui:
                subprocess.call(["sleap-label", output_path])

            # Reset output_path for next iteration
            output_path = args.output

    # running tracking on existing prediction file
    elif getattr(args, "tracking.tracker") is not None:
        provider = provider_list[0]
        data_path = data_path_list[0]

        # Load predictions
        data_path = args.data_path
        print("Loading predictions...")
        labels_pr = sleap.load_file(data_path)
        frames = sorted(labels_pr.labeled_frames, key=lambda lf: lf.frame_idx)

        print("Starting tracker...")
        frames = run_tracker(frames=frames, tracker=tracker)
        tracker.final_pass(frames)

        labels_pr = Labels(labeled_frames=frames)

        if output_path is None:
            output_path = f"{data_path}.{tracker.get_name()}.slp"

        if args.no_empty_frames:
            # Clear empty frames if specified.
            labels_pr.remove_empty_frames()

        finish_timestamp = str(datetime.now())
        total_elapsed = time() - t0
        print("Finished inference at:", finish_timestamp)
        print(f"Total runtime: {total_elapsed} secs")
        print(f"Predicted frames: {len(labels_pr)}/{len(provider)}")

        # Add provenance metadata to predictions.
        labels_pr.provenance["sleap_version"] = sleap.__version__
        labels_pr.provenance["platform"] = platform.platform()
        labels_pr.provenance["command"] = " ".join(sys.argv)
        labels_pr.provenance["data_path"] = data_path
        labels_pr.provenance["output_path"] = output_path
        labels_pr.provenance["total_elapsed"] = total_elapsed
        labels_pr.provenance["start_timestamp"] = start_timestamp
        labels_pr.provenance["finish_timestamp"] = finish_timestamp

        print("Provenance:")
        pprint(labels_pr.provenance)
        print()

        labels_pr.provenance["args"] = vars(args)

        # Save results.
        labels_pr.save(output_path)

        print("Saved output:", output_path)

        if args.open_in_gui:
            subprocess.call(["sleap-label", output_path])

    else:
        raise ValueError(
            "Neither tracker type nor path to trained models specified. "
            "Use \"sleap-track -m path/to/model ...' to specify models to use. "
            "To retrack on predictions, must specify tracker. "
            "Use \"sleap-track --tracking.tracker ...' to specify tracker to use."
        )

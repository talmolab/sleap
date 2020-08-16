"""Inference pipelines and utilities."""

import attr
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Text, Optional, List, Dict, Union

import tensorflow as tf
import numpy as np

import sleap
from sleap import util
from sleap.nn.config import TrainingJobConfig
from sleap.nn.model import Model
from sleap.nn.tracking import Tracker, run_tracker
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
    PartAffinityFieldInstanceGrouper,
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
                        f"Finished {i} examples in {elapsed_time:.2f} seconds (inference + postprocessing)"
                    )
                    if elapsed_time:
                        logger.info(f"examples/s = {i/elapsed_time}")


def make_grouped_labeled_frame(
    video_ind: int,
    frame_ind: int,
    frame_examples: List[Dict[Text, tf.Tensor]],
    videos: List[sleap.Video],
    skeleton: "Skeleton",
    points_key: Text,
    point_confidences_key: Text,
    image_key: Optional[Text] = None,
    instance_score_key: Optional[Text] = None,
    tracker: Optional[Tracker] = None,
) -> List[sleap.LabeledFrame]:

    predicted_frames = []

    # Create predicted instances from examples in the current frame.
    predicted_instances = []
    img = None
    for example in frame_examples:
        if instance_score_key is None:
            instance_scores = np.nansum(example[point_confidences_key].numpy(), axis=-1)
        else:
            instance_scores = example[instance_score_key]

        if example[points_key].ndim == 3:
            for points, confidences, instance_score in zip(
                example[points_key], example[point_confidences_key], instance_scores
            ):
                if not np.isnan(points).all():
                    predicted_instances.append(
                        sleap.PredictedInstance.from_arrays(
                            points=points,
                            point_confidences=confidences,
                            instance_score=instance_score,
                            skeleton=skeleton,
                        )
                    )
        else:
            points = example[points_key]
            confidences = example[point_confidences_key]
            instance_score = instance_scores

            if not np.isnan(points).all():
                predicted_instances.append(
                    sleap.PredictedInstance.from_arrays(
                        points=points,
                        point_confidences=confidences,
                        instance_score=instance_score,
                        skeleton=skeleton,
                    )
                )

        if image_key is not None and image_key in example:
            img = example[image_key]
        else:
            img = None

    if len(predicted_instances) > 0:
        if tracker:
            # Set tracks for predicted instances in this frame.
            predicted_instances = tracker.track(
                untracked_instances=predicted_instances, img=img, t=frame_ind
            )

        # Create labeled frame from predicted instances.
        labeled_frame = sleap.LabeledFrame(
            video=videos[video_ind], frame_idx=frame_ind, instances=predicted_instances
        )

        predicted_frames.append(labeled_frame)

    return predicted_frames


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


class CentroidCrop(tf.keras.layers.Layer):
    def __init__(
        self,
        input_scale,
        max_stride,
        keras_model,
        confmaps_stride,
        peak_threshold,
        crop_size,
    ):
        super().__init__()

        self.input_scale = input_scale
        self.max_stride = max_stride
        self.keras_model = keras_model
        self.confmaps_stride = confmaps_stride
        self.peak_threshold = peak_threshold
        self.crop_size = crop_size

    def call(self, full_imgs):

        if isinstance(full_imgs, dict):
            full_imgs = full_imgs["image"]

        imgs = full_imgs
        if self.input_scale != 1.0:
            imgs = sleap.nn.data.resizing.resize_image(imgs, self.input_scale)
        imgs = sleap.nn.data.resizing.pad_to_stride(imgs, self.max_stride)

        cms = self.keras_model(imgs)

        # Find centroids.
        (
            centroid_points,
            centroid_vals,
            crop_sample_inds,
            _,
        ) = sleap.nn.peak_finding.find_local_peaks_integral(
            cms, threshold=self.peak_threshold
        )

        # Adjust coordinates for confmaps stride.
        centroid_points = (centroid_points * self.confmaps_stride) / self.input_scale

        # Store crop offsets.
        crop_offsets = centroid_points - (self.crop_size / 2)

        # Crop instances around centroids.
        bboxes = sleap.nn.data.instance_cropping.make_centered_bboxes(
            centroid_points, self.crop_size, self.crop_size
        )
        crops = sleap.nn.peak_finding.crop_bboxes(full_imgs, bboxes, crop_sample_inds)

        # Reshape to (n_peaks, crop_height, crop_width, channels)
        n_peaks = tf.shape(centroid_points)[0]
        img_channels = tf.shape(full_imgs)[3]
        crops = tf.reshape(
            crops, [n_peaks, self.crop_size, self.crop_size, img_channels]
        )

        # Group crops by sample.
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

        return dict(
            centroids=centroids,
            centroid_vals=centroid_vals,
            crops=crops,
            crop_offsets=crop_offsets,
        )


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
            example_gt["instances"], axis=1
        )  # (batch_size, 1, n_insts, n_nodes, 2)
        a = a.to_tensor(default_value=tf.cast(np.NaN, tf.float32))
        b = tf.expand_dims(
            tf.expand_dims(crop_output["centroids"], axis=2), axis=2
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


class FindInstancePeaks(tf.keras.layers.Layer):
    def __init__(
        self, peak_threshold, input_scale, keras_model, confmaps_stride,
    ):
        super().__init__()

        self.peak_threshold = peak_threshold

        self.input_scale = input_scale
        self.keras_model = keras_model
        self.confmaps_stride = confmaps_stride

    def call(self, inputs):

        # Unpack inputs.
        crops = inputs["crops"]
        crop_offsets = inputs["crop_offsets"]
        centroids = inputs["centroids"]
        centroid_vals = inputs["centroid_vals"]

        # Flatten crops into (n_peaks, height, width, channels)
        crop_sample_inds = crops.value_rowids()
        samples = crops.nrows()
        crops = crops.merge_dims(0, 1)

        # Preprocess.
        imgs = crops
        if self.input_scale != 1.0:
            imgs = sleap.nn.data.resizing.resize_image(imgs, self.input_scale)

        # Confidence maps estimation.
        cms = self.keras_model(imgs)

        # Peak finding
        peak_points, peak_vals = sleap.nn.peak_finding.find_global_peaks_integral(
            cms, threshold=self.peak_threshold
        )

        # Adjust for scale and  offsets.
        peak_points = (peak_points * self.confmaps_stride) / self.input_scale
        peak_points = peak_points + tf.expand_dims(
            crop_offsets.merge_dims(0, 1), axis=1
        )

        # Pad peaks to full shape (samples, max_instances, nodes, 2).
        peaks = tf.RaggedTensor.from_value_rowids(
            peak_points, crop_sample_inds, nrows=samples
        )
        peak_vals = tf.RaggedTensor.from_value_rowids(
            peak_vals, crop_sample_inds, nrows=samples
        )

        return dict(
            instance_peaks=peaks,
            instance_peak_vals=peak_vals,
            centroids=centroids,
            centroid_vals=centroid_vals,
        )


class TopDownModel(tf.keras.Model):
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
        max_instances: The maximum expected number of instances within a frame. If this
            is set and fewer instances are found, the output will be padded to this
            length to maintain a fixed shape. If this is not set, the outputs will be
            padded to the batch's bounding shape, which may result in errors if using
            `model.predict()` instead of `model()` as the former expects fixed size
            batches. Also note that not setting this value will also result in retracing
            of the compute graph every time there is a new batch shape emitted. See
            `call()` for more information.
    """

    def __init__(
        self,
        centroid_crop: Union[CentroidCrop, CentroidCropGroundTruth],
        instance_peaks: Union[FindInstancePeaks, FindInstancePeaksGroundTruth],
        max_instances: Optional[int] = None,
    ):
        super().__init__()
        self.centroid_crop = centroid_crop
        self.instance_peaks = instance_peaks
        self.max_instances = max_instances
        self.use_gt_instances = isinstance(instance_peaks, FindInstancePeaksGroundTruth)

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
            `"n_valid": (batch_size,)`: Vector denoting how many of the instances were
                found in each image of the batch. This is useful for cropping the padded
                instances from the above keys.

            All of these will be concrete `tf.Tensor`s of the shapes described above.

            When calling on a single batch and if `max_instances` is not set, the
            `n_instances` axis will be the largest number of instances found in the
            batch.

            When `max_instances` is set, `n_instances` will always be that value.

            If calling via `predict()`, the `max_instances` must be set or an exception
            will be raised if the number of instances per frame changes.

            If fewer than `max_instances` are found in a frame, the tensors above will
            be NaN-padded along axis 1 and the true number of instances will be
            indicated in the `"n_valid"` key.
        """
        if isinstance(example, tf.Tensor):
            example = dict(image=example)

        crop_output = self.centroid_crop(example)

        if self.use_gt_instances:
            peaks_output = self.instance_peaks(example, crop_output)
        else:
            peaks_output = self.instance_peaks(crop_output)

        n_valid = peaks_output["instance_peaks"].row_lengths()
        if self.max_instances is None:
            peaks_output = sleap.nn.data.utils.unrag_example(peaks_output)

        else:
            for k, v in peaks_output.items():
                peaks_output[k] = sleap.nn.data.utils.unrag_tensor(
                    v, self.max_instances, axis=1
                )
        peaks_output["n_valid"] = n_valid
        return peaks_output


@attr.s(auto_attribs=True)
class TopdownPredictor(Predictor):
    centroid_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    centroid_model: Optional[Model] = attr.ib(default=None)
    confmap_config: Optional[TrainingJobConfig] = attr.ib(default=None)
    confmap_model: Optional[Model] = attr.ib(default=None)
    topdown_model: Optional[TopDownModel] = attr.ib(default=None)
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    tracker: Optional[Tracker] = attr.ib(default=None, init=False)
    batch_size: int = 16
    peak_threshold: float = 0.2
    integral_refinement: bool = True
    integral_patch_size: int = 5

    @classmethod
    def from_trained_models(
        cls,
        centroid_model_path: Optional[Text] = None,
        confmap_model_path: Optional[Text] = None,
        batch_size: int = 1,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
    ) -> "TopdownPredictor":
        """Create predictor from saved models.

        Args:
            centroid_model_path: Path to centroid model folder.
            confmap_model_path: Path to topdown confidence map model folder.

        Returns:
            An instance of TopdownPredictor with the loaded models.

            One of the two models can be left as None to perform inference with ground
            truth data. This will only work with LabelsReader as the provider.
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
            use_gt_centroid = False
        else:
            centroid_config = None
            centroid_model = None
            use_gt_centroid = True

        if confmap_model_path is not None:
            # Load confmap model.
            confmap_config = TrainingJobConfig.load_json(confmap_model_path)
            confmap_keras_model_path = get_keras_model_path(confmap_model_path)
            confmap_model = Model.from_config(confmap_config.model)
            confmap_model.keras_model = tf.keras.models.load_model(
                confmap_keras_model_path, compile=False
            )
            use_gt_confmap = False
        else:
            confmap_config = None
            confmap_model = None
            use_gt_confmap = True

        if use_gt_centroid:
            centroid_crop_layer = CentroidCropGroundTruth(
                crop_size=confmap_model.instance_cropping.crop_size
            )
        else:
            centroid_crop_layer = CentroidCrop(
                input_scale=centroid_config.data.preprocessing.input_scaling,
                max_stride=centroid_config.data.preprocessing.pad_to_stride,
                keras_model=centroid_model.keras_model,
                confmaps_stride=centroid_config.model.heads.centroid.output_stride,
                peak_threshold=peak_threshold,
                crop_size=confmap_config.data.instance_cropping.crop_size,
            )

        if use_gt_confmap:
            instance_peaks_layer = FindInstancePeaksGroundTruth()
        else:
            instance_peaks_layer = FindInstancePeaks(
                peak_threshold=peak_threshold,
                input_scale=confmap_config.data.preprocessing.input_scaling,
                keras_model=confmap_model.keras_model,
                confmaps_stride=confmap_config.model.heads.centered_instance.output_stride,
            )

        topdown_model = TopDownModel(
            centroid_crop=centroid_crop_layer, instance_peaks=instance_peaks_layer,
        )

        return cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            topdown_model=topdown_model,
            batch_size=batch_size,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
        )

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:

        keep_original_image = self.tracker and self.tracker.uses_image

        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        pipeline += sleap.nn.data.pipelines.Batcher(
            batch_size=self.batch_size, drop_remainder=False, unrag=False
        )

        # Infer colorspace preprocessing if not explicit.
        if not (
            self.confmap_config.data.preprocessing.ensure_rgb
            or self.confmap_config.data.preprocessing.ensure_grayscale
        ):
            if self.confmap_model.keras_model.inputs[0].shape[-1] == 1:
                self.confmap_config.data.preprocessing.ensure_grayscale = True
            else:
                self.confmap_config.data.preprocessing.ensure_rgb = True

        pipeline += Normalizer.from_config(
            self.confmap_config.data.preprocessing, image_key="image"
        )

        pipeline += Prefetcher()

        self.pipeline = pipeline

        return pipeline

    def predict_generator(self, data_provider: Provider):
        if self.pipeline is None:
            if self.centroid_config is not None and self.confmap_config is not None:
                self.make_pipeline()
            else:
                # Pass in data provider when mocking one of the models.
                self.make_pipeline(data_provider=data_provider)

        self.pipeline.providers = [data_provider]

        for ex in self.pipeline.make_dataset():
            preds = self.topdown_model.predict(ex)

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

            yield ex

    def make_labeled_frames_from_generator(self, generator, data_provider):

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

                frame_ind = frame_ind.numpy().squeeze()
                video_ind = video_ind.numpy().squeeze()

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
        data_provider: Provider,
        make_instances: bool = True,
        make_labels: bool = False,
    ):
        t0_gen = time.time()

        if isinstance(data_provider, sleap.Labels):
            data_provider = LabelsReader(data_provider)
        elif isinstance(data_provider, sleap.Video):
            data_provider = VideoReader(data_provider)

        generator = self.predict_generator(data_provider)

        if make_instances or make_labels:
            lfs = self.make_labeled_frames_from_generator(generator, data_provider)
            elapsed = time.time() - t0_gen
            logger.info(
                f"Predicted {len(lfs)} labeled frames in {elapsed:.3f} secs [{len(lfs)/elapsed:.1f} FPS]"
            )

            if make_labels:
                return sleap.Labels(lfs)
            else:
                return lfs

        else:
            examples = list(generator)
            elapsed = time.time() - t0_gen
            logger.info(
                f"Predicted {len(examples)} examples in {elapsed:.3f} secs [{len(examples)/elapsed:.1f} examples/s]"
            )

            return examples


@attr.s(auto_attribs=True)
class BottomupPredictor(Predictor):
    bottomup_config: TrainingJobConfig
    bottomup_model: Model
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    tracker: Optional[Tracker] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2

    @classmethod
    def from_trained_models(cls, bottomup_model_path: Text) -> "BottomupPredictor":
        """Create predictor from saved models."""
        # Load bottomup model.
        bottomup_config = TrainingJobConfig.load_json(bottomup_model_path)
        bottomup_keras_model_path = get_keras_model_path(bottomup_model_path)
        bottomup_model = Model.from_config(bottomup_config.model)
        bottomup_model.keras_model = tf.keras.models.load_model(
            bottomup_keras_model_path, compile=False
        )

        return cls(bottomup_config=bottomup_config, bottomup_model=bottomup_model)

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:
        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        # Infer colorspace preprocessing if not explicit.
        if not (
            self.bottomup_config.data.preprocessing.ensure_rgb
            or self.bottomup_config.data.preprocessing.ensure_grayscale
        ):
            if self.bottomup_model.keras_model.inputs[0].shape[-1] == 1:
                self.bottomup_config.data.preprocessing.ensure_grayscale = True
            else:
                self.bottomup_config.data.preprocessing.ensure_rgb = True

        pipeline += Normalizer.from_config(self.bottomup_config.data.preprocessing)
        pipeline += Resizer.from_config(
            self.bottomup_config.data.preprocessing,
            keep_full_image=False,
            points_key=None,
        )

        pipeline += Prefetcher()

        pipeline += KerasModelPredictor(
            keras_model=self.bottomup_model.keras_model,
            model_input_keys="image",
            model_output_keys=[
                "predicted_confidence_maps",
                "predicted_part_affinity_fields",
            ],
        )
        pipeline += LocalPeakFinder(
            confmaps_stride=self.bottomup_model.heads[0].output_stride,
            peak_threshold=self.peak_threshold,
            confmaps_key="predicted_confidence_maps",
            peaks_key="predicted_peaks",
            peak_vals_key="predicted_peak_confidences",
            peak_sample_inds_key="predicted_peak_sample_inds",
            peak_channel_inds_key="predicted_peak_channel_inds",
            keep_confmaps=False,
        )

        pipeline += LambdaFilter(filter_fn=lambda ex: len(ex["predicted_peaks"]) > 0)

        pipeline += PartAffinityFieldInstanceGrouper.from_config(
            self.bottomup_config.model.heads.multi_instance,
            max_edge_length=128,
            min_edge_score=0.05,
            n_points=10,
            min_instance_peaks=0,
            peaks_key="predicted_peaks",
            peak_scores_key="predicted_peak_confidences",
            channel_inds_key="predicted_peak_channel_inds",
            pafs_key="predicted_part_affinity_fields",
            predicted_instances_key="predicted_instances",
            predicted_peak_scores_key="predicted_peak_scores",
            predicted_instance_scores_key="predicted_instance_scores",
            keep_pafs=False,
        )

        keep_keys = [
            "scale",
            "video_ind",
            "frame_ind",
            "predicted_instances",
            "predicted_peak_scores",
            "predicted_instance_scores",
        ]

        if self.tracker and self.tracker.uses_image:
            keep_keys.append("image")

        pipeline += KeyFilter(keep_keys=keep_keys)

        pipeline += PointsRescaler(
            points_key="predicted_instances", scale_key="scale", invert=True
        )

        self.pipeline = pipeline

        return pipeline

    def make_labeled_frames_from_generator(self, generator, data_provider):
        grouped_generator = group_examples_iter(generator)

        skeleton = self.bottomup_config.data.labels.skeletons[0]

        def make_lfs(video_ind, frame_ind, frame_examples):
            return make_grouped_labeled_frame(
                video_ind=video_ind,
                frame_ind=frame_ind,
                frame_examples=frame_examples,
                videos=data_provider.videos,
                skeleton=skeleton,
                image_key="image",
                points_key="predicted_instances",
                point_confidences_key="predicted_peak_scores",
                instance_score_key="predicted_instance_scores",
                tracker=self.tracker,
            )

        predicted_frames = []
        for (video_ind, frame_ind), grouped_examples in grouped_generator:
            predicted_frames.extend(make_lfs(video_ind, frame_ind, grouped_examples))

        if self.tracker:
            self.tracker.final_pass(predicted_frames)

        return predicted_frames

    def predict_generator(self, data_provider: Provider):
        if self.pipeline is None:
            self.make_pipeline()

        self.pipeline.providers = [data_provider]

        # Yield each example from dataset, catching and logging exceptions
        return safely_generate(self.pipeline.make_dataset())

    def predict(
        self,
        data_provider: Provider,
        make_instances: bool = True,
        make_labels: bool = False,
    ):
        if isinstance(data_provider, sleap.Labels):
            data_provider = LabelsReader(data_provider)
        elif isinstance(data_provider, sleap.Video):
            data_provider = VideoReader(data_provider)
        generator = self.predict_generator(data_provider)

        if make_instances or make_labels:
            lfs = self.make_labeled_frames_from_generator(generator, data_provider)
            if make_labels:
                return sleap.Labels(lfs)
            else:
                return lfs

        return list(generator)


@attr.s(auto_attribs=True)
class SingleInstancePredictor(Predictor):
    confmap_config: TrainingJobConfig
    confmap_model: Model
    pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
    peak_threshold: float = 0.2
    integral_refinement: bool = True
    integral_patch_size: int = 5

    @classmethod
    def from_trained_models(
        cls,
        confmap_model_path: Text,
        peak_threshold: float = 0.2,
        integral_refinement: bool = True,
        integral_patch_size: int = 5,
    ) -> "SingleInstancePredictor":
        """Create predictor from saved models."""
        # Load confmap model.
        confmap_config = TrainingJobConfig.load_json(confmap_model_path)
        confmap_keras_model_path = get_keras_model_path(confmap_model_path)
        confmap_model = Model.from_config(confmap_config.model)
        confmap_model.keras_model = tf.keras.models.load_model(
            confmap_keras_model_path, compile=False
        )

        return cls(
            confmap_config=confmap_config,
            confmap_model=confmap_model,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
        )

    def make_pipeline(self, data_provider: Optional[Provider] = None) -> Pipeline:

        pipeline = Pipeline()
        if data_provider is not None:
            pipeline.providers = [data_provider]

        # Infer colorspace preprocessing if not explicit.
        if not (
            self.confmap_config.data.preprocessing.ensure_rgb
            or self.confmap_config.data.preprocessing.ensure_grayscale
        ):
            if self.confmap_model.keras_model.inputs[0].shape[-1] == 1:
                self.confmap_config.data.preprocessing.ensure_grayscale = True
            else:
                self.confmap_config.data.preprocessing.ensure_rgb = True

        pipeline += Normalizer.from_config(self.confmap_config.data.preprocessing)
        pipeline += Resizer.from_config(
            self.confmap_config.data.preprocessing, points_key=None
        )

        pipeline += Prefetcher()

        pipeline += KerasModelPredictor(
            keras_model=self.confmap_model.keras_model,
            model_input_keys="image",
            model_output_keys="predicted_instance_confidence_maps",
        )
        pipeline += GlobalPeakFinder(
            confmaps_key="predicted_instance_confidence_maps",
            peaks_key="predicted_instance",
            peak_vals_key="predicted_instance_confidences",
            confmaps_stride=self.confmap_model.heads[0].output_stride,
            peak_threshold=self.peak_threshold,
            integral=self.integral_refinement,
            integral_patch_size=self.integral_patch_size,
        )

        pipeline += KeyFilter(
            keep_keys=[
                "scale",
                "video_ind",
                "frame_ind",
                "predicted_instance",
                "predicted_instance_confidences",
            ]
        )

        pipeline += PointsRescaler(
            points_key="predicted_instance", scale_key="scale", invert=True
        )

        self.pipeline = pipeline

        return pipeline

    def make_labeled_frames_from_generator(self, generator, data_provider):
        grouped_generator = group_examples_iter(generator)

        skeleton = self.confmap_config.data.labels.skeletons[0]

        def make_lfs(video_ind, frame_ind, frame_examples):
            return make_grouped_labeled_frame(
                video_ind=video_ind,
                frame_ind=frame_ind,
                frame_examples=frame_examples,
                videos=data_provider.videos,
                skeleton=skeleton,
                points_key="predicted_instance",
                point_confidences_key="predicted_instance_confidences",
            )

        predicted_frames = []
        for (video_ind, frame_ind), grouped_examples in grouped_generator:
            predicted_frames.extend(make_lfs(video_ind, frame_ind, grouped_examples))

        return predicted_frames

    def predict_generator(self, data_provider: Provider):
        if self.pipeline is None:
            self.make_pipeline()

        self.pipeline.providers = [data_provider]

        # Yield each example from dataset, catching and logging exceptions
        return safely_generate(self.pipeline.make_dataset())

    def predict(
        self,
        data_provider: Provider,
        make_instances: bool = True,
        make_labels: bool = False,
    ):
        if isinstance(data_provider, sleap.Labels):
            data_provider = LabelsReader(data_provider)
        elif isinstance(data_provider, sleap.Video):
            data_provider = VideoReader(data_provider)
        generator = self.predict_generator(data_provider)

        if make_instances or make_labels:
            lfs = self.make_labeled_frames_from_generator(generator, data_provider)
            if make_labels:
                return sleap.Labels(lfs)
            else:
                return lfs

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
                default=None,
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


def make_predictor_from_paths(paths) -> Predictor:
    """Builds predictor object from a list of model paths."""
    return make_predictor_from_models(find_heads_for_model_paths(paths))


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
) -> Predictor:
    """Given dict of paths keyed by head name, returns appropriate predictor."""

    def get_relevant_args(key):
        if policy_args is not None and key in policy_args:
            return policy_args[key]
        return dict()

    if "multi_instance" in trained_model_paths:
        predictor = BottomupPredictor.from_trained_models(
            trained_model_paths["multi_instance"], **get_relevant_args("bottomup")
        )
    elif "single_instance" in trained_model_paths:
        predictor = SingleInstancePredictor.from_trained_models(
            confmap_model_path=trained_model_paths["single_instance"],
            **get_relevant_args("single"),
        )
    elif (
        "centroid" in trained_model_paths and "centered_instance" in trained_model_paths
    ):
        predictor = TopdownPredictor.from_trained_models(
            centroid_model_path=trained_model_paths["centroid"],
            confmap_model_path=trained_model_paths["centered_instance"],
            **get_relevant_args("topdown"),
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
        video_predicted_frames = predictor.predict(video_reader)
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

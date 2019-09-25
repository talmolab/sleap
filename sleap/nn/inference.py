import sys
import argparse
import multiprocessing
import os
import json
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import h5py
import cv2
import keras
import tensorflow as tf
import attr

from multiprocessing import Process, Pool
from multiprocessing.pool import AsyncResult, ThreadPool

from time import time, clock
from typing import Any, Dict, List, Union, Optional, Text, Tuple

from sleap.instance import LabeledFrame
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.skeleton import Skeleton
from sleap.util import usable_cpu_count

from sleap.nn.model import ModelOutputType
from sleap.nn.training import TrainingJob
from sleap.nn.tracking import FlowShiftTracker, Track
from sleap.nn.transform import DataTransform

from sleap.nn.datagen import merge_boxes_with_overlap_and_padding
from sleap.nn.peakfinding import find_all_peaks, find_all_single_peaks
from sleap.nn.peakfinding_tf import peak_tf_inference
from sleap.nn.peakmatching import match_single_peaks_all, match_peaks_paf, match_peaks_paf_par, instances_nms
from sleap.nn.util import batch, batch_count, save_visual_outputs


@attr.s(auto_attribs=True)
class InferenceModel:
    """This class provides convenience metadata and methods for running inference from a TrainingJob."""

    job: TrainingJob
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

    def compute_output_shape(self, input_shape: Tuple[int], relative=True) -> Tuple[int]:
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
            self.output_channels)

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
            model_path = os.path.join(self.job.save_dir,
                self.job.best_model_filename)

            # Try the final model if that didn't exist.
            if not os.path.exists(model_path):
                model_path = os.path.join(self.job.save_dir,
                    self.job.final_model_filename)

        # Load from disk.
        keras_model = keras.models.load_model(model_path,
            custom_objects={"tf": tf})
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
            inputs=input_tensor,
            outputs=keras_model(input_tensor))


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
            raise AttributeError("No model loaded. Call inference_model.load_model() first.")

        return self._model_path


    @property
    def trained_input_shape(self) -> Tuple[int]:
        """Returns the shape of the model when it was loaded."""

        if not self._trained_input_shape:
            raise AttributeError("No model loaded. Call inference_model.load_model() first.")

        return self._trained_input_shape

    @property
    def output_channels(self) -> int:
        """Returns the number of output channels of the model."""
        if not self._trained_input_shape:
            raise AttributeError("No model loaded. Call inference_model.load_model() first.")

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
    
    
    def predict(self, X: Union[np.ndarray, List[np.ndarray]],
        batch_size: int = 32,
        normalize: bool = True
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
                    X = X.astype("float32") / 255.
            elif isinstance(X, list):
                for i in range(len(X)):
                    if X[i].dtype == np.dtype("uint8"):
                        X[i] = X[i].astype("float32") / 255.

        return self.keras_model.predict(X, batch_size=batch_size)


@attr.s(auto_attribs=True)
class Predictor:
    """
    The Predictor class takes a set of trained sLEAP models and runs
    the full inference pipeline via the predict or predict_async method.

    Pipeline:

    * Pre-processing to load, crop and scale images
    * Inference to predict confidence maps and part affinity fields,
      and use these to generate PredictedInstances in LabeledFrames
    * Post-processing to collate data from all frames, track instances
      across frames, and save the results

    Args:
        sleap_models: Dict with a TrainingJob for each required
            ModelOutputType; can be used to construct keras model.
        skeleton: The skeleton(s) to use for prediction.
        inference_batch_size: Frames per inference batch
            (GPU memory limited)
        read_chunk_size: How many frames to read into CPU memory at a
            time (CPU memory limited)
        nms_min_thresh: A threshold of non-max suppression peak finding
            in confidence maps. All values below this minimum threshold
            will be set to zero before peak finding algorithm is run.
        nms_kernel_size: Gaussian blur is applied to confidence maps before
            non-max supression peak finding occurs. This is size of the
            kernel applied to the image.
        nms_sigma: For Gassian blur applied to confidence maps, this
            is the standard deviation of the kernel.
        min_score_to_node_ratio: FIXME
        min_score_midpts: FIXME
        min_score_integral: FIXME
        add_last_edge: FIXME
        with_tracking: whether to run tracking after inference
        flow_window: The number of frames that tracking should look back
            when trying to identify instances.
        single_per_crop: FIXME
        output_path: the output path to save the results
        save_confmaps_pafs: whether to save confmaps/pafs
        resize_hack: whether to resize images to power of 2
    """

    training_jobs: Dict[ModelOutputType, TrainingJob] = None
    inference_models: Dict[ModelOutputType, InferenceModel] = attr.ib(default=attr.Factory(dict))

    skeleton: Skeleton = None
    inference_batch_size: int = 2
    read_chunk_size: int = 256
    save_frequency: int = 100 # chunks
    nms_min_thresh = 0.3
    nms_kernel_size: int = 9
    nms_sigma: float = 3.
    min_score_to_node_ratio: float = 0.2
    min_score_midpts: float = 0.05
    min_score_integral: float = 0.6
    add_last_edge: bool = True
    with_tracking: bool = False
    flow_window: int = 15
    single_per_crop: bool = False
    crop_padding: int = 40
    crop_growth: int = 64

    output_path: Optional[str] = None
    save_confmaps_pafs: bool = False
    resize_hack: bool = True
    pool: multiprocessing.Pool = None

    gpu_peak_finding: bool = True
    supersample_window_size: int = 7  # must be odd
    supersample_factor: float = 2  # factor to upsample cropped windows by
    overlapping_instances_nms: bool = True  # suppress overlapping instances

    def __attrs_post_init__(self):

        # Create inference models from the TrainingJob metadata.
        for model_output_type, training_job in self.training_jobs.items():
            self.inference_models[model_output_type] = InferenceModel(job=training_job)
            self.inference_models[model_output_type].load_model()


    def predict(self,
                input_video: Union[dict, Video],
                frames: Optional[List[int]] = None,
                is_async: bool = False) -> List[LabeledFrame]:
        """Run the entire inference pipeline on an input video.

        Args:
            input_video: Either a `Video` object or dict that can be
                converted back to a `Video` object.
            frames (optional): List of frames to predict.
                If None, run entire video.
            is_async (optional): Whether running function from separate
                process. Default is False. If True, we won't spawn
                children.

        Returns:
            A list of LabeledFrames with predicted instances.
        """

        # Check if we have models.
        if len(self.inference_models) == 0:
            logger.warning("Predictor has no model.")
            raise ValueError("Predictor has no model.")

        self.is_async = is_async

        # Initialize parallel pool if needed.
        if not is_async and self.pool is None:
            self.pool = multiprocessing.Pool(processes=usable_cpu_count())

        # Fix the number of threads for OpenCV, not that we are using
        # anything in OpenCV that is actually multi-threaded but maybe
        # we will down the line.
        cv2.setNumThreads(usable_cpu_count())

        logger.info(f"Predict is async: {is_async}")

        # Find out if the images should be grayscale from the first model.
        # TODO: Unify this with input data normalization.
        grayscale = list(self.inference_models.values())[0].is_grayscale

        # Open the video object if needed.
        if isinstance(input_video, Video):
            vid = input_video
        elif isinstance(input_video, dict):
            vid = Video.cattr().structure(input_video, Video)
        elif isinstance(input_video, str):
            vid = Video.from_filename(input_video, grayscale=grayscale)
        else:
            raise AttributeError(f"Unable to load input video: {input_video}")

        # List of frames to process (or entire video if not specified)
        frames = frames or list(range(vid.num_frames))
        logger.info("Opened video:")
        logger.info("  Source: " + str(vid.backend))
        logger.info("  Frames: %d" % len(frames))
        logger.info("  Frame shape (H x W): %d x %d" % (vid.height, vid.width))


        # Initialize tracking
        if self.with_tracking:
            tracker = FlowShiftTracker(window=self.flow_window, verbosity=0)

        if self.output_path:
            # Delete the output file if it exists already
            if os.path.exists(self.output_path):
                os.unlink(self.output_path)
                logger.warning("Deleted existing output: " + self.output_path)

            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            logger.info("Output path: " + self.output_path)

        # Process chunk-by-chunk!
        t0_start = time()
        predicted_frames = []
        num_chunks = batch_count(frames, self.read_chunk_size)

        logger.info("Number of chunks for process: %d" % (num_chunks))

        for chunk, chunk_start, frames_idx in batch(frames, self.read_chunk_size):

            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()

            """
            Step 1: Pre-processing

            Prepare the data we need for inference:
            * load images
            * crop and scale as appropriate

            Results: a list of (images, transform) tuples.

            For instance, if we have different sized crops, we'll need a
            distinct images matrix and transform for each size.
            """

            # Read the next batch of images
            t0 = time()
            imgs_full = vid[frames_idx]
            logger.info("  Read %d frames [%.1fs]" % (len(imgs_full), time() - t0))

            # Transform images (crop or scale)
            t0 = time()

            if ModelOutputType.CENTROIDS in self.inference_models:

                # Use centroid predictions to get subchunks of crops.
                subchunks_to_process = self.centroid_crop_inference(
                    imgs_full, frames_idx)

            else:
                # Create transform object
                transform = DataTransform(frame_idxs=frames_idx)
                subchunks_to_process = [(imgs_full, transform)]

            logger.info("  Transformed images [%.1fs]" % (time() - t0))

            """
            Step 2: Inference

            This is where we predict using the trained models, and then
            convert the outputs of these models to *instances* in *frames*.

            Input: the list of (images, transform) from pre-processing

            Output: a list of LabeledFrames for each (images, transform)
                each of these is a "subchunk"
            """

            subchunk_results = []

            for subchunk_imgs_full, subchunk_transform in subchunks_to_process:

                logger.info(f"  Running inference for subchunk:")
                logger.info(f"    Shape: {subchunk_imgs_full.shape}")
                logger.info(f"    Scale: {subchunk_transform.scale}")

                if ModelOutputType.PART_AFFINITY_FIELD not in self.inference_models:
                    # Pipeline for predicting a single animal in a frame
                    # This uses only confidence maps

                    logger.warning("No PAF model! Running in SINGLE INSTANCE mode.")

                    subchunk_lfs = self.single_instance_inference(
                                            subchunk_imgs_full,
                                            subchunk_transform,
                                            vid)

                else:
                    # Pipeline for predicting multiple animals in a frame
                    # This uses confidence maps and part affinity fields
                    subchunk_lfs = self.multi_instance_inference(
                                            subchunk_imgs_full,
                                            subchunk_transform,
                                            vid)

                logger.info(f"    Subchunk frames with instances found: {len(subchunk_lfs)}")

                subchunk_results.append(subchunk_lfs)

            """
            Step 3: Post-processing

            Here we do steps that potentially involve multiple frames:

            * merge data from separate subchunks
            * track instances across frames
            * save predictions

            Inputs: the lists of lists of LabeledFrames

            Outputs: a single list of LabeledFrames
            """

            # Merge frames from multiple processing subchunks
            predicted_frames_chunk = []
            for subchunk_frames in subchunk_results:
                predicted_frames_chunk.extend(subchunk_frames)
            predicted_frames_chunk = LabeledFrame.merge_frames(predicted_frames_chunk, video=vid)

            logger.info(f"  Instances found on {len(predicted_frames_chunk)} out of {len(imgs_full)} frames.")

            if len(predicted_frames_chunk):

                # Sort by frame index
                predicted_frames_chunk.sort(key=lambda lf: lf.frame_idx)

                # Track
                if self.with_tracking and len(predicted_frames_chunk):
                    t0 = time()
                    tracker.process(imgs_full, predicted_frames_chunk)
                    logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

                # Save
                predicted_frames.extend(predicted_frames_chunk)

                if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                    t0 = time()

                    # TODO: We are re-writing the whole output each time, this is dumb.
                    #  We should save in chunks then combine at the end.
                    labels = Labels(labeled_frames=predicted_frames)
                    if self.output_path is not None:
                        if self.output_path.endswith("json"):
                            Labels.save_json(labels, filename=self.output_path, compress=True)
                        else:
                            Labels.save_hdf5(labels, filename=self.output_path)

                        logger.info("  Saved to: %s [%.1fs]" % (self.output_path, time() - t0))

            elapsed = time() - t0_chunk
            total_elapsed = time() - t0_start
            fps = len(predicted_frames) / total_elapsed
            frames_left = len(frames) - len(predicted_frames)
            eta = (frames_left / fps) if fps > 0 else 0
            logger.info("  Finished chunk [%.1fs / %.1f FPS / ETA: %.1f min]" % (elapsed, fps, eta / 60))

            sys.stdout.flush()

        logger.info("Total: %.1f min" % (total_elapsed / 60))

        # Generate Labels object from predicted frames
        labels = Labels(labeled_frames=predicted_frames)

        # Make sure we only have a single LabeledFrame for each frame idx
        labels.merge_matching_frames()

        if self.is_async:
            return labels.to_dict()
        else:
            return labels

    def predict_async(self, *args, **kwargs) -> Tuple[Pool, AsyncResult]:
        """
        Run the entire inference pipeline on an input file,
        using a background process.

        Args:
            See Predictor.predict().
            Note that video must be string rather than `Video`
            (which doesn't pickle).

        Returns:
            A tuple containing the multiprocessing.Process that is
            running predict, start() has been called. The AysncResult
            object that will contain the result when the job finishes.
        """
        kwargs["is_async"] = True
        if isinstance(kwargs["input_video"], Video):
            # unstructure input_video since it won't pickle
            kwargs["input_video"] = Video.cattr().unstructure(kwargs["input_video"])

        if self.pool is None:
            self.pool = Pool(processes=1)
        result = self.pool.apply_async(self.predict, args=args, kwds=kwargs)

        # Tell the pool to accept no new tasks
        # pool.close()

        return result


    def centroid_crop_inference(self,
                imgs: np.ndarray,
                frames_idx: List[int],
                box_size: int=None,
                do_merge: bool=True) \
                -> List[Tuple[np.ndarray, DataTransform]]:
        """
        Takes stack of images and runs centroid inference to get crops.

        Arguments:
            imgs: stack of images in a numpy matrix

        Returns:
            list of "subchunks", each an (images, transform)-tuple

        Different subchunks can thus have different images sizes,
        which allows us to merge overlapping crops into larger crops.
        """

        # Get inference models with metadata.
        centroid_model = self.inference_models[ModelOutputType.CENTROIDS]
        cm_model = self.inference_models[ModelOutputType.CONFIDENCE_MAP]

        logger.info("  Performing centroid cropping.")

        # TODO: Replace this calculation when model-specific divisibility calculation implemented.
        divisor = 2 ** centroid_model.down_blocks
        crop_within = ((imgs.shape[1] // divisor) * divisor, (imgs.shape[2] // divisor) * divisor)
        logger.info(f"    crop_within: {crop_within}")

        # Create transform
        # This lets us scale the images before we predict centroids,
        # and will also let us map the points on the scaled image to
        # points on the original images so we can crop original images.
        centroid_transform = DataTransform()
        target_shape = (int(imgs.shape[1] * centroid_model.input_scale), int(imgs.shape[2] * centroid_model.input_scale))

        # Scale to match input size of trained centroid model.
        centroid_imgs_scaled = centroid_transform.scale_to(
            imgs=imgs, target_size=target_shape)

        # Predict centroid confidence maps, then find peaks.
        t0 = time()
        centroid_confmaps = centroid_model.predict(centroid_imgs_scaled,
            batch_size=self.inference_batch_size)

        peaks, peak_vals = find_all_peaks(centroid_confmaps,
            min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)

        elapsed = time() - t0
        total_peaks = sum([len(frame_peaks[0]) for frame_peaks in peaks])
        logger.info(f"    Found {total_peaks} centroid peaks ({total_peaks / len(peaks):.2f} centroids/frame) [{elapsed:.2f}s].")

        if box_size is None:
            # Get training bounding box size to determine (min) centroid crop size.
            # TODO: fix this to use a stored value or move this logic elsewhere
            crop_size = int(max(cm_model.trained_input_shape[1:3]) // cm_model.input_scale)
            bb_half = crop_size // 2
            # bb_half = (crop_size + self.crop_padding) // 2
        else:
            bb_half = box_size // 2

        logger.info(f"    Crop box size: {bb_half * 2}")

        # Iterate over each frame to filter bounding boxes
        all_boxes = dict()
        for frame_i, (frame_peaks, frame_peak_vals) in enumerate(zip(peaks, peak_vals)):

            # If we found centroids on this frame...
            if frame_peaks[0].shape[0] > 0:

                # Pad each centroid into a bounding box
                # (We're not using the pad function because it shifts
                # boxes to fit within image.)

                boxes = []
                for peak_i in range(frame_peaks[0].shape[0]):

                    # Rescale peak back onto full-sized image
                    peak_x = int(frame_peaks[0][peak_i][0] / centroid_model.output_scale)
                    peak_y = int(frame_peaks[0][peak_i][1] / centroid_model.output_scale)

                    boxes.append((peak_x - bb_half, peak_y - bb_half,
                                  peak_x + bb_half, peak_y + bb_half))

                if do_merge:
                    # Merge overlapping boxes and pad to multiple of crop size
                    merged_boxes = merge_boxes_with_overlap_and_padding(
                                    boxes=boxes,
                                    pad_factor_box=(self.crop_growth, self.crop_growth),
                                    within=crop_within)

                else:
                    # Just return the boxes centered around each centroid.
                    # Note that these aren't guaranteed to be within the
                    # image bounds, so take care if using these to crop.
                    merged_boxes = boxes

                # Keep track of all boxes, grouped by size and frame idx
                for box in merged_boxes:

                    merged_box_size = (box[2] - box[0], box[3] - box[1])

                    if merged_box_size not in all_boxes:
                        all_boxes[merged_box_size] = dict()
                        logger.info(f"    Found box size: {merged_box_size}")

                    if frame_i not in all_boxes[merged_box_size]:
                        all_boxes[merged_box_size][frame_i] = []

                    all_boxes[merged_box_size][frame_i].append(box)

        logger.info(f"    Found {len(all_boxes)} box sizes after merging.")

        subchunks = []

        # Check if we found any boxes for this chunk of frames
        if len(all_boxes):

            # We'll make a "subchunk" for each crop size
            for crop_size in all_boxes:

                # TODO: Look into this edge case?
                # if crop_size[0] >= 1024:
                #     logger.info(f"  Skipping subchunk for size {crop_size}, would have {len(all_boxes[crop_size])} crops.")
                #     for debug_frame_idx in all_boxes[crop_size].keys():
                #         print(f"    frame {frames_idx[debug_frame_idx]}: {all_boxes[crop_size][debug_frame_idx]}")
                #     continue

                # Make list of all boxes and corresponding img index.
                subchunk_idxs = []
                subchunk_boxes = []

                for frame_i, frame_boxes in all_boxes[crop_size].items():
                    subchunk_boxes.extend(frame_boxes)
                    subchunk_idxs.extend([frame_i] * len(frame_boxes))

                # TODO: This should probably be in the main loop
                # Create transform object
                # transform = DataTransform(frame_idxs=frames_idx, scale=cm_model.output_relative_scale)
                transform = DataTransform(frame_idxs=frames_idx)

                # Do the cropping
                imgs_cropped = transform.crop(imgs, subchunk_boxes, subchunk_idxs)

                # Add subchunk
                subchunks.append((imgs_cropped, transform))

                logger.info(f"  Subchunk for size {crop_size} has {len(imgs_cropped)} crops.")

        else:
            logger.info("  No centroids found so done with this chunk.")

        return subchunks


    def single_instance_inference(self, imgs, transform, video) -> List[LabeledFrame]:
        """Run the single instance pipeline for a stack of images.

        Args:
            imgs: Subchunk of images to process.
            transform: DataTransform object tracking input transformations.
            video: Video object for building LabeledFrames with correct reference to source.

        Returns:
            A list of LabeledFrames with predicted points.
        """

        # Get confmap inference model.
        cm_model = self.inference_models[ModelOutputType.CONFIDENCE_MAP]

        # Scale to match input size of trained model.
        # Images are expected to be at full resolution, but may be cropped.
        assert(transform.scale == 1.0)
        target_shape = (int(imgs.shape[1] * cm_model.input_scale), int(imgs.shape[2] * cm_model.input_scale))
        imgs_scaled = transform.scale_to(imgs=imgs, target_size=target_shape)

        # TODO: Adjust for divisibility
        # divisor = 2 ** cm_model.down_blocks
        # crop_within = ((imgs.shape[1] // divisor) * divisor, (imgs.shape[2] // divisor) * divisor)

        # Run inference.
        t0 = time()
        confmaps = cm_model.predict(imgs_scaled, batch_size=self.inference_batch_size)
        logger.info( "  Inferred confmaps [%.1fs]" % (time() - t0))
        logger.info(f"    confmaps: shape={confmaps.shape}, ptp={np.ptp(confmaps)}")

        t0 = time()

        # TODO: Move this to GPU and add subpixel refinement.
        # Use single highest peak in channel corresponding node
        points_arrays = find_all_single_peaks(confmaps,
                                min_thresh=self.nms_min_thresh)

        # Adjust for multi-scale such that the points are at the scale of the transform.
        points_arrays = [pts / cm_model.output_relative_scale for pts in points_arrays]

        # Create labeled frames and predicted instances from the points.
        predicted_frames_chunk = match_single_peaks_all(
                                        points_arrays=points_arrays,
                                        skeleton=cm_model.skeleton,
                                        transform=transform,
                                        video=video)

        logger.info("  Used highest peaks to create instances [%.1fs]" % (time() - t0))

        # Save confmaps
        if self.output_path is not None and self.save_confmaps_pafs:
            raise NotImplementedError("Not saving confmaps/pafs because feature currently not working.")
            # Disable save_confmaps_pafs since not currently working.
            # The problem is that we can't put data for different crop sizes
            # all into a single h5 datasource. It's now possible to view live
            # predicted confmap and paf in the gui, so this isn't high priority.
            # save_visual_outputs(
            #         output_path = self.output_path,
            #         data = dict(confmaps=confmaps, box=imgs))

        return predicted_frames_chunk


    def multi_instance_inference(self, imgs, transform, video) -> List[LabeledFrame]:
        """Run the multi-instance inference pipeline for a stack of images.

        Args:
            imgs: Subchunk of images to process.
            transform: DataTransform object tracking input transformations.
            video: Video object for building LabeledFrames with correct reference to source.

        Returns:
            A list of LabeledFrames with predicted points.
        """

        # Load appropriate models as needed
        cm_model = self.inference_models[ModelOutputType.CONFIDENCE_MAP]
        paf_model = self.inference_models[ModelOutputType.PART_AFFINITY_FIELD]

        # Find peaks
        t0 = time()

        # Scale to match input resolution of model.
        # Images are expected to be at full resolution, but may be cropped.
        assert(transform.scale == 1.0)
        cm_target_shape = (int(imgs.shape[1] * cm_model.input_scale), int(imgs.shape[2] * cm_model.input_scale))
        imgs_scaled = transform.scale_to(imgs=imgs, target_size=cm_target_shape)
        if imgs_scaled.dtype == np.dtype("uint8"):  # TODO: Unify normalization.
            imgs_scaled = imgs_scaled.astype("float32") / 255.
        
        # TODO: Unfuck this whole workflow
        if self.gpu_peak_finding:
            confmaps_shape = cm_model.compute_output_shape((imgs_scaled.shape[1], imgs_scaled.shape[2]))
            peaks, peak_vals, confmaps = peak_tf_inference(
                model=cm_model.keras_model,
                confmaps_shape=confmaps_shape,
                data=imgs_scaled,
                min_thresh=self.nms_min_thresh,
                gaussian_size=self.nms_kernel_size,
                gaussian_sigma=self.nms_sigma,
                upsample_factor=int(self.supersample_factor / cm_model.output_scale),
                win_size=self.supersample_window_size,
                return_confmaps=self.save_confmaps_pafs,
                batch_size=self.inference_batch_size
                )

        else:
            confmaps = cm_model.predict(imgs_scaled, batch_size=self.inference_batch_size)
            peaks, peak_vals = find_all_peaks(confmaps, min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)

        # # Undo just the scaling so we're back to full resolution, but possibly cropped.
        for t in range(len(peaks)):  # frames
            for c in range(len(peaks[t])):  # channels
                peaks[t][c] /= cm_model.output_scale

        # Peaks should be at (refined) full resolution now.
        # Keep track of scale adjustment.
        transform.scale = 1.0

        elapsed = time() - t0
        total_peaks = sum([len(channel_peaks) for frame_peaks in peaks for channel_peaks in frame_peaks])
        logger.info(f"    Found {total_peaks} peaks ({total_peaks / len(imgs):.2f} peaks/frame) [{elapsed:.2f}s].")
        # logger.info(f"    peaks: {peaks}")

        # Scale to match input resolution of model.
        # Images are expected to be at full resolution, but may be cropped.
        paf_target_shape = (int(imgs.shape[1] * paf_model.input_scale), int(imgs.shape[2] * paf_model.input_scale))
        if (imgs_scaled.shape[1] == paf_target_shape[0]) and (imgs_scaled.shape[2] == paf_target_shape[1]):
            # No need to scale again if we're already there, so just adjust the stored scale
            transform.scale = paf_model.input_scale

        else:
            # Adjust scale from full resolution images (avoiding possible resizing up from confmaps input scale)
            imgs_scaled = transform.scale_to(imgs=imgs, target_size=paf_target_shape)

        # Infer pafs
        t0 = time()
        pafs = paf_model.predict(imgs_scaled, batch_size=self.inference_batch_size)
        logger.info( "  Inferred PAFs [%.1fs]" % (time() - t0))
        logger.info(f"    pafs: shape={pafs.shape}, ptp={np.ptp(pafs)}")

        # Adjust points to the paf output scale so we can invert later (should not incur loss of precision)
        # TODO: Check precision
        for t in range(len(peaks)):  # frames
            for c in range(len(peaks[t])):  # channels
                peaks[t][c] *= paf_model.output_scale
        transform.scale = paf_model.output_scale

        # Determine whether to use serial or parallel version of peak-finding
        # Use the serial version is we're already running in a thread pool
        match_peaks_function = match_peaks_paf_par if not self.is_async else match_peaks_paf

        # Match peaks via PAFs
        t0 = time()
        predicted_frames_chunk = match_peaks_function(
            peaks, peak_vals, pafs, paf_model.skeleton,
            transform=transform, video=video,
            min_score_to_node_ratio=self.min_score_to_node_ratio,
            min_score_midpts=self.min_score_midpts,
            min_score_integral=self.min_score_integral,
            add_last_edge=self.add_last_edge,
            single_per_crop=self.single_per_crop,
            pool=self.pool)

        total_instances = sum([len(labeled_frame) for labeled_frame in predicted_frames_chunk])
        logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))
        logger.info(f"    Found {total_instances} instances ({total_instances / len(imgs):.2f} instances/frame)")

        # Remove overlapping predicted instances
        if self.overlapping_instances_nms:
            t0 = clock()
            for lf in predicted_frames_chunk:
                n = len(lf.instances)
                instances_nms(lf.instances)
                if len(lf.instances) < n:
                    logger.info(f"    Removed {n-len(lf.instances)} overlapping instance(s) from frame {lf.frame_idx}")
            logger.info("    Instance NMS [%.1fs]" % (clock() - t0))

        # Save confmaps and pafs
        if self.output_path is not None and self.save_confmaps_pafs:
            raise NotImplementedError("Not saving confmaps/pafs because feature currently not working.")
            # Disable save_confmaps_pafs since not currently working.
            # The problem is that we can't put data for different crop sizes
            # all into a single h5 datasource. It's now possible to view live
            # predicted confmap and paf in the gui, so this isn't high priority.
            # save_visual_outputs(
            #         output_path = self.output_path,
            #         data = dict(confmaps=confmaps, pafs=pafs,
            #             frame_idxs=transform.frame_idxs, bounds=transform.bounding_boxes))

        return predicted_frames_chunk


def main():

    def frame_list(frame_str: str):

        # Handle ranges of frames. Must be of the form "1-200"
        if "-" in frame_str:
            min_max = frame_str.split("-")
            min_frame = int(min_max[0])
            max_frame = int(min_max[1])
            return list(range(min_frame, max_frame+1))

        return [int(x) for x in frame_str.split(",")] if len(frame_str) else None

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to video file")
    parser.add_argument("-m", "--model", dest='models', action='append',
                        help="Path to saved model (confmaps, pafs, ...) JSON. "
                        "Multiple models can be specified, each preceded by "
                        "--model. Confmap and PAF models are required.",
                        required=True)
    parser.add_argument("--resize-input", dest="resize_input", action="store_const",
                    const=True, default=False,
                    help="resize the input layer to image size (default False)")
    parser.add_argument("--with-tracking", dest="with_tracking", action="store_const",
                    const=True, default=False,
                    help="just visualize predicted confmaps/pafs (default False)")
    parser.add_argument("--frames", type=frame_list, default="",
                        help="list of frames to predict. Either comma separated list (e.g. 1,2,3) or "
                             "a range separated by hyphen (e.g. 1-3). (default is entire video)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="The output filename to use for the predicted data.")
    parser.add_argument("--out_format", choices=["hdf5", "json"], help="The format to use for"
                    " the output file. Either hdf5 or json. hdf5 is the default.",
                    default="hdf5")
    parser.add_argument("--save-confmaps-pafs", dest="save_confmaps_pafs", action="store_const",
                    const=True, default=False,
                        help="Whether to save the confidence maps or pafs")
    parser.add_argument("-v", "--verbose", help="Increase logging output verbosity.", action="store_true")

    args = parser.parse_args()

    if args.out_format == "json":
        output_suffix = ".predictions.json"
    else:
        output_suffix = ".predictions.h5"

    if args.frames is not None:
        output_suffix = f".frames{min(args.frames)}_{max(args.frames)}" + output_suffix

    data_path = args.data_path
    save_path = args.output if args.output else data_path + output_suffix
    frames = args.frames

    if args.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Load each model JSON
    jobs = [TrainingJob.load_json(model_filename) for model_filename in args.models]
    sleap_models = dict(zip([j.model.output_type for j in jobs], jobs))

    if ModelOutputType.CONFIDENCE_MAP not in sleap_models:
        raise ValueError("No confidence map model found in specified models!")

    if args.resize_input:
        # Load video
        vid = Video.from_filename(data_path)
        img_shape = (vid.height, vid.width, vid.channels)
    else:
        img_shape = None

    # Create a predictor to do the work.
    predictor = Predictor(training_jobs=sleap_models,
        output_path=save_path,
        save_confmaps_pafs=args.save_confmaps_pafs,
        with_tracking=args.with_tracking)

    # Run the inference pipeline
    return predictor.predict(input_video=data_path, frames=frames)


if __name__ == "__main__":
   main()

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
from typing import Dict, List, Union, Optional, Tuple

from keras.utils import multi_gpu_model

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
from sleap.nn.loadmodel import load_model, get_model_data, get_model_skeleton
from sleap.nn.peakfinding import find_all_peaks, find_all_single_peaks
from sleap.nn.peakfinding_tf import peak_tf_inference
from sleap.nn.peakmatching import match_single_peaks_all, match_peaks_paf, match_peaks_paf_par, instances_nms
from sleap.nn.util import batch, batch_count, save_visual_outputs

OVERLAPPING_INSTANCES_NMS = True

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

    sleap_models: Dict[ModelOutputType, TrainingJob] = None
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

    _models: Dict = attr.ib(default=attr.Factory(dict))

    def predict(self,
                input_video: Union[dict, Video],
                frames: Optional[List[int]] = None,
                is_async: bool = False) -> List[LabeledFrame]:
        """
        Run the entire inference pipeline on an input video.

        Args:
            input_video: Either a `Video` object or dict that can be
                converted back to a `Video` object.
            frames (optional): List of frames to predict.
                If None, run entire video.
            is_async (optional): Whether running function from separate
                process. Default is False. If True, we won't spawn
                children.

        Returns:
            list of LabeledFrame objects
        """

        self.is_async = is_async

        # Initialize parallel pool
        self.pool = None if self.is_async else multiprocessing.Pool(processes=usable_cpu_count())

        # Fix the number of threads for OpenCV, not that we are using
        # anything in OpenCV that is actually multi-threaded but maybe
        # we will down the line.
        cv2.setNumThreads(usable_cpu_count())

        logger.info(f"Predict is async: {is_async}")

        # Find out how many channels the model was trained on

        model_channels = 3 # default

        if ModelOutputType.CENTROIDS in self.sleap_models:
            centroid_model = self.fetch_model(
                                input_size = None,
                                output_types = [ModelOutputType.CENTROIDS])
            model_channels = centroid_model["model"].input_shape[-1]

        grayscale = (model_channels == 1)

        # Open the video if we need it.

        try:
            input_video.get_frame(0)
            vid = input_video
        except AttributeError:
            if isinstance(input_video, dict):
                vid = Video.cattr().structure(input_video, Video)
            elif isinstance(input_video, str):
                vid = Video.from_filename(input_video, grayscale=grayscale)
            else:
                raise AttributeError(f"Unable to load input video: {input_video}")

        # List of frames to process (or entire video if not specified)
        frames = frames or list(range(vid.num_frames))

        vid_h = vid.shape[1]
        vid_w = vid.shape[2]

        logger.info("Opened video:")
        logger.info("  Source: " + str(vid.backend))
        logger.info("  Frames: %d" % len(frames))
        logger.info("  Frame shape: %d x %d" % (vid_h, vid_w))

        # Check training models
        if len(self.sleap_models) == 0:
            logger.warning("Predictor has no model.")
            raise ValueError("Predictor has no model.")

        # Initialize tracking
        tracker = FlowShiftTracker(window=self.flow_window, verbosity=0)

        # Create output directory if it doesn't exist
        try:
            os.mkdir(os.path.dirname(self.output_path))
        except FileExistsError:
            pass
        # Delete the output file if it exists already
        if os.path.exists(self.output_path):
            os.unlink(self.output_path)

        # Process chunk-by-chunk!
        t0_start = time()
        predicted_frames: List[LabeledFrame] = []

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
            mov_full = vid[frames_idx]
            logger.info("  Read %d frames [%.1fs]" % (len(mov_full), time() - t0))

            # Transform images (crop or scale)
            t0 = time()

            if ModelOutputType.CENTROIDS in self.sleap_models:
                # Use centroid predictions to get subchunks of crops

                subchunks_to_process = self.centroid_crop_inference(
                                                mov_full, frames_idx)

            else:
                # Scale without centroid cropping

                # Get the scale that was used when training models
                model_data = get_model_data(self.sleap_models, [ModelOutputType.CONFIDENCE_MAP])
                scale = model_data["scale"]

                # Determine scaled image size
                scale_to = (int(vid.height//(1/scale)), int(vid.width//(1/scale)))

                # FIXME: Adjust to appropriate power of 2
                # It would be better to pad image to a usable size, since
                # the resize could affect aspect ratio.
                if self.resize_hack:
                    scale_to = (scale_to[0]//8*8, scale_to[1]//8*8)

                # Create transform object
                transform = DataTransform(
                                frame_idxs = frames_idx,
                                scale = model_data["multiscale"])

                # Scale if target doesn't match current size
                mov = transform.scale_to(mov_full, target_size=scale_to)

                subchunks_to_process = [(mov, transform)]

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

            for subchunk_mov, subchunk_transform in subchunks_to_process:

                logger.info(f"  Running inference for subchunk:")
                logger.info(f"    Shape: {subchunk_mov.shape}")
                logger.info(f"    Prediction Scale: {subchunk_transform.scale}")

                if ModelOutputType.PART_AFFINITY_FIELD not in self.sleap_models:
                    # Pipeline for predicting a single animal in a frame
                    # This uses only confidence maps

                    logger.warning("No PAF model! Running in SINGLE INSTANCE mode.")

                    subchunk_lfs = self.single_instance_inference(
                                            subchunk_mov,
                                            subchunk_transform,
                                            vid)

                else:
                    # Pipeline for predicting multiple animals in a frame
                    # This uses confidence maps and part affinity fields
                    subchunk_lfs = self.multi_instance_inference(
                                            subchunk_mov,
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

            logger.info(f"  Instances found on {len(predicted_frames_chunk)} out of {len(mov_full)} frames.")

            if len(predicted_frames_chunk):

                # Sort by frame index
                predicted_frames_chunk.sort(key=lambda lf: lf.frame_idx)

                # Track
                if self.with_tracking and len(predicted_frames_chunk):
                    t0 = time()
                    tracker.process(mov_full, predicted_frames_chunk)
                    logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

                # Save
                predicted_frames.extend(predicted_frames_chunk)

                if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                    t0 = time()

                    # FIXME: We are re-writing the whole output each time, this is dumb.
                    #  We should save in chunks then combine at the end.
                    labels = Labels(labeled_frames=predicted_frames)
                    if self.output_path is not None:
                        if self.output_path.endswith('json'):
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

        pool = Pool(processes=1)
        result = pool.apply_async(self.predict, args=args, kwds=kwargs)

        # Tell the pool to accept no new tasks
        pool.close()

        return pool, result

    # Methods for running inferring on components of pipeline

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

        crop_within = (imgs.shape[1]//8*8, imgs.shape[2]//8*8)

        # Fetch centroid model (uses cache if already loaded)

        model_package = self.fetch_model(
                                input_size = None,
                                output_types = [ModelOutputType.CENTROIDS])

        # Create transform

        # This lets us scale the images before we predict centroids,
        # and will also let us map the points on the scaled image to
        # points on the original images so we can crop original images.

        centroid_transform = DataTransform()

        # Scale to match input size of trained centroid model
        # Usually this will be 1/4-scale of original images

        centroid_imgs_scaled = \
            centroid_transform.scale_to(
                    imgs=imgs,
                    target_size=model_package["model"].input_shape[1:3])

        # Predict centroid confidence maps, then find peaks

        centroid_confmaps = model_package["model"].predict(centroid_imgs_scaled.astype("float32") / 255,
                                                batch_size=self.inference_batch_size)

        peaks, peak_vals = find_all_peaks(centroid_confmaps,
                                            min_thresh=self.nms_min_thresh,
                                            sigma=self.nms_sigma)


        if box_size is None:
            # Get training bounding box size to determine (min) centroid crop size
            crop_model_package = self.fetch_model(
                                    input_size = None,
                                    output_types = [ModelOutputType.CONFIDENCE_MAP])
            crop_size = crop_model_package["bounding_box_size"]
            bb_half = (crop_size + self.crop_padding)//2
        else:
            bb_half = box_size//2

        logger.info(f"  Centroid crop box size: {bb_half*2}")

        all_boxes = dict()

        # Iterate over each frame to filter bounding boxes
        for frame_i, (frame_peaks, frame_peak_vals) in enumerate(zip(peaks, peak_vals)):

            # If we found centroids on this frame...
            if frame_peaks[0].shape[0] > 0:

                # Pad each centroid into a bounding box
                # (We're not using the pad function because it shifts
                # boxes to fit within image.)

                boxes = []
                for peak_i in range(frame_peaks[0].shape[0]):
                    # Rescale peak back onto full-sized image
                    peak_x = int(frame_peaks[0][peak_i][0] / centroid_transform.scale)
                    peak_y = int(frame_peaks[0][peak_i][1] / centroid_transform.scale)

                    boxes.append((peak_x-bb_half, peak_y-bb_half,
                                  peak_x+bb_half, peak_y+bb_half))

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

                    box_size = (box[2]-box[0], box[3]-box[1])

                    if box_size not in all_boxes:
                        all_boxes[box_size] = dict()
                    if frame_i not in all_boxes[box_size]:
                        all_boxes[box_size][frame_i] = []

                    all_boxes[box_size][frame_i].append(box)

        subchunks = []

        # Check if we found any boxes for this chunk of frames
        if len(all_boxes):
            model_data = get_model_data(self.sleap_models, [ModelOutputType.CONFIDENCE_MAP])

            # We'll make a "subchunk" for each crop size
            for crop_size in all_boxes:

                if crop_size[0] >= 1024:
                    logger.info(f"  Skipping subchunk for size {crop_size}, would have {len(all_boxes[crop_size])} crops.")
                    for debug_frame_idx in all_boxes[crop_size].keys():
                        print(f"    frame {frames_idx[debug_frame_idx]}: {all_boxes[crop_size][debug_frame_idx]}")
                    continue

                # Make list of all boxes and corresponding img index.
                subchunk_idxs = []
                subchunk_boxes = []

                for frame_i, frame_boxes in all_boxes[crop_size].items():
                    subchunk_boxes.extend(frame_boxes)
                    subchunk_idxs.extend( [frame_i] * len(frame_boxes) )

                # Create transform object
                transform = DataTransform(
                                frame_idxs = frames_idx,
                                scale = model_data["multiscale"])

                # Do the cropping
                imgs_cropped = transform.crop(imgs, subchunk_boxes, subchunk_idxs)

                # Add subchunk
                subchunks.append((imgs_cropped, transform))

                logger.info(f"  Subchunk for size {crop_size} has {len(imgs_cropped)} crops.")

        else:
            logger.info("  No centroids found so done with this chunk.")

        return subchunks

    def single_instance_inference(self, imgs, transform, video) -> List[LabeledFrame]:
        """Run the single instance pipeline for a stack of images."""

        # Get confmap model for this image size
        model_package = self.fetch_model(
                            input_size = imgs.shape[1:],
                            output_types = [ModelOutputType.CONFIDENCE_MAP])

        # Run inference
        t0 = time()

        confmaps = model_package["model"].predict(imgs.astype("float32") / 255, batch_size=self.inference_batch_size)
        logger.info( "  Inferred confmaps [%.1fs]" % (time() - t0))
        logger.info(f"    confmaps: shape={confmaps.shape}, ptp={np.ptp(confmaps)}")

        t0 = time()

        # Use single highest peak in channel corresponding node
        points_arrays = find_all_single_peaks(confmaps,
                                min_thresh=self.nms_min_thresh)

        predicted_frames_chunk = match_single_peaks_all(
                                        points_arrays = points_arrays,
                                        skeleton = model_package["skeleton"],
                                        transform = transform,
                                        video = video)

        logger.info("  Used highest peaks to create instances [%.1fs]" % (time() - t0))

        # Save confmaps
        if self.output_path is not None and self.save_confmaps_pafs:
            logger.warning("Not saving confmaps because feature currently not working.")
            # Disable save_confmaps_pafs since not currently working.
            # The problem is that we can't put data for different crop sizes
            # all into a single h5 datasource. It's now possible to view live
            # predicted confmap and paf in the gui, so this isn't high priority.
            # save_visual_outputs(
            #         output_path = self.output_path,
            #         data = dict(confmaps=confmaps, box=imgs))

        return predicted_frames_chunk

    def multi_instance_inference(self, imgs, transform, video) -> List[LabeledFrame]:
        """
        Run the multi-instance inference pipeline for a stack of images.
        """

        # Load appropriate models as needed
        conf_model = self.fetch_model(
                            input_size = imgs.shape[1:],
                            output_types = [ModelOutputType.CONFIDENCE_MAP])

        paf_model = self.fetch_model(
                            input_size = imgs.shape[1:],
                            output_types = [ModelOutputType.PART_AFFINITY_FIELD])

        # Find peaks
        t0 = time()

        multiscale_diff = paf_model["multiscale"] / conf_model["multiscale"]

        peaks, peak_vals, confmaps = \
                peak_tf_inference(
                    model = conf_model["model"],
                    data = imgs.astype("float32")/255,
                    min_thresh=self.nms_min_thresh,
                    gaussian_size=self.nms_kernel_size,
                    gaussian_sigma=self.nms_sigma,
                    downsample_factor=int(1/multiscale_diff),
                    upsample_factor=int(1/conf_model["multiscale"]),
                    return_confmaps=self.save_confmaps_pafs
                    )

        transform.scale = transform.scale * multiscale_diff

        logger.info("  Inferred confmaps and found-peaks (gpu) [%.1fs]" % (time() - t0))
        logger.info(f"    peaks: {len(peaks)}")

        # Infer pafs
        t0 = time()
        pafs = paf_model["model"].predict(imgs.astype("float32") / 255, batch_size=self.inference_batch_size)

        logger.info( "  Inferred PAFs [%.1fs]" % (time() - t0))
        logger.info(f"    pafs: shape={pafs.shape}, ptp={np.ptp(pafs)}")

        # Determine whether to use serial or parallel version of peak-finding
        # Use the serial version is we're already running in a thread pool
        match_peaks_function = match_peaks_paf_par if not self.is_async else match_peaks_paf

        # Match peaks via PAFs
        t0 = time()

        predicted_frames_chunk = match_peaks_function(
                                        peaks, peak_vals, pafs, conf_model["skeleton"],
                                        transform=transform, video=video,
                                        min_score_to_node_ratio=self.min_score_to_node_ratio,
                                        min_score_midpts=self.min_score_midpts,
                                        min_score_integral=self.min_score_integral,
                                        add_last_edge=self.add_last_edge,
                                        single_per_crop=self.single_per_crop,
                                        pool=self.pool)

        logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

        # Remove overlapping predicted instances
        if OVERLAPPING_INSTANCES_NMS:
            t0 = clock()
            for lf in predicted_frames_chunk:
                n = len(lf.instances)
                instances_nms(lf.instances)
                if len(lf.instances) < n:
                    logger.info(f"    Removed {n-len(lf.instances)} overlapping instance(s) from frame {lf.frame_idx}")
            logger.info("    Instance NMS [%.1fs]" % (clock() - t0))

        # Save confmaps and pafs
        if self.output_path is not None and self.save_confmaps_pafs:
            logger.warning("Not saving confmaps/pafs because feature currently not working.")
            # Disable save_confmaps_pafs since not currently working.
            # The problem is that we can't put data for different crop sizes
            # all into a single h5 datasource. It's now possible to view live
            # predicted confmap and paf in the gui, so this isn't high priority.
            # save_visual_outputs(
            #         output_path = self.output_path,
            #         data = dict(confmaps=confmaps, pafs=pafs,
            #             frame_idxs=transform.frame_idxs, bounds=transform.bounding_boxes))

        return predicted_frames_chunk

    def fetch_model(self,
            input_size: tuple,
            output_types: List[ModelOutputType]) -> dict:
        """Loads and returns keras Model with caching."""

        key = (input_size, tuple(output_types))

        if key not in self._models:

            # Load model

            keras_model = load_model(self.sleap_models, input_size, output_types)
            first_sleap_model = self.sleap_models[output_types[0]]
            model_data = get_model_data(self.sleap_models, output_types)
            skeleton = get_model_skeleton(self.sleap_models, output_types)

            # logger.info(f"Model multiscale: {model_data['multiscale']}")

            # If no input size was specified, then use the input size
            # from original trained model.

            if input_size is None:
                input_size = keras_model.input_shape[1:]

            # Get the size of the bounding box from training data
            # (or the size of crop that model was trained on if the
            # bounding box size wasn't set).

            if first_sleap_model.trainer.instance_crop:
                bounding_box_size = \
                    first_sleap_model.trainer.bounding_box_size or keras_model.input_shape[1]
            else:
                bounding_box_size = None

            # Cache the model so we don't have to load it next time

            self._models[key] = dict(
                                    model=keras_model,
                                    skeleton=model_data["skeleton"],
                                    multiscale=model_data["multiscale"],
                                    bounding_box_size=bounding_box_size
                                    )

        # Return the keras Model
        return self._models[key]


def main():

    def frame_list(frame_str: str):

        # Handle ranges of frames. Must be of the form "1-200"
        if '-' in frame_str:
            min_max = frame_str.split('-')
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
    parser.add_argument('--resize-input', dest='resize_input', action='store_const',
                    const=True, default=False,
                    help='resize the input layer to image size (default False)')
    parser.add_argument('--with-tracking', dest='with_tracking', action='store_const',
                    const=True, default=False,
                    help='just visualize predicted confmaps/pafs (default False)')
    parser.add_argument('--frames', type=frame_list, default="",
                        help='list of frames to predict. Either comma separated list (e.g. 1,2,3) or '
                             'a range separated by hyphen (e.g. 1-3). (default is entire video)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='The output filename to use for the predicted data.')
    parser.add_argument('--out_format', choices=['hdf5', 'json'], help='The format to use for'
                    ' the output file. Either hdf5 or json. hdf5 is the default.',
                    default='hdf5')
    parser.add_argument('--save-confmaps-pafs', dest='save_confmaps_pafs', action='store_const',
                    const=True, default=False,
                        help='Whether to save the confidence maps or pafs')
    parser.add_argument('-v', '--verbose', help='Increase logging output verbosity.', action="store_true")

    args = parser.parse_args()

    if args.out_format == 'json':
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
    predictor = Predictor(sleap_models=sleap_models,
                    output_path=save_path,
                    save_confmaps_pafs=args.save_confmaps_pafs,
                    with_tracking=args.with_tracking)

    # Run the inference pipeline
    return predictor.predict(input_video=data_path, frames=frames)


if __name__ == "__main__":
   main()

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

from sleap.nn.datagen import bounding_box_nms
from sleap.nn.loadmodel import load_model, get_model_data, get_model_skeleton
from sleap.nn.peakfinding import find_all_peaks, find_all_single_peaks
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
        nms_sigma: Gaussian blur is applied to confidence maps before
            non-max supression peak finding occurs. This is the
            standard deviation of the kernel applied to the image.
        min_score_to_node_ratio: FIXME
        min_score_midpts: FIXME
        min_score_integral: FIXME
        add_last_edge: FIXME
        with_tracking: whether to run tracking after inference
        flow_window: The number of frames that tracking should look back
            when trying to identify instances.
        crop_iou_threshold: FIXME
        single_per_crop: FIXME
        output_path: the output path to save the results
        save_confmaps_pafs: whether to save confmaps/pafs
        resize_hack: whether to resize images to power of 2
    """

    sleap_models: Dict[ModelOutputType, TrainingJob] = None
    skeleton: Skeleton = None
    inference_batch_size: int = 2
    read_chunk_size: int = 256
    save_frequency: int = 1 # chunks
    nms_min_thresh = 0.3
    nms_sigma = 3
    min_score_to_node_ratio: float = 0.2
    min_score_midpts: float = 0.05
    min_score_integral: float = 0.6
    add_last_edge: bool = True
    with_tracking: bool = False
    flow_window: int = 15
    crop_iou_threshold: float = .9
    single_per_crop: bool = True

    output_path: Optional[str] = None
    save_confmaps_pafs: bool = False
    resize_hack: bool = True

    _models: Dict = attr.ib(default=attr.Factory(dict))

    def predict_centroids(self, imgs: np.ndarray, crop_size: int=None,
                iou_threshold: float=.9,
                return_confmaps=False) -> List[List[np.ndarray]]:

        keras_model = self._get_centroid_model()

        centroid_transform = DataTransform()
        centroid_imgs_scaled = centroid_transform.scale_to(
                                    imgs=imgs,
                                    target_size=keras_model.input_shape[1:3])

        # Predict centroids
        centroid_confmaps = keras_model.predict(centroid_imgs_scaled.astype("float32") / 255,
                                                batch_size=self.inference_batch_size)

        peaks, peak_vals = find_all_peaks(centroid_confmaps,
                                            min_thresh=self.nms_min_thresh,
                                            sigma=self.nms_sigma)

        if crop_size is not None:
            bb_half = crop_size//2
            peak_idxs = []

            for frame_peaks, frame_peak_vals in zip(peaks, peak_vals):
                if frame_peaks[0].shape[0] > 0:
                    boxes = np.stack([(frame_peaks[0][i][0]-bb_half,
                             frame_peaks[0][i][1]-bb_half,
                             frame_peaks[0][i][0]+bb_half,
                             frame_peaks[0][i][1]+bb_half)
                            for i in range(frame_peaks[0].shape[0])])
                    # filter boxes
                    box_select_idxs = bounding_box_nms(
                                            boxes,
                                            scores = frame_peak_vals[0],
                                            iou_threshold = iou_threshold,
                                            )
                    if len(box_select_idxs) < boxes.shape[0]:
                        logger.debug(f"    suppressed centroid crops from {boxes.shape[0]} to {len(box_select_idxs)}")
                    # get a list of peak indexes that we want to use for this frame
                    peak_idxs.append(box_select_idxs)
                else:
                    peak_idxs.append([])

        else:
            peak_idxs = [list(range(frame_peaks[0].shape[0])) for frame_peaks in peaks]

        centroids = [[np.expand_dims(frame_peaks[0][peak_idx], axis=0) / centroid_transform.scale
                        for peak_idx in frame_peak_idxs]
                     for frame_peaks, frame_peak_idxs in zip(peaks, peak_idxs)]

        # Use predicted centroids (peaks) to crop images

        if return_confmaps:
            return centroids, centroid_confmaps
        else:
            return centroids

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

        # Open the video if we need it.

        try:
            input_video.get_frame(0)
            vid = input_video
        except AttributeError:
            if isinstance(input_video, dict):
                vid = Video.cattr().structure(input_video, Video)
            elif isinstance(input_video, str):
                vid = Video.from_filename(input_video)
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

            subchunks_to_process = []

            if ModelOutputType.CENTROIDS in self.sleap_models:

                # For centroids we'll need crop-sized models so we may as well
                # load them here since we need to access data about the models.
                model_package = self.fetch_model(
                            input_size = None,
                            output_types = [ModelOutputType.CONFIDENCE_MAP,
                                ModelOutputType.PART_AFFINITY_FIELD])

                # Get training crop size
                crop_size = model_package["model"].input_shape[1]

                # Find centroids
                centroids = self.predict_centroids(mov_full, crop_size, self.crop_iou_threshold)

                # Check if we found any centroids
                if sum(map(len, centroids)):
                    # Create transform object
                    transform = DataTransform(
                                    frame_idxs = frames_idx,
                                    scale = model_data["multiscale"])

                    # Do the cropping
                    mov = transform.centroid_crop(mov_full, centroids, crop_size)

                else:
                    logger.info("  No centroids found so done with this chunk.")

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

                subchunks_to_process.append((mov, transform))

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
                logger.info(f"    Scale: {subchunk_transform.scale}")

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

                logger.info(f"    Subchunk frames with instances found: {len(subchunk_lf)}")

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

            logger.info(f"  Instances found on {len(predicted_frames_chunk)} out of {len(mov_ful)} frames.")

            if len(predicted_frames_chunk):

                # Sort by frame index
                predicted_frames_chunk.sort(key=lambda lf: lf.frame.idx)

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
                    if output_path is not None:
                        Labels.save_json(labels, filename=output_path, compress=True)

                        logger.info("  Saved to: %s [%.1fs]" % (output_path, time() - t0))

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
            save_visual_outputs(
                    output_path = self.output_path,
                    data = dict(confmaps=confmaps, box=imgs))

        return predicted_frames_chunk

    def multi_instance_inference(self, imgs, transform, video) -> List[LabeledFrame]:
        """
        Run the multi-instance inference pipeline for a stack of images.
        """

        # Load appropriate models as needed
        model_package = self.fetch_model(
                            input_size = imgs.shape[1:],
                            output_types = [ModelOutputType.CONFIDENCE_MAP,
                                ModelOutputType.PART_AFFINITY_FIELD])

        t0 = time()

        confmaps, pafs = model_package["model"].predict(imgs.astype("float32") / 255, batch_size=self.inference_batch_size)

        logger.info( "  Inferred confmaps and PAFs [%.1fs]" % (time() - t0))
        logger.info(f"    confmaps: shape={confmaps.shape}, ptp={np.ptp(confmaps)}")
        logger.info(f"    pafs: shape={pafs.shape}, ptp={np.ptp(pafs)}")

        # Find peaks
        t0 = time()
        peaks, peak_vals = find_all_peaks(confmaps, min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)
        logger.info("  Found peaks [%.1fs]" % (time() - t0))

        # Determine whether to use serial or parallel version of peak-finding
        # Use the serial version is we're already running in a thread pool
        match_peaks_function = match_peaks_paf_par if not self.is_async else match_peaks_paf

        # Match peaks via PAFs
        t0 = time()

        predicted_frames_chunk = match_peaks_function(
                                        peaks, peak_vals, pafs, model_package["skeleton"],
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
            save_visual_outputs(
                    output_path = self.output_path,
                    data = dict(confmaps=confmaps, pafs=pafs,
                        frame_idxs=transform.frame_idxs, bounds=transform.bounding_boxes))

    def fetch_model(self,
            input_size: tuple,
            output_types: List[ModelOutputType]) -> keras.Model:
        """Loads and returns keras Model with caching."""

        key = (input_size, tuple(output_types))

        if key not in self._models:

            # Load model

            keras_model = load_model(self.sleap_models, input_size, output_types)
            skeleton = get_model_skeleton(self.sleap_models, output_types)

            # If no input size was specified, then use the input size
            # from original trained model.

            if input_size is None:
                input_size = keras_model.input_shape[1:]

            # Cache the model so we don't have to load it next time

            self._models[key] = dict(
                                    model=keras_model,
                                    skeleton=skeleton
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
    parser.add_argument('--save-confmaps-pafs', dest='save_confmaps_pafs', action='store_const',
                    const=True, default=False,
                        help='Whether to save the confidence maps or pafs')
    parser.add_argument('--less-overlap', dest='less_overlap', action='store_const',
                    const=True, default=False,
                    help='use fewer crops and include all instances from each crop '
                    '(works best if crops are much larger than instance bounding boxes)')
    parser.add_argument('-v', '--verbose', help='Increase logging output verbosity.', action="store_true")

    args = parser.parse_args()

    output_suffix = ".predictions.json"
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

    if args.less_overlap:
        predictor.crop_iou_threshold = .8
        predictor.single_per_crop = False
        logger.info("Using 'less overlap' mode: crop nms iou .8, multiple instances per crop, instance nms.")

    # Run the inference pipeline
    return predictor.predict(input_video=data_path, frames=frames)


if __name__ == "__main__":
   main()

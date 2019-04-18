import sys
import argparse
import multiprocessing
import logging
logger = logging.getLogger(__name__)

import numpy as np
import cv2
import attr
import keras

from typing import Union
from time import time

from scipy.io import loadmat, savemat

from sleap.io.video import Video
from sleap.skeleton import Skeleton
from sleap.nn.inference import find_all_peaks, FlowShiftTracker, get_inference_model
from sleap.nn.paf_inference import match_peaks_paf_par
from sleap.util import usable_cpu_count


@attr.s(auto_attribs=True)
class Predictor:
    """
    The Predictor class takes a trained sLEAP model and runs
    the complete inference pipeline from confidence map/part affinity field
    inference, non-maximum suppression peak finding, paf part matching, to tracking.

    Args:
        model: A trained keras model used for confidence map and paf inference. FIXME: Should this be a keral model or a sLEAP model class
        skeleton: The skeleton to use for prediction. FIXME. This should be stored with the model I think
        inference_batch_size: Frames per inference batch (GPU memory limited)
        read_chunk_size: How many frames to read into CPU memory at a time (CPU memory limited)
        nms_min_thresh: A threshold of non-max suppression peak finding in confidence maps. All
        values below this minimum threshold will be set to zero before peak finding algorithm
        is run.
        nms_sigma: Gaussian blur is applied to confidence maps before non-max supression peak
        finding occurs. This is the standard deviation of the kernel applied to the image.
        min_score_to_node_ratio: FIXME
        min_score_midpts: FIXME
        min_score_integral: FIXME
        add_last_edge: FIXME

    """

    model: keras.Model = attr.ib()
    skeleton: str = attr.ib()
    inference_batch_size: int = 4
    read_chunk_size = 512
    nms_min_thresh = 0.3
    nms_sigma = 3
    min_score_to_node_ratio: float = 0.2
    min_score_midpts: float = 0.05
    min_score_integral: float = 0.6
    add_last_edge: bool = True
    flow_window: int = 15

    def run(self, input_video: Union[str, Video], output_path: str):
        """
        Run the entire inference pipeline on an input video or file object.

        Args:
            input_video: Either a video object or video filename.
            output_path: The output path to save the results.

        Returns:
            None
        """

        # Load skeleton: FIXME: this is using mat files, need to fix.
        skeleton = loadmat(self.skeleton)
        skeleton["nodes"] = skeleton["nodes"][0][0]  # convert to scalar
        skeleton["edges"] = skeleton["edges"] - 1  # convert to 0-based indexing
        logger.info("Skeleton (%d nodes):" % skeleton["nodes"])
        logger.info("  %s" % str(skeleton["edges"]))

        # Load model
        _, h, w, c = self.model.input_shape
        model_channels = c
        logger.info("Loaded models:")
        logger.info("  Input shape: %d x %d x %d" % (h, w, c))

        # Open the video if we need it.
        try:
            input_video.get_frame(0)
            vid = input_video
        except AttributeError:
            vid = Video.from_filename(input_video)

        num_frames = vid.num_frames
        vid_h = vid.shape[1]
        vid_w = vid.shape[2]
        scale = h / vid_h
        logger.info("Opened video:")
        logger.info("  Source: " + str(vid.backend))
        logger.info("  Frames: %d" % num_frames)
        logger.info("  Frame shape: %d x %d" % (vid_h, vid_w))
        logger.info("  Scale: %f" % scale)

        # Initialize tracking
        tracker = FlowShiftTracker(window=self.flow_window)

        # Initialize parallel pool
        pool = multiprocessing.Pool(processes=usable_cpu_count())

        # Fix the number of threads for OpenCV, not that we are using
        # anything in OpenCV that is actually multi-threaded but maybe
        # we will down the line.
        cv2.setNumThreads(usable_cpu_count())

        # Process chunk-by-chunk!
        t0_start = time()
        matched_instances = []
        match_scores = []
        matched_peak_vals = []
        num_chunks = int(np.ceil(num_frames / self.read_chunk_size))
        frame_num = 0
        for chunk in range(num_chunks):
            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()
            # Calculate how many frames to read
            # num_chunk_frames = min(read_chunk_size, num_frames - int(vid.get(cv2.CAP_PROP_POS_FRAMES)))

            # Read the next batch of images
            t0 = time()

            # Read the next chunk of frames
            frame_end = frame_num + self.read_chunk_size
            if frame_end > vid.num_frames:
                frame_end = vid.num_frames

            mov = vid[frame_num:frame_end]

            # Preprocess the frames
            if model_channels == 1:
                mov = mov[:, :, :, 0]

            # Resize the frames to the model input size
            for i in range(mov.shape[0]):
                mov[i, :, :] = cv2.resize(mov[i, :, :], (w, h))

            # Add back singleton dimension
            if model_channels == 1:
                mov = mov[..., None]
            else:
                mov = mov[..., ::-1]

            logger.info("  Read %d frames [%.1fs]" % (len(mov), time() - t0))

            # Run inference
            t0 = time()
            confmaps, pafs = self.model.predict(mov.astype("float32") / 255, batch_size=self.inference_batch_size)
            logger.info("  Inferred confmaps and PAFs [%.1fs]" % (time() - t0))

            # Find peaks
            t0 = time()
            peaks, peak_vals = find_all_peaks(confmaps, min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)
            logger.info("  Found peaks [%.1fs]" % (time() - t0))

            # Match peaks via PAFs
            t0 = time()
            # instances, scores = match_peaks_paf(peaks, peak_vals, pafs, skeleton)
            instances, scores, peak_vals = match_peaks_paf_par(peaks, peak_vals, pafs, skeleton,
                                                               min_score_to_node_ratio=self.min_score_to_node_ratio,
                                                               min_score_midpts=self.min_score_midpts,
                                                               min_score_integral=self.min_score_integral,
                                                               add_last_edge=self.add_last_edge, pool=pool)
            logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

            # # Adjust for input scale
            # for i in range(len(instances)):
            #     for j in range(len(instances[i])):
            #         instances[i][j] = instances[i][j] / scale

            # Track
            t0 = time()
            tracker.track(mov, instances)
            logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

            # Save
            matched_instances.extend(instances)
            match_scores.extend(scores)
            matched_peak_vals.extend(peak_vals)

            save_every = 3

            # Get the parameters used for this inference.
            params = attr.asdict(self, filter=lambda attr, value: attr.name != "model")

            if chunk % save_every == 0 or chunk == (num_chunks - 1):
                t0 = time()
                # FIXME: Saving as MAT file should be replaced with HDF5
                savemat(output_path, dict(params=params, skeleton=skeleton,
                                        matched_instances=matched_instances, match_scores=match_scores,
                                        matched_peak_vals=matched_peak_vals, scale=scale,
                                        uids=tracker.uids, tracked_instances=tracker.generate_tracks(matched_instances),
                                        flow_assignment_costs=tracker.flow_assignment_costs,
                                        ), do_compression=True)
                logger.info("  Saved to: %s [%.1fs]" % (output_path, time() - t0))

            elapsed = time() - t0_chunk
            total_elapsed = time() - t0_start
            fps = len(matched_instances) / total_elapsed
            frames_left = num_frames - len(matched_instances)
            logger.info("  Finished chunk [%.1fs / %.1f FPS / ETA: %.1f min]" % (elapsed, fps, (frames_left / fps) / 60))

            sys.stdout.flush()

        logger.info("Total: %.1f min" % (total_elapsed / 60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to video file")
    parser.add_argument("confmap_model_path", help="Path to saved confmap model")
    parser.add_argument("paf_model_path", help="Path to saved PAF model")
    parser.add_argument("skeleton_path", help="Path to skeleton MAT file")
    args = parser.parse_args()

    data_path = args.data_path
    confmap_model_path = args.confmap_model_path
    paf_model_path = args.paf_model_path
    save_path = data_path + ".paf_tracking.mat"
    skeleton_path = args.skeleton_path

    # Load the model
    model = get_inference_model(confmap_model_path, paf_model_path)

    # Create a predictor to do the work.
    predictor = Predictor(model=model, skeleton=skeleton_path)

    # Run the inference pipeline
    predictor.run(input_video=data_path, output_path=save_path)


if __name__ == "__main__":
    main()

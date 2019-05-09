import sys
import argparse
import multiprocessing
import logging
logger = logging.getLogger(__name__)

import numpy as np
import h5py as h5
import cv2
import keras
import attr

from time import time
from typing import List, Dict, Union

from scipy.ndimage import maximum_filter, gaussian_filter
from keras.utils import multi_gpu_model

from sleap.instance import Point, Instance
from sleap.io.dataset import LabeledFrame, Labels
from sleap.io.video import Video
from sleap.skeleton import Skeleton
from sleap.nn.inference import find_all_peaks, get_inference_model
from sleap.nn.paf_inference import match_peaks_paf_par
from sleap.util import usable_cpu_count, save_dict_to_hdf5
from sleap.nn.tracking import FlowShiftTracker


@attr.s(auto_attribs=True, slots=True)
class PredictedPoint(Point):
    """
    A predicted point is an output of the inference procedure. It has all
    the properties of a labeled point with an accompanying score.

    Args:
        score: The point level prediction score.
    """
    score: float = attr.ib(default=0.0)


@attr.s(auto_attribs=True, slots=True)
class PredictedInstance(Instance):
    """
    A predicted instance is an output of the inference procedure. It is
    the main output of the inference procedure.

    Args:
        score: The instance level prediction score.
    """
    score: float = attr.ib(default=0.0)


def get_inference_model(confmap_model_path: str, paf_model_path: str) -> keras.Model:
    """ Loads and merges confmap and PAF models into one. """

    # Load
    confmap_model = keras.models.load_model(confmap_model_path)
    paf_model = keras.models.load_model(paf_model_path)

    # Single input
    new_input = confmap_model.input

    # Rename to prevent layer naming conflict
    confmap_model.name = "confmap_" + confmap_model.name
    paf_model.name = "paf_" + paf_model.name
    for i in range(len(confmap_model.layers)):
        confmap_model.layers[i].name = "confmap_" + confmap_model.layers[i].name
    for i in range(len(paf_model.layers)):
        paf_model.layers[i].name = "paf_" + paf_model.layers[i].name

    # Get rid of first layer
    confmap_model.layers.pop(0)
    paf_model.layers.pop(0)

    # Combine models with tuple output
    model = keras.Model(new_input, [confmap_model(new_input), paf_model(new_input)])

    try:
        model = multi_gpu_model(model, gpus=4)
    except:
        logging.warning("Multi-GPU inference not available, ")
        pass

    return model


def impeaksnms(I, min_thresh=0.3, sigma=3, return_val=True):
    """ Find peaks via non-maximum suppresion. """

    # Threshold
    if min_thresh is not None:
        I[I < min_thresh] = 0

    # Blur
    if sigma is not None:
        I = gaussian_filter(I, sigma=sigma, mode="constant", cval=0, truncate=8)

    # Maximum filter
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    m = maximum_filter(I, footprint=kernel, mode="constant", cval=0)

    # Convert to points
    r, c = np.nonzero(I > m)
    pts = np.stack((c, r), axis=1)

    # Return
    if return_val:
        vals = np.array([I[pt[1],pt[0]] for pt in pts])
        return pts, vals
    else:
        return pts


def impeaksnms_cv(I, min_thresh=0.3, sigma=3, return_val=True):
    """ Find peaks via non-maximum suppresion using OpenCV. """

    # Threshold
    if min_thresh is not None:
        I[I < min_thresh] = 0

    # Blur
    if sigma is not None:
        I = cv2.GaussianBlur(I, (9,9), sigma)

    # Maximum filter
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]]).astype("uint8")
    m = cv2.dilate(I, kernel)

    # Convert to points
    r, c = np.nonzero(I > m)
    pts = np.stack((c, r), axis=1)

    # Return
    if return_val:
        vals = np.array([I[pt[1],pt[0]] for pt in pts])
        return pts.astype("float32"), vals
    else:
        return pts.astype("float32")


def find_all_peaks(confmaps, min_thresh=0.3, sigma=3):
    """ Finds peaks for all frames/channels in a stack of confidence maps """
    peaks = []
    peak_vals = []
    for confmap in confmaps:
        peaks_i = []
        peak_vals_i = []
        for i in range(confmap.shape[-1]):
            # peak, val = impeaksnms(confmap[...,i], min_thresh=min_thresh, sigma=sigma, return_val=True)
            peak, val = impeaksnms_cv(confmap[...,i], min_thresh=min_thresh, sigma=sigma, return_val=True)
            peaks_i.append(peak)
            peak_vals_i.append(val)
        peaks.append(peaks_i)
        peak_vals.append(peak_vals_i)

    return peaks, peak_vals


def improfile(I, p0, p1, max_points=None):
    """ Returns values of the image I evaluated along the line formed by points p0 and p1.

    Parameters
    ----------
    I : 2d array
        Image to get values from
    p0, p1 : 1d array with 2 elements
        Start and end coordinates of the line

    Returns
    -------
    vals : 1d array
        Vector with the images values along the line formed by p0 and p1
    """
    # Make sure image is 2d
    I = np.squeeze(I)

    # Find number of points to extract
    n = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    n = max(n, 1)
    if max_points is not None:
        n = min(n, max_points)
    n = int(n)

    # Compute coordinates
    x = np.round(np.linspace(p0[0], p1[0], n)).astype("int32")
    y = np.round(np.linspace(p0[1], p1[1], n)).astype("int32")

    # Extract values and concatenate into vector
    vals = np.stack([I[yi,xi] for xi, yi in zip(x,y)])
    return vals



@attr.s(auto_attribs=True)
class Predictor:
    """
    The Predictor class takes a trained sLEAP model and runs
    the complete inference pipeline from confidence map/part affinity field
    inference, non-maximum suppression peak finding, paf part matching, to tracking.

    Args:
        model: A trained keras model used for confidence map and paf inference. FIXME: Should this be a keras model or a sLEAP model class
        skeleton: The skeleton(s) to use for prediction. FIXME. This should be stored with the model I think
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
    skeleton: Skeleton = attr.ib()
    inference_batch_size: int = 4
    read_chunk_size: int = 512
    save_frequency: int = 30 # chunks
    nms_min_thresh = 0.3
    nms_sigma = 3
    min_score_to_node_ratio: float = 0.2
    min_score_midpts: float = 0.05
    min_score_integral: float = 0.6
    add_last_edge: bool = True
    flow_window: int = 15
    save_shifted_instances: bool = True

    def run(self, input_video: Union[str, Video], output_path: str):
        """
        Run the entire inference pipeline on an input video or file object.

        Args:
            input_video: Either a video object or video filename.
            output_path: The output path to save the results.

        Returns:
            None
        """

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
        matched_instances: List[LabeledFrame] = []
        num_chunks = int(np.ceil(num_frames / self.read_chunk_size))
        frame_idx = 0
        for chunk in range(num_chunks):
            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()
            # Calculate how many frames to read
            # num_chunk_frames = min(read_chunk_size, num_frames - int(vid.get(cv2.CAP_PROP_POS_FRAMES)))

            # Read the next batch of images
            t0 = time()

            # Read the next chunk of frames
            frame_end = frame_idx + self.read_chunk_size
            if frame_end > vid.num_frames:
                frame_end = vid.num_frames
            frames_idx = np.arange(frame_idx, frame_end)
            mov = vid[frame_idx:frame_end]

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
                # TODO: figure out when/if this is necessary for RGB videos
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
            instances = match_peaks_paf_par(peaks, peak_vals, pafs, self.skeleton,
                                            min_score_to_node_ratio=self.min_score_to_node_ratio,
                                            min_score_midpts=self.min_score_midpts,
                                            min_score_integral=self.min_score_integral,
                                            add_last_edge=self.add_last_edge, pool=pool)
            logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

            # Track
            t0 = time()
            tracker.process(mov, frames_idx, instances, self.skeleton)
            logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

            # Save
            matched_instances.extend(instances)

            # Get the parameters used for this inference.
            params = attr.asdict(self, filter=lambda attr, value: attr.name not in ["model", "skeleton"])
            print(params)

            if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                t0 = time()

                save_dict = dict(params=params,
                     matched_instances=matched_instances, scale=scale,
                     uids=tracker.uids, tracked_instances=tracker.generate_tracks(matched_instances),
                    flow_assignment_costs=tracker.flow_assignment_costs)

                if self.save_shifted_instances:
                    shifted_track_id, shifted_frame_idx, shifted_frame_idx_source, shifted_points = tracker.generate_shifted_data()
                    save_dict.update(dict(
                        shifted_track_id=shifted_track_id,
                        shifted_frame_idx=shifted_frame_idx,
                        shifted_frame_idx_source=shifted_frame_idx_source,
                        shifted_points=shifted_points, ))

                with h5.File(output_path, 'w') as f:
                    save_dict_to_hdf5(f, '/', save_dict)

                    # Save the skeleton as well, in JSON to the HDF5
                    self.skeleton.save_hdf5(f)

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
    save_path = data_path + ".paf_tracking.h5"
    skeleton_path = args.skeleton_path

    # Load the model
    model = get_inference_model(confmap_model_path, paf_model_path)

    # Load the skeleton(s)
    skeleton = Skeleton.load_json(skeleton_path)
    logger.info(f"Skeleton (name={skeleton.name}, {len(skeleton.nodes)} nodes):")

    # Create a predictor to do the work.
    predictor = Predictor(model=model, skeleton=skeleton)

    # Run the inference pipeline
    predictor.run(input_video=data_path, output_path=save_path)


if __name__ == "__main__":
    main()

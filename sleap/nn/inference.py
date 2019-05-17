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

from sleap.instance import Point, Instance, LabeledFrame
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.skeleton import Skeleton
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

    @classmethod
    def from_point(cls, point: Point, score: float = 0.0):
        """
        Create a PredictedPoint from a Point

        Args:
            point: The point to copy all data from.
            score: The score for this predicted point.

        Returns:
            A scored point based on the point passed in.
        """
        return cls(**{**attr.asdict(point), 'score': score})


@attr.s(auto_attribs=True, slots=True)
class PredictedInstance(Instance):
    """
    A predicted instance is an output of the inference procedure. It is
    the main output of the inference procedure.

    Args:
        score: The instance level prediction score.
    """
    score: float = attr.ib(default=0.0)

    @classmethod
    def from_instance(cls, instance: Instance, score):
        """
        Create a PredictedInstance from and Instance object. The fields are
        copied in a shallow manner with the exception of points. For each
        point in the instance an PredictedPoint is created with score set
        to default value.

        Args:
            instance: The Instance object to shallow copy data from.
            score: The score for this instance.

        Returns:
            A PredictedInstance for the given Instance.
        """
        kw_args = attr.asdict(instance, recurse=False)
        kw_args['points'] = {key: PredictedPoint.from_point(val)
                             for key, val in kw_args['_points'].items()}
        del kw_args['_points']
        kw_args['score'] = score
        return cls(**kw_args)


def get_available_gpus():
    """
    Get the list of available GPUs

    Returns:
        List of available GPU device names
    """

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

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

    gpu_list = get_available_gpus()

    if len(gpu_list) == 0:
        logger.warn('No GPU devices, this is going to be really slow, something is wrong, dont do this!!!')
    else:
        logger.info(f'Detected {len(gpu_list)} GPU(s) for inference')

    if len(gpu_list) > 1:
        model = multi_gpu_model(model, gpus=len(gpu_list))

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


def match_peaks_frame2(peaks_t, peak_vals_t, pafs_t, skeleton,
    min_score_to_node_ratio=0.4, min_score_midpts=0.05, min_score_integral=0.8, add_last_edge=False):
    """ Matches single frame """
    # Effectively the original implementation:
    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/demo_video.py#L107

    # Dumb
    peak_ids = []
    idx = 0
    for i in range(len(peaks_t)):
        idx_i = []
        for j in range(len(peaks_t[i])):
            idx_i.append(idx)
            idx += 1
        peak_ids.append(idx_i)

    # Score each edge
    special_k = []
    connection_all = []
    for k in range(len(skeleton["edges"])):
        edge = skeleton["edges"][k]
        paf_x = pafs_t[...,2*k]
        paf_y = pafs_t[...,2*k+1]
        peaks_src = peaks_t[edge[0]]
        peaks_dst = peaks_t[edge[1]]
        peak_vals_src = peak_vals_t[edge[0]]
        peak_vals_dst = peak_vals_t[edge[1]]

        if len(peaks_src) == 0 or len(peaks_dst) == 0:
            special_k.append(k)
            connection_all.append([])
        else:
            connection_candidates = []
            for i, peak_src in enumerate(peaks_src):
                for j, peak_dst in enumerate(peaks_dst):
                    # Vector between peaks
                    vec = peak_dst - peak_src

                    # Euclidean distance between points
                    norm = np.sqrt(np.sum(vec ** 2))

                    # Failure if points overlap
                    if norm == 0:
                        continue

                    # Convert to unit vector
                    vec = vec / norm

                    # Get PAF values along edge
                    vec_x = improfile(paf_x, peak_src, peak_dst)
                    vec_y = improfile(paf_y, peak_src, peak_dst)

                    # Compute score
                    score_midpts = vec_x * vec[0] + vec_y * vec[1]
                    score_with_dist_prior = np.mean(score_midpts) + min(0.5 * paf_x.shape[0] / norm - 1, 0)
                    score_integral = np.mean(score_midpts > min_score_midpts)
                    if score_with_dist_prior > 0 and score_integral > min_score_integral:
                        connection_candidates.append([i, j, score_with_dist_prior])

            # Sort candidates for current edge by descending score
            connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)

            # Add to list of candidates for next step
            connection = np.zeros((0,5)) # src_id, dst_id, paf_score, i, j
            for candidate in connection_candidates:
                i, j, score = candidate
                # Add to connections if node is not already included
                if (i not in connection[:, 3]) and (j not in connection[:, 4]):
                    id_i = peak_ids[skeleton["edges"][k][0]][i]
                    id_j = peak_ids[skeleton["edges"][k][1]][j]
                    connection = np.vstack([connection, [id_i, id_j, score, i, j]])

                    # Stop when reached the max number of matches possible
                    if len(connection) >= min(len(peaks_src), len(peaks_dst)):
                        break
            connection_all.append(connection)

    # Greedy matching of each edge candidate set
    subset = -1 * np.ones((0, skeleton["nodes"]+2)) # ids, overall score, number of parts
    candidate = np.array([y for x in peaks_t for y in x]) # flattened set of all points
    candidate_scores = np.array([y for x in peak_vals_t for y in x]) # flattened set of all peak scores
    for k in range(len(skeleton["edges"])):
        # No matches for this edge
        if k in special_k:
            continue

        # Get IDs for current connection
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]

        # Get edge
        indexA, indexB = skeleton["edges"][k]

        # Loop through all candidates for current edge
        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]

            # Search for current candidates in matched subset
            for j in range(len(subset)):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1

            # One of the two candidate points found in matched subset
            if found == 1:
                j = subset_idx[0]
                if subset[j][indexB] != partBs[i]: # did we already assign this part?
                    subset[j][indexB] = partBs[i] # assign part
                    subset[j][-1] += 1 # increment instance part counter
                    subset[j][-2] += candidate_scores[int(partBs[i])] + connection_all[k][i][2] # add peak + edge score

            # Both candidate points found in matched subset
            elif found == 2:
                j1, j2 = subset_idx # get indices in matched subset
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2] # count number of instances per body parts
                # All body parts are disjoint, merge them
                if np.all(membership < 2):
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, axis=0)

                # Treat them separately
                else:
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate_scores[partBs[i].astype(int)] + connection_all[k][i][2]

            # Neither point found, create a new subset (if not the last edge)
            elif found == 0 and (add_last_edge or (k < (len(skeleton["edges"])-1))):
                row = -1 * np.ones(skeleton["nodes"]+2)
                row[indexA] = partAs[i] # ID
                row[indexB] = partBs[i] # ID
                row[-1] = 2 # initial count
                row[-2] = sum(candidate_scores[connection_all[k][i, :2].astype(int)]) + connection_all[k][i][2] # score
                subset = np.vstack([subset, row]) # add to matched subset

    # Filter small instances
    score_to_node_ratio = subset[:,-2] / subset[:,-1]
    subset = subset[score_to_node_ratio > min_score_to_node_ratio, :]

    # Done with all the matching! Gather the data
    matched_instances_t = []
    match_scores_t = []
    matched_peak_vals_t = []
    for match in subset:
        pts = np.full((skeleton["nodes"], 2), np.nan)
        peak_vals = np.full((skeleton["nodes"],), np.nan)
        for i in range(len(pts)):
            if match[i] >= 0:
                pts[i,:] = candidate[int(match[i]),:2]
                peak_vals[i] = candidate_scores[int(match[i])]
        matched_instances_t.append(pts)
        match_scores_t.append(match[-2]) # score
        matched_peak_vals_t.append(peak_vals)

    return matched_instances_t, match_scores_t, matched_peak_vals_t

def match_peaks_frame(peaks_t, peak_vals_t, pafs_t, skeleton,
                      min_score_to_node_ratio=0.4,
                      min_score_midpts=0.05,
                      min_score_integral=0.8,
                      add_last_edge=False):
    """
    Matches single frame
    """

    # Effectively the original implementation:
    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/demo_video.py#L107

    # Dumb
    peak_ids = []
    idx = 0
    for i in range(len(peaks_t)):
        idx_i = []
        for j in range(len(peaks_t[i])):
            idx_i.append(idx)
            idx += 1
        peak_ids.append(idx_i)

    # Score each edge
    special_k = []
    connection_all = []
    for k, edge in enumerate(skeleton.edge_names):
        src_node_idx = skeleton.node_to_index(edge[0])
        dst_node_idx = skeleton.node_to_index(edge[1])
        paf_x = pafs_t[...,2*k]
        paf_y = pafs_t[...,2*k+1]
        peaks_src = peaks_t[src_node_idx]
        peaks_dst = peaks_t[dst_node_idx]
        peak_vals_src = peak_vals_t[src_node_idx]
        peak_vals_dst = peak_vals_t[dst_node_idx]

        if len(peaks_src) == 0 or len(peaks_dst) == 0:
            special_k.append(k)
            connection_all.append([])
        else:
            connection_candidates = []
            for i, peak_src in enumerate(peaks_src):
                for j, peak_dst in enumerate(peaks_dst):
                    # Vector between peaks
                    vec = peak_dst - peak_src

                    # Euclidean distance between points
                    norm = np.sqrt(np.sum(vec ** 2))

                    # Failure if points overlap
                    if norm == 0:
                        continue

                    # Convert to unit vector
                    vec = vec / norm

                    # Get PAF values along edge
                    vec_x = improfile(paf_x, peak_src, peak_dst)
                    vec_y = improfile(paf_y, peak_src, peak_dst)

                    # Compute score
                    score_midpts = vec_x * vec[0] + vec_y * vec[1]
                    score_with_dist_prior = np.mean(score_midpts) + min(0.5 * paf_x.shape[0] / norm - 1, 0)
                    score_integral = np.mean(score_midpts > min_score_midpts)
                    if score_with_dist_prior > 0 and score_integral > min_score_integral:
                        connection_candidates.append([i, j, score_with_dist_prior])

            # Sort candidates for current edge by descending score
            connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)

            # Add to list of candidates for next step
            connection = np.zeros((0,5)) # src_id, dst_id, paf_score, i, j
            for candidate in connection_candidates:
                i, j, score = candidate
                # Add to connections if node is not already included
                if (i not in connection[:, 3]) and (j not in connection[:, 4]):
                    id_i = peak_ids[src_node_idx][i]
                    id_j = peak_ids[dst_node_idx][j]
                    connection = np.vstack([connection, [id_i, id_j, score, i, j]])

                    # Stop when reached the max number of matches possible
                    if len(connection) >= min(len(peaks_src), len(peaks_dst)):
                        break
            connection_all.append(connection)

    # Greedy matching of each edge candidate set
    subset = -1 * np.ones((0, len(skeleton.nodes)+2)) # ids, overall score, number of parts
    candidate = np.array([y for x in peaks_t for y in x]) # flattened set of all points
    candidate_scores = np.array([y for x in peak_vals_t for y in x]) # flattened set of all peak scores
    for k, edge in enumerate(skeleton.edge_names):
        # No matches for this edge
        if k in special_k:
            continue

        # Get IDs for current connection
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]

        # Get edge
        indexA, indexB = (skeleton.node_to_index(edge[0]), skeleton.node_to_index(edge[1]))

        # Loop through all candidates for current edge
        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]

            # Search for current candidates in matched subset
            for j in range(len(subset)):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1

            # One of the two candidate points found in matched subset
            if found == 1:
                j = subset_idx[0]
                if subset[j][indexB] != partBs[i]: # did we already assign this part?
                    subset[j][indexB] = partBs[i] # assign part
                    subset[j][-1] += 1 # increment instance part counter
                    subset[j][-2] += candidate_scores[int(partBs[i])] + connection_all[k][i][2] # add peak + edge score

            # Both candidate points found in matched subset
            elif found == 2:
                j1, j2 = subset_idx # get indices in matched subset
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2] # count number of instances per body parts
                # All body parts are disjoint, merge them
                if np.all(membership < 2):
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, axis=0)

                # Treat them separately
                else:
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate_scores[partBs[i].astype(int)] + connection_all[k][i][2]

            # Neither point found, create a new subset (if not the last edge)
            elif found == 0 and (add_last_edge or (k < (len(skeleton.edges)-1))):
                row = -1 * np.ones(len(skeleton.nodes)+2)
                row[indexA] = partAs[i] # ID
                row[indexB] = partBs[i] # ID
                row[-1] = 2 # initial count
                row[-2] = sum(candidate_scores[connection_all[k][i, :2].astype(int)]) + connection_all[k][i][2] # score
                subset = np.vstack([subset, row]) # add to matched subset

    # Filter small instances
    score_to_node_ratio = subset[:,-2] / subset[:,-1]
    subset = subset[score_to_node_ratio > min_score_to_node_ratio, :]

    # Done with all the matching! Gather the data
    matched_instances_t = []
    for match in subset:

        # Get teh predicted points for this predicted instance
        pts = {}
        for node_name in skeleton.node_names:
            if match[i] >= 0:
                match_idx = int(match[i])
                pt = PredictedPoint(x=candidate[match_idx,0], y=candidate[match_idx,1],
                                    score=candidate_scores[match_idx])
            else:
                pt = PredictedPoint()
            pts[node_name] = pt

        matched_instances_t.append(PredictedInstance(skeleton=skeleton, points=pts, score=match[-2]))

    return matched_instances_t

def match_peaks_paf(peaks, peak_vals, pafs, skeleton,
                    video, frame_indices,
                    min_score_to_node_ratio=0.4, min_score_midpts=0.05,
                    min_score_integral=0.8, add_last_edge=False):
    """ Computes PAF-based peak matching via greedy assignment and other such dragons """

    # Process each frame
    predicted_frames = []
    for peaks_t, peak_vals_t, pafs_t, frame_idx in zip(peaks, peak_vals, pafs, frame_indices):
        instances = match_peaks_frame2(peaks_t, peak_vals_t, pafs_t, skeleton,
                                   min_score_to_node_ratio=min_score_to_node_ratio,
                                   min_score_midpts=min_score_midpts,
                                   min_score_integral=min_score_integral,
                                   add_last_edge=add_last_edge)
        predicted_frames.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    return predicted_frames

def match_peaks_paf_par(peaks, peak_vals, pafs, skeleton,
                        video, frame_indices,
                        min_score_to_node_ratio=0.4,
                        min_score_midpts=0.05,
                        min_score_integral=0.8,
                        add_last_edge=False,
                        pool=None):
    """ Parallel version of PAF peak matching """

    if pool is None:
        pool = multiprocessing.Pool()

    futures = []
    for peaks_t, peak_vals_t, pafs_t, frame_idx in zip(peaks, peak_vals, pafs, frame_indices):
        future = pool.apply_async(match_peaks_frame, [peaks_t, peak_vals_t, pafs_t, skeleton],
                                  dict(min_score_to_node_ratio=min_score_to_node_ratio,
                                       min_score_midpts=min_score_midpts,
                                       min_score_integral=min_score_integral,
                                       add_last_edge=add_last_edge))
        futures.append(future)

    predicted_frames = []
    for future, frame_idx in zip(futures, frame_indices):
        instances = future.get()
        predicted_frames.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    return predicted_frames


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
    read_chunk_size: int = 256
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
        tracker = FlowShiftTracker(window=self.flow_window, verbosity=0)

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
        for chunk in range(num_chunks):
            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()

            # Read the next batch of images
            t0 = time()

            # Read the next chunk of frames
            frame_start = self.read_chunk_size * chunk
            frame_end = frame_start + self.read_chunk_size
            if frame_end > vid.num_frames:
                frame_end = vid.num_frames
            frames_idx = np.arange(frame_start, frame_end)
            mov = vid[frame_start:frame_end]

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

#            from scipy.io import loadmat, savemat
#            skeleton = loadmat('skeleton_legs.mat')
#            skeleton["nodes"] = skeleton["nodes"][0][0]  # convert to scalar
#            skeleton["edges"] = skeleton["edges"] - 1  # convert to 0-based indexing
#            instance2 = match_peaks_paf(peaks, peak_vals, pafs, skeleton,
#                                            video=vid, frame_indices=frames_idx,
#                                            min_score_to_node_ratio=self.min_score_to_node_ratio,
#                                            min_score_midpts=self.min_score_midpts,
#                                            min_score_integral=self.min_score_integral,
#                                            add_last_edge=self.add_last_edge)

            # Match peaks via PAFs
            t0 = time()
            instances = match_peaks_paf_par(peaks, peak_vals, pafs, self.skeleton,
                                            video=vid, frame_indices=frames_idx,
                                            min_score_to_node_ratio=self.min_score_to_node_ratio,
                                            min_score_midpts=self.min_score_midpts,
                                            min_score_integral=self.min_score_integral,
                                            add_last_edge=self.add_last_edge, pool=pool)
            logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

            # Track
            t0 = time()
            tracker.process(mov, instances)
            logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

            # Save
            matched_instances.extend(instances)

            # Get the parameters used for this inference.
            params = attr.asdict(self, filter=lambda attr, value: attr.name not in ["model", "skeleton"])

            if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                t0 = time()

                save_dict = dict(
                    params=params,
                    matched_instances=matched_instances,
                    scale=scale,
                    tracks=tracker.generate_tracks(),
                    track_occupancy=tracker.occupancy()
                )

                if self.save_shifted_instances:
                    shifted_track_id, shifted_frame_idx, shifted_frame_idx_source, shifted_points = tracker.generate_shifted_data()
                    save_dict.update(dict(
                        shifted_track_id=shifted_track_id,
                        shifted_frame_idx=shifted_frame_idx,
                        shifted_frame_idx_source=shifted_frame_idx_source,
                        shifted_points=shifted_points, ))

                # with h5.File(output_path, 'w') as f:
                #     save_dict_to_hdf5(f, '/', save_dict)
                #
                #     # Save the skeleton as well, in JSON to the HDF5
                #     self.skeleton.save_hdf5(f)

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

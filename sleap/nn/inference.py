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

from scipy.ndimage import maximum_filter, gaussian_filter
from keras.utils import multi_gpu_model

from sleap.instance import LabeledFrame, PredictedPoint, PredictedInstance
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.skeleton import Skeleton
from sleap.util import usable_cpu_count
from sleap.info.metrics import calculate_pairwise_cost
from sleap.nn.model import ModelOutputType
from sleap.nn.training import TrainingJob
from sleap.nn.tracking import FlowShiftTracker, Track
from sleap.nn.transform import DataTransform
from sleap.nn.datagen import bounding_box_nms

OVERLAPPING_INSTANCES_NMS = True

def get_available_gpus():
    """
    Get the list of available GPUs

    Returns:
        List of available GPU device names
    """

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_inference_model(
            confmap_model_path: str,
            paf_model_path: Optional[str],
            new_input_size: None) -> keras.Model:
    """ Loads and merges confmap and PAF models into one. """

    # Load
    confmap_model = keras.models.load_model(confmap_model_path)
    logger.info(f'confmap model trained on shape {confmap_model.input_shape}')

    if paf_model_path is None:
        paf_model = None
        logger.warning("No PAF model! Running in SINGLE INSTANCE mode.")
    else:
        paf_model = keras.models.load_model(paf_model_path)
        logger.info(f'paf model trained on shape {paf_model.input_shape}')

    # Rename to prevent layer naming conflict

    confmap_model.name = "confmap_" + confmap_model.name
    for i in range(len(confmap_model.layers)):
        confmap_model.layers[i].name = "confmap_" + confmap_model.layers[i].name

    if paf_model is not None:
        paf_model.name = "paf_" + paf_model.name
        for i in range(len(paf_model.layers)):
            paf_model.layers[i].name = "paf_" + paf_model.layers[i].name

    # If we have confmap and paf models to combine
    if paf_model is not None:
        # Single input
        if new_input_size is None:
            new_input = confmap_model.input
        else:
            # create new input layout with given size (h, w, channels)
            # this allows us to use model trained on cropped images
            new_input = keras.layers.Input(new_input_size)
            logger.info(f'adjusting model input shape to {new_input_size}')

        # Get rid of first layer
        confmap_model.layers.pop(0)
        paf_model.layers.pop(0)

        # Combine models with tuple output
        model = keras.Model(new_input, [confmap_model(new_input), paf_model(new_input)])

    # If we just have a confmap model
    else:
        # Resize input layer if necessary
        if new_input_size is not None:
            logger.info(f'adjusting model input shape to {new_input_size}')
            new_input = keras.layers.Input(new_input_size)
            confmap_model.layers.pop(0)
            model = keras.Model(new_input, confmap_model(new_input))

        # Otherwise just use the confmap model as is
        else:
            model = confmap_model

    model = convert_to_gpu_model(model)

    return model

def convert_to_gpu_model(model: keras.Model) -> keras.Model:
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

def find_all_single_peaks(confmaps, min_thresh=0.3):
    """
    Finds single peak for each frames/channels in a stack of confidence maps.

    Returns:
        list of points array for each confmap
        each points array is N(=channels) x 3 for x, y, peak val
    """
    all_point_arrays = []

    for confmap in confmaps:
        peaks_vals = [image_single_peak(confmap[...,i], min_thresh) for i in range(confmap.shape[-1])]
        peaks_vals = [(*point, val) for point, val in peaks_vals]
        points_array = np.stack(peaks_vals, axis=0)
        all_point_arrays.append(points_array)

    return all_point_arrays

def match_single_peaks_frame(points_array, skeleton, transform, img_idx):
    """
    Make instance from points array returned by single peak finding.
    This is for the pipeline that assumes there's exactly one instance per frame.

    Returns:
        PredictedInstance, or None if no points.
    """
    if points_array.shape[0] == 0: return None

    # apply inverse transform to points
    points_array[...,0:2] = transform.invert(img_idx, points_array[...,0:2])

    pts = dict()
    for i, node in enumerate(skeleton.nodes):
        if not any(np.isnan(points_array[i])):
            x, y, score = points_array[i]
            # FIXME: is score just peak value or something else?
            pt = PredictedPoint(x=x, y=y, score=score)
            pts[node] = pt

    matched_instance = None
    if len(pts) > 0:
        # FIXME: how should we calculate score for instance?
        inst_score = np.sum(points_array[...,2]) / len(pts)
        matched_instance = PredictedInstance(skeleton=skeleton, points=pts, score=inst_score)

    return matched_instance

def match_single_peaks_all(points_arrays, skeleton, video, transform):
    """
    Make labeled frames for the results of single peak finding.
    This is for the pipeline that assumes there's exactly one instance per frame.

    Returns:
        list of LabeledFrames
    """
    predicted_frames = []
    for img_idx, points_array in enumerate(points_arrays):
        inst = match_single_peaks_frame(points_array, skeleton, transform, img_idx)
        if inst is not None:
            frame_idx = transform.get_frame_idxs(img_idx)
            new_lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst])
            predicted_frames.append(new_lf)
    return predicted_frames

def image_single_peak(I, min_thresh):
    peak = np.unravel_index(I.argmax(), I.shape)
    val = I[peak]

    if val < min_thresh:
        y, x = (np.nan, np.nan)
        val = np.nan
    else:
        y, x = peak

    return (x, y), val

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

def match_peaks_frame(peaks_t, peak_vals_t, pafs_t, skeleton, transform, img_idx,
                      min_score_to_node_ratio=0.4,
                      min_score_midpts=0.05,
                      min_score_integral=0.8,
                      add_last_edge=False,
                      single_per_crop=True):
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

    # apply inverse transform to points
    if candidate.shape[0] > 0:
        candidate[...,0:2] = transform.invert(img_idx, candidate[...,0:2])

    # Done with all the matching! Gather the data
    matched_instances_t = []
    for match in subset:

        # Get the predicted points for this predicted instance
        pts = dict()
        for i, node_name in enumerate(skeleton.node_names):
            if match[i] >= 0:
                match_idx = int(match[i])
                pt = PredictedPoint(x=candidate[match_idx, 0], y=candidate[match_idx, 1],
                                    score=candidate_scores[match_idx])
                pts[node_name] = pt

        if len(pts):
            matched_instances_t.append(PredictedInstance(skeleton=skeleton,
                                                         points=pts,
                                                         score=match[-2]))

    # For centroid crop just return instance closest to centroid
    if single_per_crop and len(matched_instances_t) > 1 and transform.is_cropped:

        crop_centroid = np.array(((transform.crop_size//2, transform.crop_size//2),)) # center of crop box
        crop_centroid = transform.invert(img_idx, crop_centroid) # relative to original image

        # sort by distance from crop centroid
        matched_instances_t.sort(key=lambda inst: np.linalg.norm(inst.centroid - crop_centroid))

        logger.debug(f"SINGLE_INSTANCE_PER_CROP: crop has {len(matched_instances_t)} instances, filter to 1.")

        # just use closest
        matched_instances_t = matched_instances_t[0:1]

    return matched_instances_t

def instances_nms(instances: List[PredictedInstance], thresh: float=4) -> List[PredictedInstance]:
    """Remove overlapping instances from list."""
    if len(instances) <= 1: return
    
    # Look for overlapping instances
    overlap_matrix = calculate_pairwise_cost(instances, instances,
        cost_function = lambda x: np.nan if all(np.isnan(x)) else np.nanmean(x))
    
    # Set diagonals over threshold since an instance doesn't overlap with itself
    np.fill_diagonal(overlap_matrix, thresh+1)
    overlap_matrix[np.isnan(overlap_matrix)] = thresh+1

    instances_to_remove = []

    def sort_funct(inst_idx):
        # sort by number of points in instance, then by prediction score (desc)
        return (len(instances[inst_idx].nodes), -getattr(instances[inst_idx], "score", 0))

    while np.nanmin(overlap_matrix) < thresh:
        # Find the pair of instances with greatest overlap
        idx_a, idx_b = np.unravel_index(overlap_matrix.argmin(), overlap_matrix.shape)
        
        # Keep the instance with the most points (or the highest score if tied)
        idxs = sorted([idx_a, idx_b], key=sort_funct)
        pick_idx = idxs[0]
        keep_idx = idxs[-1]
        
        # Remove this instance from overlap matrix
        overlap_matrix[pick_idx, :] = thresh+1
        overlap_matrix[:, pick_idx] = thresh+1

        # Add to list of instances that we'll remove.
        # We'll remove these later so list index doesn't change now.
        instances_to_remove.append(instances[pick_idx])

    # Remove selected instances from list
    # Note that we're modifying the original list in place
    for inst in instances_to_remove:
        instances.remove(inst)

def match_peaks_paf(peaks, peak_vals, pafs, skeleton,
                    video, transform,
                    min_score_to_node_ratio=0.4, min_score_midpts=0.05,
                    min_score_integral=0.8, add_last_edge=False, single_per_crop=True,
                    **kwargs):
    """ Computes PAF-based peak matching via greedy assignment and other such dragons """

    # Process each frame
    predicted_frames = []
    for img_idx, (peaks_t, peak_vals_t, pafs_t) in enumerate(zip(peaks, peak_vals, pafs)):
        instances = match_peaks_frame(peaks_t, peak_vals_t, pafs_t, skeleton,
                                   transform, img_idx,
                                   min_score_to_node_ratio=min_score_to_node_ratio,
                                   min_score_midpts=min_score_midpts,
                                   min_score_integral=min_score_integral,
                                   add_last_edge=add_last_edge,
                                   single_per_crop=single_per_crop)
        frame_idx = transform.get_frame_idxs(img_idx)
        predicted_frames.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    # Merge/filter predicted frames
    predicted_frames = frames_post_match_peaks(predicted_frames, video)

    return predicted_frames

def match_peaks_paf_par(peaks, peak_vals, pafs, skeleton,
                        video, transform,
                        min_score_to_node_ratio=0.4,
                        min_score_midpts=0.05,
                        min_score_integral=0.8,
                        add_last_edge=False,
                        single_per_crop=True,
                        pool=None, **kwargs):
    """ Parallel version of PAF peak matching """

    if pool is None:
        pool = multiprocessing.Pool()

    futures = []
    for img_idx, (peaks_t, peak_vals_t, pafs_t) in enumerate(zip(peaks, peak_vals, pafs)):
        future = pool.apply_async(match_peaks_frame,
                                  [peaks_t, peak_vals_t, pafs_t, skeleton],
                                  dict(transform=transform, img_idx=img_idx,
                                       min_score_to_node_ratio=min_score_to_node_ratio,
                                       min_score_midpts=min_score_midpts,
                                       min_score_integral=min_score_integral,
                                       add_last_edge=add_last_edge,
                                       single_per_crop=single_per_crop,))
        futures.append(future)

    predicted_frames = []
    for img_idx, future in enumerate(futures):
        instances = future.get()
        frame_idx = transform.get_frame_idxs(img_idx)
        # Ok, since we are doing this in parallel. Objects are getting serialized and sent
        # back and forth. This causes all instances to have a different skeleton object
        # This will cause problems later because checking if two skeletons are equal is
        # an expensive operation.
        for i in range(len(instances)):
            points = {node.name: point for node, point in instances[i].nodes_points}
            instances[i] = PredictedInstance(skeleton=skeleton, points=points, score=instances[i].score)

        predicted_frames.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    # Merge/filter predicted frames
    predicted_frames = frames_post_match_peaks(predicted_frames, video)

    return predicted_frames

def frames_post_match_peaks(predicted_frames, video):
    # Combine LabeledFrame objects for the same video frame
    predicted_frames = LabeledFrame.merge_frames(predicted_frames, video=video)

    # Remove overlapping predicted instances
    if OVERLAPPING_INSTANCES_NMS:
        t0 = clock()
        for lf in predicted_frames:
            n = len(lf.instances)
            instances_nms(lf.instances)
            if len(lf.instances) < n:
                logger.info(f"    Removed {n-len(lf.instances)} overlapping instance(s) from frame {lf.frame_idx}")
        logger.info("    Instance NMS [%.1fs]" % (clock() - t0))

    return predicted_frames

@attr.s(auto_attribs=True)
class Predictor:
    """
    The Predictor class takes a trained sLEAP model and runs
    the complete inference pipeline from confidence map/part affinity field
    inference, non-maximum suppression peak finding, paf part matching, to tracking.

    Args:
        sleap_models: Dict with a TrainingJob for each required ModelOutputType;
            can be used to construct keras model.
        model: A trained keras model used for confidence map and paf inference.
        skeleton: The skeleton(s) to use for prediction.
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
        with_tracking: Should tracking be run after inference.
        flow_window: The number of frames that tracking should look back when trying to identify
        instances.

    """

    model: keras.Model = None
    skeleton: Skeleton = None
    sleap_models: Dict[ModelOutputType, TrainingJob] = None
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

    _centroid_model: keras.Model = None

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

    def _get_centroid_model(self):
        if self._centroid_model is None:
            # Load and prepare centroid model
            centroid_job = self.sleap_models[ModelOutputType.CENTROIDS]
            centroid_model_path = os.path.join(
                                        centroid_job.save_dir,
                                        centroid_job.best_model_filename)
            keras_model = keras.models.load_model(centroid_model_path)
            keras_model = convert_to_gpu_model(keras_model)
            self._centroid_model = keras_model
        return self._centroid_model

    def predict(self, input_video: Union[dict, Video],
                output_path: Optional[str] = None,
                frames: Optional[List[int]] = None,
                save_confmaps_pafs: bool = False,
                is_async: bool = False) -> List[LabeledFrame]:
        """
        Run the entire inference pipeline on an input video or file object.

        Args:
            input_video: Either a video object or an unstructured video object (dict).
            output_path (optional): The output path to save the results.
            frames (optional): List of frames to predict. If None, run entire video.
            is_async (optional): Whether running function from separate process.
                Default is False. If True, we won't spawn children.

        Returns:
            list of LabeledFrame objects
        """

        # Open the video if we need it.
        logger.info(f"Predict is async: {is_async}")

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
        logger.info("loaded video")
        # Load model if necessary
        # We do this within predict so we don't have to pickle model if running in a thread
        keras_model = self.model
        if keras_model is None and len(self.sleap_models):
            keras_model = self.load_from_training_jobs(sleap_models=self.sleap_models,
                                         frame_shape=(vid.height, vid.width, vid.channels))

        if keras_model is None:
            logger.warning("Predictor has no model.")
            raise ValueError("Predictor has no model.")

        if self.skeleton is None:
            logger.warning("Predictor has no skeleton.")
            raise ValueError("Predictor has no skeleton.")

        _, h, w, c = keras_model.input_shape
        model_channels = c
        logger.info("Loaded models:")
        logger.info("  Input shape: %d x %d x %d" % (h, w, c))

        frames = frames or list(range(vid.num_frames))
        num_frames = len(frames)
        vid_h = vid.shape[1]
        vid_w = vid.shape[2]
        scale = h / vid_h if ModelOutputType.CENTROIDS not in self.sleap_models else 1.0
        logger.info("Opened video:")
        logger.info("  Source: " + str(vid.backend))
        logger.info("  Frames: %d" % num_frames)
        logger.info("  Frame shape: %d x %d" % (vid_h, vid_w))
        logger.info("  Scale: %f" % scale)
        logger.info(f"  True Scale: {h/vid_h, w/vid_w}")
        logger.info(f"  Crop around predicted centroids? {ModelOutputType.CENTROIDS in self.sleap_models}")
        h_w_scale = np.array((h/vid_h, w/vid_w))

        # Initialize tracking
        tracker = FlowShiftTracker(window=self.flow_window, verbosity=0)

        # Initialize parallel pool
        pool = None if is_async else multiprocessing.Pool(processes=usable_cpu_count())

        # Fix the number of threads for OpenCV, not that we are using
        # anything in OpenCV that is actually multi-threaded but maybe
        # we will down the line.
        cv2.setNumThreads(usable_cpu_count())

        # Delete the output file if it exists already
        if os.path.exists(output_path):
            os.unlink(output_path)

        # Process chunk-by-chunk!
        t0_start = time()
        predicted_frames: List[LabeledFrame] = []
        num_chunks = int(np.ceil(num_frames / self.read_chunk_size))
        for chunk in range(num_chunks):
            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()

            # Read the next batch of images
            t0 = time()

            # Read the next chunk of frames
            frame_start = self.read_chunk_size * chunk
            frame_end = frame_start + self.read_chunk_size
            if frame_end > num_frames:
                frame_end = num_frames
            frames_idx = frames[frame_start:frame_end]

            # Scale/crop using tranform object
            transform = DataTransform(frame_idxs=frames_idx)

            mov_full = vid[frames_idx]

            logger.info("  Read %d frames [%.1fs]" % (len(mov_full), time() - t0))

            # Transform images (crop or scale)
            t0 = time()
            if ModelOutputType.CENTROIDS in self.sleap_models:
                # Predict centroids and crop around these
                crop_size = h # match input of keras model
                centroids = self.predict_centroids(mov_full, crop_size, self.crop_iou_threshold)

                # Check if we found any centroids
                if sum(map(len, centroids)) == 0:
                    logger.info("  No centroids found so done with this chunk.")
                    continue

                mov = transform.centroid_crop(mov_full, centroids, crop_size)

            else:
                # Scale (if target doesn't match current size)
                mov = transform.scale_to(mov_full, target_size=(h,w))
            logger.info( "  Transformed images [%.1fs]" % (time() - t0))

            # If there's no PAF model, then we're assuming there's only a single
            # instance for each frame. We'll use the highest peak from each
            # channel as the node point.
            if ModelOutputType.PART_AFFINITY_FIELD not in self.sleap_models:
                # Run inference
                t0 = time()

                confmaps = keras_model.predict(mov.astype("float32") / 255, batch_size=self.inference_batch_size)
                logger.info( "  Inferred confmaps [%.1fs]" % (time() - t0))
                logger.info(f"    confmaps: shape={confmaps.shape}, ptp={np.ptp(confmaps)}")

                # Save confmaps
                if output_path is not None and save_confmaps_pafs:
                    save_visual_outputs(
                                output_path = output_path,
                                data = dict(confmaps=confmaps, box=mov))

                # Use single highest peak for each node
                t0 = time()

                points_arrays = find_all_single_peaks(confmaps,
                                        min_thresh=self.nms_min_thresh)

                predicted_frames_chunk = match_single_peaks_all(
                                                points_arrays = points_arrays,
                                                skeleton = self.skeleton,
                                                transform = transform,
                                                video = vid)

                logger.info("  Used highest peaks to create instances [%.1fs]" % (time() - t0))

            # Otherwise, run the regular pipeline that supports multiple
            # instances per frame.
            else:
                # Run inference
                t0 = time()

                confmaps, pafs = keras_model.predict(mov.astype("float32") / 255, batch_size=self.inference_batch_size)

                logger.info( "  Inferred confmaps and PAFs [%.1fs]" % (time() - t0))
                logger.info(f"    confmaps: shape={confmaps.shape}, ptp={np.ptp(confmaps)}")
                logger.info(f"    pafs: shape={pafs.shape}, ptp={np.ptp(pafs)}")

                # Save confmaps and pafs
                if output_path is not None and save_confmaps_pafs:
                    save_visual_outputs(
                            output_path = output_path,
                            data = dict(confmaps=confmaps, pafs=pafs,
                                frame_idxs=transform.frame_idxs, bounds=transform.bounding_boxes))

                # Find peaks
                t0 = time()
                peaks, peak_vals = find_all_peaks(confmaps, min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)
                logger.info("  Found peaks [%.1fs]" % (time() - t0))

                match_peaks_function = match_peaks_paf_par if not is_async else match_peaks_paf

                # Match peaks via PAFs
                t0 = time()
                predicted_frames_chunk = match_peaks_function(peaks, peak_vals, pafs, self.skeleton,
                                                transform=transform, video=vid, frame_indices=frames_idx,
                                                min_score_to_node_ratio=self.min_score_to_node_ratio,
                                                min_score_midpts=self.min_score_midpts,
                                                min_score_integral=self.min_score_integral,
                                                add_last_edge=self.add_last_edge,
                                                single_per_crop=self.single_per_crop,
                                                pool=pool)
                logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

            logger.info(f"  Instances found on {len(predicted_frames_chunk)} images.")

            # Track
            if self.with_tracking and len(predicted_frames_chunk):
                t0 = time()
                tracker.process(mov_full, predicted_frames_chunk)
                logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

            # Save
            predicted_frames.extend(predicted_frames_chunk)

            # Get the parameters used for this inference.
            params = attr.asdict(self, filter=lambda attr, value: attr.name not in ["model", "skeleton"])

            if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                t0 = time()

                # FIXME: We are re-writing the whole output each time, this is dumb.
                #  We should save in chunks then combine at the end.
                labels = Labels(labeled_frames=predicted_frames)
                if output_path is not None:

                    if output_path.endswith('json'):
                        Labels.save_json(labels, filename=output_path, compress=True)
                    else:
                        Labels.save_hdf5(labels, filename=output_path, append=True)

                    logger.info("  Saved to: %s [%.1fs]" % (output_path, time() - t0))

            elapsed = time() - t0_chunk
            total_elapsed = time() - t0_start
            fps = len(predicted_frames) / total_elapsed
            frames_left = num_frames - len(predicted_frames)
            eta = (frames_left / fps) if fps > 0 else 0
            logger.info("  Finished chunk [%.1fs / %.1f FPS / ETA: %.1f min]" % (elapsed, fps, eta / 60))

            sys.stdout.flush()

        logger.info("Total: %.1f min" % (total_elapsed / 60))

        labels = Labels(labeled_frames=predicted_frames)
        labels.merge_matching_frames()

        if is_async:
            return labels.to_dict()
        else:
            return labels

    def predict_async(self, *args, **kwargs) -> Tuple[Pool, AsyncResult]:
        """
        Run the entire inference pipeline on an input file, using a background process.

        Args:
            See Predictor.predict().
            Note that video must be string rather than Video (which doesn't pickle).

        Returns:
            A tuple containing the multiprocessing.Process that is running predict, start() has been called.
            And the AysncResult object that will contain the result when the job finishes.
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

    def load_from_training_jobs(self,
            sleap_models: Dict[ModelOutputType, TrainingJob],
            frame_shape: tuple,
            resize_hack=True):
        """
        Load keras model and skeleton from some TrainingJobs.

        Args:
            sleap_models: Dict with a TrainingJob for each required ModelOutputType
            frame_shape: (height, width, channels) of input for inference
        Returns:
            Predictor initialized with Keras model.
        """

        # Build the Keras Model
        # This code makes assumptions about the types of TrainingJobs we've been given

        confmap_job = sleap_models[ModelOutputType.CONFIDENCE_MAP]
        confmap_model_path = os.path.join(confmap_job.save_dir, confmap_job.best_model_filename)

        if ModelOutputType.PART_AFFINITY_FIELD in sleap_models:
            pafs_job = sleap_models[ModelOutputType.PART_AFFINITY_FIELD]
            paf_model_path = os.path.join(pafs_job.save_dir, pafs_job.best_model_filename)
        else:
            pafs_job = paf_model_path = None

        scale = confmap_job.trainer.scale

        # FIXME: we're now assuming that all the videos are the same size
        vid_height, vid_width, vid_channels = frame_shape
        img_shape = (int(vid_height//(1/scale)), int(vid_width//(1/scale)), vid_channels)

        # FIXME: hack to make inference run when image size isn't right for input layer
        if resize_hack:
            img_shape = (img_shape[0]//8*8, img_shape[1]//8*8, img_shape[2])

        # if there's a centroid model, then we don't want to resize input of other models
        # since we'll instead crop the images to match the model
        if ModelOutputType.CENTROIDS in sleap_models.keys():
            img_shape = None

        # Load the model and skeleton
        keras_model = get_inference_model(confmap_model_path, paf_model_path, img_shape)
        self.skeleton = confmap_job.model.skeletons[0]

        return keras_model


def save_visual_outputs(output_path: str, data: dict):
    # output_path is full path to labels.json, so replace "json" with "h5"
    viz_output_path = output_path
    if viz_output_path.endswith(".json"):
        viz_output_path = viz_output_path[:-(len(".json"))]
    viz_output_path += ".h5"

    # write file
    with h5py.File(viz_output_path, "a") as f:
        for key, val in data.items():
            val = np.array(val)
            if key in f:
                f[key].resize(f[key].shape[0] + val.shape[0], axis=0)
                f[key][-val.shape[0]:] = val
            else:
                maxshape = (None, *val.shape[1:])
                f.create_dataset(key, data=val, maxshape=maxshape,
                    compression="gzip", compression_opts=9)

def load_predicted_labels_json_old(
        data_path: str, parsed_json: dict = None,
        adjust_matlab_indexing: bool = True,
        fix_rel_paths: bool = True) -> Labels:
    """
    Simple utitlity code to load data from Talmo's old JSON format into newer
    Labels object. This loads the prediced instances

    Args:
        data_path: The path to the JSON file.
        parsed_json: The parsed json if already loaded. Save some time if already parsed.
        adjust_matlab_indexing: Do we need to adjust indexing from MATLAB.
        fix_rel_paths: Fix paths to videos to absolute paths.

    Returns:
        A newly constructed Labels object.
    """
    if parsed_json is None:
        data = json.loads(open(data_path).read())
    else:
        data = parsed_json

    videos = pd.DataFrame(data["videos"])
    predicted_instances = pd.DataFrame(data["predicted_instances"])
    predicted_points = pd.DataFrame(data["predicted_points"])

    if adjust_matlab_indexing:
        predicted_instances.frameIdx -= 1
        predicted_points.frameIdx -= 1

        predicted_points.node -= 1

        predicted_points.x -= 1

        predicted_points.y -= 1

    skeleton = Skeleton()
    skeleton.add_nodes(data["skeleton"]["nodeNames"])
    edges = data["skeleton"]["edges"]
    if adjust_matlab_indexing:
        edges = np.array(edges) - 1
    for (src_idx, dst_idx) in edges:
        skeleton.add_edge(data["skeleton"]["nodeNames"][src_idx], data["skeleton"]["nodeNames"][dst_idx])

    if fix_rel_paths:
        for i, row in videos.iterrows():
            p = row.filepath
            if not os.path.exists(p):
                p = os.path.join(os.path.dirname(data_path), p)
                if os.path.exists(p):
                    videos.at[i, "filepath"] = p

    # Make the video objects
    video_objects = {}
    for i, row in videos.iterrows():
        if videos.at[i, "format"] == "media":
            vid = Video.from_media(videos.at[i, "filepath"])
        else:
            vid = Video.from_hdf5(filename=videos.at[i, "filepath"], dataset=videos.at[i, "dataset"])

        video_objects[videos.at[i, "id"]] = vid

    track_ids = predicted_instances['trackId'].values
    unique_track_ids = np.unique(track_ids)

    spawned_on = {track_id: predicted_instances.loc[predicted_instances['trackId'] == track_id]['frameIdx'].values[0]
                  for track_id in unique_track_ids}
    tracks = {i: Track(name=str(i), spawned_on=spawned_on[i])
              for i in np.unique(predicted_instances['trackId'].values).tolist()}

    # A function to get all the instances for a particular video frame
    def get_frame_predicted_instances(video_id, frame_idx):
        points = predicted_points
        is_in_frame = (points["videoId"] == video_id) & (points["frameIdx"] == frame_idx)
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (points["instanceId"] == instance_id)
            track_id = predicted_instances.loc[predicted_instances['id'] == instance_id]['trackId'].values[0]
            match_score = predicted_instances.loc[predicted_instances['id'] == instance_id]['matching_score'].values[0]
            track_score = predicted_instances.loc[predicted_instances['id'] == instance_id]['tracking_score'].values[0]
            instance_points = {data["skeleton"]["nodeNames"][n]: PredictedPoint(x, y, visible=v, score=confidence)
                               for x, y, n, v, confidence in
                               zip(*[points[k][is_instance] for k in ["x", "y", "node", "visible", "confidence"]])}

            instance = PredictedInstance(skeleton=skeleton,
                                         points=instance_points,
                                         track=tracks[track_id],
                                         score=match_score)
            instances.append(instance)

        return instances

    # Get the unique labeled frames and construct a list of LabeledFrame objects for them.
    frame_keys = list({(videoId, frameIdx) for videoId, frameIdx in zip(predicted_points["videoId"], predicted_points["frameIdx"])})
    frame_keys.sort()
    labels = []
    for videoId, frameIdx in frame_keys:
        label = LabeledFrame(video=video_objects[videoId], frame_idx=frameIdx,
                             instances = get_frame_predicted_instances(videoId, frameIdx))
        labels.append(label)

    return Labels(labels)

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
    parser.add_argument('--less-overlap', dest='less_overlap', action='store_const',
                    const=True, default=False,
                    help='use fewer crops and include all instances from each crop '
                    '(works best if crops are much larger than instance bounding boxes)')
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
    predictor = Predictor(sleap_models=sleap_models, with_tracking=args.with_tracking)

    if args.less_overlap:
        predictor.crop_iou_threshold = .8
        predictor.single_per_crop = False
        logger.info("Using 'less overlap' mode: crop nms iou .8, multiple instances per crop, instance nms.")

    # Run the inference pipeline
    return predictor.predict(input_video=data_path, output_path=save_path, frames=frames,
                            save_confmaps_pafs=args.save_confmaps_pafs)


if __name__ == "__main__":
   main()

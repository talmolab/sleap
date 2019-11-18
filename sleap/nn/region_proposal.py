"""This module contains utilities for generating and working with region proposals.

Region proposals are used to extract crops from a larger image for downstream
processing. This is a technique that can drastically improve performance and memory
usage when the foreground region occupies a small fraction of the image.
"""

import attr
from typing import Tuple, List
import itertools
from collections import defaultdict
import numpy as np
import tensorflow as tf

from sleap.nn import peak_finding
from sleap.nn import utils
from sleap.nn import model


@attr.s(auto_attribs=True, slots=True)
class RegionProposalSet:
    box_size: Tuple[int, int]
    sample_inds: np.ndarray
    bboxes: np.ndarray
    patches: tf.Tensor


def make_centered_bboxes(
    centroids: np.ndarray, box_length: int, center_offset: bool = True
) -> np.ndarray:
    """Generates bounding boxes centered on a set of centroid coordinates.

    This function creates fixed size bounding boxes centered on the centroids to
    be used as region proposals.

    Args:
        centroids: Numpy array of shape (n_peaks, 4) where subscripts of centroid
            locations are specified in each row as [sample, row, col, channel], or
            of shape (n_peaks, 2) where subscripts are specified as [row, col].
        box_length: A scalar integer that specifies the width and height of the
            bounding boxes centered at each centroid location.
        center_offset: If True, add 0.5 to coordinates to adjust for integer peak
            subscripts. Set this to True when going from grid subscripts to real-valued
            image coordinates in order to offset to the center rather than the top-left
            corner of each pixel.

    Returns:
        bboxes a numpy array of shape (n_peaks, 4) specifying the bounding boxes.

        Bounding boxes are specified in the format [y1, x1, y2, x2], where the
        coordinates correspond to the top-left (y1, x1) and bottom-right (y2, x2) of
        each bounding box in absolute image coordinates.
    """

    # Pull out peak subscripts.
    if centroids.shape[1] == 2:
        centroids_y, centroids_x = np.split(centroids, 2, axis=1)

    elif centroids.shape[1] == 4:
        _, centroids_y, centroids_x, _ = np.split(centroids, 4, axis=1)

    # Initialize with centroid locations.
    bboxes = np.concatenate(
        [centroids_y, centroids_x, centroids_y, centroids_x], axis=1
    )

    # Offset by half of the box length in each direction.
    bboxes += np.array(
        [
            [
                -box_length // 2,  # top
                -box_length // 2,  # left
                box_length // 2,  # bottom
                box_length // 2,  # right
            ]
        ]
    )

    # Adjust to center of the pixel.
    if center_offset:
        bboxes += 0.5

    return bboxes


def nms_bboxes(
    bboxes: np.ndarray,
    bbox_scores: np.ndarray,
    iou_threshold: float = 0.2,
    max_boxes: int = 128,
) -> np.ndarray:
    """Selects a subset of bounding boxes by NMS to minimize overlaps.

    This function is a convenience wrapper around the `TensorFlow NMS implementation
    <https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression_with_scores>`_.

    Args:
        bboxes: An array of shape (n_bboxes, 4) with rows specifying bounding boxes in
            the format [y1, x1, y2, x2].
        bbox_scores: An array of shape (n_bboxes,) specifying the score associated with
            each bounding box. These will be used to prioritize suppression.
        iou_threshold: The minimum intersection over union between a pair of bounding
            boxes to consider them as overlapping.
        max_boxes: The maximum number of bounding boxes to output.

    Returns:
        merged_bboxes a numpy array of shape (n_merged_bboxes, 4) corresponding to a
        subset of the bounding boxes after suppressing overlaps.
    """

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        bboxes, bbox_scores, max_output_size=max_boxes, iou_threshold=iou_threshold
    )

    return bboxes[selected_indices.numpy()]


def generate_merged_bboxes(
    bboxes: np.ndarray,
    bbox_scores: np.ndarray,
    merged_box_length: int,
    merge_iou_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates new candidate region proposals by merging overlapping bounding boxes.

    This function will generate new bounding boxes by merging bounding boxes that meet
    the specified IOU threshold by placing a new centered bounding box at the midpoint
    between both boxes.

    Args:
        bboxes: A starting set of possibly overlapping bounding boxes.
        bbox_scores: The corresponding scores with each bounding box.
        merged_box_length: A scalar int specifying the width and height of merged
            bounding boxes. Set this to a larger size than the original bboxes such
            that the resulting merged bbox encompasses both of the original bounding
            boxes. A conservative value is twice the original bounding box length.
        merge_iou_threshold: Scalar float specifying the minimum IOU between each pair
            of bounding boxes in order to generate a new merged bounding box.

    Returns:
        A tuple of (merged_bboxes, merged_bbox_scores).

        merged_bboxes: A numpy array of shape (n_merged_bboxes, 4) specified in the
            [y1, x1, y2, x2] format. This is a superset of the input bboxes and the
            new merged region proposals.
        merged_bbox_scores: A numpy array of shape (n_merged_bboxes,) with the
            corresponding scores. Merged bboxes will have a score that is the sum
            of the original bboxes.
    """

    # Check every pair of bounding boxes for mergers.
    merged_centroids = []
    merged_bbox_scores = []
    for (bbox_i, score_i), (bbox_j, score_j) in itertools.combinations(
        zip(bboxes, bbox_scores), 2
    ):

        # We'll generate a new merged bounding box if the pair overlaps sufficiently.
        if utils.compute_iou(bbox_i, bbox_j) > merge_iou_threshold:

            # Compute midpoint and combined score.
            merged_centroids.append(
                [(bbox_i[0] + bbox_j[2]) / 2, (bbox_i[1] + bbox_j[3]) / 2]
            )
            merged_bbox_scores.append(score_i + score_j)

    merged_centroids = np.array(merged_centroids)
    merged_bbox_scores = np.array(merged_bbox_scores)

    if len(merged_centroids) > 0:
        # Create bounding boxes from the new centroids.
        merged_bboxes = make_centered_bboxes(
            merged_centroids, box_length=merged_box_length, center_offset=False
        )

    else:
        # No mergers detected.
        merged_bboxes = np.empty((0, 4), dtype="float32")

    # Combine with the original bboxes.
    merged_bboxes = np.concatenate((bboxes, merged_bboxes), axis=0)
    merged_bbox_scores = np.concatenate((bbox_scores, merged_bbox_scores), axis=0)

    return merged_bboxes, merged_bbox_scores


def normalize_bboxes(bboxes: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    """Normalizes bounding boxes from absolute to relative image coordinates.

    Args:
        bboxes: An array of shape (n_bboxes, 4) with rows specifying bounding boxes in
            the format [y1, x1, y2, x2] in absolute image coordinates (in pixels).
        img_height: Height of image in pixels.
        img_width: Width of image in pixels.

    Returns:
        Normalized bounding boxes where all coordinates are in the range [0, 1].
    """

    h = img_height - 1.0
    w = img_width - 1.0
    return bboxes / np.array([[h, w, h, w]])


###########################################################################


@attr.s(auto_attribs=True, eq=False)
class CentroidPredictor:
    centroid_model: model.InferenceModel
    batch_size: int = 16
    smooth_confmaps: bool = False
    smoothing_kernel_size: int = 5
    smoothing_sigma: float = 3.
    peak_thresh: float = 0.3
    refine_peaks: bool = True

    def preproc(self, imgs):
        # Scale to model input size.
        imgs = utils.resize_imgs(
            imgs,
            self.centroid_model.input_scale,
            common_divisor=2 ** self.centroid_model.down_blocks,
        )

        # Convert to float32 and scale values to [0., 1.].
        imgs = utils.normalize_imgs(imgs)

        return imgs

    @tf.function
    def inference(self, imgs):
        # Model inference
        confmaps = self.centroid_model.keras_model(imgs)

        if self.smooth_confmaps:
            confmaps = peak_finding.smooth_imgs(
                confmaps,
                kernel_size=self.smoothing_kernel_size,
                sigma=self.smoothing_sigma
            )

        return confmaps

    def postproc(self, centroid_confmaps):
        # Peak finding
        centroids, centroid_vals = peak_finding.find_local_peaks(
            centroid_confmaps, min_val=self.peak_thresh)

        if self.refine_peaks:
            centroids = peak_finding.refine_peaks_local_direction(centroid_confmaps, centroids)

        centroids /= tf.constant(
            [[1, self.centroid_model.output_scale, self.centroid_model.output_scale, 1]]
        )
        return centroids, centroid_vals

    
    def predict(self, imgs):
        imgs = self.preproc(imgs)
        confmaps = utils.batched_call(self.inference, imgs, batch_size=self.batch_size)
        return self.postproc(confmaps)


@tf.function(experimental_relax_shapes=True)
def extract_patches(imgs, bboxes, sample_inds):

    bbox = bboxes[0]
    box_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

    patches = tf.image.crop_and_resize(
        imgs,
        boxes=normalize_bboxes(bboxes, imgs.shape[1], imgs.shape[2]),
        box_indices=sample_inds,
        crop_size=box_size,
    )

    # Keep in the same dtype without changing devices.
    if patches.dtype != imgs.dtype:
        with tf.device(imgs.device):
            patches = tf.cast(patches, imgs.dtype)

    return patches


@attr.s(auto_attribs=True, eq=False)
class RegionProposalExtractor:
    """
    Attributes:
        instance_box_length: Scalar int specifying the width and height of bounding
            boxes centered on individual instances (detected by centroids).
        merge_overlapping: If True, will merge bounding boxes if overlapping.
        merged_box_length: Scalar int specifing the width and height of the bounding
            boxes that will be attempted to be created when merging overlapping
            instances. This should be >= instance_box_length.
        merge_iou_threshold: Overlap threshold in order to generate candidate merged
            boxes at the midpoint between overlapping instances.
        nms_iou_threshold: Overlap threshold to use for suppressing bounding box
            overlaps via NMS. See nms_bboxes for more info.
    """

    instance_box_length: int
    merge_overlapping: bool = True
    merged_box_length: int = 0
    merge_iou_threshold: float = 0.1
    nms_iou_threshold: float = 0.25

    def generate_initial_proposals(self, centroids):
        # Create initial region proposals from bounding boxes centered on the centroids.
        all_bboxes = make_centered_bboxes(centroids, self.instance_box_length)

        return all_bboxes

    def merge_bboxes(self, bboxes, bbox_scores):

        # Generate new candidates by merging overlapping bounding boxes.
        candidate_bboxes, candidate_bbox_scores = generate_merged_bboxes(
            bboxes,
            bbox_scores,
            merged_box_length=self.merged_box_length,
            merge_iou_threshold=self.merge_iou_threshold,
        )

        # Suppress overlaps including merged proposals.
        merged_bboxes = nms_bboxes(
            candidate_bboxes,
            candidate_bbox_scores,
            iou_threshold=self.nms_iou_threshold,
        )

        return merged_bboxes

    def merge_all_bboxes(self, centroids, centroid_vals, all_bboxes):

        # Group region proposals by sample indices.
        sample_inds = centroids[:, 0].astype(int)
        sample_grouped_bboxes = utils.group_array(all_bboxes, sample_inds)
        sample_grouped_bbox_scores = utils.group_array(centroid_vals, sample_inds)

        # Merge bounding boxes that are closely overlapping.
        merged_bboxes = dict()
        for sample in sample_grouped_bboxes.keys():

            # Suppress overlaps including merged proposals.
            merged_bboxes[sample] = self.merge_bboxes(
                sample_grouped_bboxes[sample], sample_grouped_bbox_scores[sample]
            )

        return merged_bboxes

    def size_group_bboxes(self, merged_bboxes):
        # Group merged proposals by size.
        size_grouped_bboxes = defaultdict(list)
        size_grouped_sample_inds = defaultdict(list)
        for sample_ind, sample_bboxes in merged_bboxes.items():
            for bbox in sample_bboxes:

                # Compute (height, width) of bounding box.
                box_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

                # Add to the size group.
                size_grouped_bboxes[box_size].append(bbox)
                size_grouped_sample_inds[box_size].append(sample_ind)

        for box_size in size_grouped_bboxes:
            size_grouped_bboxes[box_size] = np.stack(size_grouped_bboxes[box_size])
            size_grouped_sample_inds[box_size] = np.stack(
                size_grouped_sample_inds[box_size]
            )

        return size_grouped_bboxes, size_grouped_sample_inds

    def extract_region_proposal_sets(
        self, imgs, size_grouped_bboxes, size_grouped_sample_inds
    ):

        region_proposal_sets = []
        for box_size in size_grouped_bboxes.keys():
            # Gather size grouped data.
            sample_inds = size_grouped_sample_inds[box_size]
            bboxes = size_grouped_bboxes[box_size]

            # Extract image patches for all regions in the set.
            patches = extract_patches(
                imgs, tf.cast(bboxes, tf.float32), tf.cast(sample_inds, tf.int32)
            )

            # Save proposal set.
            region_proposal_sets.append(
                RegionProposalSet(box_size, sample_inds, bboxes, patches)
            )

        return region_proposal_sets

    def extract(self, imgs, centroids, centroid_vals=None):

        if tf.is_tensor(centroids):
            centroids = centroids.numpy()

        all_bboxes = self.generate_initial_proposals(centroids)

        if self.merge_overlapping:
            if tf.is_tensor(centroid_vals):
                centroid_vals = centroid_vals.numpy()

            merged_bboxes = self.merge_all_bboxes(centroids, centroid_vals, all_bboxes)
            size_grouped_bboxes, size_grouped_sample_inds = self.size_group_bboxes(
                merged_bboxes
            )

        else:
            size_grouped_bboxes = {self.instance_box_length: all_bboxes}
            size_grouped_sample_inds = {self.instance_box_length: centroids[:, 0].astype(int)}

        region_proposal_sets = self.extract_region_proposal_sets(
            imgs, size_grouped_bboxes, size_grouped_sample_inds
        )

        return region_proposal_sets

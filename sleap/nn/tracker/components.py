"""
Functions/classes used by multiple trackers.

Main types of functions:

1. Calculate pair-wise instance similarity; used for populating similarity/cost
   matrix.

2. Pick matches based on cost matrix.

3. Other clean-up (e.g., cull instances, connect track breaks).


"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, TypeVar

import attr
import numpy as np
from scipy.optimize import linear_sum_assignment

from sleap import Instance, LabeledFrame, PredictedInstance, Track
from sleap.nn import utils

InstanceType = TypeVar("InstanceType", Instance, PredictedInstance)


def instance_similarity(
    ref_instance: InstanceType, query_instance: InstanceType
) -> float:
    """Computes similarity between instances."""

    ref_visible = ~(np.isnan(ref_instance.points_array).any(axis=1))
    dists = np.sum(
        (query_instance.points_array - ref_instance.points_array) ** 2, axis=1
    )
    similarity = np.nansum(np.exp(-dists)) / np.sum(ref_visible)

    return similarity


def centroid_distance(
    ref_instance: InstanceType, query_instance: InstanceType, cache: dict = dict()
) -> float:
    """Returns the negative distance between the centroids of two instances.

    Uses `cache` dictionary (created with function so it persists between calls)
    since without cache this method is significantly slower than others.
    """

    if ref_instance not in cache:
        cache[ref_instance] = ref_instance.centroid

    if query_instance not in cache:
        cache[query_instance] = query_instance.centroid

    a = cache[ref_instance]
    b = cache[query_instance]

    return -np.linalg.norm(a - b)


def instance_iou(
    ref_instance: InstanceType, query_instance: InstanceType, cache: dict = dict()
) -> float:
    """Computes IOU between bounding boxes of instances."""

    if ref_instance not in cache:
        cache[ref_instance] = ref_instance.bounding_box

    if query_instance not in cache:
        cache[query_instance] = query_instance.bounding_box

    a = cache[ref_instance]
    b = cache[query_instance]

    return utils.compute_iou(a, b)


def hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Wrapper for Hungarian matching algorithm in scipy."""

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def greedy_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Performs greedy bipartite matching.
    """

    # Sort edges by ascending cost.
    rows, cols = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    unassigned_edges = list(zip(rows, cols))

    # Greedily assign edges.
    assignments = []
    while len(unassigned_edges) > 0:
        # Assign the lowest cost edge.
        row_ind, col_ind = unassigned_edges.pop(0)
        assignments.append((row_ind, col_ind))

        # Remove all other edges that contain either node (in reverse order).
        for i in range(len(unassigned_edges) - 1, -1, -1):
            if unassigned_edges[i][0] == row_ind or unassigned_edges[i][1] == col_ind:
                del unassigned_edges[i]

    return assignments


def nms_instances(
    instances: list[PredictedInstance],
    iou_threshold: float | None,
    target_count: int | None = None,
) -> tuple[list[PredictedInstance], list[PredictedInstance]]:
    """Finds `Instance`s to keep using non-maximum suppression.

    Args:
        instances: The list of `PredictedInstance` objects to filter.
        iou_threshold: The IOU threshold for suppression. If None, then no suppression
            is applied and `PredictedInstance.score` is used to determine which
            instances to keep.
        target_count: The maximum number of instances to keep. Default is None,
            which means all instances are kept.

    Returns:
        A tuple of two lists: the first list contains the instances to keep, and
        the second list contains the instances to remove.
    """
    boxes = np.array([inst.bounding_box for inst in instances])
    scores = np.array([inst.score for inst in instances])
    picks: list[int] = nms_fast(boxes, scores, iou_threshold, target_count)

    to_keep = [inst for i, inst in enumerate(instances) if i in picks]
    to_remove = [inst for i, inst in enumerate(instances) if i not in picks]

    return to_keep, to_remove


def nms_fast(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    target_count: int | None = None,
) -> list[int]:
    """Finds indices of boxes to keep using non-maximum suppression.

    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        boxes: The bounding boxes to filter. Each box is represented by its coordinates
            in the format (x1, y1, x2, y2) where (x1, y1) is the corner closest to
            (0, 0) and (x2, y2) is the corner farthest from (0, 0). Shape is (N, 4)
            where N is the number of boxes.
        scores: The scores for each bounding box (e.g., confidence scores associated
            with `PredictedInstance`). Scores are used to pick which boxes to evaluate
            first (i.e. the highest scoring box is never removed). Shape is (N,) where
            N is the number of boxes.
        iou_threshold: The IOU threshold for suppression. This is a soft threshold since
            boxes are added back if there are too few boxes picked (determined by
            `target_count`).
        target_count: The maximum number of boxes to keep. Default is None, which
            means only boxes with IOU less than the threshold are kept.

    Returns:
        A list of indices of the boxes that have an IOU less than the IOU threshold.
    """
    # Return an empty list if no boxes.
    if len(boxes) == 0:
        return []

    # Return all boxes (if target_count is None or greater than the number of boxes).
    if target_count and len(boxes) < target_count:
        return list(range(len(boxes)))

    # Convert boxes to float if they are integers (for higher precision division).
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Grab the coordinates of all the bounding boxes.
    x1 = boxes[:, 0]  # x1 <= x2
    y1 = boxes[:, 1]  # y1 <= y2
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of all the bounding boxes.
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    picked_idxs = []
    nms_idxs = []
    idxs = np.argsort(scores)  # The higher-scoring boxes are at the end of the list.
    while len(idxs) > 0:  # Each iteration, we remove the last box in `idxs`.
        # Get highest score box (last in list) and add to picked boxes.
        picked_box_idx = idxs[-1]
        picked_idxs.append(picked_box_idx)

        # Find the smallest (x, y) coordinates for corner 1 of the bounding box.
        xx1 = np.maximum(x1[picked_box_idx], x1[idxs[:-1]])
        yy1 = np.maximum(y1[picked_box_idx], y1[idxs[:-1]])

        # Find the largest (x, y) coordinates for corner 2 of the bounding box.
        xx2 = np.minimum(x2[picked_box_idx], x2[idxs[:-1]])
        yy2 = np.minimum(y2[picked_box_idx], y2[idxs[:-1]])

        # Compute the ratio of overlap.
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[:-1]]

        # Find and remove boxes with iou over threshold.
        nms_for_new_box = np.where(overlap > iou_threshold)[0]
        nms_idxs.extend(list(idxs[nms_for_new_box]))  # In case we need to add back.
        idxs = np.delete(idxs, nms_for_new_box)

        # Remove the last box (the one we just picked).
        idxs = idxs[:-1]

    # Add some boxes back if we have too few picked boxes.
    if target_count and nms_idxs and len(picked_idxs) < target_count:
        # Add back boxes with the highest scores.
        nms_idxs.sort(key=lambda idx: -scores[idx])
        add_back_count = min(len(nms_idxs), len(picked_idxs) - target_count)
        picked_idxs.extend(nms_idxs[:add_back_count])

    return picked_idxs


def cull_instances(
    frames: List[LabeledFrame],
    instance_count: int,
    iou_threshold: Optional[float] = None,
) -> None:
    """Removes instances from frames over instance per frame threshold.

    Args:
        frames: The list of `LabeledFrame` objects with predictions.
        instance_count: The maximum number of instances we want per frame.
        iou_threshold: Intersection over Union (IOU) threshold to use when
            removing overlapping instances over target count; if None, then
            only use score to determine which instances to remove.

    Returns:
        None; modifies frames in place.
    """
    if not frames:
        return

    frames.sort(key=lambda lf: lf.frame_idx)

    lf_inst_list = []
    # Find all frames with more instances than the desired threshold
    for lf in frames:
        if len(lf.predicted_instances) > instance_count:
            # List of instances which we'll pare down
            keep_instances = lf.predicted_instances

            # Use NMS to remove overlapping instances over target count
            if iou_threshold:
                keep_instances, extra_instances = nms_instances(
                    keep_instances,
                    iou_threshold=iou_threshold,
                    target_count=instance_count,
                )
                # Mark for removal
                lf_inst_list.extend([(lf, inst) for inst in extra_instances])

            # Use lower score to remove instances over target count
            if len(keep_instances) > instance_count:
                # Sort by ascending score, get target number of instances
                # from the end of list (i.e., with highest score)
                extra_instances = sorted(
                    keep_instances, key=operator.attrgetter("score")
                )[:-instance_count]

                # Mark for removal
                lf_inst_list.extend([(lf, inst) for inst in extra_instances])

    # Remove instances over per frame threshold
    for lf, inst in lf_inst_list:
        lf.instances.remove(inst)


def cull_frame_instances(
    instances_list: list[InstanceType],
    instance_count: int | None = None,
    iou_threshold: float | None = None,
    general_iou_threshold: float | None = None,
) -> list[InstanceType]:
    """Removes instances (for single frame) over instance per frame threshold.

    Args:
        instances_list: The list of instances for a single frame.
        instance_count: The maximum number of instances we want per frame. If None, then
            no limit is applied. Default is None.
        iou_threshold: Intersection over Union (IOU) threshold to use when removing
            overlapping instances over `instance_count`. If None, then only use score to
            determine which instances to remove over `instance_count`. Default is None.
        general_iou_threshold: Intersection over Union (IOU) threshold to use when
            removing overlapping instances - regardless of `instance_count`. If None,
            then no general IOU threshold is applied. Default is None.

    Returns:
        Updated `instances_list` (modified in-place).
    """
    if not instances_list:
        return

    # List of instances which we'll pare down
    keep_instances = instances_list

    # First, let's remove instances over the general IOU threshold
    if general_iou_threshold is not None:

        # Use NMS to remove overlapping instances over target count
        keep_instances, extra_instances = nms_instances(
            keep_instances,
            iou_threshold=general_iou_threshold,
        )

        # Remove the extra instances
        for inst in extra_instances:
            instances_list.remove(inst)

    # If we have no restrictions on instance count, return the list.
    if instance_count is None or len(instances_list) <= instance_count:
        return instances_list

    # Otherwise, let's determine instances to remove over the target count...
    extra_instances = []

    # ...using NMS to remove overlapping instances over target count.
    if iou_threshold is not None:
        keep_instances, extra_instances = nms_instances(
            keep_instances,
            iou_threshold=iou_threshold,
            target_count=instance_count,
        )

    # ...using lower score to remove instances over target count.
    elif len(keep_instances) > instance_count:  # Only true if no iou threshold.
        extra_instances = sorted(keep_instances, key=operator.attrgetter("score"))[
            :-instance_count
        ]

    # Remove the extra instances.
    for inst in extra_instances:
        instances_list.remove(inst)

    return instances_list


def connect_single_track_breaks(
    frames: List["LabeledFrame"], instance_count: int
) -> List["LabeledFrame"]:
    """
    Merges breaks in tracks by connecting single lost with single new track.

    Args:
        frames: The list of `LabeledFrame` objects with predictions.
        instance_count: The maximum number of instances we want per frame.

    Returns:
        Updated list of frames, also modifies frames in place.
    """
    if not frames:
        return frames

    # Move instances in new tracks into tracks that disappeared on previous frame
    fix_track_map = dict()
    last_good_frame_tracks = {inst.track for inst in frames[0].instances}
    for lf in frames:
        frame_tracks = {inst.track for inst in lf.instances}

        tracks_fixed_before = frame_tracks.intersection(set(fix_track_map.keys()))
        if tracks_fixed_before:
            for inst in lf.instances:
                if (
                    inst.track in fix_track_map
                    and fix_track_map[inst.track] not in frame_tracks
                ):
                    inst.track = fix_track_map[inst.track]
                    frame_tracks = {inst.track for inst in lf.instances}

        extra_tracks = frame_tracks - last_good_frame_tracks
        missing_tracks = last_good_frame_tracks - frame_tracks

        if len(extra_tracks) == 1 and len(missing_tracks) == 1:
            for inst in lf.instances:
                if inst.track in extra_tracks:
                    old_track = inst.track
                    new_track = missing_tracks.pop()
                    fix_track_map[old_track] = new_track
                    inst.track = new_track

                    break
        else:
            if len(frame_tracks) == instance_count:
                last_good_frame_tracks = frame_tracks

    return frames


@attr.s(auto_attribs=True, slots=True)
class Match:
    """Stores a match between a specific instance and specific track."""

    track: Track
    instance: Instance
    score: Optional[float] = None
    is_first_choice: bool = False


@attr.s(auto_attribs=True)
class FrameMatches:
    """
    Calculates (and stores) matches for a frame.

    This class encapsulates the logic to generate matches (using a custom
    matching function) from a cost matrix. One key feature is that it retains
    additional information, such as whether all the matches were first-choice
    (i.e., if each instance got the instance it would have matched to if there
    weren't other instances).

    Typically this will be created using the `from_candidate_instances` method
    which creates the cost matrix and then uses the matching function to find
    matches.

    Attributes:
        matches: the list of `Match` objects.
        cost_matrix: the cost matrix, shape is
            (number of untracked instances, number of candidate tracks).
        unmatched_instances: the instances for which we are finding matches.

    """

    matches: List[Match]
    cost_matrix: np.ndarray
    unmatched_instances: List[InstanceType] = attr.ib(factory=list)

    @property
    def has_only_first_choice_matches(self) -> bool:
        """Whether all the matches were first-choice.

        A match is a 'first-choice' for an instance if that instance would have
        matched to the same track even if there were no other instances.
        """
        return all(match.is_first_choice for match in self.matches)

    @classmethod
    def from_candidate_instances(
        cls,
        untracked_instances: List[InstanceType],
        candidate_instances: List[InstanceType],
        similarity_function: Callable,
        matching_function: Callable,
        robust_best_instance: float = 1.0,
    ):
        """Calculates (and stores) matches for a frame from candidate instance.

        Args:
            untracked_instances: list of untracked instances in the frame.
            candidate_instances: list of instances use as match.
            similarity_function: a function that returns the similarity between
                two instances (untracked and candidate).
            matching_function: function used to find the best match from the
                cost matrix. See the classmethod `from_cost_matrix`.
            robust_best_instance (float): if the value is between 0 and 1
                (excluded), use a robust quantile similarity score for the
                track. If the value is 1, use the max similarity (non-robust).
                For selecting a robust score, 0.95 is a good value.

        """
        cost = np.ndarray((0,))
        candidate_tracks = []

        if candidate_instances:

            # Group candidate instances by track.
            candidate_instances_by_track = defaultdict(list)
            for instance in candidate_instances:
                candidate_instances_by_track[instance.track].append(instance)

            # Compute similarity matrix between untracked instances and best
            # candidate for each track.
            candidate_tracks = list(candidate_instances_by_track.keys())
            dims = (len(untracked_instances), len(candidate_tracks))
            matching_similarities = np.full(dims, np.nan)

            for i, untracked_instance in enumerate(untracked_instances):

                for j, candidate_track in enumerate(candidate_tracks):
                    # Compute similarity between untracked instance and all track
                    # candidates.
                    track_instances = candidate_instances_by_track[candidate_track]
                    track_matching_similarities = [
                        similarity_function(
                            untracked_instance,
                            candidate_instance,
                        )
                        for candidate_instance in track_instances
                    ]

                    if 0 < robust_best_instance < 1:
                        # Robust, use the similarity score in the q-quantile for matching.
                        best_similarity = np.quantile(
                            track_matching_similarities,
                            robust_best_instance,
                        )
                    else:
                        # Non-robust, use the max similarity score for matching.
                        best_similarity = np.max(track_matching_similarities)
                    # Keep the best similarity score for this track.
                    matching_similarities[i, j] = best_similarity

            # Perform matching between untracked instances and candidates.
            cost = -matching_similarities
            cost[np.isnan(cost)] = np.inf

        return cls.from_cost_matrix(
            cost,
            untracked_instances,
            candidate_tracks,
            matching_function,
        )

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.ndarray,
        instances: List[InstanceType],
        tracks: List[Track],
        matching_function: Callable,
    ):
        matches = []
        match_instance_inds = []

        if instances and tracks:
            match_inds = matching_function(cost_matrix)

            # Determine the first-choice match for each instance since we want
            # to know whether or not all the matches in the frame were
            # uncontested.
            best_matches_vector = cost_matrix.argmin(axis=1)

            # Assign each matched instance.
            for i, j in match_inds:
                match_instance_inds.append(i)
                match_instance = instances[i]
                match_track = tracks[j]
                match_similarity = -cost_matrix[i, j]

                is_first_choice = best_matches_vector[i] == j

                # return matches as tuples
                matches.append(
                    Match(
                        instance=match_instance,
                        track=match_track,
                        score=match_similarity,
                        is_first_choice=is_first_choice,
                    )
                )

        # Make list of untracked instances which we didn't match to anything
        unmatched_instances = [
            untracked_instance
            for i, untracked_instance in enumerate(instances)
            if i not in match_instance_inds
        ]

        return cls(
            cost_matrix=cost_matrix,
            matches=matches,
            unmatched_instances=unmatched_instances,
        )


def first_choice_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns match indices where each row gets matched to best column.

    The means that multiple rows might be matched to the same column.
    """
    row_count = len(cost_matrix)
    best_matches_vector = cost_matrix.argmin(axis=1)
    match_indices = list(zip(range(row_count), best_matches_vector))

    return match_indices

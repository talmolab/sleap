import functools
from collections import deque, defaultdict
import attr
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Callable, List, Dict, Deque, Any, Tuple, Union, TypeVar

from sleap.nn import utils
from sleap.instance import Instance, PredictedInstance, Track
from sleap.io.dataset import LabeledFrame
from sleap.skeleton import Skeleton

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
    ref_instance: InstanceType, query_instance: InstanceType
) -> float:
    """Returns the negative distance between the centroids of two instances."""

    return -np.linalg.norm(ref_instance.centroid - query_instance.centroid)


def instance_iou(ref_instance: InstanceType, query_instance: InstanceType) -> float:
    """Computes IOU between bounding boxes of instances."""

    return utils.compute_iou(ref_instance.bounding_box, query_instance.bounding_box)


def hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Wrapper for Hungarian matching algorithm in scipy."""

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def greedy_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Performs greedy bipartite matching."""

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


@attr.s(eq=False, slots=True, auto_attribs=True)
class ShiftedInstance:

    points_array: np.ndarray = attr.ib()
    skeleton: Skeleton = attr.ib()
    frame: LabeledFrame = attr.ib()
    track: Track = attr.ib()
    shift_score: np.ndarray = attr.ib()

    @property
    def points(self):
        return self.points_array

    @classmethod
    def from_instance(
        cls,
        ref_instance: InstanceType,
        new_points_array: np.ndarray = None,
        shift_score: float = 0.0,
        with_skeleton: bool = False,
    ):

        points_array = new_points_array
        if points_array is None:
            points_array = ref_instance.points_array

        skeleton = None
        if with_skeleton:
            skeleton = ref_instance.skeleton

        return cls(
            points_array=points_array,
            skeleton=skeleton,
            frame=ref_instance.frame,
            track=ref_instance.track,
            shift_score=shift_score,
        )


def flow_shift_instances(
    ref_instances: List[InstanceType],
    ref_img: np.ndarray,
    new_img: np.ndarray,
    min_shifted_points: int = 0,
    scale: float = 1.0,
    window_size: int = 21,
    max_levels: int = 3,
) -> List[ShiftedInstance]:
    """Generates instances in a new frame by applying optical flow displacements.

    Args:
        ref_instances: Reference instances in the previous frame.
        ref_img: Previous frame image as a numpy array.
        new_img: New frame image as a numpy array.
        min_shifted_points: Minimum number of points that must be detected in the new
            frame in order to generate a new shifted instance.
        scale: Factor to scale the images by when computing optical flow. Decrease this
            to increase performance at the cost of finer accuracy. Sometimes decreasing
            the image scale can improve performance with fast movements.
        window_size: Optical flow window size to consider at each pyramid scale level.
        max_levels: Number of pyramid scale levels to consider. This is different from
            the scale parameter, which determines the initial image scaling.

    Returns:
        A list of ShiftedInstances with the optical flow displacements applied to the
        reference instance points. Points that are not found will be represented as
        NaNs in the points array for each shifted instance.

    Notes:
        This function relies on the Lucas-Kanade method for optical flow estimation.
    """

    # Convert RGB to grayscale.
    if ref_img.ndim > 2 and ref_img.shape[-1] == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # Ensure images are rank 2 in case there is a singleton channel dimension.
    if ref_img.ndim > 2:
        ref_img = np.squeeze(ref_img)
        new_img = np.squeeze(new_img)

    # Input image scaling.
    if scale != 1:
        ref_img = cv2.resize(ref_img, None, None, scale, scale)
        new_img = cv2.resize(new_img, None, None, scale, scale)

    # Gather reference points.
    ref_pts = [inst.points_array for inst in ref_instances]

    # Compute optical flow at all points.
    shifted_pts, status, errs = cv2.calcOpticalFlowPyrLK(
        ref_img,
        new_img,
        (np.concatenate(ref_pts, axis=0)).astype("float32") * scale,
        None,
        winSize=(window_size, window_size),
        maxLevel=max_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01,),
    )
    shifted_pts /= scale

    # Split results by instances.
    sections = np.cumsum([len(x) for x in ref_pts])[:-1]
    shifted_pts = np.split(shifted_pts, sections, axis=0)
    status = np.split(status, sections, axis=0)
    status_sum = [np.sum(x) for x in status]
    errs = np.split(errs, sections, axis=0)

    # Create shifted instances.
    shifted_instances = []
    for ref, pts, found, err in zip(ref_instances, shifted_pts, status, errs):
        if found.sum() > min_shifted_points:

            # Exclude points that weren't found by optical flow.
            found = found.squeeze().astype(bool)
            pts[~found] = np.nan

            # Create a shifted instance.
            shifted_instances.append(
                ShiftedInstance.from_instance(
                    ref, new_points_array=pts, shift_score=-np.mean(err[found])
                )
            )

    return shifted_instances


@attr.s(auto_attribs=True)
class FlowTracker:
    """Flow-based instance pose tracker."""

    flow_track_window: int = 5
    similarity_function: Callable = instance_similarity
    matching_function: Callable = greedy_matching
    min_shifted_points: int = 0
    img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3

    track_matching_queue: Deque[
        Tuple[int, np.ndarray, List[InstanceType]]
    ] = attr.ib()  # [..., (t, img_t, instances_t), ...]

    spawned_tracks: List[Track] = attr.ib(factory=list)

    save_tracked_instances: bool = False
    tracked_instances: Dict[int, List[InstanceType]] = attr.ib(
        factory=dict
    )  # keyed by t

    save_shifted_instances: bool = False
    shifted_instances: Dict[
        Tuple[int, int], List[ShiftedInstance]  # keyed by (src_t, dst_t)
    ] = attr.ib(factory=dict)

    @track_matching_queue.default
    def _init_matching_queue(self):
        """Factory for instantiating default matching queue with specified size."""
        return deque(maxlen=self.flow_track_window)

    @property
    def unique_tracks_in_queue(self) -> List[Track]:
        """Returns the unique tracks in the matching queue."""

        unique_tracks = {}
        for _, _, instances in self.track_matching_queue:
            for instance in instances:
                unique_tracks.add(instances.track)

        return list(unique_tracks)

    def track(
        self, untracked_instances: List[InstanceType], img: np.ndarray, t: int = None
    ) -> List[InstanceType]:
        """Performs a single step of tracking.

        Args:
            untracked_instances: List of instances to assign to tracks.
            img: Image data of the current frame for flow shifting.
            t: Current timestep. If not provided, increments from the internal queue.

        Returns:
            A list of the instances that were tracked.
        """

        # Infer timestep if not provided.
        if t is None:
            if len(self.track_matching_queue) > 0:

                # Default to last timestep + 1 if available.
                t = self.track_matching_queue[-1][0] + 1

            else:
                t = 0

        # Initialize containers for tracked instances at the current timestep.
        tracked_instances = []
        tracked_inds = []

        # Process untracked instances.
        if len(untracked_instances) > 0:

            # Build a pool of matchable candidate instances.
            candidate_instances = []
            for ref_t, ref_img, ref_instances in self.track_matching_queue:

                if len(ref_instances) > 0:
                    # Flow shift reference instances to current frame.
                    shifted_instances = flow_shift_instances(
                        ref_instances,
                        ref_img,
                        img,
                        min_shifted_points=self.min_shifted_points,
                        scale=self.img_scale,
                        window_size=self.of_window_size,
                        max_levels=self.of_max_levels,
                    )

                    # Add to candidate pool.
                    candidate_instances.extend(shifted_instances)

                    # Save shifted instances.
                    if self.save_shifted_instances:
                        self.shifted_instances[(ref_t, t)] = shifted_instances

            if len(candidate_instances) > 0:

                # Group candidate instances by track.
                candidate_instances_by_track = defaultdict(list)
                for instance in candidate_instances:
                    candidate_instances_by_track[instance.track].append(instance)

                # Compute similarity matrix between untracked instances and best
                # candidate for each track.
                candidate_tracks = list(candidate_instances_by_track.keys())
                matching_similarities = np.full(
                    (len(untracked_instances), len(candidate_tracks)), np.nan
                )
                matching_candidates = []

                for i, untracked_instance in enumerate(untracked_instances):
                    matching_candidates.append([])

                    for j, candidate_track in enumerate(candidate_tracks):

                        # Compute similarity between untracked instance and all track
                        # candidates.
                        track_instances = candidate_instances_by_track[candidate_track]
                        track_matching_similarities = [
                            self.similarity_function(
                                untracked_instance, candidate_instance
                            )
                            for candidate_instance in track_instances
                        ]

                        # Keep the best scoring instance for this track.
                        best_ind = np.argmax(track_matching_similarities)
                        matching_candidates[i].append(track_instances[best_ind])

                        # Use the best similarity score for matching.
                        best_similarity = track_matching_similarities[best_ind]
                        matching_similarities[i, j] = best_similarity

                # Perform matching between untracked instances and candidates.
                cost = -matching_similarities
                cost[np.isnan(cost)] = np.inf
                matches = self.matching_function(cost)

                # Assign each matched instance.
                for i, j in matches:
                    # Pull out matched pair.
                    matched_instance = untracked_instances[i]
                    ref_instance = matching_candidates[i][j]

                    # Save matching score.
                    match_similarity = matching_similarities[i, j]

                    # Assign to track and save.
                    tracked_instances.append(
                        attr.evolve(
                            matched_instance,
                            track=ref_instance.track,
                            tracking_score=match_similarity,
                        )
                    )

                    # Keep track of the assigned instances.
                    tracked_inds.append(i)

        # Spawn a new track for each remaining untracked instance.
        for i, inst in enumerate(untracked_instances):

            # Skip if this instance was tracked.
            if i in tracked_inds:
                continue

            # Spawn new track.
            new_track = Track(spawned_on=t, name=f"track_{len(self.spawned_tracks)}")
            self.spawned_tracks.append(new_track)

            # Assign instance to the new track and save.
            tracked_instances.append(attr.evolve(inst, track=new_track))

        # Add the tracked instances to the matching buffer.
        self.track_matching_queue.append((t, img, tracked_instances))

        # Save tracked instances internally.
        if self.save_tracked_instances:
            self.tracked_instances[t] = tracked_instances

        return tracked_instances


@attr.s(auto_attribs=True)
class SimpleTracker:
    """Simple instance tracker without image data."""

    tracking_window: int = 5
    similarity_function: Callable = instance_iou
    matching_function: Callable = hungarian_matching
    min_instance_points: int = 0

    track_matching_queue: Deque[
        Tuple[int, np.ndarray, List[InstanceType]]
    ] = attr.ib()  # [..., (t, instances_t), ...]

    spawned_tracks: List[Track] = attr.ib(factory=list)

    save_tracked_instances: bool = False
    tracked_instances: Dict[int, List[InstanceType]] = attr.ib(
        factory=dict
    )  # keyed by t

    @track_matching_queue.default
    def _init_matching_queue(self):
        """Factory for instantiating default matching queue with specified size."""
        return deque(maxlen=self.tracking_window)

    @property
    def unique_tracks_in_queue(self) -> List[Track]:
        """Returns the unique tracks in the matching queue."""

        unique_tracks = {}
        for _, _, instances in self.track_matching_queue:
            for instance in instances:
                unique_tracks.add(instances.track)

        return list(unique_tracks)

    def track(
        self, untracked_instances: List[InstanceType], t: int = None
    ) -> List[InstanceType]:
        """Performs a single step of tracking.

        Args:
            untracked_instances: List of instances to assign to tracks.
            t: Current timestep. If not provided, increments from the internal queue.

        Returns:
            A list of the instances that were tracked.
        """

        # Infer timestep if not provided.
        if t is None:
            if len(self.track_matching_queue) > 0:

                # Default to last timestep + 1 if available.
                t = self.track_matching_queue[-1][0] + 1

            else:
                t = 0

        # Initialize containers for tracked instances at the current timestep.
        tracked_instances = []
        tracked_inds = []

        # Process untracked instances.
        if len(untracked_instances) > 0:

            # Build a pool of matchable candidate instances.
            candidate_instances = []
            for ref_t, ref_instances in self.track_matching_queue:
                for ref_instance in ref_instances:
                    if ref_instance.n_visible_points >= self.min_instance_points:
                        candidate_instances.append(ref_instance)

            if len(candidate_instances) > 0:

                # Group candidate instances by track.
                candidate_instances_by_track = defaultdict(list)
                for instance in candidate_instances:
                    candidate_instances_by_track[instance.track].append(instance)

                # Compute similarity matrix between untracked instances and best
                # candidate for each track.
                candidate_tracks = list(candidate_instances_by_track.keys())
                matching_similarities = np.full(
                    (len(untracked_instances), len(candidate_tracks)), np.nan
                )
                matching_candidates = []

                for i, untracked_instance in enumerate(untracked_instances):
                    matching_candidates.append([])

                    for j, candidate_track in enumerate(candidate_tracks):

                        # Compute similarity between untracked instance and all track
                        # candidates.
                        track_instances = candidate_instances_by_track[candidate_track]
                        track_matching_similarities = [
                            self.similarity_function(
                                untracked_instance, candidate_instance
                            )
                            for candidate_instance in track_instances
                        ]

                        # Keep the best scoring instance for this track.
                        best_ind = np.argmax(track_matching_similarities)
                        matching_candidates[i].append(track_instances[best_ind])

                        # Use the best similarity score for matching.
                        best_similarity = track_matching_similarities[best_ind]
                        matching_similarities[i, j] = best_similarity

                # Perform matching between untracked instances and candidates.
                cost = -matching_similarities
                cost[np.isnan(cost)] = np.inf
                matches = self.matching_function(cost)

                # Assign each matched instance.
                for i, j in matches:
                    # Pull out matched pair.
                    matched_instance = untracked_instances[i]
                    ref_instance = matching_candidates[i][j]

                    # Save matching score.
                    match_similarity = matching_similarities[i, j]

                    # Assign to track and save.
                    tracked_instances.append(
                        attr.evolve(
                            matched_instance,
                            track=ref_instance.track,
                            tracking_score=match_similarity,
                        )
                    )

                    # Keep track of the assigned instances.
                    tracked_inds.append(i)

        # Spawn a new track for each remaining untracked instance.
        for i, inst in enumerate(untracked_instances):

            # Skip if this instance was tracked.
            if i in tracked_inds:
                continue

            # Spawn new track.
            new_track = Track(spawned_on=t, name=f"track_{len(self.spawned_tracks)}")
            self.spawned_tracks.append(new_track)

            # Assign instance to the new track and save.
            tracked_instances.append(attr.evolve(inst, track=new_track))

        # Add the tracked instances to the matching buffer.
        self.track_matching_queue.append((t, tracked_instances))

        # Save tracked instances internally.
        if self.save_tracked_instances:
            self.tracked_instances[t] = tracked_instances

        return tracked_instances

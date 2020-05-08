"""Tracking tools for linking grouped instances over time."""

from collections import deque, defaultdict
import attr
import numpy as np
import operator
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Callable, Deque, Dict, List, Optional, Tuple, TypeVar

from sleap.nn import utils
from sleap.instance import Instance, PredictedInstance, Track
from sleap.io.dataset import LabeledFrame
from sleap.skeleton import Skeleton
from sleap.nn.data.normalization import ensure_int

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


def nms_instances(
    instances, iou_threshold, target_count=None
) -> Tuple[List[PredictedInstance], List[PredictedInstance]]:
    boxes = np.array([inst.bounding_box for inst in instances])
    scores = np.array([inst.score for inst in instances])
    picks = nms_fast(boxes, scores, iou_threshold, target_count)

    to_keep = [inst for i, inst in enumerate(instances) if i in picks]
    to_remove = [inst for i, inst in enumerate(instances) if i not in picks]

    return to_keep, to_remove


def nms_fast(boxes, scores, iou_threshold, target_count=None) -> List[int]:
    """https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/"""

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if we already have fewer boxes than the target count, return all boxes
    if target_count and len(boxes) < target_count:
        return list(range(len(boxes)))

    # if the bounding boxes coordinates are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    picked_idxs = []

    # init list of boxes removed by nms
    nms_idxs = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by their scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:

        # we want to add the best box which is the last box in sorted list
        picked_box_idx = idxs[-1]

        # last = len(idxs) - 1
        # i = idxs[last]
        picked_idxs.append(picked_box_idx)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[picked_box_idx], x1[idxs[:-1]])
        yy1 = np.maximum(y1[picked_box_idx], y1[idxs[:-1]])
        xx2 = np.minimum(x2[picked_box_idx], x2[idxs[:-1]])
        yy2 = np.minimum(y2[picked_box_idx], y2[idxs[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]

        # find boxes with iou over threshold
        nms_for_new_box = np.where(overlap > iou_threshold)[0]
        nms_idxs.extend(list(idxs[nms_for_new_box]))

        # delete new box (last in list) plus nms boxes
        idxs = np.delete(idxs, nms_for_new_box)[:-1]

    # if we're below the target number of boxes, add some back
    if target_count and nms_idxs and len(picked_idxs) < target_count:
        # sort by descending score
        nms_idxs.sort(key=lambda idx: -scores[idx])

        add_back_count = min(len(nms_idxs), len(picked_idxs) - target_count)
        picked_idxs.extend(nms_idxs[:add_back_count])

    # return the list of picked boxes
    return picked_idxs


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

    @property
    def centroid(self):
        """Copy of Instance method."""
        points = self.points_array
        centroid = np.nanmedian(points, axis=0)
        return centroid

    @property
    def bounding_box(self):
        """Copy of Instance method."""
        points = self.points_array
        bbox = np.concatenate(
            [np.nanmin(points, axis=0)[::-1], np.nanmax(points, axis=0)[::-1]]
        )
        return bbox

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


@attr.s(auto_attribs=True, slots=True)
class MatchedInstance:

    t: int
    instances_t: List[InstanceType]
    img_t: Optional[np.ndarray] = None


@attr.s(auto_attribs=True)
class FlowCandidateMaker:
    """Class for producing optical flow shift matching candidates."""

    min_points: int = 0
    img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3

    save_shifted_instances: bool = False
    shifted_instances: Dict[
        Tuple[int, int], List[ShiftedInstance]  # keyed by (src_t, dst_t)
    ] = attr.ib(factory=dict)

    @property
    def uses_image(self):
        return True

    def get_candidates(
        self, track_matching_queue: Deque[MatchedInstance], t: int, img: np.ndarray
    ) -> List[ShiftedInstance]:
        candidate_instances = []
        for matched_item in track_matching_queue:
            ref_t, ref_img, ref_instances = (
                matched_item.t,
                matched_item.img_t,
                matched_item.instances_t,
            )

            if len(ref_instances) > 0:
                # Flow shift reference instances to current frame.
                shifted_instances = self.flow_shift_instances(
                    ref_instances,
                    ref_img,
                    img,
                    min_shifted_points=self.min_points,
                    scale=self.img_scale,
                    window_size=self.of_window_size,
                    max_levels=self.of_max_levels,
                )

                # Add to candidate pool.
                candidate_instances.extend(shifted_instances)

                # Save shifted instances.
                if self.save_shifted_instances:
                    self.shifted_instances[(ref_t, t)] = shifted_instances
        return candidate_instances

    @staticmethod
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
            min_shifted_points: Minimum number of points that must be detected in the
                new frame in order to generate a new shifted instance.
            scale: Factor to scale the images by when computing optical flow. Decrease
                this to increase performance at the cost of finer accuracy. Sometimes
                decreasing the image scale can improve performance with fast movements.
            window_size: Optical flow window size to consider at each pyramid scale
                level.
            max_levels: Number of pyramid scale levels to consider. This is different
                from the scale parameter, which determines the initial image scaling.

        Returns:
            A list of ShiftedInstances with the optical flow displacements applied to
            the reference instance points. Points that are not found will be represented
            as NaNs in the points array for each shifted instance.

        Notes:
            This function relies on the Lucas-Kanade method for optical flow estimation.
        """

        # Convert to uint8 for cv2.calcOpticalFlowPyrLK
        ref_img = ensure_int(ref_img)
        new_img = ensure_int(new_img)

        # Convert tensors to ndarays
        if hasattr(ref_img, "numpy"):
            ref_img = ref_img.numpy()

        if hasattr(new_img, "numpy"):
            new_img = new_img.numpy()

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
class SimpleCandidateMaker:
    """Class for producing list of matching candidates from prior frames."""

    min_points: int = 0

    @property
    def uses_image(self):
        return False

    def get_candidates(
        self, track_matching_queue: Deque[MatchedInstance], *args, **kwargs
    ) -> List[InstanceType]:
        # Build a pool of matchable candidate instances.
        candidate_instances = []
        for matched_item in track_matching_queue:
            ref_t, ref_instances = matched_item.t, matched_item.instances_t
            for ref_instance in ref_instances:
                if ref_instance.n_visible_points >= self.min_points:
                    candidate_instances.append(ref_instance)
        return candidate_instances


tracker_policies = dict(simple=SimpleCandidateMaker, flow=FlowCandidateMaker,)

similarity_policies = dict(
    instance=instance_similarity, centroid=centroid_distance, iou=instance_iou,
)

match_policies = dict(hungarian=hungarian_matching, greedy=greedy_matching,)


@attr.s(auto_attribs=True)
class Tracker:
    """
    Instance pose tracker.

    Use by instantiated with the desired parameters and then calling the
    `track` method for each frame.

    Attributes:
        track_window: How many frames back to look for candidate instances to
            match instances in the current frame against.
        similarity_function: A function that returns a numeric pairwise
            instance similarity value.
        matching_function: A function that takes a matrix of pairwise similarities
            and determines the matches to use.
        candidate_maker: A class instance with a `get_candidates` method
            which returns a list of Instances-like objects  which we can match
            the predicted instances in a frame against.
        cleaner: A class with a `run` method which attempts to clean tracks
            after the other tracking has run for all frames.
        min_new_track_points: We won't spawn a new track for an instance with
            fewer than this many points.
    """

    track_window: int = 5
    similarity_function: Optional[Callable] = instance_similarity
    matching_function: Callable = greedy_matching
    candidate_maker: object = attr.ib(factory=FlowCandidateMaker)
    cleaner: Optional[Callable] = None
    min_new_track_points: int = 0

    track_matching_queue: Deque[MatchedInstance] = attr.ib()

    spawned_tracks: List[Track] = attr.ib(factory=list)

    save_tracked_instances: bool = False
    tracked_instances: Dict[int, List[InstanceType]] = attr.ib(
        factory=dict
    )  # keyed by t

    @track_matching_queue.default
    def _init_matching_queue(self):
        """Factory for instantiating default matching queue with specified size."""
        return deque(maxlen=self.track_window)

    @property
    def unique_tracks_in_queue(self) -> List[Track]:
        """Returns the unique tracks in the matching queue."""

        unique_tracks = set()
        for match_item in self.track_matching_queue:
            for instance in match_item.instances_t:
                unique_tracks.add(instance.track)

        return list(unique_tracks)

    @property
    def uses_image(self):
        return getattr(self.candidate_maker, "uses_image", False)

    def track(
        self,
        untracked_instances: List[InstanceType],
        img: Optional[np.ndarray] = None,
        t: int = None,
    ) -> List[InstanceType]:
        """Performs a single step of tracking.

        Args:
            untracked_instances: List of instances to assign to tracks.
            img: Image data of the current frame for flow shifting.
            t: Current timestep. If not provided, increments from the internal queue.

        Returns:
            A list of the instances that were tracked.
        """

        if self.candidate_maker is None:
            return untracked_instances

        # Infer timestep if not provided.
        if t is None:
            if len(self.track_matching_queue) > 0:

                # Default to last timestep + 1 if available.
                t = self.track_matching_queue[-1].t + 1

            else:
                t = 0

        # Initialize containers for tracked instances at the current timestep.
        tracked_instances = []
        tracked_inds = []

        # Make cache so similarity function doesn't have to recompute everything.
        # similarity_cache = dict()

        # Process untracked instances.
        if len(untracked_instances) > 0:

            # Build a pool of matchable candidate instances.
            candidate_instances = self.candidate_maker.get_candidates(
                track_matching_queue=self.track_matching_queue, t=t, img=img,
            )

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
                                untracked_instance,
                                candidate_instance,
                                # cache=similarity_cache
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

            # Skip if this instance is too small to spawn a new track with.
            if inst.n_visible_points < self.min_new_track_points:
                continue

            # Spawn new track.
            new_track = Track(spawned_on=t, name=f"track_{len(self.spawned_tracks)}")
            self.spawned_tracks.append(new_track)

            # Assign instance to the new track and save.
            tracked_instances.append(attr.evolve(inst, track=new_track))

        # Add the tracked instances to the matching buffer.
        self.track_matching_queue.append(MatchedInstance(t, tracked_instances, img))

        # Save tracked instances internally.
        if self.save_tracked_instances:
            self.tracked_instances[t] = tracked_instances

        return tracked_instances

    def final_pass(self, frames: List[LabeledFrame]):
        """Called after tracking has run on all chunks."""
        if self.cleaner:
            self.cleaner.run(frames)

    def get_name(self):
        tracker_name = self.candidate_maker.__class__.__name__
        similarity_name = self.similarity_function.__name__
        match_name = self.matching_function.__name__
        return f"{tracker_name}.{similarity_name}.{match_name}"

    @classmethod
    def make_tracker_by_name(
        cls,
        tracker: str = "flow",
        similarity: str = "instance",
        match: str = "greedy",
        track_window: int = 5,
        min_new_track_points: int = 0,
        min_match_points: int = 0,
        img_scale: float = 1.0,
        of_window_size: int = 21,
        of_max_levels: int = 3,
        clean_instance_count: int = 0,
        clean_iou_threshold: Optional[float] = None,
        **kwargs,
    ) -> "Tracker":

        if tracker.lower() == "none":
            candidate_maker = None
            similarity_function = None
            matching_function = None
        else:
            if tracker not in tracker_policies:
                raise ValueError(f"{tracker} is not a valid tracker.")

            if similarity not in similarity_policies:
                raise ValueError(
                    f"{similarity} is not a valid tracker similarity function."
                )

            if match not in match_policies:
                raise ValueError(f"{match} is not a valid tracker matching function.")

            candidate_maker = tracker_policies[tracker](min_points=min_match_points)
            similarity_function = similarity_policies[similarity]
            matching_function = match_policies[match]

        if tracker == "flow":
            candidate_maker.img_scale = img_scale
            candidate_maker.of_window_size = of_window_size
            candidate_maker.of_max_levels = of_max_levels

        cleaner = None
        if clean_instance_count:
            cleaner = TrackCleaner(
                instance_count=clean_instance_count, iou_threshold=clean_iou_threshold
            )

        return cls(
            track_window=track_window,
            min_new_track_points=min_new_track_points,
            similarity_function=similarity_function,
            matching_function=matching_function,
            candidate_maker=candidate_maker,
            cleaner=cleaner,
        )

    @classmethod
    def get_by_name_factory_options(cls):

        options = []

        option = dict(name="tracker", default="None")
        option["type"] = str
        option["options"] = list(tracker_policies.keys()) + [
            "None",
        ]
        options.append(option)

        option = dict(name="clean_instance_count", default=0)
        option["type"] = int
        option["help"] = (
            "If non-zero, then attempt to clean tracking results "
            "assuming there are this many instances per frame."
        )
        options.append(option)

        option = dict(name="clean_iou_threshold", default=0)
        option["type"] = float
        option["help"] = (
            "If non-zero and clean_instance_count also set, "
            "then use IOU threshold to remove overlapping "
            "instances over count."
        )
        options.append(option)

        option = dict(name="similarity", default="instance")
        option["type"] = str
        option["options"] = list(similarity_policies.keys())
        options.append(option)

        option = dict(name="match", default="greedy")
        option["type"] = str
        option["options"] = list(match_policies.keys())
        options.append(option)

        option = dict(name="track_window", default=5)
        option["type"] = int
        option["help"] = "How many frames back to look for matches"
        options.append(option)

        option = dict(name="min_new_track_points", default=0)
        option["type"] = int
        option["help"] = "Minimum number of instance points for spawning new track"
        options.append(option)

        option = dict(name="min_match_points", default=0)
        option["type"] = int
        option["help"] = "Minimum points for match candidates"
        options.append(option)

        option = dict(name="img_scale", default=1.0)
        option["type"] = float
        option["help"] = "For optical-flow: Image scale"
        options.append(option)

        option = dict(name="of_window_size", default=21)
        option["type"] = int
        option[
            "help"
        ] = "For optical-flow: Optical flow window size to consider at each pyramid "
        "scale level"
        options.append(option)

        option = dict(name="of_max_levels", default=3)
        option["type"] = int
        option["help"] = "For optical-flow: Number of pyramid scale levels to consider"
        options.append(option)

        return options

    @classmethod
    def add_cli_parser_args(cls, parser, arg_scope: str = ""):
        for arg in cls.get_by_name_factory_options():
            help_string = arg.get("help", "")
            if arg.get("options", ""):
                help_string += " Options: " + ", ".join(arg["options"])
            help_string += f" (default: {arg['default']})"

            if arg_scope:
                arg_name = arg_scope + "." + arg["name"]
            else:
                arg_name = arg["name"]

            parser.add_argument(
                f"--{arg_name}", type=arg["type"], help=help_string,
            )


@attr.s(auto_attribs=True)
class FlowTracker(Tracker):
    """A Tracker pre-configured to use optical flow shifted candidates."""

    similarity_function: Callable = instance_similarity
    matching_function: Callable = greedy_matching
    candidate_maker: object = attr.ib(factory=FlowCandidateMaker)


@attr.s(auto_attribs=True)
class SimpleTracker(Tracker):
    """A Tracker pre-configured to use simple, non-image-based candidates."""

    similarity_function: Callable = instance_iou
    matching_function: Callable = hungarian_matching
    candidate_maker: object = attr.ib(factory=SimpleCandidateMaker)


@attr.s(auto_attribs=True)
class TrackCleaner:
    """
    Class for merging breaks in the predicted tracks.

    Method:
    1. You specify how many instances there should be in each frame.
    2. The lowest scoring instances beyond this limit are deleting from each frame.
    3. Going frame by frame, any time there's exactly one missing track and exactly
       one new track, we merge the new track into the missing track.

    You should review the results to check for "swaps". This can be done using the
    velocity threshold suggestion method.

    Attributes:
        instance_count: The maximum number of instances we want per frame.
        iou_threshold: Intersection over Union (IOU) threshold to use when
            removing overlapping instances over target count; if None, then
            only use score to determine which instances to remove.
    """

    instance_count: int
    iou_threshold: Optional[float] = None

    def run(self, frames: List["LabeledFrame"]):
        """
        Attempts to merge tracks for given frames.

        Args:
            frames: The list of `LabeldFrame` objects with predictions.

        Returns:
            None; modifies frames in place.
        """
        if not frames:
            return

        frames.sort(key=lambda lf: lf.frame_idx)

        lf_inst_list = []
        # Find all frames with more instances than the desired threshold
        for lf in frames:
            if len(lf.predicted_instances) > self.instance_count:
                # List of instances which we'll pare down
                keep_instances = lf.predicted_instances

                # Use NMS to remove overlapping instances over target count
                if self.iou_threshold:
                    keep_instances, extra_instances = nms_instances(
                        keep_instances,
                        iou_threshold=self.iou_threshold,
                        target_count=self.instance_count,
                    )
                    # Mark for removal
                    lf_inst_list.extend([(lf, inst) for inst in extra_instances])

                # Use lower score to remove instances over target count
                if len(keep_instances) > self.instance_count:
                    # Sort by ascending score, get target number of instances
                    # from the end of list (i.e., with highest score)
                    extra_instances = sorted(
                        keep_instances, key=operator.attrgetter("score")
                    )[: -self.instance_count]

                    # Mark for removal
                    lf_inst_list.extend([(lf, inst) for inst in extra_instances])

        # Remove instances over per frame threshold
        for lf, inst in lf_inst_list:
            lf.instances.remove(inst)

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
                if len(frame_tracks) == self.instance_count:
                    last_good_frame_tracks = frame_tracks


def run_tracker(frames, tracker):
    import time

    # Return original frames if we aren't retracking
    if tracker.similarity_function is None:
        return frames

    t0 = time.time()

    new_lfs = []

    # Run tracking on every frame
    for lf in frames:

        # Clear the tracks
        for inst in lf.instances:
            inst.track = None

        track_args = dict(untracked_instances=lf.instances)
        if tracker.uses_image:
            track_args["img"] = lf.video[lf.frame_idx]
        else:
            track_args["img"] = None

        new_lf = LabeledFrame(
            frame_idx=lf.frame_idx,
            video=lf.video,
            instances=tracker.track(**track_args),
        )
        new_lfs.append(new_lf)

        if lf.frame_idx % 100 == 0:
            print(lf.frame_idx, time.time() - t0)

    print(time.time() - t0)

    return new_lfs


def retrack():
    import argparse
    import operator
    import os
    import time

    from sleap import Labels

    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", help="Path to SLEAP project file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="The output filename to use for the predicted data.",
    )

    Tracker.add_cli_parser_args(parser)

    args = parser.parse_args()

    tracker_args = {key: val for key, val in vars(args).items() if val is not None}

    tracker = Tracker.make_tracker_by_name(**tracker_args)

    print(tracker)

    print("Loading predictions...")
    t0 = time.time()
    labels = Labels.load_file(args.data_path, args.data_path)
    frames = sorted(labels.labeled_frames, key=operator.attrgetter("frame_idx"))
    print(f"Done loading predictions in {time.time() - t0} seconds.")

    print("Starting tracker...")
    frames = run_tracker(frames=frames, tracker=tracker)
    tracker.final_pass(frames)

    new_labels = Labels(labeled_frames=frames)

    if args.output:
        output_path = args.output
    else:
        out_dir = os.path.dirname(args.data_path)
        out_name = os.path.basename(args.data_path) + f".{tracker.get_name()}.slp"
        output_path = os.path.join(out_dir, out_name)

    print(f"Saving: {output_path}")
    Labels.save_file(new_labels, output_path)


if __name__ == "__main__":
    retrack()

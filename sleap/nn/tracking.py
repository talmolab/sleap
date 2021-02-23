"""Tracking tools for linking grouped instances over time."""

from collections import deque, defaultdict
import abc
import attr
import numpy as np
import cv2
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

from sleap import Track, LabeledFrame, Skeleton

from sleap.nn.tracker.components import (
    instance_similarity,
    centroid_distance,
    instance_iou,
    hungarian_matching,
    greedy_matching,
    cull_instances,
    cull_frame_instances,
    connect_single_track_breaks,
    InstanceType,
    FrameMatches,
    Match,
)
from sleap.nn.tracker.kalman import BareKalmanTracker

from sleap.nn.data.normalization import ensure_int


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
class MatchedFrameInstances:
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
        self,
        track_matching_queue: Deque[MatchedFrameInstances],
        t: int,
        img: np.ndarray,
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

        # Ensure images are rank 2 in case there is a singleton channel dimension.
        if ref_img.ndim > 3:
            ref_img = np.squeeze(ref_img)
            new_img = np.squeeze(new_img)

        # Convert RGB to grayscale.
        if ref_img.ndim > 2 and ref_img.shape[-1] == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

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
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
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
        self, track_matching_queue: Deque[MatchedFrameInstances], *args, **kwargs
    ) -> List[InstanceType]:
        # Build a pool of matchable candidate instances.
        candidate_instances = []
        for matched_item in track_matching_queue:
            ref_t, ref_instances = matched_item.t, matched_item.instances_t
            for ref_instance in ref_instances:
                if ref_instance.n_visible_points >= self.min_points:
                    candidate_instances.append(ref_instance)
        return candidate_instances


tracker_policies = dict(
    simple=SimpleCandidateMaker,
    flow=FlowCandidateMaker,
)

similarity_policies = dict(
    instance=instance_similarity,
    centroid=centroid_distance,
    iou=instance_iou,
)

match_policies = dict(
    hungarian=hungarian_matching,
    greedy=greedy_matching,
)


@attr.s(auto_attribs=True)
class BaseTracker(abc.ABC):
    @property
    def is_valid(self):
        return False

    @abc.abstractmethod
    def track(
        self,
        untracked_instances: List[InstanceType],
        img: Optional[np.ndarray] = None,
        t: int = None,
    ):
        pass

    @property
    @abc.abstractmethod
    def uses_image(self):
        pass

    @abc.abstractmethod
    def final_pass(self, frames: List[LabeledFrame]):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass


@attr.s(auto_attribs=True)
class Tracker(BaseTracker):
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

    cleaner: Optional[Callable] = None  # todo: deprecate
    target_instance_count: int = 0
    pre_cull_function: Optional[Callable] = None
    post_connect_single_breaks: bool = False

    min_new_track_points: int = 0

    track_matching_queue: Deque[MatchedFrameInstances] = attr.ib()

    spawned_tracks: List[Track] = attr.ib(factory=list)

    save_tracked_instances: bool = False
    tracked_instances: Dict[int, List[InstanceType]] = attr.ib(
        factory=dict
    )  # keyed by t

    last_matches: Optional[FrameMatches] = None

    @property
    def is_valid(self):
        return self.similarity_function is not None

    @track_matching_queue.default
    def _init_matching_queue(self):
        """Factory for instantiating default matching queue with specified size."""
        return deque(maxlen=self.track_window)

    def reset_candidates(self):
        self.track_matching_queue = deque(maxlen=self.track_window)

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

        # Make cache so similarity function doesn't have to recompute everything.
        # similarity_cache = dict()

        # Process untracked instances.
        if untracked_instances:

            if self.pre_cull_function:
                self.pre_cull_function(untracked_instances)

            # Build a pool of matchable candidate instances.
            candidate_instances = self.candidate_maker.get_candidates(
                track_matching_queue=self.track_matching_queue,
                t=t,
                img=img,
            )

            # Determine matches for untracked instances in current frame.
            frame_matches = FrameMatches.from_candidate_instances(
                untracked_instances=untracked_instances,
                candidate_instances=candidate_instances,
                similarity_function=self.similarity_function,
                matching_function=self.matching_function,
            )

            # Store the most recent match data (for outside inspection).
            self.last_matches = frame_matches

            # Set track for each of the matched instances.
            tracked_instances.extend(
                self.update_matched_instance_tracks(frame_matches.matches)
            )

            # Spawn a new track for each remaining untracked instance.
            tracked_instances.extend(
                self.spawn_for_untracked_instances(frame_matches.unmatched_instances, t)
            )

        # Add the tracked instances to the matching buffer.
        self.track_matching_queue.append(
            MatchedFrameInstances(t, tracked_instances, img)
        )

        # Save tracked instances internally.
        if self.save_tracked_instances:
            self.tracked_instances[t] = tracked_instances

        return tracked_instances

    @staticmethod
    def update_matched_instance_tracks(matches: List[Match]) -> List[InstanceType]:
        inst_list = []
        for match in matches:
            # Assign to track and save.
            inst_list.append(
                attr.evolve(
                    match.instance,
                    track=match.track,
                    tracking_score=match.score,
                )
            )
        return inst_list

    def spawn_for_untracked_instances(
        self, unmatched_instances: List[InstanceType], t: int
    ) -> List[InstanceType]:
        results = []
        for inst in unmatched_instances:

            # Skip if this instance is too small to spawn a new track with.
            if inst.n_visible_points < self.min_new_track_points:
                continue

            # Spawn new track.
            new_track = Track(spawned_on=t, name=f"track_{len(self.spawned_tracks)}")
            self.spawned_tracks.append(new_track)

            # Assign instance to the new track and save.
            results.append(attr.evolve(inst, track=new_track))

        return results

    def final_pass(self, frames: List[LabeledFrame]):
        """Called after tracking has run on all frames to do any post-processing."""
        if self.cleaner:
            #     print(
            #         "DEPRECATION WARNING: "
            #         "--clean_instance_count is deprecated (but still applied to "
            #         "clean results *after* tracking). Use --target_instance_count "
            #         "and --pre_cull_to_target instead to cull instances *before* "
            #         "tracking."
            #     )
            self.cleaner.run(frames)
        elif self.target_instance_count and self.post_connect_single_breaks:
            connect_single_track_breaks(frames, self.target_instance_count)

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
        # Optical flow options
        img_scale: float = 1.0,
        of_window_size: int = 21,
        of_max_levels: int = 3,
        # Pre-tracking options to cull instances
        target_instance_count: int = 0,
        pre_cull_to_target: bool = False,
        pre_cull_iou_threshold: Optional[float] = None,
        # Post-tracking options to connect broken tracks
        post_connect_single_breaks: bool = False,
        # TODO: deprecate these post-tracking cleaning options
        clean_instance_count: int = 0,
        clean_iou_threshold: Optional[float] = None,
        # Kalman filter options
        kf_init_frame_count: int = 0,
        kf_node_indices: Optional[list] = None,
        **kwargs,
    ) -> BaseTracker:

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

        pre_cull_function = None
        if target_instance_count and pre_cull_to_target:

            def pre_cull_function(inst_list):
                cull_frame_instances(
                    inst_list,
                    instance_count=target_instance_count,
                    iou_threshold=pre_cull_iou_threshold,
                )

        tracker_obj = cls(
            track_window=track_window,
            min_new_track_points=min_new_track_points,
            similarity_function=similarity_function,
            matching_function=matching_function,
            candidate_maker=candidate_maker,
            cleaner=cleaner,
            pre_cull_function=pre_cull_function,
            target_instance_count=target_instance_count,
            post_connect_single_breaks=post_connect_single_breaks,
        )

        if target_instance_count and kf_init_frame_count:
            kalman_obj = KalmanTracker.make_tracker(
                init_tracker=tracker_obj,
                init_frame_count=kf_init_frame_count,
                node_indices=kf_node_indices,
                instance_count=target_instance_count,
                instance_iou_threshold=pre_cull_iou_threshold,
            )

            return kalman_obj
        elif kf_init_frame_count and not target_instance_count:
            raise ValueError("Kalman filter requires target instance count.")
        else:
            return tracker_obj

    @classmethod
    def get_by_name_factory_options(cls):

        options = []

        option = dict(name="tracker", default="None")
        option["type"] = str
        option["options"] = list(tracker_policies.keys()) + [
            "None",
        ]
        options.append(option)

        option = dict(name="target_instance_count", default=0)
        option["type"] = int
        option["help"] = "Target number of instances to track per frame."
        options.append(option)

        option = dict(name="pre_cull_to_target", default=0)
        option["type"] = int
        option["help"] = (
            "If non-zero and target_instance_count is also non-zero, then "
            "cull instances over target count per frame *before* tracking."
        )
        options.append(option)

        option = dict(name="pre_cull_iou_threshold", default=0)
        option["type"] = float
        option["help"] = (
            "If non-zero and pre_cull_to_target also set, "
            "then use IOU threshold to remove overlapping "
            "instances over count *before* tracking."
        )
        options.append(option)

        option = dict(name="post_connect_single_breaks", default=0)
        option["type"] = int
        option["help"] = (
            "If non-zero and target_instance_count is also non-zero, then "
            "connect track breaks when exactly one track is lost and exactly "
            "one track is spawned in frame."
        )
        options.append(option)

        option = dict(name="clean_instance_count", default=0)
        option["type"] = int
        option["help"] = "Target number of instances to clean *after* tracking."
        options.append(option)

        option = dict(name="clean_iou_threshold", default=0)
        option["type"] = float
        option["help"] = "IOU to use when culling instances *after* tracking."
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

        def int_list_func(s):
            return [int(x.strip()) for x in s.split(",")] if s else None

        option = dict(name="kf_node_indices", default="")
        option["type"] = int_list_func
        option["help"] = "For Kalman filter: Indices of nodes to track."
        options.append(option)

        option = dict(name="kf_init_frame_count", default="0")
        option["type"] = int
        option[
            "help"
        ] = "For Kalman filter: Number of frames to track with other tracker. 0 means no Kalman filters will be used."
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
                f"--{arg_name}",
                type=arg["type"],
                help=help_string,
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
class KalmanInitSet:
    init_frame_count: int
    instance_count: int
    node_indices: List[int]
    init_frames: list = attr.ib(factory=list)

    def add_frame_instances(
        self,
        instances: Iterable[InstanceType],
        frame_match: Optional[FrameMatches] = None,
    ):
        """Receives tracked results to be used for initializing Kalman filters."""
        is_good_frame = False

        # If we don't have a FrameMatch object, then just assume the tracking
        # is good (we're probably using pre-tracked data).
        if frame_match is None:
            is_good_frame = True

        # Since we're running the tracker to get data for initializing the
        # Kalman filters, we want to make sure the tracker is giving us good
        # results (otherwise we'll init the filters with bad results and they
        # won't work well).

        # Which frames are "good"? First, we'll see if the best track match
        # for each of the instances was distinct—i.e., no competition for
        # matching any track. Second, we'll make sure that there are enough
        # "usuable" instances—i.e., instances with the nodes that we'll track
        # using Kalman filters.
        elif frame_match.has_only_first_choice_matches:

            good_instances = [
                inst for inst in instances if self.is_usable_instance(inst)
            ]
            if len(good_instances) >= self.instance_count:
                is_good_frame = True

        if is_good_frame:
            self.init_frames.append(instances)
        else:
            # We got a bad frame so clear the list of init frames;
            # we want to get a certain number of *contiguous* good frames
            # that can be used to init the Kalman filters.
            self.reset()

    def reset(self):
        """Clears the data so we can start fresh."""
        self.init_frames = []

    def is_usable_instance(self, instance: InstanceType):
        """Is this instance usable for initializing Kalman filters?"""
        if not instance.track:
            return False
        if np.any(np.isnan(instance.points_array[self.node_indices, 0:2])):
            return False
        return True

    @property
    def is_set_ready(self) -> bool:
        """Do we have enough good data to initialize Kalman filters?"""
        return len(self.init_frames) >= self.init_frame_count

    @property
    def instances(self) -> List[InstanceType]:
        """The instances which will be used to initialize Kalman filters."""
        instances = [
            inst
            for frame in self.init_frames
            for inst in frame
            if self.is_usable_instance(inst)
        ]

        return instances


@attr.s(auto_attribs=True)
class KalmanTracker(BaseTracker):
    """
    Class for Kalman filter-based tracking pipeline.

    Kalman filters need to be initialized with a certain number of already
    tracked instances.

    Args:
        init_tracker: The regular Tracker we can use to track data needed
            for initializing Kalman filters. If not specified, then you can
            use pre-tracked data (i.e., track assignments already set on
            instances) if `pre_tracked` is True.
        init_set: Object to keep track of tracked "init" data and determine
            when we have enough good data to initialize filters.
        kalman_tracker: The object which handles the actual Kalman filter-based
            tracking.
        cull_function: If given, this is called to cull instances before tracking.
        init_frame_count: The target number of instances/identities per frame.
        re_init_cooldown: Number of frames to wait after initializing filters
            before checking if we need to re-init (because they aren't
            successfully matching tracks).
        re_init_after: If there's a gap of this many frames since filters
            have matched tracks (and we've also waited for cooldown frames),
            start using the regular tracker so that we can re-initialize
            Kalman filters.
        init_done: Keeps track of whether we're initialized the filters yet.
        pre_tracked: Whether to use `init_tracker` or tracks already set
            on instances.
        last_t: The last frame index we've tracked.
        last_init_t: The last frame index on which Kalman filters were
            initialized; used to checking cooldown period.
    """

    init_tracker: Optional[Tracker]
    init_set: KalmanInitSet
    kalman_tracker: BareKalmanTracker
    cull_function: Optional[Callable] = None
    init_frame_count: int = 10
    re_init_cooldown: int = 100
    re_init_after: int = 20
    init_done: bool = False
    pre_tracked: bool = False
    last_t: int = 0
    last_init_t: int = 0

    @property
    def is_valid(self):
        """Do we have everything we need to run tracking?"""
        return self.pre_tracked or (
            self.init_tracker is not None and self.init_tracker.is_valid
        )

    @classmethod
    def make_tracker(
        cls,
        init_tracker: Optional[Tracker],
        node_indices: List[int],
        instance_count: int,
        instance_iou_threshold: float = 0.8,
        init_frame_count: int = 10,
    ):
        """
        Creates KalmanTracker object.

        Args:
            init_tracker: The Kalman filters need to be initialized with data
                that's already been tracked. This is a regular Tracker which
                can be used to generate this tracked data (when needed).
            node_indices: Which nodes to track using Kalman filters; these
                should be nodes that are reliably present in the predictions.
            instance_count: The target number of instances to track per frame.
                A distinct Kalman filter is created/initialized to track each
                distinct identity. We'll also use this to cull the number of
                predicted instances before trying to track.
            instance_iou_threshold: This is the IOU threshold so that we first
                cull instances which have high overlap.
            init_frame_count: How many frames of tracked data to use when
                initializing Kalman filters.
        """
        kalman_tracker = BareKalmanTracker(
            node_indices=node_indices, instance_count=instance_count
        )

        def cull_function(inst_list):
            cull_frame_instances(
                inst_list,
                instance_count=instance_count,
                iou_threshold=instance_iou_threshold,
            )

        if init_tracker.pre_cull_function is None:
            init_tracker.pre_cull_function = cull_function

        return cls(
            init_tracker=init_tracker,
            kalman_tracker=kalman_tracker,
            cull_function=cull_function,
            init_frame_count=init_frame_count,
            init_set=KalmanInitSet(
                init_frame_count=init_frame_count,
                instance_count=instance_count,
                node_indices=node_indices,
            ),
        )

    def track(
        self,
        untracked_instances: List[InstanceType],
        img: Optional[np.ndarray] = None,
        t: int = None,
    ) -> List[InstanceType]:
        """Tracks individual frame, using Kalman filters if possible."""

        # Infer timestep if not provided.
        if t is None:
            t = self.last_t + 1

        self.last_t = t

        # Usually tracking works better if we cull instances over the target
        # number per frame before we try to match identities.
        if self.cull_function:
            self.cull_function(untracked_instances)

        # If the Kalman filter-based tracker hasn't yet been initialized,
        # use the "init" tracker until we've tracked enough frames, then
        # initialize the Kalman filters.
        if not self.init_done:
            # Run "init" tracker on this frame
            if self.pre_tracked:
                tracked_instances = untracked_instances
                frame_match_data = None
            else:
                tracked_instances = self.init_tracker.track(untracked_instances, img, t)
                frame_match_data = self.init_tracker.last_matches

            # Store this as tracked data that could be used to init filters.
            self.init_set.add_frame_instances(tracked_instances, frame_match_data)

            # Check if we have enough tracked frames, and if so, init filters.
            if self.init_set.is_set_ready:
                # Initialize the Kalman filters
                self.kalman_tracker.init_filters(self.init_set.instances)

                # print(f"Kalman filters initialized (frame {t})")

                # Clear the data used to init filters, so that if the filters
                # stop tracking and we need to re-init, we won't re-use the
                # tracked data from earlier frames.
                self.init_done = True
                self.last_init_t = t
                self.init_instances = []

        # Once the Kalman filter-based tracker has been initialized, use it
        # to track subsequent frames.
        else:
            # Clear any tracks that were set for pre-tracked instances.
            if self.pre_tracked:
                for inst in untracked_instances:
                    inst.track = None

            tracked_instances = self.kalman_tracker.track_frame(
                untracked_instances, frame_idx=t
            )

        # Check whether we've been getting good results from the Kalman filters.
        # First, has it been a while since the filters were initialized?
        if self.init_done and (t - self.last_init_t) > self.re_init_cooldown:

            # If it's been a while, then see if it's also been a while since
            # the filters successfully matched tracks to the instances.
            if self.kalman_tracker.last_frame_with_tracks < t - self.re_init_after:
                # Clear filters so we start tracking frames with the regular
                # "init" tracker and use this to re-initialize the Kalman
                # filters.
                self.init_done = False
                self.init_set.reset()

                # When we start using the regular tracker, we want it to start
                # with fresh tracks/match candidates.
                if self.init_tracker:
                    self.init_tracker.reset_candidates()

        return tracked_instances

    def get_name(self):
        return f"kalman.{self.init_tracker.get_name()}"

    @property
    def uses_image(self):
        return self.init_tracker.uses_image

    def final_pass(self, frames: List[LabeledFrame]):
        self.init_tracker.final_pass(frames)


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

    def run(self, frames: List[LabeledFrame]):
        cull_instances(frames, self.instance_count, self.iou_threshold)
        connect_single_track_breaks(frames, self.instance_count)


def run_tracker(frames: List[LabeledFrame], tracker: BaseTracker) -> List[LabeledFrame]:
    """Run a tracker on a set of labeled frames.

    Args:
        frames: A list of labeled frames with instances.
        tracker: An initialized Tracker.

    Returns:
        The input frames with the new tracks assigned. If the frames already had tracks,
        they will be cleared if the tracker has been re-initialized.
    """
    # Return original frames if we aren't retracking
    if not tracker.is_valid:
        return frames

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
    frames = frames  # [:1000]
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

"""
A SLEAP dataset collects labeled video frames, together with required metadata.

This contains labeled frame data (user annotations and/or predictions),
together with all the other data that is saved for a SLEAP project
(videos, skeletons, etc.).

To load a labels dataset file from disk:

> labels = Labels.load_file(filename)

If you're opening a dataset file created on a different computer (or if you've
moved the video files), it's likely that the paths to the original videos will
not work. We automatically check for the videos in the same directory as the
labels file, but if the videos aren't there, you can tell `load_file` where
to seach for the videos. There are various ways to do this:

> Labels.load_filename(filename, single_path_to_search)
> Labels.load_filename(filename, [path_a, path_b])
> Labels.load_filename(filename, callback_function)
> Labels.load_filename(filename, video_search=...)

The callback_function can be created via `make_video_callback()` and has the
option to make a callback with a GUI window so the user can locate the videos.

To save a labels dataset file, run:

> Labels.save_file(labels, filename)

If the filename has a supported extension (e.g., ".slp", ".h5", ".json") then
the file will be saved in the corresponding format. You can also specify the
default extension to use if none is provided in the filename.
"""
import itertools
import os
from collections import MutableSequence
from typing import Callable, List, Union, Dict, Optional, Tuple, Text, Iterable

import attr
import cattr
import h5py as h5
import numpy as np

try:
    from typing import ForwardRef
except:
    from typing import _ForwardRef as ForwardRef

from sleap.skeleton import Skeleton, Node
from sleap.instance import (
    Instance,
    LabeledFrame,
    Track,
    make_instance_cattr,
    PredictedInstance,
)

from sleap.io import pathutils
from sleap.io.video import Video
from sleap.gui.suggestions import SuggestionFrame
from sleap.gui.dialogs.missingfiles import MissingFilesDialog
from sleap.rangelist import RangeList
from sleap.util import uniquify, json_dumps

"""
The version number to put in the Labels JSON format.
"""
LABELS_JSON_FILE_VERSION = "2.0.0"

# For debugging, we can replace missing video files with a "dummy" video
USE_DUMMY_FOR_MISSING_VIDEOS = os.getenv("SLEAP_USE_DUMMY_VIDEOS", default="")


@attr.s(auto_attribs=True)
class LabelsDataCache:
    """Class for maintaining cache of data in labels dataset."""

    labels: "Labels"

    def __attrs_post_init__(self):
        self.update()

    def update(self, new_frame: Optional[LabeledFrame] = None):
        """Builds (or rebuilds) various caches."""
        # Data structures for caching

        if new_frame is None:
            self._lf_by_video = dict()
            self._frame_idx_map = dict()
            self._track_occupancy = dict()
            self._frame_count_cache = dict()

            for video in self.labels.videos:
                self._lf_by_video[video] = [
                    lf for lf in self.labels if lf.video == video
                ]
                self._frame_idx_map[video] = {
                    lf.frame_idx: lf for lf in self._lf_by_video[video]
                }
                self._track_occupancy[video] = self._make_track_occupancy(video)
        else:
            new_vid = new_frame.video

            if new_vid not in self._lf_by_video:
                self._lf_by_video[new_vid] = []
            if new_vid not in self._frame_idx_map:
                self._frame_idx_map[new_vid] = dict()
            self._lf_by_video[new_vid].append(new_frame)
            self._frame_idx_map[new_vid][new_frame.frame_idx] = new_frame

    def find_frames(
        self, video: Video, frame_idx: Optional[Union[int, Iterable[int]]] = None
    ) -> Optional[List[LabeledFrame]]:
        """Returns list of LabeledFrames matching video/frame_idx, or None."""
        if frame_idx is not None:
            if video not in self._frame_idx_map:
                return None

            if isinstance(frame_idx, Iterable):
                return [
                    self._frame_idx_map[video][idx]
                    for idx in frame_idx
                    if idx in self._frame_idx_map[video]
                ]

            if frame_idx not in self._frame_idx_map[video]:
                return None

            return [self._frame_idx_map[video][frame_idx]]
        else:
            if video not in self._lf_by_video:
                return None
            return self._lf_by_video[video]

    def find_fancy_frame_idxs(self, video, from_frame_idx, reverse):
        """Returns a list of frame idxs, with optional start position/order."""
        if video not in self._frame_idx_map:
            return None

        # Get sorted list of frame indexes for this video
        frame_idxs = sorted(self._frame_idx_map[video].keys())

        # Find the next frame index after (before) the specified frame
        if not reverse:
            next_frame_idx = min(
                filter(lambda x: x > from_frame_idx, frame_idxs), default=frame_idxs[0]
            )
        else:
            next_frame_idx = max(
                filter(lambda x: x < from_frame_idx, frame_idxs), default=frame_idxs[-1]
            )
        cut_list_idx = frame_idxs.index(next_frame_idx)

        # Shift list of frame indices to start with specified frame
        frame_idxs = frame_idxs[cut_list_idx:] + frame_idxs[:cut_list_idx]

        return frame_idxs

    def _make_track_occupancy(self, video: Video) -> Dict[Video, RangeList]:
        """Build cached track occupancy data."""
        frame_idx_map = self._frame_idx_map[video]

        tracks = dict()
        frame_idxs = sorted(frame_idx_map.keys())
        for frame_idx in frame_idxs:
            instances = frame_idx_map[frame_idx]
            for instance in instances:
                if instance.track not in tracks:
                    tracks[instance.track] = RangeList()
                tracks[instance.track].add(frame_idx)
        return tracks

    def get_track_occupancy(self, video: Video, track: Track) -> RangeList:
        """
        Accessor for track occupancy cache that adds video/track as needed.
        """
        if video not in self._track_occupancy:
            self._track_occupancy[video] = dict()

        if track not in self._track_occupancy[video]:
            self._track_occupancy[video][track] = RangeList()
        return self._track_occupancy[video][track]

    def get_video_track_occupancy(self, video: Video) -> Dict[Track, RangeList]:
        """Returns track occupancy information for specified video."""
        if video not in self._track_occupancy:
            self._track_occupancy[video] = dict()

        return self._track_occupancy[video]

    def remove_frame(self, frame: LabeledFrame):
        """Updates cache as needed."""
        self._lf_by_video[frame.video].remove(frame)
        # we'll assume that there's only a single LabeledFrame for this video
        # and frame_idx, and remove the frame_idx from the cache
        if frame.video in self._frame_idx_map:
            if frame.frame_idx in self._frame_idx_map[frame.video]:
                del self._frame_idx_map[frame.video][frame.frame_idx]

    def remove_video(self, video: Video):
        """Updates cache as needed."""
        # Remove from caches
        if video in self._lf_by_video:
            del self._lf_by_video[video]
        if video in self._frame_idx_map:
            del self._frame_idx_map[video]

    def track_swap(
        self,
        video: Video,
        new_track: Track,
        old_track: Optional[Track],
        frame_range: tuple,
    ):
        """Updates cache as needed."""

        # Get ranges in track occupancy cache
        _, within_old, _ = self.get_track_occupancy(video, old_track).cut_range(
            frame_range
        )
        _, within_new, _ = self.get_track_occupancy(video, new_track).cut_range(
            frame_range
        )

        if old_track is not None:
            # Instances that didn't already have track can't be handled here.
            # See track_set_instance for this case.
            self._track_occupancy[video][old_track].remove(frame_range)

        self._track_occupancy[video][new_track].remove(frame_range)
        self._track_occupancy[video][old_track].insert_list(within_new)
        self._track_occupancy[video][new_track].insert_list(within_old)

    def add_track(self, video: Video, track: Track):
        """Updates cache as needed."""
        self._track_occupancy[video][track] = RangeList()

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Updates cache as needed."""
        if frame.video not in self._track_occupancy:
            self._track_occupancy[frame.video] = dict()

        # Add track in its not already present in labels
        if instance.track not in self._track_occupancy[frame.video]:
            self._track_occupancy[frame.video][instance.track] = RangeList()

        self._track_occupancy[frame.video][instance.track].insert(
            (frame.frame_idx, frame.frame_idx + 1)
        )

        self.update_counts_for_frame(frame)

    def remove_instance(self, frame: LabeledFrame, instance: Instance):
        """Updates cache as needed."""
        if instance.track not in self._track_occupancy[frame.video]:
            return

        # If this is only instance in track in frame, then remove frame from track.
        if len(frame.find(track=instance.track)) == 1:
            self._track_occupancy[frame.video][instance.track].remove(
                (frame.frame_idx, frame.frame_idx + 1)
            )

        self.update_counts_for_frame(frame)

    def get_frame_count(self, video: Optional[Video] = None, filter: Text = ""):
        """
        Returns (possibly cached) count of frames matching video/filter.
        """
        if filter not in ("", "user", "predicted"):
            raise ValueError(
                f"Labels.get_labeled_frame_count() invalid filter: {filter}"
            )

        if video not in self._frame_count_cache:
            self._frame_count_cache[video] = dict()
        if self._frame_count_cache[video].get(filter, None) is None:
            self._frame_count_cache[video][filter] = self.get_filtered_frame_idxs(
                video, filter
            )

        return len(self._frame_count_cache[video][filter])

    def get_filtered_frame_idxs(self, video: Optional[Video] = None, filter: Text = ""):
        """
        Returns list of (video_idx, frame_idx) tuples matching video/filter.
        """
        if filter == "":
            filter_func = lambda lf: video is None or lf.video == video
        elif filter == "user":
            filter_func = (
                lambda lf: (video is None or lf.video == video)
                and lf.has_user_instances
            )
        elif filter == "predicted":
            filter_func = (
                lambda lf: (video is None or lf.video == video)
                and lf.has_predicted_instances
            )
        else:
            raise ValueError(f"Invalid filter: {filter}")

        # Make a set of (video_idx, frame_idx) tuples.
        # We'll use a set since it's faster to remove items, and we need the
        # video_idx so that we count frames from distinct videos with the same
        # frame index.

        if video is not None:
            video_idx = self.labels.videos.index(video)
            return {(video_idx, lf.frame_idx) for lf in self.labels if filter_func(lf)}

        return {
            (self.labels.videos.index(lf.video), lf.frame_idx)
            for lf in self.labels
            if filter_func(lf)
        }

    def update_counts_for_frame(self, frame: LabeledFrame):
        """
        Updated the cached count. Should be called after frame is modified.
        """
        video = frame.video

        if video is None or video not in self._frame_count_cache:
            return

        frame_idx = frame.frame_idx
        video_idx = self.labels.videos.index(video)

        # Update count of frames with user instances
        if frame.has_user_instances:
            self._add_count_cache(video, video_idx, frame_idx, "user")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "user")

        # Update count of frames with predicted instances
        if frame.has_predicted_instances:
            self._add_count_cache(video, video_idx, frame_idx, "predicted")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "predicted")

        # Update count of all labeled frames
        if len(frame.instances):
            self._add_count_cache(video, video_idx, frame_idx, "")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "")

    def _add_count_cache(self, video, video_idx, frame_idx, type_key: str):
        idx_pair = (video_idx, frame_idx)

        # Update count for this specific video
        if type_key in self._frame_count_cache[video]:
            self._frame_count_cache[video][type_key].add(idx_pair)

        # Update total for all videos
        if None in self._frame_count_cache:
            if type_key in self._frame_count_cache[None]:
                self._frame_count_cache[None][type_key].add(idx_pair)

    def _del_count_cache(self, video, video_idx, frame_idx, type_key: str):
        idx_pair = (video_idx, frame_idx)

        # Update count for this specific video
        if type_key in self._frame_count_cache[video]:
            self._frame_count_cache[video][type_key].discard(idx_pair)

        # Update total for all videos
        if None in self._frame_count_cache:
            if type_key in self._frame_count_cache[None]:
                self._frame_count_cache[None][type_key].discard(idx_pair)


@attr.s(auto_attribs=True)
class Labels(MutableSequence):
    """
    The :class:`Labels` class collects the data for a SLEAP project.

    This class is front-end for all interactions with loading, writing,
    and modifying these labels. The actual storage backend for the data
    is mostly abstracted away from the main interface.

    Attributes:
        labeled_frames: A list of :class:`LabeledFrame` objects
        videos: A list of :class:`Video` objects that these labels may or may
            not reference. The video for every `LabeledFrame` will be
            stored in `videos` attribute, but some videos in
            this list may not have any associated labeled frames.
        skeletons: A list of :class:`Skeleton` objects (again, that may or may
            not be referenced by an :class:`Instance` in labeled frame).
        tracks: A list of :class:`Track` that instances can belong to.
        suggestions: List that stores "suggested" frames for
            videos in project. These can be suggested frames for user
            to label or suggested frames for user to review.
        negative_anchors: Dictionary that stores center-points around
            which to crop as negative samples when training.
            Dictionary key is :class:`Video`, value is list of
            (frame index, x, y) tuples.
    """

    labeled_frames: List[LabeledFrame] = attr.ib(default=attr.Factory(list))
    videos: List[Video] = attr.ib(default=attr.Factory(list))
    skeletons: List[Skeleton] = attr.ib(default=attr.Factory(list))
    nodes: List[Node] = attr.ib(default=attr.Factory(list))
    tracks: List[Track] = attr.ib(default=attr.Factory(list))
    suggestions: List["SuggestionFrame"] = attr.ib(default=attr.Factory(list))
    negative_anchors: Dict[Video, list] = attr.ib(default=attr.Factory(dict))
    provenance: Dict[Text, Union[str, int, float, bool]] = attr.ib(
        default=attr.Factory(dict)
    )

    def __attrs_post_init__(self):
        """
        Called by attrs after the class is instantiated.

        This updates the top level contains (videos, skeletons, etc)
        from data in the labeled frames, as well as various caches.
        """

        # Add any videos/skeletons/nodes/tracks that are in labeled
        # frames but not in the lists on our object
        self._update_from_labels()

        # Update caches used to find frames by frame index
        self._cache = LabelsDataCache(self)

        # Create a variable to store a temporary storage directory
        # used when we unzip
        self.__temp_dir = None

    def _update_from_labels(self, merge: bool = False):
        """Updates top level attributes with data from labeled frames.

        Args:
            merge: If True, then update even if there's already data.

        Returns:
            None.
        """

        # Add any videos that are present in the labels but
        # missing from the video list
        if merge or len(self.videos) == 0:
            # find videos in labeled frames or suggestions
            # that aren't yet in top level videos
            lf_videos = {label.video for label in self.labels}
            suggestion_videos = {sug.video for sug in self.suggestions}
            new_videos = lf_videos.union(suggestion_videos) - set(self.videos)
            # just add the new videos so we don't re-order current list
            if len(new_videos):
                self.videos.extend(list(new_videos))

        # Ditto for skeletons
        if merge or len(self.skeletons) == 0:
            self.skeletons = list(
                set(self.skeletons).union(
                    {
                        instance.skeleton
                        for label in self.labels
                        for instance in label.instances
                    }
                )
            )

        # Ditto for nodes
        if merge or len(self.nodes) == 0:
            self.nodes = list(
                set(self.nodes).union(
                    {node for skeleton in self.skeletons for node in skeleton.nodes}
                )
            )

        # Ditto for tracks, a pattern is emerging here
        if merge or len(self.tracks) == 0:
            # Get tracks from any Instances or PredictedInstances
            other_tracks = {
                instance.track
                for frame in self.labels
                for instance in frame.instances
                if instance.track
            }

            # Add tracks from any PredictedInstance referenced by instance
            # This fixes things when there's a referenced PredictionInstance
            # which is no longer in the frame.
            other_tracks = other_tracks.union(
                {
                    instance.from_predicted.track
                    for frame in self.labels
                    for instance in frame.instances
                    if instance.from_predicted and instance.from_predicted.track
                }
            )

            # Get list of other tracks not already in track list
            new_tracks = list(other_tracks - set(self.tracks))

            # Sort the new tracks by spawned on and then name
            new_tracks.sort(key=lambda t: (t.spawned_on, t.name))

            self.tracks.extend(new_tracks)

    def _update_containers(self, new_label: LabeledFrame):
        """ Ensure that top-level containers are kept updated with new
        instances of objects that come along with new labels. """

        if new_label.video not in self.videos:
            self.videos.append(new_label.video)

        for skeleton in {instance.skeleton for instance in new_label}:
            if skeleton not in self.skeletons:
                self.skeletons.append(skeleton)
                for node in skeleton.nodes:
                    if node not in self.nodes:
                        self.nodes.append(node)

        # Add any new Tracks as well
        for instance in new_label.instances:
            if instance.track and instance.track not in self.tracks:
                self.tracks.append(instance.track)

        # Sort the tracks again
        self.tracks.sort(key=lambda t: (t.spawned_on, t.name))

        # Update cache datastructures
        self._cache.update(new_label)

    def update_cache(self):
        self._cache.update()

    # Below are convenience methods for working with Labels as list.
    # Maybe we should just inherit from list? Maybe this class shouldn't
    # exists since it is just a list really with some class methods. I
    # think more stuff might appear in this class later down the line
    # though.

    @property
    def labels(self):
        """Alias for labeled_frames."""
        return self.labeled_frames

    def __len__(self) -> int:
        """Returns number of labeled frames."""
        return len(self.labeled_frames)

    def index(self, value) -> int:
        """Returns index of labeled frame in list of labeled frames."""
        return self.labeled_frames.index(value)

    def __contains__(self, item) -> bool:
        """
        Checks if object contains the given item.

        Args:
            item: The item to look for within `Labels`.
                This can be :class:`LabeledFrame`,
                :class:`Video`, :class:`Skeleton`,
                :class:`Node`, or (:class:`Video`, frame idx) tuple.

        Returns:
            True if item is found.
        """
        if isinstance(item, LabeledFrame):
            return item in self.labeled_frames
        elif isinstance(item, Video):
            return item in self.videos
        elif isinstance(item, Skeleton):
            return item in self.skeletons
        elif isinstance(item, Node):
            return item in self.nodes
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], Video)
            and isinstance(item[1], int)
        ):
            return self.find_first(*item) is not None

    def __getitem__(self, key) -> List[LabeledFrame]:
        """Returns labeled frames matching key.

        Args:
            key: `Video` or (`Video`, frame index) to match against.

        Raises:
            KeyError: If labeled frame for `Video` or frame index
            cannot be found.

        Returns: A list with the matching labeled frame(s).
        """
        if isinstance(key, int):
            return self.labels.__getitem__(key)

        elif isinstance(key, Video):
            if key not in self.videos:
                raise KeyError("Video not found in labels.")
            return self.find(video=key)

        elif (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], Video)
            and isinstance(key[1], int)
        ):
            if key[0] not in self.videos:
                raise KeyError("Video not found in labels.")

            _hit = self.find_first(video=key[0], frame_idx=key[1])

            if _hit is None:
                raise KeyError(f"No label found for specified video at frame {key[1]}.")

            return _hit

        else:
            raise KeyError("Invalid label indexing arguments.")

    def __setitem__(self, index, value: LabeledFrame):
        """Sets labeled frame at given index."""
        # TODO: Maybe we should remove this method altogether?
        self.labeled_frames.__setitem__(index, value)
        self._update_containers(value)

    def insert(self, index, value: LabeledFrame):
        """Inserts labeled frame at given index."""
        if value in self or (value.video, value.frame_idx) in self:
            return

        self.labeled_frames.insert(index, value)
        self._update_containers(value)

    def append(self, value: LabeledFrame):
        """Adds labeled frame to list of labeled frames."""
        self.insert(len(self) + 1, value)

    def __delitem__(self, key):
        """Removes labeled frame with given index."""
        self.labeled_frames.remove(self.labeled_frames[key])

    def remove(self, value: LabeledFrame):
        """Removes given labeled frame."""
        self.labeled_frames.remove(value)
        self._cache.remove_frame(value)

    def find(
        self,
        video: Video,
        frame_idx: Optional[Union[int, Iterable[int]]] = None,
        return_new: bool = False,
    ) -> List[LabeledFrame]:
        """ Search for labeled frames given video and/or frame index.

        Args:
            video: A :class:`Video` that is associated with the project.
            frame_idx: The frame index (or indices) which we want to
                find in the video. If a range is specified, we'll return
                all frames with indices in that range. If not specific,
                then we'll return all labeled frames for video.
            return_new: Whether to return singleton of new and empty
                :class:`LabeledFrame` if none is found in project.

        Returns:
            List of `LabeledFrame` objects that match the criteria.
            Empty if no matches found, unless return_new is True,
            in which case it contains a new `LabeledFrame` with
            `video` and `frame_index` set.
        """
        null_result = (
            [LabeledFrame(video=video, frame_idx=frame_idx)] if return_new else []
        )

        result = self._cache.find_frames(video, frame_idx)
        return null_result if result is None else result

    def frames(self, video: Video, from_frame_idx: int = -1, reverse=False):
        """
        Iterator over all labeled frames in a video.

        Args:
            video: A :class:`Video` that is associated with the project.
            from_frame_idx: The frame index from which we want to start.
                Defaults to the first frame of video.
            reverse: Whether to iterate over frames in reverse order.

        Yields:
            :class:`LabeledFrame`
        """

        frame_idxs = self._cache.find_fancy_frame_idxs(video, from_frame_idx, reverse)

        # Yield the frames
        for idx in frame_idxs:
            yield self._cache._frame_idx_map[video][idx]

    def find_first(
        self, video: Video, frame_idx: Optional[int] = None
    ) -> Optional[LabeledFrame]:
        """
        Finds the first occurrence of a matching labeled frame.

        Matches on frames for the given video and/or frame index.

        Args:
            video: a `Video` instance that is associated with the
                labeled frames
            frame_idx: an integer specifying the frame index within
                the video

        Returns:
            First `LabeledFrame` that match the criteria
            or None if none were found.
        """

        if video in self.videos:
            for label in self.labels:
                if label.video == video and (
                    frame_idx is None or (label.frame_idx == frame_idx)
                ):
                    return label

    def find_last(
        self, video: Video, frame_idx: Optional[int] = None
    ) -> Optional[LabeledFrame]:
        """
        Finds the last occurrence of a matching labeled frame.

        Matches on frames for the given video and/or frame index.

        Args:
            video: a `Video` instance that is associated with the
                labeled frames
            frame_idx: an integer specifying the frame index within
                the video

        Returns:
            Last `LabeledFrame` that match the criteria
            or None if none were found.
        """

        if video in self.videos:
            for label in reversed(self.labels):
                if label.video == video and (
                    frame_idx is None or (label.frame_idx == frame_idx)
                ):
                    return label

    @property
    def user_labeled_frames(self):
        """
        Returns all labeled frames with user (non-predicted) instances.
        """
        return [lf for lf in self.labeled_frames if lf.has_user_instances]

    def get_labeled_frame_count(self, video: Optional[Video] = None, filter: Text = ""):
        return self._cache.get_frame_count(video, filter)

    # Methods for instances

    def instance_count(self, video: Video, frame_idx: int) -> int:
        """Returns number of instances matching video/frame index."""
        count = 0
        labeled_frame = self.find_first(video, frame_idx)
        if labeled_frame is not None:
            count = len(
                [inst for inst in labeled_frame.instances if type(inst) == Instance]
            )
        return count

    @property
    def all_instances(self):
        """Returns list of all instances."""
        return list(self.instances())

    @property
    def user_instances(self):
        """Returns list of all user (non-predicted) instances."""
        return [inst for inst in self.all_instances if type(inst) == Instance]

    @property
    def predicted_instances(self):
        """Returns list of all user (non-predicted) instances."""
        return [inst for inst in self.all_instances if type(inst) == PredictedInstance]

    def instances(self, video: Video = None, skeleton: Skeleton = None):
        """
        Iterate over instances in the labels, optionally with filters.

        Args:
            video: Only iterate through instances in this video
            skeleton: Only iterate through instances with this skeleton

        Yields:
            Instance: The next labeled instance
        """
        for label in self.labels:
            if video is None or label.video == video:
                for instance in label.instances:
                    if skeleton is None or instance.skeleton == skeleton:
                        yield instance

    def get_template_instance_points(self, skeleton: Skeleton):
        if not hasattr(self, "_template_instance_points"):
            self._template_instance_points = dict()

        # Use cache unless there are a small number of labeled frames so far, or
        # we don't have a cached template instance yet or the skeleton has changed.

        rebuild_template = False
        if len(self.labeled_frames) < 100:
            rebuild_template = True
        elif skeleton not in self._template_instance_points:
            rebuild_template = True
        elif skeleton.nodes != self._template_instance_points[skeleton]["nodes"]:
            rebuild_template = True

        if rebuild_template:
            # Make sure there are some labeled frames
            if self.labeled_frames and any(self.instances()):
                from sleap.info import align

                first_n_instances = itertools.islice(
                    self.instances(skeleton=skeleton), 1000
                )
                template_points = align.get_template_points_array(first_n_instances)
                self._template_instance_points[skeleton] = dict(
                    points=template_points, nodes=skeleton.nodes,
                )
            else:
                # No labeled frames so use force-directed graph layout
                import networkx as nx

                node_positions = nx.spring_layout(G=skeleton.graph, scale=50)

                template_points = np.stack(
                    [
                        node_positions[node]
                        if node in node_positions
                        else np.random.randint(0, 50, size=2)
                        for node in skeleton.nodes
                    ]
                )
                self._template_instance_points[skeleton] = dict(
                    points=template_points, nodes=skeleton.nodes,
                )

        return self._template_instance_points[skeleton]["points"]

    # Methods for tracks

    def get_track_count(self, video: Video) -> int:
        """Returns the number of occupied tracks for a given video."""
        return len(self.get_track_occupancy(video))

    def get_track_occupancy(self, video: Video) -> List:
        """Returns track occupancy list for given video"""
        return self._cache.get_video_track_occupancy(video=video)

    def add_track(self, video: Video, track: Track):
        """Adds track to labels, updating occupancy."""
        self.tracks.append(track)
        self._cache.add_track(video, track)

    def track_set_instance(
        self, frame: LabeledFrame, instance: Instance, new_track: Track
    ):
        """Sets track on given instance, updating occupancy."""
        self.track_swap(
            frame.video,
            new_track,
            instance.track,
            (frame.frame_idx, frame.frame_idx + 1),
        )
        if instance.track is None:
            self._cache.remove_instance(frame, instance)  # FIXME
        instance.track = new_track

    def track_swap(
        self,
        video: Video,
        new_track: Track,
        old_track: Optional[Track],
        frame_range: tuple,
    ):
        """
        Swaps track assignment for instances in two tracks.

        If you need to change the track to or from None, you'll need
        to use :meth:`track_set_instance` for each specific
        instance you want to modify.

        Args:
            video: The :class:`Video` for which we want to swap tracks.
            new_track: A :class:`Track` for which we want to swap
                instances with another track.
            old_track: The other :class:`Track` for swapping.
            frame_range: Tuple of (start, end) frame indexes.
                If you want to swap tracks on a single frame, use
                (frame index, frame index + 1).

        Returns:
            None.
        """

        self._cache.track_swap(video, new_track, old_track, frame_range)

        # Update tracks set on instances

        # Get all instances in old/new tracks
        # Note that this won't match on None track.
        old_track_instances = self.find_track_occupancy(video, old_track, frame_range)
        new_track_instances = self.find_track_occupancy(video, new_track, frame_range)

        # swap new to old tracks on all instances
        for instance in old_track_instances:
            instance.track = new_track
        # old_track can be `Track` or int
        # If int, it's index in instance list which we'll use as a pseudo-track,
        # but we won't set instances currently on new_track to old_track.
        if type(old_track) == Track:
            for instance in new_track_instances:
                instance.track = old_track

    def remove_instance(
        self, frame: LabeledFrame, instance: Instance, in_transaction: bool = False
    ):
        """Removes instance from frame, updating track occupancy."""
        frame.instances.remove(instance)
        if not in_transaction:
            self._cache.remove_instance(frame, instance)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Adds instance to frame, updating track occupancy."""
        # Ensure that there isn't already an Instance with this track
        tracks_in_frame = [
            inst.track
            for inst in frame
            if type(inst) == Instance and inst.track is not None
        ]
        if instance.track in tracks_in_frame:
            instance.track = None

        frame.instances.append(instance)

        self._cache.add_instance(frame, instance)

    def find_track_occupancy(
        self, video: Video, track: Union[Track, int], frame_range=None
    ) -> List[Instance]:
        """Get instances for a given video, track, and range of frames.

        Args:
            video: the `Video`
            track: the `Track` or int ("pseudo-track" index to instance list)
            frame_range (optional):
                If specified, only return instances on frames in range.
                If None, return all instances for given track.
        Returns:
            List of :class:`Instance` objects.
        """

        frame_range = range(*frame_range) if type(frame_range) == tuple else frame_range

        def does_track_match(inst, tr, labeled_frame):
            match = False
            if type(tr) == Track and inst.track is tr:
                match = True
            elif (
                type(tr) == int
                and labeled_frame.instances.index(inst) == tr
                and inst.track is None
            ):
                match = True
            return match

        track_frame_inst = [
            instance
            for lf in self.find(video)
            for instance in lf.instances
            if does_track_match(instance, track, lf)
            and (frame_range is None or lf.frame_idx in frame_range)
        ]
        return track_frame_inst

    # Methods for suggestions

    def get_video_suggestions(self, video: Video) -> List[int]:
        """
        Returns the list of suggested frames for the specified video
        or suggestions for all videos (if no video specified).
        """
        return [item.frame_idx for item in self.suggestions if item.video == video]

    def get_suggestions(self) -> list:
        """Return all suggestions as a list of SuggestionFrame items."""
        return self.suggestions

    def find_suggestion(self, video, frame_idx):
        """Find SuggestionFrame by video and frame index."""
        matches = [
            item
            for item in self.suggestions
            if item.video == video and item.frame_idx == frame_idx
        ]

        if matches:
            return matches[0]

        return None

    def get_next_suggestion(self, video, frame_idx, seek_direction=1) -> list:
        """Returns a (video, frame_idx) tuple seeking from given frame."""
        # make sure we have valid seek_direction
        if seek_direction not in (-1, 1):
            raise ValueError("seek_direction should be -1 or 1.")
        # make sure the video belongs to this Labels object
        if video not in self.videos:
            return None

        all_suggestions = self.get_suggestions()

        # If we're currently on a suggestion, then follow order of list
        match = self.find_suggestion(video, frame_idx)
        if match is not None:
            suggestion_idx = all_suggestions.index(match)
            new_idx = (suggestion_idx + seek_direction) % len(all_suggestions)
            return all_suggestions[new_idx]

        # Otherwise, find the prev/next suggestion sorted by frame order...

        # Look for next (or previous) suggestion in current video.
        if seek_direction == 1:
            frame_suggestion = min(
                (i for i in self.get_video_suggestions(video) if i > frame_idx),
                default=None,
            )
        else:
            frame_suggestion = max(
                (i for i in self.get_video_suggestions(video) if i < frame_idx),
                default=None,
            )
        if frame_suggestion is not None:
            return self.find_suggestion(video, frame_suggestion)

        # If we didn't find suggestion in current video, then we want earliest
        # frame in next video with suggestions.
        next_video_idx = (self.videos.index(video) + seek_direction) % len(self.videos)
        video = self.videos[next_video_idx]
        if seek_direction == 1:
            frame_suggestion = min(
                (i for i in self.get_video_suggestions(video)), default=None
            )
        else:
            frame_suggestion = max(
                (i for i in self.get_video_suggestions(video)), default=None
            )
        return self.find_suggestion(video, frame_suggestion)

    def set_suggestions(self, suggestions: List["SuggestionFrame"]):
        """Sets the suggested frames."""
        self.suggestions = suggestions

    def delete_suggestions(self, video):
        """Deletes suggestions for specified video."""
        self.suggestions = [item for item in self.suggestions if item.video != video]

    # Methods for videos

    def add_video(self, video: Video):
        """ Add a video to the labels if it is not already in it.

        Video instances are added automatically when adding labeled frames,
        but this function allows for adding videos to the labels before any
        labeled frames are added.

        Args:
            video: `Video` instance

        """
        if video not in self.videos:
            self.videos.append(video)

    def remove_video(self, video: Video):
        """ Removes a video from the labels and ALL associated labeled frames.

        Args:
            video: `Video` instance to be removed
        """
        if video not in self.videos:
            raise KeyError("Video is not in labels.")

        # Delete all associated labeled frames
        for label in reversed(self.labeled_frames):
            if label.video == video:
                self.labeled_frames.remove(label)

        # Delete data that's indexed by video
        self.delete_suggestions(video)
        if video in self.negative_anchors:
            del self.negative_anchors[video]

        # Delete video
        self.videos.remove(video)
        self._cache.remove_video(video)

    # Methods for saving/loading

    @classmethod
    def from_json(cls, *args, **kwargs):
        from sleap.io.format.labels_json import LabelsJsonAdaptor

        return LabelsJsonAdaptor.from_json_data(*args, **kwargs)

    def extend_from(
        self, new_frames: Union["Labels", List[LabeledFrame]], unify: bool = False
    ):
        """
        Merge data from another `Labels` object or `LabeledFrame` list.

        Arg:
            new_frames: the object from which to copy data
            unify: whether to replace objects in new frames with
                corresponding objects from current `Labels` data

        Returns:
            bool, True if we added frames, False otherwise
        """
        # allow either Labels or list of LabeledFrames
        if isinstance(new_frames, Labels):
            new_frames = new_frames.labeled_frames

        # return if this isn't non-empty list of labeled frames
        if not isinstance(new_frames, list) or len(new_frames) == 0:
            return False
        if not isinstance(new_frames[0], LabeledFrame):
            return False

        # If unify, we want to replace objects in the frames with
        # corresponding objects from the current labels.
        # We do this by deserializing/serializing with match_to.
        if unify:
            new_json = Labels(labeled_frames=new_frames).to_dict()
            new_labels = Labels.from_json(new_json, match_to=self)
            new_frames = new_labels.labeled_frames

        # copy the labeled frames
        self.labeled_frames.extend(new_frames)

        # merge labeled frames for the same video/frame idx
        self.merge_matching_frames()

        # update top level videos/nodes/skeletons/tracks
        self._update_from_labels(merge=True)
        self._cache.update()

        return True

    @classmethod
    def complex_merge_between(
        cls, base_labels: "Labels", new_labels: "Labels", unify: bool = True
    ) -> tuple:
        """
        Merge frames and other data from one dataset into another.

        Anything that can be merged cleanly is merged into base_labels.

        Frames conflict just in case each labels object has a matching
        frame (same video and frame idx) with instances not in other.

        Frames can be merged cleanly if:

        * the frame is in only one of the labels, or
        * the frame is in both labels, but all instances perfectly match
          (which means they are redundant), or
        * the frame is in both labels, maybe there are some redundant
          instances, but only one version of the frame has additional
          instances not in the other.

        Args:
            base_labels: the `Labels` that we're merging into
            new_labels: the `Labels` that we're merging from
            unify: whether to replace objects (e.g., `Video`) in
                new_labels with *matching* objects from base

        Returns:
            tuple of three items:

            * Dictionary, keys are :class:`Video`, values are
                dictionary in which keys are frame index (int)
                and value is list of :class:`Instance` objects
            * list of conflicting :class:`Instance` objects from base
            * list of conflicting :class:`Instance` objects from new

        """
        # If unify, we want to replace objects in the frames with
        # corresponding objects from the current labels.
        # We do this by deserializing/serializing with match_to.
        if unify:
            new_json = new_labels.to_dict()
            new_labels = cls.from_json(new_json, match_to=base_labels)

        # Merge anything that can be merged cleanly and get conflicts
        merged, extra_base, extra_new = LabeledFrame.complex_merge_between(
            base_labels=base_labels, new_frames=new_labels.labeled_frames
        )

        # For clean merge, finish merge now by cleaning up base object
        if not extra_base and not extra_new:
            # Add any new videos (etc) into top level lists in base
            base_labels._update_from_labels(merge=True)
            # Update caches
            base_labels.update_cache()

        # Merge suggestions and negative anchors
        base_labels.suggestions.extend(new_labels.suggestions)
        cls.merge_container_dicts(
            base_labels.negative_anchors, new_labels.negative_anchors
        )

        return merged, extra_base, extra_new

    @staticmethod
    def finish_complex_merge(
        base_labels: "Labels", resolved_frames: List[LabeledFrame]
    ):
        """
        Finish conflicted merge from complex_merge_between.

        Args:
            base_labels: the `Labels` that we're merging into
            resolved_frames: the list of frames to add into base_labels
        Returns:
            None.
        """
        # Add all the resolved frames to base
        base_labels.labeled_frames.extend(resolved_frames)

        # Combine instances when there are two LabeledFrames for same
        # video and frame index
        base_labels.merge_matching_frames()

        # Add any new videos (etc) into top level lists in base
        base_labels._update_from_labels(merge=True)
        # Update caches
        base_labels.update_cache()

    @staticmethod
    def merge_container_dicts(dict_a: Dict, dict_b: Dict) -> Dict:
        """Merge data from dict_b into dict_a."""
        for key in dict_b.keys():
            if key in dict_a:
                dict_a[key].extend(dict_b[key])
                uniquify(dict_a[key])
            else:
                dict_a[key] = dict_b[key]

    def merge_matching_frames(self, video: Optional[Video] = None):
        """
        Merge `LabeledFrame` objects that are for the same video frame.

        Args:
            video: combine for this video; if None, do all videos
        Returns:
            None
        """
        if video is None:
            for vid in {lf.video for lf in self.labeled_frames}:
                self.merge_matching_frames(video=vid)
        else:
            self.labeled_frames = LabeledFrame.merge_frames(
                self.labeled_frames, video=video
            )

    def to_dict(self, skip_labels: bool = False):
        """
        Serialize all labels in the underling list of LabeledFrames to a
        dict structure. This function returns a nested dict structure
        composed entirely of primitive python types. It is used to create
        JSON and HDF5 serialized datasets.

        Args:
            skip_labels: If True, skip labels serialization and just do the
            metadata.

        Returns:
            A dict containing the followings top level keys:
            * version - The version of the dict/json serialization format.
            * skeletons - The skeletons associated with these underlying
              instances.
            * nodes - The nodes that the skeletons represent.
            * videos - The videos that that the instances occur on.
            * labels - The labeled frames
            * tracks - The tracks associated with each instance.
            * suggestions - The suggested frames.
            * negative_anchors - The negative training sample anchors.
        """

        # FIXME: Update list of nodes
        # We shouldn't have to do this here, but for some reason we're missing nodes
        # which are in the skeleton but don't have points (in the first instance?).
        self.nodes = list(
            set(self.nodes).union(
                {node for skeleton in self.skeletons for node in skeleton.nodes}
            )
        )

        # Register some unstructure hooks since we don't want complete deserialization
        # of video and skeleton objects present in the labels. We will serialize these
        # as references to the above constructed lists to limit redundant data in the
        # json
        label_cattr = make_instance_cattr()
        label_cattr.register_unstructure_hook(
            Skeleton, lambda x: str(self.skeletons.index(x))
        )
        label_cattr.register_unstructure_hook(
            Video, lambda x: str(self.videos.index(x))
        )
        label_cattr.register_unstructure_hook(Node, lambda x: str(self.nodes.index(x)))
        label_cattr.register_unstructure_hook(
            Track, lambda x: str(self.tracks.index(x))
        )

        # Make a converter for the top level skeletons list.
        idx_to_node = {i: self.nodes[i] for i in range(len(self.nodes))}

        skeleton_cattr = Skeleton.make_cattr(idx_to_node)

        # Make attr for tracks so that we save as tuples rather than dicts;
        # this can save a lot of space when there are lots of tracks.
        track_cattr = cattr.Converter(unstruct_strat=cattr.UnstructureStrategy.AS_TUPLE)

        # Serialize the skeletons, videos, and labels
        dicts = {
            "version": LABELS_JSON_FILE_VERSION,
            "skeletons": skeleton_cattr.unstructure(self.skeletons),
            "nodes": cattr.unstructure(self.nodes),
            "videos": Video.cattr().unstructure(self.videos),
            "tracks": track_cattr.unstructure(self.tracks),
            "suggestions": label_cattr.unstructure(self.suggestions),
            "negative_anchors": label_cattr.unstructure(self.negative_anchors),
            "provenance": label_cattr.unstructure(self.provenance),
        }

        if not skip_labels:
            dicts["labels"] = label_cattr.unstructure(self.labeled_frames)

        return dicts

    def to_json(self):
        """
        Serialize all labels in the underling list of LabeledFrame(s) to a
        JSON structured string.

        Returns:
            The JSON representation of the string.
        """

        # Unstructure the data into dicts and dump to JSON.
        return json_dumps(self.to_dict())

    @classmethod
    def load_file(
        cls,
        filename: str,
        video_search: Union[Callable, List[Text], None] = None,
        *args,
        **kwargs,
    ):
        """Load file, detecting format from filename."""
        from .format import read

        return read(
            filename, for_object="labels", video_search=video_search, *args, **kwargs
        )

    @classmethod
    def save_file(
        cls, labels: "Labels", filename: str, default_suffix: str = "", *args, **kwargs
    ):
        """Save file, detecting format from filename.

        Args:
            labels: The dataset to save.
            filename: Path where we'll save it. We attempt to detect format
                from the suffix (e.g., ".json").
            default_suffix: If we can't detect valid suffix on filename,
                we can add default suffix to filename (and use corresponding
                format). Doesn't need to have "." before file extension.

        Raises:
            ValueError: If cannot detect valid filetype.

        Returns:
            None.
        """
        # Convert to full (absolute) path
        filename = os.path.abspath(filename)

        # Make sure that all directories for path exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        from .format import write

        write(filename, labels, *args, **kwargs)

    @classmethod
    def load_json(cls, filename: str, *args, **kwargs) -> "Labels":
        from .format import read

        return read(filename, for_object="labels", as_format="json", *args, **kwargs)

    @classmethod
    def save_json(cls, labels: "Labels", filename: str, *args, **kwargs):
        from .format import write

        write(filename, labels, as_format="json", *args, **kwargs)

    @classmethod
    def load_hdf5(cls, filename, *args, **kwargs):
        from .format import read

        return read(filename, for_object="labels", as_format="hdf5_v1", *args, **kwargs)

    @classmethod
    def save_hdf5(cls, labels, filename, *args, **kwargs):
        from .format import write

        write(filename, labels, as_format="hdf5_v1", *args, **kwargs)

    @classmethod
    def load_leap_matlab(cls, filename, *args, **kwargs):
        from .format import read

        return read(filename, for_object="labels", as_format="leap", *args, **kwargs)

    @classmethod
    def load_deeplabcut(cls, filename: str) -> "Labels":
        from .format import read

        return read(filename, for_object="labels", as_format="deeplabcut")

    @classmethod
    def load_coco(
        cls, filename: str, img_dir: str, use_missing_gui: bool = False,
    ) -> "Labels":
        from sleap.io.format.coco import LabelsCocoAdaptor
        from sleap.io.format.filehandle import FileHandle

        return LabelsCocoAdaptor.read(FileHandle(filename), img_dir, use_missing_gui)

    @classmethod
    def from_deepposekit(
        cls, filename: str, video_path: str, skeleton_path: str
    ) -> "Labels":
        from sleap.io.format.deepposekit import LabelsDeepPoseKitAdaptor
        from sleap.io.format.filehandle import FileHandle

        return LabelsDeepPoseKitAdaptor.read(
            FileHandle(filename), video_path, skeleton_path
        )

    def save_frame_data_imgstore(
        self, output_dir: str = "./", format: str = "png", all_labels: bool = False
    ):
        """
        Write images for labeled frames from all videos to imgstore datasets.

        This only writes frames that have been labeled. Videos without
        any labeled frames will be included as empty imgstores.

        Args:
            output_dir: Path to directory which will contain imgstores.
            format: The image format to use for the data.
                Use "png" for lossless, "jpg" for lossy.
                Other imgstore formats will probably work as well but
                have not been tested.
            all_labels: Include any labeled frames, not just the frames
                we'll use for training (i.e., those with `Instance` objects ).

        Returns:
            A list of :class:`ImgStoreVideo` objects with the stored
            frames.
        """
        # For each label
        imgstore_vids = []
        for v_idx, v in enumerate(self.videos):
            frame_nums = [
                lf.frame_idx
                for lf in self.labeled_frames
                if v == lf.video and (all_labels or lf.has_user_instances)
            ]

            # Join with "/" instead of os.path.join() since we want
            # path to work on Windows and Posix systems
            frames_filename = output_dir + f"/frame_data_vid{v_idx}"
            vid = v.to_imgstore(
                path=frames_filename, frame_numbers=frame_nums, format=format
            )

            # Close the video for now
            vid.close()

            imgstore_vids.append(vid)

        return imgstore_vids

    def save_frame_data_hdf5(
        self, output_path: str, format: str = "png", all_labels: bool = False
    ):
        """
        Write images for labeled frames from all videos to hdf5 file.

        Note that this will make an HDF5 video, not an HDF5 labels dataset.

        Args:
            output_path: Path to HDF5 file.
            format: The image format to use for the data. Defaults to png.
            all_labels: Include any labeled frames, not just the frames
                we'll use for training (i.e., those with Instances).

        Returns:
            A list of :class:`HDF5Video` objects with the stored frames.
        """
        new_vids = []
        for v_idx, v in enumerate(self.videos):
            frame_nums = [
                lf.frame_idx
                for lf in self.labeled_frames
                if v == lf.video and (all_labels or lf.has_user_instances)
            ]

            vid = v.to_hdf5(
                path=output_path,
                dataset=f"video{v_idx}",
                format=format,
                frame_numbers=frame_nums,
            )
            vid.close()
            new_vids.append(vid)

        return new_vids

    @classmethod
    def make_gui_video_callback(cls, search_paths: Optional[List] = None) -> Callable:
        return cls.make_video_callback(search_paths=search_paths, use_gui=True)

    @classmethod
    def make_video_callback(
        cls, search_paths: Optional[List] = None, use_gui: bool = False
    ) -> Callable:
        """
        Create a callback for finding missing videos.

        The callback can be used while loading a saved project and
        allows the user to find videos which have been moved (or have
        paths from a different system).

        The callback function returns True to signal "abort".

        Args:
            search_paths: If specified, this is a list of paths where
                we'll automatically try to find the missing videos.

        Returns:
            The callback function.
        """
        search_paths = search_paths or []

        def video_callback(video_list, new_paths=search_paths):
            filenames = [item["backend"]["filename"] for item in video_list]
            missing = pathutils.list_file_missing(filenames)

            # Try changing the prefix using saved patterns
            if sum(missing):
                pathutils.fix_paths_with_saved_prefix(filenames, missing)

            # Check for file in search_path directories
            if sum(missing) and new_paths:
                for i, filename in enumerate(filenames):
                    fixed_path = find_path_using_paths(filename, new_paths)
                    if fixed_path != filename:
                        filenames[i] = fixed_path
                        missing[i] = False

            if use_gui:
                # If there are still missing paths, prompt user
                if sum(missing):
                    # If we are using dummy for any video not found by user
                    # then don't require user to find everything.
                    allow_incomplete = USE_DUMMY_FOR_MISSING_VIDEOS

                    okay = MissingFilesDialog(
                        filenames, missing, allow_incomplete=allow_incomplete
                    ).exec_()

                    if not okay:
                        return True  # True for stop

            if not use_gui and sum(missing):
                # If we got the same number of paths as there are videos
                if len(filenames) == len(new_paths):
                    # and the file extensions match
                    exts_match = all(
                        (
                            old.split(".")[-1] == new.split(".")[-1]
                            for old, new in zip(filenames, new_paths)
                        )
                    )

                    if exts_match:
                        # then the search paths should be a list of all the
                        # video paths, so we can get the new path for the missing
                        # old path.
                        for i, filename in enumerate(filenames):
                            if missing[i]:
                                filenames[i] = new_paths[i]

            # Replace the video filenames with changes by user
            for i, item in enumerate(video_list):
                item["backend"]["filename"] = filenames[i]

            if USE_DUMMY_FOR_MISSING_VIDEOS and sum(missing):
                # Replace any video still missing with "dummy" video
                for is_missing, item in zip(missing, video_list):
                    from sleap.io.video import DummyVideo

                    vid = DummyVideo(filename=item["backend"]["filename"])
                    item["backend"] = cattr.unstructure(vid)

        return video_callback

    def export_training_data(self, save_path: Text):
        """Exports a set of images and points for training with minimal metadata.

        Args:
            save_path: Path to HDF5 that training data will be saved to.

        Notes:
            The exported HDF5 file will contain no SLEAP-specific metadata or
            dependencies for serialization. These files cannot be read back in for
            labeling, but are useful when training on environments where it is hard to
            install complex dependencies.
        """

        # Skeleton
        node_names = np.string_(self.skeletons[0].node_names)
        edge_inds = np.array(self.skeletons[0].edge_inds)

        # Videos metadata
        video_paths = []
        video_datasets = []
        video_shapes = []
        video_image_data_format = []
        for video in self.videos:
            video_paths.append(video.backend.filename)
            if hasattr(video.backend, "dataset"):
                video_datasets.append(video.backend.dataset)
            else:
                video_datasets.append("")
            video_shapes.append(video.shape)
            video_image_data_format.append(video.backend.input_format)

        video_paths = np.string_(video_paths)
        video_datasets = np.string_(video_datasets)
        video_shapes = np.array(video_shapes)
        video_image_data_format = np.string_(video_image_data_format)

        # Main labeling data
        video_inds = []
        frame_inds = []
        imgs = []
        peaks = []
        peak_samples = []
        peak_instances = []
        peak_channels = []
        peak_tracks = []

        # Main labeling data.
        labeled_frames_with_instances = [
            lf for lf in self.labeled_frames if lf.has_user_instances
        ]

        for sample_ind, lf in enumerate(labeled_frames_with_instances):

            # Video index into the videos metadata
            video_ind = self.videos.index(lf.video)

            # Frame index into the original images array
            frame_ind = lf.frame_idx
            if hasattr(lf.video.backend, "_HDF5Video__original_to_current_frame_idx"):
                frame_ind = lf.video.backend._HDF5Video__original_to_current_frame_idx[
                    lf.frame_idx
                ]

            # Actual image data
            img = lf.image

            frame_peaks = []
            frame_peak_samples = []
            frame_peak_instances = []
            frame_peak_channels = []
            frame_peak_tracks = []
            for instance_ind, instance in enumerate(lf.user_instances):
                instance_peaks = instance.points_array.astype("float32")
                frame_peaks.append(instance_peaks)
                frame_peak_samples.append(np.full((len(instance_peaks),), sample_ind))
                frame_peak_instances.append(
                    np.full((len(instance_peaks),), instance_ind)
                )
                frame_peak_channels.append(np.arange(len(instance_peaks)))
                track_ind = np.nan
                if instance.track is not None:
                    track_ind = self.tracks.index(instance.track)
                frame_peak_tracks.append(np.full((len(instance_peaks),), track_ind))

            # Concatenate into (n_peaks, 2) -> x, y = frame_peaks[i]
            frame_peaks = np.concatenate(frame_peaks, axis=0)

            # Concatenate metadata
            frame_peak_samples = np.concatenate(frame_peak_samples)
            frame_peak_instances = np.concatenate(frame_peak_instances)
            frame_peak_channels = np.concatenate(frame_peak_channels)
            frame_peak_tracks = np.concatenate(frame_peak_tracks)

            video_inds.append(video_ind)
            frame_inds.append(frame_ind)
            imgs.append(img)
            peaks.append(frame_peaks)
            peak_samples.append(frame_peak_samples)
            peak_instances.append(frame_peak_instances)
            peak_channels.append(frame_peak_channels)
            peak_tracks.append(peak_tracks)

        video_inds = np.array(video_inds)
        frame_inds = np.array(frame_inds)
        imgs = np.concatenate(imgs, axis=0)
        peaks = np.concatenate(peaks, axis=0)
        peak_samples = np.concatenate(peak_samples, axis=0).astype("int32")
        peak_instances = np.concatenate(peak_instances, axis=0).astype("int32")
        peak_channels = np.concatenate(peak_channels, axis=0).astype("int32")
        peak_tracks = np.concatenate(peak_channels, axis=0)

        with h5.File(save_path, "w") as f:
            f.create_dataset("skeleton/node_names", data=node_names)
            f.create_dataset("skeleton/n_nodes", data=len(node_names))
            f.create_dataset("skeleton/edges", data=edge_inds)

            f.create_dataset("videos/filepath", data=video_paths)
            f.create_dataset("videos/dataset", data=video_datasets)
            f.create_dataset("videos/shape", data=video_shapes)
            f.create_dataset("videos/image_data_format", data=video_image_data_format)

            f.create_dataset(
                "imgs",
                data=imgs,
                chunks=(1,) + imgs.shape[1:],
                compression="gzip",
                compression_opts=1,
            )
            f.create_dataset("peaks/xy", data=peaks)
            f.create_dataset("peaks/sample", data=peak_samples)
            f.create_dataset("peaks/instance", data=peak_instances)
            f.create_dataset("peaks/channel", data=peak_channels)
            f.create_dataset("peaks/track", data=peak_tracks)

    def generate_training_data(
        self,
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[np.ndarray]]:
        """Generates images and points for training.

        Returns:
            A tuple of (imgs, points).

            imgs: Array of shape (n_samples, height, width, channels) containing the
            image data for all frames with user labels. If frames are of variable size,
            imgs is a list of length n_samples with elements of shape
            (height, width, channels).

            points: List of length n_samples with elements of shape
            (n_instances, n_nodes, 2), containing all user labeled instances in the
            frame, with NaN-padded xy coordinates for each visible body part.
        """

        imgs = []
        points = []

        for lf in self.labeled_frames:
            if not lf.has_user_instances:
                continue

            imgs.append(lf.image)
            points.append(
                np.stack([inst.points_array for inst in lf.user_instances], axis=0)
            )

        # Try to stack all images into a single 4D array.
        first_shape = imgs[0].shape
        can_stack = all([img.shape == first_shape for img in imgs])
        if can_stack:
            imgs = np.stack(imgs, axis=0)

        return imgs, points


def find_path_using_paths(missing_path, search_paths):

    # Get basename (filename with directories) using current os path format
    current_basename = os.path.basename(missing_path)

    # Handle unix, windows, or mixed paths
    if current_basename.find("/") > -1:
        current_basename = current_basename.split("/")[-1]
    if current_basename.find("\\") > -1:
        current_basename = current_basename.split("\\")[-1]

    # Look for file with that name in each of the search path directories
    for search_path in search_paths:

        if os.path.isfile(search_path):
            path_dir = os.path.dirname(search_path)
        else:
            path_dir = search_path

        check_path = os.path.join(path_dir, current_basename)
        if os.path.exists(check_path):
            return check_path

    return missing_path

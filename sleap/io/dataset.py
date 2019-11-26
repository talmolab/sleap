"""
A SLEAP dataset collects labeled video frames.

This contains labeled frame data (user annotations and/or predictions),
together with all the other data that is saved for a SLEAP project
(videos, skeletons, negative training sample anchors, etc.).
"""

import os
import re
import zipfile
import atexit

import attr
import cattr
import shutil
import tempfile
import numpy as np
import scipy.io as sio
import h5py as h5

from collections import MutableSequence
from typing import Callable, List, Union, Dict, Optional, Tuple, Text

try:
    from typing import ForwardRef
except:
    from typing import _ForwardRef as ForwardRef

import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import (
    Instance,
    Point,
    LabeledFrame,
    Track,
    PredictedPoint,
    PredictedInstance,
    make_instance_cattr,
    PointArray,
    PredictedPointArray,
)

from sleap.io import pathutils
from sleap.io.legacy import load_labels_json_old
from sleap.io.video import Video
from sleap.gui.suggestions import SuggestionFrame
from sleap.gui.missingfiles import MissingFilesDialog
from sleap.rangelist import RangeList
from sleap.util import uniquify, weak_filename_match, json_dumps, json_loads


"""
The version number to put in the Labels JSON format.
"""
LABELS_JSON_FILE_VERSION = "2.0.0"


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
        self._build_lookup_caches()

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
            # find videos in labeled frames that aren't yet in top level videos
            new_videos = {label.video for label in self.labels} - set(self.videos)
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
        if new_label.video not in self._lf_by_video:
            self._lf_by_video[new_label.video] = []
        if new_label.video not in self._frame_idx_map:
            self._frame_idx_map[new_label.video] = dict()
        self._lf_by_video[new_label.video].append(new_label)
        self._frame_idx_map[new_label.video][new_label.frame_idx] = new_label

    def _build_lookup_caches(self):
        """Builds (or rebuilds) various caches."""
        # Data structures for caching
        self._lf_by_video = dict()
        self._frame_idx_map = dict()
        self._track_occupancy = dict()
        for video in self.videos:
            self._lf_by_video[video] = [lf for lf in self.labels if lf.video == video]
            self._frame_idx_map[video] = {
                lf.frame_idx: lf for lf in self._lf_by_video[video]
            }
            self._track_occupancy[video] = self._make_track_occupany(video)

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
        self._lf_by_video[value.video].remove(value)
        del self._frame_idx_map[value.video][value.frame_idx]

    def find(
        self,
        video: Video,
        frame_idx: Optional[Union[int, range]] = None,
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

        if frame_idx is not None:
            if video not in self._frame_idx_map:
                return null_result

            if type(frame_idx) == range:
                return [
                    self._frame_idx_map[video][idx]
                    for idx in frame_idx
                    if idx in self._frame_idx_map[video]
                ]

            if frame_idx not in self._frame_idx_map[video]:
                return null_result

            return [self._frame_idx_map[video][frame_idx]]
        else:
            if video not in self._lf_by_video:
                return null_result
            return self._lf_by_video[video]

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

        # Yield the frames
        for idx in frame_idxs:
            yield self._frame_idx_map[video][idx]

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

    def get_video_user_labeled_frames(self, video: Video) -> List[LabeledFrame]:
        """
        Returns labeled frames for given video with user instances.
        """
        return [
            lf
            for lf in self.labeled_frames
            if lf.has_user_instances and lf.video == video
        ]

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

    # Methods for tracks

    def get_track_occupany(self, video: Video) -> List:
        """Returns track occupancy list for given video"""
        try:
            return self._track_occupancy[video]
        except:
            return []

    def add_track(self, video: Video, track: Track):
        """Adds track to labels, updating occupancy."""
        self.tracks.append(track)
        self._track_occupancy[video][track] = RangeList()

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
            self._track_remove_instance(frame, instance)
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
        # Get ranges in track occupancy cache
        _, within_old, _ = self._get_track_occupany(video, old_track).cut_range(
            frame_range
        )
        _, within_new, _ = self._get_track_occupany(video, new_track).cut_range(
            frame_range
        )

        if old_track is not None:
            # Instances that didn't already have track can't be handled here.
            # See track_set_instance for this case.
            self._track_occupancy[video][old_track].remove(frame_range)
        self._track_occupancy[video][new_track].remove(frame_range)
        self._track_occupancy[video][old_track].insert_list(within_new)
        self._track_occupancy[video][new_track].insert_list(within_old)

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

    def _track_remove_instance(self, frame: LabeledFrame, instance: Instance):
        """Manipulates track occupancy cache."""
        if instance.track not in self._track_occupancy[frame.video]:
            return

        # If this is only instance in track in frame, then remove frame from track.
        if len(frame.find(track=instance.track)) == 1:
            self._track_occupancy[frame.video][instance.track].remove(
                (frame.frame_idx, frame.frame_idx + 1)
            )

    def remove_instance(self, frame: LabeledFrame, instance: Instance):
        """Removes instance from frame, updating track occupancy."""
        self._track_remove_instance(frame, instance)
        frame.instances.remove(instance)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Adds instance to frame, updating track occupancy."""
        if frame.video not in self._track_occupancy:
            self._track_occupancy[frame.video] = dict()

        # Ensure that there isn't already an Instance with this track
        tracks_in_frame = [
            inst.track
            for inst in frame
            if type(inst) == Instance and inst.track is not None
        ]
        if instance.track in tracks_in_frame:
            instance.track = None

        # Add track in its not already present in labels
        if instance.track not in self._track_occupancy[frame.video]:
            self._track_occupancy[frame.video][instance.track] = RangeList()

        self._track_occupancy[frame.video][instance.track].insert(
            (frame.frame_idx, frame.frame_idx + 1)
        )
        frame.instances.append(instance)

    def _make_track_occupany(self, video: Video) -> Dict[Video, RangeList]:
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

    def _get_track_occupany(self, video: Video, track: Track) -> RangeList:
        """
        Accessor for track occupancy cache that adds video/track as needed.
        """
        if video not in self._track_occupancy:
            self._track_occupancy[video] = dict()
        if track not in self._track_occupancy[video]:
            self._track_occupancy[video][track] = RangeList()
        return self._track_occupancy[video][track]

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

    def get_video_suggestions(self, video: Video) -> list:
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

        # Remove from caches
        if video in self._lf_by_video:
            del self._lf_by_video[video]
        if video in self._frame_idx_map:
            del self._frame_idx_map[video]

    # Methods for negative anchors

    def add_negative_anchor(self, video: Video, frame_idx: int, where: tuple):
        """Adds a location for a negative training sample.

        Args:
            video: the `Video` for this negative sample
            frame_idx: frame index
            where: (x, y)
        """
        if video not in self.negative_anchors:
            self.negative_anchors[video] = []
        self.negative_anchors[video].append((frame_idx, *where))

    def remove_negative_anchors(self, video: Video, frame_idx: int):
        """Removes negative training samples for given video and frame.

        Args:
            video: the `Video` for which we're removing negative samples
            frame_idx: frame index
        Returns:
            None
        """
        if video not in self.negative_anchors:
            return

        anchors = [
            (idx, x, y)
            for idx, x, y in self.negative_anchors[video]
            if idx != frame_idx
        ]
        self.negative_anchors[video] = anchors

    # Methods for saving/loading

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
        self._build_lookup_caches()

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
            base_labels._build_lookup_caches()

        # Merge suggestions and negative anchors
        base_labels.suggestions.extend(new_labels.suggestions)
        cls.merge_container_dicts(
            base_labels.negative_anchors, new_labels.negative_anchors
        )

        return merged, extra_base, extra_new

    #     @classmethod
    #     def merge_predictions_by_score(cls, extra_base: List[LabeledFrame], extra_new: List[LabeledFrame]):
    #         """
    #         Remove all predictions from input lists, return list with only
    #         the merged predictions.
    #
    #         Args:
    #             extra_base: list of `LabeledFrame` objects
    #             extra_new: list of `LabeledFrame` objects
    #                 Conflicting frames should have same index in both lists.
    #         Returns:
    #             list of `LabeledFrame` objects with merged predictions
    #         """
    #         pass

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
        base_labels._build_lookup_caches()

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

    @staticmethod
    def save_json(
        labels: "Labels",
        filename: str,
        compress: bool = False,
        save_frame_data: bool = False,
        frame_data_format: str = "png",
    ):
        """
        Save a Labels instance to a JSON format.

        Args:
            labels: The labels dataset to save.
            filename: The filename to save the data to.
            compress: Whether the data be zip compressed or not? If True,
                the JSON will be compressed using Python's shutil.make_archive
                command into a PKZIP zip file. If compress is True then
                filename will have a .zip appended to it.
            save_frame_data: Whether to save the image data for each frame.
                For each video in the dataset, all frames that have labels
                will be stored as an imgstore dataset.
                If save_frame_data is True then compress will be forced to True
                since the archive must contain both the JSON data and image
                data stored in ImgStores.
            frame_data_format: If save_frame_data is True, then this argument
                is used to set the data format to use when writing frame
                data to ImgStore objects. Supported formats should be:

                 * 'pgm',
                 * 'bmp',
                 * 'ppm',
                 * 'tif',
                 * 'png',
                 * 'jpg',
                 * 'npy',
                 * 'mjpeg/avi',
                 * 'h264/mkv',
                 * 'avc1/mp4'

                 Note: 'h264/mkv' and 'avc1/mp4' require separate installation
                 of these codecs on your system. They are excluded from SLEAP
                 because of their GPL license.

        Returns:
            None
        """

        # Lets make a temporary directory to store the image frame data or pre-compressed json
        # in case we need it.
        with tempfile.TemporaryDirectory() as tmp_dir:

            # If we are saving frame data along with the datasets. We will replace videos with
            # new video object that represent video data from just the labeled frames.
            if save_frame_data:

                # Create a set of new Video objects with imgstore backends. One for each
                # of the videos. We will only include the labeled frames though. We will
                # then replace each video with this new video
                new_videos = labels.save_frame_data_imgstore(
                    output_dir=tmp_dir, format=frame_data_format
                )

                # Make video paths relative
                for vid in new_videos:
                    tmp_path = vid.filename
                    # Get the parent dir of the YAML file.
                    # Use "/" since this works on Windows and posix
                    img_store_dir = (
                        os.path.basename(os.path.split(tmp_path)[0])
                        + "/"
                        + os.path.basename(tmp_path)
                    )
                    # Change to relative path
                    vid.backend.filename = img_store_dir

                # Convert to a dict, not JSON yet, because we need to patch up the videos
                d = labels.to_dict()
                d["videos"] = Video.cattr().unstructure(new_videos)

            else:
                d = labels.to_dict()

            if compress or save_frame_data:

                # Ensure that filename ends with .json
                # shutil will append .zip
                filename = re.sub("(\.json)?(\.zip)?$", ".json", filename)

                # Write the json to the tmp directory, we will zip it up with the frame data.
                full_out_filename = os.path.join(tmp_dir, os.path.basename(filename))
                json_dumps(d, full_out_filename)

                # Create the archive
                shutil.make_archive(base_name=filename, root_dir=tmp_dir, format="zip")

            # If the user doesn't want to compress, then just write the json to the filename
            else:
                json_dumps(d, filename)

    @classmethod
    def from_json(
        cls, data: Union[str, dict], match_to: Optional["Labels"] = None
    ) -> "Labels":
        """
        Create instance of class from data in dictionary.

        Method is used by other methods that load from JSON.

        Args:
            data: Dictionary, deserialized from JSON.
            match_to: If given, we'll replace particular objects in the
                data dictionary with *matching* objects in the match_to
                :class:`Labels` object. This ensures that the newly
                instantiated :class:`Labels` can be merged without
                duplicate matching objects (e.g., :class:`Video` objects ).
        Returns:
            A new :class:`Labels` object.
        """

        # Parse the json string if needed.
        if type(data) is str:
            dicts = json_loads(data)
        else:
            dicts = data

        dicts["tracks"] = dicts.get(
            "tracks", []
        )  # don't break if json doesn't include tracks

        # First, deserialize the skeletons, videos, and nodes lists.
        # The labels reference these so we will need them while deserializing.
        nodes = cattr.structure(dicts["nodes"], List[Node])

        idx_to_node = {i: nodes[i] for i in range(len(nodes))}
        skeletons = Skeleton.make_cattr(idx_to_node).structure(
            dicts["skeletons"], List[Skeleton]
        )
        videos = Video.cattr().structure(dicts["videos"], List[Video])

        try:
            # First try unstructuring tuple (newer format)
            track_cattr = cattr.Converter(
                unstruct_strat=cattr.UnstructureStrategy.AS_TUPLE
            )
            tracks = track_cattr.structure(dicts["tracks"], List[Track])
        except:
            # Then try unstructuring dict (older format)
            try:
                tracks = cattr.structure(dicts["tracks"], List[Track])
            except:
                raise ValueError("Unable to load tracks as tuple or dict!")

        # if we're given a Labels object to match, use its objects when they match
        if match_to is not None:
            for idx, sk in enumerate(skeletons):
                for old_sk in match_to.skeletons:
                    if sk.matches(old_sk):
                        # use nodes from matched skeleton
                        for (node, match_node) in zip(sk.nodes, old_sk.nodes):
                            node_idx = nodes.index(node)
                            nodes[node_idx] = match_node
                        # use skeleton from match
                        skeletons[idx] = old_sk
                        break
            for idx, vid in enumerate(videos):
                for old_vid in match_to.videos:
                    # compare last three parts of path
                    if vid.filename == old_vid.filename or weak_filename_match(
                        vid.filename, old_vid.filename
                    ):
                        # use video from match
                        videos[idx] = old_vid
                        break

        suggestions = []
        if "suggestions" in dicts:
            suggestions_cattr = cattr.Converter()
            suggestions_cattr.register_structure_hook(
                Video, lambda x, type: videos[int(x)]
            )
            try:
                suggestions = suggestions_cattr.structure(
                    dicts["suggestions"], List[SuggestionFrame]
                )
            except:
                try:
                    # Convert old suggestion format to new format.
                    # Old format: {video: list of frame indices}
                    # New format: [SuggestionFrames]
                    old_suggestions = suggestions_cattr.structure(
                        dicts["suggestions"], Dict[Video, List]
                    )
                    for video in old_suggestions.keys():
                        suggestions.extend(
                            [
                                SuggestionFrame(video, idx)
                                for idx in old_suggestions[video]
                            ]
                        )
                except:
                    print("Error while loading suggestions")
                    print(e)
                    pass

        if "negative_anchors" in dicts:
            negative_anchors_cattr = cattr.Converter()
            negative_anchors_cattr.register_structure_hook(
                Video, lambda x, type: videos[int(x)]
            )
            negative_anchors = negative_anchors_cattr.structure(
                dicts["negative_anchors"], Dict[Video, List]
            )
        else:
            negative_anchors = dict()

        # If there is actual labels data, get it.
        if "labels" in dicts:
            label_cattr = make_instance_cattr()
            label_cattr.register_structure_hook(
                Skeleton, lambda x, type: skeletons[int(x)]
            )
            label_cattr.register_structure_hook(Video, lambda x, type: videos[int(x)])
            label_cattr.register_structure_hook(
                Node, lambda x, type: x if isinstance(x, Node) else nodes[int(x)]
            )
            label_cattr.register_structure_hook(
                Track, lambda x, type: None if x is None else tracks[int(x)]
            )

            labels = label_cattr.structure(dicts["labels"], List[LabeledFrame])
        else:
            labels = []

        return cls(
            labeled_frames=labels,
            videos=videos,
            skeletons=skeletons,
            nodes=nodes,
            suggestions=suggestions,
            negative_anchors=negative_anchors,
            tracks=tracks,
        )

    @classmethod
    def load_json(
        cls,
        filename: str,
        video_callback: Optional[Callable] = None,
        match_to: Optional["Labels"] = None,
    ) -> "Labels":
        """
        Deserialize JSON file as new :class:`Labels` instance.

        Args:
            filename: Path to JSON file.
            video_callback: A callback function that which can modify
                video paths before we try to create the corresponding
                :class:`Video` objects. Usually you'll want to pass
                a callback created by :meth:`make_video_callback`
                or :meth:`make_gui_video_callback`.
            match_to: If given, we'll replace particular objects in the
                data dictionary with *matching* objects in the match_to
                :class:`Labels` object. This ensures that the newly
                instantiated :class:`Labels` can be merged without
                duplicate matching objects (e.g., :class:`Video` objects ).
        Returns:
            A new :class:`Labels` object.
        """

        tmp_dir = None

        # Check if the file is a zipfile for not.
        if zipfile.is_zipfile(filename):

            # Make a tmpdir, located in the directory that the file exists, to unzip
            # its contents.
            tmp_dir = os.path.join(
                os.path.dirname(filename),
                f"tmp_{os.getpid()}_{os.path.basename(filename)}",
            )
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                os.mkdir(tmp_dir)
            except FileExistsError:
                pass

            # tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(filename))

            try:

                # Register a cleanup routine that deletes the tmpdir on program exit
                # if something goes wrong. The True is for ignore_errors
                atexit.register(shutil.rmtree, tmp_dir, True)

                # Uncompress the data into the directory
                shutil.unpack_archive(filename, extract_dir=tmp_dir)

                # We can now open the JSON file, save the zip file and
                # replace file with the first JSON file we find in the archive.
                json_files = [
                    os.path.join(tmp_dir, file)
                    for file in os.listdir(tmp_dir)
                    if file.endswith(".json")
                ]

                if len(json_files) == 0:
                    raise ValueError(
                        f"No JSON file found inside {filename}. Are you sure this is a valid sLEAP dataset."
                    )

                filename = json_files[0]

            except Exception as ex:
                # If we had problems, delete the temp directory and reraise the exception.
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        # Open and parse the JSON in filename
        with open(filename, "r") as file:

            # FIXME: Peek into the json to see if there is version string.
            # We do this to tell apart old JSON data from leap_dev vs the
            # newer format for sLEAP.
            json_str = file.read()
            dicts = json_loads(json_str)

            # If we have a version number, then it is new sLEAP format
            if "version" in dicts:

                # Cache the working directory.
                cwd = os.getcwd()
                # Replace local video paths (for imagestore)
                if tmp_dir:
                    for vid in dicts["videos"]:
                        vid["backend"]["filename"] = os.path.join(
                            tmp_dir, vid["backend"]["filename"]
                        )

                # Use the callback if given to handle missing videos
                if callable(video_callback):
                    abort = video_callback(dicts["videos"])
                    if abort:
                        raise FileNotFoundError

                # Try to load the labels filename.
                try:
                    labels = Labels.from_json(dicts, match_to=match_to)

                except FileNotFoundError:

                    # FIXME: We are going to the labels JSON that has references to
                    # video files. Lets change directory to the dirname of the json file
                    # so that relative paths will be from this directory. Maybe
                    # it is better to feed the dataset dirname all the way down to
                    # the Video object. This seems like less coupling between classes
                    # though.
                    if os.path.dirname(filename) != "":
                        os.chdir(os.path.dirname(filename))

                    # Try again
                    labels = Labels.from_json(dicts, match_to=match_to)

                except Exception as ex:
                    # Ok, we give up, where the hell are these videos!
                    raise  # Re-raise.
                finally:
                    os.chdir(cwd)  # Make sure to change back if we have problems.

                return labels

            else:
                frames = load_labels_json_old(data_path=filename, parsed_json=dicts)
                return Labels(frames)

    @staticmethod
    def save_hdf5(
        labels: "Labels",
        filename: str,
        append: bool = False,
        save_frame_data: bool = False,
        frame_data_format: str = "png",
    ):
        """
        Serialize the labels dataset to an HDF5 file.

        Args:
            labels: The :class:`Labels` dataset to save
            filename: The file to serialize the dataset to.
            append: Whether to append these labeled frames to the file
                or not.
            save_frame_data: Whether to save the image frame data for
                any labeled frame as well. This is useful for uploading
                the HDF5 for model training when video files are to
                large to move. This will only save video frames that
                have some labeled instances.
            frame_data_format: If save_frame_data is True, then this argument
                is used to set the data format to use when encoding images
                saved in HDF5. Supported formats include:

                * "" for no encoding (ndarray)
                * "png"
                * "jpg"
                * anything else supported by `cv2.imencode`

        Returns:
            None
        """

        # Delete the file if it exists, we want to start from scratch since
        # h5py truncates the file which seems to not actually delete data
        # from the file. Don't if we are appending of course.
        if os.path.exists(filename) and not append:
            os.unlink(filename)

        # Serialize all the meta-data to JSON.
        d = labels.to_dict(skip_labels=True)

        if save_frame_data:
            new_videos = labels.save_frame_data_hdf5(filename, frame_data_format)

            # Replace path to video file with "." (which indicates that the
            # video is in the same file as the HDF5 labels dataset).
            # Otherwise, the video paths will break if the HDF5 labels
            # dataset file is moved.
            for vid in new_videos:
                vid.backend.filename = "."

            d["videos"] = Video.cattr().unstructure(new_videos)

        with h5.File(filename, "a") as f:

            # Add all the JSON metadata
            meta_group = f.require_group("metadata")

            # If we are appending and there already exists JSON metadata
            if append and "json" in meta_group.attrs:

                # Otherwise, we need to read the JSON and append to the lists
                old_labels = Labels.from_json(
                    meta_group.attrs["json"].tostring().decode()
                )

                # A function to join to list but only include new non-dupe entries
                # from the right hand list.
                def append_unique(old, new):
                    unique = []
                    for x in new:
                        try:
                            matches = [y.matches(x) for y in old]
                        except AttributeError:
                            matches = [x == y for y in old]

                        # If there were no matches, this is a unique object.
                        if sum(matches) == 0:
                            unique.append(x)
                        else:
                            # If we have an object that matches, replace the instance with
                            # the one from the new list. This will will make sure objects
                            # on the Instances are the same as those in the Labels lists.
                            for i, match in enumerate(matches):
                                if match:
                                    old[i] = x

                    return old + unique

                # Append the lists
                labels.tracks = append_unique(old_labels.tracks, labels.tracks)
                labels.skeletons = append_unique(old_labels.skeletons, labels.skeletons)
                labels.videos = append_unique(old_labels.videos, labels.videos)
                labels.nodes = append_unique(old_labels.nodes, labels.nodes)

                # FIXME: Do something for suggestions and negative_anchors

                # Get the dict for JSON and save it over the old data
                d = labels.to_dict(skip_labels=True)

            # Output the dict to JSON
            meta_group.attrs["json"] = np.string_(json_dumps(d))

            # FIXME: We can probably construct these from attrs fields
            # We will store Instances and PredcitedInstances in the same
            # table. instance_type=0 or Instance and instance_type=1 for
            # PredictedInstance, score will be ignored for Instances.
            instance_dtype = np.dtype(
                [
                    ("instance_id", "i8"),
                    ("instance_type", "u1"),
                    ("frame_id", "u8"),
                    ("skeleton", "u4"),
                    ("track", "i4"),
                    ("from_predicted", "i8"),
                    ("score", "f4"),
                    ("point_id_start", "u8"),
                    ("point_id_end", "u8"),
                ]
            )
            frame_dtype = np.dtype(
                [
                    ("frame_id", "u8"),
                    ("video", "u4"),
                    ("frame_idx", "u8"),
                    ("instance_id_start", "u8"),
                    ("instance_id_end", "u8"),
                ]
            )

            num_instances = len(labels.all_instances)
            max_skeleton_size = max([len(s.nodes) for s in labels.skeletons], default=0)

            # Initialize data arrays for serialization
            points = np.zeros(num_instances * max_skeleton_size, dtype=Point.dtype)
            pred_points = np.zeros(
                num_instances * max_skeleton_size, dtype=PredictedPoint.dtype
            )
            instances = np.zeros(num_instances, dtype=instance_dtype)
            frames = np.zeros(len(labels), dtype=frame_dtype)

            # Pre compute some structures to make serialization faster
            skeleton_to_idx = {
                skeleton: labels.skeletons.index(skeleton)
                for skeleton in labels.skeletons
            }
            track_to_idx = {
                track: labels.tracks.index(track) for track in labels.tracks
            }
            track_to_idx[None] = -1
            video_to_idx = {
                video: labels.videos.index(video) for video in labels.videos
            }
            instance_type_to_idx = {Instance: 0, PredictedInstance: 1}

            # Each instance we create will have and index in the dataset, keep track of
            # these so we can quickly add from_predicted links on a second pass.
            instance_to_idx = {}
            instances_with_from_predicted = []
            instances_from_predicted = []

            # If we are appending, we need look inside to see what frame, instance, and point
            # ids we need to start from. This gives us offsets to use.
            if append and "points" in f:
                point_id_offset = f["points"].shape[0]
                pred_point_id_offset = f["pred_points"].shape[0]
                instance_id_offset = f["instances"][-1]["instance_id"] + 1
                frame_id_offset = int(f["frames"][-1]["frame_id"]) + 1
            else:
                point_id_offset = 0
                pred_point_id_offset = 0
                instance_id_offset = 0
                frame_id_offset = 0

            point_id = 0
            pred_point_id = 0
            instance_id = 0

            for frame_id, label in enumerate(labels):
                frames[frame_id] = (
                    frame_id + frame_id_offset,
                    video_to_idx[label.video],
                    label.frame_idx,
                    instance_id + instance_id_offset,
                    instance_id + instance_id_offset + len(label.instances),
                )
                for instance in label.instances:

                    # Add this instance to our lookup structure we will need for from_predicted
                    # links
                    instance_to_idx[instance] = instance_id

                    parray = instance.get_points_array(copy=False, full=True)
                    instance_type = type(instance)

                    # Check whether we are working with a PredictedInstance or an Instance.
                    if instance_type is PredictedInstance:
                        score = instance.score
                        pid = pred_point_id + pred_point_id_offset
                    else:
                        score = np.nan
                        pid = point_id + point_id_offset

                        # Keep track of any from_predicted instance links, we will insert the
                        # correct instance_id in the dataset after we are done.
                        if instance.from_predicted:
                            instances_with_from_predicted.append(instance_id)
                            instances_from_predicted.append(instance.from_predicted)

                    # Copy all the data
                    instances[instance_id] = (
                        instance_id + instance_id_offset,
                        instance_type_to_idx[instance_type],
                        frame_id,
                        skeleton_to_idx[instance.skeleton],
                        track_to_idx[instance.track],
                        -1,
                        score,
                        pid,
                        pid + len(parray),
                    )

                    # If these are predicted points, copy them to the predicted point array
                    # otherwise, use the normal point array
                    if type(parray) is PredictedPointArray:
                        pred_points[
                            pred_point_id : pred_point_id + len(parray)
                        ] = parray
                        pred_point_id = pred_point_id + len(parray)
                    else:
                        points[point_id : point_id + len(parray)] = parray
                        point_id = point_id + len(parray)

                    instance_id = instance_id + 1

            # Add from_predicted links
            for instance_id, from_predicted in zip(
                instances_with_from_predicted, instances_from_predicted
            ):
                try:
                    instances[instance_id]["from_predicted"] = instance_to_idx[
                        from_predicted
                    ]
                except KeyError:
                    # If we haven't encountered the from_predicted instance yet then don't save the link.
                    # It’s possible for a user to create a regular instance from a predicted instance and then
                    # delete all predicted instances from the file, but in this case I don’t think there’s any reason
                    # to remember which predicted instance the regular instance came from.
                    pass

            # We pre-allocated our points array with max possible size considering the max
            # skeleton size, drop any unused points.
            points = points[0:point_id]
            pred_points = pred_points[0:pred_point_id]

            # Create datasets if we need to
            if append and "points" in f:
                f["points"].resize((f["points"].shape[0] + points.shape[0]), axis=0)
                f["points"][-points.shape[0] :] = points
                f["pred_points"].resize(
                    (f["pred_points"].shape[0] + pred_points.shape[0]), axis=0
                )
                f["pred_points"][-pred_points.shape[0] :] = pred_points
                f["instances"].resize(
                    (f["instances"].shape[0] + instances.shape[0]), axis=0
                )
                f["instances"][-instances.shape[0] :] = instances
                f["frames"].resize((f["frames"].shape[0] + frames.shape[0]), axis=0)
                f["frames"][-frames.shape[0] :] = frames
            else:
                f.create_dataset(
                    "points", data=points, maxshape=(None,), dtype=Point.dtype
                )
                f.create_dataset(
                    "pred_points",
                    data=pred_points,
                    maxshape=(None,),
                    dtype=PredictedPoint.dtype,
                )
                f.create_dataset(
                    "instances", data=instances, maxshape=(None,), dtype=instance_dtype
                )
                f.create_dataset(
                    "frames", data=frames, maxshape=(None,), dtype=frame_dtype
                )

    @classmethod
    def load_hdf5(
        cls, filename: str, video_callback=None, match_to: Optional["Labels"] = None
    ):
        """
        Deserialize HDF5 file as new :class:`Labels` instance.

        Args:
            filename: Path to HDF5 file.
            video_callback: A callback function that which can modify
                video paths before we try to create the corresponding
                :class:`Video` objects. Usually you'll want to pass
                a callback created by :meth:`make_video_callback`
                or :meth:`make_gui_video_callback`.
            match_to: If given, we'll replace particular objects in the
                data dictionary with *matching* objects in the match_to
                :class:`Labels` object. This ensures that the newly
                instantiated :class:`Labels` can be merged without
                duplicate matching objects (e.g., :class:`Video` objects ).

        Returns:
            A new :class:`Labels` object.
        """
        with h5.File(filename, "r") as f:

            # Extract the Labels JSON metadata and create Labels object with just
            # this metadata.
            dicts = json_loads(
                f.require_group("metadata").attrs["json"].tostring().decode()
            )

            # Video path "." means the video is saved in same file as labels,
            # so replace these paths.
            for video_item in dicts["videos"]:
                if video_item["backend"]["filename"] == ".":
                    video_item["backend"]["filename"] = filename

            # Use the callback if given to handle missing videos
            if callable(video_callback):
                video_callback(dicts["videos"])

            labels = cls.from_json(dicts, match_to=match_to)

            frames_dset = f["frames"][:]
            instances_dset = f["instances"][:]
            points_dset = f["points"][:]
            pred_points_dset = f["pred_points"][:]

            # Rather than instantiate a bunch of Point\PredictedPoint objects, we will
            # use inplace numpy recarrays. This will save a lot of time and memory
            # when reading things in.
            points = PointArray(buf=points_dset, shape=len(points_dset))
            pred_points = PredictedPointArray(
                buf=pred_points_dset, shape=len(pred_points_dset)
            )

            # Extend the tracks list with a None track. We will signify this with a -1 in the
            # data which will map to last element of tracks
            tracks = labels.tracks.copy()
            tracks.extend([None])

            # A dict to keep track of instances that have a from_predicted link. The key is the
            # instance and the value is the index of the instance.
            from_predicted_lookup = {}

            # Create the instances
            instances = []
            for i in instances_dset:
                track = tracks[i["track"]]
                skeleton = labels.skeletons[i["skeleton"]]

                if i["instance_type"] == 0:  # Instance
                    instance = Instance(
                        skeleton=skeleton,
                        track=track,
                        points=points[i["point_id_start"] : i["point_id_end"]],
                    )
                else:  # PredictedInstance
                    instance = PredictedInstance(
                        skeleton=skeleton,
                        track=track,
                        points=pred_points[i["point_id_start"] : i["point_id_end"]],
                        score=i["score"],
                    )
                instances.append(instance)

                if i["from_predicted"] != -1:
                    from_predicted_lookup[instance] = i["from_predicted"]

            # Make a second pass to add any from_predicted links
            for instance, from_predicted_idx in from_predicted_lookup.items():
                instance.from_predicted = instances[from_predicted_idx]

            # Create the labeled frames
            frames = [
                LabeledFrame(
                    video=labels.videos[frame["video"]],
                    frame_idx=frame["frame_idx"],
                    instances=instances[
                        frame["instance_id_start"] : frame["instance_id_end"]
                    ],
                )
                for i, frame in enumerate(frames_dset)
            ]

            labels.labeled_frames = frames

            # Do the stuff that should happen after we have labeled frames
            labels._build_lookup_caches()

        return labels

    @classmethod
    def load_file(cls, filename: str, *args, **kwargs):
        """Load file, detecting format from filename."""
        if filename.endswith((".h5", ".hdf5")):
            return cls.load_hdf5(filename, *args, **kwargs)
        elif filename.endswith((".json", ".json.zip")):
            return cls.load_json(filename, *args, **kwargs)
        elif filename.endswith(".mat"):
            return cls.load_mat(filename)
        elif filename.endswith(".csv"):
            # for now, the only csv we support is the DeepLabCut format
            return cls.load_deeplabcut_csv(filename)
        else:
            raise ValueError(f"Cannot detect filetype for {filename}")

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
        # Make sure that all directories for path exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Detect filetype and use appropriate save method
        if not filename.endswith((".json", ".zip", ".h5")) and default_suffix:
            filename += f".{default_suffix}"
        if filename.endswith((".json", ".zip")):
            compress = filename.endswith(".zip")
            cls.save_json(labels=labels, filename=filename, compress=compress, **kwargs)
        elif filename.endswith(".h5"):
            cls.save_hdf5(labels=labels, filename=filename, **kwargs)
        else:
            raise ValueError(f"Cannot detect filetype for {filename}")

    def save_frame_data_imgstore(
        self, output_dir: str = "./", format: str = "png", all_labels: bool = False
    ):
        """
        Write all labeled frames from all videos to imgstore datasets.

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
        Write labeled frames from all videos to hdf5 file.

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

    @staticmethod
    def _unwrap_mat_scalar(a):
        """Extract single value from nested MATLAB file data."""
        if a.shape == (1,):
            return Labels._unwrap_mat_scalar(a[0])
        else:
            return a

    @staticmethod
    def _unwrap_mat_array(a):
        """Extract list of values from nested MATLAB file data."""
        b = a[0][0]
        c = [Labels._unwrap_mat_scalar(x) for x in b]
        return c

    @classmethod
    def load_mat(cls, filename: str) -> "Labels":
        """Load LEAP MATLAB file as dataset.

        Args:
            filename: Path to csv file.
        Returns:
            The :class:`Labels` dataset.
        """
        mat_contents = sio.loadmat(filename)

        box_path = Labels._unwrap_mat_scalar(mat_contents["boxPath"])

        # If the video file isn't found, try in the same dir as the mat file
        if not os.path.exists(box_path):
            file_dir = os.path.dirname(filename)
            box_path_name = box_path.split("\\")[-1]  # assume windows path
            box_path = os.path.join(file_dir, box_path_name)

        if os.path.exists(box_path):
            vid = Video.from_hdf5(
                dataset="box", filename=box_path, input_format="channels_first"
            )
        else:
            vid = None

        # TODO: prompt user to locate video

        nodes_ = mat_contents["skeleton"]["nodes"]
        edges_ = mat_contents["skeleton"]["edges"]
        points_ = mat_contents["positions"]

        edges_ = edges_ - 1  # convert matlab 1-indexing to python 0-indexing

        nodes = Labels._unwrap_mat_array(nodes_)
        edges = Labels._unwrap_mat_array(edges_)

        nodes = list(map(str, nodes))  # convert np._str to str

        sk = Skeleton(name=filename)
        sk.add_nodes(nodes)
        for edge in edges:
            sk.add_edge(source=nodes[edge[0]], destination=nodes[edge[1]])

        labeled_frames = []
        node_count, _, frame_count = points_.shape

        for i in range(frame_count):
            new_inst = Instance(skeleton=sk)
            for node_idx, node in enumerate(nodes):
                x = points_[node_idx][0][i]
                y = points_[node_idx][1][i]
                new_inst[node] = Point(x, y)
            if len(new_inst.points):
                new_frame = LabeledFrame(video=vid, frame_idx=i)
                new_frame.instances = (new_inst,)
                labeled_frames.append(new_frame)

        labels = cls(labeled_frames=labeled_frames, videos=[vid], skeletons=[sk])

        return labels

    @classmethod
    def load_deeplabcut_csv(cls, filename: str) -> "Labels":
        """Load DeepLabCut csv file as dataset.

        Args:
            filename: Path to csv file.
        Returns:
            The :class:`Labels` dataset.
        """

        # At the moment we don't need anything from the config file,
        # but the code to read it is here in case we do in the future.

        # # Try to find the config file by walking up file path starting at csv file looking for config.csv
        # last_dir = None
        # file_dir = os.path.dirname(filename)
        # config_filename = ""

        # while file_dir != last_dir:
        #     last_dir = file_dir
        #     file_dir = os.path.dirname(file_dir)
        #     config_filename = os.path.join(file_dir, 'config.yaml')
        #     if os.path.exists(config_filename):
        #         break

        # # If we couldn't find a config file, give up
        # if not os.path.exists(config_filename): return

        # with open(config_filename, 'r') as f:
        #     config = yaml.load(f, Loader=yaml.SafeLoader)

        # x1 = config['x1']
        # y1 = config['y1']
        # x2 = config['x2']
        # y2 = config['y2']

        data = pd.read_csv(filename, header=[1, 2])

        # Create the skeleton from the list of nodes in the csv file
        # Note that DeepLabCut doesn't have edges, so these will have to be added by user later
        node_names = [n[0] for n in list(data)[1::2]]

        skeleton = Skeleton()
        skeleton.add_nodes(node_names)

        # Create an imagestore `Video` object from frame images.
        # This may not be ideal for large projects, since we're reading in
        # each image and then writing it out in a new directory.

        img_files = data.ix[:, 0]  # get list of all images

        # the image filenames in the csv may not match where the user has them
        # so we'll change the directory to match where the user has the csv
        def fix_img_path(img_dir, img_filename):
            img_filename = os.path.basename(img_filename)
            img_filename = os.path.join(img_dir, img_filename)
            return img_filename

        img_dir = os.path.dirname(filename)
        img_files = list(map(lambda f: fix_img_path(img_dir, f), img_files))

        # we'll put the new imgstore in the same directory as the current csv
        imgstore_name = os.path.join(os.path.dirname(filename), "sleap_video")

        # create the imgstore (or open if it already exists)
        if os.path.exists(imgstore_name):
            video = Video.from_filename(imgstore_name)
        else:
            video = Video.imgstore_from_filenames(img_files, imgstore_name)

        labels = []

        for i in range(len(data)):
            # get points for each node
            instance_points = dict()
            for node in node_names:
                x, y = data[(node, "x")][i], data[(node, "y")][i]
                instance_points[node] = Point(x, y)
            # create instance with points (we can assume there's only one instance per frame)
            instance = Instance(skeleton=skeleton, points=instance_points)
            # create labeledframe and add it to list
            label = LabeledFrame(video=video, frame_idx=i, instances=[instance])
            labels.append(label)

        return cls(labels)

    @classmethod
    def load_coco(
        cls, filename: str, img_dir: str, use_missing_gui: bool = False
    ) -> "Labels":
        with open(filename, "r") as file:
            json_str = file.read()
            dicts = json_loads(json_str)

        # Make skeletons from "categories"
        skeleton_map = dict()
        for category in dicts["categories"]:
            skeleton = Skeleton(name=category["name"])
            skeleton_id = category["id"]
            node_names = category["keypoints"]
            skeleton.add_nodes(node_names)

            try:
                for src_idx, dst_idx in category["skeleton"]:
                    skeleton.add_edge(node_names[src_idx], node_names[dst_idx])
            except IndexError as e:
                # According to the COCO data format specifications[^1], the edges
                # are supposed to be 1-indexed. But in some of their own
                # dataset the edges are 1-indexed! So we'll try.
                # [1]: http://cocodataset.org/#format-data

                # Clear any edges we already created using 0-indexing
                skeleton.clear_edges()

                # Add edges
                for src_idx, dst_idx in category["skeleton"]:
                    skeleton.add_edge(node_names[src_idx - 1], node_names[dst_idx - 1])

            skeleton_map[skeleton_id] = skeleton

        # Make videos from "images"

        # Remove images that aren't referenced in the annotations
        img_refs = [annotation["image_id"] for annotation in dicts["annotations"]]
        dicts["images"] = list(filter(lambda im: im["id"] in img_refs, dicts["images"]))

        # Key in JSON file should be "file_name", but sometimes it's "filename",
        # so we have to check both.
        img_filename_key = "file_name"
        if img_filename_key not in dicts["images"][0].keys():
            img_filename_key = "filename"

        # First add the img_dir to each image filename
        img_paths = [
            os.path.join(img_dir, image[img_filename_key]) for image in dicts["images"]
        ]

        # See if there are any missing files
        img_missing = [not os.path.exists(path) for path in img_paths]

        if sum(img_missing):
            if use_missing_gui:
                okay = MissingFilesDialog(img_paths, img_missing).exec_()

                if not okay:
                    return None
            else:
                raise FileNotFoundError(
                    f"Images for COCO dataset could not be found in {img_dir}."
                )

        # Update the image paths (with img_dir or user selected path)
        for image, path in zip(dicts["images"], img_paths):
            image[img_filename_key] = path

        # Create the video objects for the image files
        image_video_map = dict()

        vid_id_video_map = dict()
        for image in dicts["images"]:
            image_id = image["id"]
            image_filename = image[img_filename_key]

            # Sometimes images have a vid_id which links multiple images
            # together as one video. If so, we'll use that as the video key.
            # But if there isn't a vid_id, we'll treat each images as a
            # distinct video and use the image id as the video id.
            vid_id = image.get("vid_id", image_id)

            if vid_id not in vid_id_video_map:
                kwargs = dict(filenames=[image_filename])
                for key in ("width", "height"):
                    if key in image:
                        kwargs[key] = image[key]

                video = Video.from_image_filenames(**kwargs)
                vid_id_video_map[vid_id] = video
                frame_idx = 0
            else:
                video = vid_id_video_map[vid_id]
                frame_idx = video.num_frames
                video.backend.filenames.append(image_filename)

            image_video_map[image_id] = (video, frame_idx)

        # Make instances from "annotations"
        lf_map = dict()
        track_map = dict()
        for annotation in dicts["annotations"]:
            skeleton = skeleton_map[annotation["category_id"]]
            image_id = annotation["image_id"]
            video, frame_idx = image_video_map[image_id]
            keypoints = np.array(annotation["keypoints"], dtype="int").reshape(-1, 3)

            track = None
            if "track_id" in annotation:
                track_id = annotation["track_id"]
                if track_id not in track_map:
                    track_map[track_id] = Track(frame_idx, str(track_id))
                track = track_map[track_id]

            points = dict()
            any_visible = False
            for i in range(len(keypoints)):
                node = skeleton.nodes[i]
                x, y, flag = keypoints[i]

                if flag == 0:
                    # node not labeled for this instance
                    continue

                is_visible = flag == 2
                any_visible = any_visible or is_visible
                points[node] = Point(x, y, is_visible)

            if points:
                # If none of the points had 2 has the "visible" flag, we'll
                # assume this incorrect and just mark all as visible.
                if not any_visible:
                    for point in points.values():
                        point.visible = True

                inst = Instance(skeleton=skeleton, points=points, track=track)

                if image_id not in lf_map:
                    lf_map[image_id] = LabeledFrame(video, frame_idx)

                lf_map[image_id].insert(0, inst)

        return cls(labeled_frames=list(lf_map.values()))

    @classmethod
    def from_deepposekit(cls, filename: str, video_path: str, skeleton_path: str):
        video = Video.from_filename(video_path)

        skeleton_data = pd.read_csv(skeleton_path, header=0)
        skeleton = Skeleton()
        skeleton.add_nodes(skeleton_data["name"])
        nodes = skeleton.nodes

        for name, parent, swap in skeleton_data.itertuples(index=False, name=None):
            if parent is not np.nan:
                skeleton.add_edge(parent, name)

        lfs = []
        with h5.File(filename, "r") as f:
            pose_matrix = f["pose"][:]

            track_count, frame_count, node_count, _ = pose_matrix.shape

            tracks = [Track(0, f"Track {i}") for i in range(track_count)]
            for frame_idx in range(frame_count):
                lf_instances = []
                for track_idx in range(track_count):
                    points_array = pose_matrix[track_idx, frame_idx, :, :]
                    points = dict()
                    for p in range(len(points_array)):
                        x, y, score = points_array[p]
                        points[nodes[p]] = Point(x, y)  # TODO: score

                    inst = Instance(
                        skeleton=skeleton, track=tracks[track_idx], points=points
                    )
                    lf_instances.append(inst)
                lfs.append(
                    LabeledFrame(video, frame_idx=frame_idx, instances=lf_instances)
                )

        return cls(labeled_frames=lfs)

    @classmethod
    def make_video_callback(cls, search_paths: Optional[List] = None) -> Callable:
        """
        Create a non-GUI callback for finding missing videos.

        The callback can be used while loading a saved project and
        allows the user to find videos which have been moved (or have
        paths from a different system).

        Args:
            search_paths: If specified, this is a list of paths where
                we'll automatically try to find the missing videos.

        Returns:
            The callback function.
        """
        search_paths = search_paths or []

        def video_callback(video_list, new_paths=search_paths):
            # Check each video
            for video_item in video_list:
                if "backend" in video_item and "filename" in video_item["backend"]:
                    current_filename = video_item["backend"]["filename"]
                    # check if we can find video
                    if not os.path.exists(current_filename):

                        current_basename = os.path.basename(current_filename)
                        # handle unix, windows, or mixed paths
                        if current_basename.find("/") > -1:
                            current_basename = current_basename.split("/")[-1]
                        if current_basename.find("\\") > -1:
                            current_basename = current_basename.split("\\")[-1]

                        # First see if we can find the file in another directory,
                        # and if not, prompt the user to find the file.

                        # We'll check in the current working directory, and if the user has
                        # already found any missing videos, check in the directory of those.
                        for path_dir in new_paths:
                            check_path = os.path.join(path_dir, current_basename)
                            if os.path.exists(check_path):
                                # we found the file in a different directory
                                video_item["backend"]["filename"] = check_path
                                break

        return video_callback

    @classmethod
    def make_gui_video_callback(cls, search_paths: Optional[List] = None) -> Callable:
        """
        Create a callback with GUI for finding missing videos.

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

        def gui_video_callback(video_list, new_paths=search_paths):
            filenames = [item["backend"]["filename"] for item in video_list]
            missing = pathutils.list_file_missing(filenames)

            # First check for file in search_path directories
            if sum(missing) and new_paths:
                for i, filename in enumerate(filenames):
                    fixed_path = find_path_using_paths(filename, new_paths)
                    if fixed_path != filename:
                        filenames[i] = fixed_path
                        missing[i] = False

            # If there are still missing paths, prompt user
            if sum(missing):
                okay = MissingFilesDialog(filenames, missing).exec_()
                if not okay:
                    return True  # True for stop

            # Replace the video filenames with changes by user
            for i, item in enumerate(video_list):
                item["backend"]["filename"] = filenames[i]

        return gui_video_callback

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
    for path_dir in search_paths:
        check_path = os.path.join(path_dir, current_basename)
        if os.path.exists(check_path):
            return check_path

    return missing_path

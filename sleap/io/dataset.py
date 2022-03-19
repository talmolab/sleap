"""
A SLEAP dataset collects labeled video frames, together with required metadata.

This contains labeled frame data (user annotations and/or predictions),
together with all the other data that is saved for a SLEAP project
(videos, skeletons, etc.).

The most convenient way to load SLEAP labels files is to use the high level loader: ::

   > import sleap
   > labels = sleap.load_file(filename)

The Labels class provides additional functionality for loading SLEAP labels files. To
load a labels dataset file from disk: ::

   > labels = Labels.load_file(filename)

If you're opening a dataset file created on a different computer (or if you've
moved the video files), it's likely that the paths to the original videos will
not work. We automatically check for the videos in the same directory as the
labels file, but if the videos aren't there, you can tell `load_file` where
to search for the videos. There are various ways to do this: ::

   > Labels.load_file(filename, single_path_to_search)
   > Labels.load_file(filename, [path_a, path_b])
   > Labels.load_file(filename, callback_function)
   > Labels.load_file(filename, video_search=...)

The callback_function can be created via `make_video_callback()` and has the
option to make a callback with a GUI window so the user can locate the videos.

To save a labels dataset file, run: ::

   > Labels.save_file(labels, filename)

If the filename has a supported extension (e.g., ".slp", ".h5", ".json") then
the file will be saved in the corresponding format. You can also specify the
default extension to use if none is provided in the filename.
"""
import itertools
import os
from collections.abc import MutableSequence
from typing import (
    Callable,
    List,
    Union,
    Dict,
    Optional,
    Tuple,
    Text,
    Iterable,
    Any,
    Set,
    Callable,
)

import attr
import cattr
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split

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
from sleap.io.video import Video, ImgStoreVideo, HDF5Video
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
        """Build (or rebuilds) various caches."""
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
        """Return list of LabeledFrames matching video/frame_idx, or None."""
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
        """Return a list of frame idxs, with optional start position/order."""
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
        """Access track occupancy cache that adds video/track as needed."""
        if video not in self._track_occupancy:
            self._track_occupancy[video] = dict()

        if track not in self._track_occupancy[video]:
            self._track_occupancy[video][track] = RangeList()
        return self._track_occupancy[video][track]

    def get_video_track_occupancy(self, video: Video) -> Dict[Track, RangeList]:
        """Return track occupancy information for specified video."""
        if video not in self._track_occupancy:
            self._track_occupancy[video] = dict()

        return self._track_occupancy[video]

    def remove_frame(self, frame: LabeledFrame):
        """Remove frame and update cache as needed."""
        self._lf_by_video[frame.video].remove(frame)
        # We'll assume that there's only a single LabeledFrame for this video and
        # frame_idx, and remove the frame_idx from the cache.
        if frame.video in self._frame_idx_map:
            if frame.frame_idx in self._frame_idx_map[frame.video]:
                del self._frame_idx_map[frame.video][frame.frame_idx]

    def remove_video(self, video: Video):
        """Remove video and update cache as needed."""
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
        """Swap tracks and update cache as needed."""
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
        """Add a track to the labels."""
        self._track_occupancy[video][track] = RangeList()

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Add an instance to the labels."""
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
        """Remove an instance and update the cache as needed."""
        if instance.track not in self._track_occupancy[frame.video]:
            return

        # If this is only instance in track in frame, then remove frame from track.
        if len(frame.find(track=instance.track)) == 1:
            self._track_occupancy[frame.video][instance.track].remove(
                (frame.frame_idx, frame.frame_idx + 1)
            )

        self.update_counts_for_frame(frame)

    def get_frame_count(self, video: Optional[Video] = None, filter: Text = "") -> int:
        """Return (possibly cached) count of frames matching video/filter."""
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

    def get_filtered_frame_idxs(
        self, video: Optional[Video] = None, filter: Text = ""
    ) -> Set[Tuple[int, int]]:
        """Return list of (video_idx, frame_idx) tuples matching video/filter."""
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


@attr.s(auto_attribs=True, repr=False, str=False)
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
    suggestions: List[SuggestionFrame] = attr.ib(default=attr.Factory(list))
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
        """Ensure that top-level containers are kept updated with new
        instances of objects that come along with new labels."""

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

    @property
    def skeleton(self) -> Skeleton:
        """Return the skeleton if there is only a single skeleton in the labels."""
        if len(self.skeletons) == 1:
            return self.skeletons[0]
        else:
            raise ValueError(
                "Labels.skeleton can only be used when there is only a single skeleton "
                "saved in the labels. Use Labels.skeletons instead."
            )

    @property
    def video(self) -> Video:
        """Return the video if there is only a single video in the labels."""
        if len(self.videos) == 0:
            raise ValueError("There are no videos in the labels.")
        elif len(self.videos) == 1:
            return self.videos[0]
        else:
            raise ValueError(
                "Labels.video can only be used when there is only a single video saved "
                "in the labels. Use Labels.videos instead."
            )

    @property
    def has_missing_videos(self) -> bool:
        """Return True if any of the video files in the labels are missing."""
        return any(video.is_missing for video in self.videos)

    def __len__(self) -> int:
        """Return number of labeled frames."""
        return len(self.labeled_frames)

    def index(self, value) -> int:
        """Return index of labeled frame in list of labeled frames."""
        return self.labeled_frames.index(value)

    def __repr__(self) -> str:
        """Return a readable representation of the labels."""
        return (
            "Labels("
            f"labeled_frames={len(self.labeled_frames)}, "
            f"videos={len(self.videos)}, "
            f"skeletons={len(self.skeletons)}, "
            f"tracks={len(self.tracks)}"
            ")"
        )

    def __str__(self) -> str:
        """Return a readable representation of the labels."""
        return self.__repr__()

    def __contains__(self, item) -> bool:
        """Check if object contains the given item.

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
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Video):
            if isinstance(item[1], int):
                return self.find_first(*item) is not None
            elif isinstance(item[1], np.integer):
                return self.find_first(item[0], item[1].tolist()) is not None
        raise ValueError("Item is not an object type contained in labels.")

    def __getitem__(self, key, *args) -> Union[LabeledFrame, List[LabeledFrame]]:
        """Return labeled frames matching key.

        Args:
            key: Indexing argument to match against. If `key` is a `Video` or tuple of
                `(Video, frame_index)`, frames that match the criteria will be searched
                for. If a scalar, list, range or array of integers are provided, the
                labels with those linear indices will be returned.

        Raises:
            KeyError: If the specified key could not be found.

        Returns:
            A list with the matching `LabeledFrame`s, or a single `LabeledFrame` if a
            scalar key was provided.
        """
        if len(args) > 0:
            if type(key) != tuple:
                key = (key,)
            key = key + tuple(args)

        if isinstance(key, int):
            return self.labels.__getitem__(key)

        elif isinstance(key, Video):
            if key not in self.videos:
                raise KeyError("Video not found in labels.")
            return self.find(video=key)

        elif isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Video):
            if key[0] not in self.videos:
                raise KeyError("Video not found in labels.")

            if isinstance(key[1], int):
                _hit = self.find_first(video=key[0], frame_idx=key[1])
                if _hit is None:
                    raise KeyError(
                        f"No label found for specified video at frame {key[1]}."
                    )
                return _hit
            elif isinstance(key[1], (np.integer, np.ndarray)):
                return self.__getitem__((key[0], key[1].tolist()))
            elif isinstance(key[1], (list, range)):
                return self.find(video=key[0], frame_idx=key[1])
            else:
                raise KeyError("Invalid label indexing arguments.")

        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.__getitem__(range(start, stop, step))

        elif isinstance(key, (list, range)):
            return [self.__getitem__(i) for i in key]

        elif isinstance(key, (np.integer, np.ndarray)):
            return self.__getitem__(key.tolist())

        else:
            raise KeyError("Invalid label indexing arguments.")

    def get(self, *args) -> Union[LabeledFrame, List[LabeledFrame]]:
        """Get an item from the labels or return `None` if not found.

        This is a safe version of `labels[...]` that will not raise an exception if the
        item is not found.
        """
        try:
            return self.__getitem__(*args)
        except KeyError:
            return None

    def extract(self, inds, copy: bool = False) -> "Labels":
        """Extract labeled frames from indices and return a new `Labels` object.
        Args:
            inds: Any valid indexing keys, e.g., a range, slice, list of label indices,
                numpy array, `Video`, etc. See `__getitem__` for full list.
            copy: If `True`, create a new copy of all of the extracted labeled frames
                and associated labels. If `False` (the default), a shallow copy with
                references to the original labeled frames and other objects will be
                returned.
        Returns:
            A new `Labels` object with the specified labeled frames.
            This will preserve the other data structures even if they are not found in
            the extracted labels, including:
                - `Labels.videos`
                - `Labels.skeletons`
                - `Labels.tracks`
                - `Labels.suggestions`
                - `Labels.provenance`
        """
        lfs = self.__getitem__(inds)
        new_labels = type(self)(
            labeled_frames=lfs,
            videos=self.videos,
            skeletons=self.skeletons,
            tracks=self.tracks,
            suggestions=self.suggestions,
            provenance=self.provenance,
        )
        if copy:
            new_labels = new_labels.copy()
        return new_labels

    def copy(self) -> "Labels":
        """Return a full deep copy of the labels.
        Notes:
            All objects will be re-created by serializing and then deserializing the
            labels. This may be slow and will create new instances of all data
            structures.
        """
        return type(self).from_json(self.to_json())

    def split(
        self, n: Union[float, int], copy: bool = True
    ) -> Tuple["Labels", "Labels"]:
        """Split labels randomly.

        Args:
            n: Number or fraction of elements in the first split.
            copy: If `True` (the default), return copies of the labels.

        Returns:
            A tuple of `(labels_a, labels_b)` where both are `sleap.Labels` instances
            subsampled from these labels.

        Notes:
            If there is only 1 labeled frame, this will return two copies of the same
            labels. For `len(labels) > 1`, splits are guaranteed to be mutually
            exclusive.

        Example:
            You can generate multiple splits by calling this repeatedly:

            ```py
            # Generate a 0.8/0.1/0.1 train/val/test split.
            labels_train, labels_val_test = labels.split(n=0.8)
            labels_val, labels_test = labels_val_test.split(n=0.5)
            ```
        """
        if len(self) == 1:
            if copy:
                return self.copy(), self.copy()
            else:
                return self, self

        # Split indices.
        if type(n) != int:
            n = round(len(self) * n)
        n = max(min(n, len(self) - 1), 1)
        idx_a, idx_b = train_test_split(list(range(len(self))), train_size=n)

        return self.extract(idx_a, copy=copy), self.extract(idx_b, copy=copy)

    def __setitem__(self, index, value: LabeledFrame):
        """Set labeled frame at given index."""
        # TODO: Maybe we should remove this method altogether?
        self.labeled_frames.__setitem__(index, value)
        self._update_containers(value)

    def insert(self, index, value: LabeledFrame):
        """Insert labeled frame at given index."""
        if value in self or (value.video, value.frame_idx) in self:
            return

        self.labeled_frames.insert(index, value)
        self._update_containers(value)

    def append(self, value: LabeledFrame):
        """Add labeled frame to list of labeled frames."""
        self.insert(len(self) + 1, value)

    def __delitem__(self, key):
        """Remove labeled frame with given index."""
        self.labeled_frames.remove(self.labeled_frames[key])

    def remove(self, value: LabeledFrame):
        """Remove given labeled frame."""
        self.remove_frame(value)

    def remove_frame(self, lf: LabeledFrame, update_cache: bool = True):
        """Remove a given labeled frame.

        Args:
            lf: Labeled frame instance to remove.
            update_cache: If True, update the internal frame cache. If False, cache
                update can be postponed (useful when removing many frames).
        """
        self.labeled_frames.remove(lf)
        if update_cache:
            self._cache.remove_frame(lf)

    def remove_frames(self, lfs: List[LabeledFrame]):
        """Remove a list of frames from the labels.

        Args:
            lfs: A sequence of labeled frames to remove.
        """
        to_remove = set(lfs)
        self.labeled_frames = [lf for lf in self.labeled_frames if lf not in to_remove]
        self.update_cache()

    def remove_empty_instances(self, keep_empty_frames: bool = True):
        """Remove instances with no visible points.

        Args:
            keep_empty_frames: If True (the default), frames with no remaining instances
                will not be removed.

        Notes:
            This will modify the labels in place. If a copy is desired, call
            `labels.copy()` before this.
        """
        for lf in self.labeled_frames:
            lf.remove_empty_instances()
        self.update_cache()
        if not keep_empty_frames:
            self.remove_empty_frames()

    def remove_empty_frames(self):
        """Remove frames with no instances."""
        self.labeled_frames = [
            lf for lf in self.labeled_frames if len(lf.instances) > 0
        ]
        self.update_cache()

    def find(
        self,
        video: Video,
        frame_idx: Optional[Union[int, Iterable[int]]] = None,
        return_new: bool = False,
    ) -> List[LabeledFrame]:
        """Search for labeled frames given video and/or frame index.

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
        """Return an iterator over all labeled frames in a video.

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
        """Find the first occurrence of a matching labeled frame.

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
        """Find the last occurrence of a matching labeled frame.

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
    def user_labeled_frames(self) -> List[LabeledFrame]:
        """Return all labeled frames with user (non-predicted) instances."""
        return [lf for lf in self.labeled_frames if lf.has_user_instances]

    @property
    def user_labeled_frame_inds(self) -> List[int]:
        """Return a list of indices of frames with user labeled instances."""
        return [i for i, lf in enumerate(self.labeled_frames) if lf.has_user_instances]

    def with_user_labels_only(
        self,
        user_instances_only: bool = True,
        with_track_only: bool = False,
        copy: bool = True,
    ) -> "Labels":
        """Return a new `Labels` containing only user labels.

        This is useful as a preprocessing step to train on only user-labeled data.

        Args:
            user_instances_only: If `True` (the default), predicted instances will be
                removed from frames that also have user instances.
            with_track_only: If `True`, remove instances without a track.
            copy: If `True` (the default), create a new copy of all of the extracted
                labeled frames and associated labels. If `False`, a shallow copy with
                references to the original labeled frames and other objects will be
                returned. Warning: If returning a shallow copy, predicted and untracked
                instances will be removed from the original labels as well!

        Returns:
            A new `Labels` with only the specified subset of frames and instances.
        """
        new_labels = self.extract(self.user_labeled_frame_inds, copy=copy)
        if user_instances_only:
            new_labels.remove_predictions()
        if with_track_only:
            new_labels.remove_untracked_instances()
        new_labels.remove_empty_frames()
        return new_labels

    def get_labeled_frame_count(self, video: Optional[Video] = None, filter: Text = ""):
        return self._cache.get_frame_count(video, filter)

    def instance_count(self, video: Video, frame_idx: int) -> int:
        """Return number of instances matching video/frame index."""
        count = 0
        labeled_frame = self.find_first(video, frame_idx)
        if labeled_frame is not None:
            count = len(
                [inst for inst in labeled_frame.instances if isinstance(inst, Instance)]
            )
        return count

    @property
    def all_instances(self) -> List[Instance]:
        """Return list of all instances."""
        return list(self.instances())

    @property
    def user_instances(self) -> List[Instance]:
        """Return list of all user (non-predicted) instances."""
        return [inst for inst in self.all_instances if type(inst) == Instance]

    @property
    def predicted_instances(self) -> List[PredictedInstance]:
        """Return list of all predicted instances."""
        return [inst for inst in self.all_instances if type(inst) == PredictedInstance]

    @property
    def has_user_instances(self) -> bool:
        return any(lf.has_user_instances for lf in self.labeled_frames)

    @property
    def has_predicted_instances(self) -> bool:
        return any(lf.has_predicted_instances for lf in self.labeled_frames)

    @property
    def max_user_instances(self) -> int:
        n = 0
        for lf in self.labeled_frames:
            n = max(n, lf.n_user_instances)
        return n

    @property
    def min_user_instances(self) -> int:
        n = None
        for lf in self.labeled_frames:
            if n is not None:
                n = min(n, lf.n_user_instances)
            else:
                n = lf.n_user_instances
        return n

    @property
    def is_multi_instance(self) -> bool:
        """Returns `True` if there are multiple user instances in any frame."""
        return self.max_user_instances > 1

    def describe(self):
        """Print basic statistics about the labels dataset."""
        print(f"Skeleton: {self.skeleton}")
        print(f"Videos: {[v.filename for v in self.videos]}")
        n_user = 0
        n_pred = 0
        n_user_inst = 0
        n_pred_inst = 0
        for lf in self.labeled_frames:
            if lf.has_user_instances:
                n_user += 1
                n_user_inst += len(lf.user_instances)
            if lf.has_predicted_instances:
                n_pred += 1
                n_pred_inst += len(lf.predicted_instances)
        print(f"Frames (user/predicted): {n_user:,}/{n_pred:,}")
        print(f"Instances (user/predicted): {n_user_inst:,}/{n_pred_inst:,}")
        print("Tracks:", self.tracks)
        print(f"Suggestions: {len(self.suggestions):,}")
        print("Provenance:", self.provenance)

    def instances(
        self, video: Optional[Video] = None, skeleton: Optional[Skeleton] = None
    ):
        """Iterate over instances in the labels, optionally with filters.

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
                    points=template_points, nodes=skeleton.nodes
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
                    points=template_points, nodes=skeleton.nodes
                )

        return self._template_instance_points[skeleton]["points"]

    def get_track_count(self, video: Video) -> int:
        """Return the number of occupied tracks for a given video."""
        return len(self.get_track_occupancy(video))

    def get_track_occupancy(self, video: Video) -> List:
        """Return track occupancy list for given video."""
        return self._cache.get_video_track_occupancy(video=video)

    def add_track(self, video: Video, track: Track):
        """Add track to labels, updating occupancy."""
        self.tracks.append(track)
        self._cache.add_track(video, track)

    def remove_track(self, track: Track):
        """Remove a track from the labels, updating (but not removing) instances."""
        for inst in self.instances():
            if inst.track == track:
                inst.track = None
        self.tracks.remove(track)

    def remove_all_tracks(self):
        """Remove all tracks from labels, updating (but not removing) instances."""
        for inst in self.instances():
            inst.track = None
        self.tracks = []

    def track_set_instance(
        self, frame: LabeledFrame, instance: Instance, new_track: Track
    ):
        """Set track on given instance, updating occupancy."""
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
        """Swap track assignment for instances in two tracks.

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
        """Remove instance from frame, updating track occupancy."""
        frame.instances.remove(instance)
        if not in_transaction:
            self._cache.remove_instance(frame, instance)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Add instance to frame, updating track occupancy."""
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

    def add_suggestion(self, video: Video, frame_idx: int):
        """Add a suggested frame to the labels.

        Args:
            video: `sleap.Video` instance of the suggestion.
            frame_idx: Index of the frame of the suggestion.
        """
        for suggestion in self.suggestions:
            if suggestion.video == video and suggestion.frame_idx == frame_idx:
                return
        self.suggestions.append(SuggestionFrame(video=video, frame_idx=frame_idx))

    def remove_suggestion(self, video: Video, frame_idx: int):
        """Remove a suggestion from the list by video and frame index.

        Args:
            video: `sleap.Video` instance of the suggestion.
            frame_idx: Index of the frame of the suggestion.
        """
        for suggestion in self.suggestions:
            if suggestion.video == video and suggestion.frame_idx == frame_idx:
                self.suggestions.remove(suggestion)
                return

    def get_video_suggestions(
        self, video: Video, user_labeled: bool = True
    ) -> List[int]:
        """Return a list of suggested frame indices.

        Args:
            video: Video to get suggestions for.
            user_labeled: If `True` (the default), return frame indices for suggestions
                that already have user labels. If `False`, only suggestions with no user
                labeled instances will be returned.

        Returns:
            Indices of the suggested frames for for the specified video.
        """
        frame_indices = []
        for suggestion in self.suggestions:
            if suggestion.video == video:
                fidx = suggestion.frame_idx
                if not user_labeled:
                    lf = self.get((video, fidx))
                    if lf is not None and lf.has_user_instances:
                        continue
                frame_indices.append(fidx)
        return frame_indices

    def get_suggestions(self) -> List[SuggestionFrame]:
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

    def get_next_suggestion(self, video, frame_idx, seek_direction=1):
        """Return a (video, frame_idx) tuple seeking from given frame."""
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

    def set_suggestions(self, suggestions: List[SuggestionFrame]):
        """Set the suggested frames."""
        self.suggestions = suggestions

    def delete_suggestions(self, video):
        """Delete suggestions for specified video."""
        self.suggestions = [item for item in self.suggestions if item.video != video]

    def clear_suggestions(self):
        """Delete all suggestions."""
        self.suggestions = []

    @property
    def unlabeled_suggestions(self) -> List[SuggestionFrame]:
        """Return suggestions without user labels."""
        unlabeled_suggestions = []
        for suggestion in self.suggestions:
            lf = self.get(suggestion.video, suggestion.frame_idx)
            if lf is None or not lf.has_user_instances:
                unlabeled_suggestions.append(suggestion)
        return unlabeled_suggestions

    def get_unlabeled_suggestion_inds(self) -> List[int]:
        """Find labeled frames for unlabeled suggestions and return their indices.

        This is useful for generating a list of example indices for inference on
        unlabeled suggestions.

        Returns:
            List of indices of the labeled frames that correspond to the suggestions
            that do not have user instances.

            If a labeled frame corresponding to a suggestion does not exist, an empty
            one will be created.

        See also: `Labels.remove_empty_frames`
        """
        inds = []
        for suggestion in self.unlabeled_suggestions:
            lf = self.get((suggestion.video, suggestion.frame_idx))
            if lf is None:
                self.append(
                    LabeledFrame(video=suggestion.video, frame_idx=suggestion.frame_idx)
                )
                inds.append(len(self.labeled_frames) - 1)
            else:
                inds.append(self.index(lf))
        return inds

    def add_video(self, video: Video):
        """Add a video to the labels if it is not already in it.

        Video instances are added automatically when adding labeled frames,
        but this function allows for adding videos to the labels before any
        labeled frames are added.

        Args:
            video: `Video` instance

        """
        if video not in self.videos:
            self.videos.append(video)

    def remove_video(self, video: Video):
        """Remove a video from the labels and all associated labeled frames.

        Args:
            video: `Video` instance to be removed.
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

    @classmethod
    def from_json(cls, *args, **kwargs):
        from sleap.io.format.labels_json import LabelsJsonAdaptor

        return LabelsJsonAdaptor.from_json_data(*args, **kwargs)

    def extend_from(
        self, new_frames: Union["Labels", List[LabeledFrame]], unify: bool = False
    ):
        """Merge data from another `Labels` object or `LabeledFrame` list.

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

    def has_frame(
        self,
        lf: Optional[LabeledFrame] = None,
        video: Optional[Video] = None,
        frame_idx: Optional[int] = None,
        use_cache: bool = True,
    ) -> bool:
        """Check if the labels contain a specified frame.

        Args:
            lf: `LabeledFrame` to search for. If not provided, the `video` and
                `frame_idx` must not be `None`.
            video: `Video` of the frame. Not necessary if `lf` is given.
            frame_idx: Integer frame index of the frame. Not necessary if `lf` is given.
            use_cache: If `True` (the default), use label lookup cache for faster
                searching. If `False`, check every frame without the cache.

        Returns:
            A `bool` indicating whether the specified `LabeledFrame` is contained in the
            labels.

            This will return `True` if there is a matching frame with the same video and
            frame index, even if they contain different instances.

        Notes:
            The `Video` instance must be the same as the ones in these labels, so if
            comparing to `Video`s loaded from another file, be sure to load those labels
            with matching, i.e.: `sleap.Labels.load_file(..., match_to=labels)`.
        """
        if lf is not None:
            video = lf.video
            frame_idx = lf.frame_idx
        if video is None or frame_idx is None:
            raise ValueError("Either lf or video and frame_idx must be provided.")

        if use_cache:
            return len(self.find(video, frame_idx=frame_idx, return_new=False)) > 0

        else:
            if video not in self.videos:
                return False
            for lf in self.labeled_frames:
                if lf.video == video and lf.frame_idx == frame_idx:
                    return True
            return False

    def remove_user_instances(self, new_labels: Optional["Labels"] = None):
        """Clear user instances from the labels.

        Useful prior to merging operations to prevent overlapping instances from new
        labels.

        Args:
            new_labels: If not `None`, only user instances in frames that also contain
                user instances in the new labels will be removed. If not provided
                (the default), all user instances will be removed.

        Notes:
            If providing `new_labels`, it must have been loaded using
            `sleap.Labels.load_file(..., match_to=labels)` to ensure that conflicting
            frames can be detected.

            Labeled frames without any instances after clearing will also be removed
            from the dataset.
        """
        keep_lfs = []
        for lf in self.labeled_frames:
            if new_labels is not None:
                if not new_labels.has_frame(lf):
                    # Base frame is not in new labels, so just keep it without
                    # modification.
                    keep_lfs.append(lf)
                    continue

            if lf.has_predicted_instances:
                # Remove predictions from base frame.
                lf.instances = lf.predicted_instances
                keep_lfs.append(lf)

        # Keep only labeled frames with no conflicting predictions.
        self.labeled_frames = keep_lfs

    def remove_predictions(self, new_labels: Optional["Labels"] = None):
        """Clear predicted instances from the labels.

        Useful prior to merging operations to prevent overlapping instances from new
        predictions.

        Args:
            new_labels: If not `None`, only predicted instances in frames that also
                contain predictions in the new labels will be removed. If not provided
                (the default), all predicted instances will be removed.

        Notes:
            If providing `new_labels`, it must have been loaded using
            `sleap.Labels.load_file(..., match_to=labels)` to ensure that conflicting
            frames can be detected.

            Labeled frames without any instances after clearing will also be removed
            from the dataset.
        """
        keep_lfs = []
        for lf in self.labeled_frames:
            if new_labels is not None:
                if not new_labels.has_frame(lf):
                    # Base frame is not in new labels, so just keep it without
                    # modification.
                    keep_lfs.append(lf)
                    continue

            if lf.has_user_instances:
                # Remove predictions from base frame.
                lf.instances = lf.user_instances
                keep_lfs.append(lf)

        # Keep only labeled frames with no conflicting predictions.
        self.labeled_frames = keep_lfs

    def remove_untracked_instances(self, remove_empty_frames: bool = True):
        """Remove instances that do not have a track assignment.

        Args:
            remove_empty_frames: If `True` (the default), removes frames that do not
                contain any instances after removing untracked ones.
        """
        for lf in self.labeled_frames:
            lf.remove_untracked()
        if remove_empty_frames:
            self.remove_empty_frames()

    @classmethod
    def complex_merge_between(
        cls, base_labels: "Labels", new_labels: "Labels", unify: bool = True
    ) -> tuple:
        """Merge frames and other data from one dataset into another.

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
        """Finish conflicted merge from complex_merge_between.

        Args:
            base_labels: the `Labels` that we're merging into
            resolved_frames: the list of frames to add into base_labels
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
        """Merge `LabeledFrame` objects that are for the same video frame.

        Args:
            video: combine for this video; if None, do all videos
        """
        if video is None:
            for vid in {lf.video for lf in self.labeled_frames}:
                self.merge_matching_frames(video=vid)
        else:
            self.labeled_frames = LabeledFrame.merge_frames(
                self.labeled_frames, video=video
            )

    def to_dict(self, skip_labels: bool = False) -> Dict[str, Any]:
        """Serialize all labels to dicts.

        Serializes the labels in the underling list of LabeledFrames to a dict
        structure. This function returns a nested dict structure composed entirely of
        primitive python types. It is used to create JSON and HDF5 serialized datasets.

        Args:
            skip_labels: If True, skip labels serialization and just do the metadata.

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
        """Serialize all labels in the underling list of LabeledFrame(s) to JSON.

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
        """
        # Convert to full (absolute) path
        filename = os.path.abspath(filename)

        # Make sure that all directories for path exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        from .format import write

        write(filename, labels, *args, **kwargs)

    def save(
        self,
        filename: Text,
        with_images: bool = False,
        embed_all_labeled: bool = False,
        embed_suggested: bool = False,
    ):
        """Save the labels to a file.

        Args:
            filename: Path to save the labels to ending in `.slp`. If the filename does
                not end in `.slp`, the extension will be automatically appended.
            with_images: If `True`, the image data for frames with labels will be
                embedded in the saved labels. This is useful for generating a single
                file to be used when training remotely. Defaults to `False`.
            embed_all_labeled: If `True`, save image data for labeled frames without
                user-labeled instances (defaults to `False`). This is useful for
                selecting arbitrary frames to save by adding empty `LabeledFrame`s to
                the dataset. Labeled frame metadata will be saved regardless.
            embed_suggested: If `True`, save image data for frames in the suggestions
                (defaults to `False`). Useful for predicting on remaining suggestions
                after training. Suggestions metadata will be saved regardless.

        Notes:
            This is an instance-level wrapper for the `Labels.save_file` class method.
        """
        if os.path.splitext(filename)[1].lower() != ".slp":
            filename = filename + ".slp"
        Labels.save_file(
            self,
            filename,
            save_frame_data=with_images,
            all_labeled=embed_all_labeled,
            suggested=embed_suggested,
        )

    def export(self, filename: str):
        """Export labels to analysis HDF5 format.

        This expects the labels to contain data for a single video (e.g., predictions).

        Args:
            filename: Path to output HDF5 file.

        Notes:
            This will write the contents of the labels out as a HDF5 file without
            complete metadata.

            The resulting file will have datasets:
                - `/node_names`: List of skeleton node names.
                - `/track_names`: List of track names.
                - `/tracks`: All coordinates of the instances in the labels.
                - `/track_occupancy`: Mask denoting which instances are present in each
                    frame.
        """
        from sleap.io.format.sleap_analysis import SleapAnalysisAdaptor

        SleapAnalysisAdaptor.write(filename, self)

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
    def load_deeplabcut_folder(cls, filename: str) -> "Labels":
        csv_files = glob(f"{filename}/*/*.csv")
        merged_labels = None
        for csv_file in csv_files:
            labels = cls.load_file(csv_file, as_format="deeplabcut")
            if merged_labels is None:
                merged_labels = labels
            else:
                merged_labels.extend_from(labels, unify=True)
        return merged_labels

    @classmethod
    def load_coco(
        cls, filename: str, img_dir: str, use_missing_gui: bool = False
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
    ) -> List[ImgStoreVideo]:
        """Write images for labeled frames from all videos to imgstore datasets.

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
        self,
        output_path: str,
        format: str = "png",
        user_labeled: bool = True,
        all_labeled: bool = False,
        suggested: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[HDF5Video]:
        """Write images for labeled frames from all videos to hdf5 file.

        Note that this will make an HDF5 video, not an HDF5 labels dataset.

        Args:
            output_path: Path to HDF5 file.
            format: The image format to use for the data. Defaults to png.
            user_labeled: Include labeled frames with user instances. Defaults to
                `True`.
            all_labeled: Include all labeled frames, including those with user-labeled
                instances, predicted instances or labeled frames with no instances.
                Defaults to `False`.
            suggested: Include suggested frames even if they do not have instances.
                Useful for inference after training. Defaults to `False`.
            progress_callback: If provided, this function will be called to report the
                progress of the frame data saving. This function should be a callable
                of the form: `fn(n, n_total)` where `n` is the number of frames saved so
                far and `n_total` is the total number of frames that will be saved. This
                is called after each video is processed. If the function has a return
                value and it returns `False`, saving will be canceled and the output
                deleted.

        Returns:
            A list of :class:`HDF5Video` objects with the stored frames.
        """
        # Build list of frames to save.
        vids = []
        frame_idxs = []
        for video in self.videos:
            lfs_v = self.find(video)
            frame_nums = [
                lf.frame_idx
                for lf in lfs_v
                if all_labeled or (user_labeled and lf.has_user_instances)
            ]
            if suggested:
                frame_nums += [
                    suggestion.frame_idx
                    for suggestion in self.suggestions
                    if suggestion.video == video
                ]
            frame_nums = sorted(list(set(frame_nums)))
            vids.append(video)
            frame_idxs.append(frame_nums)

        n_total = sum([len(x) for x in frame_idxs])
        n = 0

        # Save images for each video.
        new_vids = []
        for v_idx, (video, frame_nums) in enumerate(zip(vids, frame_idxs)):
            vid = video.to_hdf5(
                path=output_path,
                dataset=f"video{v_idx}",
                format=format,
                frame_numbers=frame_nums,
            )
            n += len(frame_nums)
            if progress_callback is not None:
                # Notify update callback.
                ret = progress_callback(n, n_total)
                if ret == False:
                    vid.close()
                    return []

            vid.close()
            new_vids.append(vid)

        return new_vids

    def to_pipeline(
        self,
        batch_size: Optional[int] = None,
        prefetch: bool = True,
        frame_indices: Optional[List[int]] = None,
        user_labeled_only: bool = True,
    ) -> "sleap.pipelines.Pipeline":
        """Create a pipeline for reading the dataset.

        Args:
            batch_size: If not `None`, the video frames will be batched into rank-4
                tensors. Otherwise, single rank-3 images will be returned.
            prefetch: If `True`, pipeline will include prefetching.
            frame_indices: Labeled frame indices to limit the pipeline reader to. If not
                specified (default), pipeline will read all the labeled frames in the
                dataset.
            user_labeled_only: If `True` (default), will only read frames with user
                labeled instances.

        Returns:
            A `sleap.pipelines.Pipeline` that builds `tf.data.Dataset` for high
            throughput I/O during inference.

        See also: sleap.pipelines.LabelsReader
        """
        from sleap.nn.data import pipelines

        if user_labeled_only:
            reader = pipelines.LabelsReader.from_user_instances(self)
            reader.example_indices = frame_indices
        else:
            reader = pipelines.LabelsReader(self, example_indices=frame_indices)
        pipeline = pipelines.Pipeline(reader)
        if batch_size is not None:
            pipeline += pipelines.Batcher(
                batch_size=batch_size, drop_remainder=False, unrag=False
            )

        pipeline += pipelines.Prefetcher()
        return pipeline

    def numpy(
        self,
        video: Optional[Union[Video, int]] = None,
        all_frames: bool = True,
        untracked: bool = False,
    ) -> np.ndarray:
        """Construct a numpy array from instance points.

        Args:
            video: Video or video index to convert to numpy arrays. If `None` (the
                default), uses the first video.
            all_frames: If `True` (the default), allocate array of the same number of
                frames as the video. If `False`, only return data between the first and
                last frame with data.
            untracked: If `False` (the default), include only instances that have a
                track assignment. If `True`, includes all instances in each frame in
                arbitrary order.

        Returns:
            An array of tracks of shape `(n_frames, n_tracks, n_nodes, 2)`.

            Missing data will be replaced with `np.nan`.

            If this is a single instance project, a track does not need to be assigned.

            Only predicted instances (NOT user instances) will be returned.

        Notes:
            This method assumes that instances have tracks assigned and is intended to
            function primarily for single-video prediction results.
        """
        # Get labeled frames for specified video.
        if video is None:
            video = 0
        if type(video) == int:
            video = self.videos[video]
        lfs = self.find(video=video)

        # Figure out frame index range.
        if all_frames:
            first_frame, last_frame = 0, video.shape[0] - 1
        else:
            first_frame, last_frame = None, None
            for lf in lfs:
                if first_frame is None:
                    first_frame = lf.frame_idx
                if last_frame is None:
                    last_frame = lf.frame_idx
                first_frame = min(first_frame, lf.frame_idx)
                last_frame = max(last_frame, lf.frame_idx)

        # Figure out the number of tracks based on number of instances in each frame.
        #
        # First, let's check the max number of predicted instances (regardless of
        # whether they're tracked.
        n_preds = 0
        for lf in lfs:
            n_preds = max(n_preds, lf.n_predicted_instances)

        # Case 1: We don't care about order because there's only 1 instance per frame,
        # or we're considering untracked instances.
        untracked = untracked or n_preds == 1
        if untracked:
            n_tracks = n_preds
        else:
            # Case 2: We're considering only tracked instances.
            n_tracks = len(self.tracks)

        n_frames = last_frame - first_frame + 1
        n_nodes = len(self.skeleton.nodes)

        tracks = np.full((n_frames, n_tracks, n_nodes, 2), np.nan, dtype="float32")
        for lf in lfs:
            i = lf.frame_idx - first_frame
            if untracked:
                for j, inst in enumerate(lf.predicted_instances):
                    tracks[i, j] = inst.numpy()
            else:
                for inst in lf.tracked_instances:
                    j = self.tracks.index(inst.track)
                    tracks[i, j] = inst.numpy()

        return tracks

    def merge_nodes(self, base_node: str, merge_node: str):
        """Merge two nodes and update data accordingly.

        Args:
            base_node: Name of skeleton node that will remain after merging.
            merge_node: Name of skeleton node that will be merged into the base node.

        Notes:
            This method can be used to merge two nodes that might have been named
            differently but that should be associated with the same node.

            This is useful, for example, when merging a different set of labels where
            a node was named differently.

            If the `base_node` is visible and has data, it will not be updated.
            Otherwise, it will be updated with the data from the `merge_node` on the
            same instance.
        """
        # Update data on all instances.
        for inst in self.instances():
            inst._merge_nodes_data(base_node, merge_node)

        # Remove merge node from skeleton.
        self.skeleton.delete_node(merge_node)

        # Fix instances.
        for inst in self.instances():
            inst._fix_array()

    @classmethod
    def make_gui_video_callback(cls, search_paths: Optional[List] = None) -> Callable:
        return cls.make_video_callback(search_paths=search_paths, use_gui=True)

    @classmethod
    def make_video_callback(
        cls, search_paths: Optional[List] = None, use_gui: bool = False
    ) -> Callable:
        """Create a callback for finding missing videos.

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


def find_path_using_paths(missing_path: Text, search_paths: List[Text]) -> Text:
    """Find a path to a missing file given a set of paths to search in.

    Args:
        missing_path: Path to the missing filename.
        search_paths: List of paths to search in.

    Returns:
        The corrected path if it was found, or the original missing path if it was not.
    """
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


def load_file(
    filename: Text,
    detect_videos: bool = True,
    search_paths: Optional[Union[List[Text], Text]] = None,
    match_to: Optional[Labels] = None,
) -> Labels:
    """Load a SLEAP labels file.

    SLEAP labels files (`.slp`) contain all the metadata for a labeling project or the
    predicted labels from a video. This includes the skeleton, videos, labeled frames,
    user-labeled and predicted instances, suggestions and tracks.

    See `sleap.io.dataset.Labels` for more detailed information.

    Args:
        filename: Path to a SLEAP labels (.slp) file.
        detect_videos: If True, will attempt to detect missing videos by searching for
            their filenames in the search paths. This is useful when loading SLEAP
            labels files that were generated on another computer with different paths.
        search_paths: A path or list of paths to search for the missing videos. This can
            be the direct path to the video file or its containing folder. If not
            specified, defaults to searching for the videos in the same folder as the
            labels.
        match_to: If a `sleap.Labels` object is provided, attempt to match and reuse
            video and skeleton objects when loading. This is useful when comparing the
            contents across sets of labels.

    Returns:
        The loaded `Labels` instance.

    Notes:
        This is a convenience method to call `sleap.Labels.load_file`. See that class
        method for more functionality in the loading process.

        The video files do not need to be accessible in order to load the labels, for
        example, when only the predicted instances or user labels are required.
    """
    if detect_videos:
        if search_paths is None:
            search_paths = os.path.dirname(filename)
        return Labels.load_file(filename, search_paths, match_to=match_to)
    else:
        return Labels.load_file(filename, match_to=match_to)

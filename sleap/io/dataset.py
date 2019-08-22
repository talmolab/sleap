"""A LEAP Dataset represents annotated (labeled) video data.

A LEAP Dataset stores almost all data required for training of a model.
This includes, raw video frame data, labelled instances of skeleton _points,
confidence maps, part affinity fields, and skeleton data. A LEAP :class:`.Dataset`
is a high level API to these data structures that abstracts away their underlying
storage format.

"""

import os
import re
import zipfile
import atexit
import glob

import attr
import cattr
import json
import shutil
import tempfile
import numpy as np
import scipy.io as sio
import h5py as h5

from collections import MutableSequence
from typing import List, Union, Dict, Optional, Tuple

try:
    from typing import ForwardRef
except:
    from typing import _ForwardRef as ForwardRef

import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, Point, LabeledFrame, \
    Track, PredictedPoint, PredictedInstance
from sleap.rangelist import RangeList
from sleap.io.video import Video
from sleap.util import save_dict_to_hdf5

"""
The version number to put in the Labels JSON format.
"""
LABELS_JSON_FILE_VERSION = "2.0.0"

@attr.s(auto_attribs=True)
class Labels(MutableSequence):
    """
    The LEAP :class:`.Labels` class represents an API for accessing labeled video
    frames and other associated metadata. This class is front-end for all
    interactions with loading, writing, and modifying these labels. The actual
    storage backend for the data is mostly abstracted away from the main
    interface.

    Args:
        labeled_frames: A list of `LabeledFrame`s
        videos: A list of videos that these labels may or may not reference.
        That is, every LabeledFrame's video will be in videos but a Video
        object from videos might not have any LabeledFrame.
        skeletons: A list of skeletons that these labels may or may not reference.
        tracks: A list of tracks that instances can belong to.
        suggestions: A dict with a list for each video of suggested frames to label.
    """

    labeled_frames: List[LabeledFrame] = attr.ib(default=attr.Factory(list))
    videos: List[Video] = attr.ib(default=attr.Factory(list))
    skeletons: List[Skeleton] = attr.ib(default=attr.Factory(list))
    nodes: List[Node] = attr.ib(default=attr.Factory(list))
    tracks: List[Track] = attr.ib(default=attr.Factory(list))
    suggestions: Dict[Video, list] = attr.ib(default=attr.Factory(dict))
    negative_anchors: Dict[Video, list] = attr.ib(default=attr.Factory(dict))

    def __attrs_post_init__(self):

        # Add any videos that are present in the labels but
        # missing from the video list
        self.videos = list(set(self.videos).union({label.video for label in self.labels}))

        # Ditto for skeletons
        self.skeletons = list(set(self.skeletons).union({instance.skeleton
                                                         for label in self.labels
                                                         for instance in label.instances}))

        # Ditto for nodes
        self.nodes = list(set(self.nodes).union({node for skeleton in self.skeletons for node in skeleton.nodes}))

        # Keep the tracks we already have
        tracks = set(self.tracks)

        # Add tracks from any Instances or PredictedInstances
        tracks = tracks.union({instance.track
                               for frame in self.labels
                               for instance in frame.instances
                               if instance.track})

        # Add tracks from any PredictedInstance referenced by instance
        # This fixes things when there's a referenced PredictionInstance
        # which is no longer in the frame.
        tracks = tracks.union({instance.from_predicted.track
                               for frame in self.labels
                               for instance in frame.instances
                               if instance.from_predicted
                                 and instance.from_predicted.track})

        # Lets sort the tracks by spawned on and then name
        self.tracks = sorted(list(tracks), key=lambda t:(t.spawned_on, t.name))

        # Data structures for caching
        self._lf_by_video = dict()
        self._frame_idx_map = dict()
        self._track_occupancy = dict()
        for video in self.videos:
            self._lf_by_video[video] = [lf for lf in self.labels if lf.video == video]
            self._frame_idx_map[video] = {lf.frame_idx: lf for lf in self._lf_by_video[video]}
            self._track_occupancy[video] = self._make_track_occupany(video)

        # Create a variable to store a temporary storage directory. When we unzip
        self.__temp_dir = None

    # Below are convenience methods for working with Labels as list.
    # Maybe we should just inherit from list? Maybe this class shouldn't
    # exists since it is just a list really with some class methods. I
    # think more stuff might appear in this class later down the line
    # though.

    @property
    def labels(self):
        """ Alias for labeled_frames """
        return self.labeled_frames

    @property
    def user_labeled_frames(self):
        return [lf for lf in self.labeled_frames if lf.has_user_instances]

    def __len__(self):
        return len(self.labeled_frames)

    def index(self, value):
        return self.labeled_frames.index(value)

    def __contains__(self, item):
        if isinstance(item, LabeledFrame):
            return item in self.labeled_frames
        elif isinstance(item, Video):
            return item in self.videos
        elif isinstance(item, Skeleton):
            return item in self.skeletons
        elif isinstance(item, Node):
            return item in self.nodes
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Video) and isinstance(item[1], int):
            return self.find_first(*item) is not None

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.labels.__getitem__(key)

        elif isinstance(key, Video):
            if key not in self.videos:
                raise KeyError("Video not found in labels.")
            return self.find(video=key)

        elif isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Video) and isinstance(key[1], int):
            if key[0] not in self.videos:
                raise KeyError("Video not found in labels.")

            _hit = self.find_first(video=key[0], frame_idx=key[1])

            if _hit is None:
                raise KeyError(f"No label found for specified video at frame {key[1]}.")

            return _hit

        else:
            raise KeyError("Invalid label indexing arguments.")

    def find(self, video: Video, frame_idx: Union[int, range] = None, return_new: bool=False) -> List[LabeledFrame]:
        """ Search for labeled frames given video and/or frame index.

        Args:
            video: a `Video` instance that is associated with the labeled frames
            frame_idx: an integer specifying the frame index within the video
            return_new: return singleton of new `LabeledFrame` if none found?

        Returns:
            List of `LabeledFrame`s that match the criteria. Empty if no matches found.

        """
        null_result = [LabeledFrame(video=video, frame_idx=frame_idx)] if return_new else []

        if frame_idx is not None:
            if video not in self._frame_idx_map: return null_result

            if type(frame_idx) == range:
                return [self._frame_idx_map[video][idx] for idx in frame_idx if idx in self._frame_idx_map[video]]

            if frame_idx not in self._frame_idx_map[video]: return null_result

            return [self._frame_idx_map[video][frame_idx]]
        else:
            if video not in self._lf_by_video: return null_result
            return self._lf_by_video[video]

    def frames(self, video: Video, from_frame_idx: int = -1, reverse=False):
        """
        Iterator over all frames in a video, starting with first frame
        after specified frame_idx (or first frame in video if none specified).
        """
        if video not in self._frame_idx_map: return None

        # Get sorted list of frame indexes for this video
        frame_idxs = sorted(self._frame_idx_map[video].keys(), reverse=reverse)

        # Find the next frame index after the specified frame
        next_frame_idx = min(filter(lambda x: x > from_frame_idx, frame_idxs), default=frame_idxs[0])
        cut_list_idx = frame_idxs.index(next_frame_idx)

        # Shift list of frame indices to start with specified frame
        frame_idxs = frame_idxs[cut_list_idx:] + frame_idxs[:cut_list_idx]

        # Yield the frames
        for idx in frame_idxs:
            yield self._frame_idx_map[video][idx]

    def find_first(self, video: Video, frame_idx: int = None) -> LabeledFrame:
        """ Find the first occurrence of a labeled frame for the given video and/or frame index.

        Args:
            video: a `Video` instance that is associated with the labeled frames
            frame_idx: an integer specifying the frame index within the video

        Returns:
            First `LabeledFrame` that match the criteria or None if none were found.
        """

        if video in self.videos:
            for label in self.labels:
                if label.video == video and (frame_idx is None or (label.frame_idx == frame_idx)):
                    return label

    def find_last(self, video: Video, frame_idx: int = None) -> LabeledFrame:
        """ Find the last occurrence of a labeled frame for the given video and/or frame index.

        Args:
            video: A `Video` instance that is associated with the labeled frames
            frame_idx: An integer specifying the frame index within the video

        Returns:
            LabeledFrame: Last label that matches the criteria or None if no results.
        """

        if video in self.videos:
            for label in reversed(self.labels):
                if label.video == video and (frame_idx is None or (label.frame_idx == frame_idx)):
                    return label

    def instance_count(self, video: Video, frame_idx: int) -> int:
        count = 0
        labeled_frame = self.find_first(video, frame_idx)
        if labeled_frame is not None:
            count = len([inst for inst in labeled_frame.instances if type(inst)==Instance])
        return count

    def get_track_occupany(self, video: Video):
        try:
            return self._track_occupancy[video]
        except:
            return []

    def add_track(self, video: Video, track: Track):
        self.tracks.append(track)
        self._track_occupancy[video][track] = RangeList()

    def track_set_instance(self, frame: LabeledFrame, instance: Instance, new_track: Track):
        self.track_swap(frame.video, new_track, instance.track, (frame.frame_idx, frame.frame_idx+1))
        if instance.track is None:
            self._track_remove_instance(frame, instance)
        instance.track = new_track

    def track_swap(self, video: Video, new_track: Track, old_track: Track, frame_range: tuple):

        # Get ranges in track occupancy cache
        _, within_old, _ = self._track_occupancy[video][old_track].cut_range(frame_range)
        _, within_new, _ = self._track_occupancy[video][new_track].cut_range(frame_range)

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
        for frame, instance in old_track_instances:
            instance.track = new_track
        # old_track can be `Track` or int
        # If int, it's index in instance list which we'll use as a pseudo-track,
        # but we won't set instances currently on new_track to old_track.
        if type(old_track) == Track:
            for frame, instance in new_track_instances:
                instance.track = old_track

    def _track_remove_instance(self, frame: LabeledFrame, instance: Instance):
        if instance.track not in self._track_occupancy[frame.video]: return

        # If this is only instance in track in frame, then remove frame from track.
        if len(list(filter(lambda inst: inst.track == instance.track, frame.instances))) == 1:
            self._track_occupancy[frame.video][instance.track].remove((frame.frame_idx, frame.frame_idx+1))

    def remove_instance(self, frame: LabeledFrame, instance: Instance):
        self._track_remove_instance(frame, instance)
        frame.instances.remove(instance)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        if frame.video not in self._track_occupancy:
            self._track_occupancy[frame.video] = dict()

        # Ensure that there isn't already an Instance with this track
        tracks_in_frame = [inst.track for inst in frame
                           if type(inst) == Instance and inst.track is not None]
        if instance.track in tracks_in_frame:
            instance.track = None

        # Add track in its not already present in labels
        if instance.track not in self._track_occupancy[frame.video]:
            self._track_occupancy[frame.video][instance.track] = RangeList()

        self._track_occupancy[frame.video][instance.track].insert((frame.frame_idx, frame.frame_idx+1))
        frame.instances.append(instance)

    def _make_track_occupany(self, video):
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

    def find_track_occupancy(self, video: Video, track: Union[Track, int], frame_range=None) -> List[Tuple[LabeledFrame, Instance]]:
        """Get instances for a given track.

        Args:
            video: the `Video`
            track: the `Track` or int ("pseudo-track" index to instance list)
            frame_range (optional):
                If specified, only return instances on frames in range.
                If None, return all instances for given track.
        Returns:
            list of `Instance` objects
        """

        frame_range = range(*frame_range) if type(frame_range) == tuple else frame_range

        def does_track_match(inst, tr, labeled_frame):
            match = False
            if type(tr) == Track and inst.track is tr:
                match = True
            elif (type(tr) == int and labeled_frame.instances.index(inst) == tr
                    and inst.track is None):
                match = True
            return match

        track_frame_inst = [(lf, instance)
                            for lf in self.find(video)
                            for instance in lf.instances
                            if does_track_match(instance, track, lf)
                                and (frame_range is None or lf.frame_idx in frame_range)]
        return track_frame_inst


    def find_track_instances(self, *args, **kwargs) -> List[Instance]:
        return [inst for lf, inst in self.find_track_occupancy(*args, **kwargs)]

    @property
    def all_instances(self):
        return list(self.instances())

    @property
    def user_instances(self):
        return [inst for inst in self.all_instances if type(inst) == Instance]

    def instances(self, video: Video = None, skeleton: Skeleton = None):
        """ Iterate through all instances in the labels, optionally with filters.

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

    def __setitem__(self, index, value: LabeledFrame):
        # TODO: Maybe we should remove this method altogether?
        self.labeled_frames.__setitem__(index, value)
        self._update_containers(value)

    def insert(self, index, value: LabeledFrame):
        if value in self or (value.video, value.frame_idx) in self:
            return

        self.labeled_frames.insert(index, value)
        self._update_containers(value)

    def append(self, value: LabeledFrame):
        self.insert(len(self) + 1, value)

    def __delitem__(self, key):
        self.labeled_frames.remove(self.labeled_frames[key])

    def remove(self, value: LabeledFrame):
        self.labeled_frames.remove(value)
        self._lf_by_video[new_label.video].remove(value)
        del self._frame_idx_map[new_label.video][value.frame_idx]

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

        # Delete video
        self.videos.remove(video)

        # Remove from caches
        if video in self._lf_by_video:
            del self._lf_by_video[video]
        if video in self._frame_idx_map:
            del self._frame_idx_map[video]

    def add_negative_anchor(self, video:Video, frame_idx: int, where: tuple):
        """Adds a location for a negative training sample.

        Args:
            video: the `Video` for this negative sample
            frame_idx: frame index
            where: (x, y)
        """
        if video not in self.negative_anchors:
            self.negative_anchors[video] = []
        self.negative_anchors[video].append((frame_idx, *where))

    def get_video_suggestions(self, video:Video) -> list:
        """
        Returns the list of suggested frames for the specified video
        or suggestions for all videos (if no video specified).
        """
        return self.suggestions.get(video, list())

    def get_suggestions(self) -> list:
        """Return all suggestions as a list of (video, frame) tuples."""
        suggestion_list = [(video, frame_idx)
            for video in self.videos
            for frame_idx in self.get_video_suggestions(video)
            ]
        return suggestion_list

    def get_next_suggestion(self, video, frame_idx, seek_direction=1) -> list:
        """Returns a (video, frame_idx) tuple."""
        # make sure we have valid seek_direction
        if seek_direction not in (-1, 1): return (None, None)
        # make sure the video belongs to this Labels object
        if video not in self.videos: return (None, None)

        all_suggestions = self.get_suggestions()

        # If we're currently on a suggestion, then follow order of list
        if (video, frame_idx) in all_suggestions:
            suggestion_idx = all_suggestions.index((video, frame_idx))
            new_idx = (suggestion_idx+seek_direction)%len(all_suggestions)
            video, frame_suggestion = all_suggestions[new_idx]

        # Otherwise, find the prev/next suggestion sorted by frame order
        else:
            # look for next (or previous) suggestion in current video
            if seek_direction == 1:
                frame_suggestion = min((i for i in self.get_video_suggestions(video) if i > frame_idx), default=None)
            else:
                frame_suggestion = max((i for i in self.get_video_suggestions(video) if i < frame_idx), default=None)
            if frame_suggestion is not None: return (video, frame_suggestion)
            # if we didn't find suggestion in current video,
            # then we want earliest frame in next video with suggestions
            next_video_idx = (self.videos.index(video) + seek_direction) % len(self.videos)
            video = self.videos[next_video_idx]
            if seek_direction == 1:
                frame_suggestion = min((i for i in self.get_video_suggestions(video)), default=None)
            else:
                frame_suggestion = max((i for i in self.get_video_suggestions(video)), default=None)
        return (video, frame_suggestion)

    def set_suggestions(self, suggestions:Dict[Video, list]):
        """Sets the suggested frames."""
        self.suggestions = suggestions

    def extend_from(self, new_frames):
        """Merge data from another Labels object or list of LabeledFrames into self.

        Arg:
            new_frames: the object from which to copy data
        Returns:
            bool, True if we added frames, False otherwise
        """
        # allow either Labels or list of LabeledFrames
        if isinstance(new_frames, Labels): new_frames = new_frames.labeled_frames
        # return if this isn't non-empty list of labeled frames
        if not isinstance(new_frames, list) or len(new_frames) == 0: return False
        if not isinstance(new_frames[0], LabeledFrame): return False
        # copy the labeled frames
        self.labeled_frames.extend(new_frames)
        # merge labeled frames for the same video/frame idx
        self.merge_matching_frames()
        # update videos/skeletons/nodes/etc using all the labeled frames now present
        self.__attrs_post_init__()
        return True

    def merge_matching_frames(self, video=None):
        """
        Combine all instances from LabeledFrames that have same frame_idx.

        Args:
            video (optional): combine for this video; if None, do all videos
        Returns:
            none
        """
        if video is None:
            for vid in {lf.video for lf in self.labeled_frames}:
                self.merge_matching_frames(video=vid)
        else:
            self.labeled_frames = LabeledFrame.merge_frames(self.labeled_frames, video=video)

    def to_dict(self):
        """
        Serialize all labels in the underling list of LabeledFrames to a
        dict structure. This function returns a nested dict structure
        composed entirely of primitive python types. It is used to create
        JSON and HDF5 serialized datasets.

        Returns:
            A dict containing the followings top level keys:
            * version - The version of the dict/json serialization format.
            * skeletons - The skeletons associated with these underlying instances.
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
        self.nodes = list(set(self.nodes).union({node for skeleton in self.skeletons for node in skeleton.nodes}))

        # Register some unstructure hooks since we don't want complete deserialization
        # of video and skeleton objects present in the labels. We will serialize these
        # as references to the above constructed lists to limit redundant data in the
        # json
        label_cattr = cattr.Converter()
        label_cattr.register_unstructure_hook(Skeleton, lambda x: self.skeletons.index(x))
        label_cattr.register_unstructure_hook(Video, lambda x: self.videos.index(x))
        label_cattr.register_unstructure_hook(Node, lambda x: self.nodes.index(x))
        label_cattr.register_unstructure_hook(Track, lambda x: self.tracks.index(x))

        idx_to_node = {i: self.nodes[i] for i in range(len(self.nodes))}

        skeleton_cattr = Skeleton.make_cattr(idx_to_node)

        # Serialize the skeletons, videos, and labels
        dicts = {
            'version': LABELS_JSON_FILE_VERSION,
            'skeletons': skeleton_cattr.unstructure(self.skeletons),
            'nodes': cattr.unstructure(self.nodes),
            'videos': Video.cattr().unstructure(self.videos),
            'labels': label_cattr.unstructure(self.labeled_frames),
            'tracks': cattr.unstructure(self.tracks),
            'suggestions': label_cattr.unstructure(self.suggestions),
            'negative_anchors': label_cattr.unstructure(self.negative_anchors)
         }

        return dicts

    def to_json(self):
        """
        Serialize all labels in the underling list of LabeledFrame(s) to a
        JSON structured string.

        Returns:
            The JSON representaiton of the string.
        """

        # Unstructure the data into dicts and dump to JSON.
        return json.dumps(self.to_dict())

    @staticmethod
    def save_json(labels: 'Labels', filename: str,
                  compress: bool = False,
                  save_frame_data: bool = False,
                  frame_data_format: str = 'png'):
        """
        Save a Labels instance to a JSON format.

        Args:
            labels: The labels dataset to save.
            filename: The filename to save the data to.
            compress: Should the data be zip compressed or not? If True, the JSON will be
            compressed using Python's shutil.make_archive command into a PKZIP zip file. If
            compress is True then filename will have a .zip appended to it.
            save_frame_data: Whether to save the image data for each frame as well. For each
            video in the dataset, all frames that have labels will be stored as an imgstore
            dataset. If save_frame_data is True then compress will be forced to True since
            the archive must contain both the JSON data and image data stored in ImgStores.
            frame_data_format: If save_frame_data is True, then this argument is used to set
            the data format to use when writing frame data to ImgStore objects. Supported
            formats should be:

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

             Note: 'h264/mkv' and 'avc1/mp4' require separate installation of these codecs
             on your system. They are excluded from sLEAP because of their GPL license.

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
                new_videos = labels.save_frame_data_imgstore(output_dir=tmp_dir, format=frame_data_format)

                # Convert to a dict, not JSON yet, because we need to patch up the videos
                d = labels.to_dict()
                d['videos'] = Video.cattr().unstructure(new_videos)

                # We can't call Labels.to_json, so we need to do this here. Not as clean as I
                # would like.
                json_str = json.dumps(d)
            else:
                json_str = labels.to_json()

            if compress or save_frame_data:

                # Ensure that filename ends with .json
                # shutil will append .zip
                filename = re.sub("(\.json)?(\.zip)?$", ".json", filename)

                # Write the json to the tmp directory, we will zip it up with the frame data.
                with open(os.path.join(tmp_dir, os.path.basename(filename)), 'w') as file:
                    file.write(json_str)

                # Create the archive
                shutil.make_archive(base_name=filename, root_dir=tmp_dir, format='zip')

            # If the user doesn't want to compress, then just write the json to the filename
            else:
                with open(filename, 'w') as file:
                    file.write(json_str)

    @classmethod
    def from_json(cls, data: Union[str, dict], match_to: Optional['Labels'] = None) -> 'Labels':

        # Parse the json string if needed.
        if data is str:
            dicts = json.loads(data)
        else:
            dicts = data

        dicts['tracks'] = dicts.get('tracks', []) # don't break if json doesn't include tracks

        # First, deserialize the skeletons, videos, and nodes lists.
        # The labels reference these so we will need them while deserializing.
        nodes = cattr.structure(dicts['nodes'], List[Node])

        idx_to_node = {i:nodes[i] for i in range(len(nodes))}
        skeletons = Skeleton.make_cattr(idx_to_node).structure(dicts['skeletons'], List[Skeleton])
        videos = Video.cattr().structure(dicts['videos'], List[Video])
        tracks = cattr.structure(dicts['tracks'], List[Track])

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
                    weak_match = vid.filename.split("/")[-3:] == old_vid.filename.split("/")[-3:]
                    if vid.filename == old_vid.filename or weak_match:
                        # use video from match
                        videos[idx] = old_vid
                        break

        if "suggestions" in dicts:
            suggestions_cattr = cattr.Converter()
            suggestions_cattr.register_structure_hook(Video, lambda x,type: videos[int(x)])
            suggestions = suggestions_cattr.structure(dicts['suggestions'], Dict[Video, List])
        else:
            suggestions = dict()

        if "negative_anchors" in dicts:
            negative_anchors_cattr = cattr.Converter()
            negative_anchors_cattr.register_structure_hook(Video, lambda x,type: videos[int(x)])
            negative_anchors = negative_anchors_cattr.structure(dicts['negative_anchors'], Dict[Video, List])
        else:
            negative_anchors = dict()

        label_cattr = cattr.Converter()
        label_cattr.register_structure_hook(Skeleton, lambda x,type: skeletons[x])
        label_cattr.register_structure_hook(Video, lambda x,type: videos[x])
        label_cattr.register_structure_hook(Node, lambda x,type: x if isinstance(x,Node) else nodes[int(x)])
        label_cattr.register_structure_hook(Track, lambda x, type: None if x is None else tracks[x])

        def structure_points(x, type):
            if 'score' in x.keys():
                return cattr.structure(x, PredictedPoint)
            else:
                return cattr.structure(x, Point)

        label_cattr.register_structure_hook(Union[Point, PredictedPoint], structure_points)

        def structure_instances_list(x, type):
            inst_list = []
            for inst_data in x:
                if 'score' in inst_data.keys():
                    inst = label_cattr.structure(inst_data, PredictedInstance)
                else:
                    inst = label_cattr.structure(inst_data, Instance)
                inst_list.append(inst)
            return inst_list

        label_cattr.register_structure_hook(Union[List[Instance], List[PredictedInstance]],
                                            structure_instances_list)
        label_cattr.register_structure_hook(ForwardRef('PredictedInstance'), lambda x,type: label_cattr.structure(x, PredictedInstance))
        labels = label_cattr.structure(dicts['labels'], List[LabeledFrame])

        return cls(labeled_frames=labels,
                    videos=videos,
                    skeletons=skeletons,
                    nodes=nodes,
                    suggestions=suggestions,
                    negative_anchors=negative_anchors)

    @classmethod
    def load_json(cls, filename: str,
                  video_callback=None,
                  match_to: Optional['Labels'] = None):

        # Check if the file is a zipfile for not.
        if zipfile.is_zipfile(filename):

            # Make a tmpdir, located in the directory that the file exists, to unzip
            # its contents.
            tmp_dir = os.path.join(os.path.dirname(filename),
                                   f"tmp_{os.path.basename(filename)}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                os.mkdir(tmp_dir)
            except FileExistsError:
                pass

            #tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(filename))

            try:

                # Register a cleanup routine that deletes the tmpdir on program exit
                # if something goes wrong. The True is for ignore_errors
                atexit.register(shutil.rmtree, tmp_dir, True)

                # Uncompress the data into the directory
                shutil.unpack_archive(filename, extract_dir=tmp_dir)

                # We can now open the JSON file, save the zip file and
                # replace file with the first JSON file we find in the archive.
                json_files = [os.path.join(tmp_dir, file) for file in os.listdir(tmp_dir) if file.endswith(".json")]

                if len(json_files) == 0:
                    raise ValueError(f"No JSON file found inside {filename}. Are you sure this is a valid sLEAP dataset.")

                filename = json_files[0]

            except Exception as ex:
                # If we had problems, delete the temp directory and reraise the exception.
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        # Open and parse the JSON in filename
        with open(filename, 'r') as file:

            # FIXME: Peek into the json to see if there is version string.
            # We do this to tell apart old JSON data from leap_dev vs the
            # newer format for sLEAP.
            json_str = file.read()
            dicts = json.loads(json_str)

            # If we have a version number, then it is new sLEAP format
            if "version" in dicts:

                # Cache the working directory.
                cwd = os.getcwd()

                # Use the callback if given to handle missing videos
                if callable(video_callback):
                    video_callback(dicts["videos"])

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
                    raise # Re-raise.
                finally:
                    os.chdir(cwd)  # Make sure to change back if we have problems.

                return labels

            else:
                return load_labels_json_old(data_path=filename, parsed_json=dicts)

    def save_hdf5(self, filename: str, save_frame_data: bool = True):
        """
        Serialize the labels dataset to an HDF5 file.

        Args:
            filename: The file to serialize the dataset to.
            save_frame_data: Whether to save the image frame data for any
            labeled frame as well. This is useful for uploading the HDF5 for
            model training when video files are to large to move. This will only
            save video frames that have some labeled instances.

        Returns:
            None
        """

        # Unstructure this labels dataset to a bunch of dicts, same as we do for
        # JSON serialization.
        d = self.to_dict()

        # Delete the file if it exists, we want to start from scratch since
        # h5py truncates the file which seems to not actually delete data
        # from the file.
        if os.path.exists(filename):
            os.unlink(filename)

        with h5.File(filename, 'w') as f:

            # Save the skeletons
            #Skeleton.save_all_hdf5(filename=f, skeletons=self.skeletons)

            # Save the frame data for the videos. For each video, we will
            # save a dataset that contains only the frame data that has been
            # labelled.
            if save_frame_data:

                #
                # # All videos data will be put in the videos group
                # if 'frames' not in f:
                #     frames_group = f.create_group('frames', track_order=True)
                # else:
                #     frames_group = f.require_group('frames')
                self.save_frame_data_imgstore()

                #
                # dset = f.create_dataset(f"/frames/{v_idx}",
                #                         data=v.get_frames(frame_idxs),
                #                         compression="gzip")
                #
                # # Write the dataset to JSON string, then store it in a string
                # # attribute
                # dset.attrs[f"video_json"] = np.string_(json.dumps(d['videos'][v_idx]))

            # Save the instance level data
            Instance.save_hdf5(file=f, instances=self.all_instances)

    def save_frame_data_imgstore(self, output_dir: str = './', format: str = 'png'):
        """
        Write all labeled frames from all videos to a collection of imgstore datasets.
        This only writes frames that have been labeled. Videos without any labeled frames
        will be included as empty imgstores.

        Args:
            output_dir:
            format: The image format to use for the data. png for lossless, jpg for lossy.
            Other imgstore formats will probably work as well but have not been tested.

        Returns:
            A list of ImgStoreVideo objects that represent the stored frames.
        """
        # For each label
        imgstore_vids = []
        for v_idx, v in enumerate(self.videos):
            frame_nums = [f.frame_idx for f in self.labeled_frames if v == f.video]

            frames_filename = os.path.join(output_dir, f'frame_data_vid{v_idx}')
            vid = v.to_imgstore(path=frames_filename, frame_numbers=frame_nums, format=format)

            # Close the video for now
            vid.close()

            imgstore_vids.append(vid)

        return imgstore_vids


    @staticmethod
    def _unwrap_mat_scalar(a):
        if a.shape == (1,):
            return Labels._unwrap_mat_scalar(a[0])
        else:
            return a

    @staticmethod
    def _unwrap_mat_array(a):
        b = a[0][0]
        c = [Labels._unwrap_mat_scalar(x) for x in b]
        return c

    @classmethod
    def load_mat(cls, filename):
        mat_contents = sio.loadmat(filename)

        box_path = Labels._unwrap_mat_scalar(mat_contents["boxPath"])

        # If the video file isn't found, try in the same dir as the mat file
        if not os.path.exists(box_path):
            file_dir = os.path.dirname(filename)
            box_path_name = box_path.split("\\")[-1] # assume windows path
            box_path = os.path.join(file_dir, box_path_name)

        if os.path.exists(box_path):
            vid = Video.from_hdf5(dataset="box", filename=box_path, input_format="channels_first")
        else:
            vid = None

        # TODO: prompt user to locate video

        nodes_ = mat_contents["skeleton"]["nodes"]
        edges_ = mat_contents["skeleton"]["edges"]
        points_ = mat_contents["positions"]

        edges_ = edges_ - 1 # convert matlab 1-indexing to python 0-indexing

        nodes = Labels._unwrap_mat_array(nodes_)
        edges = Labels._unwrap_mat_array(edges_)

        nodes = list(map(str, nodes)) # convert np._str to str

        sk = Skeleton(name=filename)
        sk.add_nodes(nodes)
        for edge in edges:
            sk.add_edge(source=nodes[edge[0]], destination=nodes[edge[1]])

        labeled_frames = []
        node_count, _, frame_count = points_.shape

        for i in range(frame_count):
            new_inst = Instance(skeleton = sk)
            for node_idx, node in enumerate(nodes):
                x = points_[node_idx][0][i]
                y = points_[node_idx][1][i]
                new_inst[node] = Point(x, y)
            new_inst.drop_nan_points()
            if len(new_inst.points()):
                new_frame = LabeledFrame(video=vid, frame_idx=i)
                new_frame.instances = new_inst,
                labeled_frames.append(new_frame)

        labels = cls(labeled_frames=labeled_frames, videos=[vid], skeletons=[sk])

        return labels

    @classmethod
    def load_deeplabcut_csv(cls, filename):

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

        data = pd.read_csv(filename, header=[1,2])

        # Create the skeleton from the list of nodes in the csv file
        # Note that DeepLabCut doesn't have edges, so these will have to be added by user later
        node_names = [n[0] for n in list(data)[1::2]]

        skeleton = Skeleton()
        skeleton.add_nodes(node_names)

        # Create an imagestore `Video` object from frame images.
        # This may not be ideal for large projects, since we're reading in
        # each image and then writing it out in a new directory.

        img_files = data.ix[:,0] # get list of all images

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
                x, y = data[(node, 'x')][i], data[(node, 'y')][i]
                instance_points[node] = Point(x, y)
            # create instance with points (we can assume there's only one instance per frame)
            instance = Instance(skeleton=skeleton, points=instance_points)
            # create labeledframe and add it to list
            label = LabeledFrame(video=video, frame_idx=i, instances=[instance])
            labels.append(label)

        return cls(labels)

    @classmethod
    def make_video_callback(cls, search_paths=None):
        search_paths = search_paths or []
        def video_callback(video_list, new_paths=search_paths):
            # Check each video
            for video_item in video_list:
                if "backend" in video_item and "filename" in video_item["backend"]:
                    current_filename = video_item["backend"]["filename"]
                    # check if we can find video
                    if not os.path.exists(current_filename):
                        is_found = False

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
                                is_found = True
                                break
        return video_callback

    @classmethod
    def make_gui_video_callback(cls, search_paths):
        search_paths = search_paths or []
        def gui_video_callback(video_list, new_paths=search_paths):
            import os
            from PySide2.QtWidgets import QFileDialog, QMessageBox

            has_shown_prompt = False # have we already alerted user about missing files?

            # Check each video
            for video_item in video_list:
                if "backend" in video_item and "filename" in video_item["backend"]:
                    current_filename = video_item["backend"]["filename"]
                    # check if we can find video
                    if not os.path.exists(current_filename):
                        is_found = False

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
                                is_found = True
                                break

                        # if we found this file, then move on to the next file
                        if is_found: continue

                        # Since we couldn't find the file on our own, prompt the user.

                        if not has_shown_prompt:
                            QMessageBox(text=f"We're unable to locate one or more video files for this project. Please locate {current_filename}.").exec_()
                            has_shown_prompt = True

                        current_root, current_ext = os.path.splitext(current_basename)
                        caption = f"Please locate {current_basename}..."
                        filters = [f"{current_root} file (*{current_ext})", "Any File (*.*)"]
                        dir = None if len(new_paths) == 0 else new_paths[-1]
                        new_filename, _ = QFileDialog.getOpenFileName(None, dir=dir, caption=caption, filter=";;".join(filters))
                        # if we got an answer, then update filename for video
                        if len(new_filename):
                            video_item["backend"]["filename"] = new_filename
                            # keep track of the directory chosen by user
                            new_paths.append(os.path.dirname(new_filename))
        return gui_video_callback


def load_labels_json_old(data_path: str, parsed_json: dict = None,
                         adjust_matlab_indexing: bool = True,
                         fix_rel_paths: bool = True) -> Labels:
    """
    Simple utitlity code to load data from Talmo's old JSON format into newer
    Labels object.

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
    instances = pd.DataFrame(data["instances"])
    points = pd.DataFrame(data["points"])
    predicted_instances = pd.DataFrame(data["predicted_instances"])
    predicted_points = pd.DataFrame(data["predicted_points"])

    if adjust_matlab_indexing:
        instances.frameIdx -= 1
        points.frameIdx -= 1
        predicted_instances.frameIdx -= 1
        predicted_points.frameIdx -= 1

        points.node -= 1
        predicted_points.node -= 1

        points.x -= 1
        predicted_points.x -= 1

        points.y -= 1
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

    # A function to get all the instances for a particular video frame
    def get_frame_instances(video_id, frame_idx):
        is_in_frame = (points["videoId"] == video_id) & (points["frameIdx"] == frame_idx)
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (points["instanceId"] == instance_id)
            instance_points = {data["skeleton"]["nodeNames"][n]: Point(x, y, visible=v) for x, y, n, v in
                               zip(*[points[k][is_instance] for k in ["x", "y", "node", "visible"]])}

            instance = Instance(skeleton=skeleton, points=instance_points)
            instances.append(instance)

        return instances

    # Get the unique labeled frames and construct a list of LabeledFrame objects for them.
    frame_keys = list({(videoId, frameIdx) for videoId, frameIdx in zip(points["videoId"], points["frameIdx"])})
    frame_keys.sort()
    labels = []
    for videoId, frameIdx in frame_keys:
        label = LabeledFrame(video=video_objects[videoId], frame_idx=frameIdx,
                             instances = get_frame_instances(videoId, frameIdx))
        labels.append(label)

    return Labels(labels)

"""A LEAP Dataset represents annotated (labeled) video data.

A LEAP Dataset stores almost all data required for training of a model.
This includes, raw video frame data, labelled instances of skeleton _points,
confidence maps, part affinity fields, and skeleton data. A LEAP :class:`.Dataset`
is a high level API to these data structures that abstracts away their underlying
storage format.

"""

import os
import attr
import cattr
import json
import numpy as np

from collections import MutableSequence
from typing import List, Dict, Union

import pandas as pd

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point
from sleap.io.video import Video


@attr.s(auto_attribs=True)
class LabeledFrame:
    video: Video = attr.ib()
    frame_idx: int = attr.ib(converter=int)
    instances: List[Instance] = attr.ib(default=attr.Factory(list))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances.__getitem__(index)

    def index(self, value: Instance):
        return self.instances.index(value)

    def __delitem__(self, index):
        self.instances.__delitem__(index)

    def insert(self, index, value: Instance):
        self.instances.insert(index, value)

    def __setitem__(self, index, value: Instance):
        self.instances.__setitem__(index, value)


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
    """

    labeled_frames: List[LabeledFrame] = attr.ib(default=attr.Factory(list))
    videos: List[Video] = attr.ib(default=attr.Factory(list))
    skeletons: List[Skeleton] = attr.ib(default=attr.Factory(list))

    def __attrs_post_init__(self):

        # Add any videos that are present in the labels but
        # missing from the video list
        self.videos = self.videos + list({label.video for label in self.labels})

        # Ditto for skeletons
        self.skeletons = self.skeletons + list({instance.skeleton for label in self.labels for instance in label.instances})

    # Below are convenience methods for working with Labels as list.
    # Maybe we should just inherit from list? Maybe this class shouldn't
    # exists since it is just a list really with some class methods. I
    # think more stuff might appear in this class later down the line
    # though.

    @property
    def labels(self):
        """ Alias for labeled_frames """
        return self.labeled_frames

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
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Video) and isinstance(item[1], int):
            return self.find_first(*item) is not None

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.labels.__getitem__(index)

        elif isinstance(index, Video):
            if index not in self.videos:
                raise IndexError("Video not found in labels.")
            return self.find(video=index)

        elif isinstance(index, tuple) and len(index) == 2 and isinstance(index[0], Video) and isinstance(index[1], int):
            if index[0] not in self.videos:
                raise IndexError("Video not found in labels.")

            _hit = self.find_first(video=index[0], frame_idx=index[1])

            if _hit is None:
                raise IndexError(f"No label found for specified video at frame {index[1]}.")
                
            return _hit

        else:
            raise IndexError("Invalid label indexing arguments.")

    def find(self, video: Video, frame_idx: int = None) -> List[LabeledFrame]:
        """ Search for labeled frames given video and/or frame index. 
        
        Args:
            video: a `Video` instance that is associated with the labeled frames
            frame_idx: an integer specifying the frame index within the video

        Returns:
            List of `LabeledFrame`s that match the criteria. Empty if no matches found.

        """

        if frame_idx:
            return [label for label in self.labels if label.video == video and label.frame_idx == frame_idx]
        else:
            return [label for label in self.labels if label.video == video]

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

    def _update_containers(self, new_label: LabeledFrame):
        """ Ensure that top-level containers are kept updated with new 
        instances of objects that come along with new labels. """

        if new_label.video not in self.videos:
            self.videos.append(new_label.video)

        for skeleton in {instance.skeleton for instance in new_label}:
            if skeleton not in self.skeletons:
                self.skeletons.append(skeleton)

    def __setitem__(self, index, value: LabeledFrame):
        self.labeled_frames.__setitem__(index, value)
        self._update_containers(value)

    def insert(self, index, value: LabeledFrame):
        self.labeled_frames.insert(index, value)
        self._update_containers(value)

    def append(self, value: LabeledFrame):
        self.insert(len(self) + 1, value)

    def __delitem__(self, index):
        self.labeled_frames.__delitem__(index)

    def to_json(self):
        """
        Serialize all labels in the underling list of LabeledFrame(s) to a
        JSON structured string.

        Returns:
            The JSON representaiton of the string.
        """

        # Register some unstructure hooks since we don't want complete deserialization
        # of video and skeleton objects present in the labels. We will serialize these
        # as references to the above constructed lists to limit redundant data in the
        # json
        label_cattr = cattr.Converter()
        # label_cattr.register_unstructure_hook(Skeleton, lambda x: skeletons.index(x))
        label_cattr.register_unstructure_hook(Skeleton, lambda x: self.skeletons.index(x))
        label_cattr.register_unstructure_hook(Video, lambda x: self.videos.index(x))

        # Serialize the skeletons, videos, and labels
        dicts = {
            'version': LABELS_JSON_FILE_VERSION,
            'skeletons': Skeleton.make_cattr().unstructure(self.skeletons),
            'videos': cattr.unstructure(self.videos),
            'labels': label_cattr.unstructure(self.labeled_frames)
         }

        return json.dumps(dicts)

    @staticmethod
    def save_json(labels: 'Labels', filename: str):
        json_str = labels.to_json()

        with open(filename, 'w') as file:
            file.write(json_str)

    @classmethod
    def from_json(cls, data: Union[str, dict]):

        # Parse the json string if needed.
        if data is str:
            dicts = json.loads(data)
        else:
            dicts = data

        # First, deserialize the skeleton and videos lists, the labels reference these
        # so we will need them while deserializing.
        skeletons = Skeleton.make_cattr().structure(dicts['skeletons'], List[Skeleton])

        videos = Skeleton.make_cattr().structure(dicts['videos'], List[Video])

        label_cattr = cattr.Converter()
        label_cattr.register_structure_hook(Skeleton, lambda x,type: skeletons[x])
        label_cattr.register_structure_hook(Video, lambda x,type: videos[x])
        labels = label_cattr.structure(dicts['labels'], List[LabeledFrame])

        return cls(labeled_frames=labels, videos=videos, skeletons=skeletons)

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, 'r') as file:

            # FIXME: Peek into the json to see if there is version string.
            # I do this to tell apart old JSON data from leap_dev vs the
            # newer format for sLEAP.
            json_str = file.read()
            dicts = json.loads(json_str)

            # If we have a version number, then it is new sLEAP format
            if "version" in dicts:

                # Cache the working directory.
                cwd = os.getcwd()

                # Try to load the labels file.
                try:
                    labels = Labels.from_json(dicts)
                except FileNotFoundError:

                    # FIXME: We are going to the labels JSON that has references to
                    # video files. Lets change directory to the dirname of the json file
                    # so that relative paths will be from this director. Maybe
                    # it is better to feed the dataset dirname all the way down to
                    # the Video object. This seems like less coupling between classes
                    # though.
                    if os.path.dirname(filename) != "":
                        os.chdir(os.path.dirname(filename))

                    # Try again
                    labels = Labels.from_json(dicts)

                except Exception as ex:
                    # Ok, we give up, where the hell are these videos!
                    raise ex # Re-raise.
                finally:
                    os.chdir(cwd)  # Make sure to change back if we have problems.

                return labels

            else:
                return load_labels_json_old(data_path=filename, parsed_json=dicts)


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
    for (src, dst) in edges:
        skeleton.add_edge(skeleton.nodes[src], skeleton.nodes[dst])

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
            vid = Video.from_hdf5(file=videos.at[i, "filepath"], dataset=videos.at[i, "dataset"])

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
            instance_points = {skeleton.nodes[n]: Point(x, y, visible=v) for x, y, n, v in
                               zip(*[points[k][is_instance] for k in ["x", "y", "node", "visible"]])}

            instance = Instance(skeleton=skeleton, points=instance_points)
            instances.append(instance)

        return instances

    # Get the unique labeled frames and construct a list of LabeledFrame objects for them.
    frame_keys = {(videoId, frameIdx) for videoId, frameIdx in zip(points["videoId"], points["frameIdx"])}
    labels = []
    for videoId, frameIdx in frame_keys:
        label = LabeledFrame(video=video_objects[videoId], frame_idx=frameIdx,
                             instances = get_frame_instances(videoId, frameIdx))
        labels.append(label)

    return Labels(labels)


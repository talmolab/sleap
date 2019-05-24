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
import scipy.io as sio

from collections import MutableSequence
from typing import List, Union

import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, Point, LabeledFrame, \
    Track, PredictedPoint, PredictedInstance
from sleap.io.video import Video

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
        tracks: A list of tracks that isntances can belong to.
    """

    labeled_frames: List[LabeledFrame] = attr.ib(default=attr.Factory(list))
    videos: List[Video] = attr.ib(default=attr.Factory(list))
    skeletons: List[Skeleton] = attr.ib(default=attr.Factory(list))
    nodes: List[Node] = attr.ib(default=attr.Factory(list))
    tracks: List[Track] = attr.ib(default=attr.Factory(list))

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

        # Ditto for tracks, a pattern is emerging here
        self.tracks = list(set(self.tracks).union({instance.track
                                                   for frame in self.labels
                                                   for instance in frame.instances
                                                   if instance.track}))

        # Lets sort the tracks by spawned on and then name
        self.tracks.sort(key=lambda t:(t.spawned_on, t.name))

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

    @property
    def all_instances(self):
        return list(self.instances())

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
        self.labeled_frames.__delitem__(key)

    def remove(self, value: LabeledFrame):
        self.labeled_frames.remove(value)

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


    def to_json(self):
        """
        Serialize all labels in the underling list of LabeledFrame(s) to a
        JSON structured string.

        Returns:
            The JSON representaiton of the string.
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

        idx_to_node = {i:self.nodes[i] for i in range(len(self.nodes))}

        skeleton_cattr = Skeleton.make_cattr(idx_to_node)

        # Serialize the skeletons, videos, and labels
        dicts = {
            'version': LABELS_JSON_FILE_VERSION,
            'skeletons': skeleton_cattr.unstructure(self.skeletons),
            'nodes': cattr.unstructure(self.nodes),
            'videos': cattr.unstructure(self.videos),
            'labels': label_cattr.unstructure(self.labeled_frames),
            'tracks': cattr.unstructure(self.tracks)
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

        # First, deserialize the skeletons, videos, and nodes lists.
        # The labels reference these so we will need them while deserializing.
        nodes = cattr.structure(dicts['nodes'], List[Node])

        idx_to_node = {i:nodes[i] for i in range(len(nodes))}
        skeletons = Skeleton.make_cattr(idx_to_node).structure(dicts['skeletons'], List[Skeleton])
        videos = Skeleton.make_cattr(idx_to_node).structure(dicts['videos'], List[Video])
        tracks = cattr.structure(dicts['tracks'], List[Track])

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
            if 'score' in x[0].keys():
                return label_cattr.structure(x, List[PredictedInstance])
            else:
                return label_cattr.structure(x, List[Instance])

        label_cattr.register_structure_hook(Union[List[Instance], List[PredictedInstance]],
                                            structure_instances_list)
        labels = label_cattr.structure(dicts['labels'], List[LabeledFrame])

#         print("LABELS"); print(labels)
        return cls(labeled_frames=labels, videos=videos, skeletons=skeletons, nodes=nodes)

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
            vid = Video.from_hdf5(dataset="box", file=box_path, input_format="channels_first")
        else:
            vid = None

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


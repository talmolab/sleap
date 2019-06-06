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
import copy
import shutil
import tempfile
import numpy as np
import scipy.io as sio
import h5py as h5

from collections import MutableSequence
from typing import List, Union, Dict

import pandas as pd

from sleap.skeleton import Skeleton, Node
from sleap.instance import Instance, Point, LabeledFrame, \
    Track, PredictedPoint, PredictedInstance
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

    def instance_count(self, video: Video, frame_idx: int) -> int:
        count = 0
        labeled_frame = self.find_first(video, frame_idx)
        if labeled_frame is not None:
            count = len([inst for inst in labeled_frame.instances if type(inst)==Instance])
        return count

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
            'suggestions': label_cattr.unstructure(self.suggestions)
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
                  save_frame_data: bool = False):
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
                new_videos = labels.save_frame_data_imgstore(output_dir=tmp_dir)

                # Convert to a dict, not JSON yet, because we need to patch up the videos
                d = labels.to_dict()
                d['videos'] = Video.cattr().unstructure(new_videos)

                # We can't call Labels.to_json, so we need to do this here. Not as clean as I
                # would like.
                json_str = json.dumps(d)
            else:
                json_str = labels.to_json()

            if compress or save_frame_data:

                # Write the json to the tmp directory, we will zip it up with the frame data.
                with open(os.path.join(tmp_dir, filename), 'w') as file:
                    file.write(json_str)

                # Create the archive
                shutil.make_archive(base_name=filename, root_dir=tmp_dir, format='zip')

            # If the user doesn't want to compress, then just write the json to the filename
            else:
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

        if "suggestions" in dicts:
            suggestions_cattr = cattr.Converter()
            suggestions_cattr.register_structure_hook(Video, lambda x,type: videos[int(x)])
            suggestions = suggestions_cattr.structure(dicts['suggestions'], Dict[Video, List])
        else:
            suggestions = dict()

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

        return cls(labeled_frames=labels, videos=videos, skeletons=skeletons, nodes=nodes, suggestions=suggestions)

    @classmethod
    def load_json(cls, filename: str):

        # Check what if the file iz compressed or not, this will be denoted by
        # the filename ending in .zip.
        if filename.endswith('.zip'):
            pass
            # Uncompress the data into a directory
    
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
            #Skeleton.save_all_hdf5(file=f, skeletons=self.skeletons)

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
            vid = v.to_imgstore(path=frames_filename, frame_numbers=frame_nums, format='png')
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
            vid = Video.from_hdf5(dataset="box", file=box_path, input_format="channels_first")
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


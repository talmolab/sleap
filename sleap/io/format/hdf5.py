"""
Adaptor for reading/writing SLEAP datasets as HDF5 (including `.slp`).

Note that this is not the adaptor for reading/writing the "analysis" HDF5
format.
"""

from sleap.io import format
from . import labels_json

from sleap.instance import (
    PointArray,
    PredictedPointArray,
    Instance,
    PredictedInstance,
    LabeledFrame,
    PredictedPoint,
    Point,
)
from sleap.util import json_loads, json_dumps
from sleap import Labels, Video

import h5py
import numpy as np
import os

from typing import Optional, Callable, List, Text, Union


class LabelsV1Adaptor(format.adaptor.Adaptor):
    FORMAT_ID = 1.2

    # 1.0 points with gridline coordinates, top left corner at (0, 0)
    # 1.1 points with midpixel coordinates, top left corner at (-0.5, -0.5)
    # 1.2 adds track score to read and write functions

    @property
    def handles(self):
        return format.adaptor.SleapObjectType.labels

    @property
    def default_ext(self):
        return "slp"

    @property
    def all_exts(self):
        return ["slp", "h5", "hdf5"]

    @property
    def name(self):
        return "Labels HDF5"

    def can_read_file(self, file: format.filehandle.FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_hdf5:
            return False
        if file.format_id is not None and file.format_id >= 2:
            return False
        if "metadata" not in file.file:
            return False
        return True

    def can_write_filename(self, filename: str):
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    @classmethod
    def read_headers(
        cls,
        file: format.filehandle.FileHandle,
        video_search: Union[Callable, List[Text], None] = None,
        match_to: Optional[Labels] = None,
    ):
        f = file.file

        # Extract the Labels JSON metadata and create Labels object with just this
        # metadata.
        dicts = json_loads(f.require_group("metadata").attrs["json"].tobytes().decode())

        # These items are stored in separate lists because the metadata group got to be
        # too big.
        for key in ("videos", "tracks", "suggestions"):
            hdf5_key = f"{key}_json"
            if hdf5_key in f:
                items = [json_loads(item_json) for item_json in f[hdf5_key]]
                dicts[key] = items

        # Video path "." means the video is saved in same file as labels, so replace
        # these paths.
        for video_item in dicts["videos"]:
            if video_item["backend"]["filename"] == ".":
                video_item["backend"]["filename"] = file.filename

        # Use the video_callback for finding videos with broken paths:

        # 1. Accept single string as video search path
        if isinstance(video_search, str):
            video_search = [video_search]

        # 2. Accept list of strings as video search paths
        if hasattr(video_search, "__iter__"):
            # If the callback is an iterable, then we'll expect it to be a list of
            # strings and build a non-gui callback with those as the search paths.
            search_paths = [
                # os.path.dirname(path) if os.path.isfile(path) else path
                path
                for path in video_search
            ]

            # Make the search function from list of paths
            video_search = Labels.make_video_callback(search_paths)

        # 3. Use the callback function (either given as arg or build from paths)
        if callable(video_search):
            video_search(dicts["videos"])

        # Create the Labels object with the header data we've loaded
        labels = labels_json.LabelsJsonAdaptor.from_json_data(dicts, match_to=match_to)

        return labels

    @classmethod
    def read(
        cls,
        file: format.filehandle.FileHandle,
        video_search: Union[Callable, List[Text], None] = None,
        match_to: Optional[Labels] = None,
        *args,
        **kwargs,
    ):

        f = file.file
        labels = cls.read_headers(file, video_search, match_to)

        format_id = file.format_id

        frames_dset = f["frames"][:]
        instances_dset = f["instances"][:]
        points_dset = f["points"][:]
        pred_points_dset = f["pred_points"][:]

        # Shift the *non-predicted* points since these used to be saved with a gridline
        # coordinate system.
        if (format_id or 0) < 1.1:
            points_dset[:]["x"] -= 0.5
            points_dset[:]["y"] -= 0.5

        # Rather than instantiate a bunch of Point\PredictedPoint objects, we will use
        # inplace numpy recarrays. This will save a lot of time and memory when reading
        # things in.
        points = PointArray(buf=points_dset, shape=len(points_dset))

        pred_points = PredictedPointArray(
            buf=pred_points_dset, shape=len(pred_points_dset)
        )

        # Extend the tracks list with a None track. We will signify this with a -1 in
        # the data which will map to last element of tracks
        tracks = labels.tracks.copy()
        tracks.extend([None])

        # A dict to keep track of instances that have a from_predicted link. The key is
        # the instance and the value is the index of the instance.
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
                    tracking_score=i["tracking_score"]
                    if (format_id is not None and format_id >= 1.2)
                    else 0.0,
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
        labels.update_cache()

        return labels

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: object,
        append: bool = False,
        save_frame_data: bool = False,
        frame_data_format: str = "png",
        all_labeled: bool = False,
        suggested: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):

        labels = source_object

        # Delete the file if it exists, we want to start from scratch since
        # h5py truncates the file which seems to not actually delete data
        # from the file. Don't if we are appending of course.
        if os.path.exists(filename) and not append:
            os.unlink(filename)

        # Serialize all the meta-data to JSON.
        d = labels.to_dict(skip_labels=True)

        if save_frame_data:
            new_videos = labels.save_frame_data_hdf5(
                filename,
                format=frame_data_format,
                user_labeled=True,
                all_labeled=all_labeled,
                suggested=suggested,
                progress_callback=progress_callback,
            )

            # Replace path to video file with "." (which indicates that the
            # video is in the same file as the HDF5 labels dataset).
            # Otherwise, the video paths will break if the HDF5 labels
            # dataset file is moved.
            for vid in new_videos:
                vid.backend.filename = "."

            d["videos"] = Video.cattr().unstructure(new_videos)

        else:
            # Include the source video metadata if this was a package.
            new_videos = []
            for video in labels.videos:
                if hasattr(video.backend, "_source_video"):
                    new_videos.append(video.backend._source_video)
                else:
                    new_videos.append(video)
            d["videos"] = Video.cattr().unstructure(new_videos)

        with h5py.File(filename, "a") as f:

            # Add all the JSON metadata
            meta_group = f.require_group("metadata")

            meta_group.attrs["format_id"] = cls.FORMAT_ID

            # If we are appending and there already exists JSON metadata
            if append and "json" in meta_group.attrs:

                # Otherwise, we need to read the JSON and append to the lists
                old_labels = labels_json.LabelsJsonAdaptor.from_json_data(
                    meta_group.attrs["json"].tobytes().decode()
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
                            # If we have an object that matches, replace the instance
                            # with the one from the new list. This will will make sure
                            # objects on the Instances are the same as those in the
                            # Labels lists.
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

            if not append:
                # These items are stored in separate lists because the metadata
                # group got to be too big.
                for key in ("videos", "tracks", "suggestions"):
                    # Convert for saving in hdf5 dataset
                    data = [np.string_(json_dumps(item)) for item in d[key]]

                    hdf5_key = f"{key}_json"

                    # Save in its own dataset (e.g., videos_json)
                    f.create_dataset(hdf5_key, data=data, maxshape=(None,))

                    # Clear from dict since we don't want to save this in attribute
                    d[key] = []

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
                    ("tracking_score", "f4"),
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

            # If we are appending, we need look inside to see what frame, instance, and
            # point ids we need to start from. This gives us offsets to use.
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

                    # Add this instance to our lookup structure we will need for
                    # from_predicted links
                    instance_to_idx[instance] = instance_id

                    parray = instance.get_points_array(copy=False, full=True)
                    instance_type = type(instance)

                    # Check whether we are working with a PredictedInstance or an
                    # Instance.
                    if instance_type is PredictedInstance:
                        score = instance.score
                        pid = pred_point_id + pred_point_id_offset
                        tracking_score = instance.tracking_score
                    else:
                        score = np.nan
                        pid = point_id + point_id_offset
                        tracking_score = np.nan

                        # Keep track of any from_predicted instance links, we will
                        # insert the correct instance_id in the dataset after we are
                        # done.
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
                        tracking_score,
                    )

                    # If these are predicted points, copy them to the predicted point
                    # array otherwise, use the normal point array
                    if type(parray) is PredictedPointArray:
                        pred_points[
                            pred_point_id : (pred_point_id + len(parray))
                        ] = parray
                        pred_point_id = pred_point_id + len(parray)
                    else:
                        points[point_id : (point_id + len(parray))] = parray
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
                    # If we haven't encountered the from_predicted instance yet then
                    # don't save the link. It's possible for a user to create a regular
                    # instance from a predicted instance and then delete all predicted
                    # instances from the file, but in this case I don’t think there's
                    # any reason to remember which predicted instance the regular
                    # instance came from.
                    pass

            # We pre-allocated our points array with max possible size considering the
            # max skeleton size, drop any unused points.
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

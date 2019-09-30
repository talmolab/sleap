"""
Module for legacy LEAP dataset.
"""
import json
import os
import numpy as np
import pandas as pd

from typing import List

from sleap.util import json_loads
from sleap.io.video import Video

from sleap.instance import (
    LabeledFrame,
    PredictedPoint,
    PredictedInstance,
    Track,
    Point,
    Instance,
)
from sleap.skeleton import Skeleton


def load_predicted_labels_json_old(
    data_path: str,
    parsed_json: dict = None,
    adjust_matlab_indexing: bool = True,
    fix_rel_paths: bool = True,
) -> List[LabeledFrame]:
    """
    Load predicted instances from Talmo's old JSON format.

    Args:
        data_path: The path to the JSON file.
        parsed_json: The parsed json if already loaded, so we can save
            some time if already parsed.
        adjust_matlab_indexing: Whether to adjust indexing from MATLAB.
        fix_rel_paths: Whether to fix paths to videos to absolute paths.

    Returns:
        List of :class:`LabeledFrame` objects.
    """
    if parsed_json is None:
        data = json.loads(open(data_path).read())
    else:
        data = parsed_json

    videos = pd.DataFrame(data["videos"])
    predicted_instances = pd.DataFrame(data["predicted_instances"])
    predicted_points = pd.DataFrame(data["predicted_points"])

    if adjust_matlab_indexing:
        predicted_instances.frameIdx -= 1
        predicted_points.frameIdx -= 1

        predicted_points.node -= 1

        predicted_points.x -= 1

        predicted_points.y -= 1

    skeleton = Skeleton()
    skeleton.add_nodes(data["skeleton"]["nodeNames"])
    edges = data["skeleton"]["edges"]
    if adjust_matlab_indexing:
        edges = np.array(edges) - 1
    for (src_idx, dst_idx) in edges:
        skeleton.add_edge(
            data["skeleton"]["nodeNames"][src_idx],
            data["skeleton"]["nodeNames"][dst_idx],
        )

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
            vid = Video.from_hdf5(
                filename=videos.at[i, "filepath"], dataset=videos.at[i, "dataset"]
            )

        video_objects[videos.at[i, "id"]] = vid

    track_ids = predicted_instances["trackId"].values
    unique_track_ids = np.unique(track_ids)

    spawned_on = {
        track_id: predicted_instances.loc[predicted_instances["trackId"] == track_id][
            "frameIdx"
        ].values[0]
        for track_id in unique_track_ids
    }
    tracks = {
        i: Track(name=str(i), spawned_on=spawned_on[i])
        for i in np.unique(predicted_instances["trackId"].values).tolist()
    }

    # A function to get all the instances for a particular video frame
    def get_frame_predicted_instances(video_id, frame_idx):
        points = predicted_points
        is_in_frame = (points["videoId"] == video_id) & (
            points["frameIdx"] == frame_idx
        )
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (points["instanceId"] == instance_id)
            track_id = predicted_instances.loc[
                predicted_instances["id"] == instance_id
            ]["trackId"].values[0]
            match_score = predicted_instances.loc[
                predicted_instances["id"] == instance_id
            ]["matching_score"].values[0]
            track_score = predicted_instances.loc[
                predicted_instances["id"] == instance_id
            ]["tracking_score"].values[0]
            instance_points = {
                data["skeleton"]["nodeNames"][n]: PredictedPoint(
                    x, y, visible=v, score=confidence
                )
                for x, y, n, v, confidence in zip(
                    *[
                        points[k][is_instance]
                        for k in ["x", "y", "node", "visible", "confidence"]
                    ]
                )
            }

            instance = PredictedInstance(
                skeleton=skeleton,
                points=instance_points,
                track=tracks[track_id],
                score=match_score,
            )
            instances.append(instance)

        return instances

    # Get the unique labeled frames and construct a list of LabeledFrame objects for them.
    frame_keys = list(
        {
            (videoId, frameIdx)
            for videoId, frameIdx in zip(
                predicted_points["videoId"], predicted_points["frameIdx"]
            )
        }
    )
    frame_keys.sort()
    labels = []
    for videoId, frameIdx in frame_keys:
        label = LabeledFrame(
            video=video_objects[videoId],
            frame_idx=frameIdx,
            instances=get_frame_predicted_instances(videoId, frameIdx),
        )
        labels.append(label)

    return labels


def load_labels_json_old(
    data_path: str,
    parsed_json: dict = None,
    adjust_matlab_indexing: bool = True,
    fix_rel_paths: bool = True,
) -> List[LabeledFrame]:
    """
    Load predicted instances from Talmo's old JSON format.

    Args:
        data_path: The path to the JSON file.
        parsed_json: The parsed json if already loaded, so we can save
            some time if already parsed.
        adjust_matlab_indexing: Whether to adjust indexing from MATLAB.
        fix_rel_paths: Whether to fix paths to videos to absolute paths.

    Returns:
        A newly constructed Labels object.
    """
    if parsed_json is None:
        data = json_loads(open(data_path).read())
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
        skeleton.add_edge(
            data["skeleton"]["nodeNames"][src_idx],
            data["skeleton"]["nodeNames"][dst_idx],
        )

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
            vid = Video.from_hdf5(
                filename=videos.at[i, "filepath"], dataset=videos.at[i, "dataset"]
            )

        video_objects[videos.at[i, "id"]] = vid

    # A function to get all the instances for a particular video frame
    def get_frame_instances(video_id, frame_idx):
        """ """
        is_in_frame = (points["videoId"] == video_id) & (
            points["frameIdx"] == frame_idx
        )
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (points["instanceId"] == instance_id)
            instance_points = {
                data["skeleton"]["nodeNames"][n]: Point(x, y, visible=v)
                for x, y, n, v in zip(
                    *[points[k][is_instance] for k in ["x", "y", "node", "visible"]]
                )
            }

            instance = Instance(skeleton=skeleton, points=instance_points)
            instances.append(instance)

        return instances

    # Get the unique labeled frames and construct a list of LabeledFrame objects for them.
    frame_keys = list(
        {
            (videoId, frameIdx)
            for videoId, frameIdx in zip(points["videoId"], points["frameIdx"])
        }
    )
    frame_keys.sort()
    labels = []
    for videoId, frameIdx in frame_keys:
        label = LabeledFrame(
            video=video_objects[videoId],
            frame_idx=frameIdx,
            instances=get_frame_instances(videoId, frameIdx),
        )
        labels.append(label)

    return labels

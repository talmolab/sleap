"""Generate an HDF5 file with track occupancy and point location data.

Ignores tracks that are entirely empty. By default will also ignore
empty frames from the beginning and end of video, although
`--all-frames` argument will make it include empty frames from beginning
of video.

The HDF5 file has these datasets:

* "track_occupancy"    (shape: tracks * frames)
* "tracks"             (shape: frames * nodes * 2 * tracks)
* "track_names"        (shape: tracks)
* "node_names"         (shape: nodes)
* "edge_names"         (shape: nodes - 1)
* "edge_inds"          (shape: nodes - 1)
* "point_scores"       (shape: frames * nodes * tracks)
* "instance_scores"    (shape: frames * tracks)
* "tracking_scores"    (shape: frames * tracks)
* "labels_path":       Path to the source .slp file (if available from GUI context)
* "video_path":        Path to the source :py:class:`Video`.
* "video_ind":         Scalar integer index of the video within the :py:class:`Labels`.
* "provenance":        Dictionary that denotes the origin of the :py:class:`Labels`.

Note: the datasets are stored column-major as expected by MATLAB.
"""

import os
import re
import json
import h5py as h5
import numpy as np

from typing import Any, Dict, List, Tuple, Union

from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap import PredictedInstance


def get_tracks_as_np_strings(labels: Labels) -> List[np.string_]:
    """Get list of track names as `np.string_`."""
    return [np.string_(track.name) for track in labels.tracks]


def get_nodes_as_np_strings(labels: Labels) -> List[np.string_]:
    """Get list of node names as `np.string_`."""
    return [np.string_(node.name) for node in labels.skeletons[0].nodes]


def get_edges_as_np_strings(labels: Labels) -> List[Tuple[np.string_, np.string_]]:
    """Get list of edge names as `np.string_`."""
    return [
        (np.string_(src_name), np.string_(dst_name))
        for (src_name, dst_name) in labels.skeletons[0].edge_names
    ]


def get_occupancy_and_points_matrices(
    labels: Labels, all_frames: bool, video: Video = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds numpy matrices with track occupancy and point location data.

    Note: This function assumes either all instances have tracks or no instances have
    tracks.

    Args:
        labels: The :py:class:`Labels` from which to get data.
        all_frames: If True, then includes zeros so that frame index
            will line up with columns in the output. Otherwise,
            there will only be columns for the frames between the
            first and last frames with labeling data.
        video: The :py:class:`Video` from which to get data. If no `video` is specified,
            then the first video in `source_object` videos list will be used. If there
            are no labeled frames in the `video`, then None will be returned.

    Returns:
        tuple of arrays:

        * occupancy matrix with shape (tracks, frames)
        * point location array with shape (frames, nodes, 2, tracks)
        * point scores array with shape (frames, nodes, tracks)
        * instance scores array with shape (frames, tracks)
        * tracking scores array with shape (frames, tracks)
    """
    # Assumes either all instances have tracks or no instances have tracks
    track_count = len(labels.tracks) or 1
    node_count = len(labels.skeletons[0].nodes)

    # Retrieve frames from current video only
    try:
        if video is None:
            video = labels.videos[0]
    except IndexError:
        print(f"There are no videos in this project. No occupancy matrix to return.")
        return
    labeled_frames = labels.get(video)

    frame_idxs = [lf.frame_idx for lf in labeled_frames]
    frame_idxs.sort()

    try:
        first_frame_idx = 0 if all_frames else frame_idxs[0]

        frame_count = (
            frame_idxs[-1] - first_frame_idx + 1
        )  # count should include unlabeled frames
    except IndexError:
        print(f"No labeled frames in {video.filename}. No occupancy matrix to return.")
        return

    # Desired MATLAB format:
    # "track_occupancy"     tracks * frames
    # "tracks"              frames * nodes * 2 * tracks
    # "track_names"         tracks
    # "point_scores"        frames * nodes * tracks
    # "instance_scores"     frames * tracks
    # "tracking_scores"     frames * tracks

    occupancy_matrix = np.zeros((track_count, frame_count), dtype=np.uint8)
    locations_matrix = np.full(
        (frame_count, node_count, 2, track_count), np.nan, dtype=float
    )
    point_scores = np.full((frame_count, node_count, track_count), np.nan, dtype=float)
    instance_scores = np.full((frame_count, track_count), np.nan, dtype=float)
    tracking_scores = np.full((frame_count, track_count), np.nan, dtype=float)

    # Assumes either all instances have tracks or no instances have tracks
    # Prefer user-labeled instances over predicted instances
    tracks = labels.tracks or [None]  # Comparator in case of project with no tracks
    lfs_instances = list()
    warning_flag = False
    for lf in labeled_frames:
        user_instances = lf.user_instances
        predicted_instances = lf.predicted_instances
        for track in tracks:
            track_instances = list()
            # If a user-instance exists for this track, then use user-instance
            user_track_instances = [
                inst for inst in user_instances if inst.track == track
            ]
            if len(user_track_instances) > 0:
                track_instances = user_track_instances
            else:
                # Otherwise, if a predicted instance exists, then use the predicted
                predicted_track_instances = [
                    inst for inst in predicted_instances if inst.track == track
                ]
                if len(predicted_track_instances) > 0:
                    track_instances = predicted_track_instances

            lfs_instances.extend([(lf, inst) for inst in track_instances])

            # Set warning flag if more than one instances on a track in a single frame
            warning_flag = warning_flag or (
                (track is not None) and (len(track_instances) > 1)
            )

    if warning_flag:
        print(
            "\nWarning! "
            "There are more than one instances per track on a single frame.\n"
        )

    for lf, inst in lfs_instances:
        frame_i = lf.frame_idx - first_frame_idx
        # Assumes either all instances have tracks or no instances have tracks
        if inst.track is None:
            # We could use lf.instances.index(inst) but then we'd need
            # to calculate the number of "tracks" based on the max number of
            # instances in any frame, so for now we'll assume that there's
            # a single instance if we aren't using tracks.
            track_i = 0
        else:
            track_i = labels.tracks.index(inst.track)

        occupancy_matrix[track_i, frame_i] = 1

        locations_matrix[frame_i, ..., track_i] = inst.numpy()
        if type(inst) == PredictedInstance:
            point_scores[frame_i, ..., track_i] = inst.scores
            instance_scores[frame_i, ..., track_i] = inst.score
            tracking_scores[frame_i, ..., track_i] = inst.tracking_score

    return (
        occupancy_matrix,
        locations_matrix,
        point_scores,
        instance_scores,
        tracking_scores,
    )


def remove_empty_tracks_from_matrices(
    track_names: List,
    occupancy_matrix: np.ndarray,
    locations_matrix: np.ndarray,
    point_scores: np.ndarray,
    instance_scores: np.ndarray,
    tracking_scores: np.ndarray,
) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Removes matrix rows/columns for unoccupied tracks.

    Args:
        track_names: List of track names
        occupancy_matrix: 2d numpy matrix, rows correspond to tracks
        locations_matrix: 4d numpy matrix, last index is track
        point_scores: 3d numpy matrix, last index is track
        instance_scores: 2d numpy matrix, last index is track
        tracking_scores: 2d numpy matrix, last index is track

    Returns:
        track_names, occupancy_matrix, locations_matrix, point_scores, instance_scores
        tracking_scores but without the rows/columns corresponding to unoccupied tracks.
    """
    # Make mask with only the occupied tracks
    occupied_track_mask = np.sum(occupancy_matrix, axis=1) > 0

    # Ignore unoccupied tracks
    if np.sum(~occupied_track_mask):

        print(f"ignoring {np.sum(~occupied_track_mask)} empty tracks")

        occupancy_matrix = occupancy_matrix[occupied_track_mask]
        locations_matrix = locations_matrix[..., occupied_track_mask]
        point_scores = point_scores[..., occupied_track_mask]
        instance_scores = instance_scores[..., occupied_track_mask]
        tracking_scores = tracking_scores[..., occupied_track_mask]
        track_names = [
            track_names[i] for i in range(len(track_names)) if occupied_track_mask[i]
        ]

    return (
        track_names,
        occupancy_matrix,
        locations_matrix,
        point_scores,
        instance_scores,
        tracking_scores,
    )


def write_occupancy_file(
    output_path: str, data_dict: Dict[str, Any], transpose: bool = True
):
    """Write HDF5 file with data from given dictionary.

    Args:
        output_path: Path of HDF5 file.
        data_dict: Dictionary with data to save. Keys are dataset names,
            values are the data.
        transpose: If True, then any ndarray in data dictionary will be
            transposed before saving. This is useful for writing files
            that will be imported into MATLAB, which expects data in
            column-major format.

    Returns:
        None
    """

    with h5.File(output_path, "w") as f:
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                print(f"{key}: {val.shape}")

                if transpose:
                    # Transpose since MATLAB expects column-major
                    f.create_dataset(
                        key,
                        data=np.transpose(val),
                        compression="gzip",
                        compression_opts=9,
                    )
                else:
                    f.create_dataset(
                        key, data=val, compression="gzip", compression_opts=9
                    )
            else:
                if isinstance(val, (str, int, type(None))):
                    print(f"{key}: {val}")
                else:
                    print(f"{key}: {len(val)}")
                f.create_dataset(key, data=val)

    print(f"Saved as {output_path}")


def main(
    labels: Labels,
    output_path: str,
    labels_path: str = None,
    all_frames: bool = True,
    video: Video = None,
):
    """Writes HDF5 file with matrices of track occupancy and coordinates.

    Args:
        labels: The :class:`Labels` from which to get data.
        output_path: Path of HDF5 file to create.
        labels_path: Path of `labels` .slp file.
        all_frames: If True, then includes zeros so that frame index
            will line up with columns in the output. Otherwise,
            there will only be columns for the frames between the
            first and last frames with labeling data.
        video: The :py:class:`Video` from which to get data. If no `video` is specified,
            then the first video in `source_object` videos list will be used. If there
            are no labeled frames in the `video`, then no output file will be written.

    Returns:
        None
    """
    track_names = get_tracks_as_np_strings(labels)

    # Export analysis of current video only
    try:
        if video is None:
            video = labels.videos[0]
    except IndexError:
        print(f"There are no videos in this project. Output file will not be written.")
        return

    try:
        (
            occupancy_matrix,
            locations_matrix,
            point_scores,
            instance_scores,
            tracking_scores,
        ) = get_occupancy_and_points_matrices(labels, all_frames, video)
    except TypeError:
        print(
            f"No labeled frames in {video.filename}. "
            "Skipping the analysis for this video."
        )
        return

    (
        track_names,
        occupancy_matrix,
        locations_matrix,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = remove_empty_tracks_from_matrices(
        track_names,
        occupancy_matrix,
        locations_matrix,
        point_scores,
        instance_scores,
        tracking_scores,
    )

    data_dict = dict(
        track_names=track_names,
        node_names=get_nodes_as_np_strings(labels),
        edge_names=get_edges_as_np_strings(labels),
        edge_inds=labels.skeletons[0].edge_inds,
        tracks=locations_matrix,
        track_occupancy=occupancy_matrix,
        point_scores=point_scores,
        instance_scores=instance_scores,
        tracking_scores=tracking_scores,
        labels_path=str(labels_path),  # NoneType cannot be written to hdf5.
        video_path=video.backend.filename,
        video_ind=labels.videos.index(video),
        provenance=json.dumps(labels.provenance),  # dict cannot be written to hdf5.
    )

    write_occupancy_file(output_path, data_dict, transpose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument(
        "--all-frames",
        dest="all_frames",
        action="store_const",
        const=True,
        default=False,
        help="include all frames without predictions",
    )
    args = parser.parse_args()

    video_callback = Labels.make_video_callback([os.path.dirname(args.data_path)])
    labels = Labels.load_file(args.data_path, video_search=video_callback)

    output_path = re.sub("(\\.json(\\.zip)?|\\.h5|\\.slp)$", "", args.data_path)
    output_path = output_path + ".tracking.h5"

    main(labels, output_path=output_path, all_frames=args.all_frames)

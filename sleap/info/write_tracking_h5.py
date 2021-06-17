"""
Generate an HDF5 file with track occupancy and point location data.

Ignores tracks that are entirely empty. By default will also ignore
empty frames from the beginning and end of video, although
`--all-frames` argument will make it include empty frames from beginning
of video.

The HDF5 file has these datasets:

* "track_occupancy"     shape: tracks * frames
* "tracks"              shape: frames * nodes * 2 * tracks
* "track_names"         shape: tracks
* "node_names"         shape: nodes

Note: the datasets are stored column-major as expected by MATLAB.
"""

import os
import re
import h5py as h5
import numpy as np

from typing import Any, Dict, List, Tuple

from sleap.io.dataset import Labels
from sleap import PredictedInstance


def get_tracks_as_np_strings(labels: Labels) -> List[np.string_]:
    """Get list of track names as `np.string_`."""
    return [np.string_(track.name) for track in labels.tracks]


def get_nodes_as_np_strings(labels: Labels) -> List[np.string_]:
    """Get list of node names as `np.string_`."""
    return [np.string_(node.name) for node in labels.skeletons[0].nodes]


def get_occupancy_and_points_matrices(
    labels: Labels, all_frames: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds numpy matrices with track occupancy and point location data.

    Args:
        labels: The :class:`Labels` from which to get data.
        all_frames: If True, then includes zeros so that frame index
            will line up with columns in the output. Otherwise,
            there will only be columns for the frames between the
            first and last frames with labeling data.

    Returns:
        tuple of arrays:

        * occupancy matrix with shape (tracks, frames)
        * point location array with shape (frames, nodes, 2, tracks)
        * point scores array with shape (frames, nodes, tracks)
        * instance scores array with shape (frames, tracks)
        * tracking scores array with shape (frames, tracks)
    """
    track_count = len(labels.tracks) or 1
    node_count = len(labels.skeletons[0].nodes)

    frame_idxs = [lf.frame_idx for lf in labels]
    frame_idxs.sort()

    first_frame_idx = 0 if all_frames else frame_idxs[0]

    frame_count = (
        frame_idxs[-1] - first_frame_idx + 1
    )  # count should include unlabeled frames

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

    for lf, inst in [(lf, inst) for lf in labels for inst in lf.instances]:
        frame_i = lf.frame_idx - first_frame_idx
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
    """
    Removes matrix rows/columns for unoccupied tracks.

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
    """
    Write HDF5 file with data from given dictionary.

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
                print(f"{key}: {len(val)}")
                f.create_dataset(key, data=val)

    print(f"Saved as {output_path}")


def main(labels: Labels, output_path: str, all_frames: bool = True):
    """
    Writes HDF5 file with matrices of track occupancy and coordinates.

    Args:
        labels: The :class:`Labels` from which to get data.
        output_path: Path of HDF5 file to create.
        all_frames: If True, then includes zeros so that frame index
            will line up with columns in the output. Otherwise,
            there will only be columns for the frames between the
            first and last frames with labeling data.

    Returns:
        None
    """
    track_names = get_tracks_as_np_strings(labels)

    (
        occupancy_matrix,
        locations_matrix,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(labels, all_frames)

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
        tracks=locations_matrix,
        track_occupancy=occupancy_matrix,
        point_scores=point_scores,
        instance_scores=instance_scores,
        tracking_scores=tracking_scores,
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

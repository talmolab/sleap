import os

import h5py
import numpy as np

from sleap.info.write_tracking_h5 import (
    get_tracks_as_np_strings,
    get_occupancy_and_points_matrices,
    remove_empty_tracks_from_matrices,
    write_occupancy_file,
)


def test_output_matrices(centered_pair_predictions):

    names = get_tracks_as_np_strings(centered_pair_predictions)
    assert len(names) == 27
    assert isinstance(names[0], np.string_)

    # Remove the first labeled frame
    del centered_pair_predictions[0]
    assert len(centered_pair_predictions) == 1099

    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(centered_pair_predictions, all_frames=False)

    assert occupancy.shape == (27, 1099)
    assert points.shape == (1099, 24, 2, 27)
    assert point_scores.shape == (1099, 24, 27)
    assert instance_scores.shape == (1099, 27)
    assert tracking_scores.shape == (1099, 27)

    # Make sure "all_frames" includes the missing initial frame
    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(centered_pair_predictions, all_frames=True)

    assert occupancy.shape == (27, 1100)
    assert points.shape == (1100, 24, 2, 27)
    assert point_scores.shape == (1100, 24, 27)
    assert instance_scores.shape == (1100, 27)
    assert tracking_scores.shape == (1100, 27)

    # Make sure removing empty tracks doesn't yet change anything
    (
        names,
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = remove_empty_tracks_from_matrices(
        names, occupancy, points, point_scores, instance_scores, tracking_scores
    )

    assert len(names) == 27
    assert occupancy.shape == (27, 1100)
    assert points.shape == (1100, 24, 2, 27)
    assert point_scores.shape == (1100, 24, 27)
    assert instance_scores.shape == (1100, 27)
    assert tracking_scores.shape == (1100, 27)

    # Remove all instances from track 13
    vid = centered_pair_predictions.videos[0]
    track = centered_pair_predictions.tracks[13]
    instances = centered_pair_predictions.find_track_occupancy(vid, track)
    for instance in instances:
        centered_pair_predictions.remove_instance(instance.frame, instance)

    # Make sure that this now remove empty track
    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(centered_pair_predictions, all_frames=True)
    (
        names,
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = remove_empty_tracks_from_matrices(
        names, occupancy, points, point_scores, instance_scores, tracking_scores
    )

    assert len(names) == 26
    assert occupancy.shape == (26, 1100)
    assert points.shape == (1100, 24, 2, 26)
    assert point_scores.shape == (1100, 24, 26)
    assert instance_scores.shape == (1100, 26)
    assert tracking_scores.shape == (1100, 26)


def test_hdf5_saving(tmpdir):
    path = os.path.join(tmpdir, "occupancy.h5")

    x = np.array([[1, 2, 6], [3, 4, 5]])
    data_dict = dict(x=x)

    write_occupancy_file(path, data_dict, transpose=False)

    with h5py.File(path, "r") as f:
        assert f["x"].shape == x.shape


def test_hdf5_tranposed_saving(tmpdir):
    path = os.path.join(tmpdir, "transposed.h5")

    x = np.array([[1, 2, 6], [3, 4, 5]])
    data_dict = dict(x=x)

    write_occupancy_file(path, data_dict, transpose=True)

    with h5py.File(path, "r") as f:
        assert f["x"].shape == np.transpose(x).shape

import os
from pathlib import PurePath
import h5py
import numpy as np

from pathlib import PurePath, Path

from sleap.info.write_tracking_h5 import (
    get_tracks_as_np_strings,
    get_occupancy_and_points_matrices,
    remove_empty_tracks_from_matrices,
    write_occupancy_file,
    get_nodes_as_np_strings,
    get_edges_as_np_strings,
    main,
)
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import Instance


def test_output_matrices(centered_pair_predictions: Labels):

    names = get_tracks_as_np_strings(centered_pair_predictions)
    assert len(names) == 27
    assert isinstance(names[0], np.string_)

    # Check that node names and edges are read correctly
    node_names = [
        n.decode() for n in get_nodes_as_np_strings(centered_pair_predictions)
    ]
    edge_names = [
        (s.decode(), d.decode())
        for (s, d) in get_edges_as_np_strings(centered_pair_predictions)
    ]

    assert node_names[0] == "head"
    assert node_names[1] == "neck"
    assert node_names[2] == "thorax"
    assert node_names[3] == "abdomen"
    assert node_names[4] == "wingL"
    assert node_names[5] == "wingR"
    assert node_names[6] == "forelegL1"
    assert node_names[7] == "forelegL2"
    assert node_names[8] == "forelegL3"
    assert node_names[9] == "forelegR1"
    assert node_names[10] == "forelegR2"
    assert node_names[11] == "forelegR3"
    assert node_names[12] == "midlegL1"
    assert node_names[13] == "midlegL2"
    assert node_names[14] == "midlegL3"
    assert node_names[15] == "midlegR1"
    assert node_names[16] == "midlegR2"
    assert node_names[17] == "midlegR3"

    # Both lines check edge_names are read correctly, but latter is used in bento plugin
    assert edge_names == centered_pair_predictions.skeleton.edge_names
    for (src_node, dst_node) in edge_names:
        assert src_node in node_names
        assert dst_node in node_names

    # Remove the first labeled frame
    centered_pair_predictions.remove_frame(centered_pair_predictions[0])
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


def read_lens_hdf5(filename):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        dset_lens = {}
        for dset_name in dset_names:
            if dset_name in [
                "track_names",
                "node_names",
                "edge_names",
                "edge_inds",
            ]:
                dset_lens[dset_name] = len(f[dset_name])
            else:
                dset_lens[dset_name] = (f[dset_name][:].T).shape
    return dset_lens


def assert_dset_lens(dset_lens, num_tracks, num_frames, num_nodes):
    assert dset_lens["track_names"] == num_tracks
    assert dset_lens["node_names"] == num_nodes
    assert dset_lens["edge_names"] == num_nodes - 1
    assert dset_lens["edge_inds"] == num_nodes - 1
    assert dset_lens["tracks"] == (num_frames, num_nodes, 2, num_tracks)
    assert dset_lens["track_occupancy"] == (num_tracks, num_frames)
    assert dset_lens["point_scores"] == (num_frames, num_nodes, num_tracks)
    assert dset_lens["instance_scores"] == (num_frames, num_tracks)
    assert dset_lens["tracking_scores"] == (num_frames, num_tracks)


def test_hdf5_video_arg(
    centered_pair_predictions: Labels, small_robot_mp4_vid: Video, tmpdir
):

    labels = centered_pair_predictions
    labels.add_video(small_robot_mp4_vid)

    output_paths = []
    for video in labels.videos:
        vn = PurePath(video.filename)
        output_paths.append(PurePath(tmpdir, f"{vn.stem}.analysis.h5"))
        main(
            labels=labels,
            output_path=output_paths[-1],
            all_frames=True,
            video=video,
        )

    # Read hdf5 to ensure shapes are correct: centered_pair_low_quality
    dset_lens = read_lens_hdf5(output_paths[0])
    assert_dset_lens(dset_lens, num_tracks=27, num_frames=1100, num_nodes=24)

    # No file should exist for video with no labeled frames
    assert Path(output_paths[1]).exists() == False

    # Add labeled frames to second video, repeat process
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance(skeleton=labels.skeleton, frame=labeled_frame)
    labels.add_instance(frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)

    main(
        labels=labels,
        output_path=output_paths[1],
        all_frames=True,
        video=video,
    )
    dset_lens = read_lens_hdf5(output_paths[1])
    assert_dset_lens(dset_lens, num_tracks=1, num_frames=1, num_nodes=24)

    # Remove all videos from project and repeat process
    all_videos = list(labels.videos)
    for video in all_videos:
        labels.remove_video(labels.videos[-1])

    assert get_occupancy_and_points_matrices(labels=labels, all_frames=True) is None

    for output_path in output_paths:
        Path(output_path).unlink()
        main(
            labels=labels,
            output_path=output_path,
            all_frames=True,
        )

    # No files should exist for labels with no videos
    assert Path(output_paths[0]).exists() == False
    assert Path(output_paths[1]).exists() == False

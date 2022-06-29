import os
from pathlib import PurePath
import h5py
import json
import numpy as np

from pathlib import PurePath, Path
from typing import List

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
from sleap.instance import Instance, Point, PredictedInstance
from sleap.gui.commands import AddUserInstancesFromPredictions


def test_output_matrices(centered_pair_predictions: Labels, min_labels_robot: Labels):
    def assert_output_matrices_shape(
        num_tracks, num_frames, num_nodes, check_names: bool = False
    ):
        if check_names:
            assert len(names) == num_tracks
        assert occupancy.shape == (num_tracks, num_frames)
        assert points.shape == (num_frames, num_nodes, 2, num_tracks)
        assert point_scores.shape == (num_frames, num_nodes, num_tracks)
        assert instance_scores.shape == (num_frames, num_tracks)
        assert tracking_scores.shape == (num_frames, num_tracks)

    def assert_instance_points(points, inst: Instance, track_idx: int, frame_idx: int):
        instance_points = points[frame_idx, :, :, track_idx]
        for node_idx, _ in enumerate(inst.nodes):
            assert instance_points[node_idx][0] == inst[node_idx].x
            assert instance_points[node_idx][1] == inst[node_idx].y

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
    assert_output_matrices_shape(
        num_tracks=27, num_frames=1099, num_nodes=24, check_names=True
    )

    # Make sure "all_frames" includes the missing initial frame
    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(centered_pair_predictions, all_frames=True)
    assert_output_matrices_shape(
        num_tracks=27, num_frames=1100, num_nodes=24, check_names=True
    )

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
    assert_output_matrices_shape(
        num_tracks=27, num_frames=1100, num_nodes=24, check_names=True
    )

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
    assert_output_matrices_shape(
        num_tracks=26, num_frames=1100, num_nodes=24, check_names=True
    )

    # Create a user-instance from a predicted-instance
    lf = centered_pair_predictions[0]
    user_instance = (
        AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
            copy_instance=lf.predicted_instances[0]
        )
    )
    # Make a minor modification to the user-instance to differentiate
    node_idx = 0
    user_instance[node_idx] = Point(
        x=1,
        y=1,
        visible=True,
        complete=True,
    )
    centered_pair_predictions.add_instance(lf, user_instance)

    # Add another predicted instance (same track) incase ordering matters
    centered_pair_predictions.add_instance(lf, lf.predicted_instances[0])

    # Ensure user-instance is used in occupancy matrix instead of predicted-instance
    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(centered_pair_predictions, all_frames=True)
    assert_output_matrices_shape(num_tracks=27, num_frames=1100, num_nodes=24)
    assert_instance_points(
        points,
        user_instance,
        track_idx=user_instance.track.spawned_on,
        frame_idx=lf.frame_idx,
    )

    # Check that output matrices are correct for single instance projects
    labels = min_labels_robot
    (
        occupancy,
        points,
        point_scores,
        instance_scores,
        tracking_scores,
    ) = get_occupancy_and_points_matrices(labels, all_frames=True)
    assert_output_matrices_shape(num_tracks=1, num_frames=80, num_nodes=2)

    frame_idx = 0
    user_instance = labels[frame_idx].instances[0]
    assert_instance_points(points, user_instance, track_idx=0, frame_idx=frame_idx)


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


def read_lens_and_meta_hdf5(filename):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        dset_lens = {}
        dset_metadata = {}
        for dset_name in dset_names:
            if dset_name in [
                "track_names",
                "node_names",
                "edge_names",
                "edge_inds",
            ]:
                dset_lens[dset_name] = len(f[dset_name])
            elif dset_name in [
                "tracks",
                "track_occupancy",
                "point_scores",
                "instance_scores",
                "tracking_scores",
            ]:
                dset_lens[dset_name] = (f[dset_name][:].T).shape
            else:  # Scalar dataset.
                dset_metadata[dset_name] = read_scalar_dataset(f, dset_name)
    return dset_lens, dset_metadata


def read_scalar_dataset(f, dset_name):
    val_no_decode = f[dset_name][()]
    return val_no_decode.decode() if isinstance(val_no_decode, bytes) else val_no_decode


def extract_meta_hdf5(filename, dset_names_in: List):
    dset_names_metadata = ["labels_path", "video_path", "video_ind", "provanence"]
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        dset_names_found = list(
            set(dset_names) & set(dset_names_in) & set(dset_names_metadata)
        )
        dset_metadata = {}
        for dset_name in dset_names_found:
            dset_metadata[dset_name] = read_scalar_dataset(f, dset_name)
    return dset_metadata


def assert_dset_lens(dset_lens: dict, num_tracks: int, num_frames: int, num_nodes: int):
    assert dset_lens["track_names"] == num_tracks
    assert dset_lens["node_names"] == num_nodes
    assert dset_lens["edge_names"] == num_nodes - 1
    assert dset_lens["edge_inds"] == num_nodes - 1
    assert dset_lens["tracks"] == (num_frames, num_nodes, 2, num_tracks)
    assert dset_lens["track_occupancy"] == (num_tracks, num_frames)
    assert dset_lens["point_scores"] == (num_frames, num_nodes, num_tracks)
    assert dset_lens["instance_scores"] == (num_frames, num_tracks)
    assert dset_lens["tracking_scores"] == (num_frames, num_tracks)


def assert_dset_metadata(dset_metadata: dict, labels: Labels, video: Video):
    print(f'\nlabels_path = {dset_metadata["labels_path"]}')
    assert dset_metadata["labels_path"] == str(None)  # No labels path given.
    assert dset_metadata["video_path"] == video.backend.filename
    assert dset_metadata["video_ind"] == labels.videos.index(video)
    assert dset_metadata["provenance"] == json.dumps(labels.provenance)


def test_hdf5_video_arg(
    centered_pair_predictions: Labels, small_robot_mp4_vid: Video, tmpdir
):

    labels = centered_pair_predictions
    labels.add_video(small_robot_mp4_vid)

    output_paths = []
    for video in labels.videos:
        vn = PurePath(video.backend.filename)
        output_paths.append(PurePath(tmpdir, f"{vn.stem}.analysis.h5"))
        main(
            labels=labels,
            output_path=output_paths[-1],
            all_frames=True,
            video=video,
        )

    # Read hdf5 to ensure shapes are correct: centered_pair_low_quality
    dset_lens, dset_metadata = read_lens_and_meta_hdf5(output_paths[0])
    assert_dset_lens(dset_lens, num_tracks=27, num_frames=1100, num_nodes=24)
    assert_dset_metadata(dset_metadata, labels, video=labels.videos[0])

    # No file should exist for video with no labeled frames
    assert Path(output_paths[1]).exists() == False

    # Add labeled frames to second video, repeat process
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = AddUserInstancesFromPredictions.make_instance_from_predicted_instance(
        copy_instance=labels[0].predicted_instances[0]
    )
    labels.add_instance(frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)
    print(f"labels.tracks = {labels.tracks}")
    print(f"None in labels.tracks = {None in labels.tracks}")

    main(
        labels=labels,
        output_path=output_paths[1],
        all_frames=True,
        video=video,
    )
    dset_lens, dset_metadata = read_lens_and_meta_hdf5(output_paths[1])
    assert_dset_lens(dset_lens, num_tracks=1, num_frames=1, num_nodes=24)
    assert_dset_metadata(dset_metadata, labels, video=labels.videos[1])

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

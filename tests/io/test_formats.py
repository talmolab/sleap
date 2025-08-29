import os
from pathlib import PurePath

import pandas as pd

from sleap.io.format import filehandle
from sleap.info.write_tracking_h5 import get_nodes_as_np_strings
from sleap.io.format.sleap_analysis import SleapAnalysisAdaptor


def test_sleap_analysis_read(small_robot_3_frame_vid, small_robot_3_frame_hdf5):
    # Single instance hdf5 analysis file test
    read_labels = SleapAnalysisAdaptor.read(
        file=small_robot_3_frame_hdf5, video=small_robot_3_frame_vid
    )

    assert len(read_labels.videos) == 1
    assert len(read_labels.tracks) == 1
    assert len(read_labels.skeletons) == 1


def test_csv(tmpdir, min_labels_slp, minimal_instance_predictions_csv_path):
    from sleap.info.write_tracking_h5 import main as write_analysis

    filename_csv = str(tmpdir + "\\analysis.csv")
    write_analysis(min_labels_slp, output_path=filename_csv, all_frames=True, csv=True)

    labels_csv = pd.read_csv(filename_csv)

    csv_predictions = pd.read_csv(minimal_instance_predictions_csv_path)

    assert labels_csv.equals(csv_predictions)

    labels = min_labels_slp

    # check number of cols
    assert len(labels_csv.columns) - 3 == len(get_nodes_as_np_strings(labels)) * 3


def test_analysis_hdf5(tmpdir, centered_pair_predictions):
    from sleap.info.write_tracking_h5 import main as write_analysis

    filename = os.path.join(tmpdir, "analysis.h5")
    video = centered_pair_predictions.videos[0]

    write_analysis(centered_pair_predictions, output_path=filename, all_frames=True)

    labels = SleapAnalysisAdaptor.read(
        file=filehandle.FileHandle(filename),
        video=video,
    )

    assert len(labels) == len(centered_pair_predictions)
    assert len(labels.tracks) == len(centered_pair_predictions.tracks)
    assert len(labels.all_instances) == len(centered_pair_predictions.all_instances)


def test_matching_adaptor(centered_pair_predictions_hdf5_path):
    from sleap.io.format import read

    read(
        centered_pair_predictions_hdf5_path,
        for_object="labels",
        as_format="*",
    )

    read(
        "tests/data/json_format_v1/centered_pair.json",
        for_object="labels",
        as_format="*",
    )


def assert_read_labels_match(labels, read_labels):
    # Labeled Frames
    assert len(read_labels.labeled_frames) == len(labels.labeled_frames)

    # Instances
    frame_idx = 7
    read_instances = read_labels[frame_idx].instances
    instances = labels[frame_idx]
    assert len(instances) == len(read_instances)

    # Points
    for read_inst, inst in zip(read_instances, instances):
        for read_points, points in zip(read_inst.points, inst.points):
            assert read_points == points

    # Video
    assert len(read_labels.videos) == len(labels.videos)
    for video_idx, _ in enumerate(labels.videos):
        # The ordering of reading processing modules from NWB files seems to vary
        try:
            assert PurePath(read_labels.videos[video_idx].backend.filename) == PurePath(
                labels.videos[video_idx].backend.filename
            )
            assert isinstance(
                read_labels.videos[video_idx].backend,
                type(labels.videos[video_idx].backend),
            )
        except Exception:
            assert PurePath(read_labels.videos[video_idx].backend.filename) == PurePath(
                labels.videos[video_idx - 1].backend.filename
            )
            assert isinstance(
                read_labels.videos[video_idx].backend,
                type(labels.videos[video_idx - 1].backend),
            )

    # Skeleton
    assert read_labels.skeleton.node_names == labels.skeleton.node_names
    assert read_labels.skeleton.edge_inds == labels.skeleton.edge_inds
    assert len(read_labels.tracks) == len(labels.tracks)

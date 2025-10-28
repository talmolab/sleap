import pytest
import numpy as np

import sleap_io as sio
from sleap_io import LabeledFrame
from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io import Skeleton
from sleap_io import Labels
from sleap_io import Video

TEST_JSON_LABELS = "tests/data/json_format_v1/centered_pair.json"
TEST_SLP_LABELS = "tests/data/slp_hdf5/centered_pair.slp"
TEST_JSON_PREDICTIONS = "tests/data/json_format_v2/centered_pair_predictions.json"
TEST_JSON_MIN_LABELS = "tests/data/json_format_v2/minimal_instance.json"
TEST_SLP_MIN_LABELS = "tests/data/slp_hdf5/minimal_instance.slp"
TEST_MAT_LABELS = "tests/data/mat/labels.mat"
TEST_SLP_MIN_LABELS_ROBOT = "tests/data/slp_hdf5/small_robot_minimal.slp"
TEST_SLP_SIV_ROBOT = "tests/data/siv_format_v1/small_robot_siv.slp"
TEST_SLP_SIV_ROBOT_CACHING = "tests/data/siv_format_v2/small_robot_siv_caching.slp"
TEST_MIN_TRACKS_2NODE_LABELS = "tests/data/tracks/clip.2node.slp"
TEST_MIN_TRACKS_13NODE_LABELS = "tests/data/tracks/clip.slp"
TEST_HDF5_PREDICTIONS = "tests/data/hdf5_format_v1/centered_pair_predictions.h5"
TEST_SLP_PREDICTIONS = "tests/data/hdf5_format_v1/centered_pair_predictions.slp"
TEST_MIN_DANCE_LABELS = "tests/data/slp_hdf5/dance.mp4.labels.slp"
TEST_CSV_PREDICTIONS = (
    "tests/data/csv_format/minimal_instance.000_centered_pair_low_quality.analysis.csv"
)


@pytest.fixture
def centered_pair_labels():
    # FIXME: Legacy JSON format not supported
    # return sio.load_file(TEST_JSON_LABELS)
    return sio.load_file(TEST_SLP_LABELS)


@pytest.fixture
def centered_pair_predictions():
    # FIXME: Legacy JSON format not supported
    # return sio.load_file(TEST_JSON_PREDICTIONS)
    return sio.load_file(TEST_SLP_PREDICTIONS)


@pytest.fixture
def centered_pair_predictions_sorted(centered_pair_predictions):
    labels: Labels = centered_pair_predictions
    labels.labeled_frames.sort(key=lambda lf: lf.frame_idx)
    return labels


@pytest.fixture
def min_labels():
    # FIXME: Legacy JSON format not supported
    # return sio.load_file(TEST_JSON_MIN_LABELS
    return sio.load_file(TEST_SLP_MIN_LABELS)


@pytest.fixture
def min_labels_slp():
    return sio.load_file(TEST_SLP_MIN_LABELS)


@pytest.fixture
def min_labels_slp_path():
    return TEST_SLP_MIN_LABELS


@pytest.fixture
def min_labels_robot():
    return sio.load_file(TEST_SLP_MIN_LABELS_ROBOT)


@pytest.fixture
def siv_robot():
    """Created before grayscale attribute was added to SingleImageVideo backend."""
    return sio.load_file(TEST_SLP_SIV_ROBOT)


@pytest.fixture
def min_tracks_2node_labels():
    return sio.load_file(TEST_MIN_TRACKS_2NODE_LABELS)


@pytest.fixture
def min_tracks_2node_predictions():
    """
    Generated with:
    ```
    sleap-track -m "tests/data/models/min_tracks_2node.UNet.bottomup_multiclass" \
        "tests/data/tracks/clip.mp4"
    ```
    """
    return sio.load_file("tests/data/tracks/clip.predictions.slp")


@pytest.fixture
def min_tracks_13node_labels():
    return sio.load_file(TEST_MIN_TRACKS_13NODE_LABELS)


@pytest.fixture
def mat_labels():
    return sio.load_leap(TEST_MAT_LABELS)


TEST_LEGACY_GRID_LABELS = "tests/data/test_grid/test_grid_labels.legacy.slp"
TEST_MIDPOINT_GRID_LABELS = "tests/data/test_grid/test_grid_labels.midpoint.slp"


@pytest.fixture
def legacy_grid_labels_path():
    return TEST_LEGACY_GRID_LABELS


@pytest.fixture
def legacy_grid_labels():
    return sio.load_file(TEST_LEGACY_GRID_LABELS)


@pytest.fixture
def midpoint_grid_labels_path():
    return TEST_MIDPOINT_GRID_LABELS


@pytest.fixture
def midpoint_grid_labels():
    return sio.load_file(TEST_MIDPOINT_GRID_LABELS)


@pytest.fixture
def simple_predictions():
    video = Video.from_filename("video.mp4")

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")

    track_a = Track(name="a")
    track_b = Track(name="b")

    labels = Labels()

    instances = []
    instances.append(
        PredictedInstance.from_numpy(
            np.array([[1, 1], [1, 1]], dtype=np.float32),
            skeleton=skeleton,
            point_scores=np.array([0.5, 0.7], dtype=np.float32),
            score=2,
            track=track_a,
        )
    )
    instances.append(
        PredictedInstance.from_numpy(
            np.array([[1, 1], [1, 1]], dtype=np.float32),
            skeleton=skeleton,
            point_scores=np.array([0.5, 0.7], dtype=np.float32),
            score=5,
            track=track_b,
        )
    )

    labeled_frame = LabeledFrame(video, frame_idx=0, instances=instances)
    labels.append(labeled_frame)

    instances = []
    instances.append(
        PredictedInstance.from_numpy(
            np.array([[4, 5], [1, 1]], dtype=np.float32),
            skeleton=skeleton,
            point_scores=np.array([1.0, 1.6], dtype=np.float32),
            score=3,
            track=track_a,
        )
    )
    instances.append(
        PredictedInstance.from_numpy(
            np.array([[6, 13], [1, 1]], dtype=np.float32),
            skeleton=skeleton,
            point_scores=np.array([1.0, 1.6], dtype=np.float32),
            score=6,
            track=track_b,
        )
    )

    labeled_frame = LabeledFrame(video, frame_idx=1, instances=instances)
    labels.append(labeled_frame)

    return labels


@pytest.fixture
def multi_skel_vid_labels(hdf5_vid, small_robot_mp4_vid, skeleton, stickman):
    """
    Build a big list of LabeledFrame objects and wrap it in Labels class.

    Args:
        hdf5_vid: An HDF5 video fixture
        small_robot_mp4_vid: An MP4 video fixture
        skeleton: A fly skeleton.
        stickman: A stickman skeleton

    Returns:
        The Labels object containing all the labeled frames
    """
    labels = []
    stick_tracks = [Track(name=f"Stickman {i}") for i in range(6)]
    fly_tracks = [Track(name=f"Fly {i}") for i in range(6)]

    # Make some tracks None to test that
    fly_tracks[3] = None
    stick_tracks[2] = None

    for f in range(500):
        from sleap.sleap_io_adaptors.video_utils import (
            video_get_frames,
            video_get_height,
            video_get_width,
        )

        vid = [hdf5_vid, small_robot_mp4_vid][f % 2]
        label = LabeledFrame(video=vid, frame_idx=f % video_get_frames(vid))

        fly_instances = []
        for i in range(6):
            fly_instances.append(Instance(skeleton=skeleton, track=fly_tracks[i]))
            for node in skeleton.nodes:
                fly_instances[i][node] = (
                    [i % video_get_width(vid), i % video_get_height(vid)],
                    True,
                    False,
                )  # (xy, visible, complete)

        stickman_instances = []
        for i in range(6):
            stickman_instances.append(
                Instance(skeleton=stickman, track=stick_tracks[i])
            )
            for node in stickman.nodes:
                stickman_instances[i][node] = (
                    [i % video_get_width(vid), i % video_get_height(vid)],
                    True,
                    False,
                )  # (xy, visible, complete)

        label.instances = stickman_instances + fly_instances
        labels.append(label)

    labels = Labels(labels)

    return labels


@pytest.fixture
def centered_pair_predictions_hdf5_path():
    return TEST_HDF5_PREDICTIONS


@pytest.fixture
def minimal_instance_predictions_csv_path():
    return TEST_CSV_PREDICTIONS


@pytest.fixture
def centered_pair_predictions_slp_path():
    return TEST_SLP_PREDICTIONS


@pytest.fixture
def min_dance_labels():
    return sio.load_file(TEST_MIN_DANCE_LABELS)


@pytest.fixture
def movenet_video():
    return Video.from_filename("tests/data/videos/dance.mp4")

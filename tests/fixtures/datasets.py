import os
import pytest

from sleap.instance import (
    Instance,
    PredictedInstance,
    Point,
    PredictedPoint,
    LabeledFrame,
    Track,
)
from sleap.skeleton import Skeleton
from sleap.io.dataset import Labels
from sleap.io.video import Video

TEST_JSON_LABELS = "tests/data/json_format_v1/centered_pair.json"
TEST_JSON_PREDICTIONS = "tests/data/json_format_v2/centered_pair_predictions.json"
TEST_JSON_MIN_LABELS = "tests/data/json_format_v2/minimal_instance.json"
TEST_SLP_MIN_LABELS = "tests/data/slp_hdf5/minimal_instance.slp"
TEST_MAT_LABELS = "tests/data/mat/labels.mat"
TEST_SLP_MIN_LABELS_ROBOT = "tests/data/slp_hdf5/small_robot_minimal.slp"
TEST_MIN_TRACKS_2NODE_LABELS = "tests/data/tracks/clip.2node.slp"
TEST_MIN_TRACKS_13NODE_LABELS = "tests/data/tracks/clip.slp"
TEST_HDF5_PREDICTIONS = "tests/data/hdf5_format_v1/centered_pair_predictions.h5"
TEST_SLP_PREDICTIONS = "tests/data/hdf5_format_v1/centered_pair_predictions.slp"


@pytest.fixture
def centered_pair_labels():
    return Labels.load_file(TEST_JSON_LABELS)


@pytest.fixture
def centered_pair_predictions():
    return Labels.load_file(TEST_JSON_PREDICTIONS)


@pytest.fixture
def min_labels():
    return Labels.load_file(TEST_JSON_MIN_LABELS)


@pytest.fixture
def min_labels_slp():
    return Labels.load_file(TEST_SLP_MIN_LABELS)


@pytest.fixture
def min_labels_slp_path():
    return TEST_SLP_MIN_LABELS


@pytest.fixture
def min_labels_robot():
    return Labels.load_file(TEST_SLP_MIN_LABELS_ROBOT)


@pytest.fixture
def min_tracks_2node_labels():
    return Labels.load_file(
        TEST_MIN_TRACKS_2NODE_LABELS, video_search=["tests/data/tracks/clip.mp4"]
    )


@pytest.fixture
def min_tracks_13node_labels():
    return Labels.load_file(
        TEST_MIN_TRACKS_13NODE_LABELS, video_search=["tests/data/tracks/clip.mp4"]
    )


@pytest.fixture
def mat_labels():
    return Labels.load_leap_matlab(TEST_MAT_LABELS, gui=False)


TEST_LEGACY_GRID_LABELS = "tests/data/test_grid/test_grid_labels.legacy.h5"
TEST_MIDPOINT_GRID_LABELS = "tests/data/test_grid/test_grid_labels.midpoint.h5"


@pytest.fixture
def legacy_grid_labels_path():
    return TEST_LEGACY_GRID_LABELS


@pytest.fixture
def legacy_grid_labels():
    return Labels.load_file(
        TEST_LEGACY_GRID_LABELS, video_search=TEST_LEGACY_GRID_LABELS
    )


@pytest.fixture
def midpoint_grid_labels_path():
    return TEST_MIDPOINT_GRID_LABELS


@pytest.fixture
def midpoint_grid_labels():
    return Labels.load_file(
        TEST_MIDPOINT_GRID_LABELS, video_search=TEST_MIDPOINT_GRID_LABELS
    )


@pytest.fixture
def simple_predictions():

    video = Video.from_filename("video.mp4")

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")

    track_a = Track(0, "a")
    track_b = Track(0, "b")

    labels = Labels()

    instances = []
    instances.append(
        PredictedInstance(
            skeleton=skeleton,
            score=2,
            track=track_a,
            points=dict(
                a=PredictedPoint(1, 1, score=0.5), b=PredictedPoint(1, 1, score=0.5)
            ),
        )
    )
    instances.append(
        PredictedInstance(
            skeleton=skeleton,
            score=5,
            track=track_b,
            points=dict(
                a=PredictedPoint(1, 1, score=0.7), b=PredictedPoint(1, 1, score=0.7)
            ),
        )
    )

    labeled_frame = LabeledFrame(video, frame_idx=0, instances=instances)
    labels.append(labeled_frame)

    instances = []
    instances.append(
        PredictedInstance(
            skeleton=skeleton,
            score=3,
            track=track_a,
            points=dict(
                a=PredictedPoint(4, 5, score=1.5), b=PredictedPoint(1, 1, score=1.0)
            ),
        )
    )
    instances.append(
        PredictedInstance(
            skeleton=skeleton,
            score=6,
            track=track_b,
            points=dict(
                a=PredictedPoint(6, 13, score=1.7), b=PredictedPoint(1, 1, score=1.0)
            ),
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
    stick_tracks = [Track(spawned_on=0, name=f"Stickman {i}") for i in range(6)]
    fly_tracks = [Track(spawned_on=0, name=f"Fly {i}") for i in range(6)]

    # Make some tracks None to test that
    fly_tracks[3] = None
    stick_tracks[2] = None

    for f in range(500):
        vid = [hdf5_vid, small_robot_mp4_vid][f % 2]
        label = LabeledFrame(video=vid, frame_idx=f % vid.frames)

        fly_instances = []
        for i in range(6):
            fly_instances.append(Instance(skeleton=skeleton, track=fly_tracks[i]))
            for node in skeleton.nodes:
                fly_instances[i][node] = Point(x=i % vid.width, y=i % vid.height)

        stickman_instances = []
        for i in range(6):
            stickman_instances.append(
                Instance(skeleton=stickman, track=stick_tracks[i])
            )
            for node in stickman.nodes:
                stickman_instances[i][node] = Point(x=i % vid.width, y=i % vid.height)

        label.instances = stickman_instances + fly_instances
        labels.append(label)

    labels = Labels(labels)

    return labels


@pytest.fixture
def centered_pair_predictions_hdf5_path():
    return TEST_HDF5_PREDICTIONS


@pytest.fixture
def centered_pair_predictions_slp_path():
    return TEST_SLP_PREDICTIONS

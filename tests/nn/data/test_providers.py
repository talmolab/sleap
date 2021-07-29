import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test
from tests.fixtures.videos import TEST_H5_FILE, TEST_SMALL_ROBOT_MP4_FILE
import sleap
from sleap.nn.data import providers


def test_labels_reader(min_labels):
    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    assert not labels_reader.is_from_multi_size_videos

    example = next(iter(ds))

    assert len(labels_reader) == 1

    assert example["image"].shape == (384, 384, 1)
    assert example["image"].dtype == tf.uint8

    assert example["raw_image_size"].dtype == tf.int32
    np.testing.assert_array_equal(example["raw_image_size"], (384, 384, 1))

    assert example["example_ind"] == 0
    assert example["example_ind"].dtype == tf.int64

    assert example["video_ind"] == 0
    assert example["video_ind"].dtype == tf.int32

    assert example["frame_ind"] == 0
    assert example["frame_ind"].dtype == tf.int64

    assert example["instances"].shape == (2, 2, 2)
    assert example["instances"].dtype == tf.float32

    np.testing.assert_array_equal(example["scale"], (1.0, 1.0))
    assert example["scale"].dtype == tf.float32

    np.testing.assert_array_equal(example["skeleton_inds"], [0, 0])
    assert example["skeleton_inds"].dtype == tf.int32


def test_labels_reader_no_visible_points(min_labels):

    # There should be two instances in the labels dataset
    labels = min_labels.copy()
    assert len(labels.labeled_frames[0].instances) == 2

    # Non-visible ones will be removed in place
    inst = labels.labeled_frames[0].instances[0]
    for pt in inst.points:
        pt.visible = False
    labels_reader = providers.LabelsReader.from_user_instances(labels)
    assert len(labels.labeled_frames[0].instances) == 1

    # Make sure there's only one included with the instances for training
    example = next(iter(labels_reader.make_dataset()))
    assert len(example["instances"]) == 1

    # Now try with no visible instances
    labels = min_labels.copy()
    for inst in labels.labeled_frames[0].instances:
        for pt in inst.points:
            pt.visible = False
    labels_reader = providers.LabelsReader.from_user_instances(labels)
    assert len(labels) == 0
    assert len(labels_reader) == 0


def test_labels_reader_subset(min_labels):
    labels = sleap.Labels([min_labels[0], min_labels[0], min_labels[0]])
    assert len(labels) == 3

    labels_reader = providers.LabelsReader(labels, example_indices=[2, 1])
    assert len(labels_reader) == 2
    examples = list(iter(labels_reader.make_dataset()))
    assert len(examples) == 2
    assert examples[0]["example_ind"] == 2
    assert examples[1]["example_ind"] == 1


def test_video_reader_mp4():
    video_reader = providers.VideoReader.from_filepath(TEST_SMALL_ROBOT_MP4_FILE)
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 166

    assert example["image"].shape == (320, 560, 3)
    assert example["image"].dtype == tf.uint8

    assert example["raw_image_size"].dtype == tf.int32
    np.testing.assert_array_equal(example["raw_image_size"], (320, 560, 3))

    assert example["frame_ind"] == 0
    assert example["frame_ind"].dtype == tf.int64

    np.testing.assert_array_equal(example["scale"], (1.0, 1.0))
    assert example["scale"].dtype == tf.float32


def test_video_reader_mp4_subset():
    video_reader = providers.VideoReader.from_filepath(
        TEST_SMALL_ROBOT_MP4_FILE, example_indices=[2, 1, 4]
    )

    assert len(video_reader) == 3

    ds = video_reader.make_dataset()
    examples = list(iter(ds))

    assert examples[0]["frame_ind"].dtype == tf.int64
    assert examples[0]["frame_ind"] == 2
    assert examples[1]["frame_ind"] == 1
    assert examples[2]["frame_ind"] == 4


def test_video_reader_mp4_grayscale():
    video_reader = providers.VideoReader.from_filepath(
        TEST_SMALL_ROBOT_MP4_FILE, grayscale=True
    )
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 166

    assert example["image"].shape == (320, 560, 1)
    assert example["image"].dtype == tf.uint8

    assert example["raw_image_size"].dtype == tf.int32
    np.testing.assert_array_equal(example["raw_image_size"], (320, 560, 1))


def test_video_reader_hdf5():
    video_reader = providers.VideoReader.from_filepath(
        TEST_H5_FILE, dataset="/box", input_format="channels_first"
    )
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 42

    assert example["image"].shape == (512, 512, 1)
    assert example["image"].dtype == tf.uint8

    assert example["raw_image_size"].dtype == tf.int32
    np.testing.assert_array_equal(example["raw_image_size"], (512, 512, 1))


def test_labels_reader_multi_size():
    # Create some fake data using two different size videos.
    skeleton = sleap.Skeleton.from_names_and_edge_inds(["A"])
    labels = sleap.Labels(
        [
            sleap.LabeledFrame(
                frame_idx=0,
                video=sleap.Video.from_filename(
                    TEST_SMALL_ROBOT_MP4_FILE, grayscale=True
                ),
                instances=[
                    sleap.Instance.from_pointsarray(
                        np.array([[128, 128]]), skeleton=skeleton
                    )
                ],
            ),
            sleap.LabeledFrame(
                frame_idx=0,
                video=sleap.Video.from_filename(
                    TEST_H5_FILE, dataset="/box", input_format="channels_first"
                ),
                instances=[
                    sleap.Instance.from_pointsarray(
                        np.array([[128, 128]]), skeleton=skeleton
                    )
                ],
            ),
        ]
    )

    # Create a loader for those labels.
    labels_reader = providers.LabelsReader(labels)
    ds = labels_reader.make_dataset()
    ds_iter = iter(ds)

    # Check LabelReader can provide different shapes of individual samples
    assert next(ds_iter)["image"].shape == (320, 560, 1)
    assert next(ds_iter)["image"].shape == (512, 512, 1)

    # Check util functions
    h, w = labels_reader.max_height_and_width
    assert h == 512
    assert w == 560
    assert labels_reader.is_from_multi_size_videos

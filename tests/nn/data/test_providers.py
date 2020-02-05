import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test
from tests.fixtures.videos import TEST_H5_FILE, TEST_SMALL_ROBOT_MP4_FILE

from sleap.nn.data import providers

def test_labels_reader(min_labels):
    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example = next(iter(ds))

    assert len(labels_reader) == 1

    assert example["image"].shape == (384, 384, 1)
    assert example["image"].dtype == tf.uint8

    assert example["video_ind"] == 0
    assert example["video_ind"].dtype == tf.int32

    assert example["frame_ind"] == 0
    assert example["frame_ind"].dtype == tf.int64

    assert example["instances"].shape == (2, 2, 2)
    assert example["instances"].dtype == tf.float32

    assert example["scale"] == 1.0
    assert example["scale"].dtype == tf.float32

    np.testing.assert_array_equal(example["skeleton_inds"], [0, 0])
    assert example["skeleton_inds"].dtype == tf.int32


def test_video_reader_mp4():
    video_reader = providers.VideoReader.from_filepath(TEST_SMALL_ROBOT_MP4_FILE)
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 166

    assert example["image"].shape == (320, 560, 3)
    assert example["image"].dtype == tf.uint8

    assert example["frame_ind"] == 0
    assert example["frame_ind"].dtype == tf.int64

    assert example["scale"] == 1.0
    assert example["scale"].dtype == tf.float32


def test_video_reader_mp4_grayscale():
    video_reader = providers.VideoReader.from_filepath(
        TEST_SMALL_ROBOT_MP4_FILE, grayscale=True)
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 166

    assert example["image"].shape == (320, 560, 1)
    assert example["image"].dtype == tf.uint8


def test_video_reader_hdf5():
    video_reader = providers.VideoReader.from_filepath(
        TEST_H5_FILE, dataset="/box", input_format="channels_first")
    ds = video_reader.make_dataset()
    example = next(iter(ds))

    assert len(video_reader) == 42

    assert example["image"].shape == (512, 512, 1)
    assert example["image"].dtype == tf.uint8

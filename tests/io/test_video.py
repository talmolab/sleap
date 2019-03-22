import pytest
import numpy as np

from sleap.io.video import Video
from tests.fixtures.videos import TEST_H5_FILE, TEST_SMALL_ROBOT_MP4_FILE

# FIXME:
# Parameterizing fixtures with fixtures is annoying so this leads to a lot
# of redundant test code here.
# See: https://github.com/pytest-dev/pytest/issues/349

def test_hdf5_get_shape(hdf5_vid):
    assert(hdf5_vid.shape == (42, 512, 512, 1))


def test_hdf5_len(hdf5_vid):
    assert(len(hdf5_vid) == 42)


def test_hdf5_dtype(hdf5_vid):
    assert(hdf5_vid.dtype == np.uint8)


def test_hdf5_get_frame(hdf5_vid):
    assert(hdf5_vid.get_frame(0).shape == (512, 512, 1))


def test_hdf5_get_frames(hdf5_vid):
    assert(hdf5_vid.get_frames(0).shape == (1, 512, 512, 1))
    assert(hdf5_vid.get_frames([0,1]).shape == (2, 512, 512, 1))


def test_hdf5_get_item(hdf5_vid):
    assert(hdf5_vid[0].shape == (1, 512, 512, 1))
    assert(np.alltrue(hdf5_vid[1:10:3] == hdf5_vid.get_frames([1, 4, 7])))

def test_hd5f_file_not_found():
    with pytest.raises(FileNotFoundError):
        Video.from_hdf5("non-existent-file.h5", 'dataset_name')

def test_mp4_get_shape(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.shape == (166, 3, 560, 320))


def test_mp4_len(small_robot_mp4_vid):
    assert(len(small_robot_mp4_vid) == 166)


def test_mp4_dtype(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.dtype == np.uint8)


def test_mp4_get_frame(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.get_frame(0).shape == (320, 560, 3))


def test_mp4_get_frames(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.get_frames(0).shape == (1, 320, 560, 3))
    assert(small_robot_mp4_vid.get_frames([0,1]).shape == (2, 320, 560, 3))


def test_mp4_get_item(small_robot_mp4_vid):
    assert(small_robot_mp4_vid[0].shape == (1, 320, 560, 3))
    assert(np.alltrue(small_robot_mp4_vid[1:10:3] == small_robot_mp4_vid.get_frames([1, 4, 7])))

def test_mp4_file_not_found():
    with pytest.raises(FileNotFoundError):
        Video.from_media("non-existent-file.mp4")
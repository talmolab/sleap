import pytest
import numpy as np

from sleap.io.video import HDF5Video

TEST_H5_FILE = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
TEST_H5_DSET = "/box"
TEST_H5_INPUT_FORMAT = "channels_first"

@pytest.fixture
def hdf5_vid():
    return HDF5Video(TEST_H5_FILE, dset=TEST_H5_DSET, input_format=TEST_H5_INPUT_FORMAT)


def test_get_shape(hdf5_vid):
    assert(hdf5_vid.shape == (42, 512, 512, 1))

def test_len(hdf5_vid):
    assert(len(hdf5_vid) == 42)

def test_dtype(hdf5_vid):
    assert(hdf5_vid.dtype == np.uint8)

def test_get_frame(hdf5_vid):
    assert(hdf5_vid.get_frame(0).shape == (512, 512, 1))

def test_get_frames(hdf5_vid):
    assert(hdf5_vid.get_frames(0).shape == (1, 512, 512, 1))
    assert(hdf5_vid.get_frames([0,1]).shape == (2, 512, 512, 1))

def test_get_item(hdf5_vid):
    assert(hdf5_vid[0].shape == (1, 512, 512, 1))
    assert(np.alltrue(hdf5_vid[1:10:3] == hdf5_vid.get_frames([1, 4, 7])))


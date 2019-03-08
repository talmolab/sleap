import pytest

import h5py

from sleap.io.video import HDF5Video

TEST_H5_FILE = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
TEST_H5_DSET = "/box"
TEST_H5_INPUT_FORMAT = "channels_first"

@pytest.fixture
def hdf5_vid():
    return HDF5Video(TEST_H5_FILE, dset=TEST_H5_DSET, input_format=TEST_H5_INPUT_FORMAT)


def test_get_shape(hdf5_vid):
    with h5py.File(TEST_H5_FILE, "r") as f:
        shape = f[TEST_H5_DSET].shape

        if TEST_H5_INPUT_FORMAT == "channels_first":
            shape = (shape[0], shape[3], shape[2], shape[1])

    assert(hdf5_vid.shape == shape)


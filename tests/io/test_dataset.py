import pytest

from leap.io.dataset import Dataset

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def test_load_dataset():
    dataset = Dataset(path=TEST_H5_DATASET)

    # Basic sanity checks on the dataset.
    # FIXME: Need to do some more sophisticated checks
    assert(dataset.confmaps is not None)


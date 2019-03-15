import pytest

from sleap.io.dataset import Dataset

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def test_load_dataset():
    dataset = Dataset.load(path=TEST_H5_DATASET)

    # Basic sanity checks on the dataset.
    # FIXME: Need to do some more sophisticated checks
    assert(dataset.confmaps is not None)
    assert (dataset.skeleton is not None)
    assert (dataset.frames is not None)
    assert (dataset.pafs is not None)


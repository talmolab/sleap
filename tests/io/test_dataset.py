import os
import pytest

from sleap.io.dataset import Dataset

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def test_dataset_hdf5_save_load(dataset_hdf5):
    dataset_hdf5.save()

    assert os.path.isfile(dataset_hdf5.path)


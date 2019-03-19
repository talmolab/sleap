import os
import pytest

from sleap.io.dataset import Dataset

@pytest.fixture
def dataset_hdf5(multi_skel_instances, tmpdir):
    filename = os.path.join('.', 'dataset.h5')

    if os.path.isfile(filename):
        os.remove(filename)

    # Create or load the dataset
    dataset = Dataset.load(path=filename, create=True)
    dataset.instances = multi_skel_instances

    return dataset
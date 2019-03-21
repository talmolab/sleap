import os
import pytest

from sleap.instance import Instance, Point
from sleap.io.video import Video
from sleap.io.dataset import LabeledFrame, Labels

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def test_labels_json(tmpdir, multi_skel_vid_labels):
    json_file_path = os.path.join('.', 'dataset.json')

    if os.path.isfile(json_file_path):
        os.remove(json_file_path)

    # Save to json
    Labels.save_json(labels=multi_skel_vid_labels, filename=json_file_path)

    # Make sure the file is there
    assert os.path.isfile(json_file_path)

    # Lets load the labels back in and make sure we haven't lost anything.
    #loaded_labels = Labels.load_json(json_file_path)
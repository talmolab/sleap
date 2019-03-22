import os
import pytest
import numpy as np

from sleap.instance import Instance, Point
from sleap.io.video import Video
from sleap.io.dataset import LabeledFrame, Labels, load_labels_json_old

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def test_labels_json(tmpdir, multi_skel_vid_labels):
    json_file_path = os.path.join(tmpdir, 'dataset.json')

    if os.path.isfile(json_file_path):
        os.remove(json_file_path)

    # Save to json
    Labels.save_json(labels=multi_skel_vid_labels, filename=json_file_path)

    # Make sure the file is there
    assert os.path.isfile(json_file_path)

    # Lets load the labels back in and make sure we haven't lost anything.
    loaded_labels = Labels.load_json(json_file_path)

    # Check that we have the same thing
    for expected_label, label in zip(multi_skel_vid_labels.labels, loaded_labels.labels):
        assert expected_label.frame_idx == label.frame_idx

        # Compare the first frames of the videos, do it on a small sub-region to
        # make the test reasonable in time.
        np.allclose(expected_label.video.get_frame(0)[0:15,0:15,:], label.video.get_frame(0)[0:15,0:15,:])

        # Compare the instances
        assert expected_label.instances == label.instances


def test_load_labels_json_old(tmpdir):
    new_file_path = os.path.join(tmpdir, 'centered_pair_v2.json')

    # Function to run some checks on loaded labels
    def check_labels(labels):
        skel_node_names = ['head', 'neck', 'thorax', 'abdomen', 'wingL',
                           'wingR', 'forelegL1', 'forelegL2', 'forelegL3',
                           'forelegR1', 'forelegR2', 'forelegR3', 'midlegL1',
                           'midlegL2', 'midlegL3', 'midlegR1', 'midlegR2',
                           'midlegR3', 'hindlegL1', 'hindlegL2', 'hindlegL3',
                           'hindlegR1', 'hindlegR2', 'hindlegR3']

        # Do some basic checks
        assert len(labels) == 70

        # Make sure we only create one video object and it works
        assert len({label.video for label in labels}) == 1
        assert labels[0].video.get_frame(0).shape == (384, 384, 3)

        # Check some frame objects.
        assert labels[0].frame_idx == 118
        assert labels[40].frame_idx == 494

        # Check the skeleton
        assert labels[0].instances[0].skeleton.node_names == skel_node_names

    labels = Labels.load_json("tests/data/json_format_v1/centered_pair.json")
    check_labels(labels)

    # Save out to new JSON format
    Labels.save_json(labels, new_file_path)

    # Reload and check again.
    new_labels = Labels.load_json(new_file_path)
    check_labels(new_labels)


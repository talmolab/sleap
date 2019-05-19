import os

from sleap.nn.inference import load_predicted_labels_json_old
from sleap.instance import PredictedPoint, PredictedInstance

from sleap.io.dataset import Labels

def check_labels(labels):

    # Make sure there are 1100 frames
    assert len(labels) == 1100

    for i in labels.all_instances:
        assert type(i) == PredictedInstance
        assert type(i.points()[0]) == PredictedPoint

    # Make sure frames are in order
    for i, frame in enumerate(labels):
        assert frame.frame_idx == i

    # Make sure that we only found [2,3,4,5] number of instances
    assert set([len(f.instances) for f in labels]) == {2, 3, 4, 5}

    # Make sure we only get the correct number of tracks
    assert len(labels.tracks) == 27

    # FIXME: We need more checks here.

def test_load_old_json():
    labels = load_predicted_labels_json_old("tests/data/json_format_v1/centered_pair.json")

    check_labels(labels)

    #Labels.save_json(labels, 'tests/data/json_format_v2/centered_pair_predictions.json')

def test_save_load_json(centered_pair_predictions, tmpdir):
    test_out_file = os.path.join(tmpdir, 'test_tmp.json')

    # Check the labels
    check_labels(centered_pair_predictions)

    # Save to JSON
    Labels.save_json(centered_pair_predictions, test_out_file)

    # Load and check again
    new_labels = Labels.load_json(test_out_file)

    check_labels(new_labels)
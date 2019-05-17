from sleap.nn.inference import PredictedInstance, PredictedPoint, \
    load_predicted_labels_json_old

from sleap.io.dataset import Labels

def test_load_old_json():
    labels = load_predicted_labels_json_old("tests/data/json_format_v1/centered_pair.json")

    # Make sure there are 1100 frames
    assert len(labels) == 1100

    # Make sure frames are in order
    for i, frame in enumerate(labels):
        assert frame.frame_idx == i

    # Make sure that we only found [2,3,4,5] number of instances
    assert set([len(f.instances) for f in labels]) == {2,3,4,5}

    # Make sure we only get the correct number of tracks
    assert len(labels.tracks) == 27

    # FIXME: We need more checks here.

    #Labels.save_json(labels, 'tests/data/json_format_v2/centered_pair_predictions.json')

def test_save_load_json(centered_pair_predictions):
    labels = centered_pair_predictions

    # Make sure there are 1100 frames
    assert len(labels) == 1100

    # Make sure frames are in order
    for i, frame in enumerate(labels):
        assert frame.frame_idx == i

    # Make sure that we only found [2,3,4,5] number of instances
    assert set([len(f.instances) for f in labels]) == {2, 3, 4, 5}

    # Make sure we only get the correct number of tracks
    assert len(labels.tracks) == 27

    # FIXME: We need more checks here.
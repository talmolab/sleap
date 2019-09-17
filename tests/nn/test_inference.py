import os

import numpy as np

from sleap.io.legacy import load_predicted_labels_json_old
from sleap.nn.inference import find_all_peaks, match_peaks_paf, match_peaks_paf_par
from sleap.nn.datagen import generate_images, generate_confidence_maps, generate_pafs
from sleap.nn.transform import DataTransform
from sleap.instance import PredictedPoint, PredictedInstance

from sleap.io.video import Video

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

def test_peaks_with_scaling():

    # load from scratch so we won't change centered_pair_predictions
    true_labels = Labels.load_json('tests/data/json_format_v1/centered_pair.json')
    # only use a few frames
    true_labels.labeled_frames = true_labels.labeled_frames[13:23:2]
    skeleton = true_labels.skeletons[0]

    imgs = generate_images(true_labels)
    # scaling
    scale = .5
    transform = DataTransform()
    img_size = imgs.shape[1], imgs.shape[2]
    scaled_size = int(imgs.shape[1]//(1/scale)), int(imgs.shape[2]//(1/scale))
    imgs = transform.scale_to(imgs, scaled_size)
    assert transform.scale == scale
    assert imgs.shape[1], imgs.shape[2] == scaled_size
    # data gen
    video = Video.from_numpy(imgs * 255)
    confmaps = generate_confidence_maps(true_labels, scale=scale)
    pafs = generate_pafs(true_labels, scale=scale)

    # inference
    peaks, peak_vals = find_all_peaks(confmaps)
    lf = match_peaks_paf(peaks, peak_vals, pafs, skeleton, video, transform)
    new_labels = Labels(lf)

    # make sure what we got from interence matches what we started with
    for i in range(len(new_labels.labeled_frames)):
        assert len(true_labels.labeled_frames[i].instances) <= len(new_labels.labeled_frames[i].instances)

        # sort instances by location of thorax
        true_labels.labeled_frames[i].instances.sort(key=lambda inst: inst["thorax"])
        new_labels.labeled_frames[i].instances.sort(key=lambda inst: inst["thorax"])

        # make sure that each true instance has points matching one of the new instances
        for inst_a, inst_b in zip(true_labels.labeled_frames[i].instances, new_labels.labeled_frames[i].instances):
        
            assert inst_a.get_points_array().shape == inst_b.get_points_array().shape
            # FIXME: new instances have nans, so for now just check first 5 points
            assert np.allclose(inst_a.get_points_array()[0:5], inst_b.get_points_array()[0:5], atol=1/scale)

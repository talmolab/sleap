import os

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame, PredictedInstance
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.nn.model import ModelOutputType
from sleap.gui.active import make_default_training_jobs, find_saved_jobs, add_frames_from_json

def test_make_default_training_jobs():
    jobs = make_default_training_jobs()

    assert ModelOutputType.CONFIDENCE_MAP in jobs
    assert ModelOutputType.PART_AFFINITY_FIELD in jobs

    for output_type in jobs:
        assert jobs[output_type].model.output_type == output_type
        assert jobs[output_type].best_model_filename is None

def test_find_saved_jobs():
    jobs_a = find_saved_jobs("tests/data/training_profiles/set_a")
    assert len(jobs_a) == 3
    assert len(jobs_a[ModelOutputType.CONFIDENCE_MAP]) == 1

    jobs_b = find_saved_jobs("tests/data/training_profiles/set_b")
    assert len(jobs_b) == 1

    path, job = jobs_b[ModelOutputType.CONFIDENCE_MAP][0]
    assert os.path.basename(path) == "test_confmaps.json"
    assert job.trainer.num_epochs == 17

    # Add jobs from set_a to already loaded jobs from set_b
    jobs_c = find_saved_jobs("tests/data/training_profiles/set_a", jobs_b)
    assert len(jobs_c) == 3

    # Make sure we now have two confmap jobs
    assert len(jobs_c[ModelOutputType.CONFIDENCE_MAP]) == 2

    # Make sure set_a was added after items from set_b
    paths = [name for (name, job) in jobs_c[ModelOutputType.CONFIDENCE_MAP]]
    assert os.path.basename(paths[0]) == "test_confmaps.json"
    assert os.path.basename(paths[1]) == "default_confmaps.json"

def test_add_frames_from_json():
    vid_a = Video.from_filename("foo.mp4")
    vid_b = Video.from_filename("bar.mp4")

    skeleton_a = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    lf_a = LabeledFrame(vid_a, frame_idx=2, instances=[Instance(skeleton_a)])
    lf_b = LabeledFrame(vid_b, frame_idx=3, instances=[Instance(skeleton_b)])

    empty_labels = Labels()
    labels_with_video = Labels(videos=[vid_a])
    labels_with_skeleton = Labels(skeletons=[skeleton_a])

    new_labels_a = Labels(labeled_frames=[lf_a])
    new_labels_b = Labels(labeled_frames=[lf_b])

    json_a = new_labels_a.to_dict()
    json_b = new_labels_b.to_dict()

    # Test with empty labels

    assert len(empty_labels.labeled_frames) == 0
    assert len(empty_labels.skeletons) == 0
    assert len(empty_labels.skeletons) == 0

    add_frames_from_json(empty_labels, json_a)
    assert len(empty_labels.labeled_frames) == 1
    assert len(empty_labels.videos) == 1
    assert len(empty_labels.skeletons) == 1

    add_frames_from_json(empty_labels, json_b)
    assert len(empty_labels.labeled_frames) == 2
    assert len(empty_labels.videos) == 2
    assert len(empty_labels.skeletons) == 1

    empty_labels.to_dict()

    # Test with labels that have video

    assert len(labels_with_video.labeled_frames) == 0
    assert len(labels_with_video.skeletons) == 0
    assert len(labels_with_video.videos) == 1

    add_frames_from_json(labels_with_video, json_a)
    assert len(labels_with_video.labeled_frames) == 1
    assert len(labels_with_video.videos) == 1
    assert len(labels_with_video.skeletons) == 1

    add_frames_from_json(labels_with_video, json_b)
    assert len(labels_with_video.labeled_frames) == 2
    assert len(labels_with_video.videos) == 2
    assert len(labels_with_video.skeletons) == 1

    labels_with_video.to_dict()

    # Test with labels that have skeleton

    assert len(labels_with_skeleton.labeled_frames) == 0
    assert len(labels_with_skeleton.skeletons) == 1
    assert len(labels_with_skeleton.videos) == 0

    add_frames_from_json(labels_with_skeleton, json_a)
    assert len(labels_with_skeleton.labeled_frames) == 1
    assert len(labels_with_skeleton.videos) == 1
    assert len(labels_with_skeleton.skeletons) == 1

    add_frames_from_json(labels_with_skeleton, json_b)
    assert len(labels_with_skeleton.labeled_frames) == 2
    assert len(labels_with_skeleton.videos) == 2
    assert len(labels_with_skeleton.skeletons) == 1

    labels_with_skeleton.to_dict()
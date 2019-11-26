import os
import pytest

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame, PredictedInstance
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.nn.model import ModelOutputType
from sleap.gui.inference import InferenceDialog, JobMenuManager


def test_active_gui(qtbot, centered_pair_labels):
    win = InferenceDialog(
        labels_filename="foo.json", labels=centered_pair_labels, mode="expert"
    )
    win.show()
    qtbot.addWidget(win)

    # Make sure we include pafs by default
    jobs = win._get_current_training_jobs()
    assert ModelOutputType.PART_AFFINITY_FIELD in jobs

    # Test option to not include pafs
    assert "_dont_use_pafs" in win.form_widget.fields
    win.form_widget.set_form_data(dict(_dont_use_pafs=True))
    jobs = win._get_current_training_jobs()
    assert ModelOutputType.PART_AFFINITY_FIELD not in jobs


def test_inference_gui(qtbot, centered_pair_labels):
    win = InferenceDialog(
        labels_filename="foo.json", labels=centered_pair_labels, mode="inference"
    )
    win.show()
    qtbot.addWidget(win)

    # There aren't any trained models, so there should be no options shown for
    # inference
    jobs = win._get_current_training_jobs()
    assert len(jobs) == 0


def test_training_gui(qtbot, centered_pair_labels):
    win = InferenceDialog(
        labels_filename="foo.json", labels=centered_pair_labels, mode="learning"
    )
    win.show()
    qtbot.addWidget(win)

    # Make sure we include pafs and centroids by default
    jobs = win._get_current_training_jobs()
    assert ModelOutputType.PART_AFFINITY_FIELD in jobs
    assert ModelOutputType.CENTROIDS in jobs

    # Test option to not include pafs
    assert "_multi_instance_mode" in win.form_widget.fields
    win.form_widget.set_form_data(dict(_multi_instance_mode="single"))
    jobs = win._get_current_training_jobs()
    assert ModelOutputType.PART_AFFINITY_FIELD not in jobs

    # Test option to not include centroids
    assert "_region_proposal_mode" in win.form_widget.fields
    win.form_widget.set_form_data(dict(_region_proposal_mode="full frame"))
    jobs = win._get_current_training_jobs()
    assert ModelOutputType.CENTROIDS not in jobs


def test_find_saved_jobs():
    job_manager = JobMenuManager(None, dict())

    conf_menu_name = job_manager.menu_name_from_model_type(
        ModelOutputType.CONFIDENCE_MAP
    )

    jobs_a = job_manager.find_saved_jobs("tests/data/training_profiles/set_a")
    assert len(jobs_a) == 3
    assert len(jobs_a[conf_menu_name]) == 1

    jobs_b = job_manager.find_saved_jobs("tests/data/training_profiles/set_b")
    assert len(jobs_b) == 1

    path, job = jobs_b[conf_menu_name][0]
    assert os.path.basename(path) == "test_confmaps.json"
    assert job.trainer.num_epochs == 17

    # Add jobs from set_a to already loaded jobs from set_b
    jobs_c = job_manager.find_saved_jobs("tests/data/training_profiles/set_a", jobs_b)
    assert len(jobs_c) == 3

    # Make sure we now have two confmap jobs
    assert len(jobs_c[conf_menu_name]) == 2

    # Make sure set_a was added after items from set_b
    paths = [name for (name, job) in jobs_c[conf_menu_name]]
    assert os.path.basename(paths[0]) == "test_confmaps.json"
    assert os.path.basename(paths[1]) == "default_confmaps.json"


@pytest.mark.skip(reason="for old merging method")
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

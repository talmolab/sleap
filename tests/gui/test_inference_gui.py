from sleap.gui.learning import runners
from sleap.gui.learning.configs import TrainingConfigsGetter
from sleap.gui.learning.scopedkeydict import ScopedKeyDict

from sleap import Labels, LabeledFrame, Instance, PredictedInstance, Skeleton
from sleap.io.video import Video, MediaVideo

import numpy as np


def test_config_list_load():
    configs = TrainingConfigsGetter.make_from_labels_filename("").get_filtered_configs(
        "centroid"
    )

    assert len(configs) > 0
    for cfg in configs:
        assert cfg.config.model.heads.which_oneof_attrib_name() == "centroid"


def test_config_list_order():
    configs = TrainingConfigsGetter.make_from_labels_filename("").get_filtered_configs()

    # Check that all 'old' configs (if any) are last in the collected configs list
    for i in range(len(configs) - 1):
        # if current config is 'old', next must be 'old' as well
        assert not configs[i].filename.startswith("old.") or configs[
            i + 1
        ].filename.startswith("old.")


def test_scoped_key_dict():
    d = {"foo": 1, "bar": {"cat": {"dog": 2}, "elephant": 3}}

    x = ScopedKeyDict.from_hierarchical_dict(d).key_val_dict

    assert x["foo"] == 1
    assert x["bar.cat.dog"] == 2
    assert x["bar.elephant"] == 3


def test_inference_cli_builder():

    inference_task = runners.InferenceTask(
        trained_job_paths=["model1", "model2"],
        inference_params={"tracking.tracker": "simple"},
    )

    item_for_inference = runners.VideoItemForInference(
        video=Video.from_filename("video.mp4"),
        frames=[1, 2, 3],
    )

    cli_args, output_path = inference_task.make_predict_cli_call(item_for_inference)

    assert cli_args[0] == "sleap-track"
    assert cli_args[1] == "video.mp4"
    assert "model1" in cli_args
    assert "model2" in cli_args
    assert "--frames" in cli_args
    assert "--tracking.tracker" in cli_args

    assert output_path.startswith("video.mp4")
    assert output_path.endswith("predictions.slp")


def test_inference_cli_output_path():
    inference_task = runners.InferenceTask(
        trained_job_paths=["model1", "model2"],
        inference_params=dict(),
    )

    item_for_inference = runners.VideoItemForInference(
        video=Video.from_filename("video.mp4"),
        frames=[1, 2, 3],
    )

    # Try with specified output path
    cli_args, output_path = inference_task.make_predict_cli_call(
        item_for_inference,
        output_path="another_output_path.slp",
    )

    assert output_path == "another_output_path.slp"
    assert "another_output_path.slp" in cli_args


def test_inference_merging():
    skeleton = Skeleton.from_names_and_edge_inds(["A", "B"])
    video = Video.from_filename("video.mp4")
    lf_user_only = LabeledFrame(
        video=video, frame_idx=0, instances=[Instance(skeleton=skeleton)]
    )
    lf_pred_only = LabeledFrame(
        video=video, frame_idx=1, instances=[PredictedInstance(skeleton=skeleton)]
    )
    # Test pre-existing predicted instance is removed after merge
    lf_both = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[Instance(skeleton=skeleton), PredictedInstance(skeleton=skeleton)],
    )
    labels = Labels([lf_user_only, lf_pred_only, lf_both])

    task = runners.InferenceTask(
        trained_job_paths=None,
        inference_params=None,
        labels=labels,
        results=[
            LabeledFrame(
                video=labels.video,
                frame_idx=2,
                instances=[
                    # This instance should be merged
                    PredictedInstance.from_numpy(
                        points=np.array([[1, 1], [2, 2]]),
                        point_confidences=np.array([5.0, 6.0]),
                        instance_score=5.5,
                        skeleton=skeleton,
                    ),
                    # This instance should be merged
                    PredictedInstance.from_numpy(
                        points=np.array([[3, 3], [2, 2]]),
                        point_confidences=np.array([6.0, 7.0]),
                        instance_score=5.5,
                        skeleton=skeleton,
                    ),
                    # This instance should not be merged (all nan points)
                    PredictedInstance.from_numpy(
                        points=np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                        point_confidences=np.array([0.0, 0.0]),
                        instance_score=0,
                        skeleton=skeleton,
                    ),
                    # This instance should not be merged (no points)
                    PredictedInstance(skeleton=skeleton),
                ],
            )
        ],
    )
    task.merge_results()

    assert len(labels) == 3
    assert labels[0].frame_idx == 0
    assert labels[0].has_user_instances
    assert labels[1].frame_idx == 1
    assert labels[1].has_predicted_instances
    assert labels[2].frame_idx == 2
    assert len(labels[2].user_instances) == 1
    # Only predicted instances with graphable points should be merged
    assert len(labels[2].predicted_instances) == 2

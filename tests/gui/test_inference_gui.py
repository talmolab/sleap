from sleap.gui.learning import runners
from sleap.gui.learning.configs import TrainingConfigsGetter
from sleap.gui.learning.utils import ScopedKeyDict

from sleap.io.video import Video


def test_config_list_load():
    configs = TrainingConfigsGetter.make_from_labels_filename("").get_filtered_configs(
        "centroid"
    )

    assert 1 == len(configs)


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
        video=Video.from_filename("video.mp4"), frames=[1, 2, 3],
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
        trained_job_paths=["model1", "model2"], inference_params=dict(),
    )

    item_for_inference = runners.VideoItemForInference(
        video=Video.from_filename("video.mp4"), frames=[1, 2, 3],
    )

    # Try with specified output path
    cli_args, output_path = inference_task.make_predict_cli_call(
        item_for_inference, output_path="another_output_path.slp",
    )

    assert output_path == "another_output_path.slp"
    assert "another_output_path.slp" in cli_args

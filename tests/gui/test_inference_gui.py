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
    cli_args, output_path = runners.make_predict_cli_call(
        video=Video.from_filename("video.mp4"),
        trained_job_paths=["model1", "model2"],
        kwargs={"tracking.tracker": "simple"},
        frames=[1, 2, 3],
    )

    assert cli_args[0] == "sleap-track"
    assert cli_args[1] == "video.mp4"
    assert "model1" in cli_args
    assert "model2" in cli_args
    assert "--frames" in cli_args
    assert "--tracking.tracker" in cli_args

    assert output_path.startswith("video.mp4")
    assert output_path.endswith("predictions.slp")

    # Try with specified video path
    cli_args, output_path = runners.make_predict_cli_call(
        video=Video.from_filename("video.mp4"),
        video_path="another_video_path.mp4",
        trained_job_paths=["model1", "model2"],
        kwargs=dict(),
        frames=[1, 2, 3],
    )

    assert cli_args[1] == "another_video_path.mp4"
    assert "--tracking.tracker" not in cli_args

    # Try with specified output path
    cli_args, output_path = runners.make_predict_cli_call(
        video=Video.from_filename("video.mp4"),
        video_path="another_video_path.mp4",
        trained_job_paths=["model1", "model2"],
        kwargs=dict(),
        frames=[1, 2, 3],
        output_path="another_output_path.slp",
    )

    assert output_path == "another_output_path.slp"
    assert "another_output_path.slp" in cli_args

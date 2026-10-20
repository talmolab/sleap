"""Tests for the base config the training dialog builds a training job from.

Uses the real dialog with a real config list, and checks the result with
sleap-nn's own config validator and geometric augmentation.
"""

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.data.augmentation import apply_geometric_augmentation

from sleap.gui.config_utils import filter_cfg
from sleap.gui.learning import configs
from sleap.gui.learning.dialog import LearningDialog

SINGLE_PROFILE = "sleap/training_profiles/baseline_medium_rf.single.yaml"
GEOMETRIC_KEYS = (
    "rotation_min",
    "rotation_max",
    "rotation_p",
    "scale_min",
    "scale_max",
    "scale_p",
    "translate_width",
    "translate_height",
    "translate_p",
    "affine_p",
)


@pytest.fixture
def labels_path(tmp_path):
    path = tmp_path / "robot.slp"
    labels = sio.load_slp("tests/data/slp_hdf5/small_robot_minimal.slp")
    labels.replace_filenames(
        new_filenames=[str(Path("tests/data/videos/small_robot.mp4").absolute())]
    )
    sio.save_slp(labels, str(path))
    return str(path)


@pytest.fixture
def custom_profile(tmp_path):
    """A user's single-instance config with settings the form doesn't show."""
    cfg = OmegaConf.load(SINGLE_PROFILE)
    cfg.trainer_config.optimizer_name = "AdamW"
    cfg.trainer_config.lr_scheduler = {
        "step_lr": {"step_size": 7, "gamma": 0.5},
        "reduce_lr_on_plateau": None,
    }
    cfg.trainer_config.min_train_steps_per_epoch = 50
    cfg.model_config.init_weights = "xavier"
    cfg.data_config.augmentation_config.geometric.affine_p = 1.0
    # Dataset/run-specific values that must not carry over to a new run.
    cfg.data_config.val_labels_path = ["/other/project.val.slp"]
    cfg.data_config.test_file_path = "/other/project.test.slp"
    path = tmp_path / "custom.single.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


@pytest.fixture
def trained_model_config(tmp_path, custom_profile):
    """A trained single-instance model (config + best.ckpt) with affine_p = 1.0."""
    model_dir = tmp_path / "models" / "trained.single_instance"
    model_dir.mkdir(parents=True)
    shutil.copy(custom_profile, model_dir / "training_config.yaml")
    (model_dir / "best.ckpt").touch()
    return str(model_dir / "training_config.yaml")


def build_single_instance_config(qtbot, labels_path, config_path, form_data=None):
    """Select `config_path` in the dialog's config menu and build the training job."""
    labels = sio.load_slp(labels_path)
    dialog = LearningDialog(
        mode="training",
        labels_filename=labels_path,
        labels=labels,
        skeleton=labels.skeletons[0],
    )
    qtbot.addWidget(dialog)
    dialog.pipeline_form_widget.current_pipeline = "single"
    tab = dialog.tabs["single_instance"]
    tab._cfg_list_widget._add_file_selection_to_menu(
        configs.ConfigFileInfo.from_config_file(config_path)
    )
    assert tab._cfg_list_widget.getSelectedConfigInfo().path == config_path
    for form_name, data in (form_data or {}).items():
        tab.form_widgets[form_name].set_form_data(data)

    pipeline_form_data = dialog.pipeline_form_widget.get_form_data()
    (cfg_info,) = [
        cfg_info
        for cfg_info in dialog.get_every_head_config_data(pipeline_form_data)
        if cfg_info.head_name == "single_instance"
    ]
    return cfg_info.config


def geometric_aug_moves_points(geometric, n=30):
    """Whether sleap-nn's geometric augmentation moves keypoints with these params."""
    kwargs = {key: geometric[key] for key in GEOMETRIC_KEYS}
    image = torch.rand(1, 1, 64, 64)
    points = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
    return any(
        not torch.equal(
            apply_geometric_augmentation(image.clone(), points.clone(), **kwargs)[1],
            points,
        )
        for _ in range(n)
    )


def test_untrained_config_keeps_hidden_settings(qtbot, labels_path, custom_profile):
    """Settings the form doesn't show come from the selected config, not defaults."""
    cfg = build_single_instance_config(qtbot, labels_path, custom_profile)

    assert cfg.trainer_config.optimizer_name == "AdamW"
    assert cfg.trainer_config.lr_scheduler.step_lr.step_size == 7
    assert cfg.trainer_config.lr_scheduler.step_lr.gamma == 0.5
    assert cfg.trainer_config.min_train_steps_per_epoch == 50
    assert cfg.model_config.init_weights == "xavier"
    # The base config's own dataset/run fields don't carry over.
    assert cfg.data_config.val_labels_path is None
    assert cfg.data_config.test_file_path is None
    assert cfg.model_config.pretrained_backbone_weights is None
    # And the result is a valid sleap-nn training config.
    verify_training_cfg(filter_cfg(deepcopy(cfg)))


def test_form_values_override_untrained_config(qtbot, labels_path, custom_profile):
    """Values set in the form still win over the selected config."""
    cfg = build_single_instance_config(
        qtbot,
        labels_path,
        custom_profile,
        form_data={"optimization": {"trainer_config.max_epochs": 3}},
    )

    assert cfg.trainer_config.max_epochs == 3
    assert cfg.trainer_config.optimizer_name == "AdamW"


def test_untrained_config_does_not_modify_menu_copy(qtbot, labels_path, custom_profile):
    """Building a job loads a fresh copy of the selected config."""
    labels = sio.load_slp(labels_path)
    dialog = LearningDialog(
        mode="training",
        labels_filename=labels_path,
        labels=labels,
        skeleton=labels.skeletons[0],
    )
    qtbot.addWidget(dialog)
    dialog.pipeline_form_widget.current_pipeline = "single"
    tab = dialog.tabs["single_instance"]
    tab._cfg_list_widget._add_file_selection_to_menu(
        configs.ConfigFileInfo.from_config_file(custom_profile)
    )

    base = tab.untrained_config_info_to_use

    assert base.config.data_config.val_labels_path is None
    menu_cfg = tab._cfg_list_widget.getSelectedConfigInfo().config
    assert menu_cfg.data_config.val_labels_path == ["/other/project.val.slp"]


@pytest.mark.parametrize("config_fixture", ["custom_profile", "trained_model_config"])
def test_rotation_and_scale_off_disable_geometric_aug(
    qtbot, labels_path, request, config_fixture
):
    """Rotation Off + Scale unchecked turns geometric augmentation off.

    With both written as None, sleap-nn fell back to the base config's bundled
    `affine_p` (1.0 here, as in the bundled profiles) and rotated/scaled anyway.
    """
    config_path = request.getfixturevalue(config_fixture)

    cfg = build_single_instance_config(
        qtbot,
        labels_path,
        config_path,
        form_data={
            "augmentation": {"_rotation_preset": "Off", "_scale_enabled": False}
        },
    )

    # Validate the way `train_subprocess` does, which fills in sleap-nn defaults.
    geometric = verify_training_cfg(
        filter_cfg(deepcopy(cfg))
    ).data_config.augmentation_config.geometric
    assert geometric.affine_p == 1.0
    assert geometric.rotation_p == 0.0
    assert geometric.scale_p == 0.0
    assert not geometric_aug_moves_points(geometric)


def test_rotation_on_still_augments(qtbot, labels_path, custom_profile):
    """Turning rotation on still applies rotation."""
    cfg = build_single_instance_config(
        qtbot,
        labels_path,
        custom_profile,
        form_data={
            "augmentation": {"_rotation_preset": "±180°", "_scale_enabled": False}
        },
    )

    geometric = verify_training_cfg(
        filter_cfg(deepcopy(cfg))
    ).data_config.augmentation_config.geometric
    assert geometric.rotation_p == 1.0
    assert geometric.scale_p == 0.0
    assert geometric_aug_moves_points(geometric)


@pytest.mark.parametrize(
    "pipeline", ["single", "top-down", "bottom-up", "top-down-id", "bottom-up-id"]
)
def test_bundled_profiles_as_base(qtbot, pipeline):
    """The default bundled profiles build valid jobs when used as the base config.

    Bundled profiles have no `wandb` section and hard-code ZMQ ports 9000/9001.
    """
    labels_path = "tests/data/tracks/clip.slp"  # Has tracks, for the ID pipelines.
    labels = sio.load_slp(labels_path)
    dialog = LearningDialog(
        mode="training",
        labels_filename=labels_path,
        labels=labels,
        skeleton=labels.skeletons[0],
    )
    qtbot.addWidget(dialog)
    dialog.pipeline_form_widget.current_pipeline = pipeline

    cfg_infos = dialog.get_every_head_config_data(
        dialog.pipeline_form_widget.get_form_data()
    )

    assert cfg_infos
    for cfg_info in cfg_infos:
        selected = dialog.tabs[cfg_info.head_name]._cfg_list_widget
        assert "training_profiles" in selected.getSelectedConfigInfo().path
        cfg = verify_training_cfg(filter_cfg(deepcopy(cfg_info.config)))
        assert cfg.trainer_config.zmq.controller_port is None
        assert cfg.trainer_config.zmq.publish_port is None
        assert cfg.data_config.augmentation_config.geometric.rotation_p is not None
        assert cfg.data_config.augmentation_config.geometric.scale_p is not None
        if cfg_info.head_name.startswith("multi_class"):
            # Only set by these profiles, and not shown in the form.
            assert cfg.model_config.init_weights == "xavier"


def test_trained_config_is_not_an_untrained_base(
    qtbot, labels_path, trained_model_config
):
    """A config with a trained model goes through the trained-config path instead."""
    labels = sio.load_slp(labels_path)
    dialog = LearningDialog(
        mode="training",
        labels_filename=labels_path,
        labels=labels,
        skeleton=labels.skeletons[0],
    )
    qtbot.addWidget(dialog)
    dialog.pipeline_form_widget.current_pipeline = "single"
    tab = dialog.tabs["single_instance"]
    tab._cfg_list_widget._add_file_selection_to_menu(
        configs.ConfigFileInfo.from_config_file(trained_model_config)
    )

    assert tab.untrained_config_info_to_use is None
    assert tab.trained_config_info_to_use is not None

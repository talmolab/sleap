import shutil
from typing import Optional, List, Callable, Set
from pathlib import Path
import traceback

import cattr
import pytest
from qtpy import QtWidgets

from sleap.gui.learning.dialog import LearningDialog, TrainingEditorWidget
from sleap.gui.learning.configs import (
    TrainingConfigFilesWidget,
    ConfigFileInfo,
    TrainingConfigsGetter,
)
from sleap.gui.learning.scopedkeydict import (
    ScopedKeyDict,
    apply_cfg_transforms_to_key_val_dict,
)
from sleap.gui.app import MainWindow
from sleap.io.dataset import Labels
from sleap.nn.config import TrainingJobConfig, UNetConfig
from sleap.util import get_package_file


def test_use_hidden_params_from_loaded_config(
    qtbot, min_labels_slp, min_bottomup_model_path
):

    model_path = Path(min_bottomup_model_path)

    # Create a learning dialog
    app = MainWindow(no_usage_data=True)
    ld = LearningDialog(
        mode="training",
        labels_filename=Path(
            model_path
        ).parent.absolute(),  # Hack to get correct config
        labels=min_labels_slp,
    )

    # Make pipeline_form_widget
    ld.pipeline_form_widget.current_pipeline = "bottom-up"
    tab_name = "multi_instance"

    # Select a loaded config for pipeline form data
    bottom_up_tab: TrainingEditorWidget = ld.tabs[tab_name]
    cfg_list_widget: TrainingConfigFilesWidget = bottom_up_tab._cfg_list_widget
    cfg_list_widget.update()
    training_cfg_info: ConfigFileInfo = list(
        filter(
            lambda cfg: model_path.name in cfg.path,
            cfg_list_widget._cfg_list,
        )
    )[0]
    training_cfg_info_dict: dict = ScopedKeyDict.from_hierarchical_dict(
        cattr.unstructure(training_cfg_info)
    ).key_val_dict
    menu_idx = cfg_list_widget._cfg_list.index(training_cfg_info)
    cfg_list_widget.onSelectionIdxChange(menu_idx)

    # Make some changes to pipeline form data
    training_cfg_setting = training_cfg_info.config.data.preprocessing.input_scaling
    bottom_up_tab.set_fields_from_key_val_dict(
        {"data.preprocessing.input_scaling": training_cfg_setting - 0.1}
    )

    # Create config to use in new round of training
    pipeline_form_data = ld.pipeline_form_widget.get_form_data()
    config_info = ld.get_every_head_config_data(pipeline_form_data)[0]
    config_info_dict: dict = ScopedKeyDict.from_hierarchical_dict(
        cattr.unstructure(config_info)
    ).key_val_dict

    # Load pipeline form data
    tab_cfg_key_val_dict = bottom_up_tab.get_all_form_data()
    apply_cfg_transforms_to_key_val_dict(tab_cfg_key_val_dict)
    assert (
        tab_cfg_key_val_dict["data.preprocessing.input_scaling"] != training_cfg_setting
    )

    # Assert that config info list:
    params_set_in_tab = [f"config.{k}" for k in tab_cfg_key_val_dict.keys()]
    params_reset = [
        "config.data.labels.skeletons",
        "config.outputs.run_name",
        "config.outputs.run_name_suffix",
        "config.outputs.tags",
        "path",
        "filename",
    ]
    for k, _ in config_info_dict.items():
        if k in params_set_in_tab:
            # 1. Prefers data from widget over loaded config
            try:
                assert config_info_dict[k] == tab_cfg_key_val_dict[k[7:]]
            except:
                assert str(config_info_dict[k]) == tab_cfg_key_val_dict[k[7:]]
        elif k not in params_reset:
            # 2. Uses hidden parameters from loaded config
            assert config_info_dict[k] == training_cfg_info_dict[k]


def test_update_loaded_config():
    base_cfg = TrainingJobConfig()
    base_cfg.data.preprocessing.input_scaling = 0.5
    base_cfg.model.backbone.unet = UNetConfig(max_stride=32, output_stride=2)
    base_cfg.optimization.augmentation_config.rotation_max_angle = 180
    base_cfg.optimization.augmentation_config.rotation_min_angle = -180

    gui_vals = {
        "data.preprocessing.input_scaling": 1.0,
        "model.backbone.pretrained_encoder.encoder": "vgg16",
    }

    scoped_cfg = LearningDialog.update_loaded_config(base_cfg, gui_vals)
    assert scoped_cfg.key_val_dict["data.preprocessing.input_scaling"] == 1.0
    assert scoped_cfg.key_val_dict["model.backbone.unet"] is None
    assert (
        scoped_cfg.key_val_dict["model.backbone.pretrained_encoder.encoder"] == "vgg16"
    )
    assert (
        scoped_cfg.key_val_dict["optimization.augmentation_config.rotation_max_angle"]
        == 180
    )
    assert (
        scoped_cfg.key_val_dict["optimization.augmentation_config.rotation_min_angle"]
        == -180
    )


# Parameters: LearningDialog type (training, inference), config type (trained, untrained)
def test_training_editor_checkbox_states(
    qtbot, tmpdir, min_labels: Labels, min_centroid_model_path: str
):
    """Test that Use Trained Model and Resume Training checkboxes operate correctly."""

    def assert_checkbox_states(
        ted: TrainingEditorWidget,
        use_trained: Optional[bool] = None,
        resume_training: Optional[bool] = None,
    ):
        assert (
            ted._use_trained_model.isChecked() == use_trained
            if use_trained is not None
            else True
        )
        assert (
            ted._resume_training.isChecked() == resume_training
            if resume_training is not None
            else True
        )

    def switch_states(
        ted: TrainingEditorWidget,
        prev_use_trained: Optional[bool] = None,
        prev_resume_training: Optional[bool] = None,
        new_use_trained: Optional[bool] = None,
        new_resume_training: Optional[bool] = None,
    ):
        """Switch the states of the checkboxes."""

        # Assert previous checkbox state
        assert_checkbox_states(
            ted, use_trained=prev_use_trained, resume_training=prev_resume_training
        )

        # Switch states
        if new_use_trained is not None:
            ted._use_trained_model.setChecked(new_use_trained)
        if new_resume_training is not None:
            ted._resume_training.setChecked(new_resume_training)

        # Assert new checkbox state
        assert_checkbox_states(
            ted, use_trained=new_use_trained, resume_training=new_resume_training
        )

    def check_resume_training(
        ted: TrainingEditorWidget, prev_use_trained: Optional[bool] = None
    ):
        """Check the Resume Training checkbox."""
        switch_states(
            ted,
            prev_use_trained=prev_use_trained,
            new_use_trained=True,
            new_resume_training=True,
        )
        assert not ted.use_trained
        assert ted.resume_training

    def check_resume_training_00(ted: TrainingEditorWidget):
        """Check the Resume Training checkbox when Use Trained is unchecked."""
        check_resume_training(ted, prev_use_trained=False)

    def check_resume_training_10(ted: TrainingEditorWidget):
        """Check the Resume Training checkbox when Use Trained is checked."""
        check_resume_training(ted, prev_use_trained=True)

    def check_use_trained(ted: TrainingEditorWidget):
        """Check the Use Trained checkbox when Resume Training is unchecked."""
        switch_states(ted, prev_resume_training=False, new_use_trained=True)
        assert ted.use_trained
        assert not ted.resume_training

    def uncheck_resume_training(ted: TrainingEditorWidget):
        """Uncheck the Resume Training checkbox when Use Trained is checked."""
        switch_states(ted, prev_use_trained=True, new_resume_training=False)
        assert ted.use_trained
        assert not ted.resume_training

    def uncheck_use_trained(
        ted: TrainingEditorWidget, prev_resume_training: Optional[bool] = None
    ):
        """Uncheck the Use Trained checkbox."""
        switch_states(
            ted,
            prev_resume_training=prev_resume_training,
            new_use_trained=False,
            new_resume_training=False,
        )
        assert not ted.use_trained
        assert not ted.resume_training

    def uncheck_use_trained_10(ted: TrainingEditorWidget):
        """Uncheck the Use Trained checkbox when Resume Training is unchecked."""
        uncheck_use_trained(ted, prev_resume_training=False)

    def uncheck_use_trained_11(ted: TrainingEditorWidget):
        """Uncheck the Use Trained checkbox when Resume Training is checked."""
        uncheck_use_trained(ted, prev_resume_training=True)

    def assert_form_state(
        change_state: Callable,
        ted: TrainingEditorWidget,
        og_form_data: dict,
        reset_causing_actions: Set[Callable] = {
            check_use_trained,
            uncheck_resume_training,
        },
    ):
        expected_form_data = dict()
        actual_form_data = dict()

        # Read form values before changing state
        if change_state not in reset_causing_actions:
            for key in ted.form_widgets.keys():
                expected_form_data[key] = ted.form_widgets[key].get_form_data()

        # Change state
        change_state(ted)

        # Modify expected form values depending on state, and check if form is enabled
        if ted.resume_training:
            expected_form_data["model"] = og_form_data["model"]
        elif ted.use_trained:
            expected_form_data = og_form_data

        # Read form values after changing state
        for key in ted.form_widgets.keys():
            actual_form_data[key] = ted.form_widgets[key].get_form_data()
        assert expected_form_data == actual_form_data

    # Load the data
    labels: Labels = min_labels
    video = labels.video
    skeleton = labels.skeleton
    model_path = Path(min_centroid_model_path)
    head_name = (model_path.name).split(".")[-1]
    mode = "training"

    # Create a training TrainingEditorWidget
    cfg_getter = TrainingConfigsGetter(
        dir_paths=[str(model_path)], head_filter=head_name
    )
    ted = TrainingEditorWidget(
        video=video,
        skeleton=skeleton,
        head=head_name,
        cfg_getter=cfg_getter,
        require_trained=(mode == "inference"),
    )
    ted.update_file_list()

    og_form_data = dict()
    for key in ted.form_widgets.keys():
        og_form_data[key] = ted.form_widgets[key].get_form_data()

    # The action trajectory below should cover the entire state space of the checkboxes
    action_trajectory: List[Callable] = [
        check_resume_training_00,
        uncheck_use_trained_11,
        check_use_trained,
        check_resume_training_10,
        uncheck_resume_training,
        uncheck_use_trained_10,
    ]
    for action in action_trajectory:
        assert_form_state(action, ted, og_form_data)

    # TODO (LM): Add test for when an untrained model is selected (check that boxes are unchecked)
    # TODO (LM): Add test for when mode is inference (check that boxes are unchecked)


# TODO (LM): Remove after testing
if __name__ == "__main__":
    pytest.main([f"{__file__}::test_training_editor_checkbox_states", "-vv", "-rP"])

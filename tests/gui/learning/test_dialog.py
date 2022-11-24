from sleap.gui.learning.dialog import LearningDialog, TrainingEditorWidget
from sleap.gui.learning.configs import TrainingConfigFilesWidget
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.gui.learning.scopedkeydict import (
    make_training_config_from_key_val_dict,
    ScopedKeyDict,
    apply_cfg_transforms_to_key_val_dict,
)
from sleap.gui.app import MainWindow

import pytest
import cattr
from pathlib import Path
from qtpy import QtWidgets


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

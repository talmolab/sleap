"""Tests for config field bindings.

This module tests that GUI widget values are correctly mapped to config keys,
and that config transforms are applied correctly to produce valid sleap-nn configs.
"""

import pytest
from unittest.mock import patch

from sleap.gui.learning.main_tab import (
    MainTabWidget,
)
from sleap.gui.config_utils import (
    apply_cfg_transforms_to_key_val_dict,
    get_omegaconf_from_gui_form,
    filter_cfg,
    get_keyval_dict_from_omegaconf,
)
from sleap.gui.widgets.frame_target_selector import FrameTargetSelection


# =============================================================================
# Fixtures
# =============================================================================

# Mock preferences for clean state
CLEAN_PREFS = {
    "training data pipeline framework": None,
    "training num workers": None,
    "training accelerator": None,
    "training num devices": None,
}


@pytest.fixture
def training_widget(qtbot):
    """Create a MainTabWidget in training mode with clean preferences."""
    with patch("sleap.gui.learning.main_tab.prefs", CLEAN_PREFS):
        with patch(
            "sleap.gui.learning.main_tab.check_wandb_login_status",
            return_value=(False, None, None),
        ):
            widget = MainTabWidget(mode="training")
            qtbot.addWidget(widget)
            return widget


@pytest.fixture
def inference_widget(qtbot):
    """Create a MainTabWidget in inference mode."""
    widget = MainTabWidget(mode="inference")
    qtbot.addWidget(widget)
    return widget


# =============================================================================
# Pipeline Field Mapping Tests
# =============================================================================


class TestPipelineFieldMapping:
    """Tests for pipeline-specific field mappings."""

    def test_bottom_up_sigma_fields_mapping(self, training_widget):
        """Bottom-up sigma fields should map to correct config keys."""
        training_widget.current_pipeline = "bottom-up"

        # Set sigma values
        training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ].setValue(7.0)
        training_widget._fields[
            "model_config.head_configs.bottomup.pafs.sigma"
        ].setValue(20.0)

        data = training_widget.get_form_data()

        assert data["model_config.head_configs.bottomup.confmaps.sigma"] == 7.0
        assert data["model_config.head_configs.bottomup.pafs.sigma"] == 20.0

    def test_top_down_sigma_fields_mapping(self, training_widget):
        """Top-down sigma fields should map to correct config keys."""
        training_widget.current_pipeline = "top-down"

        # Set sigma values
        training_widget._fields[
            "model_config.head_configs.centroid.confmaps.sigma"
        ].setValue(5.0)
        training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.sigma"
        ].setValue(3.0)

        data = training_widget.get_form_data()

        assert data["model_config.head_configs.centroid.confmaps.sigma"] == 5.0
        assert data["model_config.head_configs.centered_instance.confmaps.sigma"] == 3.0

    def test_top_down_anchor_part_mapping(self, training_widget):
        """Top-down anchor part should map to correct config key."""
        training_widget.current_pipeline = "top-down"

        # Add some nodes to choose from
        training_widget.set_node_options(["head", "thorax", "tail"])

        # Select an anchor part
        anchor_combo = training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]
        idx = anchor_combo.findData("thorax")
        anchor_combo.setCurrentIndex(idx)

        data = training_widget.get_form_data()

        assert (
            data["model_config.head_configs.centered_instance.confmaps.anchor_part"]
            == "thorax"
        )

    def test_single_instance_sigma_mapping(self, training_widget):
        """Single instance sigma should map to correct config key."""
        training_widget.current_pipeline = "single"

        training_widget._fields[
            "model_config.head_configs.single_instance.confmaps.sigma"
        ].setValue(4.5)

        data = training_widget.get_form_data()

        assert data["model_config.head_configs.single_instance.confmaps.sigma"] == 4.5

    def test_bottom_up_id_sigma_fields_mapping(self, training_widget):
        """Bottom-up-id sigma fields should map to correct config keys."""
        training_widget.current_pipeline = "bottom-up-id"

        training_widget._fields[
            "model_config.head_configs.multi_class_bottomup.confmaps.sigma"
        ].setValue(6.0)
        training_widget._fields[
            "model_config.head_configs.multi_class_bottomup.class_maps.sigma"
        ].setValue(8.0)

        data = training_widget.get_form_data()

        assert (
            data["model_config.head_configs.multi_class_bottomup.confmaps.sigma"] == 6.0
        )
        assert (
            data["model_config.head_configs.multi_class_bottomup.class_maps.sigma"]
            == 8.0
        )

    def test_top_down_id_sigma_fields_mapping(self, training_widget):
        """Top-down-id sigma fields should map to correct config keys."""
        training_widget.current_pipeline = "top-down-id"

        training_widget._fields[
            "model_config.head_configs.centroid.confmaps.sigma"
        ].setValue(5.5)
        training_widget._fields[
            "model_config.head_configs.multi_class_topdown.confmaps.sigma"
        ].setValue(4.0)

        data = training_widget.get_form_data()

        assert data["model_config.head_configs.centroid.confmaps.sigma"] == 5.5
        assert (
            data["model_config.head_configs.multi_class_topdown.confmaps.sigma"] == 4.0
        )


# =============================================================================
# Performance Field Mapping Tests
# =============================================================================


class TestPerformanceFieldMapping:
    """Tests for performance section field mappings."""

    def test_accelerator_mapping(self, training_widget):
        """Accelerator dropdown should map to trainer_config.trainer_accelerator."""
        combo = training_widget._fields["trainer_config.trainer_accelerator"]
        idx = combo.findData("cuda")
        combo.setCurrentIndex(idx)

        data = training_widget.get_form_data()

        assert data["trainer_config.trainer_accelerator"] == "cuda"

    def test_devices_mapping(self, training_widget):
        """Devices spinbox should map to trainer_config.trainer_devices."""
        # Uncheck auto first
        training_widget._fields["_trainer_devices_auto"].setChecked(False)
        training_widget._fields["trainer_config.trainer_devices"].setValue(2)

        data = training_widget.get_form_data()

        assert data["trainer_config.trainer_devices"] == 2

    def test_devices_auto_mapping(self, training_widget):
        """Devices auto checkbox should map to _trainer_devices_auto."""
        training_widget._fields["_trainer_devices_auto"].setChecked(True)

        data = training_widget.get_form_data()

        assert data["_trainer_devices_auto"] is True

    def test_num_workers_mapping(self, training_widget):
        """Num workers should map to trainer_config.train_data_loader.num_workers."""
        training_widget._fields[
            "trainer_config.train_data_loader.num_workers"
        ].setValue(8)

        data = training_widget.get_form_data()

        assert data["trainer_config.train_data_loader.num_workers"] == 8

    def test_data_pipeline_mapping(self, training_widget):
        """Data pipeline dropdown should map to _data_pipeline_fw."""
        combo = training_widget._fields["_data_pipeline_fw"]
        idx = combo.findData("cache_disk")
        combo.setCurrentIndex(idx)

        data = training_widget.get_form_data()

        assert data["_data_pipeline_fw"] == "cache_disk"


# =============================================================================
# WandB Field Mapping Tests
# =============================================================================


class TestWandBFieldMapping:
    """Tests for WandB section field mappings."""

    def test_wandb_enable_mapping(self, training_widget):
        """WandB enable checkbox should map to trainer_config.use_wandb."""
        training_widget._fields["trainer_config.use_wandb"].setChecked(True)

        data = training_widget.get_form_data()

        assert data["trainer_config.use_wandb"] is True

    def test_wandb_upload_viz_mapping(self, training_widget):
        """WandB upload viz should map to trainer_config.wandb.save_viz_imgs_wandb."""
        training_widget._fields["trainer_config.wandb.save_viz_imgs_wandb"].setChecked(
            True
        )

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.save_viz_imgs_wandb"] is True

    def test_wandb_entity_mapping(self, training_widget):
        """WandB entity should map to trainer_config.wandb.entity."""
        training_widget._fields["trainer_config.wandb.entity"].setText("my-team")

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.entity"] == "my-team"

    def test_wandb_project_mapping(self, training_widget):
        """WandB project should map to trainer_config.wandb.project."""
        training_widget._fields["trainer_config.wandb.project"].setText("my-project")

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.project"] == "my-project"

    def test_wandb_api_key_mapping(self, training_widget):
        """WandB API key should map to trainer_config.wandb.api_key."""
        training_widget._fields["trainer_config.wandb.api_key"].setText("secret-key")

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.api_key"] == "secret-key"

    def test_wandb_prv_runid_mapping(self, training_widget):
        """WandB previous run ID should map to trainer_config.wandb.prv_runid."""
        training_widget._fields["trainer_config.wandb.prv_runid"].setText("run123")

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.prv_runid"] == "run123"

    def test_wandb_group_mapping(self, training_widget):
        """WandB group should map to trainer_config.wandb.group."""
        training_widget._fields["trainer_config.wandb.group"].setText("experiment-1")

        data = training_widget.get_form_data()

        assert data["trainer_config.wandb.group"] == "experiment-1"


# =============================================================================
# Output Field Mapping Tests
# =============================================================================


class TestOutputFieldMapping:
    """Tests for output section field mappings."""

    def test_run_name_mapping(self, training_widget):
        """Run name should map to trainer_config.run_name."""
        training_widget._fields["trainer_config.run_name"].setText("test_run")

        data = training_widget.get_form_data()

        assert data["trainer_config.run_name"] == "test_run"

    def test_ckpt_dir_mapping(self, training_widget):
        """Checkpoint dir should map to trainer_config.ckpt_dir."""
        training_widget._fields["trainer_config.ckpt_dir"].setText("custom_models")

        data = training_widget.get_form_data()

        assert data["trainer_config.ckpt_dir"] == "custom_models"

    def test_save_best_mapping(self, training_widget):
        """Save best checkbox should map to trainer_config.save_ckpt."""
        training_widget._fields["trainer_config.save_ckpt"].setChecked(False)

        data = training_widget.get_form_data()

        assert data["trainer_config.save_ckpt"] is False

    def test_save_latest_mapping(self, training_widget):
        """Save latest checkbox should map to trainer_config.model_ckpt.save_last."""
        training_widget._fields["trainer_config.model_ckpt.save_last"].setChecked(True)

        data = training_widget.get_form_data()

        assert data["trainer_config.model_ckpt.save_last"] is True

    def test_visualize_mapping(self, training_widget):
        """Visualize checkbox maps to visualize_preds_during_training."""
        training_widget._fields[
            "trainer_config.visualize_preds_during_training"
        ].setChecked(False)

        data = training_widget.get_form_data()

        assert data["trainer_config.visualize_preds_during_training"] is False

    def test_keep_viz_mapping(self, training_widget):
        """Keep viz checkbox should map to trainer_config.keep_viz."""
        training_widget._fields["trainer_config.keep_viz"].setChecked(True)

        data = training_widget.get_form_data()

        assert data["trainer_config.keep_viz"] is True


# =============================================================================
# Preprocessing Field Mapping Tests
# =============================================================================


class TestPreprocessingFieldMapping:
    """Tests for preprocessing section field mappings."""

    def test_ensure_channels_mapping(self, training_widget):
        """Color conversion should map to _ensure_channels."""
        combo = training_widget._fields["_ensure_channels"]
        idx = combo.findData("RGB")
        combo.setCurrentIndex(idx)

        data = training_widget.get_form_data()

        assert data["_ensure_channels"] == "RGB"

    def test_max_instances_mapping(self, training_widget):
        """Max instances should map to _max_instances."""
        training_widget._fields["_max_instances_disabled"].setChecked(False)
        training_widget._fields["_max_instances"].setValue(10)

        data = training_widget.get_form_data()

        assert data["_max_instances"] == 10

    def test_max_instances_disabled_mapping(self, training_widget):
        """Max instances disabled should map to _max_instances_disabled."""
        training_widget._fields["_max_instances_disabled"].setChecked(True)

        data = training_widget.get_form_data()

        assert data["_max_instances_disabled"] is True


# =============================================================================
# Tracker Field Mapping Tests (Inference Only)
# =============================================================================


class TestTrackerFieldMapping:
    """Tests for tracker section field mappings (inference mode only)."""

    def test_tracker_type_mapping(self, inference_widget):
        """Tracker dropdown should map to tracking.tracker."""
        combo = inference_widget._fields["tracking.tracker"]
        idx = combo.findData("flow")
        combo.setCurrentIndex(idx)

        data = inference_widget.get_form_data()

        assert data["tracking.tracker"] == "flow"

    def test_flow_tracker_max_tracks_mapping(self, inference_widget):
        """Flow tracker max tracks should map correctly."""
        # Disable the "No limit" checkbox
        inference_widget._fields["tracking.max_tracks_disabled.flow"].setChecked(False)
        inference_widget._fields["tracking.max_tracks.flow"].setValue(5)

        data = inference_widget.get_form_data()

        assert data["tracking.max_tracks.flow"] == 5

    def test_flow_tracker_similarity_mapping(self, inference_widget):
        """Flow tracker similarity should map correctly."""
        combo = inference_widget._fields["tracking.similarity.flow"]
        # Get actual available values from the combo
        available_values = [combo.itemData(i) for i in range(combo.count())]
        if "instance" in available_values:
            idx = combo.findData("instance")
        else:
            # Use first non-empty value
            idx = 0
        combo.setCurrentIndex(idx)

        data = inference_widget.get_form_data()

        # Verify the value matches what's in the combo
        assert data["tracking.similarity.flow"] == combo.currentData()

    def test_flow_tracker_match_mapping(self, inference_widget):
        """Flow tracker match should map correctly."""
        combo = inference_widget._fields["tracking.match.flow"]
        idx = combo.findData("greedy")
        combo.setCurrentIndex(idx)

        data = inference_widget.get_form_data()

        assert data["tracking.match.flow"] == "greedy"

    def test_flow_tracker_track_window_mapping(self, inference_widget):
        """Flow tracker track window should map correctly."""
        inference_widget._fields["tracking.track_window.flow"].setValue(10)

        data = inference_widget.get_form_data()

        assert data["tracking.track_window.flow"] == 10

    def test_simple_tracker_fields_mapping(self, inference_widget):
        """Simple tracker fields should map correctly."""
        # Disable the "No limit" checkbox
        inference_widget._fields["tracking.max_tracks_disabled.simple"].setChecked(
            False
        )
        inference_widget._fields["tracking.max_tracks.simple"].setValue(3)
        inference_widget._fields["tracking.track_window.simple"].setValue(15)

        data = inference_widget.get_form_data()

        assert data["tracking.max_tracks.simple"] == 3
        assert data["tracking.track_window.simple"] == 15


# =============================================================================
# Frame Target Selector Field Mapping Tests
# =============================================================================


class TestFrameTargetFieldMapping:
    """Tests for frame target selector field mappings."""

    def test_predict_target_mapping(self, training_widget):
        """Target selection should map to _predict_target."""
        training_widget.frame_target_selector.set_selection(
            FrameTargetSelection(target_key="video")
        )

        data = training_widget.get_form_data()

        assert data["_predict_target"] == "video"

    def test_exclude_user_labeled_mapping(self, training_widget):
        """Skip user labeled should map to _exclude_user_labeled."""
        training_widget.frame_target_selector.set_selection(
            FrameTargetSelection(exclude_user_labeled=True)
        )

        data = training_widget.get_form_data()

        assert data["_exclude_user_labeled"] is True

    def test_prediction_mode_mapping(self, training_widget):
        """Prediction mode should map to _prediction_mode."""
        training_widget.frame_target_selector.set_selection(
            FrameTargetSelection(prediction_mode="replace")
        )

        data = training_widget.get_form_data()

        assert data["_prediction_mode"] == "replace"

    def test_clear_all_first_mapping(self, training_widget):
        """Clear all first should map to _clear_all_first."""
        training_widget.frame_target_selector.set_selection(
            FrameTargetSelection(clear_all_first=True)
        )

        data = training_widget.get_form_data()

        assert data["_clear_all_first"] is True


# =============================================================================
# Config Transform Tests
# =============================================================================


class TestConfigTransforms:
    """Tests for apply_cfg_transforms_to_key_val_dict.

    Note: apply_cfg_transforms_to_key_val_dict requires
    trainer_config.train_data_loader.batch_size and num_workers to be present.
    """

    # Base data required by apply_cfg_transforms_to_key_val_dict
    BASE_DATA = {
        "trainer_config.train_data_loader.batch_size": 4,
        "trainer_config.train_data_loader.num_workers": 0,
    }

    def _make_data(self, **kwargs):
        """Create test data dict with required base keys plus custom keys."""
        data = self.BASE_DATA.copy()
        data.update(kwargs)
        return data

    def test_ensure_channels_rgb_transform(self):
        """_ensure_channels='RGB' should set ensure_rgb=True."""
        data = self._make_data(_ensure_channels="RGB")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.preprocessing.ensure_rgb"] is True
        assert data["data_config.preprocessing.ensure_grayscale"] is False

    def test_ensure_channels_grayscale_transform(self):
        """_ensure_channels='grayscale' should set ensure_grayscale=True."""
        data = self._make_data(_ensure_channels="grayscale")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.preprocessing.ensure_rgb"] is False
        assert data["data_config.preprocessing.ensure_grayscale"] is True

    def test_ensure_channels_none_transform(self):
        """Empty _ensure_channels should set both to False."""
        data = self._make_data(_ensure_channels="")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.preprocessing.ensure_rgb"] is False
        assert data["data_config.preprocessing.ensure_grayscale"] is False

    def test_data_pipeline_stream_transform(self):
        """Stream data pipeline should transform to torch_dataset."""
        data = self._make_data(**{"_data_pipeline_fw": "Stream (no caching)"})
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.data_pipeline_fw"] == "torch_dataset"

    def test_data_pipeline_cache_memory_transform(self):
        """Cache in Memory should transform correctly."""
        data = self._make_data(**{"_data_pipeline_fw": "Cache in Memory"})
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.data_pipeline_fw"] == "torch_dataset_cache_img_memory"

    def test_data_pipeline_cache_disk_transform(self):
        """Cache to Disk should transform correctly."""
        data = self._make_data(**{"_data_pipeline_fw": "Cache to Disk"})
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.data_pipeline_fw"] == "torch_dataset_cache_img_disk"

    def test_trainer_devices_auto_checked_sets_none(self):
        """When auto checkbox is checked, trainer_devices should be set to None."""
        data = self._make_data(
            **{
                "_trainer_devices_auto": True,
                "trainer_config.trainer_devices": 0,  # Ignored when auto
            }
        )
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["trainer_config.trainer_devices"] is None

    def test_trainer_devices_auto_unchecked_preserves_value(self):
        """When auto checkbox is unchecked, trainer_devices should be preserved."""
        data = self._make_data(
            **{
                "_trainer_devices_auto": False,
                "trainer_config.trainer_devices": 2,
            }
        )
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["trainer_config.trainer_devices"] == 2

    def test_trainer_devices_auto_not_present_preserves_value(self):
        """When auto checkbox is not in data, trainer_devices should be preserved."""
        data = self._make_data(**{"trainer_config.trainer_devices": 1})
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["trainer_config.trainer_devices"] == 1

    def test_rotation_preset_off_transform(self):
        """Rotation preset Off should set rotation_p to None."""
        data = self._make_data(_rotation_preset="Off")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.rotation_p"] is None

    def test_rotation_preset_15_transform(self):
        """Rotation preset ±15° should set rotation to ±15."""
        data = self._make_data(_rotation_preset="±15°")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.rotation_p"] == 1.0
        assert data["data_config.augmentation_config.geometric.rotation_min"] == -15
        assert data["data_config.augmentation_config.geometric.rotation_max"] == 15

    def test_rotation_preset_180_transform(self):
        """Rotation preset ±180° should set rotation to ±180."""
        data = self._make_data(_rotation_preset="±180°")
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.rotation_p"] == 1.0
        assert data["data_config.augmentation_config.geometric.rotation_min"] == -180
        assert data["data_config.augmentation_config.geometric.rotation_max"] == 180

    def test_rotation_preset_custom_transform(self):
        """Custom rotation should use _rotation_custom_angle."""
        data = self._make_data(_rotation_preset="Custom", _rotation_custom_angle=45)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.rotation_p"] == 1.0
        assert data["data_config.augmentation_config.geometric.rotation_min"] == -45
        assert data["data_config.augmentation_config.geometric.rotation_max"] == 45

    def test_scale_enabled_transform(self):
        """_scale_enabled=True should set scale_p=1.0."""
        data = self._make_data(_scale_enabled=True)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.scale_p"] == 1.0

    def test_scale_disabled_transform(self):
        """_scale_enabled=False should set scale_p=None."""
        data = self._make_data(_scale_enabled=False)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.geometric.scale_p"] is None

    def test_uniform_noise_enabled_transform(self):
        """_uniform_noise_enabled=True should set uniform_noise_p=1.0."""
        data = self._make_data(_uniform_noise_enabled=True)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.intensity.uniform_noise_p"] == 1.0

    def test_uniform_noise_disabled_transform(self):
        """_uniform_noise_enabled=False should set uniform_noise_p=0.0."""
        data = self._make_data(_uniform_noise_enabled=False)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.intensity.uniform_noise_p"] == 0.0

    def test_gaussian_noise_enabled_transform(self):
        """_gaussian_noise_enabled=True should set gaussian_noise_p=1.0."""
        data = self._make_data(_gaussian_noise_enabled=True)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.intensity.gaussian_noise_p"] == 1.0

    def test_contrast_enabled_transform(self):
        """_contrast_enabled=True should set contrast_p=1.0."""
        data = self._make_data(_contrast_enabled=True)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.intensity.contrast_p"] == 1.0

    def test_brightness_enabled_transform(self):
        """_brightness_enabled=True should set brightness_p=1.0."""
        data = self._make_data(_brightness_enabled=True)
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["data_config.augmentation_config.intensity.brightness_p"] == 1.0

    def test_batch_size_synced_to_val(self):
        """Train batch size should be synced to val batch size."""
        data = {
            "trainer_config.train_data_loader.batch_size": 16,
            "trainer_config.train_data_loader.num_workers": 4,
        }
        apply_cfg_transforms_to_key_val_dict(data)

        assert data["trainer_config.val_data_loader.batch_size"] == 16
        assert data["trainer_config.val_data_loader.num_workers"] == 4


# =============================================================================
# Filter Config Tests
# =============================================================================


class TestFilterConfig:
    """Tests for filter_cfg which removes underscore-prefixed keys."""

    def test_filter_removes_underscore_keys(self):
        """filter_cfg should remove keys starting with underscore."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "_pipeline": "bottom-up",
                "_ensure_channels": "RGB",
                "trainer_config": {
                    "use_wandb": True,
                    "_internal": "value",
                },
            }
        )

        filtered = filter_cfg(cfg)

        assert "_pipeline" not in filtered
        assert "_ensure_channels" not in filtered
        assert "trainer_config" in filtered
        assert filtered.trainer_config.use_wandb is True

    def test_filter_preserves_non_underscore_keys(self):
        """filter_cfg should preserve keys not starting with underscore."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "bottomup": {"confmaps": {"sigma": 5.0}},
                    }
                }
            }
        )

        filtered = filter_cfg(cfg)

        assert filtered.model_config.head_configs.bottomup.confmaps.sigma == 5.0


# =============================================================================
# OmegaConf Conversion Tests
# =============================================================================


class TestOmegaConfConversion:
    """Tests for conversion between flat dict and OmegaConf."""

    def test_flat_dict_to_omegaconf(self):
        """get_omegaconf_from_gui_form should create nested OmegaConf."""
        flat_dict = {
            "model_config.head_configs.bottomup.confmaps.sigma": 5.0,
            "trainer_config.use_wandb": True,
            "trainer_config.wandb.entity": "my-team",
        }

        cfg = get_omegaconf_from_gui_form(flat_dict)

        assert cfg.model_config.head_configs.bottomup.confmaps.sigma == 5.0
        assert cfg.trainer_config.use_wandb is True
        assert cfg.trainer_config.wandb.entity == "my-team"

    def test_omegaconf_to_flat_dict(self):
        """get_keyval_dict_from_omegaconf should flatten OmegaConf."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "bottomup": {"confmaps": {"sigma": 5.0}},
                    }
                },
                "trainer_config": {"use_wandb": True},
            }
        )

        flat = get_keyval_dict_from_omegaconf(cfg)

        assert flat["model_config.head_configs.bottomup.confmaps.sigma"] == 5.0
        assert flat["trainer_config.use_wandb"] is True

    def test_roundtrip_conversion(self):
        """Converting to OmegaConf and back should preserve values."""
        original = {
            "model_config.head_configs.bottomup.confmaps.sigma": 7.5,
            "trainer_config.use_wandb": False,
            "trainer_config.wandb.entity": "test",
        }

        cfg = get_omegaconf_from_gui_form(original)
        result = get_keyval_dict_from_omegaconf(cfg)

        for key, value in original.items():
            assert result[key] == value


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullConfigPipeline:
    """Tests for complete config generation pipeline.

    Note: MainTabWidget doesn't include all fields for cfg_transforms (like
    trainer_config.train_data_loader.batch_size in TrainingEditorWidget).
    These tests add required fields or skip transforms where appropriate.
    """

    # Required fields for apply_cfg_transforms_to_key_val_dict
    REQUIRED_TRANSFORM_FIELDS = {
        "trainer_config.train_data_loader.batch_size": 4,
        "trainer_config.train_data_loader.num_workers": 0,
    }

    def test_training_widget_to_config(self, training_widget):
        """Full pipeline from training widget to OmegaConf config."""
        # Configure widget
        training_widget.current_pipeline = "bottom-up"
        training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ].setValue(6.0)
        training_widget._fields["trainer_config.use_wandb"].setChecked(True)
        training_widget._fields["trainer_config.wandb.entity"].setText("my-org")

        # Get form data
        data = training_widget.get_form_data()

        # Add required fields for transform (normally from TrainingEditorWidget)
        data.update(self.REQUIRED_TRANSFORM_FIELDS)

        # Apply transforms
        apply_cfg_transforms_to_key_val_dict(data)

        # Convert to OmegaConf
        cfg = get_omegaconf_from_gui_form(data)

        # Verify
        assert cfg.model_config.head_configs.bottomup.confmaps.sigma == 6.0
        assert cfg.trainer_config.use_wandb is True
        assert cfg.trainer_config.wandb.entity == "my-org"

    def test_inference_widget_to_config(self, inference_widget):
        """Full pipeline from inference widget to OmegaConf config."""
        # Configure widget
        inference_widget.current_pipeline = "bottom-up"

        # Select flow tracker
        combo = inference_widget._fields["tracking.tracker"]
        idx = combo.findData("flow")
        combo.setCurrentIndex(idx)

        inference_widget._fields["tracking.track_window.flow"].setValue(20)

        # Get form data
        data = inference_widget.get_form_data()

        # Convert to OmegaConf (no transforms needed for inference-only fields)
        cfg = get_omegaconf_from_gui_form(data)

        # Verify tracking config (suffixed keys consolidated to unsuffixed)
        assert cfg.tracking.tracker == "flow"
        assert cfg.tracking.track_window == 20

    def test_filtered_config_ready_for_sleap_nn(self, training_widget):
        """Config should be properly filtered for sleap-nn after full pipeline."""
        # Configure widget
        training_widget.current_pipeline = "single"
        training_widget._fields[
            "model_config.head_configs.single_instance.confmaps.sigma"
        ].setValue(4.0)

        # Get and transform data
        data = training_widget.get_form_data()

        # Add required fields for transform (normally from TrainingEditorWidget)
        data.update(self.REQUIRED_TRANSFORM_FIELDS)

        apply_cfg_transforms_to_key_val_dict(data)

        # Convert to OmegaConf and filter
        cfg = get_omegaconf_from_gui_form(data)
        filtered_cfg = filter_cfg(cfg)

        # Verify underscore keys are removed
        assert not any(
            k.startswith("_") for k in get_keyval_dict_from_omegaconf(filtered_cfg)
        )

        # Verify essential config is preserved
        assert (
            filtered_cfg.model_config.head_configs.single_instance.confmaps.sigma == 4.0
        )


# =============================================================================
# Config Loading Reverse Mapping Tests
# =============================================================================


class TestConfigLoadingReverseMapping:
    """Tests for reverse mapping when loading configs into the GUI.

    These tests verify that augmentation settings from loaded configs
    (including legacy configs using affine_p) are correctly mapped to
    GUI form fields.
    """

    def test_rotation_preset_from_rotation_p(self):
        """rotation_p should be used to determine _rotation_preset."""
        key_val_dict = {
            "data_config.augmentation_config.geometric.rotation_min": -180,
            "data_config.augmentation_config.geometric.rotation_max": 180,
            "data_config.augmentation_config.geometric.rotation_p": 1.0,
        }

        # Simulate the reverse mapping logic from _load_config
        rot_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_min"
        )
        rot_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_max"
        )
        rot_p = key_val_dict.get("data_config.augmentation_config.geometric.rotation_p")
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        effective_rot_p = rot_p if rot_p is not None else affine_p

        assert effective_rot_p == 1.0
        assert rot_min == -180 and rot_max == 180

    def test_rotation_preset_fallback_to_affine_p(self):
        """affine_p should be used as fallback when rotation_p is None.

        This is the case for legacy/baseline configs that use affine_p
        instead of the new rotation_p/scale_p parameters.
        """
        # This is how baseline configs look - they have affine_p but no rotation_p
        key_val_dict = {
            "data_config.augmentation_config.geometric.rotation_min": -180.0,
            "data_config.augmentation_config.geometric.rotation_max": 180.0,
            "data_config.augmentation_config.geometric.affine_p": 1.0,
            # rotation_p is NOT present (None)
        }

        rot_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_min"
        )
        rot_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_max"
        )
        rot_p = key_val_dict.get("data_config.augmentation_config.geometric.rotation_p")
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        # Check rotation_p first, fall back to affine_p for legacy configs
        effective_rot_p = rot_p if rot_p is not None else affine_p

        # With affine_p=1.0 and rotation_min=-180, rotation_max=180,
        # should NOT be "Off"
        assert effective_rot_p == 1.0
        assert rot_min == -180.0 and rot_max == 180.0

        # Determine preset
        if effective_rot_p is None or effective_rot_p == 0:
            preset = "Off"
        elif rot_min == -180 and rot_max == 180:
            preset = "±180°"
        elif rot_min == -15 and rot_max == 15:
            preset = "±15°"
        else:
            preset = "Custom"

        assert preset == "±180°"

    def test_scale_enabled_from_scale_p(self):
        """scale_p should be used to determine _scale_enabled."""
        key_val_dict = {
            "data_config.augmentation_config.geometric.scale_min": 0.9,
            "data_config.augmentation_config.geometric.scale_max": 1.1,
            "data_config.augmentation_config.geometric.scale_p": 1.0,
        }

        scale_p = key_val_dict.get("data_config.augmentation_config.geometric.scale_p")
        scale_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_min"
        )
        scale_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_max"
        )
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        effective_scale_p = scale_p if scale_p is not None else affine_p

        scale_enabled = (
            effective_scale_p is not None
            and effective_scale_p > 0
            and scale_min is not None
            and scale_max is not None
            and scale_min != scale_max
        )

        assert scale_enabled is True

    def test_scale_enabled_fallback_to_affine_p(self):
        """affine_p should be used as fallback when scale_p is None.

        This is the case for legacy/baseline configs that use affine_p
        instead of the new rotation_p/scale_p parameters.
        """
        # This is how baseline configs look - they have affine_p but no scale_p
        key_val_dict = {
            "data_config.augmentation_config.geometric.scale_min": 0.9,
            "data_config.augmentation_config.geometric.scale_max": 1.1,
            "data_config.augmentation_config.geometric.affine_p": 1.0,
            # scale_p is NOT present (None)
        }

        scale_p = key_val_dict.get("data_config.augmentation_config.geometric.scale_p")
        scale_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_min"
        )
        scale_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_max"
        )
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        # Check scale_p first, fall back to affine_p for legacy configs
        effective_scale_p = scale_p if scale_p is not None else affine_p

        # Scale is enabled if probability > 0 AND min != max
        scale_enabled = (
            effective_scale_p is not None
            and effective_scale_p > 0
            and scale_min is not None
            and scale_max is not None
            and scale_min != scale_max
        )

        assert scale_enabled is True

    def test_scale_disabled_when_min_equals_max(self):
        """Scale should be disabled when min == max (no actual scaling)."""
        key_val_dict = {
            "data_config.augmentation_config.geometric.scale_min": 1.0,
            "data_config.augmentation_config.geometric.scale_max": 1.0,
            "data_config.augmentation_config.geometric.affine_p": 1.0,
        }

        scale_p = key_val_dict.get("data_config.augmentation_config.geometric.scale_p")
        scale_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_min"
        )
        scale_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.scale_max"
        )
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        effective_scale_p = scale_p if scale_p is not None else affine_p

        scale_enabled = (
            effective_scale_p is not None
            and effective_scale_p > 0
            and scale_min is not None
            and scale_max is not None
            and scale_min != scale_max
        )

        # scale_min == scale_max means no actual scaling occurs
        assert scale_enabled is False

    def test_rotation_off_when_affine_p_zero(self):
        """Rotation should be Off when affine_p is 0."""
        key_val_dict = {
            "data_config.augmentation_config.geometric.rotation_min": -180.0,
            "data_config.augmentation_config.geometric.rotation_max": 180.0,
            "data_config.augmentation_config.geometric.affine_p": 0.0,
        }

        rot_p = key_val_dict.get("data_config.augmentation_config.geometric.rotation_p")
        affine_p = key_val_dict.get(
            "data_config.augmentation_config.geometric.affine_p"
        )

        effective_rot_p = rot_p if rot_p is not None else affine_p

        if effective_rot_p is None or effective_rot_p == 0:
            preset = "Off"
        else:
            preset = "±180°"

        assert preset == "Off"

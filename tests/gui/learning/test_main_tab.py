"""Tests for MainTabWidget.

This module tests the native Qt main tab widget that replaces
TrainingPipelineWidget's YAML form builder approach.
"""

import pytest
from unittest.mock import patch
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
)

from sleap.gui.learning.main_tab import (
    MainTabWidget,
    PIPELINE_OPTIONS_TRAINING,
    PIPELINE_OPTIONS_INFERENCE,
)


# =============================================================================
# Fixtures
# =============================================================================


# Mock preferences that return None/default for all training settings
CLEAN_PREFS = {
    "training data pipeline framework": None,
    "training num workers": None,
    "training accelerator": None,
    "training num devices": None,
}


@pytest.fixture
def training_widget(qtbot):
    """Create a MainTabWidget in training mode."""
    widget = MainTabWidget(mode="training")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def training_widget_clean(qtbot):
    """Create a MainTabWidget in training mode with no saved preferences."""
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


@pytest.fixture
def training_widget_with_skeleton(qtbot, skeleton):
    """Create a MainTabWidget in training mode with skeleton for anchor parts."""
    widget = MainTabWidget(mode="training", skeleton=skeleton)
    qtbot.addWidget(widget)
    return widget


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestMainTabWidgetInstantiation:
    """Tests for MainTabWidget basic instantiation."""

    def test_training_mode_instantiation(self, training_widget):
        """Test that training mode widget instantiates correctly."""
        assert training_widget._mode == "training"
        assert training_widget._pipeline_combo is not None
        assert training_widget._pipeline_stack is not None
        assert training_widget.frame_target_selector is not None

    def test_inference_mode_instantiation(self, inference_widget):
        """Test that inference mode widget instantiates correctly."""
        assert inference_widget._mode == "inference"
        assert inference_widget._pipeline_combo is not None
        assert inference_widget._pipeline_stack is not None
        assert inference_widget.frame_target_selector is not None

    def test_training_mode_has_correct_pipeline_count(self, training_widget):
        """Training mode should have 5 pipeline options."""
        assert training_widget._pipeline_combo.count() == 5
        assert len(PIPELINE_OPTIONS_TRAINING) == 5

    def test_inference_mode_has_correct_pipeline_count(self, inference_widget):
        """Inference mode should have 8 pipeline options."""
        assert inference_widget._pipeline_combo.count() == 8
        assert len(PIPELINE_OPTIONS_INFERENCE) == 8

    def test_training_mode_has_wandb_section(self, training_widget):
        """Training mode should have WandB fields."""
        assert "trainer_config.use_wandb" in training_widget._fields
        assert "trainer_config.wandb.entity" in training_widget._fields
        assert "trainer_config.wandb.project" in training_widget._fields

    def test_inference_mode_no_wandb_section(self, inference_widget):
        """Inference mode should NOT have WandB fields."""
        assert "trainer_config.use_wandb" not in inference_widget._fields
        assert "trainer_config.wandb.entity" not in inference_widget._fields

    def test_training_mode_has_output_section(self, training_widget):
        """Training mode should have output fields."""
        assert "trainer_config.run_name" in training_widget._fields
        assert "trainer_config.ckpt_dir" in training_widget._fields
        assert "trainer_config.save_ckpt" in training_widget._fields

    def test_inference_mode_no_output_section(self, inference_widget):
        """Inference mode should NOT have output fields."""
        assert "trainer_config.run_name" not in inference_widget._fields
        assert "trainer_config.ckpt_dir" not in inference_widget._fields

    def test_training_mode_has_data_pipeline(self, training_widget):
        """Training mode should have data pipeline fields."""
        assert "_data_pipeline_fw" in training_widget._fields
        assert "trainer_config.train_data_loader.num_workers" in training_widget._fields

    def test_inference_mode_no_data_pipeline(self, inference_widget):
        """Inference mode should NOT have data pipeline fields."""
        assert "_data_pipeline_fw" not in inference_widget._fields
        assert (
            "trainer_config.train_data_loader.num_workers"
            not in inference_widget._fields
        )

    def test_inference_mode_has_tracker_section(self, inference_widget):
        """Inference mode should have tracker fields."""
        assert "tracking.tracker" in inference_widget._fields

    def test_training_mode_no_tracker_section(self, training_widget):
        """Training mode should NOT have tracker fields."""
        assert "tracking.tracker" not in training_widget._fields


# =============================================================================
# Pipeline Selection Tests
# =============================================================================


class TestPipelineSelection:
    """Tests for pipeline selection and switching."""

    def test_default_pipeline_is_bottom_up(self, training_widget):
        """Default pipeline should be multi-animal bottom-up."""
        assert training_widget.current_pipeline == "bottom-up"
        assert training_widget._pipeline_combo.currentIndex() == 0

    def test_set_pipeline_by_short_name(self, training_widget):
        """Should be able to set pipeline by short name."""
        training_widget.current_pipeline = "top-down"
        assert training_widget._pipeline_combo.currentIndex() == 1
        assert training_widget.current_pipeline == "top-down"

    def test_set_pipeline_single_animal(self, training_widget):
        """Should be able to set single animal pipeline."""
        training_widget.current_pipeline = "single"
        assert training_widget._pipeline_combo.currentIndex() == 4
        assert training_widget.current_pipeline == "single"

    def test_set_pipeline_id_variants(self, training_widget):
        """Should be able to set ID pipeline variants."""
        training_widget.current_pipeline = "bottom-up-id"
        assert training_widget.current_pipeline == "bottom-up-id"

        training_widget.current_pipeline = "top-down-id"
        assert training_widget.current_pipeline == "top-down-id"

    def test_stacked_widget_follows_dropdown(self, training_widget):
        """Stacked widget should update when dropdown changes."""
        training_widget._pipeline_combo.setCurrentIndex(0)
        assert training_widget._pipeline_stack.currentIndex() == 0

        training_widget._pipeline_combo.setCurrentIndex(1)
        assert training_widget._pipeline_stack.currentIndex() == 1

        training_widget._pipeline_combo.setCurrentIndex(2)
        assert training_widget._pipeline_stack.currentIndex() == 2

    def test_pipeline_change_emits_signal(self, training_widget, qtbot):
        """Pipeline change should emit updatePipeline signal."""
        with qtbot.waitSignal(training_widget.updatePipeline, timeout=1000) as blocker:
            training_widget._pipeline_combo.setCurrentIndex(1)

        assert blocker.args[0] == "top-down"

    def test_invalid_pipeline_name_ignored(self, training_widget):
        """Invalid pipeline name should be ignored."""
        original_index = training_widget._pipeline_combo.currentIndex()
        training_widget.current_pipeline = "invalid-pipeline"
        assert training_widget._pipeline_combo.currentIndex() == original_index


# =============================================================================
# Pipeline-Specific Fields Tests
# =============================================================================


class TestPipelineSpecificFields:
    """Tests for pipeline-specific field bindings."""

    def test_bottom_up_has_sigma_fields(self, training_widget):
        """Bottom-up pipeline should have sigma fields for nodes and edges."""
        assert (
            "model_config.head_configs.bottomup.confmaps.sigma"
            in training_widget._fields
        )
        assert (
            "model_config.head_configs.bottomup.pafs.sigma" in training_widget._fields
        )

    def test_top_down_has_anchor_and_sigma_fields(self, training_widget):
        """Top-down pipeline should have anchor part and sigma fields."""
        assert (
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
            in training_widget._fields
        )
        assert (
            "model_config.head_configs.centroid.confmaps.sigma"
            in training_widget._fields
        )
        assert (
            "model_config.head_configs.centered_instance.confmaps.sigma"
            in training_widget._fields
        )

    def test_single_animal_has_sigma_field(self, training_widget):
        """Single animal pipeline should have sigma field."""
        assert (
            "model_config.head_configs.single_instance.confmaps.sigma"
            in training_widget._fields
        )

    def test_sigma_field_is_double_spinbox(self, training_widget):
        """Sigma fields should be QDoubleSpinBox widgets."""
        sigma_field = training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ]
        assert isinstance(sigma_field, QDoubleSpinBox)

    def test_sigma_field_default_value(self, training_widget):
        """Sigma fields should have correct default values."""
        nodes_sigma = training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ]
        edges_sigma = training_widget._fields[
            "model_config.head_configs.bottomup.pafs.sigma"
        ]
        assert nodes_sigma.value() == 5.0
        assert edges_sigma.value() == 15.0

    def test_anchor_part_is_combobox(self, training_widget):
        """Anchor part field should be a QComboBox."""
        anchor_field = training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]
        assert isinstance(anchor_field, QComboBox)

    def test_anchor_part_has_empty_option(self, training_widget):
        """Anchor part dropdown should have empty option (bounding box midpoint)."""
        anchor_field = training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]
        # First item should be empty string
        assert anchor_field.itemText(0) == ""
        assert anchor_field.itemData(0) is None

    def test_anchor_part_populated_from_skeleton(
        self, training_widget_with_skeleton, skeleton
    ):
        """Anchor part should be populated with skeleton node names."""
        anchor_field = training_widget_with_skeleton._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]
        # Should have empty + all skeleton nodes
        assert anchor_field.count() == len(skeleton.node_names) + 1
        # Check that node names are in dropdown
        for i, name in enumerate(skeleton.node_names):
            assert anchor_field.itemText(i + 1) == name


# =============================================================================
# Form Data Tests
# =============================================================================


class TestFormData:
    """Tests for get_form_data and set_form_data."""

    def test_get_form_data_returns_dict(self, training_widget):
        """get_form_data should return a dictionary."""
        data = training_widget.get_form_data()
        assert isinstance(data, dict)

    def test_get_form_data_includes_pipeline(self, training_widget):
        """Form data should include _pipeline key."""
        data = training_widget.get_form_data()
        assert "_pipeline" in data
        assert data["_pipeline"] == "bottom-up"

    def test_get_form_data_includes_all_fields(self, training_widget):
        """Form data should include all registered fields."""
        data = training_widget.get_form_data()
        for key in training_widget._fields.keys():
            assert key in data, f"Missing field: {key}"

    def test_get_form_data_includes_frame_target(self, training_widget):
        """Form data should include frame target selector data."""
        data = training_widget.get_form_data()
        assert "_predict_target" in data
        assert "_exclude_user_labeled" in data
        assert "_prediction_mode" in data

    def test_set_form_data_pipeline(self, training_widget):
        """set_form_data should update pipeline selection."""
        training_widget.set_form_data({"_pipeline": "top-down"})
        assert training_widget.current_pipeline == "top-down"

    def test_set_form_data_checkbox(self, training_widget):
        """set_form_data should update checkbox fields."""
        training_widget.set_form_data({"trainer_config.use_wandb": True})
        checkbox = training_widget._fields["trainer_config.use_wandb"]
        assert checkbox.isChecked() is True

        training_widget.set_form_data({"trainer_config.use_wandb": False})
        assert checkbox.isChecked() is False

    def test_set_form_data_spinbox(self, training_widget):
        """set_form_data should update spinbox fields."""
        training_widget.set_form_data(
            {"trainer_config.train_data_loader.num_workers": 4}
        )
        spinbox = training_widget._fields[
            "trainer_config.train_data_loader.num_workers"
        ]
        assert spinbox.value() == 4

    def test_set_form_data_double_spinbox(self, training_widget):
        """set_form_data should update double spinbox fields."""
        training_widget.set_form_data(
            {"model_config.head_configs.bottomup.confmaps.sigma": 7.5}
        )
        spinbox = training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ]
        assert spinbox.value() == 7.5

    def test_set_form_data_combobox(self, training_widget):
        """set_form_data should update combobox fields."""
        training_widget.set_form_data({"trainer_config.trainer_accelerator": "cuda"})
        combo = training_widget._fields["trainer_config.trainer_accelerator"]
        assert combo.currentData() == "cuda"

    def test_set_form_data_lineedit(self, training_widget):
        """set_form_data should update line edit fields."""
        training_widget.set_form_data({"trainer_config.run_name": "test_run"})
        lineedit = training_widget._fields["trainer_config.run_name"]
        assert lineedit.text() == "test_run"

    def test_form_data_roundtrip(self, training_widget):
        """Setting and getting form data should be consistent."""
        original_data = {
            "_pipeline": "top-down",
            "trainer_config.use_wandb": True,
            "trainer_config.train_data_loader.num_workers": 8,
            "trainer_config.trainer_accelerator": "cuda",
            "trainer_config.run_name": "my_run",
        }
        training_widget.set_form_data(original_data)
        result_data = training_widget.get_form_data()

        for key, value in original_data.items():
            assert result_data[key] == value, f"Mismatch for {key}"


# =============================================================================
# Widget Value Tests
# =============================================================================


class TestWidgetValues:
    """Tests for individual widget value get/set."""

    def test_get_checkbox_value(self, training_widget):
        """Should get correct value from checkbox."""
        checkbox = training_widget._fields["trainer_config.use_wandb"]
        checkbox.setChecked(True)
        assert training_widget._get_widget_value(checkbox) is True

        checkbox.setChecked(False)
        assert training_widget._get_widget_value(checkbox) is False

    def test_get_spinbox_value(self, training_widget):
        """Should get correct value from spinbox."""
        spinbox = training_widget._fields["trainer_config.trainer_devices"]
        spinbox.setValue(2)
        assert training_widget._get_widget_value(spinbox) == 2

    def test_get_double_spinbox_value(self, training_widget):
        """Should get correct value from double spinbox."""
        spinbox = training_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ]
        spinbox.setValue(3.5)
        assert training_widget._get_widget_value(spinbox) == 3.5

    def test_get_combobox_value_with_data(self, training_widget):
        """Should get currentData from combobox when available."""
        combo = training_widget._fields["trainer_config.trainer_accelerator"]
        combo.setCurrentIndex(1)  # cuda
        assert training_widget._get_widget_value(combo) == "cuda"

    def test_get_lineedit_value(self, training_widget):
        """Should get text from line edit."""
        lineedit = training_widget._fields["trainer_config.run_name"]
        lineedit.setText("test_value")
        assert training_widget._get_widget_value(lineedit) == "test_value"

    def test_get_lineedit_empty_returns_none(self, training_widget):
        """Empty line edit should return None."""
        lineedit = training_widget._fields["trainer_config.run_name"]
        lineedit.setText("")
        assert training_widget._get_widget_value(lineedit) is None

    def test_get_lineedit_whitespace_returns_none(self, training_widget):
        """Whitespace-only line edit should return None."""
        lineedit = training_widget._fields["trainer_config.run_name"]
        lineedit.setText("   ")
        assert training_widget._get_widget_value(lineedit) is None


# =============================================================================
# Coupled Fields Tests
# =============================================================================


class TestCoupledFields:
    """Tests for coupled field behavior (checkbox enables/disables spinbox)."""

    def test_max_instances_disabled_by_default(self, training_widget):
        """Max instances spinbox should be disabled when 'No max' is checked."""
        spinbox = training_widget._fields["_max_instances"]
        checkbox = training_widget._fields["_max_instances_disabled"]

        # Default: No max is checked, spinbox is disabled
        assert checkbox.isChecked() is True
        assert spinbox.isEnabled() is False

    def test_max_instances_enabled_when_unchecked(self, training_widget):
        """Max instances spinbox should be enabled when 'No max' is unchecked."""
        spinbox = training_widget._fields["_max_instances"]
        checkbox = training_widget._fields["_max_instances_disabled"]

        checkbox.setChecked(False)
        assert spinbox.isEnabled() is True

    def test_devices_auto_disabled_by_default(self, training_widget_clean):
        """Devices spinbox should be disabled when 'Auto' is checked."""
        spinbox = training_widget_clean._fields["trainer_config.trainer_devices"]
        checkbox = training_widget_clean._fields["_trainer_devices_auto"]

        # Default (without saved preferences): Auto is checked, spinbox is disabled
        assert checkbox.isChecked() is True
        assert spinbox.isEnabled() is False

    def test_devices_enabled_when_auto_unchecked(self, training_widget):
        """Devices spinbox should be enabled when 'Auto' is unchecked."""
        spinbox = training_widget._fields["trainer_config.trainer_devices"]
        checkbox = training_widget._fields["_trainer_devices_auto"]

        checkbox.setChecked(False)
        assert spinbox.isEnabled() is True


# =============================================================================
# Tracker Section Tests (Inference Only)
# =============================================================================


class TestTrackerSection:
    """Tests for tracker section in inference mode."""

    def test_tracker_dropdown_options(self, inference_widget):
        """Tracker dropdown should have none, flow, simple options."""
        tracker_combo = inference_widget._fields["tracking.tracker"]
        assert tracker_combo.count() == 3
        assert tracker_combo.itemData(0) == "none"
        assert tracker_combo.itemData(1) == "flow"
        assert tracker_combo.itemData(2) == "simple"

    def test_tracker_default_is_none(self, inference_widget):
        """Default tracker should be 'none'."""
        tracker_combo = inference_widget._fields["tracking.tracker"]
        assert tracker_combo.currentData() == "none"

    def test_flow_tracker_fields_exist(self, inference_widget):
        """Flow tracker should have its specific fields."""
        assert "tracking.max_tracks.flow" in inference_widget._fields
        assert "tracking.similarity.flow" in inference_widget._fields
        assert "tracking.match.flow" in inference_widget._fields
        assert "tracking.track_window.flow" in inference_widget._fields
        assert "tracking.post_connect_single_breaks.flow" in inference_widget._fields

    def test_simple_tracker_fields_exist(self, inference_widget):
        """Simple tracker should have its specific fields."""
        assert "tracking.max_tracks.simple" in inference_widget._fields
        assert "tracking.similarity.simple" in inference_widget._fields
        assert "tracking.match.simple" in inference_widget._fields
        assert "tracking.track_window.simple" in inference_widget._fields
        assert "tracking.post_connect_single_breaks.simple" in inference_widget._fields

    def test_tracker_stack_switches_with_dropdown(self, inference_widget):
        """Tracker stacked widget should switch with dropdown."""
        tracker_combo = inference_widget._fields["tracking.tracker"]

        tracker_combo.setCurrentIndex(0)  # none
        assert inference_widget._tracker_stack.currentIndex() == 0

        tracker_combo.setCurrentIndex(1)  # flow
        assert inference_widget._tracker_stack.currentIndex() == 1

        tracker_combo.setCurrentIndex(2)  # simple
        assert inference_widget._tracker_stack.currentIndex() == 2

    def test_tracker_max_tracks_disabled_by_default(self, inference_widget):
        """Max tracks spinbox should be disabled by default (No limit checked)."""
        for tracker_type in ["flow", "simple"]:
            spinbox = inference_widget._fields[f"tracking.max_tracks.{tracker_type}"]
            checkbox = inference_widget._fields[
                f"tracking.max_tracks_disabled.{tracker_type}"
            ]
            assert checkbox.isChecked() is True
            assert spinbox.isEnabled() is False

    def test_robust_fields_exist(self, inference_widget):
        """Robust quantile fields should exist for flow and simple trackers."""
        for tracker_type in ["flow", "simple"]:
            robust_key = f"tracking.robust.{tracker_type}"
            disabled_key = f"tracking.robust_disabled.{tracker_type}"
            assert robust_key in inference_widget._fields
            assert disabled_key in inference_widget._fields

    def test_robust_disabled_by_default(self, inference_widget):
        """Robust quantile should be disabled by default (Use max checked)."""
        for tracker_type in ["flow", "simple"]:
            spinbox = inference_widget._fields[f"tracking.robust.{tracker_type}"]
            checkbox = inference_widget._fields[
                f"tracking.robust_disabled.{tracker_type}"
            ]
            assert checkbox.isChecked() is True
            assert spinbox.isEnabled() is False

    def test_robust_default_value(self, inference_widget):
        """Robust quantile default value should be 0.95."""
        for tracker_type in ["flow", "simple"]:
            spinbox = inference_widget._fields[f"tracking.robust.{tracker_type}"]
            assert spinbox.value() == 0.95

    def test_robust_consolidation_in_form_data(self, inference_widget):
        """Form data should consolidate robust field from tracker-specific key."""
        # Select flow tracker
        inference_widget._fields["tracking.tracker"].setCurrentIndex(1)
        # Enable robust and set a value
        inference_widget._fields["tracking.robust_disabled.flow"].setChecked(False)
        inference_widget._fields["tracking.robust.flow"].setValue(0.75)

        data = inference_widget.get_form_data()

        assert data["tracking.robust"] == 0.75
        assert data["tracking.robust_disabled"] is False

    def test_robust_disabled_sets_value_to_1(self, inference_widget):
        """When robust is disabled, form data should set it to 1.0."""
        # Select flow tracker (robust is disabled by default)
        inference_widget._fields["tracking.tracker"].setCurrentIndex(1)

        data = inference_widget.get_form_data()

        assert data["tracking.robust"] == 1.0
        assert data["tracking.robust_disabled"] is True


# =============================================================================
# Performance Section Tests
# =============================================================================


class TestPerformanceSection:
    """Tests for performance section fields."""

    def test_accelerator_options(self, training_widget):
        """Accelerator dropdown should have correct options."""
        combo = training_widget._fields["trainer_config.trainer_accelerator"]
        options = [combo.itemData(i) for i in range(combo.count())]
        assert "auto" in options
        assert "cuda" in options
        assert "cpu" in options
        assert "mps" in options

    def test_accelerator_default_is_auto(self, training_widget_clean):
        """Default accelerator should be 'auto' (no saved prefs)."""
        combo = training_widget_clean._fields["trainer_config.trainer_accelerator"]
        assert combo.currentData() == "auto"

    def test_data_pipeline_options(self, training_widget):
        """Data pipeline dropdown should have correct options."""
        combo = training_widget._fields["_data_pipeline_fw"]
        options = [combo.itemData(i) for i in range(combo.count())]
        assert "stream" in options
        assert "cache_memory" in options
        assert "cache_disk" in options

    def test_data_pipeline_default_is_cache_memory(self, training_widget):
        """Default data pipeline should be 'cache_memory'."""
        combo = training_widget._fields["_data_pipeline_fw"]
        assert combo.currentData() == "cache_memory"

    def test_batch_size_exists_in_inference(self, inference_widget):
        """Batch size field should exist in inference mode."""
        assert "_batch_size" in inference_widget._fields
        assert "_batch_size_default" in inference_widget._fields

    def test_batch_size_not_in_training(self, training_widget):
        """Batch size field should NOT exist in training mode."""
        assert "_batch_size" not in training_widget._fields
        assert "_batch_size_default" not in training_widget._fields

    def test_batch_size_default_checkbox_checked(self, inference_widget):
        """Default checkbox should be checked by default."""
        cb = inference_widget._fields["_batch_size_default"]
        assert cb.isChecked()

    def test_batch_size_spinbox_disabled_when_default(self, inference_widget):
        """Batch size spinbox should be disabled when default is checked."""
        spinbox = inference_widget._fields["_batch_size"]
        assert not spinbox.isEnabled()

    def test_batch_size_spinbox_enabled_when_unchecked(self, inference_widget):
        """Batch size spinbox should be enabled when default is unchecked."""
        cb = inference_widget._fields["_batch_size_default"]
        spinbox = inference_widget._fields["_batch_size"]

        cb.setChecked(False)
        assert spinbox.isEnabled()

    def test_batch_size_not_in_form_data_when_default(self, inference_widget):
        """Batch size should NOT be in form data when default is checked."""
        data = inference_widget.get_form_data()
        assert "_batch_size" not in data

    def test_batch_size_in_form_data_when_specified(self, inference_widget):
        """Batch size should be in form data when default is unchecked."""
        cb = inference_widget._fields["_batch_size_default"]
        spinbox = inference_widget._fields["_batch_size"]

        cb.setChecked(False)
        spinbox.setValue(16)

        data = inference_widget.get_form_data()
        assert "_batch_size" in data
        assert data["_batch_size"] == 16


# =============================================================================
# WandB Section Tests (Training Only)
# =============================================================================


class TestWandBSection:
    """Tests for WandB section in training mode."""

    def test_wandb_enable_checkbox_exists(self, training_widget):
        """WandB enable checkbox should exist."""
        assert "trainer_config.use_wandb" in training_widget._fields
        checkbox = training_widget._fields["trainer_config.use_wandb"]
        assert isinstance(checkbox, QCheckBox)

    def test_wandb_fields_exist(self, training_widget):
        """All WandB fields should exist."""
        wandb_fields = [
            "trainer_config.use_wandb",
            "trainer_config.wandb.save_viz_imgs_wandb",
            "trainer_config.wandb.api_key",
            "trainer_config.wandb.entity",
            "trainer_config.wandb.project",
            "trainer_config.wandb.prv_runid",
            "trainer_config.wandb.group",
        ]
        for field in wandb_fields:
            assert field in training_widget._fields, f"Missing WandB field: {field}"


# =============================================================================
# Output Section Tests (Training Only)
# =============================================================================


class TestOutputSection:
    """Tests for output section in training mode."""

    def test_output_fields_exist(self, training_widget):
        """All output fields should exist."""
        output_fields = [
            "trainer_config.run_name",
            "trainer_config.ckpt_dir",
            "trainer_config.save_ckpt",
            "trainer_config.model_ckpt.save_last",
            "trainer_config.visualize_preds_during_training",
            "trainer_config.keep_viz",
        ]
        for field in output_fields:
            assert field in training_widget._fields, f"Missing output field: {field}"

    def test_save_best_checked_by_default(self, training_widget):
        """Save best model should be checked by default."""
        checkbox = training_widget._fields["trainer_config.save_ckpt"]
        assert checkbox.isChecked() is True

    def test_visualize_checked_by_default(self, training_widget):
        """Visualize predictions should be checked by default."""
        checkbox = training_widget._fields[
            "trainer_config.visualize_preds_during_training"
        ]
        assert checkbox.isChecked() is True

    def test_runs_folder_default_value(self, training_widget):
        """Runs folder should default to 'models'."""
        lineedit = training_widget._fields["trainer_config.ckpt_dir"]
        assert lineedit.text() == "models"


# =============================================================================
# Preprocessing Section Tests
# =============================================================================


class TestPreprocessingSection:
    """Tests for preprocessing section fields."""

    def test_max_instances_exists(self, training_widget):
        """Max instances field should exist."""
        assert "_max_instances" in training_widget._fields
        assert "_max_instances_disabled" in training_widget._fields

    def test_ensure_channels_exists_in_training(self, training_widget):
        """Color conversion dropdown should exist in training mode."""
        assert "_ensure_channels" in training_widget._fields

    def test_ensure_channels_not_in_inference(self, inference_widget):
        """Color conversion dropdown should NOT exist in inference mode."""
        # Actually, looking at the code, _ensure_channels is only in training mode
        assert "_ensure_channels" not in inference_widget._fields

    def test_ensure_channels_options(self, training_widget):
        """Color conversion dropdown should have correct options."""
        combo = training_widget._fields["_ensure_channels"]
        options = [combo.itemData(i) for i in range(combo.count())]
        assert "" in options  # No conversion
        assert "RGB" in options
        assert "grayscale" in options


# =============================================================================
# Signal Tests
# =============================================================================


class TestSignals:
    """Tests for signal emissions."""

    def test_value_changed_on_checkbox_toggle(self, training_widget, qtbot):
        """valueChanged should emit when checkbox is toggled."""
        checkbox = training_widget._fields["trainer_config.use_wandb"]
        with qtbot.waitSignal(training_widget.valueChanged, timeout=1000):
            checkbox.setChecked(not checkbox.isChecked())

    def test_value_changed_on_spinbox_change(self, training_widget, qtbot):
        """valueChanged should emit when spinbox value changes."""
        spinbox = training_widget._fields["trainer_config.trainer_devices"]
        with qtbot.waitSignal(training_widget.valueChanged, timeout=1000):
            spinbox.setValue(spinbox.value() + 1)

    def test_value_changed_on_combobox_change(self, training_widget, qtbot):
        """valueChanged should emit when combobox selection changes."""
        combo = training_widget._fields["trainer_config.trainer_accelerator"]
        with qtbot.waitSignal(training_widget.valueChanged, timeout=1000):
            combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    def test_value_changed_on_lineedit_change(self, training_widget, qtbot):
        """valueChanged should emit when line edit text changes."""
        lineedit = training_widget._fields["trainer_config.run_name"]
        with qtbot.waitSignal(training_widget.valueChanged, timeout=1000):
            lineedit.setText("new_value")

    def test_update_pipeline_emits_on_change(self, training_widget, qtbot):
        """updatePipeline should emit when pipeline changes."""
        with qtbot.waitSignal(training_widget.updatePipeline, timeout=1000):
            training_widget._pipeline_combo.setCurrentIndex(1)


# =============================================================================
# Node Options Tests
# =============================================================================


class TestNodeOptions:
    """Tests for set_node_options method."""

    def test_set_node_options_populates_anchor_dropdowns(self, training_widget):
        """set_node_options should populate anchor part dropdowns."""
        node_names = ["head", "thorax", "tail"]
        training_widget.set_node_options(node_names)

        anchor_field = training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]
        # Should have empty + all nodes
        assert anchor_field.count() == len(node_names) + 1
        for i, name in enumerate(node_names):
            assert anchor_field.itemText(i + 1) == name

    def test_set_node_options_preserves_selection(self, training_widget):
        """set_node_options should preserve current selection if still valid."""
        # First set options
        training_widget.set_node_options(["head", "thorax", "tail"])
        anchor_field = training_widget._fields[
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
        ]

        # Select "thorax"
        idx = anchor_field.findData("thorax")
        anchor_field.setCurrentIndex(idx)
        assert anchor_field.currentData() == "thorax"

        # Update options (still includes thorax)
        training_widget.set_node_options(["head", "thorax", "abdomen"])

        # Selection should be preserved
        assert anchor_field.currentData() == "thorax"


# =============================================================================
# Frame Target Selector Integration Tests
# =============================================================================


class TestFrameTargetSelectorIntegration:
    """Tests for frame target selector integration."""

    def test_frame_target_selector_exists(self, training_widget):
        """Frame target selector should be accessible."""
        assert training_widget.frame_target_selector is not None

    def test_frame_target_in_training_mode(self, training_widget):
        """Frame target selector should be in training mode."""
        assert training_widget.frame_target_selector._mode == "training"

    def test_frame_target_in_inference_mode(self, inference_widget):
        """Frame target selector should be in inference mode."""
        assert inference_widget.frame_target_selector._mode == "inference"

    def test_frame_target_data_in_form_data(self, training_widget):
        """Form data should include frame target selector data."""
        data = training_widget.get_form_data()
        assert "_predict_target" in data
        assert "_exclude_user_labeled" in data
        assert "_prediction_mode" in data
        assert "_clear_all_first" in data

    def test_frame_target_value_change_emits_signal(self, training_widget, qtbot):
        """Frame target selector changes should emit valueChanged."""
        with qtbot.waitSignal(training_widget.valueChanged, timeout=1000):
            # Toggle skip user labeled checkbox
            training_widget.frame_target_selector.skip_user_labeled_cb.setChecked(True)

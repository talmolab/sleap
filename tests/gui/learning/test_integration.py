"""Integration tests for training/inference dialog.

This module tests the integration between components including:
- Config assembly from multiple dialog components
- Preference persistence across sessions
- Full dialog workflows
"""

import pytest
from unittest.mock import patch, MagicMock


from sleap_io import Labels, Skeleton, Video

from sleap.gui.learning.dialog import LearningDialog, TrainingEditorWidget
from sleap.gui.learning.main_tab import MainTabWidget
from sleap.gui.widgets.frame_target_selector import FrameTargetSelection


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_skeleton():
    """Create a minimal skeleton for testing."""
    skeleton = Skeleton(nodes=["head", "thorax", "tail"], name="test")
    skeleton.add_edge("head", "thorax")
    skeleton.add_edge("thorax", "tail")
    return skeleton


@pytest.fixture
def minimal_labels(minimal_skeleton):
    """Create minimal Labels with a skeleton."""
    return Labels(skeletons=[minimal_skeleton])


@pytest.fixture
def mock_cfg_getter():
    """Create a mock TrainingConfigsGetter."""
    getter = MagicMock()
    getter.get_first.return_value = None
    getter.update.return_value = None
    getter.get_filtered.return_value = []
    return getter


@pytest.fixture
def mock_prefs():
    """Create a mock preferences object."""
    prefs_data = {
        "training data pipeline framework": None,
        "training num workers": None,
        "training accelerator": None,
        "training num devices": None,
    }

    class MockPrefs:
        def __init__(self):
            self._data = prefs_data.copy()
            self._saved = False

        def __getitem__(self, key):
            return self._data.get(key)

        def __setitem__(self, key, value):
            self._data[key] = value

        def get(self, key, default=None):
            return self._data.get(key, default)

        def save(self):
            self._saved = True

    return MockPrefs()


@pytest.fixture
def training_dialog(qtbot, minimal_labels, minimal_skeleton, tmp_path, mock_cfg_getter):
    """Create a LearningDialog in training mode."""
    labels_file = tmp_path / "test.slp"
    labels_file.touch()

    with patch(
        "sleap.gui.learning.dialog.configs.TrainingConfigsGetter.make_from_labels_filename",
        return_value=mock_cfg_getter,
    ):
        dialog = LearningDialog(
            mode="training",
            labels_filename=str(labels_file),
            labels=minimal_labels,
            skeleton=minimal_skeleton,
        )
        qtbot.addWidget(dialog)
        return dialog


# =============================================================================
# Config Assembly Tests
# =============================================================================


class TestConfigAssembly:
    """Tests for config assembly from dialog components."""

    def test_merge_pipeline_and_head_config_data(self, training_dialog):
        """Pipeline data should merge with head data correctly."""
        head_data = {
            "model_config.head_configs.bottomup.confmaps.sigma": 5.0,
            "trainer_config.max_epochs": 100,
        }
        pipeline_data = {
            "model_config.head_configs.bottomup.confmaps.sigma": 6.0,  # Override
            "model_config.head_configs.bottomup.pafs.sigma": 15.0,  # Add
            "trainer_config.use_wandb": True,  # Add non-head-specific
            "model_config.head_configs.centroid.confmaps.sigma": 5.0,  # Different head
        }

        training_dialog.merge_pipeline_and_head_config_data(
            head_name="bottomup",
            head_data=head_data,
            pipeline_data=pipeline_data,
        )

        # bottomup fields should be merged
        assert head_data["model_config.head_configs.bottomup.confmaps.sigma"] == 6.0
        assert head_data["model_config.head_configs.bottomup.pafs.sigma"] == 15.0
        # Non-head-specific fields should be merged
        assert head_data["trainer_config.use_wandb"] is True
        # Other head's fields should NOT be merged
        assert "model_config.head_configs.centroid.confmaps.sigma" not in head_data

    def test_adjust_data_syncs_anchor_parts(self, training_dialog):
        """Anchor parts should be synced across all top-down heads."""
        source_data = {
            "model_config.head_configs.centered_instance.confmaps.anchor_part": "thorax"
        }
        updated_data = {}

        training_dialog.adjust_data_to_update_other_tabs(source_data, updated_data)

        # All anchor part fields should be synced
        assert (
            updated_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ]
            == "thorax"
        )
        assert (
            updated_data["model_config.head_configs.centroid.confmaps.anchor_part"]
            == "thorax"
        )
        assert (
            updated_data[
                "model_config.head_configs.multi_class_topdown.confmaps.anchor_part"
            ]
            == "thorax"
        )

    def test_adjust_data_centroid_anchor_syncs(self, training_dialog):
        """Centroid anchor part should sync to other heads."""
        source_data = {
            "model_config.head_configs.centroid.confmaps.anchor_part": "head"
        }
        updated_data = {}

        training_dialog.adjust_data_to_update_other_tabs(source_data, updated_data)

        assert (
            updated_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ]
            == "head"
        )

    def test_update_loaded_config(self, training_dialog):
        """update_loaded_config should override loaded config with GUI values."""
        loaded_cfg = {
            "trainer_config.max_epochs": 100,
            "trainer_config.use_wandb": False,
            "model_config.head_configs.bottomup.confmaps.sigma": 5.0,
        }
        gui_values = {
            "trainer_config.max_epochs": 200,  # Override
            "trainer_config.run_name": "new_run",  # Add new
        }

        result = LearningDialog.update_loaded_config(loaded_cfg, gui_values)

        assert result["trainer_config.max_epochs"] == 200
        assert result["trainer_config.run_name"] == "new_run"
        assert result["trainer_config.use_wandb"] is False  # Preserved
        assert result["model_config.head_configs.bottomup.confmaps.sigma"] == 5.0


class TestConfigAssemblyWithTabs:
    """Tests for config assembly involving TrainingEditorWidget tabs."""

    @pytest.fixture
    def dialog_with_tabs(
        self, qtbot, minimal_labels, minimal_skeleton, tmp_path, mock_cfg_getter
    ):
        """Create a dialog with tabs initialized for bottom-up pipeline."""
        labels_file = tmp_path / "test.slp"
        labels_file.touch()

        with patch(
            "sleap.gui.learning.dialog.configs.TrainingConfigsGetter.make_from_labels_filename",
            return_value=mock_cfg_getter,
        ):
            dialog = LearningDialog(
                mode="training",
                labels_filename=str(labels_file),
                labels=minimal_labels,
                skeleton=minimal_skeleton,
            )
            # Set to bottom-up to create the tab
            dialog.set_pipeline("bottom-up")
            qtbot.addWidget(dialog)
            return dialog

    def test_tabs_initialized_for_pipeline(self, dialog_with_tabs):
        """Tabs should be created for the selected pipeline."""
        assert "bottomup" in dialog_with_tabs.tabs
        assert isinstance(dialog_with_tabs.tabs["bottomup"], TrainingEditorWidget)

    def test_on_tab_data_change_updates_tabs(self, dialog_with_tabs):
        """on_tab_data_change should propagate data to tabs."""
        # Simulate pipeline form data change
        dialog_with_tabs.disconnect_signals()

        # Set some data in the pipeline widget
        dialog_with_tabs.pipeline_form_widget._fields[
            "model_config.head_configs.bottomup.confmaps.sigma"
        ].setValue(7.0)

        # Trigger data change
        dialog_with_tabs.on_tab_data_change(tab_name=None)

        dialog_with_tabs.connect_signals()


# =============================================================================
# Preferences Persistence Tests
# =============================================================================


class TestTrainingPreferences:
    """Tests for training settings preference persistence."""

    def test_init_training_settings_loads_prefs(self, qtbot, mock_prefs):
        """Training settings should be loaded from preferences on init."""
        mock_prefs["training accelerator"] = "cuda"
        mock_prefs["training num workers"] = 8
        mock_prefs["training num devices"] = 2

        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            with patch(
                "sleap.gui.learning.main_tab.check_wandb_login_status",
                return_value=(False, None, None),
            ):
                widget = MainTabWidget(mode="training")
                qtbot.addWidget(widget)

        # Check that values were loaded
        assert (
            widget._fields["trainer_config.trainer_accelerator"].currentData() == "cuda"
        )
        assert (
            widget._fields["trainer_config.train_data_loader.num_workers"].value() == 8
        )
        assert widget._fields["trainer_config.trainer_devices"].value() == 2
        # Auto should be unchecked when specific device count is loaded
        assert widget._fields["_trainer_devices_auto"].isChecked() is False

    def test_save_training_preferences(self, qtbot, mock_prefs):
        """Training settings should be saved to preferences."""
        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            with patch(
                "sleap.gui.learning.main_tab.check_wandb_login_status",
                return_value=(False, None, None),
            ):
                widget = MainTabWidget(mode="training")
                qtbot.addWidget(widget)

                # Set values
                accel_combo = widget._fields["trainer_config.trainer_accelerator"]
                idx = accel_combo.findData("cuda")
                accel_combo.setCurrentIndex(idx)

                widget._fields["trainer_config.train_data_loader.num_workers"].setValue(
                    4
                )

                # Save preferences
                form_data = widget.get_form_data()
                widget._save_training_preferences(form_data)

        # Check preferences were saved
        assert mock_prefs["training accelerator"] == "cuda"
        assert mock_prefs["training num workers"] == 4
        assert mock_prefs._saved is True

    def test_training_prefs_not_in_inference_mode(self, qtbot, mock_prefs):
        """Training-only preferences should not exist in inference mode."""
        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            widget = MainTabWidget(mode="inference")
            qtbot.addWidget(widget)

        # Training-only fields should not exist
        assert "_data_pipeline_fw" not in widget._fields
        assert "trainer_config.train_data_loader.num_workers" not in widget._fields


# =============================================================================
# Full Workflow Integration Tests
# =============================================================================


class TestFullWorkflow:
    """Tests for complete dialog workflows."""

    def test_training_workflow_form_data_complete(self, training_dialog):
        """Training workflow should produce complete form data."""
        # Set pipeline via the widget (not just the dialog)
        training_dialog.pipeline_form_widget.current_pipeline = "bottom-up"
        training_dialog.set_pipeline("bottom-up")

        # Get form data
        form_data = training_dialog.pipeline_form_widget.get_form_data()

        # Should have pipeline
        assert "_pipeline" in form_data
        assert form_data["_pipeline"] == "bottom-up"

        # Should have sigma fields
        assert "model_config.head_configs.bottomup.confmaps.sigma" in form_data
        assert "model_config.head_configs.bottomup.pafs.sigma" in form_data

        # Should have frame target data
        assert "_predict_target" in form_data
        assert "_prediction_mode" in form_data

    def test_inference_workflow_form_data_complete(
        self, qtbot, minimal_labels, minimal_skeleton, tmp_path, mock_cfg_getter
    ):
        """Inference workflow should produce complete form data."""
        labels_file = tmp_path / "test.slp"
        labels_file.touch()

        with patch(
            "sleap.gui.learning.dialog.configs.TrainingConfigsGetter.make_from_labels_filename",
            return_value=mock_cfg_getter,
        ):
            dialog = LearningDialog(
                mode="inference",
                labels_filename=str(labels_file),
                labels=minimal_labels,
                skeleton=minimal_skeleton,
            )
            qtbot.addWidget(dialog)

        # Get form data
        form_data = dialog.pipeline_form_widget.get_form_data()

        # Should have tracker data
        assert "tracking.tracker" in form_data

        # Should NOT have WandB (inference only)
        assert "trainer_config.use_wandb" not in form_data

        # Should have frame target data
        assert "_predict_target" in form_data

    def test_pipeline_switch_updates_form_data(self, training_dialog):
        """Switching pipeline should update available form data."""
        # Start with bottom-up
        training_dialog.set_pipeline("bottom-up")
        data1 = training_dialog.pipeline_form_widget.get_form_data()
        assert "model_config.head_configs.bottomup.confmaps.sigma" in data1

        # Switch to single
        training_dialog.set_pipeline("single")
        data2 = training_dialog.pipeline_form_widget.get_form_data()
        assert "model_config.head_configs.single_instance.confmaps.sigma" in data2

    def test_frame_selection_to_items_for_inference(self, training_dialog):
        """Frame selection should produce correct inference items."""
        mock_video = MagicMock(spec=Video)
        mock_video.__class__ = Video

        # Set up frame selection
        training_dialog._frame_selection = {
            "frame": {mock_video: [5]},
            "video": {mock_video: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        }

        # Select frame target
        training_dialog.frame_target_selector.set_selection(
            FrameTargetSelection(target_key="video")
        )

        # Get frames to predict
        frames = training_dialog.get_selected_frames_to_predict({})

        assert mock_video in frames
        assert len(frames[mock_video]) == 10


# =============================================================================
# Cross-Component Signal Tests
# =============================================================================


class TestCrossComponentSignals:
    """Tests for signals between components."""

    def test_pipeline_change_triggers_tab_update(self, training_dialog):
        """Pipeline change should trigger tab updates."""
        # Connect signal to track calls
        signal_received = []

        def on_pipeline_change(pipeline):
            signal_received.append(pipeline)

        training_dialog.pipeline_form_widget.updatePipeline.connect(on_pipeline_change)

        # Change pipeline
        training_dialog.pipeline_form_widget.current_pipeline = "single"

        assert "single" in signal_received

    def test_value_change_triggers_validation(self, training_dialog):
        """Value changes should trigger validation."""
        training_dialog.set_pipeline("bottom-up")

        # Validation should have been called (run button enabled/disabled)
        # Just verify no errors occur
        training_dialog._validate_pipeline()
        assert training_dialog.message_widget is not None


# =============================================================================
# Data Pipeline Tests
# =============================================================================


class TestDataPipelinePreferences:
    """Tests for data pipeline preference handling."""

    def test_data_pipeline_values(self, qtbot, mock_prefs):
        """Data pipeline dropdown should have correct values."""
        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            with patch(
                "sleap.gui.learning.main_tab.check_wandb_login_status",
                return_value=(False, None, None),
            ):
                widget = MainTabWidget(mode="training")
                qtbot.addWidget(widget)

        combo = widget._fields["_data_pipeline_fw"]
        values = [combo.itemData(i) for i in range(combo.count())]

        assert "stream" in values
        assert "cache_memory" in values
        assert "cache_disk" in values

    def test_data_pipeline_preference_loaded(self, qtbot, mock_prefs):
        """Saved data pipeline preference should be loaded."""
        mock_prefs["training data pipeline framework"] = "cache_disk"

        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            with patch(
                "sleap.gui.learning.main_tab.check_wandb_login_status",
                return_value=(False, None, None),
            ):
                widget = MainTabWidget(mode="training")
                qtbot.addWidget(widget)

        assert widget._fields["_data_pipeline_fw"].currentData() == "cache_disk"


# =============================================================================
# Color Conversion Integration Tests
# =============================================================================


class TestColorConversionIntegration:
    """Tests for color conversion field integration."""

    def test_ensure_channels_in_form_data(self, qtbot, mock_prefs):
        """Color conversion should be in form data."""
        with patch("sleap.gui.learning.main_tab.prefs", mock_prefs):
            with patch(
                "sleap.gui.learning.main_tab.check_wandb_login_status",
                return_value=(False, None, None),
            ):
                widget = MainTabWidget(mode="training")
                qtbot.addWidget(widget)

                # Set value
                combo = widget._fields["_ensure_channels"]
                idx = combo.findData("grayscale")
                combo.setCurrentIndex(idx)

        form_data = widget.get_form_data()
        assert form_data["_ensure_channels"] == "grayscale"

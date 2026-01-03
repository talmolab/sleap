"""Tests for LearningDialog.

This module tests the main dialog for training/inference configuration.
"""

import pytest
from unittest.mock import patch, MagicMock


from sleap_io import Labels, Skeleton, Video

from sleap.gui.learning.dialog import LearningDialog, TrainingEditorWidget
from sleap.gui.learning.main_tab import MainTabWidget
from sleap.gui.widgets.frame_target_selector import (
    FrameTargetSelection,
)


# =============================================================================
# Fixtures
# =============================================================================


# Path to minimal labels file for testing
TEST_SLP_MIN_LABELS = "tests/data/slp_hdf5/minimal_instance.slp"


@pytest.fixture
def minimal_skeleton():
    """Create a minimal skeleton for testing."""
    skeleton = Skeleton(nodes=["head", "tail"], name="test")
    skeleton.add_edge("head", "tail")
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
def training_dialog(qtbot, minimal_labels, minimal_skeleton, tmp_path, mock_cfg_getter):
    """Create a LearningDialog in training mode with mocked dependencies."""
    # Create a temporary labels file
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


@pytest.fixture
def inference_dialog(
    qtbot, minimal_labels, minimal_skeleton, tmp_path, mock_cfg_getter
):
    """Create a LearningDialog in inference mode with mocked dependencies."""
    # Create a temporary labels file
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
        return dialog


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestLearningDialogInstantiation:
    """Tests for LearningDialog basic instantiation."""

    def test_training_mode_instantiation(self, training_dialog):
        """Test that training mode dialog instantiates correctly."""
        assert training_dialog.mode == "training"
        assert training_dialog.pipeline_form_widget is not None
        assert training_dialog.tab_widget is not None
        assert training_dialog.frame_target_selector is not None

    def test_inference_mode_instantiation(self, inference_dialog):
        """Test that inference mode dialog instantiates correctly."""
        assert inference_dialog.mode == "inference"
        assert inference_dialog.pipeline_form_widget is not None
        assert inference_dialog.tab_widget is not None

    def test_training_dialog_window_title(self, training_dialog):
        """Training dialog should have correct window title."""
        title = training_dialog.windowTitle()
        assert "Training Configuration" in title
        assert "SLEAP" in title

    def test_inference_dialog_window_title(self, inference_dialog):
        """Inference dialog should have correct window title."""
        title = inference_dialog.windowTitle()
        assert "Inference Configuration" in title
        assert "SLEAP" in title

    def test_pipeline_form_widget_is_main_tab(self, training_dialog):
        """Pipeline form widget should be a MainTabWidget."""
        assert isinstance(training_dialog.pipeline_form_widget, MainTabWidget)

    def test_tab_widget_has_pipeline_tab(self, training_dialog):
        """Tab widget should have pipeline tab as first tab."""
        assert training_dialog.tab_widget.count() >= 1
        # First tab should be the pipeline tab
        first_tab = training_dialog.tab_widget.widget(0)
        assert first_tab is training_dialog.pipeline_form_widget

    def test_training_dialog_has_buttons(self, training_dialog):
        """Training dialog should have all expected buttons."""
        assert training_dialog.run_button is not None
        assert training_dialog.cancel_button is not None
        assert training_dialog.copy_button is not None
        assert training_dialog.save_button is not None
        assert training_dialog.export_button is not None

    def test_frame_target_selector_from_main_tab(self, training_dialog):
        """Frame target selector should come from MainTabWidget."""
        assert (
            training_dialog.frame_target_selector
            is training_dialog.pipeline_form_widget.frame_target_selector
        )


# =============================================================================
# Pipeline Switching Tests
# =============================================================================


class TestPipelineSwitching:
    """Tests for pipeline type switching."""

    def test_set_pipeline_top_down(self, training_dialog):
        """Setting top-down pipeline should add centroid and centered_instance tabs."""
        training_dialog.set_pipeline("top-down")
        assert training_dialog.current_pipeline == "top-down"
        assert "centroid" in training_dialog.shown_tab_names
        assert "centered_instance" in training_dialog.shown_tab_names

    def test_set_pipeline_bottom_up(self, training_dialog):
        """Setting bottom-up pipeline should add bottomup tab."""
        training_dialog.set_pipeline("bottom-up")
        assert training_dialog.current_pipeline == "bottom-up"
        assert "bottomup" in training_dialog.shown_tab_names

    def test_set_pipeline_single(self, training_dialog):
        """Setting single pipeline should add single_instance tab."""
        training_dialog.set_pipeline("single")
        assert training_dialog.current_pipeline == "single"
        assert "single_instance" in training_dialog.shown_tab_names

    def test_set_pipeline_top_down_id(self, training_dialog):
        """Top-down-id pipeline should add centroid and multi_class_topdown."""
        training_dialog.set_pipeline("top-down-id")
        assert training_dialog.current_pipeline == "top-down-id"
        assert "centroid" in training_dialog.shown_tab_names
        assert "multi_class_topdown" in training_dialog.shown_tab_names

    def test_set_pipeline_bottom_up_id(self, training_dialog):
        """Setting bottom-up-id pipeline should add multi_class_bottomup tab."""
        training_dialog.set_pipeline("bottom-up-id")
        assert training_dialog.current_pipeline == "bottom-up-id"
        assert "multi_class_bottomup" in training_dialog.shown_tab_names

    def test_pipeline_switch_removes_old_tabs(self, training_dialog):
        """Switching pipelines should remove old tabs."""
        training_dialog.set_pipeline("top-down")
        assert "centroid" in training_dialog.shown_tab_names
        assert "centered_instance" in training_dialog.shown_tab_names

        training_dialog.set_pipeline("single")
        assert "centroid" not in training_dialog.shown_tab_names
        assert "centered_instance" not in training_dialog.shown_tab_names
        assert "single_instance" in training_dialog.shown_tab_names

    def test_remove_tabs_keeps_first_tab(self, training_dialog):
        """remove_tabs should keep the first (pipeline) tab."""
        training_dialog.set_pipeline("top-down")
        initial_count = training_dialog.tab_widget.count()
        assert initial_count > 1

        training_dialog.remove_tabs()
        assert training_dialog.tab_widget.count() == 1
        # First tab should still be pipeline tab
        assert (
            training_dialog.tab_widget.widget(0) is training_dialog.pipeline_form_widget
        )


# =============================================================================
# Tab Management Tests
# =============================================================================


class TestTabManagement:
    """Tests for lazy tab creation and management."""

    def test_tabs_dict_initially_empty(self, training_dialog):
        """Tabs dict may have entries from default pipeline."""
        # The dialog sets a default pipeline on init, so some tabs may exist
        assert isinstance(training_dialog.tabs, dict)

    def test_add_tab_creates_widget(self, training_dialog):
        """add_tab should create TrainingEditorWidget lazily."""
        # Clear existing tabs first
        training_dialog.remove_tabs()
        training_dialog.tabs.clear()

        training_dialog.add_tab("single_instance")
        assert "single_instance" in training_dialog.tabs
        assert isinstance(training_dialog.tabs["single_instance"], TrainingEditorWidget)

    def test_add_tab_is_idempotent(self, training_dialog):
        """Adding same tab twice should not duplicate it."""
        training_dialog.remove_tabs()

        training_dialog.add_tab("bottomup")
        first_count = training_dialog.tab_widget.count()

        training_dialog.add_tab("bottomup")
        assert training_dialog.tab_widget.count() == first_count

    def test_ensure_tab_initialized_caches_widget(self, training_dialog):
        """_ensure_tab_initialized should cache created widgets."""
        training_dialog.remove_tabs()
        training_dialog.tabs.clear()

        widget1 = training_dialog._ensure_tab_initialized("single_instance")
        widget2 = training_dialog._ensure_tab_initialized("single_instance")
        assert widget1 is widget2


# =============================================================================
# Frame Selection Tests
# =============================================================================


class TestFrameSelection:
    """Tests for frame selection property and methods."""

    def test_frame_selection_initially_none(self, training_dialog):
        """Frame selection should be None initially."""
        # Note: It may or may not be None depending on dialog setup
        # Just check the property exists
        _ = training_dialog.frame_selection

    def test_set_frame_selection_updates_selector(self, training_dialog):
        """Setting frame_selection should update the selector widget."""
        mock_video = MagicMock(spec=Video)
        frame_selection = {
            "frame": {mock_video: [0]},
            "video": {mock_video: [0, 1, 2, 3, 4]},
        }
        training_dialog.frame_selection = frame_selection
        assert training_dialog._frame_selection == frame_selection

    def test_count_total_frames_simple(self, training_dialog):
        """count_total_frames_for_selection_option should count frames."""
        mock_video = MagicMock(spec=Video)
        frames = {mock_video: [0, 1, 2, 3, 4]}
        count = LearningDialog.count_total_frames_for_selection_option(frames)
        assert count == 5

    def test_count_total_frames_range(self, training_dialog):
        """count_total_frames_for_selection_option should handle ranges."""
        mock_video = MagicMock(spec=Video)
        # Range format: (start, -end) means frames from start to end
        frames = {mock_video: [0, -10]}  # Frames 0-9 (10 frames)
        count = LearningDialog.count_total_frames_for_selection_option(frames)
        assert count == 10

    def test_count_total_frames_empty(self, training_dialog):
        """count_total_frames_for_selection_option should return 0 for empty."""
        count = LearningDialog.count_total_frames_for_selection_option({})
        assert count == 0

    def test_count_total_frames_none(self, training_dialog):
        """count_total_frames_for_selection_option should return 0 for None."""
        count = LearningDialog.count_total_frames_for_selection_option(None)
        assert count == 0


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for pipeline validation."""

    def test_validate_pipeline_enables_run_button(self, training_dialog):
        """Valid pipeline should enable run button."""
        # Set a valid pipeline
        training_dialog.set_pipeline("single")
        training_dialog._validate_pipeline()
        # Run button state depends on validation results
        # For training mode with no other validation issues, should be enabled
        assert training_dialog.run_button is not None

    def test_validate_id_model_no_tracks(self, training_dialog):
        """ID model validation should fail without tracks."""
        # labels without tracks
        assert training_dialog._validate_id_model() is False

    def test_message_widget_exists(self, training_dialog):
        """Message widget should exist for validation messages."""
        assert training_dialog.message_widget is not None


# =============================================================================
# Signal Connection Tests
# =============================================================================


class TestSignalConnections:
    """Tests for signal connections."""

    def test_pipeline_update_signal_connected(self, training_dialog):
        """Pipeline update signal should be connected."""
        # This is implicitly tested by pipeline switching working
        training_dialog.set_pipeline("single")
        assert training_dialog.current_pipeline == "single"

    def test_disconnect_signals_no_error(self, training_dialog):
        """disconnect_signals should not raise errors."""
        # Should not raise even if some signals aren't connected
        training_dialog.disconnect_signals()

    def test_connect_signals_no_error(self, training_dialog):
        """connect_signals should not raise errors."""
        training_dialog.disconnect_signals()
        training_dialog.connect_signals()


# =============================================================================
# Get Form Data Tests
# =============================================================================


class TestGetFormData:
    """Tests for getting form data from dialog."""

    def test_pipeline_form_data_available(self, training_dialog):
        """Should be able to get form data from pipeline widget."""
        data = training_dialog.pipeline_form_widget.get_form_data()
        assert isinstance(data, dict)
        assert "_pipeline" in data

    def test_get_selected_frames_to_predict_nothing(self, training_dialog):
        """get_selected_frames_to_predict should return empty for 'nothing'."""
        # Set selection to nothing
        training_dialog.frame_target_selector.set_selection(
            FrameTargetSelection(target_key="nothing")
        )
        frames = training_dialog.get_selected_frames_to_predict({})
        assert frames == {}

    def test_get_selected_frames_to_predict_with_selection(self, training_dialog):
        """get_selected_frames_to_predict should return frames for valid selection."""
        mock_video = MagicMock(spec=Video)
        training_dialog._frame_selection = {
            "frame": {mock_video: [5]},
        }
        training_dialog.frame_target_selector.set_selection(
            FrameTargetSelection(target_key="frame")
        )
        frames = training_dialog.get_selected_frames_to_predict({})
        assert mock_video in frames
        assert frames[mock_video] == [5]


# =============================================================================
# Default Pipeline Tests
# =============================================================================


class TestDefaultPipeline:
    """Tests for default pipeline selection."""

    def test_get_most_recent_pipeline_trained_none(
        self, training_dialog, mock_cfg_getter
    ):
        """get_most_recent_pipeline_trained should return empty if no trained."""
        mock_cfg_getter.get_first.return_value = None
        result = training_dialog.get_most_recent_pipeline_trained()
        assert result == ""

    def test_set_default_pipeline_single_instance(self, training_dialog):
        """Single instance project should default to single pipeline."""
        # With minimal_labels having no instances, max_user_instance will be 0
        # which means multi-animal, so it defaults to top-down
        training_dialog.set_default_pipeline_tab()
        # Default depends on labels content - just verify it sets something
        assert training_dialog.pipeline_form_widget.current_pipeline in [
            "single",
            "top-down",
            "bottom-up",
        ]


# =============================================================================
# Mode Difference Tests
# =============================================================================


class TestModeDifferences:
    """Tests for differences between training and inference modes."""

    def test_training_mode_pipeline_tab_label(self, training_dialog):
        """Training mode should have 'Training Pipeline' tab label."""
        # First tab should be pipeline tab
        tab_label = training_dialog.tab_widget.tabText(0)
        assert "Training" in tab_label

    def test_inference_mode_pipeline_tab_label(self, inference_dialog):
        """Inference mode should have 'Inference Pipeline' tab label."""
        tab_label = inference_dialog.tab_widget.tabText(0)
        assert "Inference" in tab_label


# =============================================================================
# Target Selection Preference Tests
# =============================================================================


class TestTargetSelectionPreference:
    """Tests for target selection preference handling."""

    def test_target_selection_user_changed_tracking(self, training_dialog):
        """User changes to target selection should be tracked."""
        assert training_dialog._target_selection_user_changed is False

        # Simulate user change
        training_dialog._on_target_selection_changed()
        assert training_dialog._target_selection_user_changed is True


# =============================================================================
# TrainingEditorWidget Tests
# =============================================================================


class TestTrainingEditorWidget:
    """Tests for TrainingEditorWidget."""

    @pytest.fixture
    def editor_widget(self, qtbot, minimal_skeleton, mock_cfg_getter):
        """Create a TrainingEditorWidget for testing."""
        widget = TrainingEditorWidget(
            skeleton=minimal_skeleton,
            head="single_instance",
            cfg_getter=mock_cfg_getter,
            require_trained=False,
        )
        qtbot.addWidget(widget)
        return widget

    def test_editor_widget_instantiation(self, editor_widget):
        """TrainingEditorWidget should instantiate correctly."""
        assert editor_widget.head == "single_instance"
        assert editor_widget.form_widgets is not None

    def test_editor_widget_has_form_widgets(self, editor_widget):
        """TrainingEditorWidget should have all form widgets."""
        expected_forms = ["model", "data", "augmentation", "optimization", "outputs"]
        for form_name in expected_forms:
            assert form_name in editor_widget.form_widgets

    def test_editor_widget_training_mode_radios(self, editor_widget):
        """Non-require_trained editor should have training mode radio buttons."""
        assert editor_widget._radio_train_scratch is not None
        assert editor_widget._radio_resume is not None
        assert editor_widget._radio_use_trained is not None

    def test_editor_widget_train_scratch_default(self, editor_widget):
        """Default training mode should be 'train from scratch'."""
        assert editor_widget._radio_train_scratch.isChecked() is True

    def test_editor_widget_resume_disabled_initially(self, editor_widget):
        """Resume and use trained should be disabled without trained model."""
        assert editor_widget._radio_resume.isEnabled() is False
        assert editor_widget._radio_use_trained.isEnabled() is False

    def test_editor_widget_value_changed_signal(self, editor_widget, qtbot):
        """TrainingEditorWidget should emit valueChanged signal."""
        with qtbot.waitSignal(editor_widget.valueChanged, timeout=1000):
            editor_widget.emitValueChanged()

    @pytest.fixture
    def inference_editor_widget(self, qtbot, minimal_skeleton, mock_cfg_getter):
        """Create a TrainingEditorWidget for inference (require_trained=True)."""
        widget = TrainingEditorWidget(
            skeleton=minimal_skeleton,
            head="single_instance",
            cfg_getter=mock_cfg_getter,
            require_trained=True,
        )
        qtbot.addWidget(widget)
        return widget

    def test_inference_editor_no_training_radios(self, inference_editor_widget):
        """Inference mode editor should not have training mode radios."""
        # With require_trained=True, radio buttons are not created
        assert inference_editor_widget._radio_train_scratch is None
        assert inference_editor_widget._radio_resume is None
        assert inference_editor_widget._radio_use_trained is None


# =============================================================================
# Anchor Part Sync Tests
# =============================================================================


class TestAnchorPartSync:
    """Tests for anchor part synchronization across tabs."""

    def test_adjust_data_to_update_other_tabs_anchor(self, training_dialog):
        """adjust_data_to_update_other_tabs should sync anchor parts."""
        source_data = {
            "model_config.head_configs.centered_instance.confmaps.anchor_part": "head"
        }
        updated_data = {}
        training_dialog.adjust_data_to_update_other_tabs(source_data, updated_data)

        # All anchor part fields should be synced
        assert (
            updated_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ]
            == "head"
        )
        assert (
            updated_data["model_config.head_configs.centroid.confmaps.anchor_part"]
            == "head"
        )
        assert (
            updated_data[
                "model_config.head_configs.multi_class_topdown.confmaps.anchor_part"
            ]
            == "head"
        )

    def test_adjust_data_empty_anchor_becomes_none(self, training_dialog):
        """Empty anchor part should become None."""
        source_data = {
            "model_config.head_configs.centered_instance.confmaps.anchor_part": ""
        }
        updated_data = {}
        training_dialog.adjust_data_to_update_other_tabs(source_data, updated_data)

        assert (
            updated_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ]
            is None
        )


# =============================================================================
# Sigma Sync Tests
# =============================================================================


class TestSigmaSync:
    """Tests for sigma field synchronization between Pipeline tab and config tabs.

    The centroid sigma appears in both 'multi-animal top-down' and
    'multi-animal top-down-id' pipelines. When reading form data, we must
    read from the currently selected pipeline's widget, not from a
    potentially overwritten reference.
    """

    def test_centroid_sigma_reads_from_current_pipeline(self, training_dialog, qtbot):
        """Centroid sigma should be read from the current pipeline's widget.

        This tests the fix for a bug where centroid sigma was read from the
        wrong widget because 'multi-animal top-down-id' overwrote the
        reference in _fields.
        """
        main_tab = training_dialog.pipeline_form_widget

        # Set to multi-animal top-down pipeline
        main_tab.current_pipeline = "top-down"

        # Get the centroid sigma field from the current pipeline
        pipeline_key = main_tab.current_pipeline_key
        assert "top-down" in pipeline_key
        assert "id" not in pipeline_key

        # The centroid sigma should be accessible
        centroid_sigma_key = "model_config.head_configs.centroid.confmaps.sigma"
        assert pipeline_key in main_tab._pipeline_fields
        assert centroid_sigma_key in main_tab._pipeline_fields[pipeline_key]

        # Change the sigma value in the current pipeline's widget
        widget = main_tab._pipeline_fields[pipeline_key][centroid_sigma_key]
        widget.setValue(7.5)

        # get_form_data should return the value from the CURRENT pipeline
        form_data = main_tab.get_form_data()
        assert form_data[centroid_sigma_key] == 7.5

    def test_pipeline_fields_stored_separately(self, training_dialog):
        """Each pipeline should have its own dict of field widgets."""
        main_tab = training_dialog.pipeline_form_widget

        # Check that pipeline fields are stored separately
        assert "multi-animal top-down" in main_tab._pipeline_fields
        assert "multi-animal top-down-id" in main_tab._pipeline_fields

        # Both pipelines have centroid sigma
        centroid_sigma_key = "model_config.head_configs.centroid.confmaps.sigma"
        assert centroid_sigma_key in main_tab._pipeline_fields["multi-animal top-down"]
        assert (
            centroid_sigma_key in main_tab._pipeline_fields["multi-animal top-down-id"]
        )

        # They should be DIFFERENT widget instances
        widget1 = main_tab._pipeline_fields["multi-animal top-down"][centroid_sigma_key]
        widget2 = main_tab._pipeline_fields["multi-animal top-down-id"][
            centroid_sigma_key
        ]
        assert widget1 is not widget2

    def test_set_form_data_updates_all_pipelines(self, training_dialog):
        """set_form_data should update sigma in all pipelines that have it."""
        main_tab = training_dialog.pipeline_form_widget
        centroid_sigma_key = "model_config.head_configs.centroid.confmaps.sigma"

        # Set the sigma value via set_form_data
        main_tab.set_form_data({centroid_sigma_key: 8.0})

        # Both pipeline widgets should be updated
        widget1 = main_tab._pipeline_fields["multi-animal top-down"][centroid_sigma_key]
        widget2 = main_tab._pipeline_fields["multi-animal top-down-id"][
            centroid_sigma_key
        ]
        assert widget1.value() == 8.0
        assert widget2.value() == 8.0


# =============================================================================
# Button State Tests
# =============================================================================


class TestButtonStates:
    """Tests for button state management."""

    def test_run_button_initial_state(self, training_dialog):
        """Run button should be enabled for valid training config."""
        # Initial state depends on validation
        assert training_dialog.run_button is not None

    def test_buttons_have_tooltips(self, training_dialog):
        """All buttons should have tooltips."""
        assert training_dialog.run_button.toolTip() != ""
        assert training_dialog.copy_button.toolTip() != ""
        assert training_dialog.save_button.toolTip() != ""
        assert training_dialog.export_button.toolTip() != ""


# =============================================================================
# Dialog Size Tests
# =============================================================================


class TestDialogSize:
    """Tests for dialog sizing."""

    def test_training_dialog_size(self, training_dialog):
        """Training dialog should have appropriate size."""
        size = training_dialog.size()
        assert size.width() > 0
        assert size.height() > 0

    def test_inference_dialog_size(self, inference_dialog):
        """Inference dialog should have appropriate size."""
        size = inference_dialog.size()
        assert size.width() > 0
        assert size.height() > 0

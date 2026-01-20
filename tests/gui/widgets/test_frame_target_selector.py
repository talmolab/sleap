"""Tests for FrameTargetSelector widget.

This module tests the dropdown-based target selector for training/inference dialogs.
"""

import pytest

from sleap.gui.widgets.frame_target_selector import (
    FrameTargetSelector,
    FrameTargetOption,
    FrameTargetSelection,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def training_selector(qtbot):
    """Create a FrameTargetSelector in training mode."""
    selector = FrameTargetSelector(mode="training")
    qtbot.addWidget(selector)
    return selector


@pytest.fixture
def inference_selector(qtbot):
    """Create a FrameTargetSelector in inference mode."""
    selector = FrameTargetSelector(mode="inference")
    qtbot.addWidget(selector)
    return selector


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestFrameTargetSelectorInstantiation:
    """Tests for FrameTargetSelector basic instantiation."""

    def test_training_mode_instantiation(self, training_selector):
        """Test that training mode selector instantiates correctly."""
        assert training_selector._mode == "training"
        assert training_selector.target_dropdown is not None
        assert training_selector.skip_user_labeled_cb is not None
        assert training_selector.predictions_button_group is not None

    def test_inference_mode_instantiation(self, inference_selector):
        """Test that inference mode selector instantiates correctly."""
        assert inference_selector._mode == "inference"
        assert inference_selector.target_dropdown is not None

    def test_training_mode_has_10_options(self, training_selector):
        """Training mode should have 10 target options."""
        assert training_selector.target_dropdown.count() == 10

    def test_inference_mode_has_9_options(self, inference_selector):
        """Inference mode should have 9 target options (no 'nothing')."""
        assert inference_selector.target_dropdown.count() == 9

    def test_training_mode_includes_nothing(self, training_selector):
        """Training mode should include 'nothing' option."""
        options = [
            training_selector.target_dropdown.itemData(i)
            for i in range(training_selector.target_dropdown.count())
        ]
        assert "nothing" in options

    def test_inference_mode_excludes_nothing(self, inference_selector):
        """Inference mode should NOT include 'nothing' option."""
        options = [
            inference_selector.target_dropdown.itemData(i)
            for i in range(inference_selector.target_dropdown.count())
        ]
        assert "nothing" not in options

    def test_default_target_is_frame(self, training_selector):
        """Default target should be 'frame'."""
        # In training mode, first option is "nothing", but default selection is "frame"
        assert training_selector._selected_key == "frame"

    def test_group_box_title(self, inference_selector):
        """Group box should have correct title."""
        assert inference_selector.target_group_box.title() == "Inference Target"


# =============================================================================
# Target Options Tests
# =============================================================================


class TestTargetOptions:
    """Tests for target option availability and content."""

    def test_all_training_options_present(self, training_selector):
        """All expected training options should be present."""
        expected_keys = [
            "nothing",
            "frame",
            "clip",
            "video",
            "all_videos",
            "random",
            "suggestions",
            "user_labeled",
            "predicted",
        ]
        actual_keys = [
            training_selector.target_dropdown.itemData(i)
            for i in range(training_selector.target_dropdown.count())
        ]
        for key in expected_keys:
            assert key in actual_keys, f"Missing option: {key}"

    def test_all_inference_options_present(self, inference_selector):
        """All expected inference options should be present."""
        expected_keys = [
            "frame",
            "clip",
            "video",
            "all_videos",
            "random",
            "suggestions",
            "user_labeled",
            "predicted",
        ]
        actual_keys = [
            inference_selector.target_dropdown.itemData(i)
            for i in range(inference_selector.target_dropdown.count())
        ]
        for key in expected_keys:
            assert key in actual_keys, f"Missing option: {key}"

    def test_options_have_labels(self, training_selector):
        """All options should have display labels."""
        for i in range(training_selector.target_dropdown.count()):
            label = training_selector.target_dropdown.itemText(i)
            assert label, f"Option at index {i} has empty label"

    def test_option_labels_match_keys(self, training_selector):
        """Option labels should be human-readable versions of keys."""
        expected_labels = {
            "nothing": "Nothing",
            "frame": "Current frame",
            "clip": "Selected clip",
            "video": "Entire video",
            "all_videos": "All videos",
            "random_video": "Random sample (current video)",
            "random": "Random sample (all videos)",
            "suggestions": "Suggestions",
            "user_labeled": "User labeled",
            "predicted": "Frames with predictions",
        }
        for i in range(training_selector.target_dropdown.count()):
            key = training_selector.target_dropdown.itemData(i)
            label = training_selector.target_dropdown.itemText(i)
            assert label == expected_labels[key], f"Label mismatch for {key}"


# =============================================================================
# Selection Tests
# =============================================================================


class TestSelection:
    """Tests for get_selection and set_selection."""

    def test_get_selection_returns_dataclass(self, training_selector):
        """get_selection should return a FrameTargetSelection."""
        selection = training_selector.get_selection()
        assert isinstance(selection, FrameTargetSelection)

    def test_get_selection_default_values(self, inference_selector):
        """get_selection should return correct default values."""
        selection = inference_selector.get_selection()
        assert selection.target_key == "frame"
        assert selection.exclude_user_labeled is False
        assert selection.exclude_predicted is False
        assert selection.prediction_mode == "replace"
        assert selection.clear_all_first is False

    def test_set_selection_target_key(self, training_selector):
        """set_selection should update target dropdown."""
        selection = FrameTargetSelection(target_key="video")
        training_selector.set_selection(selection)
        assert training_selector._selected_key == "video"
        assert training_selector.target_dropdown.currentData() == "video"

    def test_set_selection_exclude_user_labeled(self, training_selector):
        """set_selection should update skip user labeled checkbox."""
        selection = FrameTargetSelection(exclude_user_labeled=True)
        training_selector.set_selection(selection)
        assert training_selector.skip_user_labeled_cb.isChecked() is True

    def test_set_selection_clear_all_first(self, training_selector):
        """set_selection should select Clear all radio when clear_all_first=True."""
        selection = FrameTargetSelection(clear_all_first=True)
        training_selector.set_selection(selection)
        assert training_selector.predictions_clear_radio.isChecked() is True

    def test_set_selection_replace_mode(self, training_selector):
        """set_selection should select Replace radio when prediction_mode='replace'."""
        selection = FrameTargetSelection(prediction_mode="replace")
        training_selector.set_selection(selection)
        assert training_selector.predictions_replace_radio.isChecked() is True

    def test_set_selection_keep_mode(self, training_selector):
        """set_selection should select Keep radio when prediction_mode='add'."""
        selection = FrameTargetSelection(prediction_mode="add", clear_all_first=False)
        training_selector.set_selection(selection)
        assert training_selector.predictions_keep_radio.isChecked() is True

    def test_selection_roundtrip(self, training_selector):
        """Setting and getting selection should be consistent."""
        original = FrameTargetSelection(
            target_key="suggestions",
            exclude_user_labeled=True,
            prediction_mode="replace",
            clear_all_first=False,
        )
        training_selector.set_selection(original)
        result = training_selector.get_selection()

        assert result.target_key == original.target_key
        assert result.exclude_user_labeled == original.exclude_user_labeled
        assert result.prediction_mode == original.prediction_mode


# =============================================================================
# Prediction Mode Radio Button Tests
# =============================================================================


class TestPredictionModeRadios:
    """Tests for prediction mode radio button behavior."""

    def test_default_is_replace(self, training_selector):
        """Default prediction mode should be 'Replace'."""
        assert training_selector.predictions_replace_radio.isChecked() is True
        assert training_selector.predictions_keep_radio.isChecked() is False
        assert training_selector.predictions_clear_radio.isChecked() is False

    def test_clear_all_selection(self, training_selector):
        """Selecting Clear all should set clear_all_first=True."""
        training_selector.predictions_clear_radio.setChecked(True)
        selection = training_selector.get_selection()
        assert selection.clear_all_first is True
        assert selection.prediction_mode == "add"

    def test_replace_selection(self, training_selector):
        """Selecting Replace should set prediction_mode='replace'."""
        training_selector.predictions_replace_radio.setChecked(True)
        selection = training_selector.get_selection()
        assert selection.prediction_mode == "replace"
        assert selection.clear_all_first is False

    def test_keep_selection(self, training_selector):
        """Selecting Keep should set prediction_mode='add', clear_all_first=False."""
        # First set to something else
        training_selector.predictions_replace_radio.setChecked(True)
        # Then back to keep
        training_selector.predictions_keep_radio.setChecked(True)
        selection = training_selector.get_selection()
        assert selection.prediction_mode == "add"
        assert selection.clear_all_first is False

    def test_radios_are_mutually_exclusive(self, training_selector):
        """Only one prediction mode radio can be selected at a time."""
        training_selector.predictions_clear_radio.setChecked(True)
        assert training_selector.predictions_keep_radio.isChecked() is False
        assert training_selector.predictions_replace_radio.isChecked() is False

        training_selector.predictions_replace_radio.setChecked(True)
        assert training_selector.predictions_clear_radio.isChecked() is False
        assert training_selector.predictions_keep_radio.isChecked() is False


# =============================================================================
# Skip Checkbox Tests
# =============================================================================


class TestSkipCheckbox:
    """Tests for skip user labeled checkbox."""

    def test_skip_checkbox_default_unchecked(self, training_selector):
        """Skip user labeled checkbox should be unchecked by default."""
        assert training_selector.skip_user_labeled_cb.isChecked() is False

    def test_skip_checkbox_affects_selection(self, training_selector):
        """Skip checkbox state should be reflected in selection."""
        training_selector.skip_user_labeled_cb.setChecked(True)
        selection = training_selector.get_selection()
        assert selection.exclude_user_labeled is True

        training_selector.skip_user_labeled_cb.setChecked(False)
        selection = training_selector.get_selection()
        assert selection.exclude_user_labeled is False


# =============================================================================
# Form Data Tests
# =============================================================================


class TestFormData:
    """Tests for get_form_data method."""

    def test_get_form_data_returns_dict(self, training_selector):
        """get_form_data should return a dictionary."""
        data = training_selector.get_form_data()
        assert isinstance(data, dict)

    def test_get_form_data_has_all_keys(self, training_selector):
        """get_form_data should include all expected keys."""
        data = training_selector.get_form_data()
        expected_keys = [
            "_predict_target",
            "_exclude_user_labeled",
            "_exclude_predicted",
            "_prediction_mode",
            "_clear_all_first",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_get_form_data_keys_have_underscore_prefix(self, training_selector):
        """All form data keys should have underscore prefix (GUI-only fields)."""
        data = training_selector.get_form_data()
        for key in data.keys():
            assert key.startswith("_"), f"Key {key} missing underscore prefix"

    def test_get_form_data_default_values(self, inference_selector):
        """Form data should have correct default values."""
        data = inference_selector.get_form_data()
        assert data["_predict_target"] == "frame"
        assert data["_exclude_user_labeled"] is False
        assert data["_exclude_predicted"] is False
        assert data["_prediction_mode"] == "replace"
        assert data["_clear_all_first"] is False

    def test_get_form_data_after_changes(self, training_selector):
        """Form data should reflect UI changes."""
        # Change target
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("video")
        )
        # Check skip user labeled
        training_selector.skip_user_labeled_cb.setChecked(True)
        # Select Replace mode
        training_selector.predictions_replace_radio.setChecked(True)

        data = training_selector.get_form_data()
        assert data["_predict_target"] == "video"
        assert data["_exclude_user_labeled"] is True
        assert data["_prediction_mode"] == "replace"


# =============================================================================
# Frame Count Tests
# =============================================================================


class TestFrameCount:
    """Tests for frame count display and updates."""

    def test_update_option_frame_count(self, training_selector):
        """update_option_frame_count should update the stored count."""
        training_selector.update_option_frame_count("video", 1000)
        assert training_selector._options["video"].frame_count == 1000

    def test_frame_count_in_description_singular(self, training_selector):
        """Description should show singular 'frame' for count of 1."""
        # Set to frame which has count 1
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("frame")
        )
        description = training_selector.description_label.text()
        assert "1 frame)" in description

    def test_frame_count_in_description_plural(self, training_selector):
        """Description should show plural 'frames' for count > 1."""
        training_selector.update_option_frame_count("video", 500)
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("video")
        )
        description = training_selector.description_label.text()
        assert "500 frames)" in description

    def test_frame_count_formatted_with_commas(self, training_selector):
        """Large frame counts should be formatted with commas."""
        training_selector.update_option_frame_count("all_videos", 10000)
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("all_videos")
        )
        description = training_selector.description_label.text()
        assert "10,000 frames)" in description

    def test_nothing_option_no_frame_count(self, training_selector):
        """'Nothing' option should not show frame count."""
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("nothing")
        )
        description = training_selector.description_label.text()
        assert "frames)" not in description


# =============================================================================
# Signal Tests
# =============================================================================


class TestSignals:
    """Tests for signal emissions."""

    def test_value_changed_on_target_change(self, training_selector, qtbot):
        """valueChanged should emit when target changes."""
        with qtbot.waitSignal(training_selector.valueChanged, timeout=1000):
            training_selector.target_dropdown.setCurrentIndex(2)

    def test_value_changed_on_skip_checkbox_toggle(self, training_selector, qtbot):
        """valueChanged should emit when skip checkbox is toggled."""
        with qtbot.waitSignal(training_selector.valueChanged, timeout=1000):
            training_selector.skip_user_labeled_cb.setChecked(True)

    def test_value_changed_on_prediction_mode_change(self, training_selector, qtbot):
        """valueChanged should emit when prediction mode radio is clicked."""
        # Note: setChecked(True) doesn't trigger buttonClicked signal,
        # but actual user clicks do. Use click() to simulate user interaction.
        with qtbot.waitSignal(training_selector.valueChanged, timeout=1000):
            training_selector.predictions_replace_radio.click()


# =============================================================================
# Mode Switching Tests
# =============================================================================


class TestModeSwitching:
    """Tests for mode switching behavior."""

    def test_get_mode(self, training_selector, inference_selector):
        """get_mode should return current mode."""
        assert training_selector.get_mode() == "training"
        assert inference_selector.get_mode() == "inference"

    def test_set_mode_training_to_inference(self, training_selector):
        """Switching from training to inference should remove 'nothing' option."""
        assert training_selector.target_dropdown.count() == 10
        training_selector.set_mode("inference")
        assert training_selector._mode == "inference"
        assert training_selector.target_dropdown.count() == 9
        options = [
            training_selector.target_dropdown.itemData(i)
            for i in range(training_selector.target_dropdown.count())
        ]
        assert "nothing" not in options

    def test_set_mode_inference_to_training(self, inference_selector):
        """Switching from inference to training updates mode.

        Note: When switching from inference to training, the 'nothing' option
        is NOT restored because it was already filtered from _options during
        initial construction in inference mode. The rebuild uses current
        _options, not DEFAULT_OPTIONS. This is current behavior - if full
        restoration is needed, options should be reset from DEFAULT_OPTIONS.
        """
        assert inference_selector.target_dropdown.count() == 9
        inference_selector.set_mode("training")
        assert inference_selector._mode == "training"
        # Options count stays 9 because 'nothing' was never in _options
        # (filtered during inference mode construction)
        assert inference_selector.target_dropdown.count() == 9

    def test_set_mode_same_mode_no_change(self, training_selector):
        """Setting same mode should be a no-op."""
        original_count = training_selector.target_dropdown.count()
        training_selector.set_mode("training")
        assert training_selector.target_dropdown.count() == original_count


# =============================================================================
# Custom Options Tests
# =============================================================================


class TestCustomOptions:
    """Tests for set_options method."""

    def test_set_options_replaces_dropdown(self, training_selector):
        """set_options should replace dropdown contents."""
        custom_options = {
            "opt1": FrameTargetOption(
                key="opt1", label="Option 1", description="First option"
            ),
            "opt2": FrameTargetOption(
                key="opt2", label="Option 2", description="Second option"
            ),
        }
        training_selector.set_options(custom_options)
        assert training_selector.target_dropdown.count() == 2

    def test_set_options_with_frame_counts(self, training_selector):
        """set_options should preserve frame counts."""
        custom_options = {
            "opt1": FrameTargetOption(
                key="opt1",
                label="Option 1",
                description="First option",
                frame_count=100,
            ),
        }
        training_selector.set_options(custom_options)
        training_selector.target_dropdown.setCurrentIndex(0)
        description = training_selector.description_label.text()
        assert "100 frames)" in description

    def test_set_options_respects_training_only(self, inference_selector):
        """set_options should filter training_only options in inference mode."""
        custom_options = {
            "opt1": FrameTargetOption(
                key="opt1",
                label="Option 1",
                description="First option",
                training_only=False,
            ),
            "opt2": FrameTargetOption(
                key="opt2",
                label="Training Only",
                description="Training only option",
                training_only=True,
            ),
        }
        inference_selector.set_options(custom_options)
        # Should only have 1 option (training_only filtered out)
        assert inference_selector.target_dropdown.count() == 1
        assert inference_selector.target_dropdown.itemData(0) == "opt1"


# =============================================================================
# UI Configuration Tests
# =============================================================================


class TestUIConfiguration:
    """Tests for UI configuration methods."""

    def test_set_title(self, training_selector):
        """set_title should update group box title."""
        training_selector.set_title("Custom Title")
        assert training_selector.target_group_box.title() == "Custom Title"

    def test_set_compact_mode_no_error(self, training_selector):
        """set_compact_mode should not raise error (no-op for dropdown design)."""
        # Should not raise
        training_selector.set_compact_mode(True)
        training_selector.set_compact_mode(False)

    def test_apply_compact_styling_no_error(self, training_selector):
        """apply_compact_styling should not raise error (no-op for dropdown design)."""
        # Should not raise
        training_selector.apply_compact_styling()

    def test_setup_for_side_panel_no_error(self, training_selector):
        """setup_for_side_panel should not raise error (no-op for dropdown design)."""
        # Should not raise
        training_selector.setup_for_side_panel(min_height=400)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_set_selection_invalid_target_key(self, training_selector):
        """set_selection with invalid target_key should be handled gracefully."""
        original_key = training_selector._selected_key
        selection = FrameTargetSelection(target_key="invalid_key")
        training_selector.set_selection(selection)
        # Should keep original selection since key not found
        assert training_selector._selected_key == original_key

    def test_update_frame_count_invalid_key(self, training_selector):
        """update_option_frame_count with invalid key should be handled gracefully."""
        # Should not raise
        training_selector.update_option_frame_count("invalid_key", 100)

    def test_empty_options_dict(self, training_selector):
        """Setting empty options should be handled gracefully."""
        training_selector.set_options({})
        assert training_selector.target_dropdown.count() == 0

    def test_description_updates_on_selection_change(self, training_selector):
        """Description label should update when selection changes."""
        # Get initial description
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("frame")
        )
        desc1 = training_selector.description_label.text()

        # Change selection
        training_selector.target_dropdown.setCurrentIndex(
            training_selector._option_keys.index("video")
        )
        desc2 = training_selector.description_label.text()

        assert desc1 != desc2

    def test_suggestions_auto_configures_replace_and_skip(self, inference_selector):
        """Selecting 'Suggestions' should auto-select Replace and enable Skip."""
        # Verify initial state - Replace is selected (new default) and skip is unchecked
        assert inference_selector.predictions_replace_radio.isChecked()
        assert not inference_selector.skip_user_labeled_cb.isChecked()

        # Select Suggestions
        suggestions_index = inference_selector._option_keys.index("suggestions")
        inference_selector.target_dropdown.setCurrentIndex(suggestions_index)

        # Verify auto-configuration (Replace stays selected, Skip gets enabled)
        assert inference_selector.predictions_replace_radio.isChecked()
        assert inference_selector.skip_user_labeled_cb.isChecked()
        assert not inference_selector.predictions_keep_radio.isChecked()
        assert not inference_selector.predictions_clear_radio.isChecked()

    def test_user_labeled_auto_unchecks_skip(self, inference_selector):
        """Selecting 'User labeled' should uncheck Skip user labeled frames."""
        # First enable skip user labeled
        inference_selector.skip_user_labeled_cb.setChecked(True)
        assert inference_selector.skip_user_labeled_cb.isChecked()

        # Select User labeled
        user_labeled_index = inference_selector._option_keys.index("user_labeled")
        inference_selector.target_dropdown.setCurrentIndex(user_labeled_index)

        # Verify skip is unchecked
        assert not inference_selector.skip_user_labeled_cb.isChecked()


# =============================================================================
# FrameTargetOption Dataclass Tests
# =============================================================================


class TestFrameTargetOption:
    """Tests for FrameTargetOption dataclass."""

    def test_option_default_values(self):
        """FrameTargetOption should have correct default values."""
        option = FrameTargetOption(key="test", label="Test", description="Test desc")
        assert option.frame_count == 0
        assert option.available is True
        assert option.training_only is False

    def test_option_custom_values(self):
        """FrameTargetOption should accept custom values."""
        option = FrameTargetOption(
            key="test",
            label="Test",
            description="Test desc",
            frame_count=100,
            available=False,
            training_only=True,
        )
        assert option.frame_count == 100
        assert option.available is False
        assert option.training_only is True


# =============================================================================
# FrameTargetSelection Dataclass Tests
# =============================================================================


class TestFrameTargetSelection:
    """Tests for FrameTargetSelection dataclass."""

    def test_selection_default_values(self):
        """FrameTargetSelection should have correct default values."""
        selection = FrameTargetSelection()
        assert selection.target_key == "frame"
        assert selection.exclude_user_labeled is False
        assert selection.exclude_predicted is False
        assert selection.prediction_mode == "replace"
        assert selection.clear_all_first is False

    def test_selection_custom_values(self):
        """FrameTargetSelection should accept custom values."""
        selection = FrameTargetSelection(
            target_key="video",
            exclude_user_labeled=True,
            exclude_predicted=True,
            prediction_mode="replace",
            clear_all_first=True,
        )
        assert selection.target_key == "video"
        assert selection.exclude_user_labeled is True
        assert selection.exclude_predicted is True
        assert selection.prediction_mode == "replace"
        assert selection.clear_all_first is True

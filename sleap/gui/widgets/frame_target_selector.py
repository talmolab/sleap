"""Widget for selecting prediction targets with dropdown design.

This widget provides a clean UI for selecting which frames to run inference on,
with a dropdown for target selection and inline description.

Example::

    >>> selector = FrameTargetSelector(mode="training")
    >>> selector.set_options(options_dict)
    >>> selector.valueChanged.connect(on_selection_changed)
    >>> selection = selector.get_selection()

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QSpinBox,
)


@dataclass
class FrameTargetOption:
    """Represents a single target option in the selector.

    Attributes:
        key: Internal key for the option (e.g., "clip", "suggestions").
        label: Display name (e.g., "Selected clip").
        description: Help text explaining the option.
        frame_count: Number of frames in this selection.
        available: Whether this option is currently available.
        training_only: If True, only show in training mode (e.g., "nothing").
    """

    key: str
    label: str
    description: str
    frame_count: int = 0
    available: bool = True
    training_only: bool = False


@dataclass
class FrameTargetSelection:
    """Represents the user's complete selection.

    Field names use underscore prefix per SLEAP architecture conventions
    (GUI-only fields are filtered before sleap-nn execution).

    Attributes:
        target_key: Which target option is selected.
        exclude_user_labeled: Whether to skip user-labeled frames.
        exclude_predicted: Whether to skip already-predicted frames.
        prediction_mode: "add" (keep existing) or "replace" (overwrite).
        clear_all_first: Pre-action to clear all predictions before running.
        sample_count: Number of frames for random sample options.
    """

    target_key: str = "frame"
    exclude_user_labeled: bool = False
    exclude_predicted: bool = False
    prediction_mode: str = "add"
    clear_all_first: bool = False
    sample_count: int = 20


class FrameTargetSelector(QWidget):
    """Widget for selecting prediction targets with dropdown design.

    This widget follows SLEAP patterns:
    - Emits valueChanged signal for dialog integration
    - Accepts mode parameter for training/inference differences
    - Returns data compatible with underscore-prefix field naming

    Attributes:
        mode: "training" or "inference" - affects available options.
            Training mode includes the "nothing" option.
            Inference mode does not include "nothing".
    """

    valueChanged = Signal()

    # Default target options with descriptions
    DEFAULT_OPTIONS = [
        FrameTargetOption(
            key="nothing",
            label="Nothing",
            description="Skip predictions, training only",
            frame_count=0,
            training_only=True,
        ),
        FrameTargetOption(
            key="frame",
            label="Current frame",
            description="Predict on just this frame",
            frame_count=1,
        ),
        FrameTargetOption(
            key="clip",
            label="Selected clip",
            description="Predict on the frame range you selected",
            frame_count=0,
        ),
        FrameTargetOption(
            key="video",
            label="Entire video",
            description="Predict on all frames in current video",
            frame_count=0,
        ),
        FrameTargetOption(
            key="all_videos",
            label="All videos",
            description="Predict on every frame across all videos",
            frame_count=0,
        ),
        FrameTargetOption(
            key="random_video",
            label="Random sample (current video)",
            description="Random frames from current video",
            frame_count=20,
        ),
        FrameTargetOption(
            key="random",
            label="Random sample (all videos)",
            description="Random frames from all videos",
            frame_count=20,
        ),
        FrameTargetOption(
            key="suggestions",
            label="Suggestions",
            description="Selected frames for labeling",
            frame_count=0,
        ),
        FrameTargetOption(
            key="user_labeled",
            label="User labeled",
            description="Frames you've annotated (for evaluation)",
            frame_count=0,
        ),
        FrameTargetOption(
            key="predicted",
            label="Frames with predictions",
            description="Only frames that already have predictions",
            frame_count=0,
        ),
    ]

    def __init__(self, mode: str = "inference", parent: Optional[QWidget] = None):
        """Initialize the frame target selector.

        Args:
            mode: "training" or "inference" - affects available options.
                Training mode includes the "nothing" option.
                Inference mode does not include "nothing".
            parent: Parent widget.
        """
        super().__init__(parent)
        self._mode = mode
        self._options: Dict[str, FrameTargetOption] = {}
        self._option_keys: List[str] = []  # Maintains order of keys in dropdown
        self._selected_key: str = "frame"

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        from qtpy.QtWidgets import QCheckBox

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # === Target Selection Group Box ===
        self.target_group_box = QGroupBox("Inference Target")
        target_layout = QVBoxLayout(self.target_group_box)
        target_layout.setSpacing(8)

        # Row 1: Dropdown + sample count spinbox + description label (same row)
        dropdown_row = QHBoxLayout()
        dropdown_row.setSpacing(12)

        # Dropdown for target selection
        self.target_dropdown = QComboBox()
        self.target_dropdown.setMinimumWidth(150)
        dropdown_row.addWidget(self.target_dropdown)

        # Sample count spinbox (only visible for random sample options)
        self.sample_count_label = QLabel("Frames:")
        self.sample_count_label.setStyleSheet("font-size: 11px;")
        dropdown_row.addWidget(self.sample_count_label)

        self.sample_count_spinbox = QSpinBox()
        self.sample_count_spinbox.setRange(1, 1000)
        self.sample_count_spinbox.setValue(20)
        self.sample_count_spinbox.setMinimumWidth(60)
        self.sample_count_spinbox.setStyleSheet("font-size: 11px;")
        dropdown_row.addWidget(self.sample_count_spinbox)

        # Description label (shows description + frame count) - to the right
        self.description_label = QLabel()
        self.description_label.setStyleSheet("color: #666; font-size: 11px;")
        dropdown_row.addWidget(self.description_label, stretch=1)

        target_layout.addLayout(dropdown_row)

        # Row 2: Skip user labeled checkbox + Existing predictions radios
        options_layout = QHBoxLayout()
        options_layout.setSpacing(12)

        # Skip user labeled checkbox
        self.skip_user_labeled_cb = QCheckBox("Skip user labeled frames")
        self.skip_user_labeled_cb.setStyleSheet("font-size: 11px;")
        options_layout.addWidget(self.skip_user_labeled_cb)

        # Separator
        options_layout.addSpacing(16)

        # Existing predictions label and radios
        self.predictions_label = QLabel("Existing predictions:")
        self.predictions_label.setStyleSheet("font-size: 11px;")
        options_layout.addWidget(self.predictions_label)

        self.predictions_button_group = QButtonGroup(self)

        self.predictions_clear_radio = QRadioButton("Clear all")
        self.predictions_clear_radio.setStyleSheet("font-size: 11px;")
        self.predictions_button_group.addButton(self.predictions_clear_radio, 0)
        options_layout.addWidget(self.predictions_clear_radio)

        self.predictions_replace_radio = QRadioButton("Replace")
        self.predictions_replace_radio.setStyleSheet("font-size: 11px;")
        self.predictions_button_group.addButton(self.predictions_replace_radio, 1)
        options_layout.addWidget(self.predictions_replace_radio)

        self.predictions_keep_radio = QRadioButton("Keep")
        self.predictions_keep_radio.setStyleSheet("font-size: 11px;")
        self.predictions_keep_radio.setChecked(True)  # Default to Keep
        self.predictions_button_group.addButton(self.predictions_keep_radio, 2)
        options_layout.addWidget(self.predictions_keep_radio)

        options_layout.addStretch()
        target_layout.addLayout(options_layout)

        layout.addWidget(self.target_group_box)
        layout.addStretch()

        # Initialize with default options
        self._build_options_from_list(self.DEFAULT_OPTIONS)
        self._update_description()

    def _connect_signals(self):
        """Connect internal signals."""
        self.target_dropdown.currentIndexChanged.connect(self._on_target_changed)
        self.predictions_button_group.buttonClicked.connect(
            self._on_predictions_changed
        )
        self.skip_user_labeled_cb.stateChanged.connect(
            self._on_skip_user_labeled_changed
        )
        self.sample_count_spinbox.valueChanged.connect(self._on_sample_count_changed)

    def _on_skip_user_labeled_changed(self, state):
        """Handle skip user labeled checkbox change."""
        self._update_description()
        self.valueChanged.emit()

    def _on_sample_count_changed(self, value):
        """Handle sample count spinbox change."""
        self._update_description()
        self.valueChanged.emit()

    def _build_options_from_list(self, options: List[FrameTargetOption]):
        """Build the dropdown from options."""
        # Block signals during rebuild
        self.target_dropdown.blockSignals(True)
        self.target_dropdown.clear()
        self._options.clear()
        self._option_keys.clear()

        first_available_key = None

        for opt in options:
            # Skip training-only options in inference mode
            if opt.training_only and self._mode != "training":
                continue

            self._options[opt.key] = opt
            self._option_keys.append(opt.key)

            # Add to dropdown with just the label
            self.target_dropdown.addItem(opt.label, opt.key)

            if first_available_key is None and opt.available:
                first_available_key = opt.key

        # Select appropriate option
        if self._selected_key not in self._options:
            self._selected_key = first_available_key or "frame"

        # Set current selection in dropdown
        if self._selected_key in self._option_keys:
            index = self._option_keys.index(self._selected_key)
            self.target_dropdown.setCurrentIndex(index)

        self.target_dropdown.blockSignals(False)

        # Apply auto-configuration for initial selection
        self._apply_target_auto_configuration()

    def _on_target_changed(self, index: int):
        """Handle target selection change from dropdown."""
        if 0 <= index < len(self._option_keys):
            self._selected_key = self._option_keys[index]
            self._update_description()
            self._apply_target_auto_configuration()
            self.valueChanged.emit()

    def _apply_target_auto_configuration(self):
        """Apply sensible defaults and enable/disable widgets based on target.

        Different targets have different valid filter combinations:
        - "suggestions": Already excludes user-labeled frames (sleap-nn behavior),
          so "Skip user labeled" is redundant and should be forced on + disabled.
        - "user_labeled": Predicting on user-labeled frames, so "Skip user labeled"
          would result in zero frames. Force off + disabled.
        - "nothing": No inference runs, so all filters are irrelevant. Disable all
          and force "Keep" for predictions.
        - "random", "random_video": Show sample count spinbox.
        - All other targets: All filter combinations are valid.
        """
        # Default: enable everything, hide sample count spinbox
        self.skip_user_labeled_cb.setEnabled(True)
        self.predictions_label.setEnabled(True)
        self.predictions_clear_radio.setEnabled(True)
        self.predictions_replace_radio.setEnabled(True)
        self.predictions_keep_radio.setEnabled(True)

        # Show sample count spinbox only for random sample options
        is_random_sample = self._selected_key in ("random", "random_video")
        self.sample_count_label.setVisible(is_random_sample)
        self.sample_count_spinbox.setVisible(is_random_sample)

        if self._selected_key == "suggestions":
            # Suggestions already excludes user-labeled frames in sleap-nn
            # Force checked and disable to indicate this is automatic
            self.skip_user_labeled_cb.setChecked(True)
            self.skip_user_labeled_cb.setEnabled(False)
            # Default to Replace for suggestions (common workflow)
            self.predictions_replace_radio.setChecked(True)

        elif self._selected_key == "user_labeled":
            # Can't skip user-labeled when targeting user-labeled frames
            self.skip_user_labeled_cb.setChecked(False)
            self.skip_user_labeled_cb.setEnabled(False)

        elif self._selected_key == "nothing":
            # No inference runs, so filters are irrelevant
            # Disable everything and force "Keep" (no changes to predictions)
            self.skip_user_labeled_cb.setChecked(False)
            self.skip_user_labeled_cb.setEnabled(False)
            self.predictions_label.setEnabled(False)
            self.predictions_clear_radio.setEnabled(False)
            self.predictions_replace_radio.setEnabled(False)
            self.predictions_keep_radio.setEnabled(False)
            # Force "Keep" so no predictions are accidentally cleared
            self.predictions_keep_radio.setChecked(True)

    def _on_predictions_changed(self, button):
        """Handle predictions mode change."""
        self.valueChanged.emit()

    def _update_description(self):
        """Update the description label based on current selection."""
        if self._selected_key not in self._options:
            self.description_label.setText("")
            return

        opt = self._options[self._selected_key]
        description = opt.description

        # Build frame count text with proper singular/plural
        if opt.frame_count > 0:
            frame_word = "frame" if opt.frame_count == 1 else "frames"
            count_str = f"{opt.frame_count:,}"
            frame_text = f" ({count_str} {frame_word})"
        elif opt.key != "nothing":
            frame_text = " (0 frames)"
        else:
            frame_text = ""

        self.description_label.setText(f"{description}{frame_text}")

    def set_options(self, options: Dict[str, FrameTargetOption]):
        """Set the available target options.

        Args:
            options: Dictionary mapping option keys to FrameTargetOption objects.
        """
        # Convert dict to list maintaining order
        option_list = list(options.values())
        self._build_options_from_list(option_list)
        self._update_description()

    def update_option_frame_count(self, key: str, frame_count: int):
        """Update the frame count for a specific option.

        Args:
            key: The option key to update.
            frame_count: The new frame count.
        """
        if key in self._options:
            self._options[key].frame_count = frame_count
            # Update description if this is the currently selected option
            if key == self._selected_key:
                self._update_description()

    def get_selection(self) -> FrameTargetSelection:
        """Get the current selection.

        Returns:
            FrameTargetSelection with all current settings.
        """
        # Determine prediction mode and clear_all_first from radio buttons
        if self.predictions_clear_radio.isChecked():
            prediction_mode = "add"
            clear_all_first = True
        elif self.predictions_replace_radio.isChecked():
            prediction_mode = "replace"
            clear_all_first = False
        else:  # Keep
            prediction_mode = "add"
            clear_all_first = False

        return FrameTargetSelection(
            target_key=self._selected_key,
            exclude_user_labeled=self.skip_user_labeled_cb.isChecked(),
            exclude_predicted=False,
            prediction_mode=prediction_mode,
            clear_all_first=clear_all_first,
            sample_count=self.sample_count_spinbox.value(),
        )

    def set_selection(self, selection: FrameTargetSelection):
        """Set the current selection.

        Args:
            selection: FrameTargetSelection with settings to apply.
        """
        # Set target
        if selection.target_key in self._option_keys:
            self._selected_key = selection.target_key
            index = self._option_keys.index(selection.target_key)
            self.target_dropdown.setCurrentIndex(index)

        # Set skip user labeled checkbox
        self.skip_user_labeled_cb.setChecked(selection.exclude_user_labeled)

        # Set predictions mode radio
        if selection.clear_all_first:
            self.predictions_clear_radio.setChecked(True)
        elif selection.prediction_mode == "replace":
            self.predictions_replace_radio.setChecked(True)
        else:
            self.predictions_keep_radio.setChecked(True)

        # Set sample count spinbox
        self.sample_count_spinbox.setValue(selection.sample_count)

        self._update_description()
        self._apply_target_auto_configuration()

    def get_form_data(self) -> Dict[str, Any]:
        """Return data with underscore-prefixed keys for GUI-only fields.

        This follows SLEAP's convention where underscore-prefixed fields
        are filtered out before sleap-nn execution.

        Returns:
            Dictionary with form data.
        """
        selection = self.get_selection()
        return {
            "_predict_target": selection.target_key,
            "_exclude_user_labeled": selection.exclude_user_labeled,
            "_exclude_predicted": selection.exclude_predicted,
            "_prediction_mode": selection.prediction_mode,
            "_clear_all_first": selection.clear_all_first,
            "_sample_count": selection.sample_count,
        }

    def get_mode(self) -> str:
        """Get the current mode (training/inference)."""
        return self._mode

    def set_mode(self, mode: str):
        """Set the mode and rebuild options.

        Args:
            mode: "training" or "inference"
        """
        if mode != self._mode:
            self._mode = mode
            # Rebuild with current options to show/hide training-only items
            current_options = list(self._options.values())
            if current_options:
                self._build_options_from_list(current_options)
            else:
                self._build_options_from_list(self.DEFAULT_OPTIONS)
            self._update_description()

    def set_title(self, title: str):
        """Set the group box title.

        Args:
            title: The new title for the group box.
        """
        self.target_group_box.setTitle(title)

    def set_compact_mode(self, compact: bool):
        """Set compact mode (no-op for dropdown design, kept for API compatibility).

        Args:
            compact: Ignored in dropdown design.
        """
        pass  # No-op for dropdown design

    def apply_compact_styling(self):
        """Apply compact styling.

        No-op for dropdown design, kept for API compatibility.
        """
        pass  # No-op for dropdown design

    def setup_for_side_panel(self, min_height: Optional[int] = None):
        """Configure sizing for use in a side panel layout (API compatibility).

        Args:
            min_height: Ignored in dropdown design.
        """
        pass  # No-op for dropdown design


if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Test in training mode
    widget = FrameTargetSelector(mode="training")
    widget.setWindowTitle("Frame Target Selector (Training Mode)")
    widget.resize(400, 200)
    widget.show()

    sys.exit(app.exec_())

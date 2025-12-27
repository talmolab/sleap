"""Widget for selecting prediction targets with C1 radio list design.

This widget provides a progressive disclosure UI for selecting which frames
to run inference on, with advanced options for exclusions and prediction mode.

Example::

    >>> selector = FrameTargetSelector(mode="training")
    >>> selector.set_options(options_dict)
    >>> selector.valueChanged.connect(on_selection_changed)
    >>> selection = selector.get_selection()

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QFrame,
    QToolButton,
    QScrollArea,
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
    """

    target_key: str = "frame"
    exclude_user_labeled: bool = False
    exclude_predicted: bool = False
    prediction_mode: str = "add"
    clear_all_first: bool = False


class TargetOptionItem(QFrame):
    """A single target option with radio button, label, and description.

    Clicking anywhere on the item selects it.
    """

    clicked = Signal()

    def __init__(
        self,
        key: str,
        label: str,
        description: str,
        frame_count: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.key = key
        self.setFrameStyle(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 8)
        layout.setSpacing(2)

        # Top row: radio + label + count
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.radio = QRadioButton()
        top_row.addWidget(self.radio)

        # Build label text with frame count
        label_text = f"<b>{label}</b>"
        if frame_count > 0:
            count_str = f"{frame_count:,}"
            label_text += f" <span style='color: #666;'>({count_str} frames)</span>"
        elif frame_count == 0 and key != "nothing":
            label_text += " <span style='color: #999;'>(0 frames)</span>"

        self.label_widget = QLabel(label_text)
        self.label_widget.setStyleSheet("font-size: 12px;")
        top_row.addWidget(self.label_widget)
        top_row.addStretch()

        layout.addLayout(top_row)

        # Description
        self.desc_widget = QLabel(description)
        self.desc_widget.setWordWrap(True)
        self.desc_widget.setStyleSheet(
            "color: #666; font-size: 10px; margin-left: 24px;"
        )
        layout.addWidget(self.desc_widget)

        # Connect radio to signal
        self.radio.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool):
        if checked:
            self.clicked.emit()

    def mousePressEvent(self, event):
        self.radio.setChecked(True)
        super().mousePressEvent(event)

    def isChecked(self) -> bool:
        return self.radio.isChecked()

    def setChecked(self, checked: bool):
        self.radio.setChecked(checked)

    def update_frame_count(self, frame_count: int):
        """Update the displayed frame count."""
        label = self.label_widget.text()
        # Extract the label text before the count span
        if "<span" in label:
            label = label.split("<span")[0].strip()
        else:
            label = label.replace("</b>", "").replace("<b>", "")
            label = f"<b>{label}</b>"

        if frame_count > 0:
            count_str = f"{frame_count:,}"
            label += f" <span style='color: #666;'>({count_str} frames)</span>"
        elif frame_count == 0 and self.key != "nothing":
            label += " <span style='color: #999;'>(0 frames)</span>"

        self.label_widget.setText(label)


class CollapsibleSection(QWidget):
    """A collapsible section with toggle button and summary text."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_expanded = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background-color: #e8e8e8; border-radius: 3px; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._toggle)
        header_layout.addWidget(self.toggle_button)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(self.summary_label)
        header_layout.addStretch()

        main_layout.addWidget(header)

        # Content area
        self.content_area = QFrame()
        self.content_area.setStyleSheet(
            "QFrame { background-color: #f5f5f5; border-radius: 3px; }"
        )
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 8, 12, 8)
        self.content_area.setVisible(False)
        main_layout.addWidget(self.content_area)

    def _toggle(self):
        self._is_expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.DownArrow if self._is_expanded else Qt.RightArrow
        )
        self.content_area.setVisible(self._is_expanded)

    def set_summary(self, text: str):
        """Set the summary text shown next to the toggle button."""
        self.summary_label.setText(text)

    def add_widget(self, widget: QWidget):
        """Add a widget to the collapsible content area."""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded: bool):
        """Programmatically expand or collapse the section."""
        if expanded != self._is_expanded:
            self.toggle_button.setChecked(expanded)
            self._toggle()

    def is_expanded(self) -> bool:
        """Return whether the section is currently expanded."""
        return self._is_expanded


class FrameTargetSelector(QWidget):
    """Widget for selecting prediction targets with C1 radio list design.

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
            key="random",
            label="Random sample",
            description="20 random frames for quick model check",
            frame_count=20,
        ),
        FrameTargetOption(
            key="suggestions",
            label="Suggested frames",
            description="AI-selected frames good for labeling",
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
        self._option_items: Dict[str, TargetOptionItem] = {}
        self._selected_key: str = "frame"

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # === Target Selection (Radio List) ===
        self.target_group_box = QGroupBox("Predict On")
        target_layout = QVBoxLayout(self.target_group_box)
        target_layout.setSpacing(0)

        # Scrollable area for options
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        # Note: Height is managed dynamically based on mode (training/inference)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(0)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.target_button_group = QButtonGroup(self)
        self.target_button_group.setExclusive(True)

        self.scroll_area.setWidget(self.scroll_content)
        target_layout.addWidget(self.scroll_area)
        layout.addWidget(self.target_group_box)

        # === Advanced Options (collapsible) ===
        self.advanced_section = CollapsibleSection("Advanced Options")

        # Exclusions
        self.advanced_section.add_widget(QLabel("<b>Exclusions</b>"))

        self.exclude_user_labeled_cb = QCheckBox("Skip user-labeled frames")
        self.advanced_section.add_widget(self.exclude_user_labeled_cb)

        self.exclude_predicted_cb = QCheckBox("Skip already-predicted frames")
        self.advanced_section.add_widget(self.exclude_predicted_cb)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        self.advanced_section.add_widget(sep1)

        # Mode
        self.advanced_section.add_widget(QLabel("<b>Prediction Handling</b>"))

        self.mode_button_group = QButtonGroup(self)
        self.mode_add_radio = QRadioButton("Add alongside existing")
        self.mode_add_radio.setChecked(True)
        self.mode_button_group.addButton(self.mode_add_radio)
        self.advanced_section.add_widget(self.mode_add_radio)

        self.mode_replace_radio = QRadioButton("Replace on target frames")
        self.mode_button_group.addButton(self.mode_replace_radio)
        self.advanced_section.add_widget(self.mode_replace_radio)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        self.advanced_section.add_widget(sep2)

        # Pre-action
        self.advanced_section.add_widget(QLabel("<b>Before Running</b>"))
        self.clear_all_first_cb = QCheckBox("Clear ALL predictions first")
        self.clear_all_first_cb.setStyleSheet("color: #c00;")
        self.advanced_section.add_widget(self.clear_all_first_cb)

        layout.addWidget(self.advanced_section)

        # === What Will Happen (Preview) ===
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_frame.setStyleSheet("background-color: #f0f8ff; padding: 8px;")
        preview_layout = QVBoxLayout(preview_frame)

        preview_header = QLabel("<b>What will happen:</b>")
        preview_layout.addWidget(preview_header)

        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("font-size: 11px;")
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_frame)
        layout.addStretch()

        # Initialize with default options
        self._build_options_from_list(self.DEFAULT_OPTIONS)
        self._update_display()

    def _connect_signals(self):
        """Connect internal signals."""
        self.target_button_group.buttonClicked.connect(self._on_target_changed)
        self.exclude_user_labeled_cb.stateChanged.connect(self._update_display)
        self.exclude_predicted_cb.stateChanged.connect(self._update_display)
        self.mode_add_radio.toggled.connect(self._update_display)
        self.mode_replace_radio.toggled.connect(self._update_display)
        self.clear_all_first_cb.stateChanged.connect(self._update_display)

    def _build_options_from_list(self, options: List[FrameTargetOption]):
        """Build the radio button list from options."""
        # Clear existing items
        for item in self._option_items.values():
            self.target_button_group.removeButton(item.radio)
            item.setParent(None)
            item.deleteLater()
        self._option_items.clear()
        self._options.clear()

        # Clear layout
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Build new items
        button_id = 0
        first_available_key = None

        for opt in options:
            # Skip training-only options in inference mode
            if opt.training_only and self._mode != "training":
                continue

            self._options[opt.key] = opt

            item = TargetOptionItem(
                key=opt.key,
                label=opt.label,
                description=opt.description,
                frame_count=opt.frame_count,
            )
            item.clicked.connect(lambda k=opt.key: self._on_item_clicked(k))

            self.target_button_group.addButton(item.radio, button_id)
            self._option_items[opt.key] = item
            self.scroll_layout.addWidget(item)

            # Add separator
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #e0e0e0;")
            self.scroll_layout.addWidget(sep)

            if first_available_key is None and opt.available:
                first_available_key = opt.key

            button_id += 1

        self.scroll_layout.addStretch()

        # Select first available option if current selection is not available
        if self._selected_key not in self._option_items:
            self._selected_key = first_available_key or "frame"

        if self._selected_key in self._option_items:
            self._option_items[self._selected_key].setChecked(True)

    def _on_target_changed(self, button):
        """Handle target selection change from button group."""
        for key, item in self._option_items.items():
            if item.radio == button:
                self._selected_key = key
                break
        self._update_display()
        self.valueChanged.emit()

    def _on_item_clicked(self, key: str):
        """Handle target option item click."""
        self._selected_key = key
        self._update_display()
        self.valueChanged.emit()

    def _update_display(self):
        """Update the preview panel and advanced summary."""
        selection = self.get_selection()

        # Update advanced summary
        advanced_parts = []
        if selection.exclude_user_labeled:
            advanced_parts.append("-labeled")
        if selection.exclude_predicted:
            advanced_parts.append("-predicted")
        if selection.prediction_mode == "replace":
            advanced_parts.append("replace")
        if selection.clear_all_first:
            advanced_parts.append("clear first")

        if advanced_parts:
            self.advanced_section.set_summary(f"({', '.join(advanced_parts)})")
        else:
            self.advanced_section.set_summary("(defaults)")

        # Build preview
        preview_parts = []
        step = 1

        if selection.clear_all_first:
            msg = f"{step}. Delete ALL existing predictions"
            preview_parts.append(f"<span style='color:#c00;'>{msg}</span>")
            step += 1

        # Effective target
        if selection.target_key in self._options:
            opt = self._options[selection.target_key]
            if opt.frame_count > 0:
                effective = f"<b>{opt.label}</b> ({opt.frame_count:,} frames)"
            else:
                effective = f"<b>{opt.label}</b>"
        else:
            effective = f"<b>{selection.target_key}</b>"

        exclusions = []
        if selection.exclude_user_labeled:
            exclusions.append("user-labeled")
        if selection.exclude_predicted:
            exclusions.append("predicted")
        if exclusions:
            effective += f" minus {', '.join(exclusions)}"

        preview_parts.append(f"{step}. Run predictions on: {effective}")
        step += 1

        if selection.prediction_mode == "replace":
            preview_parts.append(
                f"{step}. Old predictions on these frames → <b>replaced</b>"
            )
        else:
            preview_parts.append(
                f"{step}. New predictions → <b>added</b> (keeping existing)"
            )

        self.preview_label.setText("<br>".join(preview_parts))

    def set_options(self, options: Dict[str, FrameTargetOption]):
        """Set the available target options.

        Args:
            options: Dictionary mapping option keys to FrameTargetOption objects.
        """
        # Convert dict to list maintaining order
        option_list = list(options.values())
        self._build_options_from_list(option_list)
        self._update_display()

    def update_option_frame_count(self, key: str, frame_count: int):
        """Update the frame count for a specific option.

        Args:
            key: The option key to update.
            frame_count: The new frame count.
        """
        if key in self._options:
            self._options[key].frame_count = frame_count
        if key in self._option_items:
            self._option_items[key].update_frame_count(frame_count)
        self._update_display()

    def get_selection(self) -> FrameTargetSelection:
        """Get the current selection.

        Returns:
            FrameTargetSelection with all current settings.
        """
        return FrameTargetSelection(
            target_key=self._selected_key,
            exclude_user_labeled=self.exclude_user_labeled_cb.isChecked(),
            exclude_predicted=self.exclude_predicted_cb.isChecked(),
            prediction_mode="replace" if self.mode_replace_radio.isChecked() else "add",
            clear_all_first=self.clear_all_first_cb.isChecked(),
        )

    def set_selection(self, selection: FrameTargetSelection):
        """Set the current selection.

        Args:
            selection: FrameTargetSelection with settings to apply.
        """
        # Set target
        if selection.target_key in self._option_items:
            self._selected_key = selection.target_key
            self._option_items[selection.target_key].setChecked(True)

        # Set advanced options
        self.exclude_user_labeled_cb.setChecked(selection.exclude_user_labeled)
        self.exclude_predicted_cb.setChecked(selection.exclude_predicted)

        if selection.prediction_mode == "replace":
            self.mode_replace_radio.setChecked(True)
        else:
            self.mode_add_radio.setChecked(True)

        self.clear_all_first_cb.setChecked(selection.clear_all_first)

        self._update_display()

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
            self._update_display()

    def apply_compact_styling(self):
        """Make option items more compact by reducing spacing and font sizes.

        Call this method when the selector is used in a side panel layout
        where vertical space is limited.
        """

        # Reduce spacing in the scroll layout
        self.scroll_layout.setSpacing(0)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Make each option item more compact
        for key, item in self._option_items.items():
            # Reduce item margins
            item.layout().setContentsMargins(2, 2, 2, 4)
            item.layout().setSpacing(1)

            # Make description text smaller
            item.desc_widget.setStyleSheet(
                "color: #666; font-size: 9px; margin-left: 20px; margin-top: 0px;"
            )

    def setup_for_side_panel(self, min_height: Optional[int] = None):
        """Configure sizing for use in a side panel layout.

        Removes height constraints and sets up expanding size policy so the
        selector uses available vertical space.

        Args:
            min_height: Optional minimum height for the scroll area. If None,
                calculates based on number of options (42px per option).
        """
        from qtpy.QtWidgets import QSizePolicy

        # Calculate needed height based on mode if not specified
        if min_height is None:
            # Training has 9 options, inference has 8
            num_options = 9 if self._mode == "training" else 8
            height_per_option = 42  # Compact option height
            min_height = num_options * height_per_option + 20  # padding

        # Remove max height constraints
        self.scroll_area.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        self.target_group_box.setMaximumHeight(16777215)

        # Set minimum height to fit all options
        self.scroll_area.setMinimumHeight(min_height)

        # Force expanding size policy
        self.scroll_area.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )

        # Apply compact styling
        self.apply_compact_styling()


if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Test in training mode
    widget = FrameTargetSelector(mode="training")
    widget.setWindowTitle("Frame Target Selector (Training Mode)")
    widget.resize(450, 600)
    widget.show()

    sys.exit(app.exec_())

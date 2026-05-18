"""
Dockable widget for viewing label QC results.

Provides a QDockWidget that wraps the QCWidget with navigation support.
Starts floating by default but can be docked to left or right side.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from sleap.gui.widgets.qc import QCWidget

if TYPE_CHECKING:
    import sleap_io as sio


class QCDockWidget(QtWidgets.QDockWidget):
    """Dockable widget for label quality control analysis.

    This dock widget wraps the QCWidget and provides docking capabilities.
    It starts floating by default but can be docked to the left or right
    side of the main window for convenient review workflows.

    Args:
        labels: The Labels object containing labeled frames.
        navigate_callback: Optional callback function that takes
            (video_idx, frame_idx, instance_idx) arguments. Called when
            user selects an instance to navigate to.
        parent: Parent widget (typically the main window).
    """

    def __init__(
        self,
        labels: Optional["sio.Labels"] = None,
        navigate_callback: Optional[Callable[[int, int, int], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the dock widget.

        Args:
            labels: The Labels object containing labeled frames, or None if
                not yet loaded.
            navigate_callback: Optional callback for navigation.
            parent: Parent widget.
        """
        super().__init__("Label Quality Control", parent)

        self._labels = labels
        self._navigate_callback = navigate_callback
        self._tab_visible = True  # Track if we're the visible tab when docked
        # Store main window reference (self.parent() may change after docking)
        self._main_window = parent

        self._setup_ui()
        self._setup_dock()
        self._connect_signals()

    def _setup_dock(self):
        """Configure dock widget properties."""
        self.setObjectName("LabelQCDock")

        # Allow docking on left or right side
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Start docked by default (not floating) - Qt state restoration will
        # override this if the user previously changed it
        self.setFloating(False)

        # Set minimum size for when docked
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        # Enable close button and allow floating
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )

    def _setup_ui(self):
        """Set up the dock widget UI."""
        # Create container widget
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        # Main QC widget
        self._widget = QCWidget()
        self._widget.set_labels(self._labels)
        layout.addWidget(self._widget, stretch=1)

        # Button row: Add to Suggestions, Export CSV, Dock/Undock, Close
        button_layout = QtWidgets.QHBoxLayout()

        self._suggestions_button = QtWidgets.QPushButton("Add to Suggestions")
        self._suggestions_button.setToolTip(
            "Add flagged frames to the Labeling Suggestions list for review"
        )
        self._suggestions_button.clicked.connect(self._on_add_to_suggestions)
        button_layout.addWidget(self._suggestions_button)

        self._export_button = QtWidgets.QPushButton("Export CSV...")
        self._export_button.setToolTip("Export all QC results to a CSV file")
        self._export_button.clicked.connect(self._widget.export_results)
        button_layout.addWidget(self._export_button)

        button_layout.addStretch()

        # Fit to Selection checkbox - mirrors View menu option
        self._fit_selection_checkbox = QtWidgets.QCheckBox("Fit to Selection")
        self._fit_selection_checkbox.setToolTip(
            "Auto-zoom to fit the selected instance when navigating"
        )
        self._fit_selection_checkbox.stateChanged.connect(
            self._on_fit_selection_changed
        )
        button_layout.addWidget(self._fit_selection_checkbox)

        # Dock/Undock toggle button - starts with "Undock" since we default to docked
        self._dock_button = QtWidgets.QPushButton("Undock")
        self._dock_button.setToolTip("Undock this panel to a floating window")
        self._dock_button.clicked.connect(self._toggle_dock)
        button_layout.addWidget(self._dock_button)

        layout.addLayout(button_layout)

        self.setWidget(container)

    def _connect_signals(self):
        """Connect widget signals."""
        if self._navigate_callback is not None:
            self._widget.navigate_to_instance.connect(self._on_navigate)

        # Update dock button text when dock state changes (e.g., via dragging)
        self.topLevelChanged.connect(self._update_dock_button)

        # Track tab visibility for navigation precedence
        # visibilityChanged is emitted when tab is selected/deselected in tabified mode
        self.visibilityChanged.connect(self._on_visibility_changed)

        # Connect to parent state for fit_selection sync (do this early)
        # This ensures the checkbox stays synced even if the dock hasn't been shown yet
        parent = self._main_window
        if parent is not None and hasattr(parent, "state"):
            # Initial sync of checkbox state
            self._sync_fit_selection_checkbox()
            # Connect to state changes
            parent.state.connect("fit_selection", self._on_fit_selection_state_changed)
            self._state_connected = True

    def _on_visibility_changed(self, visible: bool):
        """Track visibility changes and update labels if needed.

        When tabified, this signal tells us if our tab is selected.
        Also updates labels when dock becomes visible (handles case where
        dock is opened via View menu instead of Analyze > Label QC).
        """
        self._tab_visible = visible

        # When dock becomes visible, sync state from parent
        if visible:
            parent = self._main_window
            if parent is not None:
                # Sync labels
                if hasattr(parent, "labels"):
                    parent_labels = parent.labels
                    if parent_labels is not None and self._labels is not parent_labels:
                        self.update_labels(parent_labels)

                # Sync fit_selection checkbox (connection made in _connect_signals)
                if hasattr(parent, "state"):
                    self._sync_fit_selection_checkbox()

    def _on_navigate(self, video_idx: int, frame_idx: int, instance_idx: int):
        """Handle navigation request from widget."""
        if self._navigate_callback is not None:
            self._navigate_callback(video_idx, frame_idx, instance_idx)

    def _toggle_dock(self):
        """Toggle between docked and floating states."""
        if self.isFloating():
            # Currently floating, dock it
            self.setFloating(False)
            # Bring the QC tab to front after docking
            self.raise_()
        else:
            # Currently docked, float it
            self.setFloating(True)

    def _update_dock_button(self, floating: bool):
        """Update dock button text based on current state.

        Args:
            floating: True if the widget is now floating, False if docked.
        """
        if floating:
            self._dock_button.setText("Dock to Right")
            self._dock_button.setToolTip(
                "Dock this panel to the right side of the main window"
            )
        else:
            self._dock_button.setText("Undock")
            self._dock_button.setToolTip("Undock this panel to a floating window")

    def _on_add_to_suggestions(self):
        """Handle Add to Suggestions button click."""
        n_added = self._widget.export_to_suggestions()
        if n_added > 0:
            # Trigger update of suggestions dock in main window
            parent = self._main_window
            if parent is not None and hasattr(parent, "on_data_update"):
                from sleap.gui.commands import UpdateTopic

                parent.on_data_update([UpdateTopic.suggestions])

    def _on_fit_selection_changed(self, state: int):
        """Handle Fit to Selection checkbox change.

        Updates the main window's state to mirror the checkbox and applies
        the zoom effect immediately.
        """
        parent = self._main_window
        if parent is not None and hasattr(parent, "state"):
            # stateChanged passes int: 0=Unchecked, 2=Checked
            # Note: Qt.Checked is an enum, so compare to int value directly
            enabled = state == Qt.Checked.value
            parent.state["fit_selection"] = enabled
            # Apply immediate zoom effect (state change will also trigger this,
            # but that's fine - it's idempotent)
            self._apply_fit_selection_zoom(enabled)

    def _on_fit_selection_state_changed(self, value: bool):
        """Handle fit_selection state change from menu or elsewhere.

        Syncs the checkbox and applies immediate zoom effect.
        """
        self._sync_fit_selection_checkbox(value)
        self._apply_fit_selection_zoom(value)

    def _apply_fit_selection_zoom(self, enabled: bool):
        """Apply or clear the fit-to-selection zoom.

        Args:
            enabled: If True, zoom to current selection. If False, clear zoom.
        """
        parent = self._main_window
        if parent is None or not hasattr(parent, "player"):
            return

        player = parent.player
        if enabled:
            # Zoom to fit the current selection
            player.zoomToSelection()
        else:
            # Clear the zoom to show full frame
            player.view.clearZoom()

    def _sync_fit_selection_checkbox(self, value=None):
        """Sync checkbox state from main window's state.

        Args:
            value: Optional value from state callback. If None, reads from state.
        """
        if value is None:
            parent = self._main_window
            if parent is not None and hasattr(parent, "state"):
                value = parent.state.get("fit_selection", False)
            else:
                value = False

        # Block signals to avoid feedback loop
        self._fit_selection_checkbox.blockSignals(True)
        self._fit_selection_checkbox.setChecked(bool(value))
        self._fit_selection_checkbox.blockSignals(False)

    def update_labels(self, labels: "sio.Labels"):
        """Update the labels being analyzed.

        Args:
            labels: New Labels object.
        """
        self._labels = labels
        self._widget.set_labels(labels)

    @property
    def is_active_for_navigation(self) -> bool:
        """Check if QC dock should take precedence for space navigation.

        Returns True if:
        - The dock is visible (not closed)
        - The widget has flagged results to navigate
        - AND either: dock is floating, OR dock is the visible tab when tabified

        Returns:
            True if QC navigation should take precedence over suggestions.
        """
        # Must be visible (not closed) and have flags to navigate
        if not self.isVisible():
            return False
        if not self._widget.has_flags:
            return False

        # If floating, always active
        if self.isFloating():
            return True

        # If docked, use tracked tab visibility from visibilityChanged signal
        return self._tab_visible

    @property
    def has_flags(self) -> bool:
        """Check if the widget has flagged items."""
        return self._widget.has_flags

    def goto_next_flag(self) -> bool:
        """Navigate to next flagged instance.

        Returns:
            True if navigation occurred.
        """
        return self._widget.goto_next_flag()

    def goto_prev_flag(self) -> bool:
        """Navigate to previous flagged instance.

        Returns:
            True if navigation occurred.
        """
        return self._widget.goto_prev_flag()

    def closeEvent(self, event):
        """Handle dock widget close event."""
        # Clean up the widget's resources (e.g., stop running analysis)
        self._widget.cleanup()
        super().closeEvent(event)


# Keep QCDialog as an alias for backwards compatibility
QCDialog = QCDockWidget

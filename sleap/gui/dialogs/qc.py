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

        # Dock/Undock toggle button - starts with "Undock" since we default to docked
        self._dock_button = QtWidgets.QPushButton("Undock")
        self._dock_button.setToolTip("Undock this panel to a floating window")
        self._dock_button.clicked.connect(self._toggle_dock)
        button_layout.addWidget(self._dock_button)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setWidget(container)

    def _connect_signals(self):
        """Connect widget signals."""
        if self._navigate_callback is not None:
            self._widget.navigate_to_instance.connect(self._on_navigate)

        # Update dock button text when dock state changes (e.g., via dragging)
        self.topLevelChanged.connect(self._update_dock_button)

    def _on_navigate(self, video_idx: int, frame_idx: int, instance_idx: int):
        """Handle navigation request from widget."""
        if self._navigate_callback is not None:
            self._navigate_callback(video_idx, frame_idx, instance_idx)

    def _toggle_dock(self):
        """Toggle between docked and floating states."""
        if self.isFloating():
            # Currently floating, dock it
            self.setFloating(False)
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
            self._dock_button.setToolTip(
                "Undock this panel to a floating window"
            )

    def _on_add_to_suggestions(self):
        """Handle Add to Suggestions button click."""
        n_added = self._widget.export_to_suggestions()
        if n_added > 0:
            # Trigger update of suggestions dock in main window
            parent = self.parent()
            if parent is not None and hasattr(parent, "on_data_update"):
                from sleap.gui.commands import UpdateTopic

                parent.on_data_update([UpdateTopic.suggestions])

    def update_labels(self, labels: "sio.Labels"):
        """Update the labels being analyzed.

        Args:
            labels: New Labels object.
        """
        self._labels = labels
        self._widget.set_labels(labels)

    def closeEvent(self, event):
        """Handle dock widget close event."""
        # Clean up the widget's resources (e.g., stop running analysis)
        self._widget.cleanup()
        super().closeEvent(event)


# Keep QCDialog as an alias for backwards compatibility
QCDialog = QCDockWidget

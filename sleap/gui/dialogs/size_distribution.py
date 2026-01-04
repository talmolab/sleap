"""
Dialog for viewing instance size distribution.

Provides a standalone dialog that wraps the SizeDistributionWidget
with navigation support back to the main window.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from qtpy import QtWidgets

from sleap.gui.widgets.size_distribution import SizeDistributionWidget

if TYPE_CHECKING:
    import sleap_io as sio


class SizeDistributionDialog(QtWidgets.QDialog):
    """Dialog for viewing instance size distribution with navigation.

    This dialog displays the SizeDistributionWidget and optionally
    connects navigation signals to the main window.

    Args:
        labels: The Labels object containing labeled frames.
        navigate_callback: Optional callback function that takes
            (video_idx, frame_idx, instance_idx) arguments. Called when
            user clicks "Go to Frame" on a selected instance.
        parent: Parent widget.
    """

    def __init__(
        self,
        labels: "sio.Labels",
        navigate_callback: Optional[Callable[[int, int, int], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the dialog.

        Args:
            labels: The Labels object containing labeled frames.
            navigate_callback: Optional callback for navigation.
            parent: Parent widget.
        """
        super().__init__(parent)

        self._labels = labels
        self._navigate_callback = navigate_callback

        self.setWindowTitle("Instance Size Distribution")
        self.setMinimumSize(750, 750)
        self.resize(850, 850)

        # Make dialog non-modal so user can interact with main window
        self.setModal(False)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Main widget
        self._widget = SizeDistributionWidget()
        self._widget.set_labels(self._labels)
        layout.addWidget(self._widget, stretch=1)

        # Button box
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

    def _connect_signals(self):
        """Connect widget signals."""
        if self._navigate_callback is not None:
            self._widget.navigate_to_frame.connect(self._on_navigate)

    def _on_navigate(self, video_idx: int, frame_idx: int, instance_idx: int):
        """Handle navigation request from widget."""
        if self._navigate_callback is not None:
            self._navigate_callback(video_idx, frame_idx, instance_idx)

    def set_rotation_preset(self, preset: str):
        """Set the rotation preset on the widget.

        Args:
            preset: One of "Off", "+/-15", "+/-180", "Custom"
        """
        self._widget.set_rotation_preset(preset)

    def set_custom_angle(self, angle: int):
        """Set the custom rotation angle on the widget.

        Args:
            angle: Angle in degrees (0-180).
        """
        self._widget.set_custom_angle(angle)

"""
Dialog for viewing label QC results.

Provides a standalone dialog that wraps the QCWidget
with navigation support back to the main window.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from qtpy import QtWidgets

from sleap.gui.widgets.qc import QCWidget

if TYPE_CHECKING:
    import sleap_io as sio


class QCDialog(QtWidgets.QDialog):
    """Dialog for label quality control analysis with navigation.

    This dialog displays the QCWidget and optionally connects
    navigation signals to the main window.

    Args:
        labels: The Labels object containing labeled frames.
        navigate_callback: Optional callback function that takes
            (video_idx, frame_idx, instance_idx) arguments. Called when
            user selects an instance to navigate to.
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

        self.setWindowTitle("Label Quality Control")
        self.setMinimumSize(700, 900)
        self.resize(800, 950)

        # Make dialog non-modal so user can interact with main window
        self.setModal(False)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Main widget
        self._widget = QCWidget()
        self._widget.set_labels(self._labels)
        layout.addWidget(self._widget, stretch=1)

        # Button box
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

    def _connect_signals(self):
        """Connect widget signals."""
        if self._navigate_callback is not None:
            self._widget.navigate_to_instance.connect(self._on_navigate)

    def _on_navigate(self, video_idx: int, frame_idx: int, instance_idx: int):
        """Handle navigation request from widget."""
        if self._navigate_callback is not None:
            self._navigate_callback(video_idx, frame_idx, instance_idx)

    def update_labels(self, labels: "sio.Labels"):
        """Update the labels being analyzed.

        Args:
            labels: New Labels object.
        """
        self._labels = labels
        self._widget.set_labels(labels)

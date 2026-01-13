"""Widget for live preview of rendered frames using sleap-io.

This module provides a preview widget that displays rendered frames using
sleap-io's `render_image()` function, allowing real-time preview of
rendering settings before exporting a video.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    import sleap_io as sio


class RenderingPreviewWidget(QtWidgets.QWidget):
    """Live preview of rendered frames using sleap-io.

    This widget displays a rendered frame with the current render settings and
    allows navigation through labeled frames. It uses sleap-io's `render_image()`
    function for actual rendering.

    Attributes:
        labels: The Labels object containing labeled frames.
        video: The video to render frames from.
        frameChanged: Signal emitted when frame changes (int: frame index).
    """

    frameChanged = QtCore.Signal(int)

    def __init__(
        self,
        labels: "sio.Labels",
        video: "sio.Video | None" = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        """Initialize the rendering preview widget.

        Args:
            labels: The Labels object containing labeled frames.
            video: The video to render. If None, uses first video in labels.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.labels = labels
        self.video = video or (labels.videos[0] if labels.videos else None)
        self._current_frame_idx = 0
        self._render_params: dict = {}
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_render)

        # Build list of labeled frame indices for this video
        self._labeled_frame_indices = self._get_labeled_frame_indices()

        self._setup_ui()
        self._update_slider_range()

        # Initial render
        if self._labeled_frame_indices:
            self._current_frame_idx = self._labeled_frame_indices[0]
            self._do_render()

    def _get_labeled_frame_indices(self) -> list[int]:
        """Get sorted list of frame indices with labels for current video."""
        if self.video is None:
            return [lf.frame_idx for lf in self.labels.labeled_frames]
        return sorted(
            lf.frame_idx for lf in self.labels.labeled_frames if lf.video == self.video
        )

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Preview image display
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a;")
        self.preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.preview_label, stretch=1)

        # Frame navigation controls
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.setContentsMargins(4, 4, 4, 4)

        # Previous frame button
        self.prev_btn = QtWidgets.QPushButton("<")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.setToolTip("Previous labeled frame")
        self.prev_btn.clicked.connect(self._go_prev_frame)
        nav_layout.addWidget(self.prev_btn)

        # Frame slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self.frame_slider, stretch=1)

        # Next frame button
        self.next_btn = QtWidgets.QPushButton(">")
        self.next_btn.setFixedWidth(30)
        self.next_btn.setToolTip("Next labeled frame")
        self.next_btn.clicked.connect(self._go_next_frame)
        nav_layout.addWidget(self.next_btn)

        # Frame label
        self.frame_label = QtWidgets.QLabel("Frame: 0")
        self.frame_label.setMinimumWidth(80)
        nav_layout.addWidget(self.frame_label)

        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-render current frame")
        self.refresh_btn.clicked.connect(self._do_render)
        nav_layout.addWidget(self.refresh_btn)

        layout.addLayout(nav_layout)

    def _update_slider_range(self):
        """Update slider to match labeled frames."""
        n_frames = len(self._labeled_frame_indices)
        self.frame_slider.setMaximum(max(0, n_frames - 1))
        self.prev_btn.setEnabled(n_frames > 1)
        self.next_btn.setEnabled(n_frames > 1)

    def _on_slider_changed(self, slider_idx: int):
        """Handle slider value change."""
        if 0 <= slider_idx < len(self._labeled_frame_indices):
            self._current_frame_idx = self._labeled_frame_indices[slider_idx]
            self.frame_label.setText(f"Frame: {self._current_frame_idx}")
            self._schedule_render()
            self.frameChanged.emit(self._current_frame_idx)

    def _go_prev_frame(self):
        """Navigate to previous labeled frame."""
        current_slider = self.frame_slider.value()
        if current_slider > 0:
            self.frame_slider.setValue(current_slider - 1)

    def _go_next_frame(self):
        """Navigate to next labeled frame."""
        current_slider = self.frame_slider.value()
        if current_slider < self.frame_slider.maximum():
            self.frame_slider.setValue(current_slider + 1)

    def set_render_params(self, **params):
        """Update rendering parameters and refresh preview.

        Args:
            **params: Keyword arguments passed to `sio.render_image()`.
        """
        self._render_params = params
        self._schedule_render()

    def set_frame(self, frame_idx: int):
        """Jump to a specific frame index.

        Args:
            frame_idx: Frame index to display.
        """
        if frame_idx in self._labeled_frame_indices:
            slider_idx = self._labeled_frame_indices.index(frame_idx)
            self.frame_slider.setValue(slider_idx)

    def _schedule_render(self, delay_ms: int = 100):
        """Schedule a debounced render.

        Args:
            delay_ms: Delay in milliseconds before rendering.
        """
        self._debounce_timer.start(delay_ms)

    def _do_render(self):
        """Render current frame with current settings."""
        import sleap_io as sio

        if not self._labeled_frame_indices:
            self._show_no_frames_message()
            return

        # Find the labeled frame
        lfs = self.labels.find(self.video, self._current_frame_idx)
        if not lfs:
            self._show_no_frames_message()
            return

        lf = lfs[0]

        try:
            # Render using sleap-io
            img = sio.render_image(lf, **self._render_params)
            self._display_image(img)
        except Exception as e:
            self._show_error_message(str(e))

    def _display_image(self, img: np.ndarray):
        """Convert numpy array to QPixmap and display.

        Args:
            img: Numpy array of image data (H, W, C).
        """
        # Ensure contiguous array
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1

        if c == 3:
            # RGB
            qimg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        elif c == 4:
            # RGBA
            qimg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGBA8888)
        else:
            # Grayscale
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap.fromImage(qimg)

        # Scale to fit while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.preview_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _show_no_frames_message(self):
        """Show message when no labeled frames available."""
        self.preview_label.setText("No labeled frames")
        self.preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")

    def _show_error_message(self, error: str):
        """Show error message.

        Args:
            error: Error message to display.
        """
        self.preview_label.setText(f"Render error:\n{error}")
        self.preview_label.setStyleSheet("background-color: #1a1a1a; color: #f88;")

    def resizeEvent(self, event):
        """Re-render on resize to fit new size."""
        super().resizeEvent(event)
        self._schedule_render(delay_ms=200)

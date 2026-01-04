"""
Widget for visualizing crop size distribution across instances.

Provides histogram and scatter plot views of instance bounding box sizes,
with support for rotation augmentation preview and click-to-navigate.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets

# Matplotlib setup with proper backend handling
import matplotlib
import os

if os.environ.get("MPLBACKEND") != "Agg":
    try:
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
    except ImportError:
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
else:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

from matplotlib.figure import Figure

if TYPE_CHECKING:
    import sleap_io as sio
    from sleap.gui.learning.crop_size import InstanceCropInfo


class CropSizeHistogramCanvas(Canvas):
    """Matplotlib canvas for displaying crop size distribution.

    Provides both scatter and histogram views with click-to-select
    functionality in scatter mode.

    Signals:
        point_clicked: Emitted when a point is clicked in scatter mode.
            Arguments are (video_idx, frame_idx, instance_idx).
    """

    point_clicked = QtCore.Signal(int, int, int)

    def __init__(self, width: int = 7, height: int = 5, dpi: int = 100):
        """Initialize the canvas.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.
            dpi: Dots per inch for the figure.
        """
        # Use constrained_layout for robust spacing that adapts to content
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setMinimumSize(400, 300)
        self.updateGeometry()

        self._data: List["InstanceCropInfo"] = []
        self._rotation_angle: float = 0.0
        self._scatter = None
        self._selected_idx: Optional[int] = None
        self._view_mode = "scatter"  # "scatter" or "histogram"

        # Store axis limits for stability
        self._x_limits: Optional[tuple] = None
        self._y_limits: Optional[tuple] = None

        # Connect pick event for scatter selection
        self.mpl_connect("pick_event", self._on_pick)

        self._setup_axes()

    def _setup_axes(self):
        """Configure the axes appearance."""
        self.axes.set_xlabel("Crop Size (pixels)", fontsize=10)
        self.axes.set_ylabel("Instance Index", fontsize=10)
        self.axes.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        self.axes.tick_params(labelsize=9)

    def set_data(self, data: List["InstanceCropInfo"]):
        """Set the instance crop size data.

        Args:
            data: List of InstanceCropInfo objects.
        """
        self._data = data
        self._selected_idx = None
        # Reset axis limits when data changes
        self._x_limits = None
        self._y_limits = None
        self.update_plot()

    def set_rotation_angle(self, angle: float):
        """Set the rotation angle for crop size calculation.

        Args:
            angle: Maximum rotation angle in degrees.
        """
        self._rotation_angle = angle
        # Reset x limits when rotation changes (y stays the same)
        self._x_limits = None
        self.update_plot()

    def set_view_mode(self, mode: str):
        """Set view mode.

        Args:
            mode: Either 'scatter' or 'histogram'.
        """
        self._view_mode = mode
        # Reset limits when switching views
        self._x_limits = None
        self._y_limits = None
        self.update_plot()

    def update_plot(self):
        """Redraw the plot with current data and settings."""
        self.axes.clear()
        self._setup_axes()
        self._scatter = None

        if not self._data:
            self.axes.text(
                0.5,
                0.5,
                "No data loaded\n\nLoad labels or click Recompute",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.draw()
            return

        # Calculate crop sizes with rotation
        crop_sizes = np.array(
            [d.get_rotated_crop_size(self._rotation_angle) for d in self._data]
        )

        if self._view_mode == "scatter":
            self._draw_scatter(crop_sizes)
        else:
            self._draw_histogram(crop_sizes)

        self.draw()

    def _draw_scatter(self, crop_sizes: np.ndarray):
        """Draw scatter plot of crop sizes.

        Args:
            crop_sizes: Array of computed crop sizes.
        """
        indices = np.arange(len(crop_sizes))

        # Color by relative size (outliers are redder)
        median = np.median(crop_sizes)
        if median > 0:
            colors = crop_sizes / median
        else:
            colors = np.ones_like(crop_sizes)

        self._scatter = self.axes.scatter(
            crop_sizes,
            indices,
            c=colors,
            cmap="RdYlBu_r",
            alpha=0.7,
            picker=True,
            pickradius=5,
            s=30,
            vmin=0.5,
            vmax=1.5,
        )

        # Add vertical lines for statistics
        mean_val = np.mean(crop_sizes)
        median_val = np.median(crop_sizes)
        max_val = np.max(crop_sizes)

        self.axes.axvline(
            median_val,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Median: {median_val:.0f}",
        )
        self.axes.axvline(
            mean_val,
            color="blue",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label=f"Mean: {mean_val:.0f}",
        )
        self.axes.axvline(
            max_val,
            color="red",
            linestyle="-",
            alpha=0.5,
            linewidth=1.5,
            label=f"Max: {max_val:.0f}",
        )

        # Highlight selected point
        if self._selected_idx is not None and self._selected_idx < len(crop_sizes):
            self.axes.scatter(
                [crop_sizes[self._selected_idx]],
                [self._selected_idx],
                s=150,
                facecolors="none",
                edgecolors="red",
                linewidths=2.5,
                zorder=10,
            )

        # Set fixed axis limits for stability
        if self._x_limits is None:
            min_val = np.min(crop_sizes)
            x_margin = (max_val - min_val) * 0.1 if max_val > min_val else 10
            self._x_limits = (max(0, min_val - x_margin), max_val + x_margin)

        if self._y_limits is None:
            self._y_limits = (-len(crop_sizes) * 0.02, len(crop_sizes) * 1.02)

        self.axes.set_xlim(self._x_limits)
        self.axes.set_ylim(self._y_limits)

        self.axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.axes.set_ylabel("Instance Index", fontsize=10)
        self.axes.set_title(
            f"Crop Size Distribution (n={len(crop_sizes)})", fontsize=11
        )

    def _draw_histogram(self, crop_sizes: np.ndarray):
        """Draw histogram of crop sizes.

        Args:
            crop_sizes: Array of computed crop sizes.
        """
        n_bins = min(50, max(10, len(crop_sizes) // 5 + 1))

        counts, bins, patches = self.axes.hist(
            crop_sizes, bins=n_bins, alpha=0.7, color="steelblue", edgecolor="white"
        )

        # Add statistics
        mean_val = np.mean(crop_sizes)
        median_val = np.median(crop_sizes)
        max_val = np.max(crop_sizes)

        self.axes.axvline(
            median_val,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Median: {median_val:.0f}",
        )
        self.axes.axvline(
            mean_val,
            color="blue",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label=f"Mean: {mean_val:.0f}",
        )
        self.axes.axvline(
            max_val,
            color="red",
            linestyle="-",
            alpha=0.5,
            linewidth=1.5,
            label=f"Max: {max_val:.0f}",
        )

        # Set fixed axis limits for stability
        if self._x_limits is None:
            min_val = np.min(crop_sizes)
            x_margin = (max_val - min_val) * 0.1 if max_val > min_val else 10
            self._x_limits = (max(0, min_val - x_margin), max_val + x_margin)

        self.axes.set_xlim(self._x_limits)

        self.axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.axes.set_ylabel("Count", fontsize=10)
        self.axes.set_title(f"Crop Size Histogram (n={len(crop_sizes)})", fontsize=11)

    def _on_pick(self, event):
        """Handle pick event on scatter points."""
        if event.artist != self._scatter:
            return

        if len(event.ind) == 0:
            return

        # Get the clicked point index
        idx = event.ind[0]
        self._selected_idx = idx

        if idx < len(self._data):
            info = self._data[idx]
            self.point_clicked.emit(info.video_idx, info.frame_idx, info.instance_idx)

        # Redraw to show selection highlight (keep axis limits stable)
        self.update_plot()


class CropSizeDistributionWidget(QtWidgets.QWidget):
    """Widget for visualizing crop size distribution with navigation.

    Provides controls for rotation augmentation preview, view mode selection,
    and click-to-navigate functionality for exploring outliers.

    Signals:
        navigate_to_frame: Emitted when user wants to navigate to a frame.
            Arguments are (video_idx, frame_idx, instance_idx).
    """

    navigate_to_frame = QtCore.Signal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._labels: Optional["sio.Labels"] = None
        self._data: List["InstanceCropInfo"] = []
        self._selected_info: Optional["InstanceCropInfo"] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title and recompute button row
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Crop Size Distribution</b>")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._recompute_button = QtWidgets.QPushButton("Recompute")
        self._recompute_button.setToolTip(
            "Recalculate crop sizes from current labels (user instances only)"
        )
        self._recompute_button.setFixedWidth(90)
        title_layout.addWidget(self._recompute_button)
        layout.addLayout(title_layout)

        # Rotation controls
        rotation_group = QtWidgets.QGroupBox("Rotation Augmentation")
        rotation_layout = QtWidgets.QHBoxLayout(rotation_group)
        rotation_layout.setContentsMargins(8, 8, 8, 8)

        self._rotation_combo = QtWidgets.QComboBox()
        self._rotation_combo.addItems(["Off", "+/-15", "+/-180", "Custom"])
        self._rotation_combo.setFixedWidth(100)
        rotation_layout.addWidget(QtWidgets.QLabel("Preset:"))
        rotation_layout.addWidget(self._rotation_combo)

        self._custom_angle_spin = QtWidgets.QSpinBox()
        self._custom_angle_spin.setRange(0, 180)
        self._custom_angle_spin.setValue(45)
        self._custom_angle_spin.setSuffix(" deg")
        self._custom_angle_spin.setEnabled(False)
        self._custom_angle_spin.setFixedWidth(80)
        rotation_layout.addWidget(QtWidgets.QLabel("Custom:"))
        rotation_layout.addWidget(self._custom_angle_spin)

        rotation_layout.addStretch()
        layout.addWidget(rotation_group)

        # View mode toggle
        view_layout = QtWidgets.QHBoxLayout()
        self._scatter_radio = QtWidgets.QRadioButton("Scatter (clickable)")
        self._histogram_radio = QtWidgets.QRadioButton("Histogram")
        self._scatter_radio.setChecked(True)
        view_layout.addWidget(self._scatter_radio)
        view_layout.addWidget(self._histogram_radio)
        view_layout.addStretch()
        layout.addLayout(view_layout)

        # Matplotlib canvas
        self._canvas = CropSizeHistogramCanvas(width=7, height=5)
        layout.addWidget(self._canvas, stretch=1)

        # Bottom panel: info and stats side by side
        bottom_layout = QtWidgets.QHBoxLayout()

        # Selection info panel
        info_group = QtWidgets.QGroupBox("Selected Instance")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)

        self._info_label = QtWidgets.QLabel("Click on a point to select an instance")
        self._info_label.setWordWrap(True)
        self._info_label.setMinimumHeight(80)
        info_layout.addWidget(self._info_label)

        self._goto_button = QtWidgets.QPushButton("Go to Frame")
        self._goto_button.setEnabled(False)
        info_layout.addWidget(self._goto_button)

        bottom_layout.addWidget(info_group)

        # Statistics panel
        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        self._stats_label = QtWidgets.QLabel("No data loaded")
        self._stats_label.setWordWrap(True)
        self._stats_label.setMinimumHeight(80)
        stats_layout.addWidget(self._stats_label)

        bottom_layout.addWidget(stats_group)

        layout.addLayout(bottom_layout)

    def _connect_signals(self):
        """Connect UI signals."""
        self._rotation_combo.currentTextChanged.connect(self._on_rotation_changed)
        self._custom_angle_spin.valueChanged.connect(self._on_custom_angle_changed)
        self._scatter_radio.toggled.connect(self._on_view_mode_changed)
        self._canvas.point_clicked.connect(self._on_point_clicked)
        self._goto_button.clicked.connect(self._on_goto_clicked)
        self._recompute_button.clicked.connect(self._on_recompute)

    def _get_rotation_angle(self) -> float:
        """Get the current rotation angle setting.

        Returns:
            Maximum rotation angle in degrees.
        """
        preset = self._rotation_combo.currentText()
        if preset == "Off":
            return 0.0
        elif preset == "+/-15":
            return 15.0
        elif preset == "+/-180":
            return 180.0
        else:  # Custom
            return float(self._custom_angle_spin.value())

    def _on_rotation_changed(self, text: str):
        """Handle rotation preset change."""
        self._custom_angle_spin.setEnabled(text == "Custom")
        self._canvas.set_rotation_angle(self._get_rotation_angle())
        self._update_statistics()
        self._update_selected_info()

    def _on_custom_angle_changed(self, value: int):
        """Handle custom angle change."""
        if self._rotation_combo.currentText() == "Custom":
            self._canvas.set_rotation_angle(float(value))
            self._update_statistics()
            self._update_selected_info()

    def _on_view_mode_changed(self, checked: bool):
        """Handle view mode toggle."""
        if checked:  # Scatter selected
            self._canvas.set_view_mode("scatter")
        else:
            self._canvas.set_view_mode("histogram")

    def _on_point_clicked(self, video_idx: int, frame_idx: int, instance_idx: int):
        """Handle point click in scatter plot."""
        # Find the clicked instance info
        for info in self._data:
            if (
                info.video_idx == video_idx
                and info.frame_idx == frame_idx
                and info.instance_idx == instance_idx
            ):
                self._selected_info = info
                break

        self._update_selected_info()
        self._goto_button.setEnabled(self._selected_info is not None)

    def _update_selected_info(self):
        """Update the selected instance info label."""
        if self._selected_info is None:
            self._info_label.setText("Click on a point to select an instance")
            return

        angle = self._get_rotation_angle()
        rotated_size = self._selected_info.get_rotated_crop_size(angle)

        info = self._selected_info
        raw_dims = f"({info.raw_width:.1f} x {info.raw_height:.1f})"
        self._info_label.setText(
            f"<b>Frame:</b> {info.frame_idx}<br/>"
            f"<b>Instance:</b> {info.instance_idx}<br/>"
            f"<b>Video:</b> {info.video_idx}<br/>"
            f"<b>Raw Size:</b> {info.raw_crop_size:.1f}px {raw_dims}<br/>"
            f"<b>Rotated Size:</b> {rotated_size:.1f}px"
        )

    def _on_goto_clicked(self):
        """Handle go-to-frame button click."""
        if self._selected_info:
            self.navigate_to_frame.emit(
                self._selected_info.video_idx,
                self._selected_info.frame_idx,
                self._selected_info.instance_idx,
            )

    def _on_recompute(self):
        """Handle recompute button click."""
        if self._labels is not None:
            self._compute_and_update()

    def _compute_and_update(self):
        """Compute crop sizes from labels and update display."""
        if self._labels is None:
            return

        # Import here to avoid circular imports
        from sleap.gui.learning.crop_size import compute_instance_crop_sizes

        self._data = compute_instance_crop_sizes(self._labels, user_instances_only=True)
        self._selected_info = None
        self._info_label.setText("Click on a point to select an instance")
        self._goto_button.setEnabled(False)

        self._canvas.set_data(self._data)
        self._update_statistics()

    def _update_statistics(self):
        """Update the statistics panel."""
        if not self._data:
            self._stats_label.setText("No data loaded")
            return

        angle = self._get_rotation_angle()
        crop_sizes = np.array([d.get_rotated_crop_size(angle) for d in self._data])

        # Calculate statistics
        mean_val = np.mean(crop_sizes)
        median_val = np.median(crop_sizes)
        std_val = np.std(crop_sizes)
        min_val = np.min(crop_sizes)
        max_val = np.max(crop_sizes)

        # Count potential outliers (>2 std from mean)
        outlier_threshold = mean_val + 2 * std_val
        n_outliers = int(np.sum(crop_sizes > outlier_threshold))

        # Percentiles
        p90 = np.percentile(crop_sizes, 90)
        p95 = np.percentile(crop_sizes, 95)
        p99 = np.percentile(crop_sizes, 99)

        pct = 100 * n_outliers / len(crop_sizes) if len(crop_sizes) > 0 else 0

        self._stats_label.setText(
            f"<b>Count:</b> {len(crop_sizes)}<br/>"
            f"<b>Range:</b> {min_val:.0f} - {max_val:.0f}px<br/>"
            f"<b>Mean +/- Std:</b> {mean_val:.0f} +/- {std_val:.0f}px<br/>"
            f"<b>Median:</b> {median_val:.0f}px<br/>"
            f"<b>90th/95th/99th:</b> {p90:.0f} / {p95:.0f} / {p99:.0f}px<br/>"
            f"<b>Outliers (>2 sigma):</b> {n_outliers} ({pct:.1f}%)"
        )

    def set_labels(self, labels: "sio.Labels"):
        """Set the labels data and compute crop sizes.

        Args:
            labels: A sleap_io.Labels object.
        """
        self._labels = labels
        self._compute_and_update()

    def set_rotation_preset(self, preset: str):
        """Set the rotation preset programmatically.

        Args:
            preset: One of "Off", "+/-15", "+/-180", "Custom"
        """
        index = self._rotation_combo.findText(preset)
        if index >= 0:
            self._rotation_combo.setCurrentIndex(index)

    def set_custom_angle(self, angle: int):
        """Set the custom rotation angle.

        Args:
            angle: Angle in degrees (0-180).
        """
        self._custom_angle_spin.setValue(angle)

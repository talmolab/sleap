"""
Widget for visualizing instance size distribution.

Provides histogram and scatter plot views of instance bounding box sizes,
with support for rotation augmentation preview and click-to-navigate.
Useful for determining crop sizes and identifying outlier annotations.
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
    from sleap.gui.learning.size import InstanceSizeInfo


class SizeHistogramCanvas(Canvas):
    """Matplotlib canvas for displaying instance size distribution.

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

        self._data: List["InstanceSizeInfo"] = []
        self._rotation_angle: float = 0.0
        self._scatter = None
        self._selected_idx: Optional[int] = None
        self._view_mode = "scatter"  # "scatter" or "histogram"

        # Histogram settings
        self._hist_bins: int = 30
        self._hist_x_min: Optional[float] = None  # None = auto
        self._hist_x_max: Optional[float] = None  # None = auto

        # Store axis limits for stability
        self._x_limits: Optional[tuple] = None
        self._y_limits: Optional[tuple] = None

        # Connect pick event for scatter selection
        self.mpl_connect("pick_event", self._on_pick)

        self._setup_axes()

    def _setup_axes(self):
        """Configure the axes appearance."""
        self.axes.set_xlabel("Size (pixels)", fontsize=10)
        self.axes.set_ylabel("Instance Index", fontsize=10)
        self.axes.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        self.axes.tick_params(labelsize=9)

    def set_data(self, data: List["InstanceSizeInfo"]):
        """Set the instance size data.

        Args:
            data: List of InstanceSizeInfo objects.
        """
        self._data = data
        self._selected_idx = None
        # Reset axis limits when data changes
        self._x_limits = None
        self._y_limits = None
        self.update_plot()

    def set_rotation_angle(self, angle: float):
        """Set the rotation angle for size calculation.

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

    def set_histogram_bins(self, bins: int):
        """Set the number of histogram bins.

        Args:
            bins: Number of bins for the histogram.
        """
        self._hist_bins = max(5, min(100, bins))
        if self._view_mode == "histogram":
            self.update_plot()

    def set_histogram_range(
        self, x_min: Optional[float] = None, x_max: Optional[float] = None
    ):
        """Set the histogram x-axis range.

        Args:
            x_min: Minimum x value, or None for auto.
            x_max: Maximum x value, or None for auto.
        """
        self._hist_x_min = x_min
        self._hist_x_max = x_max
        self._x_limits = None  # Force recalculation
        if self._view_mode == "histogram":
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

        # Calculate sizes with rotation
        sizes = np.array([d.get_rotated_size(self._rotation_angle) for d in self._data])

        if self._view_mode == "scatter":
            self._draw_scatter(sizes)
        else:
            self._draw_histogram(sizes)

        self.draw()

    def _draw_scatter(self, sizes: np.ndarray):
        """Draw scatter plot of sizes.

        Args:
            sizes: Array of computed sizes.
        """
        indices = np.arange(len(sizes))

        # Color by relative size (outliers are redder)
        median = np.median(sizes)
        if median > 0:
            colors = sizes / median
        else:
            colors = np.ones_like(sizes)

        self._scatter = self.axes.scatter(
            sizes,
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
        mean_val = np.mean(sizes)
        median_val = np.median(sizes)
        max_val = np.max(sizes)

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
        if self._selected_idx is not None and self._selected_idx < len(sizes):
            self.axes.scatter(
                [sizes[self._selected_idx]],
                [self._selected_idx],
                s=150,
                facecolors="none",
                edgecolors="red",
                linewidths=2.5,
                zorder=10,
            )

        # Set fixed axis limits for stability
        if self._x_limits is None:
            min_val = np.min(sizes)
            x_margin = (max_val - min_val) * 0.1 if max_val > min_val else 10
            self._x_limits = (max(0, min_val - x_margin), max_val + x_margin)

        if self._y_limits is None:
            self._y_limits = (-len(sizes) * 0.02, len(sizes) * 1.02)

        self.axes.set_xlim(self._x_limits)
        self.axes.set_ylim(self._y_limits)

        # Legend outside to the right
        self.axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            framealpha=0.9,
            borderaxespad=0,
        )
        self.axes.set_ylabel("Instance Index", fontsize=10)
        self.axes.set_title(f"Size Distribution (n={len(sizes)})", fontsize=11)

    def _draw_histogram(self, sizes: np.ndarray):
        """Draw histogram of sizes.

        Args:
            sizes: Array of computed sizes.
        """
        # Determine range
        data_min = np.min(sizes)
        data_max = np.max(sizes)

        hist_min = self._hist_x_min if self._hist_x_min is not None else data_min
        hist_max = self._hist_x_max if self._hist_x_max is not None else data_max

        # Filter data to range for histogram
        mask = (sizes >= hist_min) & (sizes <= hist_max)
        filtered_sizes = sizes[mask]

        if len(filtered_sizes) == 0:
            filtered_sizes = sizes  # Fall back to all data

        counts, bins, patches = self.axes.hist(
            filtered_sizes,
            bins=self._hist_bins,
            range=(hist_min, hist_max),
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )

        # Add statistics (computed on all data)
        mean_val = np.mean(sizes)
        median_val = np.median(sizes)
        max_val = np.max(sizes)

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

        # Set axis limits
        x_margin = (hist_max - hist_min) * 0.05 if hist_max > hist_min else 10
        self.axes.set_xlim(hist_min - x_margin, hist_max + x_margin)

        # Legend outside to the right
        self.axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            framealpha=0.9,
            borderaxespad=0,
        )
        self.axes.set_ylabel("Count", fontsize=10)
        self.axes.set_title(f"Size Histogram (n={len(sizes)})", fontsize=11)

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


class SizeDistributionWidget(QtWidgets.QWidget):
    """Widget for visualizing instance size distribution with navigation.

    Provides controls for rotation augmentation preview, view mode selection,
    histogram configuration, and click-to-navigate functionality for exploring
    outliers.

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
        self._data: List["InstanceSizeInfo"] = []
        self._selected_info: Optional["InstanceSizeInfo"] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title and recompute button row
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Instance Size Distribution</b>")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._recompute_button = QtWidgets.QPushButton("Recompute")
        self._recompute_button.setToolTip(
            "Recalculate sizes from current labels (user instances only)"
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

        # View mode toggle and histogram controls
        view_layout = QtWidgets.QHBoxLayout()
        self._scatter_radio = QtWidgets.QRadioButton("Scatter (clickable)")
        self._histogram_radio = QtWidgets.QRadioButton("Histogram")
        self._scatter_radio.setChecked(True)
        view_layout.addWidget(self._scatter_radio)
        view_layout.addWidget(self._histogram_radio)

        view_layout.addSpacing(20)

        # Histogram controls (enabled only in histogram mode)
        view_layout.addWidget(QtWidgets.QLabel("Bins:"))
        self._bins_spin = QtWidgets.QSpinBox()
        self._bins_spin.setRange(5, 100)
        self._bins_spin.setValue(30)
        self._bins_spin.setFixedWidth(60)
        self._bins_spin.setEnabled(False)
        view_layout.addWidget(self._bins_spin)

        view_layout.addWidget(QtWidgets.QLabel("X-min:"))
        self._xmin_spin = QtWidgets.QSpinBox()
        self._xmin_spin.setRange(0, 10000)
        self._xmin_spin.setValue(0)
        self._xmin_spin.setSpecialValueText("Auto")
        self._xmin_spin.setFixedWidth(70)
        self._xmin_spin.setEnabled(False)
        view_layout.addWidget(self._xmin_spin)

        view_layout.addWidget(QtWidgets.QLabel("X-max:"))
        self._xmax_spin = QtWidgets.QSpinBox()
        self._xmax_spin.setRange(0, 10000)
        self._xmax_spin.setValue(0)
        self._xmax_spin.setSpecialValueText("Auto")
        self._xmax_spin.setFixedWidth(70)
        self._xmax_spin.setEnabled(False)
        view_layout.addWidget(self._xmax_spin)

        view_layout.addStretch()
        layout.addLayout(view_layout)

        # Matplotlib canvas
        self._canvas = SizeHistogramCanvas(width=7, height=5)
        layout.addWidget(self._canvas, stretch=1)

        # Bottom panel: info and stats side by side
        bottom_layout = QtWidgets.QHBoxLayout()

        # Selection info panel
        info_group = QtWidgets.QGroupBox("Selected Instance")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)

        self._info_label = QtWidgets.QLabel(
            "Click on a point to select and navigate to instance"
        )
        self._info_label.setWordWrap(True)
        self._info_label.setMinimumHeight(80)
        info_layout.addWidget(self._info_label)

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
        self._recompute_button.clicked.connect(self._on_recompute)

        # Histogram controls
        self._bins_spin.valueChanged.connect(self._on_bins_changed)
        self._xmin_spin.valueChanged.connect(self._on_xrange_changed)
        self._xmax_spin.valueChanged.connect(self._on_xrange_changed)

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
        is_histogram = not checked
        self._bins_spin.setEnabled(is_histogram)
        self._xmin_spin.setEnabled(is_histogram)
        self._xmax_spin.setEnabled(is_histogram)

        if checked:  # Scatter selected
            self._canvas.set_view_mode("scatter")
        else:
            self._canvas.set_view_mode("histogram")

    def _on_bins_changed(self, value: int):
        """Handle histogram bins change."""
        self._canvas.set_histogram_bins(value)

    def _on_xrange_changed(self):
        """Handle histogram x-range change."""
        x_min = self._xmin_spin.value() if self._xmin_spin.value() > 0 else None
        x_max = self._xmax_spin.value() if self._xmax_spin.value() > 0 else None
        self._canvas.set_histogram_range(x_min, x_max)

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

        # Navigate directly to the frame
        if self._selected_info is not None:
            self.navigate_to_frame.emit(
                self._selected_info.video_idx,
                self._selected_info.frame_idx,
                self._selected_info.instance_idx,
            )

    def _update_selected_info(self):
        """Update the selected instance info label."""
        if self._selected_info is None:
            self._info_label.setText(
                "Click on a point to select and navigate to instance"
            )
            return

        angle = self._get_rotation_angle()
        rotated_size = self._selected_info.get_rotated_size(angle)

        info = self._selected_info
        raw_dims = f"({info.raw_width:.1f} x {info.raw_height:.1f})"
        self._info_label.setText(
            f"<b>Frame:</b> {info.frame_idx}<br/>"
            f"<b>Instance:</b> {info.instance_idx}<br/>"
            f"<b>Video:</b> {info.video_idx}<br/>"
            f"<b>Raw Size:</b> {info.raw_size:.1f}px {raw_dims}<br/>"
            f"<b>Rotated Size:</b> {rotated_size:.1f}px"
        )

    def _on_recompute(self):
        """Handle recompute button click."""
        if self._labels is not None:
            self._compute_and_update()

    def _compute_and_update(self):
        """Compute sizes from labels and update display."""
        if self._labels is None:
            return

        # Import here to avoid circular imports
        from sleap.gui.learning.size import compute_instance_sizes

        self._data = compute_instance_sizes(self._labels, user_instances_only=True)
        self._selected_info = None
        self._info_label.setText("Click on a point to select and navigate to instance")

        self._canvas.set_data(self._data)
        self._update_statistics()

    def _update_statistics(self):
        """Update the statistics panel."""
        if not self._data:
            self._stats_label.setText("No data loaded")
            return

        angle = self._get_rotation_angle()
        sizes = np.array([d.get_rotated_size(angle) for d in self._data])

        # Calculate statistics
        mean_val = np.mean(sizes)
        median_val = np.median(sizes)
        std_val = np.std(sizes)
        min_val = np.min(sizes)
        max_val = np.max(sizes)

        # Count potential outliers (>2 std from mean)
        outlier_threshold = mean_val + 2 * std_val
        n_outliers = int(np.sum(sizes > outlier_threshold))

        # Percentiles
        p90 = np.percentile(sizes, 90)
        p95 = np.percentile(sizes, 95)
        p99 = np.percentile(sizes, 99)

        pct = 100 * n_outliers / len(sizes) if len(sizes) > 0 else 0

        self._stats_label.setText(
            f"<b>Count:</b> {len(sizes)}<br/>"
            f"<b>Range:</b> {min_val:.0f} - {max_val:.0f}px<br/>"
            f"<b>Mean +/- Std:</b> {mean_val:.0f} +/- {std_val:.0f}px<br/>"
            f"<b>Median:</b> {median_val:.0f}px<br/>"
            f"<b>90th/95th/99th:</b> {p90:.0f} / {p95:.0f} / {p99:.0f}px<br/>"
            f"<b>Outliers (>2 sigma):</b> {n_outliers} ({pct:.1f}%)"
        )

    def set_labels(self, labels: "sio.Labels"):
        """Set the labels data and compute sizes.

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

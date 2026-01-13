"""
Widget for visualizing label QC results.

Provides histogram and table views of instance anomaly scores,
with click-to-navigate support for reviewing flagged annotations.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import QThread, Signal as QSignal

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
    from sleap.qc.results import QCResults, QCFlag


class QCScoreCanvas(Canvas):
    """Matplotlib canvas for displaying QC score distribution.

    Provides histogram visualization with threshold indicator and
    click-to-select functionality.

    Signals:
        threshold_changed: Emitted when user clicks to set threshold.
            Argument is the new threshold value (0-1).
    """

    threshold_changed = QtCore.Signal(float)

    def __init__(self, width: int = 6, height: int = 3, dpi: int = 100):
        """Initialize the canvas.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.
            dpi: Dots per inch for the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        # when docked. This respects size hints without fighting the splitter.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 150)

        self._scores: np.ndarray = np.array([])
        self._threshold: float = 0.7
        self._threshold_line = None

        # Connect click event for threshold adjustment
        self.mpl_connect("button_press_event", self._on_click)

        self._setup_axes()

    def _setup_axes(self):
        """Configure the axes appearance."""
        self.axes.set_xlabel("Anomaly Score", fontsize=10)
        self.axes.set_ylabel("Count", fontsize=10)
        self.axes.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        self.axes.tick_params(labelsize=9)

    def set_scores(self, scores: np.ndarray):
        """Set the anomaly scores to display.

        Args:
            scores: Array of anomaly scores (0-1).
        """
        self._scores = scores
        self.update_plot()

    def set_threshold(self, threshold: float):
        """Set the threshold line position.

        Args:
            threshold: Threshold value (0-1).
        """
        self._threshold = threshold
        self.update_plot()

    def update_plot(self):
        """Redraw the plot with current data and threshold."""
        self.axes.clear()
        self._setup_axes()

        if len(self._scores) == 0:
            self.axes.text(
                0.5,
                0.5,
                "No data\n\nClick 'Run Analysis' to start",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.draw()
            return

        # Draw histogram with fixed bins from 0 to 1
        bins = np.linspace(0, 1, 51)  # 50 bins for finer detail
        n_flagged = np.sum(self._scores >= self._threshold)
        n_total = len(self._scores)

        # Color bars based on threshold
        counts, bin_edges, patches = self.axes.hist(
            self._scores,
            bins=bins,
            alpha=0.7,
            edgecolor="white",
        )

        # Color bars based on whether they're above/below threshold
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            if left_edge >= self._threshold:
                patch.set_facecolor("#dc3545")  # Red for flagged
            else:
                patch.set_facecolor("#6c757d")  # Gray for normal

        # Draw threshold line
        self._threshold_line = self.axes.axvline(
            self._threshold,
            color="#007bff",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {self._threshold:.2f}",
        )

        # Add annotation for flagged count
        self.axes.annotate(
            f"{n_flagged} flagged\n({100 * n_flagged / n_total:.1f}%)",
            xy=(self._threshold + 0.02, self.axes.get_ylim()[1] * 0.9),
            fontsize=9,
            color="#dc3545",
            fontweight="bold",
        )

        self.axes.set_xlim(0, 1)
        self.axes.set_title(
            f"Score Distribution (n={n_total})",
            fontsize=11,
        )
        self.axes.legend(loc="upper left", fontsize=8)

        self.draw()

    def _on_click(self, event):
        """Handle click event to set threshold."""
        if event.inaxes != self.axes:
            return

        # Get x coordinate of click
        x = event.xdata
        if x is not None and 0 <= x <= 1:
            self.threshold_changed.emit(float(x))


class QCBreakdownCanvas(Canvas):
    """Matplotlib canvas for displaying error type breakdown.

    Shows a horizontal bar chart of top issues.
    """

    def __init__(self, width: int = 6, height: int = 2.5, dpi: int = 100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 120)

        self._issue_counts: dict = {}

    def set_issue_counts(self, issue_counts: dict):
        """Set the issue type counts to display.

        Args:
            issue_counts: Dict mapping issue name to count.
        """
        self._issue_counts = issue_counts
        self.update_plot()

    def update_plot(self):
        """Redraw the breakdown chart."""
        self.axes.clear()

        if not self._issue_counts:
            self.axes.text(
                0.5,
                0.5,
                "No flagged instances",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.axes.set_title("Issue Breakdown", fontsize=11)
            self.draw()
            return

        # Sort by count descending, show ALL issue types
        sorted_issues = sorted(
            self._issue_counts.items(), key=lambda x: x[1], reverse=True
        )

        labels = [item[0] for item in sorted_issues]
        counts = [item[1] for item in sorted_issues]
        max_count = max(counts) if counts else 1

        # Horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = self.axes.barh(y_pos, counts, color="#dc3545", alpha=0.7)

        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels(labels, fontsize=9)
        self.axes.invert_yaxis()  # Top to bottom
        self.axes.set_xlabel("Count", fontsize=10)
        self.axes.set_title("Issue Breakdown", fontsize=11)

        # Add count labels - inside bar (white) if bar is wide enough, else outside
        for bar, count in zip(bars, counts):
            bar_width = bar.get_width()
            y_center = bar.get_y() + bar.get_height() / 2

            # If bar is at least 20% of max width, put label inside
            if bar_width >= max_count * 0.2:
                self.axes.text(
                    bar_width - max_count * 0.02,  # Slightly inside right edge
                    y_center,
                    str(count),
                    va="center",
                    ha="right",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            else:
                # Put label outside the bar
                self.axes.text(
                    bar_width + max_count * 0.02,
                    y_center,
                    str(count),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#333",
                )

        # Add some padding on the right for labels
        self.axes.set_xlim(0, max_count * 1.15)

        self.draw()


class QCFeatureCanvas(Canvas):
    """Matplotlib canvas for displaying feature distributions.

    Shows box plots comparing flagged vs non-flagged instances across
    top contributing features.
    """

    def __init__(self, width: int = 6, height: int = 2.5, dpi: int = 100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 120)

        self._feature_data: dict = {}  # {feature_name: (normal_values, flagged_values)}
        self._top_features: list = []

    def set_feature_data(
        self,
        feature_contributions: dict,
        instance_scores: dict,
        threshold: float,
        feature_names: list,
    ):
        """Set the feature data to display.

        Args:
            feature_contributions: Dict mapping InstanceKey to feature dict.
            instance_scores: Dict mapping InstanceKey to score.
            threshold: Threshold for flagging instances.
            feature_names: List of all feature names.
        """
        if not feature_contributions or not feature_names:
            self._feature_data = {}
            self._top_features = []
            self.update_plot()
            return

        # Separate flagged vs normal
        normal_features = {name: [] for name in feature_names}
        flagged_features = {name: [] for name in feature_names}

        for key, contributions in feature_contributions.items():
            score = instance_scores.get(key, 0)
            target = flagged_features if score >= threshold else normal_features

            for name in feature_names:
                val = contributions.get(name, 0)
                if np.isfinite(val):
                    target[name].append(val)

        # Find top discriminating features by difference in means
        feature_scores = []
        for name in feature_names:
            normal_vals = normal_features.get(name, [])
            flagged_vals = flagged_features.get(name, [])

            if normal_vals and flagged_vals:
                normal_mean = np.mean(normal_vals)
                normal_std = np.std(normal_vals) or 1.0
                flagged_mean = np.mean(flagged_vals)
                # Z-score of difference
                diff = abs(flagged_mean - normal_mean) / normal_std
                feature_scores.append((name, diff))

        # Sort by discriminating power and take top 6
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        self._top_features = [name for name, _ in feature_scores[:6]]

        # Store the data
        self._feature_data = {
            name: (normal_features[name], flagged_features[name])
            for name in self._top_features
        }

        self.update_plot()

    def update_plot(self):
        """Redraw the feature comparison chart."""
        self.axes.clear()

        if not self._feature_data or not self._top_features:
            self.axes.text(
                0.5,
                0.5,
                "No feature data\n\nRun analysis to see feature distributions",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.axes.set_title("Feature Comparison", fontsize=11)
            self.draw()
            return

        # Prepare data for box plots
        positions = []
        box_data = []
        colors = []
        tick_labels = []

        for i, name in enumerate(self._top_features):
            normal_vals, flagged_vals = self._feature_data[name]
            base_pos = i * 2.5

            # Normal values
            if normal_vals:
                positions.append(base_pos)
                box_data.append(normal_vals)
                colors.append("#6c757d")  # Gray for normal
                tick_labels.append("")

            # Flagged values
            if flagged_vals:
                positions.append(base_pos + 0.8)
                box_data.append(flagged_vals)
                colors.append("#dc3545")  # Red for flagged
                tick_labels.append("")

        if not box_data:
            self.axes.text(
                0.5,
                0.5,
                "Insufficient data for comparison",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.draw()
            return

        # Create box plots
        bp = self.axes.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Hide outliers for cleaner view
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set x-axis labels
        feature_positions = [i * 2.5 + 0.4 for i in range(len(self._top_features))]
        self.axes.set_xticks(feature_positions)
        # Shorten feature names
        short_names = [
            name.replace("_", " ").replace(" zscore", "")[:12]
            for name in self._top_features
        ]
        self.axes.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#6c757d", alpha=0.7, label="Normal"),
            Patch(facecolor="#dc3545", alpha=0.7, label="Flagged"),
        ]
        self.axes.legend(handles=legend_elements, loc="upper right", fontsize=8)

        self.axes.set_ylabel("Feature Value", fontsize=9)
        self.axes.set_title("Top Discriminating Features", fontsize=11)
        self.axes.grid(True, alpha=0.3, axis="y")

        self.draw()


class QCFlagTableModel(QtCore.QAbstractTableModel):
    """Table model for QC flagged instances."""

    COLUMNS = ["Frame", "Instance", "Score", "Confidence", "Issue"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: List["QCFlag"] = []

    @property
    def items(self) -> List["QCFlag"]:
        """Get the current items."""
        return self._items

    @items.setter
    def items(self, value: List["QCFlag"]):
        """Set items and refresh the model."""
        self.beginResetModel()
        self._items = value
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        return len(self._items)

    def columnCount(self, parent=None) -> int:
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None

        item = self._items[index.row()]
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:  # Frame
                return str(item.frame_idx)
            elif col == 1:  # Instance
                return str(item.instance_idx)
            elif col == 2:  # Score
                return f"{item.score:.3f}"
            elif col == 3:  # Confidence
                return item.confidence.title()
            elif col == 4:  # Issue
                return item.top_issue.replace("_", " ").title()

        elif role == QtCore.Qt.ForegroundRole:
            if col == 2:  # Score column
                if item.score >= 0.8:
                    return QtGui.QBrush(QtGui.QColor(220, 53, 69))  # Red
                elif item.score >= 0.6:
                    return QtGui.QBrush(QtGui.QColor(255, 193, 7))  # Yellow
            elif col == 3:  # Confidence column
                if item.confidence == "high":
                    return QtGui.QBrush(QtGui.QColor(220, 53, 69))
                elif item.confidence == "medium":
                    return QtGui.QBrush(QtGui.QColor(255, 193, 7))
                else:
                    return QtGui.QBrush(QtGui.QColor(108, 117, 125))

        return None

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder):
        """Sort the model by the given column.

        Args:
            column: Column index to sort by.
            order: Sort order (AscendingOrder or DescendingOrder).
        """
        self.beginResetModel()

        reverse = order == QtCore.Qt.DescendingOrder

        # Define sort key for each column
        if column == 0:  # Frame
            key = lambda x: x.frame_idx
        elif column == 1:  # Instance
            key = lambda x: x.instance_idx
        elif column == 2:  # Score
            key = lambda x: x.score
        elif column == 3:  # Confidence (high > medium > low)
            conf_order = {"high": 2, "medium": 1, "low": 0}
            key = lambda x: conf_order.get(x.confidence, -1)
        elif column == 4:  # Issue (alphabetical)
            key = lambda x: x.top_issue
        else:
            key = lambda x: 0

        self._items.sort(key=key, reverse=reverse)
        self.endResetModel()


class QCAnalysisWorker(QThread):
    """Worker thread for running QC analysis in background.

    Signals:
        progress: Emitted with (step_name, progress_pct, detail) during analysis.
        finished: Emitted with QCResults when analysis completes.
        error: Emitted with error message if analysis fails.
    """

    progress = QSignal(str, int, str)  # (step_name, progress_percent, detail)
    finished = QSignal(object)  # QCResults
    error = QSignal(str)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._labels = labels
        self._results = None
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the analysis."""
        self._cancelled = True

    def run(self):
        """Run the QC analysis."""
        try:
            from sleap.qc import LabelQCDetector

            def progress_callback(step_name, progress_fraction, detail=None):
                """Handle progress updates from detector."""
                if self._cancelled:
                    raise InterruptedError("Analysis cancelled")
                progress_pct = int(progress_fraction * 100)
                self.progress.emit(step_name, progress_pct, detail or "")

            # Create detector
            self.progress.emit("Initializing...", 0, "")
            detector = LabelQCDetector()

            # Fit model with progress callback
            detector.fit(self._labels, progress_callback=progress_callback)

            if self._cancelled:
                return

            # Score instances with progress callback
            results = detector.score(self._labels, progress_callback=progress_callback)

            if self._cancelled:
                return

            # Complete
            self.progress.emit("Complete", 100, "")
            self.finished.emit(results)

        except InterruptedError:
            # Analysis was cancelled, just return silently
            pass
        except Exception as e:
            self.error.emit(str(e))


class QCWidget(QtWidgets.QWidget):
    """Widget for label quality control analysis with visualizations.

    Provides controls for running QC analysis, viewing score distributions,
    and navigating to flagged instances.

    Signals:
        navigate_to_instance: Emitted when user wants to navigate to an instance.
            Arguments are (video_idx, frame_idx, instance_idx).
    """

    navigate_to_instance = QtCore.Signal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._labels: Optional["sio.Labels"] = None
        self._detector = None
        self._results: Optional["QCResults"] = None
        self._selected_flag: Optional["QCFlag"] = None
        self._worker: Optional[QCAnalysisWorker] = None
        self._last_export_dir: Optional[str] = None  # Persist export directory

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # === Top row: title and run button ===
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Label Quality Control</b>")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._run_button = QtWidgets.QPushButton("Run Analysis")
        self._run_button.setToolTip(
            "Analyze all labeled instances for potential annotation errors"
        )
        self._run_button.setFixedWidth(100)
        title_layout.addWidget(self._run_button)
        layout.addLayout(title_layout)

        # Progress bar with status and cancel button (hidden by default)
        progress_layout = QtWidgets.QHBoxLayout()
        self._progress_label = QtWidgets.QLabel("")
        self._progress_label.setVisible(False)
        progress_layout.addWidget(self._progress_label)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        progress_layout.addWidget(self._progress_bar, stretch=1)

        # Cancel button
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.setVisible(False)
        self._cancel_button.setFixedWidth(60)
        self._cancel_button.setToolTip("Cancel the running analysis")
        progress_layout.addWidget(self._cancel_button)

        layout.addLayout(progress_layout)

        # Timer for spinner animation during analysis
        self._spinner_timer = QtCore.QTimer(self)
        self._spinner_timer.setInterval(100)  # 100ms
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0

        # === Threshold control ===
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("Sensitivity:"))
        threshold_layout.addWidget(QtWidgets.QLabel("More"))

        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setMinimum(30)
        self._threshold_slider.setMaximum(90)
        self._threshold_slider.setValue(70)
        self._threshold_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._threshold_slider.setTickInterval(10)
        self._threshold_slider.setToolTip(
            "Lower threshold = more instances flagged (higher sensitivity)\n"
            "Click on the histogram to set threshold visually"
        )
        threshold_layout.addWidget(self._threshold_slider, stretch=1)

        threshold_layout.addWidget(QtWidgets.QLabel("Fewer"))

        self._threshold_label = QtWidgets.QLabel("0.70")
        self._threshold_label.setMinimumWidth(40)
        self._threshold_label.setAlignment(QtCore.Qt.AlignCenter)
        self._threshold_label.setStyleSheet(
            "font-weight: bold; background: #f8f9fa; "
            "padding: 2px 6px; border-radius: 3px;"
        )
        threshold_layout.addWidget(self._threshold_label)

        layout.addLayout(threshold_layout)

        # === Tabbed visualization area ===
        self._viz_tabs = QtWidgets.QTabWidget()
        self._viz_tabs.setMinimumHeight(180)
        # Let the tabs shrink when space is limited but not expand unboundedly
        self._viz_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )

        # Score distribution tab
        self._score_canvas = QCScoreCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._score_canvas, "Score Distribution")

        # Issue breakdown tab
        self._breakdown_canvas = QCBreakdownCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._breakdown_canvas, "Issue Breakdown")

        # Features tab
        self._feature_canvas = QCFeatureCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._feature_canvas, "Features")

        layout.addWidget(self._viz_tabs)

        # === Flagged instances table ===
        table_group = QtWidgets.QGroupBox("Flagged Instances")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        table_layout.setContentsMargins(4, 4, 4, 4)

        self._table_model = QCFlagTableModel()
        self._table_view = QtWidgets.QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self._table_view.setSelectionMode(QtWidgets.QTableView.SingleSelection)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSortingEnabled(True)
        self._table_view.setMinimumHeight(120)

        # Set column widths
        header = self._table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        table_layout.addWidget(self._table_view)
        layout.addWidget(table_group, stretch=1)

        # === Bottom panel: selected instance info and statistics ===
        bottom_layout = QtWidgets.QHBoxLayout()

        # Selected instance details
        details_group = QtWidgets.QGroupBox("Selected Instance")
        details_layout = QtWidgets.QVBoxLayout(details_group)
        details_layout.setContentsMargins(6, 6, 6, 6)

        self._details_label = QtWidgets.QLabel(
            "Click a row in the table to select an instance"
        )
        self._details_label.setWordWrap(True)
        self._details_label.setMinimumHeight(70)
        details_layout.addWidget(self._details_label)

        bottom_layout.addWidget(details_group)

        # Statistics panel
        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(6, 6, 6, 6)

        self._stats_label = QtWidgets.QLabel("No analysis run yet")
        self._stats_label.setWordWrap(True)
        self._stats_label.setMinimumHeight(70)
        stats_layout.addWidget(self._stats_label)

        bottom_layout.addWidget(stats_group)

        layout.addLayout(bottom_layout)

    def _connect_signals(self):
        """Connect UI signals."""
        self._run_button.clicked.connect(self._on_run_analysis)
        self._cancel_button.clicked.connect(self._on_cancel_analysis)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._score_canvas.threshold_changed.connect(self._on_canvas_threshold_changed)
        self._table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._table_view.doubleClicked.connect(self._on_row_double_clicked)

    def _update_spinner(self):
        """Update the spinner animation character."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        # Update the progress label with spinner
        current_text = self._progress_label.text()
        # Remove old spinner if present
        for char in self._spinner_chars:
            if current_text.startswith(char + " "):
                current_text = current_text[2:]
                break
        self._progress_label.setText(
            f"{self._spinner_chars[self._spinner_idx]} {current_text}"
        )

    def _on_cancel_analysis(self):
        """Handle cancel button click."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._progress_label.setText("Cancelling...")
            self._cancel_button.setEnabled(False)

    def set_labels(self, labels: "sio.Labels"):
        """Set the labels to analyze.

        Args:
            labels: A sleap_io.Labels object.
        """
        self._labels = labels
        self._detector = None
        self._results = None
        self._selected_flag = None

        # Update UI
        self._score_canvas.set_scores(np.array([]))
        self._breakdown_canvas.set_issue_counts({})
        self._table_model.items = []
        self._update_statistics()
        self._details_label.setText("Click a row in the table to select an instance")

    def _on_run_analysis(self):
        """Run QC analysis on current labels."""
        if self._labels is None:
            QtWidgets.QMessageBox.warning(
                self, "No Labels", "Please load a labels file first."
            )
            return

        n_instances = sum(len(lf.instances) for lf in self._labels)
        if n_instances < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least 2 instances to run QC analysis.",
            )
            return

        # If already running, don't start another
        if self._worker is not None and self._worker.isRunning():
            return

        # Show progress UI
        self._run_button.setEnabled(False)
        self._progress_label.setVisible(True)
        self._progress_label.setText("Starting...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._cancel_button.setVisible(True)
        self._cancel_button.setEnabled(True)

        # Start spinner animation
        self._spinner_idx = 0
        self._spinner_timer.start()

        # Create and start worker thread
        self._worker = QCAnalysisWorker(self._labels)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_progress(self, step_name: str, progress: int, detail: str):
        """Handle progress update from worker."""
        # Format the label: step name with optional detail
        if detail:
            text = f"{step_name} ({detail})"
        else:
            text = step_name
        self._progress_label.setText(text)
        self._progress_bar.setValue(progress)

    def _on_analysis_finished(self, results):
        """Handle successful analysis completion."""
        self._results = results

        # Stop spinner and hide progress UI
        self._spinner_timer.stop()
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)

        # Update all displays
        self._update_all_displays()

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        # Stop spinner and hide progress UI
        self._spinner_timer.stop()
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)

        QtWidgets.QMessageBox.critical(
            self, "Analysis Error", f"Error during QC analysis:\n{error_msg}"
        )

    def _on_threshold_changed(self, value: int):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self._threshold_label.setText(f"{threshold:.2f}")
        self._score_canvas.set_threshold(threshold)

        if self._results is not None:
            self._update_flagged_display()

    def _on_canvas_threshold_changed(self, threshold: float):
        """Handle threshold change from clicking on histogram."""
        # Clamp to slider range
        slider_value = int(threshold * 100)
        slider_value = max(30, min(90, slider_value))
        self._threshold_slider.setValue(slider_value)

    def _update_all_displays(self):
        """Update all display components after analysis."""
        if self._results is None:
            return

        # Get all scores for histogram
        scores = np.array(list(self._results.instance_scores.values()))
        self._score_canvas.set_scores(scores)

        threshold = self._threshold_slider.value() / 100.0
        self._score_canvas.set_threshold(threshold)

        self._update_flagged_display()
        self._update_statistics()

    def _update_flagged_display(self):
        """Update the flagged instances table and breakdown chart."""
        if self._results is None:
            return

        threshold = self._threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)

        # Update table
        self._table_model.items = flagged

        # Update breakdown chart
        issue_counts = {}
        for flag in flagged:
            issue = flag.top_issue.replace("_", " ").title()
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        self._breakdown_canvas.set_issue_counts(issue_counts)

        # Update feature comparison chart
        self._feature_canvas.set_feature_data(
            self._results.feature_contributions,
            self._results.instance_scores,
            threshold,
            self._results.feature_names,
        )

    def _update_statistics(self):
        """Update the statistics panel."""
        if self._labels is None:
            self._stats_label.setText("No labels loaded")
            return

        n_instances = sum(len(lf.instances) for lf in self._labels)
        n_frames = len(self._labels)

        if self._results is None:
            self._stats_label.setText(
                f"<b>Ready to analyze:</b><br/>"
                f"• {n_instances} instances<br/>"
                f"• {n_frames} frames"
            )
            return

        threshold = self._threshold_slider.value() / 100.0
        scores = np.array(list(self._results.instance_scores.values()))
        flagged = self._results.get_flagged(threshold=threshold)
        n_flagged = len(flagged)
        pct_flagged = (n_flagged / n_instances * 100) if n_instances > 0 else 0

        # Score statistics
        mean_score = np.mean(scores) if len(scores) > 0 else 0
        median_score = np.median(scores) if len(scores) > 0 else 0
        max_score = np.max(scores) if len(scores) > 0 else 0

        # Count by confidence
        high_conf = sum(1 for f in flagged if f.confidence == "high")
        med_conf = sum(1 for f in flagged if f.confidence == "medium")

        # Frame-level issues
        frame_issues = self._results.get_frame_issues()
        n_frame_issues = len(frame_issues)

        self._stats_label.setText(
            f"<b>Flagged:</b> {n_flagged} / {n_instances} ({pct_flagged:.1f}%)<br/>"
            f"<b>By confidence:</b> {high_conf} high, {med_conf} medium<br/>"
            f"<b>Frame issues:</b> {n_frame_issues}<br/>"
            f"<b>Scores:</b> mean={mean_score:.2f}, "
            f"median={median_score:.2f}, max={max_score:.2f}"
        )

    def _on_selection_changed(self, selected, deselected):
        """Handle selection change in table."""
        indexes = self._table_view.selectionModel().selectedRows()
        if indexes:
            row = indexes[0].row()
            if row < len(self._table_model.items):
                self._selected_flag = self._table_model.items[row]
                self._update_selected_details()

                # Navigate to the instance
                self.navigate_to_instance.emit(
                    self._selected_flag.video_idx,
                    self._selected_flag.frame_idx,
                    self._selected_flag.instance_idx,
                )
        else:
            self._selected_flag = None
            self._details_label.setText(
                "Click a row in the table to select an instance"
            )

    def _on_row_double_clicked(self, index):
        """Handle double-click on table row."""
        row = index.row()
        if row < len(self._table_model.items):
            flag = self._table_model.items[row]
            self.navigate_to_instance.emit(
                flag.video_idx,
                flag.frame_idx,
                flag.instance_idx,
            )

    def _update_selected_details(self):
        """Update the selected instance details panel."""
        if self._selected_flag is None:
            self._details_label.setText(
                "Click a row in the table to select an instance"
            )
            return

        flag = self._selected_flag

        # Get top contributing features
        contributions = flag.feature_contributions
        top_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        features_text = "<br/>".join(
            f"  • {name.replace('_', ' ')}: {value:.3f}" for name, value in top_features
        )

        self._details_label.setText(
            f"<b>Frame:</b> {flag.frame_idx} | "
            f"<b>Instance:</b> {flag.instance_idx}<br/>"
            f"<b>Score:</b> {flag.score:.3f} ({flag.confidence} confidence)<br/>"
            f"<b>Primary Issue:</b> {flag.top_issue.replace('_', ' ').title()}<br/>"
            f"<b>Top Features:</b><br/>{features_text}"
        )

    @property
    def has_results(self) -> bool:
        """Return True if analysis results are available."""
        return self._results is not None

    @property
    def has_flags(self) -> bool:
        """Return True if there are flagged items to navigate."""
        return len(self._table_model.items) > 0

    def goto_next_flag(self) -> bool:
        """Navigate to the next flagged instance in the table.

        Returns:
            True if navigation occurred, False if no items or at end.
        """
        if not self.has_flags:
            return False

        # Get current selection
        indexes = self._table_view.selectionModel().selectedRows()
        current_row = indexes[0].row() if indexes else -1

        # Move to next row (wrap around)
        next_row = (current_row + 1) % len(self._table_model.items)

        # Select the row (this triggers navigation via _on_selection_changed)
        self._table_view.selectRow(next_row)
        return True

    def goto_prev_flag(self) -> bool:
        """Navigate to the previous flagged instance in the table.

        Returns:
            True if navigation occurred, False if no items.
        """
        if not self.has_flags:
            return False

        # Get current selection
        indexes = self._table_view.selectionModel().selectedRows()
        n_items = len(self._table_model.items)
        current_row = indexes[0].row() if indexes else 0

        # Move to previous row (wrap around)
        prev_row = (current_row - 1) % n_items

        # Select the row (this triggers navigation via _on_selection_changed)
        self._table_view.selectRow(prev_row)
        return True

    def export_results(self):
        """Export QC results to CSV (public method for dialog)."""
        import os

        if self._results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Results", "Please run analysis first."
            )
            return

        # Determine default directory: use last export dir, or labels folder, or CWD
        default_dir = self._last_export_dir
        if default_dir is None and self._labels is not None:
            # Try to get directory from labels provenance
            provenance = getattr(self._labels, "provenance", None)
            if provenance is not None:
                labels_path = getattr(provenance, "filename", None)
                if labels_path:
                    default_dir = os.path.dirname(labels_path)

        default_filename = "qc_results.csv"
        if default_dir:
            default_path = os.path.join(default_dir, default_filename)
        else:
            default_path = default_filename

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Results",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )

        if filepath:
            try:
                df = self._results.to_dataframe()
                df.to_csv(filepath, index=False)
                # Persist the directory for next export
                self._last_export_dir = os.path.dirname(filepath)
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", f"Results exported to:\n{filepath}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Error exporting results:\n{str(e)}"
                )

    def export_to_suggestions(self) -> int:
        """Export flagged frames to the suggestions list.

        Creates SuggestionFrame objects for each unique frame that contains
        flagged instances and adds them to labels.suggestions.

        Returns:
            Number of suggestions added, or -1 if export failed.
        """
        from sleap_io import SuggestionFrame

        if self._results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Results", "Please run analysis first."
            )
            return -1

        if self._labels is None:
            QtWidgets.QMessageBox.warning(self, "No Labels", "No labels file loaded.")
            return -1

        threshold = self._threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)

        if not flagged:
            QtWidgets.QMessageBox.information(
                self,
                "No Flagged Instances",
                "No instances are flagged at the current threshold.",
            )
            return 0

        # Get unique frames (video_idx, frame_idx pairs)
        # Track the highest score for each frame for metadata
        unique_frames = {}
        for flag in flagged:
            key = (flag.video_idx, flag.frame_idx)
            if key not in unique_frames or flag.score > unique_frames[key].score:
                unique_frames[key] = flag

        # Filter out frames that are already in suggestions
        existing_suggestions = set()
        for sugg in self._labels.suggestions:
            video_idx = self._labels.videos.index(sugg.video)
            existing_suggestions.add((video_idx, sugg.frame_idx))

        new_frames = {
            key: flag
            for key, flag in unique_frames.items()
            if key not in existing_suggestions
        }

        if not new_frames:
            QtWidgets.QMessageBox.information(
                self,
                "Already Added",
                f"All {len(unique_frames)} flagged frames are already in suggestions.",
            )
            return 0

        # Create SuggestionFrame objects
        suggestions = []
        for (video_idx, frame_idx), flag in new_frames.items():
            video = self._labels.videos[video_idx]
            suggestion = SuggestionFrame(video=video, frame_idx=frame_idx)
            suggestions.append(suggestion)

        # Add to labels
        self._labels.suggestions.extend(suggestions)

        n_added = len(suggestions)
        n_skipped = len(unique_frames) - n_added

        msg = f"Added {n_added} frame(s) to suggestions."
        if n_skipped > 0:
            msg += f"\n({n_skipped} already in suggestions)"

        QtWidgets.QMessageBox.information(self, "Export Complete", msg)

        return n_added

    def cleanup(self):
        """Clean up resources, stopping any running analysis.

        Should be called before the widget is destroyed.
        """
        # Stop spinner timer
        self._spinner_timer.stop()

        # Cancel and wait for worker thread
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            # Wait up to 2 seconds for thread to finish
            if not self._worker.wait(2000):
                # Thread didn't finish, terminate it
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

    def closeEvent(self, event):
        """Handle widget close event."""
        self.cleanup()
        super().closeEvent(event)

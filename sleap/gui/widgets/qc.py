"""
Widget for visualizing label QC results.

Provides histogram and table views of instance anomaly scores,
with click-to-navigate support for reviewing flagged annotations.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui

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

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setMinimumSize(400, 200)
        self.updateGeometry()

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
        bins = np.linspace(0, 1, 21)  # 20 bins
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
            f"{n_flagged} flagged\n({100*n_flagged/n_total:.1f}%)",
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

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setMinimumSize(400, 150)
        self.updateGeometry()

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

        # Sort by count descending, take top 8
        sorted_issues = sorted(
            self._issue_counts.items(), key=lambda x: x[1], reverse=True
        )[:8]

        labels = [item[0] for item in sorted_issues]
        counts = [item[1] for item in sorted_issues]

        # Horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = self.axes.barh(y_pos, counts, color="#dc3545", alpha=0.7)

        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels(labels, fontsize=9)
        self.axes.invert_yaxis()  # Top to bottom
        self.axes.set_xlabel("Count", fontsize=10)
        self.axes.set_title("Issue Breakdown", fontsize=11)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            self.axes.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9,
            )

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

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

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

        # Progress bar (hidden by default)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # === Threshold control ===
        threshold_group = QtWidgets.QGroupBox("Sensitivity Threshold")
        threshold_layout = QtWidgets.QHBoxLayout(threshold_group)
        threshold_layout.setContentsMargins(8, 8, 8, 8)

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
            "font-weight: bold; background: #f8f9fa; padding: 2px 6px; border-radius: 3px;"
        )
        threshold_layout.addWidget(self._threshold_label)

        layout.addWidget(threshold_group)

        # === Score histogram ===
        self._score_canvas = QCScoreCanvas(width=6, height=2.5)
        layout.addWidget(self._score_canvas)

        # === Issue breakdown ===
        self._breakdown_canvas = QCBreakdownCanvas(width=6, height=2)
        layout.addWidget(self._breakdown_canvas)

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
        self._table_view.setMinimumHeight(150)

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
        details_layout.setContentsMargins(8, 8, 8, 8)

        self._details_label = QtWidgets.QLabel(
            "Click a row in the table to select an instance"
        )
        self._details_label.setWordWrap(True)
        self._details_label.setMinimumHeight(80)
        details_layout.addWidget(self._details_label)

        bottom_layout.addWidget(details_group)

        # Statistics panel
        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 8, 8, 8)

        self._stats_label = QtWidgets.QLabel("No analysis run yet")
        self._stats_label.setWordWrap(True)
        self._stats_label.setMinimumHeight(80)
        stats_layout.addWidget(self._stats_label)

        bottom_layout.addWidget(stats_group)

        layout.addLayout(bottom_layout)

        # === Export button ===
        export_layout = QtWidgets.QHBoxLayout()
        export_layout.addStretch()

        self._export_button = QtWidgets.QPushButton("Export to CSV...")
        self._export_button.setToolTip("Export all QC results to a CSV file")
        self._export_button.setEnabled(False)
        export_layout.addWidget(self._export_button)

        layout.addLayout(export_layout)

    def _connect_signals(self):
        """Connect UI signals."""
        self._run_button.clicked.connect(self._on_run_analysis)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._score_canvas.threshold_changed.connect(self._on_canvas_threshold_changed)
        self._table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._table_view.doubleClicked.connect(self._on_row_double_clicked)
        self._export_button.clicked.connect(self._on_export)

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

        self._run_button.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # Indeterminate

        try:
            # Import here to avoid circular imports
            from sleap.qc import LabelQCDetector

            # Create and fit detector
            self._detector = LabelQCDetector()
            self._detector.fit(self._labels)

            # Score all instances
            self._results = self._detector.score(self._labels)

            # Update all displays
            self._update_all_displays()
            self._export_button.setEnabled(True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Analysis Error", f"Error during QC analysis:\n{str(e)}"
            )

        finally:
            self._run_button.setEnabled(True)
            self._progress_bar.setVisible(False)

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
        top_features = sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        )[:3]
        features_text = "<br/>".join(
            f"  • {name.replace('_', ' ')}: {value:.3f}"
            for name, value in top_features
        )

        self._details_label.setText(
            f"<b>Frame:</b> {flag.frame_idx} | "
            f"<b>Instance:</b> {flag.instance_idx}<br/>"
            f"<b>Score:</b> {flag.score:.3f} ({flag.confidence} confidence)<br/>"
            f"<b>Primary Issue:</b> {flag.top_issue.replace('_', ' ').title()}<br/>"
            f"<b>Top Features:</b><br/>{features_text}"
        )

    def _on_export(self):
        """Export QC results to CSV."""
        if self._results is None:
            return

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Results",
            "qc_results.csv",
            "CSV Files (*.csv);;All Files (*)",
        )

        if filepath:
            try:
                df = self._results.to_dataframe()
                df.to_csv(filepath, index=False)
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", f"Results exported to:\n{filepath}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Error exporting results:\n{str(e)}"
                )

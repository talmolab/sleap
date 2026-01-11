"""Label QC dock widget for flagging annotation errors."""

from typing import Any, Optional

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QMessageBox,
)

from sleap.gui.dataviews import GenericTableModel, GenericTableView
from sleap.gui.state import GuiState


class QCFlagTableModel(GenericTableModel):
    """Table model for QC flagged instances.

    Displays flagged annotation instances with their scores and top issues.
    """

    properties = ("video", "frame", "instance", "score", "confidence", "top_issue")
    show_row_numbers = True

    def __init__(
        self,
        items: Optional[list] = None,
        context=None,
    ):
        super().__init__(items=items, context=context)

    def item_to_data(self, obj, item):
        """Convert QCFlag item to display data."""
        return {
            "video": item.instance_key[0],
            "frame": item.instance_key[1],
            "instance": item.instance_key[2],
            "score": f"{item.score:.3f}",
            "confidence": item.confidence.title(),
            "top_issue": item.top_issue.replace("_", " ").title(),
        }

    def get_item_color(self, item: Any, key: str):
        """Return color based on confidence level."""
        if key == "confidence":
            conf = item.confidence
            if conf == "high":
                return QtGui.QBrush(QtGui.QColor(220, 53, 69))  # Red
            elif conf == "medium":
                return QtGui.QBrush(QtGui.QColor(255, 193, 7))  # Yellow
            else:
                return QtGui.QBrush(QtGui.QColor(108, 117, 125))  # Gray
        elif key == "score":
            score = item.score
            if score >= 0.8:
                return QtGui.QBrush(QtGui.QColor(220, 53, 69))  # Red
            elif score >= 0.6:
                return QtGui.QBrush(QtGui.QColor(255, 193, 7))  # Yellow
        return None


class QCWidget(QWidget):
    """Widget for label quality control analysis.

    Provides controls for running QC analysis and viewing flagged instances.

    Signals:
        navigate_to_frame: Emitted when user wants to navigate to a flagged instance.
            Args: (video_idx, frame_idx)
        analysis_started: Emitted when QC analysis begins.
        analysis_finished: Emitted when QC analysis completes.
    """

    navigate_to_frame = Signal(int, int)
    analysis_started = Signal()
    analysis_finished = Signal()

    def __init__(
        self,
        state: GuiState = None,
        labels=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.state = state or GuiState()
        self.labels = labels
        self._detector = None
        self._results = None

        self._setup_ui()
        self._connect_signals()
        self._update_summary()  # Initialize summary with current labels state

    def _setup_ui(self):
        """Create the widget UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # === Controls Group ===
        controls_group = QGroupBox("QC Analysis")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(4)

        # Run button row
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run QC Analysis")
        self.run_btn.setToolTip(
            "Analyze all labeled instances for potential annotation errors"
        )
        run_row.addWidget(self.run_btn)
        controls_layout.addLayout(run_row)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Threshold slider row
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Sensitivity:"))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(70)  # Default: 0.7 threshold
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setToolTip(
            "Lower values = more instances flagged (higher sensitivity)"
        )
        threshold_row.addWidget(self.threshold_slider, stretch=1)

        self.threshold_label = QLabel("0.70")
        self.threshold_label.setMinimumWidth(35)
        threshold_row.addWidget(self.threshold_label)

        controls_layout.addLayout(threshold_row)

        # Sensitivity presets
        presets_row = QHBoxLayout()
        self.low_btn = QPushButton("Low")
        self.low_btn.setToolTip("Flag ~5% of instances (threshold=0.8)")
        self.low_btn.clicked.connect(lambda: self.threshold_slider.setValue(80))

        self.medium_btn = QPushButton("Medium")
        self.medium_btn.setToolTip("Flag ~10% of instances (threshold=0.7)")
        self.medium_btn.clicked.connect(lambda: self.threshold_slider.setValue(70))

        self.high_btn = QPushButton("High")
        self.high_btn.setToolTip("Flag ~20% of instances (threshold=0.5)")
        self.high_btn.clicked.connect(lambda: self.threshold_slider.setValue(50))

        presets_row.addWidget(self.low_btn)
        presets_row.addWidget(self.medium_btn)
        presets_row.addWidget(self.high_btn)
        controls_layout.addLayout(presets_row)

        layout.addWidget(controls_group)

        # === Summary Group ===
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_label = QLabel("No analysis run yet")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_group)

        # === Results Table ===
        results_group = QGroupBox("Flagged Instances")
        results_layout = QVBoxLayout(results_group)

        # Table for flagged instances
        self.table_model = QCFlagTableModel(items=[])
        self.table_view = GenericTableView(
            state=self.state,
            row_name="qc_flag",
            is_activatable=True,
            is_sortable=True,
            model=self.table_model,
        )
        self.table_view.setMinimumHeight(150)
        results_layout.addWidget(self.table_view)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self.goto_btn = QPushButton("Go to Frame")
        self.goto_btn.setToolTip("Navigate to the selected flagged instance")
        self.goto_btn.setEnabled(False)
        nav_row.addWidget(self.goto_btn)

        self.export_btn = QPushButton("Export...")
        self.export_btn.setToolTip("Export QC results to CSV")
        self.export_btn.setEnabled(False)
        nav_row.addWidget(self.export_btn)

        results_layout.addLayout(nav_row)

        layout.addWidget(results_group, stretch=1)

    def _connect_signals(self):
        """Connect widget signals to handlers."""
        self.run_btn.clicked.connect(self._on_run_analysis)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.goto_btn.clicked.connect(self._on_goto_frame)
        self.export_btn.clicked.connect(self._on_export)
        self.table_view.doubleClicked.connect(self._on_row_double_clicked)

        # Enable goto button when row selected
        self.state.connect("selected_qc_flag", self._on_selection_changed)

    def set_labels(self, labels):
        """Set the labels to analyze."""
        self.labels = labels
        self._detector = None
        self._results = None
        self._update_summary()

    def _on_threshold_changed(self, value: int):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")

        # Update flagged list and summary if we have results
        if self._results is not None:
            self._update_flagged_list()
            self._update_summary()

    def _on_run_analysis(self):
        """Run QC analysis on current labels."""
        if self.labels is None:
            QMessageBox.warning(
                self, "No Labels", "Please load a labels file first."
            )
            return

        n_instances = sum(len(lf.instances) for lf in self.labels)
        if n_instances < 2:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least 2 instances to run QC analysis.",
            )
            return

        self.analysis_started.emit()
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        try:
            # Import here to avoid circular imports
            from sleap.qc import LabelQCDetector

            # Create and fit detector
            self._detector = LabelQCDetector()
            self._detector.fit(self.labels)

            # Score all instances
            self._results = self._detector.score(self.labels)

            # Update display
            self._update_flagged_list()
            self._update_summary()

            self.export_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(
                self, "Analysis Error", f"Error during QC analysis:\n{str(e)}"
            )

        finally:
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.analysis_finished.emit()

    def _update_flagged_list(self):
        """Update the flagged instances table."""
        if self._results is None:
            self.table_model.items = []
            return

        threshold = self.threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)

        # Sort by score descending
        flagged_sorted = sorted(flagged, key=lambda x: x.score, reverse=True)
        self.table_model.items = flagged_sorted

    def _update_summary(self):
        """Update the summary label."""
        if self.labels is None:
            self.summary_label.setText("No labels loaded")
            return

        n_instances = sum(len(lf.instances) for lf in self.labels)
        n_frames = len(self.labels)

        if self._results is None:
            self.summary_label.setText(
                f"Ready to analyze:\n"
                f"• {n_instances} instances\n"
                f"• {n_frames} frames"
            )
            return

        threshold = self.threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)
        n_flagged = len(flagged)
        pct_flagged = (n_flagged / n_instances * 100) if n_instances > 0 else 0

        # Count by confidence
        high_conf = sum(1 for f in flagged if f.confidence == "high")
        med_conf = sum(1 for f in flagged if f.confidence == "medium")
        low_conf = sum(1 for f in flagged if f.confidence == "low")

        # Frame-level issues
        frame_issues = self._results.get_frame_issues()
        n_frame_issues = len(frame_issues)

        self.summary_label.setText(
            f"Analysis complete:\n"
            f"• {n_flagged} flagged ({pct_flagged:.1f}%)\n"
            f"  - High confidence: {high_conf}\n"
            f"  - Medium confidence: {med_conf}\n"
            f"  - Low confidence: {low_conf}\n"
            f"• {n_frame_issues} frame-level issues"
        )

    def _on_selection_changed(self, item):
        """Handle selection change in table."""
        self.goto_btn.setEnabled(item is not None)

    def _on_goto_frame(self):
        """Navigate to the selected flagged instance."""
        flag = self.state.get("selected_qc_flag")
        if flag is not None:
            video_idx, frame_idx, _ = flag.instance_key
            self.navigate_to_frame.emit(video_idx, frame_idx)

    def _on_row_double_clicked(self, index):
        """Handle double-click on table row."""
        flag = self.table_model.original_items[index.row()]
        video_idx, frame_idx, _ = flag.instance_key
        self.navigate_to_frame.emit(video_idx, frame_idx)

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
                QMessageBox.information(
                    self, "Export Complete", f"Results exported to:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Error exporting results:\n{str(e)}"
                )


class QCDock(QDockWidget):
    """Dock widget for label quality control.

    Provides a dockable panel for running QC analysis on annotations.
    """

    def __init__(
        self,
        main_window=None,
        tab_with=None,
    ):
        super().__init__("Label QC")
        self.main_window = main_window
        self.setObjectName("LabelQCDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create the QC widget
        self.qc_widget = QCWidget(
            state=main_window.state if main_window else None,
            labels=main_window.labels if main_window else None,
            parent=self,
        )
        self.setWidget(self.qc_widget)

        # Add to main window if provided
        if main_window is not None:
            self._add_to_window(main_window, tab_with)

        # Connect navigation signal
        self.qc_widget.navigate_to_frame.connect(self._navigate_to_frame)

    def _add_to_window(self, main_window, tab_with=None):
        """Add dock to main window."""
        main_window.addDockWidget(Qt.RightDockWidgetArea, self)
        main_window.viewMenu.addAction(self.toggleViewAction())

        if tab_with is not None:
            main_window.tabifyDockWidget(tab_with, self)

    def _navigate_to_frame(self, video_idx: int, frame_idx: int):
        """Navigate main window to specific video/frame."""
        if self.main_window is None:
            return

        # Set video if needed
        videos = self.main_window.labels.videos
        if video_idx < len(videos):
            self.main_window.state["video"] = videos[video_idx]

        # Set frame
        self.main_window.state["frame_idx"] = frame_idx

    def update_labels(self, labels):
        """Update the labels being analyzed."""
        self.qc_widget.set_labels(labels)

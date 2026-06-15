"""Tests for QC widget components."""

from unittest.mock import MagicMock, PropertyMock, patch

from qtpy import QtCore, QtWidgets

from sleap.gui.widgets.qc import (
    DETECTOR_HELP,
    CheckableFilterMenu,
    CollapsibleGroupBox,
    QCAnalysisWorker,
    QCChainTraceDialog,
    QCFlagTableModel,
    QCSkeletonTraceCanvas,
    QCWidget,
    _friendly_issue,
)
from sleap.qc.config import QCConfig


def _fake_results(flags):
    """Build a QCResults-like stub whose get_flagged returns ``flags``.

    Mirrors only the attributes the QCWidget display path touches, so the
    flagged-list filter/reviewed tests can drive ``_update_flagged_display``
    without running real detectors.
    """
    results = MagicMock()
    results.get_flagged.return_value = list(flags)
    results.feature_contributions = {}
    results.instance_scores = {}
    results.feature_names = []
    return results


class MockQCFlag:
    """Mock QCFlag for testing."""

    def __init__(
        self, video_idx, frame_idx, instance_idx, score, confidence, top_issue
    ):
        self.instance_key = (video_idx, frame_idx, instance_idx)
        self.video_idx = video_idx
        self.frame_idx = frame_idx
        self.instance_idx = instance_idx
        self.score = score
        self.confidence = confidence
        self.top_issue = top_issue
        self.feature_contributions = {"edge_zscore": 0.5, "visibility": 0.3}


class TestQCFlagTableModel:
    """Tests for QCFlagTableModel."""

    def test_columns(self):
        """Test table has expected columns."""
        model = QCFlagTableModel()
        assert "Frame" in model.COLUMNS
        assert "Instance" in model.COLUMNS
        assert "Score" in model.COLUMNS
        assert "Issue" in model.COLUMNS
        assert "Confidence" in model.COLUMNS
        # Reviewed checkmark column (issue #2769, item 6), appended last so the
        # existing Frame..Issue column indices are unchanged.
        assert "Reviewed" in model.COLUMNS
        assert model.COLUMNS[model.REVIEWED_COL] == "Reviewed"

    def test_empty_model(self):
        """Test model can be created empty."""
        model = QCFlagTableModel()
        assert model.rowCount() == 0
        # 5 data columns + the trailing Reviewed checkmark column.
        assert model.columnCount() == 6

    def test_data_display_role(self):
        """Test data retrieval with DisplayRole."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(
                video_idx=0,
                frame_idx=10,
                instance_idx=0,
                score=0.85,
                confidence="high",
                top_issue="edge_zscore",
            )
        ]
        model.items = flags

        # Frame column (0)
        frame_data = model.data(model.index(0, 0), QtCore.Qt.DisplayRole)
        assert frame_data == "10"

        # Instance column (1)
        instance_data = model.data(model.index(0, 1), QtCore.Qt.DisplayRole)
        assert instance_data == "0"

        # Score column (2)
        score_data = model.data(model.index(0, 2), QtCore.Qt.DisplayRole)
        assert score_data == "0.850"

        # Confidence column (3)
        conf_data = model.data(model.index(0, 3), QtCore.Qt.DisplayRole)
        assert conf_data == "High"

        # Issue column (4)
        issue_data = model.data(model.index(0, 4), QtCore.Qt.DisplayRole)
        assert issue_data == "Edge Zscore"

    def test_items_setter(self):
        """Test setting items on model."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
        ]
        model.items = flags
        assert model.rowCount() == 2

    def test_header_data(self):
        """Test header data returns column names."""
        model = QCFlagTableModel()
        assert model.headerData(0, QtCore.Qt.Horizontal) == "Frame"
        assert model.headerData(1, QtCore.Qt.Horizontal) == "Instance"
        assert model.headerData(2, QtCore.Qt.Horizontal) == "Score"

    def test_sort_by_frame(self):
        """Test sorting by frame column."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(0, 10, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 5, 1, 0.7, "medium", "visibility"),
            MockQCFlag(0, 15, 0, 0.6, "low", "scale"),
        ]
        model.items = flags

        # Sort ascending by frame
        model.sort(0, QtCore.Qt.AscendingOrder)
        assert model._items[0].frame_idx == 5
        assert model._items[1].frame_idx == 10
        assert model._items[2].frame_idx == 15

        # Sort descending by frame
        model.sort(0, QtCore.Qt.DescendingOrder)
        assert model._items[0].frame_idx == 15
        assert model._items[1].frame_idx == 10
        assert model._items[2].frame_idx == 5

    def test_sort_by_score(self):
        """Test sorting by score column."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(0, 10, 0, 0.7, "medium", "edge_error"),
            MockQCFlag(0, 5, 1, 0.9, "high", "visibility"),
            MockQCFlag(0, 15, 0, 0.6, "low", "scale"),
        ]
        model.items = flags

        # Sort ascending by score (lowest first)
        model.sort(2, QtCore.Qt.AscendingOrder)
        assert model._items[0].score == 0.6
        assert model._items[1].score == 0.7
        assert model._items[2].score == 0.9

        # Sort descending by score (highest first)
        model.sort(2, QtCore.Qt.DescendingOrder)
        assert model._items[0].score == 0.9
        assert model._items[1].score == 0.7
        assert model._items[2].score == 0.6

    def test_sort_by_confidence(self):
        """Test sorting by confidence column."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(0, 10, 0, 0.7, "medium", "edge_error"),
            MockQCFlag(0, 5, 1, 0.9, "high", "visibility"),
            MockQCFlag(0, 15, 0, 0.6, "low", "scale"),
        ]
        model.items = flags

        # Sort ascending by confidence (low first)
        model.sort(3, QtCore.Qt.AscendingOrder)
        assert model._items[0].confidence == "low"
        assert model._items[1].confidence == "medium"
        assert model._items[2].confidence == "high"

        # Sort descending by confidence (high first)
        model.sort(3, QtCore.Qt.DescendingOrder)
        assert model._items[0].confidence == "high"
        assert model._items[1].confidence == "medium"
        assert model._items[2].confidence == "low"


class TestQCWidget:
    """Tests for QCWidget."""

    def test_widget_creation(self, qtbot):
        """Test widget can be created."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget is not None
        assert widget._labels is None

    def test_widget_has_controls(self, qtbot):
        """Test widget has expected controls."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._run_button is not None
        assert widget._threshold_slider is not None
        assert widget._table_view is not None
        assert widget._score_canvas is not None
        assert widget._breakdown_canvas is not None
        assert widget._viz_tabs is not None  # Tabbed visualization

    def test_threshold_slider_default(self, qtbot):
        """Test default threshold is 0.7."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._threshold_slider.value() == 70

    def test_threshold_slider_range(self, qtbot):
        """Test threshold slider has expected range."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._threshold_slider.minimum() == 30
        assert widget._threshold_slider.maximum() == 90

    def test_threshold_label_updates(self, qtbot):
        """Test threshold label updates with slider."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._threshold_slider.setValue(50)
        assert "0.50" in widget._threshold_label.text()

    def test_set_labels(self, qtbot):
        """Test setting labels on widget."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        widget.set_labels(mock_labels)
        assert widget._labels is mock_labels

    def test_run_analysis_no_labels(self, qtbot):
        """Test run analysis shows warning with no labels."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Should show warning dialog
        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox") as mock_msgbox:
            widget._on_run_analysis()
            mock_msgbox.warning.assert_called_once()

    def test_stats_no_labels(self, qtbot):
        """Test stats shows 'No labels loaded' when no labels provided."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._labels = None
        widget._update_statistics()
        assert "No labels loaded" in widget._stats_label.text()

    def test_stats_with_labels_before_analysis(self, qtbot):
        """Test stats shows 'Ready to analyze' when labels loaded but not analyzed."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)

        # Create mock labeled frames with mock instances. Stats count only
        # user-labeled instances, so mock `user_instances` (not `instances`).
        mock_lf1 = MagicMock()
        mock_lf1.user_instances = [MagicMock(), MagicMock()]  # 2 user instances
        mock_lf2 = MagicMock()
        mock_lf2.user_instances = [MagicMock()]  # 1 user instance
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf1, mock_lf2]))

        widget.set_labels(mock_labels)

        # Should show "Ready to analyze: 3 instances, 10 frames"
        assert "Ready to analyze" in widget._stats_label.text()
        assert "3 instances" in widget._stats_label.text()

    def test_navigate_signal_emitted(self, qtbot):
        """Test navigate_to_instance signal is emitted on selection."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Add some flags to the table
        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
        ]
        widget._table_model.items = flags

        # Track signal emission
        received_signals = []

        def on_navigate(video_idx, frame_idx, instance_idx):
            received_signals.append((video_idx, frame_idx, instance_idx))

        widget.navigate_to_instance.connect(on_navigate)

        # Select first row
        widget._table_view.selectRow(0)
        qtbot.wait(50)  # Allow signal to propagate

        assert len(received_signals) == 1
        assert received_signals[0] == (0, 5, 0)

    def test_has_results_property(self, qtbot):
        """Test has_results property is False before analysis."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget.has_results


class TestQCDialog:
    """Tests for QCDialog."""

    def test_dialog_creation(self, qtbot):
        """Test dialog can be created."""
        from sleap.gui.dialogs.qc import QCDialog

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dialog = QCDialog(labels=mock_labels)
        qtbot.addWidget(dialog)
        assert dialog is not None

    def test_dialog_has_widget(self, qtbot):
        """Test dialog contains QCWidget."""
        from sleap.gui.dialogs.qc import QCDialog

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dialog = QCDialog(labels=mock_labels)
        qtbot.addWidget(dialog)
        assert dialog._widget is not None
        assert isinstance(dialog._widget, QCWidget)

    def test_dialog_navigate_callback(self, qtbot):
        """Test dialog navigation callback is called."""
        from sleap.gui.dialogs.qc import QCDialog

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        callback_calls = []

        def callback(video_idx, frame_idx, instance_idx):
            callback_calls.append((video_idx, frame_idx, instance_idx))

        dialog = QCDialog(labels=mock_labels, navigate_callback=callback)
        qtbot.addWidget(dialog)

        # Emit navigate signal from widget
        dialog._widget.navigate_to_instance.emit(0, 42, 1)
        qtbot.wait(50)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (0, 42, 1)

    def test_dialog_is_non_modal(self, qtbot):
        """Test dialog is non-modal."""
        from sleap.gui.dialogs.qc import QCDialog

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dialog = QCDialog(labels=mock_labels)
        qtbot.addWidget(dialog)
        assert not dialog.isModal()


class TestQCDockWidget:
    """Tests for QCDockWidget docking functionality."""

    def test_dock_widget_is_dockable(self, qtbot):
        """Test that QCDockWidget is a QDockWidget."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        assert isinstance(dock, QDockWidget)

    def test_dock_widget_starts_docked(self, qtbot):
        """Test that dock widget starts in docked (not floating) mode."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QMainWindow
        from qtpy.QtCore import Qt

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        # Need a main window for docking to work
        main_window = QMainWindow()
        qtbot.addWidget(main_window)

        dock = QCDockWidget(labels=mock_labels, parent=main_window)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        qtbot.addWidget(dock)

        # Now starts docked by default (not floating) so Qt state saving works
        assert not dock.isFloating()

    def test_dock_widget_allowed_areas(self, qtbot):
        """Test that dock widget can be docked to left or right."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtCore import Qt

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        allowed_areas = dock.allowedAreas()
        assert allowed_areas & Qt.LeftDockWidgetArea
        assert allowed_areas & Qt.RightDockWidgetArea

    def test_dock_widget_has_suggestions_button(self, qtbot):
        """Test that dock widget has Add to Suggestions button."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        assert dock._suggestions_button is not None
        assert "Suggestions" in dock._suggestions_button.text()

    def test_dock_widget_has_dock_button(self, qtbot):
        """Test that dock widget has dock/undock toggle button."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        assert dock._dock_button is not None
        # Initially docked, so button should say "Undock"
        assert "Undock" in dock._dock_button.text()

    def test_dock_button_toggles_state(self, qtbot):
        """Test that dock button toggles between docked and floating."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QMainWindow
        from qtpy.QtCore import Qt

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        # Need a main window for docking to work
        main_window = QMainWindow()
        qtbot.addWidget(main_window)

        dock = QCDockWidget(labels=mock_labels, parent=main_window)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        qtbot.addWidget(dock)

        # Initially docked (not floating)
        assert not dock.isFloating()
        assert "Undock" in dock._dock_button.text()

        # Click to undock (float)
        dock._dock_button.click()
        qtbot.wait(50)
        assert dock.isFloating()
        assert "Dock" in dock._dock_button.text()

        # Click to dock again
        dock._dock_button.click()
        qtbot.wait(50)
        assert not dock.isFloating()
        assert "Undock" in dock._dock_button.text()

    def test_update_labels_noop_when_unchanged(self, qtbot):
        """update_labels should not reset the widget when given the same object."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)

        # Passing the same object should not re-run set_labels (which would
        # clear any existing analysis results).
        with patch.object(dock._widget, "set_labels") as mock_set:
            dock.update_labels(mock_labels)
            mock_set.assert_not_called()

        # A new object should propagate to the inner widget.
        other_labels = MagicMock()
        other_labels.__len__ = MagicMock(return_value=5)
        other_labels.__iter__ = MagicMock(return_value=iter([]))
        with patch.object(dock._widget, "set_labels") as mock_set:
            dock.update_labels(other_labels)
            mock_set.assert_called_once_with(other_labels)

    def test_qc_dock_repoints_to_newly_loaded_project(self, qtbot, min_labels):
        """The dock re-points to a project loaded while it holds stale labels.

        Regression test: the dock is created once and persists across project
        loads. Previously it only refreshed via the Analyze menu or a visibility
        change, so opening a project while the dock was already visible left it
        pointing at a stale/empty Labels object, and "Run Analysis" reported
        "Need at least 2 instances" until something re-synced it.

        ``MainWindow.on_data_update`` now calls ``update_labels`` on the
        ``UpdateTopic.project`` path; this exercises that call directly (without
        a full ``MainWindow``, whose video-backed teardown is flaky offscreen).
        """
        from sleap.gui.dialogs.qc import QCDockWidget
        from sleap_io import Labels

        # Start stale: an empty project, as the dock holds before a load.
        stale = Labels()
        dock = QCDockWidget(labels=stale)
        qtbot.addWidget(dock)
        assert sum(len(lf.user_instances) for lf in dock._widget._labels) == 0

        # Mimic what on_data_update does when a project is loaded.
        dock.update_labels(min_labels)

        # The dock now tracks the loaded project, so Run Analysis would proceed.
        assert dock._widget._labels is min_labels
        n_user = sum(len(lf.user_instances) for lf in dock._widget._labels)
        assert n_user == sum(len(lf.user_instances) for lf in min_labels)
        assert n_user >= 2


class TestExportToSuggestions:
    """Tests for export_to_suggestions functionality."""

    def test_export_no_results(self, qtbot):
        """Test export fails gracefully when no results available."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox") as mock_msgbox:
            result = widget.export_to_suggestions()
            mock_msgbox.warning.assert_called_once()
            assert result == -1

    def test_export_no_labels(self, qtbot):
        """Test export fails gracefully when no labels loaded."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Set up mock results but no labels
        widget._results = MagicMock()
        widget._labels = None

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox") as mock_msgbox:
            result = widget.export_to_suggestions()
            mock_msgbox.warning.assert_called_once()
            assert result == -1

    def test_export_no_flagged_instances(self, qtbot):
        """Test export handles no flagged instances."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Set up mock results with no flagged instances
        mock_results = MagicMock()
        mock_results.get_flagged.return_value = []
        widget._results = mock_results
        widget._labels = MagicMock()

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox") as mock_msgbox:
            result = widget.export_to_suggestions()
            mock_msgbox.information.assert_called_once()
            assert result == 0

    def test_export_creates_suggestions(self, qtbot):
        """Test export creates SuggestionFrame objects for flagged frames."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Create mock labels with videos
        mock_video = MagicMock()
        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.suggestions = []

        # Create mock results with flagged instances
        mock_flags = [
            MockQCFlag(0, 10, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.85, "high", "visibility"),  # Same frame
            MockQCFlag(0, 20, 0, 0.75, "medium", "edge_error"),  # Different frame
        ]
        mock_results = MagicMock()
        mock_results.get_flagged.return_value = mock_flags
        widget._results = mock_results
        widget._labels = mock_labels
        widget._threshold_slider.setValue(70)

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox"):
            result = widget.export_to_suggestions()

        # Should add 2 unique frames (10 and 20)
        assert result == 2
        assert len(mock_labels.suggestions) == 2

    def test_export_skips_existing_suggestions(self, qtbot):
        """Test export doesn't duplicate existing suggestions."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Create mock labels with one existing suggestion
        mock_video = MagicMock()
        existing_suggestion = MagicMock()
        existing_suggestion.video = mock_video
        existing_suggestion.frame_idx = 10

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.suggestions = [existing_suggestion]

        # Create mock results with flagged instances
        mock_flags = [
            MockQCFlag(0, 10, 0, 0.9, "high", "edge_error"),  # Already in suggestions
            MockQCFlag(0, 20, 0, 0.75, "medium", "edge_error"),  # New frame
        ]
        mock_results = MagicMock()
        mock_results.get_flagged.return_value = mock_flags
        widget._results = mock_results
        widget._labels = mock_labels
        widget._threshold_slider.setValue(70)

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox"):
            result = widget.export_to_suggestions()

        # Should only add 1 new frame (frame 20)
        assert result == 1
        assert len(mock_labels.suggestions) == 2  # 1 existing + 1 new


class TestQCNavigation:
    """Tests for QC flag navigation functionality."""

    def test_has_flags_property_no_items(self, qtbot):
        """Test has_flags is False when no items in table."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget.has_flags

    def test_has_flags_property_with_items(self, qtbot):
        """Test has_flags is True when items in table."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        flags = [MockQCFlag(0, 5, 0, 0.9, "high", "edge_error")]
        widget._table_model.items = flags
        assert widget.has_flags

    def test_goto_next_flag_no_items(self, qtbot):
        """Test goto_next_flag returns False with no items."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget.goto_next_flag()

    def test_goto_next_flag_advances_selection(self, qtbot):
        """Test goto_next_flag advances to next row."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
            MockQCFlag(0, 15, 0, 0.6, "medium", "edge_error"),
        ]
        widget._table_model.items = flags

        # No selection initially, should start at row 0
        assert widget.goto_next_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert len(indexes) == 1
        assert indexes[0].row() == 0

        # Move to row 1
        assert widget.goto_next_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 1

        # Move to row 2
        assert widget.goto_next_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 2

        # Wrap around to row 0
        assert widget.goto_next_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 0

    def test_goto_prev_flag_no_items(self, qtbot):
        """Test goto_prev_flag returns False with no items."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget.goto_prev_flag()

    def test_goto_prev_flag_goes_backward(self, qtbot):
        """Test goto_prev_flag goes to previous row."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
            MockQCFlag(0, 15, 0, 0.6, "medium", "edge_error"),
        ]
        widget._table_model.items = flags

        # Start at row 1
        widget._table_view.selectRow(1)
        qtbot.wait(10)

        # Move to row 0
        assert widget.goto_prev_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 0

        # Wrap to last row
        assert widget.goto_prev_flag()
        indexes = widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 2


class TestQCDockNavigation:
    """Tests for QC dock widget navigation precedence."""

    def test_dock_has_flags_property(self, qtbot):
        """Test dock widget exposes has_flags property."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)

        # Initially no flags
        assert not dock.has_flags

        # Add flags
        flags = [MockQCFlag(0, 5, 0, 0.9, "high", "edge_error")]
        dock._widget._table_model.items = flags
        assert dock.has_flags

    def test_dock_goto_methods(self, qtbot):
        """Test dock widget exposes navigation methods."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)

        # Add flags
        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
        ]
        dock._widget._table_model.items = flags

        # Test goto_next_flag
        assert dock.goto_next_flag()
        indexes = dock._widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 0

        # Test goto_prev_flag (wraps to end)
        assert dock.goto_prev_flag()
        indexes = dock._widget._table_view.selectionModel().selectedRows()
        assert indexes[0].row() == 1

    def test_is_active_for_navigation_not_visible(self, qtbot):
        """Test is_active_for_navigation is False when dock is not visible."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)

        # Add flags but keep dock hidden
        flags = [MockQCFlag(0, 5, 0, 0.9, "high", "edge_error")]
        dock._widget._table_model.items = flags
        dock.hide()

        assert not dock.is_active_for_navigation

    def test_is_active_for_navigation_no_flags(self, qtbot):
        """Test is_active_for_navigation is False when no flags."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        dock.show()

        # No flags, so should not be active
        assert not dock.is_active_for_navigation

    def test_is_active_for_navigation_floating(self, qtbot):
        """Test is_active_for_navigation is True when floating with flags."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QMainWindow
        from qtpy.QtCore import Qt

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        main_window = QMainWindow()
        qtbot.addWidget(main_window)

        dock = QCDockWidget(labels=mock_labels, parent=main_window)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        qtbot.addWidget(dock)

        # Add flags and float the dock
        flags = [MockQCFlag(0, 5, 0, 0.9, "high", "edge_error")]
        dock._widget._table_model.items = flags
        dock.setFloating(True)
        dock.show()
        main_window.show()
        qtbot.wait(50)

        assert dock.is_active_for_navigation

    def test_visibility_changed_updates_labels(self, qtbot):
        """Test that making dock visible syncs labels from parent."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QMainWindow
        from qtpy.QtCore import Qt

        # Create mock labels
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        # Create main window with labels attribute
        main_window = QMainWindow()
        main_window.labels = mock_labels
        qtbot.addWidget(main_window)

        # Create dock WITHOUT labels (simulating init-time creation)
        dock = QCDockWidget(labels=None, parent=main_window)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        qtbot.addWidget(dock)

        # Initially dock has no labels
        assert dock._labels is None
        assert dock._widget._labels is None

        # Simulate visibility change (like when View menu toggle activates dock)
        dock._on_visibility_changed(True)
        qtbot.wait(10)

        # Now dock should have labels from parent
        assert dock._labels is mock_labels
        assert dock._widget._labels is mock_labels

    def test_fit_selection_checkbox_exists(self, qtbot):
        """Test that dock widget has Fit to Selection checkbox."""
        from sleap.gui.dialogs.qc import QCDockWidget

        dock = QCDockWidget()
        qtbot.addWidget(dock)

        assert hasattr(dock, "_fit_selection_checkbox")
        assert dock._fit_selection_checkbox.text() == "Fit to Selection"

    def test_fit_selection_checkbox_syncs_with_parent_state(self, qtbot):
        """Test checkbox syncs with parent's fit_selection state."""
        from sleap.gui.dialogs.qc import QCDockWidget
        from qtpy.QtWidgets import QMainWindow
        from qtpy.QtCore import Qt

        # Create main window with mock state
        main_window = QMainWindow()
        main_window.state = MagicMock()
        main_window.state.get = MagicMock(return_value=True)
        main_window.state.connect = MagicMock()
        qtbot.addWidget(main_window)

        dock = QCDockWidget(parent=main_window)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        qtbot.addWidget(dock)

        # Sync checkbox - should read True from state
        dock._sync_fit_selection_checkbox()
        assert dock._fit_selection_checkbox.isChecked()

        # Sync with False
        main_window.state.get = MagicMock(return_value=False)
        dock._sync_fit_selection_checkbox()
        assert not dock._fit_selection_checkbox.isChecked()


class TestQCDetectorSettings:
    """Tests for the per-detector settings controls and config building."""

    def test_detector_controls_exist_with_defaults(self, qtbot):
        """Detector controls exist with documented default states."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Group box header (checkable). Advanced panel starts COLLAPSED so the
        # first-time view stays clean (issue #2769, item 4).
        assert widget._detector_settings_group is not None
        assert widget._detector_settings_group.isCheckable()
        assert not widget._detector_settings_group.isChecked()

        # Reliable detectors default-ON.
        assert widget._cb_flip.isChecked()
        assert widget._cb_chimera.isChecked()
        assert widget._cb_duplicate.isChecked()

        # Experimental detectors default-OFF.
        assert not widget._cb_chain.isChecked()
        assert not widget._cb_missing.isChecked()

        # Threshold spinbox defaults.
        assert widget._sb_flip_thr.value() == 0.5
        assert widget._sb_dup_thr.value() == 0.5
        assert widget._sb_chain_angle.value() == 60
        assert widget._sb_order_thr.value() == 0.3
        assert widget._sb_missing_thr.value() == 0.9

        # Ordered-chains edit starts empty.
        assert widget._ordered_chains_edit.toPlainText() == ""

    def test_chimera_has_no_threshold_widget(self, qtbot):
        """Chimera detector exposes no tunable threshold spinbox."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not hasattr(widget, "_sb_chimera_thr")

    def test_build_qc_config_defaults(self, qtbot):
        """_build_qc_config reflects the default control states."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        config = widget._build_qc_config()
        assert isinstance(config, QCConfig)

        # Toggles map to the default checkbox states.
        assert config.use_chirality is True
        assert config.use_split_detection is True
        assert config.use_duplicate_score is True
        assert config.use_chain_ordering is False
        assert config.use_missing_node_check is False

        # Thresholds map to the default spinbox values.
        assert config.chirality_flip_threshold == 0.5
        assert config.duplicate_score_threshold == 0.5
        assert config.chain_turn_angle_deg == 60.0
        assert config.order_inversion_threshold == 0.3
        assert config.missing_node_prob_threshold == 0.9

        # No chains entered by default.
        assert config.ordered_chains == []

    def test_build_qc_config_reflects_toggles(self, qtbot):
        """Toggling checkboxes is reflected in the built config."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._cb_flip.setChecked(False)
        widget._cb_chimera.setChecked(False)
        widget._cb_duplicate.setChecked(False)
        widget._cb_missing.setChecked(True)

        config = widget._build_qc_config()
        assert config.use_chirality is False
        assert config.use_split_detection is False
        assert config.use_duplicate_score is False
        assert config.use_missing_node_check is True

    def test_build_qc_config_reflects_thresholds(self, qtbot):
        """Changing spinboxes is reflected in the built config."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._sb_flip_thr.setValue(0.75)
        widget._sb_dup_thr.setValue(0.6)
        widget._sb_chain_angle.setValue(90)
        widget._sb_order_thr.setValue(0.45)
        widget._sb_missing_thr.setValue(0.8)

        config = widget._build_qc_config()
        assert config.chirality_flip_threshold == 0.75
        assert config.duplicate_score_threshold == 0.6
        assert config.chain_turn_angle_deg == 90.0
        assert config.order_inversion_threshold == 0.45
        assert config.missing_node_prob_threshold == 0.8

    def test_build_qc_config_parses_ordered_chains(self, qtbot):
        """Enabling chain order + entering chains populates the config."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._cb_chain.setChecked(True)
        widget._ordered_chains_edit.setPlainText("A, B, C\nD, E, F")

        config = widget._build_qc_config()
        assert config.use_chain_ordering is True
        assert config.ordered_chains == [["A", "B", "C"], ["D", "E", "F"]]

    def test_parse_ordered_chains_strips_and_drops_empties(self, qtbot):
        """Chain parsing strips whitespace and drops empty lines/tokens."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Blank lines, trailing commas, and extra spaces should be cleaned up.
        widget._ordered_chains_edit.setPlainText(
            "  TTI , Tail_0 ,, Tail_1 \n\n  \nHead,Neck,\n"
        )
        chains = widget._parse_ordered_chains()
        assert chains == [["TTI", "Tail_0", "Tail_1"], ["Head", "Neck"]]

    def test_threshold_widgets_disable_when_unchecked(self, qtbot):
        """A detector's threshold widgets disable when its checkbox is off."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Flip is on by default -> threshold enabled.
        assert widget._sb_flip_thr.isEnabled()
        widget._cb_flip.setChecked(False)
        assert not widget._sb_flip_thr.isEnabled()

        # Chain is off by default -> chain widgets + skeleton-trace panel
        # disabled. The advanced free-text edit now lives inside the trace panel
        # (issue #2769, item 2), so we check the panel's enabled state and the
        # edit's enabled-state relative to its own collapsible section.
        assert not widget._sb_chain_angle.isEnabled()
        assert not widget._sb_order_thr.isEnabled()
        assert not widget._chain_trace_panel.isEnabled()
        edit_parent = widget._ordered_chains_edit.parent()
        widget._cb_chain.setChecked(True)
        assert widget._sb_chain_angle.isEnabled()
        assert widget._sb_order_thr.isEnabled()
        assert widget._chain_trace_panel.isEnabled()
        # The free-text edit is reachable once its advanced section is expanded.
        assert widget._ordered_chains_edit.isEnabledTo(edit_parent)


class TestQCB2DetectorSettings:
    """Tests for the two B2 channel controls (appearance + in-sample model)."""

    def test_b2_controls_exist_with_defaults(self, qtbot):
        """The appearance + in-sample controls exist and default OFF."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Appearance / wrong-object checkbox, experimental default-OFF.
        assert widget._cb_appearance is not None
        assert widget._cb_appearance.text() == "Appearance / wrong-object"
        assert not widget._cb_appearance.isChecked()

        # In-sample model prediction checkbox, experimental default-OFF.
        assert widget._cb_insample is not None
        assert widget._cb_insample.text() == "In-sample model prediction"
        assert not widget._cb_insample.isChecked()

        # Model-path picker exists: a placeholder line edit + a Browse button.
        assert widget._insample_model_edit is not None
        assert widget._insample_model_edit.text() == ""
        assert "model" in widget._insample_model_edit.placeholderText().lower()
        assert widget._insample_browse_btn is not None
        assert "Browse" in widget._insample_browse_btn.text()

    def test_insample_tooltip_warns_about_slow_inference(self, qtbot):
        """The in-sample checkbox tooltip warns that it runs full inference."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        tip = widget._cb_insample.toolTip().lower()
        assert "inference" in tip
        assert "slow" in tip

    def test_insample_picker_disabled_when_unchecked(self, qtbot):
        """The model picker + Browse button disable when in-sample is off."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Off by default -> picker + browse disabled.
        assert not widget._insample_model_edit.isEnabled()
        assert not widget._insample_browse_btn.isEnabled()

        # Enabling the checkbox enables both.
        widget._cb_insample.setChecked(True)
        assert widget._insample_model_edit.isEnabled()
        assert widget._insample_browse_btn.isEnabled()

        # Disabling again disables both.
        widget._cb_insample.setChecked(False)
        assert not widget._insample_model_edit.isEnabled()
        assert not widget._insample_browse_btn.isEnabled()

    def test_browse_sets_model_path_from_dialog(self, qtbot):
        """Clicking Browse opens getExistingDirectory and sets the line edit."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        with patch(
            "sleap.gui.widgets.qc.QtWidgets.QFileDialog.getExistingDirectory",
            return_value="/path/to/model",
        ) as mock_dialog:
            widget._on_browse_insample_model()
            mock_dialog.assert_called_once()
        assert widget._insample_model_edit.text() == "/path/to/model"

    def test_browse_cancel_leaves_path_unchanged(self, qtbot):
        """Cancelling the folder dialog (empty return) leaves the path empty."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        with patch(
            "sleap.gui.widgets.qc.QtWidgets.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            widget._on_browse_insample_model()
        assert widget._insample_model_edit.text() == ""

    def test_build_qc_config_b2_defaults(self, qtbot):
        """_build_qc_config maps the B2 controls; both channels default OFF."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        config = widget._build_qc_config()
        assert config.use_appearance is False
        assert config.use_insample_prediction is False
        assert config.insample_model_path == ""

    def test_build_qc_config_reflects_b2_toggles(self, qtbot):
        """Toggling the B2 checkboxes + path is reflected in the built config."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._cb_appearance.setChecked(True)
        widget._cb_insample.setChecked(True)
        widget._insample_model_edit.setText("  /models/best  ")

        config = widget._build_qc_config()
        assert config.use_appearance is True
        assert config.use_insample_prediction is True
        # Path is stripped of surrounding whitespace.
        assert config.insample_model_path == "/models/best"


class TestQCAnalysisWorkerConfig:
    """Tests for threading the QCConfig into the analysis worker."""

    def test_worker_stores_config(self, qtbot):
        """QCAnalysisWorker stores the config passed to it."""
        labels = MagicMock()
        cfg = QCConfig()
        worker = QCAnalysisWorker(labels, config=cfg)
        assert worker._config is cfg

    def test_worker_config_defaults_to_none(self, qtbot):
        """QCAnalysisWorker config defaults to None when omitted."""
        worker = QCAnalysisWorker(MagicMock())
        assert worker._config is None

    def test_run_analysis_builds_worker_with_config(self, qtbot):
        """_on_run_analysis builds a worker with a non-None config.

        Patches QCAnalysisWorker to capture the config arg without spinning up
        the background thread.
        """
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Tweak a couple of controls so the built config is non-default.
        widget._cb_chain.setChecked(True)
        widget._ordered_chains_edit.setPlainText("A, B, C")

        # Provide labels with enough user instances to pass the guard.
        mock_lf = MagicMock()
        mock_lf.user_instances = [MagicMock(), MagicMock()]
        mock_labels = MagicMock()
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))
        widget._labels = mock_labels

        captured = {}

        def fake_worker(labels, config=None, parent=None):
            captured["labels"] = labels
            captured["config"] = config
            # Return a stub that looks like a non-running worker.
            stub = MagicMock()
            stub.isRunning.return_value = False
            return stub

        with patch("sleap.gui.widgets.qc.QCAnalysisWorker", side_effect=fake_worker):
            widget._on_run_analysis()

        assert captured["labels"] is mock_labels
        assert isinstance(captured["config"], QCConfig)
        assert captured["config"].use_chain_ordering is True
        assert captured["config"].ordered_chains == [["A", "B", "C"]]


class TestCollapsibleGroupBox:
    """Tests for the CollapsibleGroupBox helper (issue #2769, item 4)."""

    def test_starts_expanded_by_default(self, qtbot):
        """Default (collapsed=False) starts checked with the body visible."""
        box = CollapsibleGroupBox("Title")
        qtbot.addWidget(box)
        assert box.isCheckable()
        assert box.isChecked()
        assert box.content.isVisible() or not box.isVisible()  # body tracks header

    def test_starts_collapsed_when_requested(self, qtbot):
        """collapsed=True starts unchecked with the body hidden."""
        box = CollapsibleGroupBox("Title", collapsed=True)
        qtbot.addWidget(box)
        assert not box.isChecked()
        assert not box.content.isVisible()

    def test_toggle_shows_and_hides_body(self, qtbot):
        """Toggling the header shows/hides the content frame."""
        box = CollapsibleGroupBox("Title", collapsed=True)
        qtbot.addWidget(box)
        box.show()

        # Expand.
        box.setChecked(True)
        assert box.content.isVisible()

        # Collapse again.
        box.setChecked(False)
        assert not box.content.isVisible()

    def test_body_children_stay_enabled_when_collapsed(self, qtbot):
        """Collapsing must not disable body widgets (visibility-only collapse).

        A plain checkable QGroupBox would disable all descendants when
        unchecked; CollapsibleGroupBox must keep the body enabled so per-widget
        enable/disable logic survives collapsing.
        """
        box = CollapsibleGroupBox("Title", collapsed=True)
        qtbot.addWidget(box)

        line = QtWidgets.QLineEdit()
        layout = QtWidgets.QVBoxLayout(box.content)
        layout.addWidget(line)
        box.apply_collapsed_state()

        # Even though the box is collapsed, the child's effective enabled state
        # follows its own flag, not the collapsed header.
        assert line.isEnabled()
        line.setEnabled(False)
        assert not line.isEnabled()


class TestQCUXLayout:
    """Tests for the overall UX/layout revamp (issue #2769, items 1, 3, 4, 7)."""

    # --- Item 1: two-line progress row ------------------------------------

    def test_progress_label_wraps_on_its_own_line(self, qtbot):
        """Status text wraps and sits separately from the progress bar."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        # Word wrap lets the status text use its own line without squeezing.
        assert widget._progress_label.wordWrap()
        # The status label and progress bar are distinct widgets (two lines).
        assert widget._progress_label is not widget._progress_bar

    # --- Item 3: per-detector "?" help buttons ----------------------------

    def test_detector_help_has_all_detectors(self):
        """DETECTOR_HELP covers every detector mentioned in #2769."""
        for key in [
            "flip",
            "chimera",
            "duplicate",
            "chain",
            "missing",
            "appearance",
            "insample",
        ]:
            assert key in DETECTOR_HELP
            title, body = DETECTOR_HELP[key]
            assert title and body
            # Friendly, biologist-facing copy: reasonably descriptive.
            assert len(body) > 40

    def test_help_buttons_present_for_each_detector(self, qtbot):
        """A "?" tool button exists in the settings grid for each detector."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        help_buttons = [
            b
            for b in widget._detector_settings_group.findChildren(QtWidgets.QToolButton)
            if b.text() == "?"
        ]
        # One per detector row (7 detectors).
        assert len(help_buttons) == 7

    def test_show_detector_help_pops_message_box(self, qtbot):
        """Clicking help shows a QMessageBox carrying the plain-language body."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        created = {}

        class FakeBox:
            def __init__(self, *a, **k):
                created["instance"] = self

            def setIcon(self, *a):
                pass

            def setWindowTitle(self, t):
                created["title"] = t

            def setText(self, t):
                created["text"] = t

            def setInformativeText(self, t):
                created["informative"] = t

            def setStandardButtons(self, *a):
                pass

            def exec_(self):
                created["shown"] = True

        with patch("sleap.gui.widgets.qc.QtWidgets.QMessageBox") as mock_box:
            mock_box.side_effect = FakeBox
            mock_box.Information = 0
            mock_box.Ok = 0
            widget._show_detector_help("flip")

        assert created.get("shown")
        # The informative text is the friendly body for the flip detector.
        assert created["informative"] == DETECTOR_HELP["flip"][1]
        assert "left" in created["informative"].lower()

    def test_help_button_click_invokes_help(self, qtbot):
        """The "?" button is wired to _show_detector_help."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        help_buttons = [
            b
            for b in widget._detector_settings_group.findChildren(QtWidgets.QToolButton)
            if b.text() == "?"
        ]
        with patch.object(widget, "_show_detector_help") as mock_help:
            help_buttons[0].click()
        mock_help.assert_called_once()

    # --- Item 4: collapsible panels --------------------------------------

    def test_collapsible_panels_exist(self, qtbot):
        """Charts + Selected Instance + Statistics are collapsible sections."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        for group in (
            widget._charts_group,
            widget._details_group,
            widget._stats_group,
            widget._detector_settings_group,
        ):
            assert isinstance(group, CollapsibleGroupBox)

    def test_detector_settings_collapsed_charts_expanded_by_default(self, qtbot):
        """Advanced panel starts collapsed; the everyday panels start expanded."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        # Advanced -> collapsed.
        assert not widget._detector_settings_group.isChecked()
        # Everyday panels -> expanded.
        assert widget._charts_group.isChecked()
        assert widget._details_group.isChecked()
        assert widget._stats_group.isChecked()

    def test_collapsing_charts_hides_tabs(self, qtbot):
        """Collapsing the charts group hides the visualization tabs."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget.show()

        assert widget._viz_tabs.isVisible()
        widget._charts_group.setChecked(False)
        assert not widget._viz_tabs.isVisible()
        widget._charts_group.setChecked(True)
        assert widget._viz_tabs.isVisible()

    def test_charts_height_capped_and_panel_scrolls(self, qtbot):
        """Charts can't expand without bound; the panel is scrollable.

        Regression for #2769 item 4: the matplotlib canvases used to grow the
        panel (and the whole window) without limit, so users couldn't reach the
        flagged-instances table to review. The charts now have a locked maximum
        height and the whole panel lives in a resizable scroll area.
        """
        widget = QCWidget()
        qtbot.addWidget(widget)
        # Charts height is locked so the canvases can't expand without bound.
        assert 0 < widget._viz_tabs.maximumHeight() <= 300
        assert (
            widget._viz_tabs.sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Maximum
        )
        # The whole panel lives inside a resizable scroll area...
        assert isinstance(widget._scroll, QtWidgets.QScrollArea)
        assert widget._scroll.widgetResizable()
        # ...and the table is still reachable through the scroll container.
        container = widget._scroll.widget()
        assert container is not None
        assert widget._table_view in container.findChildren(QtWidgets.QTableView)

    # --- Item 7: plain-language Selected Instance + Statistics ------------

    def test_friendly_issue_maps_known_labels(self):
        """_friendly_issue turns raw issue labels into plain-language clauses."""
        assert _friendly_issue("Whole-instance L/R flip") == (
            "left/right sides look swapped"
        )
        assert "out of order" in _friendly_issue("Wrong keypoint order along chain")
        assert _friendly_issue("Unusual joint angle") == (
            "a joint bends at an unusual angle"
        )

    def test_friendly_issue_falls_back_gracefully(self):
        """Unknown / raw labels are cleaned up rather than shown verbatim."""
        # "High <feature>" fallbacks drop the prefix and underscores.
        assert _friendly_issue("High pose_split_score") == "pose split score"
        # A bare unknown label is lowercased.
        assert _friendly_issue("Some Weird Thing") == "some weird thing"

    def test_selected_details_plain_language(self, qtbot):
        """Selected Instance panel says WHY in plain language, with location."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._selected_flag = MockQCFlag(
            video_idx=0,
            frame_idx=1837,
            instance_idx=1,
            score=0.91,
            confidence="high",
            top_issue="Whole-instance L/R flip",
        )
        widget._update_selected_details()
        text = widget._details_label.text()

        assert "Flagged:" in text
        # The clause is capitalized into a sentence.
        assert "Left/right sides look swapped" in text
        assert "0.91" in text
        assert "Frame 1837" in text
        assert "instance 1" in text
        # No raw feature names leaking through.
        assert "edge_zscore" not in text

    def test_statistics_plain_language_summary(self, qtbot):
        """Statistics panel summarizes counts + most common issue in words."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # 1200 user instances across the labels.
        mock_lf = MagicMock()
        mock_lf.user_instances = [MagicMock()] * 1200
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=300)
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))
        widget._labels = mock_labels

        # 45 flagged, most common issue "Unusual joint angle" (x27).
        flagged = []
        for i in range(27):
            flagged.append(MockQCFlag(0, i, 0, 0.95, "high", "Unusual joint angle"))
        for i in range(18):
            flagged.append(MockQCFlag(0, 100 + i, 0, 0.72, "medium", "Likely L/R swap"))

        mock_results = MagicMock()
        mock_results.get_flagged.return_value = flagged
        widget._results = mock_results
        widget._threshold_slider.setValue(70)

        widget._update_statistics()
        text = widget._stats_label.text()

        assert "45" in text and "1,200" in text  # comma-formatted totals
        assert "3.8%" in text
        assert "Most common issue" in text
        assert "Unusual Joint Angle" in text  # titled like the table column
        assert "(27)" in text

    def test_statistics_no_flags_is_reassuring(self, qtbot):
        """With nothing flagged, Statistics reassures instead of dumping zeros."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        mock_lf = MagicMock()
        mock_lf.user_instances = [MagicMock()] * 10
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=5)
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))
        widget._labels = mock_labels

        mock_results = MagicMock()
        mock_results.get_flagged.return_value = []
        widget._results = mock_results
        widget._update_statistics()

        text = widget._stats_label.text()
        assert "No issues flagged" in text

    def test_statistics_ready_message_before_analysis(self, qtbot):
        """Pre-analysis Statistics reads as a friendly call to action."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        mock_lf = MagicMock()
        mock_lf.user_instances = [MagicMock()] * 1500
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=42)
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))
        widget.set_labels(mock_labels)

        text = widget._stats_label.text()
        assert "Ready to analyze" in text
        assert "1,500" in text  # comma-formatted
        assert "Run Analysis" in text


class TestQCFlagReviewedColumn:
    """Tests for the model's Reviewed checkmark column (issue #2769, item 6)."""

    def test_reviewed_column_is_appended_last(self):
        """Reviewed is the last column; data indices Frame..Issue are unchanged."""
        model = QCFlagTableModel()
        assert model.COLUMNS[-1] == "Reviewed"
        assert model.REVIEWED_COL == len(model.COLUMNS) - 1
        # The original data columns keep their positions.
        assert model.COLUMNS[:5] == [
            "Frame",
            "Instance",
            "Score",
            "Confidence",
            "Issue",
        ]

    def test_reviewed_column_is_user_checkable(self):
        """The Reviewed column carries the user-checkable item flag."""
        model = QCFlagTableModel()
        model.items = [MockQCFlag(0, 1, 0, 0.9, "high", "flip")]
        idx = model.index(0, QCFlagTableModel.REVIEWED_COL)
        assert model.flags(idx) & QtCore.Qt.ItemIsUserCheckable
        # Other columns are not user-checkable.
        other = model.index(0, 0)
        assert not (model.flags(other) & QtCore.Qt.ItemIsUserCheckable)

    def test_reviewed_defaults_unchecked(self):
        """A fresh flag shows an unchecked Reviewed box."""
        model = QCFlagTableModel()
        model.items = [MockQCFlag(0, 1, 0, 0.9, "high", "flip")]
        idx = model.index(0, QCFlagTableModel.REVIEWED_COL)
        assert model.data(idx, QtCore.Qt.CheckStateRole) == QtCore.Qt.Unchecked

    def test_setdata_toggles_reviewed(self):
        """Setting CheckStateRole marks the flag reviewed and back."""
        model = QCFlagTableModel()
        flag = MockQCFlag(0, 1, 0, 0.9, "high", "flip")
        model.items = [flag]
        idx = model.index(0, QCFlagTableModel.REVIEWED_COL)

        assert model.setData(idx, QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
        assert model.is_reviewed(flag)
        assert model.data(idx, QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked
        assert model.reviewed_count() == 1

        assert model.setData(idx, QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        assert not model.is_reviewed(flag)
        assert model.reviewed_count() == 0

    def test_setdata_ignores_non_reviewed_columns(self):
        """setData on a non-Reviewed column / wrong role is a no-op."""
        model = QCFlagTableModel()
        model.items = [MockQCFlag(0, 1, 0, 0.9, "high", "flip")]
        # Wrong column.
        assert not model.setData(
            model.index(0, 0), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole
        )
        # Wrong role on the Reviewed column.
        assert not model.setData(
            model.index(0, QCFlagTableModel.REVIEWED_COL), "x", QtCore.Qt.EditRole
        )

    def test_reviewed_state_keyed_by_identity_not_row(self):
        """Reviewed-state follows the instance identity across row reshuffles."""
        shared = set()
        model = QCFlagTableModel(reviewed_keys=shared)
        f1 = MockQCFlag(0, 1, 0, 0.9, "high", "flip")
        f2 = MockQCFlag(0, 2, 1, 0.8, "medium", "angle")
        model.items = [f1, f2]

        model.set_reviewed(f1, True)
        assert (0, 1, 0) in shared

        # Replace the row list with NEW flag objects that share f1's identity:
        # the model must still treat that identity as reviewed.
        f1b = MockQCFlag(0, 1, 0, 0.95, "high", "flip")
        f3 = MockQCFlag(0, 3, 0, 0.7, "low", "scale")
        model.items = [f3, f1b]
        idx_f1b = model.index(1, QCFlagTableModel.REVIEWED_COL)
        assert model.data(idx_f1b, QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked
        idx_f3 = model.index(0, QCFlagTableModel.REVIEWED_COL)
        assert model.data(idx_f3, QtCore.Qt.CheckStateRole) == QtCore.Qt.Unchecked

    def test_reviewed_count_only_counts_shown_rows(self):
        """reviewed_count reflects only the rows currently in the model."""
        shared = set()
        model = QCFlagTableModel(reviewed_keys=shared)
        f1 = MockQCFlag(0, 1, 0, 0.9, "high", "flip")
        f2 = MockQCFlag(0, 2, 0, 0.8, "medium", "angle")
        model.items = [f1, f2]
        model.set_reviewed(f1, True)
        model.set_reviewed(f2, True)
        assert model.reviewed_count() == 2

        # Show only f2's row; the count drops even though f1 is still reviewed.
        model.items = [f2]
        assert model.reviewed_count() == 1
        assert (0, 1, 0) in shared  # f1 remains reviewed in the shared set

    def test_sort_by_reviewed_column(self):
        """Sorting by the Reviewed column groups reviewed/unreviewed rows."""
        shared = set()
        model = QCFlagTableModel(reviewed_keys=shared)
        f1 = MockQCFlag(0, 1, 0, 0.9, "high", "flip")
        f2 = MockQCFlag(0, 2, 0, 0.8, "medium", "angle")
        f3 = MockQCFlag(0, 3, 0, 0.7, "low", "scale")
        model.items = [f1, f2, f3]
        model.set_reviewed(f2, True)

        # Ascending: unreviewed (False) first.
        model.sort(QCFlagTableModel.REVIEWED_COL, QtCore.Qt.AscendingOrder)
        assert not model.is_reviewed(model.items[0])
        assert model.is_reviewed(model.items[-1])

        # Descending: reviewed first.
        model.sort(QCFlagTableModel.REVIEWED_COL, QtCore.Qt.DescendingOrder)
        assert model.is_reviewed(model.items[0])


class TestCheckableFilterMenu:
    """Tests for the multi-select menu that stays open (issue #2769, item 5)."""

    def test_checkable_action_toggles_without_closing(self, qtbot):
        """Releasing the mouse on a checkable action toggles it in place."""
        menu = CheckableFilterMenu()
        qtbot.addWidget(menu)
        action = menu.addAction("Type A")
        action.setCheckable(True)
        action.setChecked(False)

        # Make it the active action and simulate a mouse release on it.
        menu.setActiveAction(action)
        with patch.object(CheckableFilterMenu, "activeAction", return_value=action):
            # super().mouseReleaseEvent must NOT be called for checkable items;
            # the action is toggled directly instead.
            with patch(
                "sleap.gui.widgets.qc.QtWidgets.QMenu.mouseReleaseEvent"
            ) as super_release:
                menu.mouseReleaseEvent(MagicMock())
        assert action.isChecked()  # toggled on
        super_release.assert_not_called()  # menu kept open

    def test_non_checkable_action_uses_default_behavior(self, qtbot):
        """A non-checkable action falls through to the default (closes menu)."""
        menu = CheckableFilterMenu()
        qtbot.addWidget(menu)
        action = menu.addAction("Select all")  # not checkable

        with patch.object(CheckableFilterMenu, "activeAction", return_value=action):
            with patch(
                "sleap.gui.widgets.qc.QtWidgets.QMenu.mouseReleaseEvent"
            ) as super_release:
                menu.mouseReleaseEvent(MagicMock())
        super_release.assert_called_once()


class TestQCListIssueFilter:
    """Tests for the flagged-list issue-type filter (issue #2769, item 5)."""

    def _make_widget_with_flags(self, qtbot, flags):
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._results = _fake_results(flags)
        widget._update_flagged_display()
        return widget

    def test_filter_button_disabled_until_results(self, qtbot):
        """The issue-type filter button is disabled before any results exist."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._issue_filter_button is not None
        assert not widget._issue_filter_button.isEnabled()
        assert widget._issue_filter_button.text() == "Issue types: all"

    def test_menu_lists_present_issue_types(self, qtbot):
        """The menu has one checkable entry per issue type present, all shown."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Whole-instance L/R flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Unusual joint angle"),
            MockQCFlag(0, 3, 0, 0.85, "medium", "Unusual joint angle"),
            MockQCFlag(0, 4, 0, 0.80, "medium", "Appearance outlier"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # One action per unique raw issue (3 of them), button enabled.
        assert widget._issue_filter_button.isEnabled()
        assert set(widget._issue_filter_actions.keys()) == {
            "Whole-instance L/R flip",
            "Unusual joint angle",
            "Appearance outlier",
        }
        # All present and selected by default -> all four rows shown.
        assert widget._table_model.rowCount() == 4
        assert widget._issue_filter_button.text() == "Issue types: all"
        # Menu labels are the friendly, title-cased category names.
        labels = {a.text() for a in widget._issue_filter_actions.values()}
        assert "Whole-Instance L/R Flip" in labels

    def test_deselecting_issue_type_filters_table(self, qtbot):
        """Unchecking an issue type removes those rows from the table live."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Whole-instance L/R flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Unusual joint angle"),
            MockQCFlag(0, 3, 0, 0.85, "medium", "Unusual joint angle"),
            MockQCFlag(0, 4, 0, 0.80, "medium", "Appearance outlier"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        widget._on_issue_type_toggled("Unusual joint angle", False)
        assert widget._table_model.rowCount() == 2
        shown = {
            widget._table_model.items[r].top_issue
            for r in range(widget._table_model.rowCount())
        }
        assert shown == {"Whole-instance L/R flip", "Appearance outlier"}
        assert widget._issue_filter_button.text() == "Issue types: 2 of 3"

    def test_select_none_and_all(self, qtbot):
        """Select-none empties the table; select-all restores every row."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        widget._set_all_issue_types(False)
        assert widget._table_model.rowCount() == 0
        assert widget._issue_filter_button.text() == "Issue types: none"

        widget._set_all_issue_types(True)
        assert widget._table_model.rowCount() == 2
        assert widget._issue_filter_button.text() == "Issue types: all"

    def test_filter_selection_survives_threshold_refilter(self, qtbot):
        """A de-selected issue type stays hidden across a threshold re-filter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.80, "medium", "Appearance"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Hide "Flip".
        widget._on_issue_type_toggled("Flip", False)
        assert "Flip" not in widget._visible_issue_types

        # Simulate raising the threshold so the 0.80 Appearance flag drops out.
        widget._results.get_flagged.return_value = [f for f in flags if f.score >= 0.85]
        widget._update_flagged_display()

        # Flip is still present but remains hidden; only Angle shows.
        assert set(widget._issue_filter_actions.keys()) == {"Flip", "Angle"}
        assert widget._visible_issue_types == {"Angle"}
        shown = {
            widget._table_model.items[r].top_issue
            for r in range(widget._table_model.rowCount())
        }
        assert shown == {"Angle"}

    def test_breakdown_reflects_full_set_not_filter(self, qtbot):
        """The Issue Breakdown chart shows ALL flagged, ignoring the table filter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.85, "medium", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        widget._on_issue_type_toggled("Angle", False)  # hide Angle in the table
        # Table shows only Flip now...
        assert widget._table_model.rowCount() == 1
        # ...but the breakdown still counts all three (Flip=1, Angle=2).
        counts = widget._breakdown_canvas._issue_counts
        assert counts.get("Flip") == 1
        assert counts.get("Angle") == 2


class TestQCListReviewedIntegration:
    """Widget-level reviewed-state + counter tests (issue #2769, item 6)."""

    def _make_widget_with_flags(self, qtbot, flags):
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._results = _fake_results(flags)
        widget._update_flagged_display()
        return widget

    def test_reviewed_counter_widgets_exist(self, qtbot):
        """The running reviewed counter starts at 0 / 0."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._reviewed_count_label is not None
        assert widget._reviewed_count_label.text() == "0 / 0 reviewed"

    def test_counter_updates_when_box_checked(self, qtbot):
        """Ticking a Reviewed box updates the running counter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)
        assert widget._reviewed_count_label.text() == "0 / 2 reviewed"

        idx = widget._table_model.index(0, QCFlagTableModel.REVIEWED_COL)
        widget._table_model.setData(idx, QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"

    def test_navigate_auto_marks_reviewed(self, qtbot):
        """Selecting a row (navigating) auto-marks that instance reviewed."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        widget._table_view.selectRow(0)
        qtbot.wait(20)
        assert widget._table_model.is_reviewed(flags[0])
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"

    def test_reviewed_survives_threshold_refilter(self, qtbot):
        """Reviewed marks persist across a threshold re-filter (identity-keyed)."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.80, "medium", "Appearance"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Mark the highest-score flag (frame 1) reviewed.
        idx = widget._table_model.index(0, QCFlagTableModel.REVIEWED_COL)
        widget._table_model.setData(idx, QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
        assert (0, 1, 0) in widget._reviewed_keys

        # Raise threshold so frame-3 (0.80) drops; frame-1 stays and is still
        # reviewed -> counter is 1 / 2.
        widget._results.get_flagged.return_value = [f for f in flags if f.score >= 0.85]
        widget._update_flagged_display()
        assert (0, 1, 0) in widget._reviewed_keys
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"

    def test_reviewed_survives_issue_filter(self, qtbot):
        """Reviewed marks persist when an issue type is hidden then re-shown."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Review the Angle flag.
        widget._table_model.set_reviewed(flags[1], True)
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"

        # Hide Angle: counter now reflects only the shown (Flip) rows.
        widget._on_issue_type_toggled("Angle", False)
        assert widget._reviewed_count_label.text() == "0 / 1 reviewed"

        # Re-show Angle: its reviewed mark is intact.
        widget._on_issue_type_toggled("Angle", True)
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"
        assert widget._table_model.is_reviewed(flags[1])

    def test_set_labels_resets_filter_and_reviewed(self, qtbot):
        """Loading a new project clears reviewed marks and the issue filter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)
        widget._table_model.set_reviewed(flags[0], True)
        widget._on_issue_type_toggled("Angle", False)
        widget._hide_reviewed_check.setChecked(True)
        assert widget._reviewed_keys
        assert widget._visible_issue_types is not None

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=3)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))
        widget.set_labels(mock_labels)

        assert widget._reviewed_keys == set()
        assert widget._visible_issue_types is None
        assert widget._all_flagged == []
        assert widget._reviewed_count_label.text() == "0 / 0 reviewed"
        assert not widget._issue_filter_button.isEnabled()
        # The "Hide reviewed" filter is reset and disabled for a fresh project.
        assert not widget._hide_reviewed_check.isChecked()
        assert not widget._hide_reviewed_check.isEnabled()


class TestQCListHideReviewedFilter:
    """Tests for the "Hide reviewed" not-reviewed filter (Group C / #2769).

    The filter shows only flagged instances NOT in ``_reviewed_keys``, combined
    (logical AND) with the issue-type filter, and updates the table live as rows
    are marked reviewed/unreviewed.
    """

    def _make_widget_with_flags(self, qtbot, flags):
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._results = _fake_results(flags)
        widget._update_flagged_display()
        return widget

    def test_hide_reviewed_checkbox_exists_and_gated_on_results(self, qtbot):
        """The control exists, defaults off, and is disabled before results."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._hide_reviewed_check is not None
        assert widget._hide_reviewed_check.text() == "Hide reviewed"
        # Unchecked + disabled until there are flagged rows to hide.
        assert not widget._hide_reviewed_check.isChecked()
        assert not widget._hide_reviewed_check.isEnabled()

        # Once results with flags exist, the checkbox becomes usable.
        flags = [MockQCFlag(0, 1, 0, 0.95, "high", "Flip")]
        widget._results = _fake_results(flags)
        widget._update_flagged_display()
        assert widget._hide_reviewed_check.isEnabled()

    def test_hide_reviewed_shows_only_unreviewed(self, qtbot):
        """Enabling the filter shows exactly the not-reviewed instances."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.85, "medium", "Appearance"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Mark two of the three reviewed.
        widget._table_model.set_reviewed(flags[0], True)
        widget._table_model.set_reviewed(flags[2], True)

        # Turn on "Hide reviewed": only the single unreviewed flag remains.
        widget._hide_reviewed_check.setChecked(True)
        shown = {
            widget._table_model.items[r].instance_key
            for r in range(widget._table_model.rowCount())
        }
        assert shown == {(0, 2, 0)}
        # The shown rows are all unreviewed, so the counter reads 0 / 1.
        assert widget._reviewed_count_label.text() == "0 / 1 reviewed"

        # Unchecking restores the full (issue-filtered) set.
        widget._hide_reviewed_check.setChecked(False)
        assert widget._table_model.rowCount() == 3

    def test_hide_reviewed_intersects_with_issue_filter(self, qtbot):
        """The not-reviewed filter AND-combines with the issue-type filter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.85, "medium", "Angle"),
            MockQCFlag(0, 4, 0, 0.80, "medium", "Appearance"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Review one Angle flag (frame 2) and the Flip flag (frame 1).
        widget._table_model.set_reviewed(flags[0], True)  # Flip
        widget._table_model.set_reviewed(flags[1], True)  # Angle (frame 2)

        # Show only the "Angle" issue type...
        widget._set_all_issue_types(False)
        widget._on_issue_type_toggled("Angle", True)
        # ...and hide reviewed. Intersection = unreviewed Angle = frame 3 only.
        widget._hide_reviewed_check.setChecked(True)

        shown = {
            widget._table_model.items[r].instance_key
            for r in range(widget._table_model.rowCount())
        }
        assert shown == {(0, 3, 0)}

    def test_marking_shown_row_reviewed_removes_it_live(self, qtbot):
        """Ticking a shown row reviewed drops it from the unreviewed view live."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Hide reviewed with nothing reviewed yet -> both rows show.
        widget._hide_reviewed_check.setChecked(True)
        assert widget._table_model.rowCount() == 2

        # Tick the first shown row reviewed via the model (as the checkbox does).
        idx = widget._table_model.index(0, QCFlagTableModel.REVIEWED_COL)
        first_key = widget._table_model.items[0].instance_key
        widget._table_model.setData(idx, QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)

        # The re-filter is deferred to the next event-loop turn; let it run.
        qtbot.wait(20)

        keys = {
            widget._table_model.items[r].instance_key
            for r in range(widget._table_model.rowCount())
        }
        assert first_key not in keys
        assert widget._table_model.rowCount() == 1
        # Only the remaining unreviewed row is shown -> 0 / 1 reviewed.
        assert widget._reviewed_count_label.text() == "0 / 1 reviewed"

    def test_navigate_auto_mark_removes_row_live_when_hiding(self, qtbot):
        """Selecting a row auto-marks it reviewed; it leaves the hidden view."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)
        widget._hide_reviewed_check.setChecked(True)

        selected_key = widget._table_model.items[0].instance_key
        widget._table_view.selectRow(0)
        # Selection auto-marks reviewed and schedules the deferred re-filter.
        qtbot.wait(20)

        keys = {
            widget._table_model.items[r].instance_key
            for r in range(widget._table_model.rowCount())
        }
        assert selected_key not in keys
        assert selected_key in widget._reviewed_keys
        assert widget._table_model.rowCount() == 1

    def test_unhiding_brings_reviewed_rows_back(self, qtbot):
        """Toggling the filter off re-shows reviewed rows (state preserved)."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        widget._table_model.set_reviewed(flags[0], True)
        widget._hide_reviewed_check.setChecked(True)
        assert widget._table_model.rowCount() == 1  # only the unreviewed one

        widget._hide_reviewed_check.setChecked(False)
        # Both rows back; the reviewed mark on frame 1 is intact.
        assert widget._table_model.rowCount() == 2
        assert widget._table_model.is_reviewed(flags[0])
        assert widget._reviewed_count_label.text() == "1 / 2 reviewed"

    def test_hide_reviewed_survives_threshold_refilter(self, qtbot):
        """A raised threshold re-applies the active not-reviewed filter."""
        flags = [
            MockQCFlag(0, 1, 0, 0.95, "high", "Flip"),
            MockQCFlag(0, 2, 0, 0.90, "high", "Angle"),
            MockQCFlag(0, 3, 0, 0.80, "medium", "Appearance"),
        ]
        widget = self._make_widget_with_flags(qtbot, flags)

        # Review frame 2, then hide reviewed -> frames 1 and 3 show.
        widget._table_model.set_reviewed(flags[1], True)
        widget._hide_reviewed_check.setChecked(True)
        assert widget._table_model.rowCount() == 2

        # Raise the threshold so the 0.80 Appearance flag (frame 3) drops out;
        # the still-active "Hide reviewed" filter leaves only the unreviewed
        # frame 1.
        widget._results.get_flagged.return_value = [f for f in flags if f.score >= 0.85]
        widget._update_flagged_display()

        shown = {
            widget._table_model.items[r].instance_key
            for r in range(widget._table_model.rowCount())
        }
        assert shown == {(0, 1, 0)}


class _FakeNode:
    """Minimal stand-in for a sleap-io skeleton Node (just a ``.name``)."""

    def __init__(self, name):
        self.name = name


class _FakeEdge:
    """Minimal stand-in for a sleap-io skeleton Edge (source/destination)."""

    def __init__(self, src, dst):
        self.source = _FakeNode(src)
        self.destination = _FakeNode(dst)


class _FakeSkeleton:
    """Minimal skeleton exposing ``node_names`` and ``edges`` like sleap-io."""

    def __init__(self, node_names, edges):
        self.node_names = list(node_names)
        self.edges = [_FakeEdge(s, d) for s, d in edges]


class TestQCSkeletonTraceCanvas:
    """Tests for the click-to-trace skeleton canvas (issue #2769, item 2)."""

    def test_starts_empty(self, qtbot):
        """A fresh canvas has no skeleton and an empty trace."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        assert canvas._positions == {}
        assert canvas.trace == []

    def test_set_skeleton_lays_out_nodes(self, qtbot):
        """Setting a skeleton computes a position for every node."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B", "C"], [("A", "B"), ("B", "C")])
        assert set(canvas._positions.keys()) == {"A", "B", "C"}
        # Edges are kept as validated name pairs.
        assert ("A", "B") in canvas._edges

    def test_set_skeleton_drops_unknown_edges(self, qtbot):
        """Edges referencing missing nodes are filtered out."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B"], [("A", "B"), ("B", "Z")])
        assert canvas._edges == [("A", "B")]

    def test_set_trace_filters_to_known_nodes(self, qtbot):
        """set_trace keeps only nodes that exist in the current skeleton."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B", "C"], [])
        canvas.set_trace(["A", "Q", "C"])
        assert canvas.trace == ["A", "C"]

    def test_node_at_hit_and_miss(self, qtbot):
        """node_at returns the nearest node within radius, else None."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B", "C"], [("A", "B"), ("B", "C")])
        ax, ay = canvas._positions["A"]
        assert canvas.node_at(ax, ay) == "A"
        # A point far outside the spring layout extent hits nothing.
        assert canvas.node_at(999.0, 999.0) is None

    def test_click_emits_node_clicked(self, qtbot):
        """A button-press over a node emits node_clicked with its name."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B", "C"], [("A", "B"), ("B", "C")])
        canvas.resize(400, 300)
        canvas.draw()

        received = []
        canvas.node_clicked.connect(received.append)

        disp = canvas.axes.transData.transform(canvas._positions["B"])
        event = MouseEvent("button_press_event", canvas, disp[0], disp[1], button=1)
        canvas._on_click(event)
        assert received == ["B"]

    def test_empty_skeleton_redraw_is_safe(self, qtbot):
        """Updating the plot with no skeleton does not raise."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton([], [])
        canvas.update_plot()  # Should render the placeholder without error.
        assert canvas._positions == {}


class TestQCChainTracePanel:
    """Tests for the chain-order skeleton-tracing UI (issue #2769, item 2)."""

    def test_trace_panel_widgets_exist(self, qtbot):
        """The trace panel and its key controls are constructed."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._chain_trace_panel is not None
        assert isinstance(widget._skeleton_canvas, QCSkeletonTraceCanvas)
        assert widget._chains_list is not None
        # The advanced free-text fallback still exists.
        assert widget._ordered_chains_edit is not None
        # The zoom/pan canvas exposes a Reset view affordance.
        assert widget._reset_view_btn is not None

    def test_clicking_nodes_builds_trace(self, qtbot):
        """Node clicks append to the in-progress trace; the canvas mirrors it."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._on_trace_node_clicked("Base")
        widget._on_trace_node_clicked("Mid")
        widget._on_trace_node_clicked("Tip")
        assert widget._trace_in_progress == ["Base", "Mid", "Tip"]
        assert widget._skeleton_canvas.trace == []  # no skeleton loaded -> filtered

    def test_consecutive_duplicate_click_ignored(self, qtbot):
        """Clicking the same node twice in a row does not duplicate it."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._on_trace_node_clicked("Base")
        widget._on_trace_node_clicked("Base")
        assert widget._trace_in_progress == ["Base"]

    def test_undo_and_clear_trace(self, qtbot):
        """Undo removes the last node; clear empties the trace."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        for name in ["A", "B", "C"]:
            widget._on_trace_node_clicked(name)
        widget._on_trace_undo()
        assert widget._trace_in_progress == ["A", "B"]
        widget._on_trace_clear()
        assert widget._trace_in_progress == []

    def test_add_chain_requires_two_nodes(self, qtbot):
        """A chain of fewer than two nodes is rejected (warning shown)."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._on_trace_node_clicked("OnlyOne")
        with patch.object(QtWidgets.QMessageBox, "information") as info:
            widget._on_trace_add_chain()
        info.assert_called_once()
        assert widget._traced_chains == []

    def test_add_chain_commits_and_resets(self, qtbot):
        """Adding a valid chain stores it and clears the in-progress trace."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        for name in ["Base", "Mid", "Tip"]:
            widget._on_trace_node_clicked(name)
        widget._on_trace_add_chain()
        assert widget._traced_chains == [["Base", "Mid", "Tip"]]
        assert widget._trace_in_progress == []
        assert widget._chains_list.count() == 1

    def test_reorder_saved_chains(self, qtbot):
        """Up/down moves a selected saved chain within the list."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._traced_chains = [["A", "B"], ["C", "D"], ["E", "F"]]
        widget._refresh_chains_list()
        widget._chains_list.setCurrentRow(2)
        widget._move_selected_chain(-1)
        assert widget._traced_chains == [["A", "B"], ["E", "F"], ["C", "D"]]
        assert widget._chains_list.currentRow() == 1
        # Moving past the end is a no-op.
        widget._chains_list.setCurrentRow(2)
        widget._move_selected_chain(1)
        assert widget._traced_chains == [["A", "B"], ["E", "F"], ["C", "D"]]

    def test_remove_saved_chain(self, qtbot):
        """Removing the selected chain drops it from the list and state."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._traced_chains = [["A", "B"], ["C", "D"]]
        widget._refresh_chains_list()
        widget._chains_list.setCurrentRow(0)
        widget._on_remove_selected_chain()
        assert widget._traced_chains == [["C", "D"]]
        assert widget._chains_list.count() == 1

    def test_collect_ordered_chains_merges_traced_and_text(self, qtbot):
        """Traced chains come first, then free-text chains, de-duplicated."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._traced_chains = [["Base", "Mid", "Tip"]]
        widget._ordered_chains_edit.setPlainText("Head, Neck\nBase, Mid, Tip")
        # The duplicate (Base, Mid, Tip) typed after tracing is dropped.
        assert widget._collect_ordered_chains() == [
            ["Base", "Mid", "Tip"],
            ["Head", "Neck"],
        ]

    def test_build_qc_config_uses_traced_chains(self, qtbot):
        """Traced chains flow into QCConfig.ordered_chains."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._cb_chain.setChecked(True)
        widget._traced_chains = [["TTI", "Tail_0", "Tail_1"]]
        config = widget._build_qc_config()
        assert config.ordered_chains == [["TTI", "Tail_0", "Tail_1"]]

    def test_set_labels_loads_skeleton_into_canvas(self, qtbot):
        """Loading labels with a skeleton populates the trace canvas."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        skeleton = _FakeSkeleton(
            ["TTI", "Tail_0", "Tail_1"], [("TTI", "Tail_0"), ("Tail_0", "Tail_1")]
        )
        labels = MagicMock()
        labels.skeletons = [skeleton]
        labels.__len__ = MagicMock(return_value=0)
        labels.__iter__ = MagicMock(return_value=iter([]))

        widget.set_labels(labels)
        assert set(widget._skeleton_canvas._positions.keys()) == {
            "TTI",
            "Tail_0",
            "Tail_1",
        }

    def test_set_labels_without_skeleton_clears_canvas(self, qtbot):
        """Labels whose skeleton access raises leave the canvas empty."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Seed a skeleton, then load labels that expose no usable skeleton.
        widget._skeleton_canvas.set_skeleton(["A", "B"], [("A", "B")])

        labels = MagicMock()
        labels.skeletons = []  # no skeletons -> canvas cleared
        labels.__len__ = MagicMock(return_value=0)
        labels.__iter__ = MagicMock(return_value=iter([]))

        widget.set_labels(labels)
        assert widget._skeleton_canvas._positions == {}

    def test_trace_panel_disabled_with_chain_off(self, qtbot):
        """The trace panel follows the chain detector checkbox."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget._cb_chain.isChecked()
        assert not widget._chain_trace_panel.isEnabled()
        widget._cb_chain.setChecked(True)
        assert widget._chain_trace_panel.isEnabled()

    def test_chain_order_warns_when_enabled_without_chains(self, qtbot):
        """Ticking 'Wrong chain order' with no chains prompts to open the editor."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._collect_ordered_chains() == []
        with patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ) as mq:
            with patch.object(widget, "_open_chain_trace_dialog") as mopen:
                widget._on_chain_checked(True)
        mq.assert_called_once()
        mopen.assert_called_once()

    def test_chain_order_no_warning_when_chains_exist_or_unchecked(self, qtbot):
        """No prompt when chains already exist, or when the box is unchecked."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        # Chains already defined -> ticking on does not prompt.
        with patch.object(widget, "_collect_ordered_chains", return_value=[["A", "B"]]):
            with patch.object(QtWidgets.QMessageBox, "question") as mq:
                widget._on_chain_checked(True)
            mq.assert_not_called()
        # Unchecking never prompts.
        with patch.object(QtWidgets.QMessageBox, "question") as mq2:
            widget._on_chain_checked(False)
        mq2.assert_not_called()


class TestCollapsibleDisclosure:
    """Tests for the ``<details>``-style disclosure arrow on CollapsibleGroupBox.

    The header should read like a GitHub ``<details>``/``<summary>`` disclosure:
    a ``▶`` arrow when collapsed and a ``▼`` arrow when expanded, while keeping
    the caller's title queryable without the arrow (issue #2769 follow-up).
    """

    def test_collapsed_shows_right_arrow(self, qtbot):
        """A collapsed box prefixes its title with a ▶ arrow."""
        box = CollapsibleGroupBox("Detector Settings", collapsed=True)
        qtbot.addWidget(box)
        # ``title()`` returns the clean caller text; the displayed (Qt) title
        # carries the arrow prefix.
        assert box.title() == "Detector Settings"
        assert QtWidgets.QGroupBox.title(box).startswith("▶")
        assert "Detector Settings" in QtWidgets.QGroupBox.title(box)

    def test_expanded_shows_down_arrow(self, qtbot):
        """An expanded box prefixes its title with a ▼ arrow."""
        box = CollapsibleGroupBox("Charts", collapsed=False)
        qtbot.addWidget(box)
        assert QtWidgets.QGroupBox.title(box).startswith("▼")

    def test_arrow_flips_on_toggle(self, qtbot):
        """Toggling the header flips the disclosure arrow ▶ <-> ▼."""
        box = CollapsibleGroupBox("Stats", collapsed=True)
        qtbot.addWidget(box)
        assert QtWidgets.QGroupBox.title(box).startswith("▶")

        box.setChecked(True)
        assert QtWidgets.QGroupBox.title(box).startswith("▼")

        box.setChecked(False)
        assert QtWidgets.QGroupBox.title(box).startswith("▶")

    def test_set_title_keeps_arrow(self, qtbot):
        """Re-setting the title keeps the leading disclosure arrow in sync."""
        box = CollapsibleGroupBox("Old", collapsed=False)
        qtbot.addWidget(box)
        box.setTitle("New title")
        assert box.title() == "New title"
        displayed = QtWidgets.QGroupBox.title(box)
        assert displayed.startswith("▼")
        assert "New title" in displayed

    def test_detector_settings_header_has_arrow(self, qtbot):
        """The Detector Settings panel renders the disclosure arrow."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        group = widget._detector_settings_group
        # Collapsed by default -> right arrow; clean title still queryable.
        assert group.title() == "Detector Settings"
        assert QtWidgets.QGroupBox.title(group).startswith("▶")

    def test_disclosure_hides_native_checkbox(self, qtbot):
        """Header shows only the arrow -- the native checkbox is hidden.

        A checkable QGroupBox draws a native checkbox next to the title (very
        visible on macOS); we keep it checkable (so isChecked()/setChecked()/
        click-to-expand work) but style its indicator to zero size so only the
        ▶/▼ disclosure arrow shows (issue #2769 follow-up).
        """
        box = CollapsibleGroupBox("Detector Settings", collapsed=True)
        qtbot.addWidget(box)
        # Still checkable, so the toggle API + click-to-expand keep working.
        assert box.isCheckable()
        box.setChecked(True)
        assert box.isChecked()
        # The native checkbox indicator is styled to nothing.
        ss = box.styleSheet().replace(" ", "").lower()
        assert "qgroupbox::indicator" in ss
        assert "width:0" in ss and "height:0" in ss


class TestQCRestoreDefaults:
    """Tests for the Detector Settings "Restore defaults" button (item #2769).

    Changing several controls and then clicking Restore defaults must put every
    control back to a fresh ``QCConfig()`` -- enable checkboxes, thresholds,
    ordered chains, the in-sample model path and the B2 toggles.
    """

    def test_restore_button_exists_in_panel(self, qtbot):
        """A "Restore defaults" button lives inside the Detector Settings panel."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._restore_defaults_btn is not None
        assert widget._restore_defaults_btn.text() == "Restore defaults"
        # It is a descendant of the Detector Settings group.
        buttons = widget._detector_settings_group.findChildren(QtWidgets.QPushButton)
        assert widget._restore_defaults_btn in buttons

    def test_restore_resets_every_control_to_qcconfig_defaults(self, qtbot):
        """Mutating many controls then restoring matches QCConfig() defaults."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # --- Mutate a wide spread of controls away from their defaults. ---
        widget._cb_flip.setChecked(False)
        widget._cb_chimera.setChecked(False)
        widget._cb_duplicate.setChecked(False)
        widget._cb_chain.setChecked(True)
        widget._cb_missing.setChecked(True)
        widget._cb_appearance.setChecked(True)
        widget._cb_insample.setChecked(True)

        widget._sb_flip_thr.setValue(0.95)
        widget._sb_dup_thr.setValue(0.1)
        widget._sb_chain_angle.setValue(120)
        widget._sb_order_thr.setValue(0.8)
        widget._sb_missing_thr.setValue(0.2)

        widget._insample_model_edit.setText("/tmp/some/model")
        widget._ordered_chains_edit.setPlainText("a, b, c")
        widget._traced_chains = [["head", "tail"]]
        widget._trace_in_progress = ["head"]

        # --- Restore. ---
        widget._restore_defaults_btn.click()

        defaults = QCConfig()

        # Enable checkboxes back to documented GUI defaults.
        assert widget._cb_flip.isChecked()
        assert widget._cb_chimera.isChecked()
        assert widget._cb_duplicate.isChecked()
        assert not widget._cb_chain.isChecked()
        assert not widget._cb_missing.isChecked()
        assert not widget._cb_appearance.isChecked()
        assert not widget._cb_insample.isChecked()

        # Thresholds back to defaults.
        assert widget._sb_flip_thr.value() == defaults.chirality_flip_threshold
        assert widget._sb_dup_thr.value() == defaults.duplicate_score_threshold
        assert widget._sb_chain_angle.value() == int(defaults.chain_turn_angle_deg)
        assert widget._sb_order_thr.value() == defaults.order_inversion_threshold
        assert widget._sb_missing_thr.value() == defaults.missing_node_prob_threshold

        # In-sample path + ordered chains cleared (QCConfig defaults are ""/[]).
        assert widget._insample_model_edit.text() == defaults.insample_model_path
        assert widget._ordered_chains_edit.toPlainText() == ""
        assert widget._traced_chains == []
        assert widget._trace_in_progress == []
        assert widget._chains_list.count() == 0

    def test_restore_yields_config_equal_to_defaults(self, qtbot):
        """After restore, the rebuilt config matches a fresh QCConfig()."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._cb_chain.setChecked(True)
        widget._sb_missing_thr.setValue(0.42)
        widget._traced_chains = [["x", "y"]]

        widget._on_restore_detector_defaults()

        built = widget._build_qc_config()
        defaults = QCConfig()
        # The GUI exposes a subset of fields; assert each round-trips to default.
        assert built.use_chirality is True  # "auto" -> ON in GUI
        assert built.use_split_detection == defaults.use_split_detection
        assert built.use_duplicate_score == defaults.use_duplicate_score
        assert built.use_chain_ordering is False
        assert built.use_missing_node_check is False
        assert built.use_appearance is False
        assert built.use_insample_prediction is False
        assert built.chirality_flip_threshold == defaults.chirality_flip_threshold
        assert built.duplicate_score_threshold == defaults.duplicate_score_threshold
        assert built.chain_turn_angle_deg == defaults.chain_turn_angle_deg
        assert built.order_inversion_threshold == defaults.order_inversion_threshold
        assert built.missing_node_prob_threshold == defaults.missing_node_prob_threshold
        assert built.insample_model_path == defaults.insample_model_path
        assert built.ordered_chains == defaults.ordered_chains

    def test_apply_config_to_widgets_loads_values(self, qtbot):
        """_apply_config_to_widgets pushes a non-default config into the panel."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        config = QCConfig(
            use_chirality=False,
            use_chain_ordering=True,
            use_appearance=True,
            chirality_flip_threshold=0.25,
            chain_turn_angle_deg=90.0,
            missing_node_prob_threshold=0.55,
            insample_model_path="/models/best",
            ordered_chains=[["base", "mid", "tip"]],
        )
        widget._apply_config_to_widgets(config)

        assert not widget._cb_flip.isChecked()
        assert widget._cb_chain.isChecked()
        assert widget._cb_appearance.isChecked()
        assert widget._sb_flip_thr.value() == 0.25
        assert widget._sb_chain_angle.value() == 90
        assert widget._sb_missing_thr.value() == 0.55
        assert widget._insample_model_edit.text() == "/models/best"
        assert widget._traced_chains == [["base", "mid", "tip"]]
        assert widget._chains_list.count() == 1


def _real_labels(node_names, edges, instance_coords):
    """Build a real sleap-io ``Labels`` with one frame of user instances.

    Args:
        node_names: Ordered node names for the skeleton.
        edges: List of ``(src, dst)`` name pairs for skeleton edges.
        instance_coords: List of ``(n_nodes, 2)`` array-likes, one per instance.
            ``np.nan`` rows are treated as invisible nodes.

    Returns:
        A ``Labels`` object whose single frame holds the given user instances,
        all sharing one ``Skeleton``.
    """
    import numpy as np
    from sleap_io.model.skeleton import Skeleton
    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video

    skeleton = Skeleton(list(node_names), edges=[list(e) for e in edges])
    video = Video(filename="dummy.mp4")
    instances = [
        Instance.from_numpy(np.asarray(coords, dtype=float), skeleton=skeleton)
        for coords in instance_coords
    ]
    lf = LabeledFrame(video=video, frame_idx=0, instances=instances)
    return Labels([lf])


class TestQCSkeletonTraceCanvasRealCoords:
    """Real-animal layout for the trace canvas (issue #2769 follow-up).

    The canvas should draw the skeleton using real labeled coordinates when they
    are provided, and fall back to the spring/line layout when they are not.
    """

    def test_set_node_positions_drives_layout(self, qtbot):
        """Real coords are normalized into the display frame and used as-is."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        # An L-shaped animal: B is right of A, C is below B (image coords, y down).
        canvas.set_skeleton(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            node_positions={"A": (0.0, 0.0), "B": (100.0, 0.0), "C": (100.0, 50.0)},
        )
        pos = canvas._positions
        assert set(pos) == {"A", "B", "C"}
        # Normalized to ~[-1, 1] (max half-extent maps to 1).
        for x, y in pos.values():
            assert -1.0001 <= x <= 1.0001
            assert -1.0001 <= y <= 1.0001
        # Horizontal order from the real coords is preserved (A left of B).
        assert pos["A"][0] < pos["B"][0]
        # y is flipped (image y grows down): C is *below* B, so its drawn y is
        # smaller than B's.
        assert pos["C"][1] < pos["B"][1]
        # The layout came from the real coords, not the seeded spring layout.
        assert canvas._node_coords != {}

    def test_set_node_positions_keeps_hit_testing(self, qtbot):
        """Click hit-testing works against the real-coordinate positions."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            node_positions={"A": (0.0, 0.0), "B": (100.0, 0.0), "C": (200.0, 0.0)},
        )
        bx, by = canvas._positions["B"]
        assert canvas.node_at(bx, by) == "B"
        # A far-away point still misses everything.
        assert canvas.node_at(50.0, 50.0) is None

    def test_falls_back_to_spring_when_no_coords(self, qtbot):
        """With no real coords the seeded spring/line layout is used."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(["A", "B", "C"], [("A", "B"), ("B", "C")])
        assert canvas._node_coords == {}
        assert set(canvas._positions) == {"A", "B", "C"}

    def test_single_real_coord_falls_back(self, qtbot):
        """Fewer than two finite coords cannot define a scale -> fall back."""
        import numpy as np

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (5.0, 5.0), "B": (np.nan, np.nan)},
        )
        # Only one finite coord -> _layout_from_coords returns None, spring used,
        # but both nodes still get a position.
        assert set(canvas._positions) == {"A", "B"}

    def test_node_without_coord_is_not_drawn(self, qtbot):
        """A node missing a real coord gets no position (and edges skip it)."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        canvas.set_skeleton(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            node_positions={"A": (0.0, 0.0), "B": (10.0, 0.0)},  # C omitted
        )
        assert set(canvas._positions) == {"A", "B"}
        # Redraw must not raise even though edge B->C references a missing node.
        canvas.update_plot()


class TestQCRealAnimalLayout:
    """Widget-level real-animal layout fed from labeled instances."""

    def test_set_labels_computes_median_node_positions(self, qtbot):
        """set_labels derives per-node medians and feeds them to the canvas."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Three instances of the same L-shaped pose, translated to different
        # parts of the frame. After per-instance centering the medians recover
        # the shared shape, so the canvas uses real coords (not a spring layout).
        base = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0]]

        def shifted(dx, dy):
            return [[x + dx, y + dy] for x, y in base]

        labels = _real_labels(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            [shifted(0, 0), shifted(500, 10), shifted(-300, 200)],
        )
        widget.set_labels(labels)

        coords = widget._skeleton_canvas._node_coords
        assert set(coords) == {"A", "B", "C"}
        # The recovered shape keeps B to the right of A and C below B (the L).
        assert coords["B"][0] > coords["A"][0]
        assert coords["C"][1] > coords["B"][1]  # raw image coords: y grows down

    def test_representative_positions_skips_invisible_nodes(self, qtbot):
        """A node never visible in any instance gets no representative coord."""
        import numpy as np

        widget = QCWidget()
        qtbot.addWidget(widget)

        nan = [np.nan, np.nan]
        labels = _real_labels(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            [
                [[0.0, 0.0], [10.0, 0.0], nan],
                [[1.0, 1.0], [11.0, 1.0], nan],
            ],
        )
        widget.set_labels(labels)
        coords = widget._skeleton_canvas._node_coords
        # C was invisible everywhere -> no representative position for it.
        assert "C" not in coords
        assert {"A", "B"} <= set(coords)

    def test_set_labels_without_instances_uses_spring(self, qtbot):
        """Labels with a skeleton but no instances fall back to spring layout."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        skeleton = _FakeSkeleton(["A", "B", "C"], [("A", "B"), ("B", "C")])
        labels = MagicMock()
        labels.skeletons = [skeleton]
        labels.__len__ = MagicMock(return_value=0)
        labels.__iter__ = MagicMock(return_value=iter([]))

        widget.set_labels(labels)
        # No instances -> no real coords, but the spring layout still positions
        # every node so the user can trace.
        assert widget._skeleton_canvas._node_coords == {}
        assert set(widget._skeleton_canvas._positions) == {"A", "B", "C"}


def _synthetic_image(h=80, w=120, channels=None):
    """Build a small deterministic image for trace-canvas tests.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        channels: ``None`` for a 2D grayscale image, or an int (1 or 3) for a
            trailing channel dimension.

    Returns:
        A ``uint8`` ``np.ndarray`` of shape ``(h, w)``, ``(h, w, 1)``, or
        ``(h, w, 3)``.
    """
    import numpy as np

    base = (np.add.outer(np.linspace(0, 200, h), np.linspace(0, 50, w))).astype("uint8")
    if channels is None:
        return base
    if channels == 1:
        return base[..., None]
    return np.stack([base, base, base], axis=-1)


class TestQCTraceCanvasImageMode:
    """Image-backed trace canvas: photo background + pixel-space overlay.

    The user traces directly on a real labeled frame, so the canvas shows the
    frame image and overlays the skeleton at true pixel coordinates, with mouse
    zoom/pan (issue #2769 follow-up).
    """

    def test_image_sets_background_and_pixel_positions(self, qtbot):
        """With an image, the canvas shows it and uses pixel-space node coords."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(80, 120)
        coords = {"A": (40.0, 30.0), "B": (80.0, 50.0), "C": (100.0, 70.0)}
        canvas.set_skeleton(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            node_positions=coords,
            image=img,
        )
        # The background image is stored and an AxesImage was drawn.
        assert canvas._background_image is not None
        assert canvas._image_artist is not None
        from matplotlib.image import AxesImage

        assert any(isinstance(a, AxesImage) for a in canvas.axes.get_images())
        # Nodes sit at their *pixel* coords, not a normalized [-1, 1] value.
        assert canvas._positions["A"] == (40.0, 30.0)
        assert canvas._positions["C"] == (100.0, 70.0)

    def test_image_view_uses_top_left_origin(self, qtbot):
        """The view spans the image extent with y inverted (top-left origin)."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(80, 120)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (10.0, 10.0), "B": (90.0, 70.0)},
            image=img,
        )
        x0, x1 = canvas.axes.get_xlim()
        y0, y1 = canvas.axes.get_ylim()
        # x grows left->right; the default view fits the instance, so it spans
        # at most the full image width and contains the instance's x-extent.
        assert x0 < x1
        assert (x1 - x0) <= 120 + 2.0
        assert x0 <= 10.0 and x1 >= 90.0
        # y is inverted so (0, 0) is at the top, like an image.
        assert y0 > y1

    def test_image_mode_defaults_to_instance_fit(self, qtbot):
        """The default image-mode view zooms to the instance, not the full frame."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(400, 600)  # 600 wide x 400 tall
        # Instance occupies a small central region of the frame.
        coords = {"A": (280.0, 190.0), "B": (320.0, 210.0)}
        canvas.set_skeleton(["A", "B"], [("A", "B")], node_positions=coords, image=img)
        x0, x1 = canvas.axes.get_xlim()
        y_bottom, y_top = canvas.axes.get_ylim()  # inverted: bottom > top
        # Tighter than the full image extent on every side...
        assert -0.5 < x0 < 280.0 and 320.0 < x1 < 599.5
        assert -0.5 < y_top < 190.0 and 210.0 < y_bottom < 399.5
        # ...but still contains the whole instance bounding box.
        assert x0 <= 280.0 and x1 >= 320.0
        assert y_top <= 190.0 and y_bottom >= 210.0

    def test_grayscale_channel_image_is_squeezed(self, qtbot):
        """An (H, W, 1) grayscale image is squeezed to 2D for imshow."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(60, 90, channels=1)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (10.0, 10.0), "B": (50.0, 40.0)},
            image=img,
        )
        assert canvas._background_image is not None
        assert canvas._background_image.ndim == 2

    def test_rgb_image_passes_through(self, qtbot):
        """An (H, W, 3) RGB image is kept 3-channel for imshow."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(60, 90, channels=3)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (10.0, 10.0), "B": (50.0, 40.0)},
            image=img,
        )
        assert canvas._background_image is not None
        assert canvas._background_image.ndim == 3
        assert canvas._background_image.shape[-1] == 3

    def test_node_at_pixel_space(self, qtbot):
        """node_at hits a node at its pixel coordinate and misses far away."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            node_positions={"A": (20.0, 20.0), "B": (75.0, 50.0), "C": (130.0, 90.0)},
            image=img,
        )
        assert canvas.node_at(75.0, 50.0) == "B"
        # A pixel between far-apart nodes hits nothing.
        assert canvas.node_at(0.0, 99.0) is None

    def test_node_at_still_works_after_zoom(self, qtbot):
        """Hit-testing keeps working after a wheel zoom (pixel pick radius)."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        # Zoom in around node B.
        disp = canvas.axes.transData.transform((120.0, 80.0))
        ev = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        ev.button = "up"
        canvas._on_scroll(ev)
        # B is still selectable at its pixel coordinate after zooming.
        assert canvas.node_at(120.0, 80.0) == "B"

    def test_scroll_zooms_changes_limits(self, qtbot):
        """Scrolling up shrinks the view; scrolling down grows it."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()

        def _width():
            lo, hi = canvas.axes.get_xlim()
            return abs(hi - lo)

        before = _width()
        disp = canvas.axes.transData.transform((75.0, 50.0))
        up = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        up.button = "up"
        canvas._on_scroll(up)
        zoomed_in = _width()
        assert zoomed_in < before

        down = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        down.button = "down"
        canvas._on_scroll(down)
        assert _width() > zoomed_in

    def test_reset_view_restores_default_view(self, qtbot):
        """reset_view returns to the default (instance-fit) view after zooming."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        lo0, hi0 = canvas.axes.get_xlim()
        default_w = abs(hi0 - lo0)
        # Zoom in, then reset.
        disp = canvas.axes.transData.transform((75.0, 50.0))
        up = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        up.button = "up"
        canvas._on_scroll(up)
        zoomed = abs(canvas.axes.get_xlim()[1] - canvas.axes.get_xlim()[0])
        assert zoomed < default_w
        canvas.reset_view()
        lo, hi = canvas.axes.get_xlim()
        assert abs(abs(hi - lo) - default_w) < 2.0

    def test_trace_edit_preserves_zoom(self, qtbot):
        """Editing the trace after zooming keeps the user's zoomed view."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        disp = canvas.axes.transData.transform((75.0, 50.0))
        up = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        up.button = "up"
        canvas._on_scroll(up)
        zoomed = canvas.axes.get_xlim()
        # A trace change triggers a redraw but must not snap back to full extent.
        canvas.set_trace(["A", "B"])
        assert canvas.axes.get_xlim() == zoomed

    def test_pan_with_right_button_shifts_view(self, qtbot):
        """Dragging with the right button pans the view; left does not."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        start = canvas.axes.transData.transform((75.0, 50.0))
        press = MouseEvent("button_press_event", canvas, start[0], start[1], button=3)
        canvas._on_pan_press(press)
        # Move 40 display px to the right.
        move = MouseEvent(
            "motion_notify_event", canvas, start[0] + 40, start[1], button=3
        )
        canvas._on_pan_move(move)
        x0, x1 = canvas.axes.get_xlim()
        # Panning right shows smaller-x content (limits decrease).
        assert x0 < -0.5
        release = MouseEvent(
            "button_release_event", canvas, start[0] + 40, start[1], button=3
        )
        canvas._on_pan_release(release)
        assert canvas._pan_anchor is None

    def test_left_click_does_not_pan(self, qtbot):
        """The left button is reserved for selection and never starts a pan."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        start = canvas.axes.transData.transform((75.0, 50.0))
        press = MouseEvent("button_press_event", canvas, start[0], start[1], button=1)
        canvas._on_pan_press(press)
        assert canvas._pan_anchor is None

    def test_double_click_resets_view(self, qtbot):
        """A left double-click resets the zoomed view to the default view."""
        from matplotlib.backend_bases import MouseEvent

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(100, 150)
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (20.0, 20.0), "B": (120.0, 80.0)},
            image=img,
        )
        canvas.resize(500, 400)
        canvas.draw()
        default_w = abs(canvas.axes.get_xlim()[1] - canvas.axes.get_xlim()[0])
        # Zoom in first.
        disp = canvas.axes.transData.transform((75.0, 50.0))
        up = MouseEvent("scroll_event", canvas, disp[0], disp[1], step=1)
        up.button = "up"
        canvas._on_scroll(up)
        # A double-click resets.
        dbl = MouseEvent("button_press_event", canvas, disp[0], disp[1], button=1)
        dbl.dblclick = True
        canvas._on_click(dbl)
        lo, hi = canvas.axes.get_xlim()
        assert abs(abs(hi - lo) - default_w) < 2.0

    def test_image_without_coords_falls_back_to_abstract(self, qtbot):
        """An image with no real coords cannot place nodes -> abstract layout."""
        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        img = _synthetic_image(60, 90)
        # No node_positions -> nothing to overlay on the photo.
        canvas.set_skeleton(["A", "B", "C"], [("A", "B"), ("B", "C")], image=img)
        assert canvas._background_image is None
        assert set(canvas._positions) == {"A", "B", "C"}

    def test_bad_image_is_ignored(self, qtbot):
        """A malformed image (wrong rank) is dropped, not drawn."""
        import numpy as np

        canvas = QCSkeletonTraceCanvas()
        qtbot.addWidget(canvas)
        bad = np.zeros((4, 4, 4, 4), dtype="uint8")  # rank-4, not an image
        canvas.set_skeleton(
            ["A", "B"],
            [("A", "B")],
            node_positions={"A": (1.0, 1.0), "B": (2.0, 2.0)},
            image=bad,
        )
        # Falls back to the abstract layout (no usable background).
        assert canvas._background_image is None


def _real_labels_with_image(node_names, edges, instance_coords, image):
    """Build a ``Labels`` whose single frame returns ``image`` from ``.image``.

    Mirrors :func:`_real_labels` but returns the frame plus the ``Labels`` so the
    caller can patch ``LabeledFrame.image`` to yield a synthetic frame without a
    real video file on disk.

    Args:
        node_names: Ordered node names for the skeleton.
        edges: List of ``(src, dst)`` name pairs for skeleton edges.
        instance_coords: List of ``(n_nodes, 2)`` array-likes, one per instance.
        image: The synthetic frame image the patched ``.image`` should return.

    Returns:
        A ``(labels, frame, image)`` tuple.
    """
    import numpy as np
    from sleap_io.model.skeleton import Skeleton
    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video

    skeleton = Skeleton(list(node_names), edges=[list(e) for e in edges])
    video = Video(filename="dummy.mp4")
    instances = [
        Instance.from_numpy(np.asarray(coords, dtype=float), skeleton=skeleton)
        for coords in instance_coords
    ]
    lf = LabeledFrame(video=video, frame_idx=0, instances=instances)
    return Labels([lf]), lf, image


class TestQCBestLabeledInstanceImage:
    """Widget-level best-instance + frame-image feeding (issue #2769 follow-up).

    ``set_labels`` should pick the labeled instance with the most present nodes,
    decode its frame image, and feed the canvas real pixel coordinates + the
    photo, degrading gracefully when no image can be decoded.
    """

    def test_picks_instance_with_most_present_nodes(self, qtbot):
        """The fully-labeled instance wins over a partially-labeled one."""
        import numpy as np

        widget = QCWidget()
        qtbot.addWidget(widget)

        nan = [np.nan, np.nan]
        # Frame holds two instances: a partial one (2 nodes) and a full one (3).
        partial = [[10.0, 10.0], [20.0, 20.0], nan]
        full = [[30.0, 30.0], [40.0, 40.0], [50.0, 60.0]]
        labels, lf, img = _real_labels_with_image(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            [partial, full],
            _synthetic_image(80, 120),
        )

        with patch.object(
            type(lf), "image", new_callable=PropertyMock, return_value=img
        ):
            widget.set_labels(labels)

        canvas = widget._skeleton_canvas
        # Image mode is active and the FULL instance drove the overlay.
        assert canvas._background_image is not None
        assert canvas._positions["A"] == (30.0, 30.0)
        assert canvas._positions["C"] == (50.0, 60.0)

    def test_best_instance_helper_returns_positions_and_image(self, qtbot):
        """The helper returns pixel coords for present nodes + the frame image."""
        import numpy as np

        widget = QCWidget()
        qtbot.addWidget(widget)

        nan = [np.nan, np.nan]
        labels, lf, img = _real_labels_with_image(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            [[[5.0, 5.0], [15.0, 25.0], nan]],
            _synthetic_image(70, 100),
        )
        widget._labels = labels
        skeleton = labels.skeletons[0]

        with patch.object(
            type(lf), "image", new_callable=PropertyMock, return_value=img
        ):
            positions, image = widget._best_labeled_instance_image(
                skeleton, list(skeleton.node_names)
            )
        assert positions["A"] == (5.0, 5.0)
        assert positions["B"] == (15.0, 25.0)
        # The invisible node is omitted (no pixel coordinate).
        assert "C" not in positions
        assert image is not None
        assert np.asarray(image).shape == (70, 100)

    def test_falls_back_to_abstract_when_image_decode_fails(self, qtbot):
        """A frame whose image cannot be decoded uses the abstract layout."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # No patched image: the dummy video cannot be opened, so .image raises.
        base = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0]]

        def shifted(dx, dy):
            return [[x + dx, y + dy] for x, y in base]

        labels = _real_labels(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            [shifted(0, 0), shifted(200, 10)],
        )
        widget.set_labels(labels)
        canvas = widget._skeleton_canvas
        # No image -> abstract layout (normalized coords, no background).
        assert canvas._background_image is None
        assert set(canvas._positions) == {"A", "B", "C"}
        for x, y in canvas._positions.values():
            assert -1.0001 <= x <= 1.0001
            assert -1.0001 <= y <= 1.0001

    def test_no_instances_returns_empty(self, qtbot):
        """With a skeleton but no instances the helper returns no image/coords."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        skeleton = _FakeSkeleton(["A", "B"], [("A", "B")])
        labels = MagicMock()
        labels.labeled_frames = []
        positions, image = widget._best_labeled_instance_image(skeleton, ["A", "B"])
        assert positions == {}
        assert image is None


class TestQCChainTraceDialog:
    """Pop-up ordered-chains editor (issue #2769 follow-up).

    The full tracing UI now lives in a dialog opened from a "Configure
    chains..." button; the inline summary reflects the configured chains and the
    chains still feed ``QCConfig.ordered_chains``.
    """

    def test_configure_button_and_summary_exist(self, qtbot):
        """The chain row exposes a button + summary instead of the inline panel."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget._chain_config_btn is not None
        assert widget._chain_summary_label is not None
        # Nothing configured yet.
        assert widget._chain_summary_label.text() == "No chains"
        # The trace panel is NOT a child of the detector grid container now; it
        # is held off to the side for the dialog to host.
        assert widget._chain_trace_panel is not None

    def test_summary_updates_when_chains_added(self, qtbot):
        """Adding/removing chains updates the inline summary text."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget._traced_chains = [["TTI", "Tail_0", "TailTip"], ["Head", "Neck"]]
        widget._refresh_chains_list()

        text = widget._chain_summary_label.text()
        assert text.startswith("2 chains:")
        # Long chains are abbreviated to endpoints; short ones shown in full.
        assert "TTI→…→TailTip" in text
        assert "Head→Neck" in text

        # Removing all chains resets the summary.
        widget._traced_chains = []
        widget._refresh_chains_list()
        assert widget._chain_summary_label.text() == "No chains"

    def test_summary_includes_free_text_chains(self, qtbot):
        """Advanced free-text chains count toward the summary too."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._ordered_chains_edit.setPlainText("A, B, C")
        widget._refresh_chain_summary()
        assert widget._chain_summary_label.text().startswith("1 chain:")
        assert "A→…→C" in widget._chain_summary_label.text()

    def test_open_dialog_hosts_panel_and_edits_chains(self, qtbot):
        """Opening the dialog hosts the panel; edits flow back into the config."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget._cb_chain.setChecked(True)

        opened = {}

        def _drive_dialog():
            dialog = widget._chain_trace_dialog
            opened["dialog"] = dialog
            # The live trace panel is hosted inside the dialog.
            assert widget._chain_trace_panel.parent() is dialog
            assert widget._chain_trace_panel.isVisible()
            # Trace a chain by clicking nodes, then commit it -- exactly the
            # interaction a user performs in the pop-up.
            for name in ["TTI", "Tail_0", "TailTip"]:
                widget._on_trace_node_clicked(name)
            widget._on_trace_add_chain()
            dialog.accept()

        QtCore.QTimer.singleShot(0, _drive_dialog)
        widget._open_chain_trace_dialog()

        assert isinstance(opened["dialog"], QCChainTraceDialog)
        # The chain edited in the dialog is now in the saved list + the summary.
        assert widget._traced_chains == [["TTI", "Tail_0", "TailTip"]]
        assert widget._chain_summary_label.text().startswith("1 chain:")
        # ...and feeds QCConfig.ordered_chains so analysis picks it up.
        config = widget._build_qc_config()
        assert config.ordered_chains == [["TTI", "Tail_0", "TailTip"]]

    def test_dialog_reused_across_opens(self, qtbot):
        """The dialog instance is created once and reused on subsequent opens."""

        widget = QCWidget()
        qtbot.addWidget(widget)

        def _close():
            widget._chain_trace_dialog.reject()

        QtCore.QTimer.singleShot(0, _close)
        widget._open_chain_trace_dialog()
        first = widget._chain_trace_dialog

        QtCore.QTimer.singleShot(0, _close)
        widget._open_chain_trace_dialog()
        assert widget._chain_trace_dialog is first
        # Panel is still hosted by (the same) dialog after reopening.
        assert widget._chain_trace_panel.parent() is first

    def test_configure_controls_follow_chain_checkbox(self, qtbot):
        """The button + summary enable/disable with the chain detector."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert not widget._cb_chain.isChecked()
        assert not widget._chain_config_btn.isEnabled()
        assert not widget._chain_summary_label.isEnabled()
        widget._cb_chain.setChecked(True)
        assert widget._chain_config_btn.isEnabled()
        assert widget._chain_summary_label.isEnabled()

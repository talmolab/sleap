"""Tests for QC widget components."""

from unittest.mock import MagicMock, patch

import pytest
from qtpy import QtCore

from sleap.gui.widgets.qc import QCFlagTableModel, QCWidget


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

    def test_empty_model(self):
        """Test model can be created empty."""
        model = QCFlagTableModel()
        assert model.rowCount() == 0
        assert model.columnCount() == 5

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

        # Create mock labeled frames with mock instances
        mock_lf1 = MagicMock()
        mock_lf1.instances = [MagicMock(), MagicMock()]  # 2 instances
        mock_lf2 = MagicMock()
        mock_lf2.instances = [MagicMock()]  # 1 instance
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

    def test_dock_widget_starts_floating(self, qtbot):
        """Test that dock widget starts in floating mode."""
        from sleap.gui.dialogs.qc import QCDockWidget

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        dock = QCDockWidget(labels=mock_labels)
        qtbot.addWidget(dock)
        assert dock.isFloating()

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

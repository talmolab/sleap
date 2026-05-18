"""Tests for QC widget components."""

from unittest.mock import MagicMock, patch

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

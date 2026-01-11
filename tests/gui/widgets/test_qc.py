"""Tests for QC widget components."""

from unittest.mock import MagicMock, patch

from sleap.gui.widgets.qc import QCFlagTableModel, QCWidget


class MockQCFlag:
    """Mock QCFlag for testing."""

    def __init__(
        self, video_idx, frame_idx, instance_idx, score, confidence, top_issue
    ):
        self.instance_key = (video_idx, frame_idx, instance_idx)
        self.score = score
        self.confidence = confidence
        self.top_issue = top_issue


class TestQCFlagTableModel:
    """Tests for QCFlagTableModel."""

    def test_properties(self):
        """Test table has expected columns."""
        model = QCFlagTableModel()
        assert "video" in model.properties
        assert "frame" in model.properties
        assert "score" in model.properties
        assert "top_issue" in model.properties

    def test_empty_model(self):
        """Test model can be created empty."""
        model = QCFlagTableModel()
        assert model.rowCount() == 0

    def test_item_to_data(self):
        """Test item conversion to display data."""
        model = QCFlagTableModel()
        flag = MockQCFlag(
            video_idx=0,
            frame_idx=10,
            instance_idx=0,
            score=0.85,
            confidence="high",
            top_issue="edge_zscore",
        )

        data = model.item_to_data(None, flag)
        assert data["video"] == 0
        assert data["frame"] == 10
        assert data["instance"] == 0
        assert data["score"] == "0.850"
        assert data["confidence"] == "High"
        assert data["top_issue"] == "Edge Zscore"

    def test_items_setter(self):
        """Test setting items on model."""
        model = QCFlagTableModel()
        flags = [
            MockQCFlag(0, 5, 0, 0.9, "high", "edge_error"),
            MockQCFlag(0, 10, 1, 0.7, "medium", "visibility"),
        ]
        model.items = flags
        assert model.rowCount() == 2


class TestQCWidget:
    """Tests for QCWidget."""

    def test_widget_creation(self, qtbot):
        """Test widget can be created."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget is not None
        assert widget.labels is None

    def test_widget_has_controls(self, qtbot):
        """Test widget has expected controls."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget.run_btn is not None
        assert widget.threshold_slider is not None
        assert widget.table_view is not None
        assert widget.goto_btn is not None
        assert widget.export_btn is not None

    def test_threshold_slider_default(self, qtbot):
        """Test default threshold is 0.7."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        assert widget.threshold_slider.value() == 70

    def test_threshold_label_updates(self, qtbot):
        """Test threshold label updates with slider."""
        widget = QCWidget()
        qtbot.addWidget(widget)
        widget.threshold_slider.setValue(50)
        assert "0.50" in widget.threshold_label.text()

    def test_set_labels(self, qtbot):
        """Test setting labels on widget."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)
        mock_labels.__iter__ = MagicMock(return_value=iter([]))

        widget.set_labels(mock_labels)
        assert widget.labels is mock_labels

    def test_run_analysis_no_labels(self, qtbot):
        """Test run analysis shows warning with no labels."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Should show warning dialog
        with patch("sleap.gui.widgets.qc.QMessageBox") as mock_msgbox:
            widget._on_run_analysis()
            mock_msgbox.warning.assert_called_once()

    def test_sensitivity_presets(self, qtbot):
        """Test sensitivity preset buttons."""
        widget = QCWidget()
        qtbot.addWidget(widget)

        widget.low_btn.click()
        assert widget.threshold_slider.value() == 80

        widget.medium_btn.click()
        assert widget.threshold_slider.value() == 70

        widget.high_btn.click()
        assert widget.threshold_slider.value() == 50

    def test_summary_no_labels(self, qtbot):
        """Test summary shows 'No labels loaded' when no labels provided.

        Regression test for bug where summary wasn't updated on init.
        """
        widget = QCWidget(labels=None)
        qtbot.addWidget(widget)
        assert "No labels loaded" in widget.summary_label.text()

    def test_summary_with_labels_before_analysis(self, qtbot):
        """Test summary shows 'Ready to analyze' when labels loaded but not analyzed.

        Regression test for bug where summary wasn't updated on init with labels.
        """
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=10)

        # Create mock labeled frames with mock instances
        mock_lf1 = MagicMock()
        mock_lf1.instances = [MagicMock(), MagicMock()]  # 2 instances
        mock_lf2 = MagicMock()
        mock_lf2.instances = [MagicMock()]  # 1 instance
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf1, mock_lf2]))

        widget = QCWidget(labels=mock_labels)
        qtbot.addWidget(widget)

        # Should show "Ready to analyze: 3 instances, 10 frames"
        assert "Ready to analyze" in widget.summary_label.text()
        assert "3 instances" in widget.summary_label.text()

    def test_summary_updates_on_threshold_change(self, qtbot):
        """Test summary updates when threshold slider changes.

        Regression test for bug where summary wasn't updated on threshold change.
        """
        widget = QCWidget()
        qtbot.addWidget(widget)

        # Create mock results
        widget._results = MagicMock()

        # Create mock flags with different scores
        mock_flag_high = MagicMock()
        mock_flag_high.score = 0.9
        mock_flag_high.confidence = "high"

        mock_flag_medium = MagicMock()
        mock_flag_medium.score = 0.7
        mock_flag_medium.confidence = "medium"

        mock_flag_low = MagicMock()
        mock_flag_low.score = 0.5
        mock_flag_low.confidence = "low"

        all_flags = [mock_flag_high, mock_flag_medium, mock_flag_low]

        def get_flagged_impl(threshold):
            return [f for f in all_flags if f.score >= threshold]

        widget._results.get_flagged = MagicMock(side_effect=get_flagged_impl)
        widget._results.get_frame_issues = MagicMock(return_value=[])

        # Mock labels with 10 instances
        mock_labels = MagicMock()
        mock_labels.__len__ = MagicMock(return_value=5)
        mock_lf = MagicMock()
        mock_lf.instances = [MagicMock() for _ in range(10)]
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))
        widget.labels = mock_labels

        # Set high threshold (0.8) - should flag 1
        widget.threshold_slider.setValue(80)
        assert "1 flagged" in widget.summary_label.text()

        # Set medium threshold (0.6) - should flag 2
        widget.threshold_slider.setValue(60)
        assert "2 flagged" in widget.summary_label.text()

        # Set low threshold (0.4) - should flag 3
        widget.threshold_slider.setValue(40)
        assert "3 flagged" in widget.summary_label.text()

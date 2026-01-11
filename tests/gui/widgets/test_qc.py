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

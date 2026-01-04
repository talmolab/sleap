"""Tests for instance size distribution widget."""

import pytest

from sleap.gui.learning.size import InstanceSizeInfo
from sleap.gui.widgets.size_distribution import (
    SizeDistributionWidget,
    SizeHistogramCanvas,
)


@pytest.fixture
def sample_size_data():
    """Create sample size data for testing."""
    return [
        InstanceSizeInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=80.0,
            raw_size=100.0,
        ),
        InstanceSizeInfo(
            video_idx=0,
            frame_idx=10,
            instance_idx=0,
            raw_width=90.0,
            raw_height=90.0,
            raw_size=90.0,
        ),
        InstanceSizeInfo(
            video_idx=0,
            frame_idx=20,
            instance_idx=0,
            raw_width=150.0,
            raw_height=50.0,
            raw_size=150.0,
        ),
    ]


class TestSizeHistogramCanvas:
    """Tests for the matplotlib canvas."""

    def test_canvas_creation(self, qtbot):
        """Test that canvas can be created."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        assert canvas is not None
        assert canvas.fig is not None
        assert canvas.axes is not None

    def test_set_data(self, qtbot, sample_size_data):
        """Test setting data on the canvas."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)
        assert canvas._data == sample_size_data

    def test_set_rotation_angle(self, qtbot, sample_size_data):
        """Test setting rotation angle."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)
        canvas.set_rotation_angle(15.0)
        assert canvas._rotation_angle == 15.0

    def test_set_view_mode(self, qtbot, sample_size_data):
        """Test switching view modes."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)

        canvas.set_view_mode("scatter")
        assert canvas._view_mode == "scatter"

        canvas.set_view_mode("histogram")
        assert canvas._view_mode == "histogram"

    def test_set_histogram_bins(self, qtbot, sample_size_data):
        """Test setting histogram bins."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)
        canvas.set_histogram_bins(20)
        assert canvas._hist_bins == 20

    def test_set_histogram_range(self, qtbot, sample_size_data):
        """Test setting histogram range."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)
        canvas.set_histogram_range(50.0, 200.0)
        assert canvas._hist_x_min == 50.0
        assert canvas._hist_x_max == 200.0

    def test_point_clicked_signal(self, qtbot, sample_size_data):
        """Test that point_clicked signal is emitted."""
        canvas = SizeHistogramCanvas()
        qtbot.addWidget(canvas)

        canvas.set_data(sample_size_data)

        # Verify signal is defined
        assert hasattr(canvas, "point_clicked")


class TestSizeDistributionWidget:
    """Tests for the main distribution widget."""

    def test_widget_creation(self, qtbot):
        """Test that widget can be created."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        assert widget is not None
        assert widget._canvas is not None
        assert widget._rotation_combo is not None
        assert widget._recompute_button is not None

    def test_rotation_presets(self, qtbot):
        """Test rotation preset selection."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        # Test each preset
        widget.set_rotation_preset("Off")
        assert widget._get_rotation_angle() == 0.0

        widget.set_rotation_preset("+/-15")
        assert widget._get_rotation_angle() == 15.0

        widget.set_rotation_preset("+/-180")
        assert widget._get_rotation_angle() == 180.0

    def test_custom_angle(self, qtbot):
        """Test custom angle setting."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_rotation_preset("Custom")
        widget.set_custom_angle(30)

        assert widget._get_rotation_angle() == 30.0

    def test_navigate_signal(self, qtbot):
        """Test that navigate_to_frame signal exists."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        assert hasattr(widget, "navigate_to_frame")

    def test_set_labels(self, qtbot, centered_pair_labels):
        """Test setting labels on the widget."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_labels(centered_pair_labels)

        # Should have computed data
        assert len(widget._data) > 0

    def test_view_mode_toggle(self, qtbot, centered_pair_labels):
        """Test toggling between scatter and histogram views."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_labels(centered_pair_labels)

        # Toggle to histogram
        widget._histogram_radio.setChecked(True)
        assert widget._canvas._view_mode == "histogram"

        # Toggle back to scatter
        widget._scatter_radio.setChecked(True)
        assert widget._canvas._view_mode == "scatter"

    def test_histogram_controls_enabled(self, qtbot, centered_pair_labels):
        """Test that histogram controls are enabled in histogram mode."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_labels(centered_pair_labels)

        # In scatter mode, controls should be disabled
        assert not widget._bins_spin.isEnabled()
        assert not widget._xmin_spin.isEnabled()
        assert not widget._xmax_spin.isEnabled()

        # Switch to histogram mode
        widget._histogram_radio.setChecked(True)

        # Controls should now be enabled
        assert widget._bins_spin.isEnabled()
        assert widget._xmin_spin.isEnabled()
        assert widget._xmax_spin.isEnabled()

    def test_statistics_update(self, qtbot, centered_pair_labels):
        """Test that statistics are computed and displayed."""
        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_labels(centered_pair_labels)

        # Statistics label should have content
        stats_text = widget._stats_label.text()
        assert "Count:" in stats_text
        assert "Mean" in stats_text
        assert "Median" in stats_text

    def test_empty_data_handling(self, qtbot, centered_pair_labels):
        """Test widget handles empty data gracefully."""
        # Remove all instances
        for lf in centered_pair_labels:
            lf.instances = []

        widget = SizeDistributionWidget()
        qtbot.addWidget(widget)

        widget.set_labels(centered_pair_labels)

        # Should handle empty data without crashing
        assert widget._data == []
        assert "No data" in widget._stats_label.text()

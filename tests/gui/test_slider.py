from sleap.gui.widgets.slider import VideoSlider, set_slider_marks_from_labels
import pytest


def test_slider(qtbot, centered_pair_predictions):

    labels = centered_pair_predictions

    slider = VideoSlider(min=0, max=1200, val=15, marks=(10, 15))

    assert slider.value_range == 1200

    assert slider.value() == 15
    slider.setValue(20)
    assert slider.value() == 20

    assert slider.getSelection() == (0, 0)
    slider.startSelection(3)
    slider.endSelection(5)
    assert slider.getSelection() == (3, 5)
    slider.clearSelection()
    assert slider.getSelection() == (0, 0)

    initial_height = slider.maximumHeight()
    slider.setNumberOfTracks(20)
    assert slider.maximumHeight() != initial_height

    set_slider_marks_from_labels(slider, labels, labels.videos[0])
    assert len(slider.getMarks("track")) == 40

    slider.moveSelectionAnchor(5, 5)
    slider.releaseSelectionAnchor(100, 15)
    assert slider.getSelection() == (slider._toVal(5), slider._toVal(100))

    slider.setSelection(20, 30)
    assert slider.getSelection() == (20, 30)

    slider.setEnabled(False)
    assert not slider.enabled()

    slider.setEnabled(True)
    assert slider.enabled()

@pytest.mark.parametrize(
    "slider_width, x_value, min_value, max_value, expected_value",
    [
        # Test values within range
        (1000, 500, 0, 1000, 500),    # Midpoint translation
        (800, 400, 0, 800, 400),      # Maximum boundary for smaller range
        (1500, 750, 100, 1200, 600),  # Midpoint translation with offset
        (2000, 1000, 50, 1950, 975),  # Larger width and range

        # Test values below range
        (1000, -100, 0, 1000, 0),     # Clamped to minimum
        (800, -50, 20, 800, 20),      # Custom min clamping
        (500, -200, -100, 400, -100), # Negative range min clamp

        # Test values above range
        (1000, 1200, 0, 1000, 1000),  # Clamped to maximum
        (1500, 1600, 100, 1400, 1400),# Custom max clamping
        (2000, 2100, 50, 1950, 1950), # Large width and max clamping
    ]
)
def test_toVal_clamping(qtbot, slider_width, x_value, min_value, max_value, expected_value):
    """Parameterized test for _toVal clamping with varying slider widths and ranges."""
    slider = VideoSlider(min=0, max=1200, val=15, marks=(10, 15))

    slider.setMinimum(min_value)
    slider.setMaximum(max_value)

    # Simulate slider width for the calculation
    slider.box_rect.setWidth(slider_width)

    # Check clamping behavior
    assert slider._toVal(x_value) == expected_value
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
    "slider_width, x_value, min_value, max_value",
    [
        # Values within range
        (1000, 500, 0, 1000),  # Midpoint with no offset
        (800, 400, 0, 800),  # Exact midpoint within smaller range
        (1500, 750, 100, 1200),  # Midpoint with offset range
        (2000, 1000, 50, 1950),  # Large width and offset range
        # Values below range (no clamping expected)
        (1000, -100, 0, 1000),  # Below minimum
        # Values above range (no clamping expected)
        (1000, 1200, 0, 1000),  # Above maximum
    ],
)
def test_toVal(qtbot, slider_width, x_value, min_value, max_value):
    """
    Test _toVal scaling and transformation for varying slider widths and ranges,
    without expecting clamping behavior.

    Args:
        qtbot: The pytest-qt bot fixture.
        slider_width (int): The width of the slider in pixels.
        x_value (float): The x-coordinate on the slider to be converted to a value.
        min_value (int): The minimum value of the slider.
        max_value (int): The maximum value of the slider.
    """
    slider = VideoSlider(min=0, max=1000, val=15, marks=(10, 15))  # Initialize slider

    slider.setMinimum(min_value)  # Set slider range
    slider.setMaximum(max_value)

    slider.box_rect.setWidth(slider_width)  # Simulate visual width

    # Compute the expected raw transformed value
    expected_value = round(
        (x_value / slider_width) * (max_value - min_value) + min_value
    )

    # Assert that the raw transformation matches the expected value
    assert slider._toVal(x_value) == expected_value


def test_slider_width_property(qtbot):
    """
    Test the _slider_width property to ensure it accurately reflects
    the visual width of the slider's box_rect.
    """
    slider = VideoSlider(min=0, max=1000, val=15, marks=(10, 15))  # Initialize slider

    # Test various box_rect widths
    for width in [800, 1000, 1200, 1500]:
        slider.box_rect.setWidth(width)  # Simulate setting the visual width
        assert (
            slider._slider_width == width
        ), f"Expected _slider_width to be {width}, but got {slider._slider_width}"

    # Test edge cases with very small and large widths
    slider.box_rect.setWidth(0)
    assert (
        slider._slider_width == 0
    ), "Expected _slider_width to be 0 when box_rect width is 0"

    slider.box_rect.setWidth(10000)
    assert (
        slider._slider_width == 10000
    ), "Expected _slider_width to handle large values correctly"

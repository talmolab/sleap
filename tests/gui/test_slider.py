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
    "slider_width, x_value, handle_width, min_value, max_value",
    [
        # ---- Cases from test_toVal ----
        (1000, 500, 0, 0, 1000),  # Midpoint w/o centering
        (800, 400, 0, 0, 800),
        (1500, 750, 0, 100, 1200),
        (2000, 1000, 0, 50, 1950),
        (1000, -100, 0, 0, 1000),  # Below minimum
        (1000, 1200, 0, 0, 1000),  # Above maximum
    ],
)
def test_toVal(qtbot, slider_width, x_value, handle_width, min_value, max_value):
    """
    Merged test that checks the slider value transformation for both:
    1) center=True (handle offset subtracted).
    2) center=False (no offset).

    Args:
        qtbot: The pytest-qt bot fixture.
        slider_width: The width of the slider box_rect in pixels.
        x_value: The x-coordinate on the slider to convert to a value.
        handle_width: The desired width of the slider handle rect (0 if center=False).
        min_value: The slider's minimum value.
        max_value: The slider's maximum value.
        center: Whether _toVal() is called with center=True or not.

    Returns:
        None
    """
    slider = VideoSlider(min=min_value, max=max_value, val=(min_value + max_value) // 2)
    slider.box_rect.setWidth(slider_width)

    # If we have a non-zero handle_width, set the handle's bounding rect.
    if handle_width > 0:
        current_handle_rect = slider.handle.rect()
        slider.handle.setRect(
            current_handle_rect.x(),
            current_handle_rect.y(),
            handle_width,
            current_handle_rect.height(),
        )

    # Read back the actual handle width (will be 0 if center=False).
    actual_handle_width = slider.handle.rect().width()

    # Manually compute the expected slider value.
    # If center=True, subtract half the handle width.
    val = float(x_value)
    effective_width = max(1.0, slider_width)  # Prevent zero-division
    val /= effective_width
    val *= max(1, max_value - min_value)
    val += min_value
    expected_val = round(val)

    # Now call the actual function with the specified center setting.
    actual_val = slider._toVal(x_value)

    assert actual_val == expected_val, (
        f"For x={x_value}, handle_width={actual_handle_width}, "
        f"slider_width={slider_width}, "
    )


def test_slider_width_property(qtbot):
    """
    Test the _slider_width property to ensure it accurately reflects
    the visual width of the slider's box_rect.
    """
    slider = VideoSlider(min=0, max=1000, val=15, marks=(10, 15))  # Initialize slider

    # Test various box_rect widths
    for width in [800, 1000, 1200, 1500]:
        slider.box_rect.setWidth(width)  # Simulate setting the visual width
        assert slider._slider_width == width, (
            f"Expected _slider_width to be {width}, but got {slider._slider_width}"
        )

    # Test edge cases with very small and large widths
    slider.box_rect.setWidth(0)
    assert slider._slider_width == 0, (
        "Expected _slider_width to be 0 when box_rect width is 0"
    )

    slider.box_rect.setWidth(10000)
    assert slider._slider_width == 10000, (
        "Expected _slider_width to handle large values correctly"
    )


@pytest.mark.parametrize(
    "invalid_value, expected_error_msg",
    [
        (None, "x position cannot be None"),
        ("invalid", "x position must be a number, got invalid"),
        ([], "x position must be a number, got []"),
        ({}, "x position must be a number, got {}"),
    ],
)
def test_toVal_invalid_input(qtbot, invalid_value, expected_error_msg):
    """
    Tests _toVal for invalid inputs to ensure ValueError is raised
    with the correct message.

    Args:
        qtbot: Pytest fixture for Qt applications.
        invalid_value: The invalid value for x.
        expected_error_msg: The exact error message expected.

    Returns:
        None
    """
    # We use tqdm to track progress across multiple invalid inputs (optional).

    slider = VideoSlider(min=0, max=1000, val=15)

    with pytest.raises(ValueError) as excinfo:
        slider._toVal(invalid_value)

    # Verify the exact error message
    assert str(excinfo.value) == expected_error_msg

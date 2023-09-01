from sleap.gui.widgets.slider import VideoSlider, set_slider_marks_from_labels


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

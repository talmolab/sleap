from sleap.gui.slider import VideoSlider

def test_slider(qtbot, centered_pair_predictions):
    
    labels = centered_pair_predictions
    
    slider = VideoSlider(min=0, max=1200, val=15, marks=(10,15))
    
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
    slider.setTracks(20)
    assert slider.maximumHeight() != initial_height
    
    slider.setTracksFromLabels(labels, labels.videos[0])
    assert len(slider.getMarks()) == 2274

    slider.moveSelectionAnchor(5, 5)
    slider.releaseSelectionAnchor(100, 15)
    assert slider.getSelection() == (31, 619)

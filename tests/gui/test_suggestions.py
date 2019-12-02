from sleap.gui.suggestions import VideoFrameSuggestions


def test_velocity_suggestions(centered_pair_predictions):
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params=dict(method="velocity", node="", threshold=0.5),
    )
    assert len(suggestions) == 12
    assert suggestions[0].frame_idx == 80
    assert suggestions[1].frame_idx == 145

from sleap.gui.suggestions import VideoFrameSuggestions
from sleap.io.dataset import Labels


def test_velocity_suggestions(centered_pair_predictions):
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params=dict(method="velocity", node="", threshold=0.5),
    )
    assert len(suggestions) == 45
    assert suggestions[0].frame_idx == 21
    assert suggestions[1].frame_idx == 45


def test_frame_increment(centered_pair_predictions: Labels):

    # Testing videos that have less frames than desired Samples per Video value using stride method.
    # Expected result is one suggested frame.

    vid_frames = centered_pair_predictions.video.num_frames
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "method": "sample",
            "per_video": 2 * vid_frames,
            "sampling_method": "stride",
        },
    )
    assert len(suggestions) == 1

    # Testing typical videos that have more frames than desired Samples per Video value using stride method.
    # Expected result is the desired Samples per Video number of frames.

    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "method": "sample",
            "per_video": 20,
            "sampling_method": "stride",
        },
    )
    assert len(suggestions) == 20

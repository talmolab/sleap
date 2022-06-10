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

    # Testing videos that have less frames than desired Samples per Video (stride)
    # Expected result is there should be n suggestions where n is equal to the frames
    # in the video.
    vid_frames = centered_pair_predictions.video.num_frames
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos
            "method": "sample",
            "per_video": 2 * vid_frames,
            "sampling_method": "stride",
        },
    )
    assert len(suggestions) == vid_frames

    # Testing typical videos that have more frames than Samples per Video (stride)
    # Expected result is the desired Samples per Video number of frames.
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos
            "method": "sample",
            "per_video": 20,
            "sampling_method": "stride",
        },
    )
    assert len(suggestions) == 20

    # Testing videos that have less frames than desired Samples per Video (random)
    # Expected result is there should be n suggestions where n is equal to the frames
    # in the video.
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos
            "method": "sample",
            "per_video": 2 * vid_frames,
            "sampling_method": "random",
        },
    )
    assert len(suggestions) == vid_frames

    # Testing typical videos that have more frames than Samples per Video (random)
    # Expected result is the desired Samples per Video number of frames.
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos
            "method": "sample",
            "per_video": 20,
            "sampling_method": "random",
        },
    )
    assert len(suggestions) == 20

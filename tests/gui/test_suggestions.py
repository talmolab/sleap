from sleap.gui.suggestions import VideoFrameSuggestions
from sleap.io.dataset import Labels
from sleap.io.video import Video


def test_velocity_suggestions(centered_pair_predictions):
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params=dict(
            videos=centered_pair_predictions.videos,
            method="velocity",
            node="",
            threshold=0.5,
        ),
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
            "videos": centered_pair_predictions.videos,
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
            "videos": centered_pair_predictions.videos,
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
            "videos": centered_pair_predictions.videos,
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
            "videos": centered_pair_predictions.videos,
            "method": "sample",
            "per_video": 20,
            "sampling_method": "random",
        },
    )
    assert len(suggestions) == 20
    print(centered_pair_predictions.videos)


def test_video_selection(centered_pair_predictions: Labels):

    # Testing the functionality of choosing a specific video in a project and
    # only generating suggestions for the video

    centered_pair_predictions.add_video(Video.from_filename(filename="test.mp4"))
    # Testing suggestion generation from Image Features
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[0]],
            "method": "image features",
            "per_video": 5,
            "sample_method": "stride",
            "scale": 1,
            "merge_video_features": "per_video",
            "feature_type": "raw_images",
            "pca_components": 5,
            "n_clusters": 5,
            "per_cluster": 5,
        },
    )
    for i in range(len(suggestions)):
        # Confirming every suggestion is only for the video that is chosen and no other videos
        assert suggestions[i].video == centered_pair_predictions.videos[0]

    # Testing suggestion generation from Sample
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[0]],
            "method": "sample",
            "per_video": 3,
            "sampling_method": "random",
        },
    )

    for i in range(len(suggestions)):
        # Confirming every suggestion is only for the video that is chosen and no other videos
        assert suggestions[i].video == centered_pair_predictions.videos[0]

    # Testing suggestion generation from prediction score
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[0]],
            "method": "prediction_score",
            "score_limit": 2,
            "instance_limit": 1,
        },
    )

    for i in range(len(suggestions)):
        # Confirming every suggestion is only for the video that is chosen and no other videos
        assert suggestions[i].video == centered_pair_predictions.videos[0]

    # Testing suggestion generation from velocity
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[0]],
            "method": "velocity",
            "node": "",
            "threshold": 0.8,
        },
    )
    for i in range(len(suggestions)):
        # Confirming every suggestion is only for the video that is chosen and no other videos
        assert suggestions[i].video == centered_pair_predictions.videos[0]

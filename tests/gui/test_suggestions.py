from typing import List
import pytest
from sleap.gui.suggestions import SuggestionFrame, VideoFrameSuggestions
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import LabeledFrame, PredictedInstance, Track, PredictedPoint
from sleap.io.dataset import Labels
from sleap.skeleton import Skeleton


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
    # assert suggestions[0].frame_idx == 21
    # assert suggestions[1].frame_idx == 45
    assert suggestions[0].frame_idx == 131
    assert suggestions[1].frame_idx == 765


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


@pytest.mark.parametrize(
    "params",
    [
        {
            "per_video": 2,
            "method": "sample",
            "sample_method": "random",
        },
        {
            "per_video": 2,
            "method": "sample",
            "sample_method": "stride",
        },
        {
            "per_video": 2,
            "method": "image features",
            "sample_method": "random",
            "scale": 1,
            "merge_video_features": "across all videos",
            "feature_type": "raw",
            "pca_components": 2,
            "n_clusters": 1,
            "per_cluster": 1,
        },
        {
            "per_video": 2,
            "method": "prediction_score",
            "score_limit": 1.1,
            "instance_limit": 1,
        },
        {
            "per_video": 2,
            "method": "velocity",
            "node": "a",
            "threshold": 0.1,
        },
    ],
)
def test_unqiue_suggestions(params, small_robot_image_vid):
    """Ensure only unique suggestions are returned and that suggestions are appended."""

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    track_a = Track(0, "a")

    instances1 = []
    instances1.extend(
        [
            PredictedInstance(
                skeleton=skeleton,
                score=1,
                track=track_a,
                points=dict(
                    a=PredictedPoint(1, 3, score=0.5), b=PredictedPoint(7, 9, score=0.5)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(2, 4, score=0.7), b=PredictedPoint(6, 5, score=0.7)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(7, 3, score=0.7),
                    b=PredictedPoint(8, 10, score=0.7),
                ),
            ),
        ]
    )

    instances2 = []
    instances2.extend(
        [
            PredictedInstance(
                skeleton=skeleton,
                score=1,
                track=track_a,
                points=dict(
                    a=PredictedPoint(2, 4, score=0.5), b=PredictedPoint(7, 9, score=0.5)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(3, 7, score=0.7), b=PredictedPoint(6, 5, score=0.7)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(8, 1, score=0.7),
                    b=PredictedPoint(8, 10, score=0.7),
                ),
            ),
        ]
    )

    instances3 = []
    instances3.extend(
        [
            PredictedInstance(
                skeleton=skeleton,
                score=1,
                track=track_a,
                points=dict(
                    a=PredictedPoint(8, 9, score=0.5), b=PredictedPoint(7, 9, score=0.5)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(2, 3, score=0.7), b=PredictedPoint(6, 5, score=0.7)
                ),
            ),
            PredictedInstance(
                skeleton=skeleton,
                score=5,
                track=track_a,
                points=dict(
                    a=PredictedPoint(5, 9, score=0.7),
                    b=PredictedPoint(8, 10, score=0.7),
                ),
            ),
        ]
    )

    labeled_frame1 = LabeledFrame(
        small_robot_image_vid, frame_idx=0, instances=instances1
    )
    labeled_frame2 = LabeledFrame(
        small_robot_image_vid, frame_idx=1, instances=instances2
    )
    labeled_frame3 = LabeledFrame(
        small_robot_image_vid, frame_idx=2, instances=instances3
    )

    lfs = [labeled_frame1, labeled_frame2, labeled_frame3]
    labels = Labels(lfs)
    params["videos"] = labels.videos

    suggestions = VideoFrameSuggestions.suggest(labels=labels, params=params)
    labels.suggestions.extend(suggestions)

    print("old_suggestions", suggestions)

    new_suggestions = VideoFrameSuggestions.suggest(labels=labels, params=params)

    print("new_suggestions", new_suggestions)

    if params["method"] == "image features":
        assert len(suggestions) == 1
        assert len(new_suggestions) == 1
    elif params["method"] == "prediction_score":
        assert len(suggestions) == (
            (params["per_video"] + 1) * params["instance_limit"]
        )
        assert len(new_suggestions) == 0
    elif params["method"] == "velocity":
        assert len(suggestions) == 1
        assert len(new_suggestions) == 0
    else:
        assert len(suggestions) == params["per_video"]
        assert len(new_suggestions) == 1


def test_basic_append_suggestions(small_robot_image_vid):
    """Ensure only unique suggestions are returned and that suggestions are appended."""

    def assert_suggestions_unique(
        labels: Labels, new_suggestions: List[SuggestionFrame]
    ):
        for sugg in labels.suggestions:
            for new_sugg in new_suggestions:
                assert sugg.frame_idx != new_sugg.frame_idx

    labels = Labels(videos=[small_robot_image_vid])

    # Generate some suggestions
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 3,
            "method": "sample",
            "sample_method": "stride",
            "videos": labels.videos,
        },
    )
    assert len(suggestions) == 3
    labels.suggestions.extend(suggestions[0:2])

    # Sample with stride method
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 3,
            "method": "sample",
            "sample_method": "stride",
            "videos": labels.videos,
        },
    )

    # Check that stride method returns only unique suggestions
    assert len(suggestions) == 1
    assert_suggestions_unique(labels, suggestions)
    labels.suggestions.extend(suggestions)

    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 3,
            "method": "sample",
            "sample_method": "stride",
            "videos": labels.videos,
        },
    )
    assert len(suggestions) == 0
    assert_suggestions_unique(labels, suggestions)

    # Sample with random method
    labels.suggestions.pop()
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 3,
            "method": "sample",
            "sample_method": "random",
            "videos": labels.videos,
        },
    )

    # Check that random method only returns unique suggestions
    assert len(suggestions) == 1
    assert_suggestions_unique(labels, suggestions)
    labels.suggestions.extend(suggestions)

    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 3,
            "method": "sample",
            "sample_method": "random",
            "videos": labels.videos,
        },
    )
    assert len(suggestions) == 0
    assert_suggestions_unique(labels, suggestions)

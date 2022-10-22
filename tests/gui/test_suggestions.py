from typing import List
import pytest
from sqlalchemy import true
from sleap.gui.suggestions import SuggestionFrame, VideoFrameSuggestions
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import LabeledFrame, PredictedInstance, Track, PredictedPoint
from sleap.io.dataset import Labels
from sleap.skeleton import Skeleton
import numpy as np


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


def test_video_selection(
    centered_pair_predictions: Labels, small_robot_3_frame_vid: Video
):
    # Testing the functionality of choosing a specific video in a project and
    # only generating suggestions for the video

    centered_pair_predictions.add_video(small_robot_3_frame_vid)
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
            "instance_limit_upper": 2,
            "instance_limit_lower": 1,
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

    # Ensure video target works given suggestions from another video already exist
    centered_pair_predictions.set_suggestions(suggestions)
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[1]],
            "method": "sample",
            "per_video": 3,
            "sampling_method": "random",
        },
    )

    # Testing suggestion generation from frame chunk targeting selected video or all videos
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[1]],
            "method": "frame_chunk",
            "frame_from": 1,
            "frame_to": 3,
        },
    )
    # Verify that frame 1-3 of video 1 are selected
    for i in range(len(suggestions)):
        assert suggestions[i].video == centered_pair_predictions.videos[1]

    # Testing suggestion generation from frame chunk targeting all videos and frame to exceeding one video
    centered_pair_predictions_copy = centered_pair_predictions
    # Clear existing suggestions so that generated suggestions will be kept intact at the uniqueness check step
    centered_pair_predictions_copy.suggestions.clear()
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions_copy,
        params={
            "videos": centered_pair_predictions_copy.videos,
            "method": "frame_chunk",
            "frame_from": 1,
            "frame_to": 1000,
        },
    )
    # Verify that frame 1-1000 of video 0 and 1-3 of video 1 are selected
    assert len(suggestions) == 1003

    correct_sugg = True
    for i in range(len(suggestions)):
        if (
            suggestions[i].video == centered_pair_predictions.videos[1]
            and suggestions[i].frame_idx > 2
        ):
            correct_sugg = False
            break
        elif (
            suggestions[i].video == centered_pair_predictions.videos[0]
            and suggestions[i].frame_idx > 999
        ):
            correct_sugg = False
            break

    assert correct_sugg

    # Testing when range exceeds video 1, only frames from video 0 are selected
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos,
            "method": "frame_chunk",
            "frame_from": 501,
            "frame_to": 600,
        },
    )
    # Verify that frame 501-600 of video 0 are selected
    assert len(suggestions) == 100
    correct_sugg = True
    for i in range(len(suggestions)):
        if suggestions[i].video == centered_pair_predictions.videos[1]:
            correct_sugg = False
            break
        elif suggestions[i].frame_idx < 500 or suggestions[i].frame_idx > 599:
            correct_sugg = False
            break
    assert correct_sugg


def assert_suggestions_unique(labels: Labels, new_suggestions: List[SuggestionFrame]):
    for sugg in labels.suggestions:
        for new_sugg in new_suggestions:
            assert sugg.frame_idx != new_sugg.frame_idx


def test_append_suggestions(small_robot_3_frame_vid: Video, stickman: Skeleton):
    """Ensure only unique suggestions are returned and that suggestions are appended."""
    track_a = Track(0, "a")
    track_b = Track(0, "b")

    lfs = [
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=0,
            instances=[
                PredictedInstance(
                    skeleton=stickman,
                    score=0.1,
                    points=dict(
                        head=PredictedPoint(1, 2, score=0.5),
                        neck=PredictedPoint(2, 3, score=0.5),
                    ),
                    track=track_a,
                ),
                PredictedInstance(
                    skeleton=stickman,
                    score=0.5,
                    points=dict(
                        head=PredictedPoint(11, 12, score=0.5),
                        neck=PredictedPoint(12, 13, score=0.5),
                    ),
                    track=track_b,
                ),
            ],
        ),
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=1,
            instances=[
                PredictedInstance(
                    skeleton=stickman,
                    score=0.1,
                    points=dict(
                        head=PredictedPoint(2, 1, score=0.5),
                        neck=PredictedPoint(3, 2, score=0.5),
                    ),
                    track=track_a,
                ),
                PredictedInstance(
                    skeleton=stickman,
                    score=0.5,
                    points=dict(
                        head=PredictedPoint(2, 1, score=0.5),
                        neck=PredictedPoint(3, 2, score=0.5),
                    ),
                    track=track_b,
                ),
            ],
        ),
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=2,
            instances=[
                PredictedInstance(
                    skeleton=stickman,
                    score=0.5,
                    points=dict(
                        head=PredictedPoint(11, 12, score=0.5),
                        neck=PredictedPoint(12, 13, score=0.5),
                    ),
                    track=track_a,
                ),
                PredictedInstance(
                    skeleton=stickman,
                    score=0.5,
                    points=dict(
                        head=PredictedPoint(1, 2, score=0.5),
                        neck=PredictedPoint(2, 3, score=0.5),
                    ),
                    track=track_b,
                ),
            ],
        ),
    ]
    labels = Labels(lfs)

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
    labels.append_suggestions(suggestions[0:2])

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
    labels.append_suggestions(suggestions)

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
    labels.append_suggestions(suggestions)

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

    # Generate some suggestions using image features
    labels.suggestions.pop()
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "per_video": 2,
            "method": "image features",
            "sample_method": "random",
            "scale": 1,
            "merge_video_features": "across all videos",
            "feature_type": "raw",
            "pca_components": 2,
            "n_clusters": 1,
            "per_cluster": 1,
            "videos": labels.videos,
        },
    )

    # Test that image features returns only unique suggestions
    assert_suggestions_unique(labels, suggestions)

    # Generate suggestions using prediction score
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "method": "prediction_score",
            "score_limit": 1,
            "instance_limit_upper": 2,
            "instance_limit_lower": 1,
            "videos": labels.videos,
        },
    )
    assert_suggestions_unique(labels, suggestions)

    # Generate suggestions using velocity
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "method": "velocity",
            "node": "head",
            "threshold": 0.1,
            "videos": labels.videos,
        },
    )
    assert_suggestions_unique(labels, suggestions)


def test_limits_prediction_score(centered_pair_predictions: Labels):
    """Testing suggestion generation using instance limits and prediction score."""
    labels = centered_pair_predictions
    score_limit = 20
    instance_lower_limit = 3
    instance_upper_limit = 3

    # Generate suggestions
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "videos": labels.videos,
            "method": "prediction_score",
            "score_limit": score_limit,
            "instance_limit_upper": instance_upper_limit,
            "instance_limit_lower": instance_lower_limit,
        },
    )

    # Confirming every suggested frame meets criteria
    for sugg in suggestions:
        lf = labels.get((sugg.video, sugg.frame_idx))
        pred_instances = [
            inst for inst in lf.instances_to_show if isinstance(inst, PredictedInstance)
        ]
        n_instance_below_score = np.nansum(
            [True for inst in pred_instances if inst.score <= score_limit]
        )
        assert n_instance_below_score >= instance_lower_limit
        assert n_instance_below_score <= instance_upper_limit

    # Confirming all frames meeting the criteria are captured
    def check_all_predicted_instances(sugg, labels):
        lfs = labels.labeled_frames
        for lf in lfs:
            pred_instances = [
                inst
                for inst in lf.instances_to_show
                if isinstance(inst, PredictedInstance)
            ]
            n_instance_below_score = np.nansum(
                [True for inst in pred_instances if inst.score <= score_limit]
            )
            if (
                n_instance_below_score <= instance_upper_limit
                and n_instance_below_score >= instance_lower_limit
            ):
                temp_suggest = SuggestionFrame(
                    labels.video, pred_instances[0].frame_idx
                )
                if not (temp_suggest in sugg):
                    return False

        return True

    suggestions_correct = check_all_predicted_instances(suggestions, labels)
    assert suggestions_correct

    # Generate suggestions using frame chunk
    suggestions = VideoFrameSuggestions.suggest(
        labels=labels,
        params={
            "method": "frame_chunk",
            "frame_from": 1,
            "frame_to": 15,
            "videos": labels.videos,
        },
    )
    assert_suggestions_unique(labels, suggestions)

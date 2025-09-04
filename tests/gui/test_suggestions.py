from typing import List
from sleap.gui.suggestions import VideoFrameSuggestions
from sleap_io import Video
from sleap_io import LabeledFrame, Labels, SuggestionFrame
from sleap_io.model.instance import PredictedInstance, Track
from sleap_io import Skeleton
from sleap.sleap_io_adaptors.lf_labels_utils import labels_get, get_instances_to_show
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


def test_max_point_displacement_suggestions(centered_pair_predictions):
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params=dict(
            videos=centered_pair_predictions.videos,
            method="max_point_displacement",
            displacement_threshold=6,
        ),
    )
    assert len(suggestions) == 19
    assert suggestions[0].frame_idx == 28
    assert suggestions[1].frame_idx == 82


def test_frame_increment(centered_pair_predictions: Labels):
    # Testing videos that have less frames than desired Samples per Video (stride)
    # Expected result is there should be n suggestions where n is equal to the frames
    # in the video.
    vid_frames = len(centered_pair_predictions.video)
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

    from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video

    labels_add_video(centered_pair_predictions, small_robot_3_frame_vid)
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
        # Confirming every suggestion is only for the video that is chosen and no other
        # videos
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
        # Confirming every suggestion is only for the video that is chosen and no other
        # videos
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
        # Confirming every suggestion is only for the video that is chosen and no other
        # videos
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
        # Confirming every suggestion is only for the video that is chosen and no other
        # videos
        assert suggestions[i].video == centered_pair_predictions.videos[0]

    # Ensure video target works given suggestions from another video already exist
    centered_pair_predictions.suggestions = suggestions
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": [centered_pair_predictions.videos[1]],
            "method": "sample",
            "per_video": 3,
            "sampling_method": "random",
        },
    )

    # Testing suggestion generation from frame chunk targeting selected video or all
    # videos
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

    # Testing suggestion generation from frame chunk targeting all videos
    # Clear existing suggestions so that generated suggestions will be kept intact at
    # the uniqueness check step
    centered_pair_predictions.suggestions = []
    suggestions = VideoFrameSuggestions.suggest(
        labels=centered_pair_predictions,
        params={
            "videos": centered_pair_predictions.videos,
            "method": "frame_chunk",
            "frame_from": 1,
            "frame_to": 20,
        },
    )
    # Verify that frame 1-20 of video 0 and 1-3 of video 1 are selected
    assert len(suggestions) == 23

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
            and suggestions[i].frame_idx > 19
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
            "frame_to": 510,
        },
    )
    # Verify that frame 501-600 of video 0 are selected
    assert len(suggestions) == 10
    correct_sugg = True
    for i in range(len(suggestions)):
        if suggestions[i].video == centered_pair_predictions.videos[1]:
            correct_sugg = False
            break
        elif suggestions[i].frame_idx < 500 or suggestions[i].frame_idx > 509:
            correct_sugg = False
            break
    assert correct_sugg


def assert_suggestions_unique(labels: Labels, new_suggestions: List[SuggestionFrame]):
    for sugg in labels.suggestions:
        for new_sugg in new_suggestions:
            assert sugg.frame_idx != new_sugg.frame_idx


def test_append_suggestions(small_robot_3_frame_vid: Video, stickman: Skeleton):
    """Ensure only unique suggestions are returned and that suggestions are appended."""
    import numpy as np

    def _create_points(point_dict, skeleton):
        """Helper to convert old dict format to numpy arrays"""
        points_array = np.full((len(skeleton), 2), np.nan, dtype=np.float32)
        point_scores = np.full(len(skeleton), 0.0, dtype=np.float32)
        for node_name, data in point_dict.items():
            node_idx = skeleton.node_names.index(node_name)
            xy_coords = data[0] if isinstance(data[0], (list, tuple)) else data[:2]
            score = (
                data[1] if len(data) > 1 and isinstance(data[0], (list, tuple)) else 0.5
            )
            points_array[node_idx] = xy_coords
            point_scores[node_idx] = score
        return points_array, point_scores

    track_a = Track(name="a")
    track_b = Track(name="b")

    # Frame 0 instances
    points_a0, scores_a0 = _create_points(
        {"head": ([1, 2], 0.5), "neck": ([2, 3], 0.5)}, stickman
    )
    points_b0, scores_b0 = _create_points(
        {"head": ([11, 12], 0.5), "neck": ([12, 13], 0.5)}, stickman
    )

    # Frame 1 instances
    points_a1, scores_a1 = _create_points(
        {"head": ([2, 1], 0.5), "neck": ([3, 2], 0.5)}, stickman
    )
    points_b1, scores_b1 = _create_points(
        {"head": ([2, 1], 0.5), "neck": ([3, 2], 0.5)}, stickman
    )

    # Frame 2 instances
    points_a2, scores_a2 = _create_points(
        {"head": ([11, 12], 0.5), "neck": ([12, 13], 0.5)}, stickman
    )
    points_b2, scores_b2 = _create_points(
        {"head": ([1, 2], 0.5), "neck": ([2, 3], 0.5)}, stickman
    )

    lfs = [
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=0,
            instances=[
                PredictedInstance.from_numpy(
                    points_a0,
                    skeleton=stickman,
                    point_scores=scores_a0,
                    score=0.1,
                    track=track_a,
                ),
                PredictedInstance.from_numpy(
                    points_b0,
                    skeleton=stickman,
                    point_scores=scores_b0,
                    score=0.5,
                    track=track_b,
                ),
            ],
        ),
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=1,
            instances=[
                PredictedInstance.from_numpy(
                    points_a1,
                    skeleton=stickman,
                    point_scores=scores_a1,
                    score=0.1,
                    track=track_a,
                ),
                PredictedInstance.from_numpy(
                    points_b1,
                    skeleton=stickman,
                    point_scores=scores_b1,
                    score=0.5,
                    track=track_b,
                ),
            ],
        ),
        LabeledFrame(
            small_robot_3_frame_vid,
            frame_idx=2,
            instances=[
                PredictedInstance.from_numpy(
                    points_a2,
                    skeleton=stickman,
                    point_scores=scores_a2,
                    score=0.5,
                    track=track_a,
                ),
                PredictedInstance.from_numpy(
                    points_b2,
                    skeleton=stickman,
                    point_scores=scores_b2,
                    score=0.5,
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
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_append_suggestions

    labels_append_suggestions(labels, suggestions[0:2])

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
    labels_append_suggestions(labels, suggestions)

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
    labels_append_suggestions(labels, suggestions)

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
        lf = labels_get(labels, (sugg.video, sugg.frame_idx))
        pred_instances = [
            inst
            for inst in get_instances_to_show(lf)
            if isinstance(inst, PredictedInstance)
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
                for inst in get_instances_to_show(lf)
                if isinstance(inst, PredictedInstance)
            ]
            n_instance_below_score = np.nansum(
                [True for inst in pred_instances if inst.score <= score_limit]
            )
            if (
                n_instance_below_score <= instance_upper_limit
                and n_instance_below_score >= instance_lower_limit
            ):
                temp_suggest = SuggestionFrame(labels.video, lf.frame_idx)
                if temp_suggest not in sugg:
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

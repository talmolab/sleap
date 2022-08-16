"""
Module for generating lists of suggested frames (for labeling or reviewing).
"""

import attr
import numpy as np
import random

from typing import List, Optional, Union

from sleap.io.video import Video
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)

GroupType = int


@attr.s(auto_attribs=True, slots=True)
class SuggestionFrame:
    """Object for storing a single suggested frame item."""

    video: Video
    frame_idx: int
    group: Optional[GroupType] = None


class VideoFrameSuggestions(object):
    """
    Class for generating lists of suggested frames.

    Implements various algorithms as methods:
    * sample (either random or evenly spaced sample frames from each video)
    * image features (raw images/brisk -> pca -> k-means)
    * prediction_score (frames with number of instances below specified score)

    Each of algorithm method should accept `labels`; other parameters will be
    passed from the `params` dict given to :meth:`suggest`.
    """

    @classmethod
    def suggest(cls, params: dict, labels: "Labels" = None) -> List[SuggestionFrame]:
        """
        This is the main entry point for generating lists of suggested frames.

        Args:
            params: A dictionary with all params to control how we generate
                suggestions, minimally this will have a "method" key with
                the name of one of the class methods.
            labels: A `Labels` object for which we are generating suggestions.

        Returns:
            List of `SuggestionFrame` objects.
        """

        # map from method param value to corresponding class method
        method_functions = dict(
            sample=cls.basic_sample_suggestion_method,
            image_features=cls.image_feature_based_method,
            prediction_score=cls.prediction_score,
            velocity=cls.velocity,
        )

        method = str.replace(params["method"], " ", "_")
        if method_functions.get(method, None) is not None:
            return method_functions[method](labels=labels, **params)
        else:
            print(f"No {method} method found for generating suggestions.")
            return []

    # Functions corresponding to "method" param

    @classmethod
    def basic_sample_suggestion_method(
        cls,
        labels,
        videos: List[Video],
        per_video: int = 20,
        sampling_method: str = "random",
        **kwargs,
    ):
        """Method to generate suggestions by taking strides through video."""

        suggestions = []
        sugg_idx_dict = {video: [] for video in videos}

        sugg_idx = [(sugg.video, sugg.frame_idx) for sugg in labels.suggestions]

        # Turn list of tuples into dict where first argument in tuple is dict key
        for vid, frame in sugg_idx:
            sugg_idx_dict[vid].append(frame)

        for video in videos:
            vid_idx = list(range(video.frames))
            vid_sugg_idx = sugg_idx_dict[video]
            unique_idx = set(vid_idx) ^ set(vid_sugg_idx)

            if sampling_method == "stride":
                frame_increment = video.frames // per_video
                frame_increment = 1 if frame_increment == 0 else frame_increment
                vid_suggestions = list(range(0, len(unique_idx), frame_increment))[
                    :per_video
                ]
            else:
                # random sampling
                frames_num = per_video
                frames_num = video.frames if (frames_num > video.frames) else frames_num
                if len(unique_idx) == 1:
                    vid_suggestions = list(unique_idx)
                else:
                    vid_suggestions = random.sample(unique_idx, frames_num)

            group = labels.videos.index(video)
            suggestions.extend(
                cls.idx_list_to_frame_list(vid_suggestions, video, group)
            )

        return suggestions

    @classmethod
    def image_feature_based_method(
        cls,
        labels,
        videos: List[Video],
        per_video,
        sample_method,
        scale,
        merge_video_features,
        feature_type,
        pca_components,
        n_clusters,
        per_cluster,
        **kwargs,
    ):
        """
        Method to generate suggestions based on image features.

        This is a wrapper for `feature_suggestion_pipeline` implemented in
        `sleap.info.feature_suggestions`.
        """

        brisk_threshold = kwargs.get("brisk_threshold", 80)
        vocab_size = kwargs.get("vocab_size", 20)

        pipeline = FeatureSuggestionPipeline(
            per_video=per_video,
            scale=scale,
            sample_method=sample_method,
            feature_type=feature_type,
            brisk_threshold=brisk_threshold,
            vocab_size=vocab_size,
            n_components=pca_components,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
        )

        if merge_video_features == "across all videos":
            # Run single pipeline with all videos
            suggestions = pipeline.get_suggestion_frames(videos=videos)
        else:
            # Run pipeline separately (in parallel) for each video
            suggestions = ParallelFeaturePipeline.run(pipeline, videos)

        prev_idx = [
            sugg.frame_idx for sugg in labels.suggestions  # for sugg in sugg_lst
        ]

        vid_idx = [frame.frame_idx for frame in labels.labeled_frames]

        unique_idx = set(vid_idx) ^ set(prev_idx)

        unique_suggestions = [
            sugg for sugg in suggestions if sugg.frame_idx in unique_idx
        ]

        return unique_suggestions

    @classmethod
    def prediction_score(
        cls,
        labels: "Labels",
        videos: List[Video],
        score_limit,
        instance_limit,
        **kwargs,
    ):
        """
        Method to generate suggestions for proofreading frames with low score.
        """
        score_limit = float(score_limit)
        instance_limit = int(instance_limit)

        suggestions = []
        for video in videos:
            suggestions.extend(
                cls._prediction_score_video(video, labels, score_limit, instance_limit)
            )
        return suggestions

    @classmethod
    def _prediction_score_video(
        cls, video: "Video", labels: "Labels", score_limit: float, instance_limit: int
    ):
        lfs = labels.find(video)
        frames = len(lfs)
        idxs = np.ndarray((frames), dtype="int")
        scores = np.full((frames, instance_limit), 100.0, dtype="float")

        # Build matrix with scores for instances in frames
        for i, lf in enumerate(lfs):
            # Scores from instances in frame
            frame_scores = [inst.score for inst in lf if hasattr(inst, "score")]
            # Just get the lowest scores
            if len(frame_scores) > instance_limit:
                frame_scores = sorted(frame_scores)[:instance_limit]
            # Add to matrix
            scores[i, : len(frame_scores)] = frame_scores
            idxs[i] = lf.frame_idx

        # Find instances below score of <score_limit>
        low_instances = np.nansum(scores < score_limit, axis=1)

        # Find all the frames with at least <instance_limit> low scoring instances
        result = idxs[low_instances >= instance_limit].tolist()

        # Generate unique suggestions
        prev_idx = [
            sugg.frame_idx for sugg in labels.suggestions if sugg.video == video
        ]

        unique_idx = set(result) ^ set(prev_idx)
        suggestions = cls.idx_list_to_frame_list(list(unique_idx), video)

        return suggestions

    @classmethod
    def velocity(
        cls,
        labels: "Labels",
        videos: List[Video],
        node: Union[int, str],
        threshold: float,
        **kwargs,
    ):
        """
        Finds frames for proofreading with high node velocity.
        """

        if isinstance(node, str):
            node_name = node
        else:
            try:
                node_name = labels.skeletons[0].nodes[node]
            except IndexError:
                node_name = ""

        suggestions = []
        for video in videos:
            suggestions.extend(cls._velocity_video(video, labels, node_name, threshold))
        return suggestions

    @classmethod
    def _velocity_video(
        cls, video: "Video", labels: "Labels", node_name: str, threshold: float
    ):
        from sleap.info.summary import StatisticSeries

        displacements = StatisticSeries(labels).get_primary_point_displacement_series(
            video=video, reduction="sum", primary_node=node_name
        )

        data_range = np.ptp(displacements)
        data_min = np.min(displacements)

        frame_idxs = list(
            map(int, np.argwhere(displacements - data_min > data_range * threshold))
        )

        prev_idx = [sugg_lst.frame_idx for sugg_lst in labels.suggestions]
        unique_idx = set(frame_idxs) - set(prev_idx)
        suggestions = cls.idx_list_to_frame_list(unique_idx, video)

        return suggestions

    # Utility functions

    @staticmethod
    def idx_list_to_frame_list(
        idx_list, video: "Video", group: Optional[GroupType] = None
    ) -> List[SuggestionFrame]:
        return [SuggestionFrame(video, frame_idx, group) for frame_idx in idx_list]


def demo_gui():
    from sleap.gui.dialogs.formbuilder import YamlFormWidget
    from sleap import Labels
    from PySide2.QtWidgets import QApplication

    labels = Labels.load_file(
        "tests/data/json_format_v2/centered_pair_predictions.json"
    )

    options_lists = dict(node=labels.skeletons[0].node_names)

    app = QApplication()
    win = YamlFormWidget.from_name(
        "suggestions", title="Generate Suggestions", field_options_lists=options_lists
    )

    def demo_suggestions(params):
        print(params)
        x = VideoFrameSuggestions.suggest(params=params, labels=labels)

        for suggested_frame in x:
            print(
                suggested_frame.video.backend.filename,
                suggested_frame.frame_idx,
                suggested_frame.group,
            )

    win.mainAction.connect(demo_suggestions)
    win.show()

    app.exec_()


if __name__ == "__main__":
    demo_gui()

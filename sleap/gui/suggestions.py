"""
Module for generating lists of suggested frames (for labeling or reviewing).
"""

import attr
import numpy as np
import random

from typing import Dict, List, Optional, Union

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
            frame_chunk=cls.frame_chunk,
            max_point_displacement=cls.max_point_displacement,
        )

        method = str.replace(params["method"], " ", "_")
        if method_functions.get(method, None) is not None:
            return method_functions[method](labels=labels, **params)
        else:
            raise ValueError(
                f"No {'' if method == '_' else method + ' '}method found for "
                "generating suggestions."
            )

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
        """Method to generate suggestions randomly or by taking strides through video."""
        suggestions = []
        sugg_idx_dict: Dict[Video, list] = {video: [] for video in labels.videos}

        for sugg in labels.suggestions:
            sugg_idx_dict[sugg.video].append(sugg.frame_idx)

        for video in videos:
            # Get unique sample space
            vid_idx = list(range(video.frames))
            vid_sugg_idx = sugg_idx_dict[video]
            unique_idx = list(set(vid_idx) - set(vid_sugg_idx))
            n_frames = len(unique_idx)

            if sampling_method == "stride":
                frame_increment = n_frames // per_video
                frame_increment = 1 if frame_increment == 0 else frame_increment
                stride_idx = list(range(0, n_frames, frame_increment))[:per_video]
                vid_suggestions = [unique_idx[idx] for idx in stride_idx]
            else:
                # random sampling
                frames_num = per_video
                frames_num = n_frames if (frames_num > n_frames) else frames_num
                if n_frames == 1:
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

        # Propose new suggestions
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
            proposed_suggestions = pipeline.get_suggestion_frames(videos=videos)
        else:
            # Run pipeline separately (in parallel) for each video
            proposed_suggestions = ParallelFeaturePipeline.run(pipeline, videos)

        suggestions = VideoFrameSuggestions.filter_unique_suggestions(
            labels, videos, proposed_suggestions
        )

        return suggestions

    @classmethod
    def prediction_score(
        cls,
        labels: "Labels",
        videos: List[Video],
        score_limit,
        instance_limit_upper,
        instance_limit_lower,
        **kwargs,
    ):
        """Method to generate suggestions for proofreading frames with low score."""
        score_limit = float(score_limit)
        instance_limit_upper = int(instance_limit_upper)
        instance_limit_lower = int(instance_limit_lower)

        proposed_suggestions = []
        for video in videos:
            proposed_suggestions.extend(
                cls._prediction_score_video(
                    video,
                    labels,
                    score_limit,
                    instance_limit_upper,
                    instance_limit_lower,
                )
            )

        suggestions = VideoFrameSuggestions.filter_unique_suggestions(
            labels, videos, proposed_suggestions
        )

        return suggestions

    @classmethod
    def _prediction_score_video(
        cls,
        video: Video,
        labels: "Labels",
        score_limit: float,
        instance_limit_upper: int,
        instance_limit_lower: int,
    ):
        lfs = labels.find(video)
        frames = len(lfs)

        # initiate an array filled with -1 to store frame index (starting from 0).
        idxs = np.full((frames), -1, dtype="int")

        for i, lf in enumerate(lfs):
            # Scores from visible instances in frame
            pred_fs = lf.instances_to_show
            frame_scores = np.array(
                [inst.score for inst in pred_fs if hasattr(inst, "score")]
            )
            # Gets the number of instances with scores lower than <score_limit>
            n_qualified_instance = np.nansum(frame_scores <= score_limit)

            if (
                n_qualified_instance >= instance_limit_lower
                and n_qualified_instance <= instance_limit_upper
            ):
                # idxs saves qualified frame index at corresponding entry, otherwise the entry is -1
                idxs[i] = lf.frame_idx

        # Finds non-negative entries in idxs
        result = sorted(idxs[idxs >= 0].tolist())

        return cls.idx_list_to_frame_list(result, video)

    @classmethod
    def velocity(
        cls,
        labels: "Labels",
        videos: List[Video],
        node: Union[int, str],
        threshold: float,
        **kwargs,
    ):
        """Finds frames for proofreading with high node velocity."""

        if isinstance(node, str):
            node_name = node
        else:
            try:
                node_name = labels.skeletons[0].nodes[node]
            except IndexError:
                node_name = ""

        proposed_suggestions = []
        for video in videos:
            proposed_suggestions.extend(
                cls._velocity_video(video, labels, node_name, threshold)
            )

        suggestions = VideoFrameSuggestions.filter_unique_suggestions(
            labels, videos, proposed_suggestions
        )

        return suggestions

    @classmethod
    def _velocity_video(
        cls, video: Video, labels: "Labels", node_name: str, threshold: float
    ):
        from sleap.info.summary import StatisticSeries

        displacements = StatisticSeries(labels).get_primary_point_displacement_series(
            video=video, reduction="sum", primary_node=node_name
        )
        data_range = np.ptp(displacements)
        data_min = np.min(displacements)

        frame_idxs = list(
            map(
                int,
                np.squeeze(
                    np.argwhere(displacements - data_min > data_range * threshold)
                ),
            )
        )

        return cls.idx_list_to_frame_list(frame_idxs, video)

    @classmethod
    def max_point_displacement(
        cls,
        labels: "Labels",
        videos: List[Video],
        displacement_threshold: float,
        **kwargs,
    ):
        """Finds frames with maximum point displacement above a threshold."""

        proposed_suggestions = []
        for video in videos:
            proposed_suggestions.extend(
                cls._max_point_displacement_video(video, labels, displacement_threshold)
            )

        suggestions = VideoFrameSuggestions.filter_unique_suggestions(
            labels, videos, proposed_suggestions
        )

        return suggestions

    @classmethod
    def _max_point_displacement_video(
        cls, video: Video, labels: "Labels", displacement_threshold: float
    ):
        # Get numpy of shape (frames, tracks, nodes, x, y)
        labels_numpy = labels.numpy(video=video, all_frames=True, untracked=False)

        # Return empty list if not enough frames
        n_frames, n_tracks, n_nodes, _ = labels_numpy.shape

        if n_frames < 2:
            return []

        # Calculate displacements
        diff = labels_numpy[1:] - labels_numpy[:-1]  # (frames - 1, tracks, nodes, x, y)
        euc_norm = np.linalg.norm(diff, axis=-1)  # (frames - 1, tracks, nodes)
        mean_euc_norm = np.nanmean(euc_norm, axis=-1)  # (frames - 1, tracks)

        # Find frames where mean displacement is above threshold
        threshold_mask = np.any(
            mean_euc_norm > displacement_threshold, axis=-1
        )  # (frames - 1,)
        frame_idxs = list(
            np.argwhere(threshold_mask).flatten() + 1
        )  # [0, len(frames - 1)]

        return cls.idx_list_to_frame_list(frame_idxs, video)

    @classmethod
    def frame_chunk(
        cls,
        labels: "Labels",
        videos: List[Video],
        frame_from: int,
        frame_to: int,
        **kwargs,
    ):
        """Add consecutive frame chunk to label suggestion"""

        proposed_suggestions = []

        # Check the validity of inputs, frame_from <= frame_to
        if frame_from > frame_to:
            return proposed_suggestions

        for video in videos:
            # Make sure when targeting all videos the from and to do not exceed frame number
            if frame_from > video.num_frames:
                continue
            this_video_frame_to = min(frame_to, video.num_frames)
            # Generate list of frame numbers
            idx = list(range(frame_from - 1, this_video_frame_to))
            proposed_suggestions.extend(cls.idx_list_to_frame_list(idx, video))

        suggestions = VideoFrameSuggestions.filter_unique_suggestions(
            labels, videos, proposed_suggestions
        )
        return suggestions

    # Utility functions

    @staticmethod
    def idx_list_to_frame_list(
        idx_list, video: "Video", group: Optional[GroupType] = None
    ) -> List[SuggestionFrame]:
        return [SuggestionFrame(video, frame_idx, group) for frame_idx in idx_list]

    @staticmethod
    def filter_unique_suggestions(
        labels: "Labels",
        videos: List[Video],
        proposed_suggestions: List[SuggestionFrame],
    ) -> List[SuggestionFrame]:
        # Create log of suggestions that already exist
        sugg_idx_dict: Dict[Video, list] = {video: [] for video in labels.videos}
        for sugg in labels.suggestions:
            sugg_idx_dict[sugg.video].append(sugg.frame_idx)

        # Filter for suggestions that already exist
        unique_suggestions = [
            sugg
            for sugg in proposed_suggestions
            if sugg.frame_idx not in sugg_idx_dict[sugg.video]
        ]

        return unique_suggestions


def demo_gui():
    from sleap.gui.dialogs.formbuilder import YamlFormWidget
    from sleap import Labels
    from qtpy.QtWidgets import QApplication

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

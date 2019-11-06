"""
Module for getting a series which gives some statistic based on labeling
data for each frame of some labeled video.
"""

import attr
import numpy as np

from typing import Callable, Dict

from sleap.io.dataset import Labels
from sleap.io.video import Video


@attr.s(auto_attribs=True)
class StatisticSeries:
    """
    Class to calculate various statistical series for labeled frames.

    Each method returns a series which is a dictionary in which keys
    are frame index and value are some numerical value for the frame.

    Args:
        labels: The :class:`Labels` for which to calculate series.
    """

    labels: Labels

    def get_point_count_series(self, video: Video) -> Dict[int, float]:
        """Get series with total number of labeled points in each frame."""
        series = dict()

        for lf in self.labels.find(video):
            val = sum(len(inst.points) for inst in lf if hasattr(inst, "score"))
            series[lf.frame_idx] = val
        return series

    def get_point_score_series(
        self, video: Video, reduction: str = "sum"
    ) -> Dict[int, float]:
        """Get series with statistic of point scores in each frame.

        Args:
            video: The :class:`Video` for which to calculate statistic.
            reduction: name of function applied to scores:
                * sum
                * min

        Returns:
            The series dictionary (see class docs for details)
        """
        reduce_funct = dict(sum=sum, min=lambda x: min(x, default=0))[reduction]

        series = dict()

        for lf in self.labels.find(video):
            val = reduce_funct(
                point.score
                for inst in lf
                for point in inst.points
                if hasattr(inst, "score")
            )
            series[lf.frame_idx] = val
        return series

    def get_instance_score_series(self, video, reduction="sum") -> Dict[int, float]:
        """Get series with statistic of instance scores in each frame.

        Args:
            video: The :class:`Video` for which to calculate statistic.
            reduction: name of function applied to scores:
                * sum
                * min

        Returns:
            The series dictionary (see class docs for details)
        """
        reduce_funct = dict(sum=sum, min=lambda x: min(x, default=0))[reduction]

        series = dict()

        for lf in self.labels.find(video):
            val = reduce_funct(inst.score for inst in lf if hasattr(inst, "score"))
            series[lf.frame_idx] = val
        return series

    def get_point_displacement_series(self, video, reduction="sum") -> Dict[int, float]:
        """
        Get series with statistic of point displacement in each frame.

        Point displacement is the distance between the point location in
        frame and the location of the corresponding point (same node,
        same track) from the closest earlier labeled frame.

        Args:
            video: The :class:`Video` for which to calculate statistic.
            reduction: name of function applied to point scores:
                * sum
                * mean
                * max

        Returns:
            The series dictionary (see class docs for details)
        """
        reduce_funct = dict(sum=np.sum, mean=np.nanmean, max=np.max)[reduction]

        series = dict()

        last_lf = None
        for lf in self.labels.find(video):
            val = self._calculate_frame_velocity(lf, last_lf, reduce_funct)
            last_lf = lf
            if not np.isnan(val):
                series[lf.frame_idx] = val  # len(lf.instances)
        return series

    def get_primary_point_displacement_series(
        self, video, reduction="sum", primary_node=None
    ):
        """
        Get sum of displacement for single node of each instance per frame.

        Args:
            video: The :class:`Video` for which to calculate statistic.
            reduction: name of function applied to point scores:
                * sum
                * mean
                * max
            primary_node: The node for which we'll calculate displacement.
                This can be name of node or `Node` object. If not specified,
                then defaults to first node.

        Returns:
            The series dictionary (see class docs for details)
        """
        reduce_funct = dict(sum=np.sum, mean=np.nanmean, max=np.max)[reduction]

        track_count = len(self.labels.find_first(video).instances)

        try:
            primary_node_idx = self.labels.skeletons[0].node_to_index(primary_node)
        except:
            print(f"Unable to locate node {primary_node} so using node 0")
            primary_node_idx = 0

        last_frame_idx = self.labels.find_last(video).frame_idx
        location_matrix = np.full(
            (last_frame_idx + 1, track_count, 2), np.nan, dtype=np.float
        )
        for lf in self.labels.find(video):
            for inst in lf.instances:
                if inst.track is not None:
                    track_idx = self.labels.tracks.index(inst.track)
                    if track_idx < track_count:
                        frame_idx = lf.frame_idx
                        point = inst.points_array[primary_node_idx, :2]
                        location_matrix[frame_idx, track_idx] = point

        displacement = location_matrix[1:, ...] - location_matrix[:-1, ...]

        displacement_distances = np.linalg.norm(displacement, axis=2)

        result = reduce_funct(displacement_distances, axis=1)
        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _calculate_frame_velocity(
        lf: "LabeledFrame", last_lf: "LabeledFrame", reduce_function: Callable
    ) -> float:
        """
        Calculate total point displacement between two given frames.

        Args:
            lf: The :class:`LabeledFrame` for which we want velocity
            last_lf: The frame from which to calculate displacement.
            reduce_function: Numpy function (e.g., np.sum, np.nanmean)
                is applied to *point* displacement, and then those
                instance values are summed for the whole frame.

        Returns:
            The total velocity for instances in frame.
        """
        val = 0
        for inst in lf:
            if last_lf is not None:
                last_inst = last_lf.find(track=inst.track)
                if last_inst:
                    points_a = inst.points_array
                    points_b = last_inst[0].points_array
                    point_dist = np.linalg.norm(points_a - points_b, axis=1)
                    inst_dist = reduce_function(point_dist)
                    val += inst_dist if not np.isnan(inst_dist) else 0
        return val

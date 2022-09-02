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

        track_count = self.labels.get_track_count(video)

        try:
            primary_node_idx = self.labels.skeletons[0].node_to_index(primary_node)
        except ValueError:
            print(f"Unable to locate node {primary_node} so using node 0")
            primary_node_idx = 0

        last_frame_idx = video.num_frames - 1
        location_matrix = np.full(
            (last_frame_idx + 1, track_count, 2), np.nan, dtype=float
        )
        last_track_pos = np.full((track_count, 2), 0, dtype=float)

        has_seen_track_idx = set()

        for frame_idx in range(last_frame_idx + 1):
            lfs = self.labels.find(video, frame_idx)

            # Start by setting all track positions to where they were last,
            # so that we won't get "jumps" when an instance is missing for
            # some frames.
            location_matrix[frame_idx] = last_track_pos

            # Now update any positions we do have for the frame
            if lfs:
                lf = lfs[0]
                for inst in lf.instances:
                    if inst.track is not None:
                        track_idx = self.labels.tracks.index(inst.track)
                        if track_idx < track_count:
                            point = inst.points_array[primary_node_idx, :2]
                            location_matrix[frame_idx, track_idx] = point

                            if not np.all(np.isnan(point)):
                                # Keep track of where this track was last.
                                last_track_pos[track_idx] = point

                                # If this is the first time we've seen this
                                # track, then use initial location for all
                                # previous frames so first occurrence doesn't
                                # have high displacement.
                                if track_idx not in has_seen_track_idx:
                                    location_matrix[:frame_idx, track_idx] = point
                                    has_seen_track_idx.add(track_idx)

        # Calculate the displacements. Note these will be offset by 1 frame
        # since we're starting from frame 1 rather than 0.
        displacement = location_matrix[1:, ...] - location_matrix[:-1, ...]

        displacement_distances = np.linalg.norm(displacement, axis=2)

        result = reduce_funct(displacement_distances, axis=1)
        result[np.isnan(result)] = 0

        # Shift back by 1 frame so offsets line up with frame index.
        result[1:] = result[:-1]

        return result

    def get_min_centroid_proximity_series(self, video):
        series = dict()

        def min_centroid_dist(instances):
            if len(instances) < 2:
                return np.nan
            # centroids for all instances in frame
            centroids = np.array([inst.centroid for inst in instances])
            # calculate distance between each pair of instance centroids
            distances = np.linalg.norm(
                centroids[np.newaxis, :, :] - centroids[:, np.newaxis, :], axis=-1
            )
            # clear distance from each instance to itself
            np.fill_diagonal(distances, np.nan)
            # return the min
            return np.nanmin(distances)

        for lf in self.labels.find(video):
            val = min_centroid_dist(lf.instances)
            if not np.isnan(val):
                series[lf.frame_idx] = val
        return series

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

"""
Module to use Kalman filters for tracking instance identities.

The Kalman filters needs a small number of frames already tracked in order
to initialize the filters. Then you can use the module for tracking on the
remaining frames.

It's a good idea to cull the instances (i.e., N best instances per frame) before
trying to track with the Kalman filter, since the skeleton fragments can mess
up the filters.

Usage:

> filter_frames(frames, instance_count=2, node_indices=[0, 1, 2])
"""
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Text, Tuple

import attr
import numpy as np
import pykalman

from numpy import ma
from pykalman import KalmanFilter

from sleap import Instance, PredictedInstance, LabeledFrame, Track


TOO_CLOSE_DIST = 5


@attr.s(auto_attribs=True, slots=True)
class Match:
    track: Track
    instance: Instance
    score: Optional[float] = None


@attr.s(auto_attribs=True)
class TrackKalman:
    kalman_filters: Dict[Track, pykalman.KalmanFilter]
    last_results: Dict[Track, Dict[Text, Any]]
    tracks: List[Track]
    instance_count: int
    instance_score_thresh: float
    node_indices: List[int]  # indices of rows for points to use

    @classmethod
    def initialize(
        cls,
        frames: List[LabeledFrame],
        instance_count: int,
        node_indices: List[int],
        instance_score_thresh: float = 0.3,
    ) -> "TrackKalman":
        frame_array_dict = defaultdict(list)

        track_list = []
        filters = dict()
        last_results = dict()

        instances = [inst for lf in frames for inst in lf.instances]

        if not instances:
            raise ValueError("Kalman filter must be initialized with instances.")

        # TODO: make arg optional and use algorithm to find best nodes to track

        for inst in instances:
            point_coords = inst.points_array[node_indices, 0:2].flatten()
            frame_array_dict[inst.track].append(point_coords)

        for track, frame_array in frame_array_dict.items():

            frame_array = ma.asarray(frame_array)
            frame_array = ma.masked_invalid(frame_array)

            initial_frame = frame_array[0]
            initial_frame_size = initial_frame.size
            initial_state_means = [0] * (initial_frame_size * 2)

            for coord_idx, coord_value in enumerate(initial_frame.flatten()):
                initial_state_means[(coord_idx * 2)] = coord_value

            transition_matrix = []

            for coord_idx in range(0, initial_frame_size):
                transition_matrix.append(
                    [
                        int(x in [(coord_idx * 2), (coord_idx * 2) + 1])
                        for x in range(initial_frame_size * 2)
                    ]
                )

                transition_matrix.append(
                    [
                        int(x == ((coord_idx * 2) + 1))
                        for x in range(initial_frame_size * 2)
                    ]
                )

            observation_matrix = []

            for coord_idx in range(0, initial_frame_size):
                observation_matrix.append(
                    [int(x == (coord_idx * 2)) for x in range(initial_frame_size * 2)]
                )

            # Make the filter for this track
            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_means,
            )
            kf = kf.em(frame_array, n_iter=20)
            state_means, state_covariances = kf.filter(frame_array)

            # Store necessary objects/data for this track
            track_list.append(track)
            filters[track] = kf
            last_results[track] = {
                "means": list(state_means)[-1],
                "covariances": list(state_covariances)[-1],
            }

        return cls(
            kalman_filters=filters,
            last_results=last_results,
            tracks=track_list,
            instance_count=instance_count,
            node_indices=node_indices,
            instance_score_thresh=instance_score_thresh,
        )

    def track_frames(self, frames: List[LabeledFrame]):
        """
        Runs tracking for every frame in list using initialized Kalman filters.
        """
        for lf in frames:
            # Only track predicted instances in frame
            # (one reason is that we use predicted point score, which doesn't
            # exist for user instances).
            untracked_instances = lf.predicted_instances

            # Get expected positions for each track.
            # Doesn't update "last results" since we don't want this updated
            # until after we process the frame (below).
            filter_results = self.update_filters(only_update_matches=False)

            # Measure similarity (inverse cost) between each instance
            # and each track, based on expected position for track.
            cost_matrix = self.frame_cost_matrix(
                untracked_instances=untracked_instances, filter_results=filter_results
            )

            # FIXME: why all nans? is this the right thing to do?
            if np.all(np.isnan(cost_matrix)):
                continue

            # Only count best matches which are sufficiently better than next
            # best match (by threshold determined from data).
            best_vs_second_thresh = self.get_best_vs_second_threshold(
                cost_matrix, untracked_instances
            )

            cost_matrix = remove_second_bests_from_cost_matrix(
                cost_matrix, thresh=best_vs_second_thresh
            )

            # Match instances to tracks based on similarity matrix.
            matches = self.get_track_instance_matches(
                cost_matrix, instances=untracked_instances
            )

            track_inst_matches = {match.track: match.instance for match in matches}

            # Update filters with points for each matched instance.
            self.last_results.update(
                self.update_filters(track_inst_matches, only_update_matches=True)
            )

            # Set tracks on matched instances
            for match in matches:
                # print(f"set track to {match.track.name} ({match.score})")
                match.instance.track = match.track

    def update_filters(
        self,
        track_instance_matches: Optional[Dict[Track, Instance]] = None,
        only_update_matches: bool = False,
    ):
        """
        Updates state of Kalman filters.

        For matching tracks to instances on a frame, we update the filters
        to get the expected means and covariances for each tracked identity.

        After matching tracks and instances on a frame, we update the filters
        which matched with the points from the matched instance.

        Args:
            track_instance_matches: Dictionary with instance that matched to
                each track. Only used when updating after matches for frame.
            only_update_matches: Whether to update all filters (using
                ma.masked as points when we don't have match) or to skip
                updating filters without a match. Should be False when updating
                when getting data to use for frame matching and True when
                updating after we've determined matches.

        Returns:
            None; modifies `last_results` attribute.
        """
        results = dict()

        # Update each Kalman filter, one per tracked identity
        for track, kf in self.kalman_filters.items():

            if track_instance_matches and track in track_instance_matches:

                inst = track_instance_matches[track]

                # x1, 0, y1, 0, x2, 0, y2, 0, ...
                # points_array = np.zeros(len(self.node_indices) * 4)
                # points_array[::2] = inst.points_array[self.node_indices, 0:2].flatten()
                points_array = inst.points_array[self.node_indices, 0:2].flatten()

                # convert to masked array
                points_array = ma.masked_invalid(ma.asarray(points_array))

            elif only_update_matches:
                continue

            else:
                points_array = ma.masked

            exp_mean, exp_covariance = kf.filter_update(
                self.last_results[track]["means"],
                self.last_results[track]["covariances"],
                points_array,
            )

            # The outputs from filter_update are lists which give 4 values for
            # for each node: x, x_velocity, y, y_velocity.

            # When matching instances to tracks we just use (x, y), so make
            # list of (x0, y0, x1, y1, ...) with (x_n, y_n) for each node.
            exp_coord_means = np.array(exp_mean[::2])

            results[track] = {
                "means": exp_mean,
                "covariances": exp_covariance,
                "coordinate_means": exp_coord_means,
            }

        return results

    def get_instance_points_weight(
        self, instance: PredictedInstance
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns points (and weights, i.e., scores) for tracked nodes."""
        # For predicted instances, the *full* points array will be (N, 5)
        # where N is the number of nodes.
        # Each row has: x, y, visible, complete, score
        point_array = instance.get_points_array(
            copy=True, invisible_as_nan=True, full=True
        )

        # inst_points: [x1, y1, x2, y2, ...]
        # weights: [score1, score1, score2, score2, ...]
        # distances: [abs(x1 - x1_hat), ...]
        inst_points = point_array[self.node_indices, 0:2].flatten()
        weights = point_array[self.node_indices, 4].flatten().repeat(2)

        return inst_points, weights

    def get_best_vs_second_threshold(
        self, cost_matrix: np.ndarray, instances: List[PredictedInstance]
    ) -> float:
        """"Returns threshold to use when comparing best and second-best matches."""

        # Best vs second-best threshold (see below) determined by:
        #  cost of best best match (i.e., min for whole cost matrix),
        #  min mean dist between relevant nodes in instances (pairwise).
        best_best_match_cost = np.nanmin(cost_matrix)
        min_mean_dist = self.min_mean_inst_dist(instances)

        # If the mean point distance between the closest pair of instances
        # is less than TOO_CLOSE_DIST pixels, make sure the best match cost is at least
        # TOO_CLOSE_DIST better than the second-best cost when matching.
        if min_mean_dist < TOO_CLOSE_DIST:
            best_vs_second_thresh = max(best_best_match_cost, TOO_CLOSE_DIST)
        else:
            # Otherwise, use best match value as threshold since this is
            # the minimum mean distance between actual and expected point.
            best_vs_second_thresh = best_best_match_cost

        return best_vs_second_thresh

    @staticmethod
    def instance_points_match_cost(
        instance_points: np.ndarray,
        instance_weights: np.ndarray,
        expected_points: np.ndarray,
    ) -> float:
        """Returns match cost between instance and expected (filter) points."""
        distances = np.absolute(expected_points - instance_points)

        if all(np.isnan(distances)):
            return np.nan

        distances = ma.MaskedArray(distances, mask=np.isnan(distances))

        # "cost" for matching mean point distance weighted by point score
        return ma.average(distances, weights=instance_weights)

    def min_mean_inst_dist(self, instances: List[PredictedInstance]) -> float:
        """Returns minimum mean distance between instances compared pairwise."""
        inst_points = dict()
        for inst in instances:
            inst_points[inst], _ = self.get_instance_points_weight(inst)

        def pair_mean_dist(inst_a, inst_b):
            d = np.absolute(inst_points[inst_a] - inst_points[inst_b])
            return np.nanmean(d) if not np.all(np.isnan(d)) else np.nan

        return min(
            (
                pair_mean_dist(inst_a, inst_b)
                for inst_a, inst_b in itertools.combinations(instances, 2)
            ),
            default=np.nan,
        )

    def frame_cost_matrix(
        self,
        untracked_instances: List[PredictedInstance],
        filter_results: Dict[Track, Dict[Text, Any]],
    ) -> np.ndarray:
        """
        Returns full cost matrix for matches.

        Instances are rows, tracks (filters) are columns.
        """
        # Matrix of matching similarity: [inst, track]
        matching_similarity = np.full(
            (len(untracked_instances), len(self.kalman_filters)), np.nan
        )

        for inst_idx, inst in enumerate(untracked_instances):

            if hasattr(inst, "score") and inst.score < self.instance_score_thresh:
                # Don't try to match to instances with sufficiently low score.
                # FIXME: Maybe we should still do match since otherwise we might
                #  match a track that should have matched the low scoring instance
                #  to another instance.
                continue

            inst_points, inst_weights = self.get_instance_points_weight(inst)

            for track_idx, track in enumerate(self.tracks):
                inst_sim = self.instance_points_match_cost(
                    inst_points,
                    inst_weights,
                    expected_points=filter_results[track]["coordinate_means"],
                )

                matching_similarity[inst_idx, track_idx] = inst_sim

        return matching_similarity

    def get_track_instance_matches(
        self, cost_matrix: np.ndarray, instances: List[Instance]
    ) -> List[Match]:
        """
        Greedily matches tracks to instances using similarity matrix.
        """
        from sleap.nn.tracking import greedy_matching

        match_indices = greedy_matching(cost_matrix)

        matches = []

        for i, j in match_indices:
            match_score = cost_matrix[i, j]
            if np.isfinite(match_score):
                matches.append(
                    Match(
                        track=self.tracks[j], instance=instances[i], score=match_score
                    )
                )

        return matches


def remove_second_bests_from_cost_matrix(
    cost_matrix: np.ndarray, thresh: float, invalid_val: float = np.nan,
) -> np.ndarray:
    """
    Removes unclear matches from cost matrix.

    If the best match for a given track is too close to the second best match,
    then this will clear all the matches for that track (and ensure that any
    instance where that track was the best match won't be matched to another
    track).

    It removes the matches by setting the appropriate rows/columns to the
    specified invalid_val (usually nan or inf).

    Args:
        cost_matrix: This is a negative cost matrix.
        thresh: Best match must be better than second best + threshold
            to be valid.
        invalid_val: Value to set invalid rows/columns to.

    Returns:
         cost matrix with invalid matches set to specified invalid value.
    """

    valid_match_mask = np.full_like(cost_matrix, True, dtype=np.bool)

    rows, columns = cost_matrix.shape

    # Invalidate columns with best match too close to second best match.
    for c in range(columns):
        column = cost_matrix[:, c]

        # Skip columns with all nans
        if all(np.isnan(column)):
            continue

        # Get best match value for this column.
        col_min_val = column.min()

        # Count the number of column within threshold of best match.
        close_match_count = (column < (col_min_val + thresh)).sum()

        # Best match is already 1, so check if more than one close match
        if close_match_count > 1:
            valid_match_mask[:, c] = False

    # Invalidate rows where best match is already invalidated or is too close
    # to second best match.
    for r in range(rows):
        # Similar logic to the loop over columns, except for the additional
        # check whether the best match in row is still valid.
        row = cost_matrix[r]

        if np.all(np.isnan(row)):
            continue

        row_validity_mask = valid_match_mask[r]

        row_min_idx = row.argmin()
        row_min_val = row[row_min_idx]
        is_min_item_valid = row_validity_mask[row_min_idx]

        close_match_count = (row < (row_min_val + thresh)).sum()

        # Make sure the best match for row isn't too close to second best match
        # and hasn't already been ruled out (this would happen if the column was
        # invalidated because the best and second best matches were too close).
        # For instance there could be a track (column) which is ruled out
        # and an instance (row) where the best match for that *instance* is
        # the row that's already ruled out. In this case, we want to make sure
        # that we don't match anything (i.e., the best valid match) for that
        # instance.
        if close_match_count > 1 or not is_min_item_valid:
            valid_match_mask[r] = False

    # Copy the similarity matrix (so we don't modify in place) and set invalid
    # matches to specified invalid value.
    valid_cost_matrix = np.copy(cost_matrix)
    valid_cost_matrix[~valid_match_mask] = invalid_val

    return valid_cost_matrix


def filter_frames(
    frames: List[LabeledFrame],
    instance_count: int,
    node_indices: List[int],
    init_len: int = 10,
):
    """
    Attempts to track N instances using a Kalman Filter.

    Args:
        frames: The list of `LabeledFrame` objects with predictions.
        instance_count: The number of expected instances per frame.
        node_indices: Indices of nodes to use for tracking.
        init_len: The number of frames that should be used to initialize
            the Kalman filter.

    Returns:
        None; modifies frames in place.
    """

    # Initialize the filter
    kalman_filter = TrackKalman.initialize(
        frames[:init_len], instance_count, node_indices
    )

    # Run the filter, frame by frame
    kalman_filter.track_frames(frames[init_len:])

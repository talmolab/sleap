"""
Module to use Kalman filters for tracking instance identities.

The Kalman filters needs a small number of frames already tracked in order
to initialize the filters. Then you can use the module for tracking on the
remaining frames.

It's a good idea to cull the instances (i.e., N best instances per frame) before
trying to track with the Kalman filter, since the skeleton fragments can mess
up the filters.
"""

import attr
import itertools

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple

import numpy as np
from numpy import ma

import pykalman
from pykalman import KalmanFilter

from sleap import Instance, PredictedInstance, LabeledFrame, Track
from sleap.nn.tracker.components import (
    greedy_matching,
    InstanceType,
    Match,
    first_choice_matching,
)


@attr.s(auto_attribs=True)
class BareKalmanTracker:
    node_indices: List[int]  # indices of rows for points to use
    instance_count: int

    instance_score_thresh: float = 0.3
    reset_gap_size: int = 5

    kalman_filters: Dict[Track, pykalman.KalmanFilter] = attr.ib(factory=dict)
    last_results: Dict[Track, Dict[Text, Any]] = attr.ib(factory=dict)
    tracks: List[Track] = attr.ib(factory=list)
    last_frame_for_track: Dict[Track, int] = attr.ib(factory=dict)

    @classmethod
    def initialize(
        cls,
        frames: List[LabeledFrame],
        instance_count: int,
        node_indices: List[int],
        instance_score_thresh: float = 0.3,
        reset_gap_size: int = 5,
    ) -> "BareKalmanTracker":

        kf_obj = cls(
            instance_count=instance_count,
            node_indices=node_indices,
            instance_score_thresh=instance_score_thresh,
            reset_gap_size=reset_gap_size,
        )

        instances_lists = [lf.predicted_instances for lf in frames]

        kf_obj.init_filters(instances_lists)

        return kf_obj

    def init_filters(
        self, instances: Iterable[PredictedInstance]
    ):  # tracked_instances_lists: Iterable[Iterable[PredictedInstance]]):
        frame_array_dict = defaultdict(list)

        track_list = []
        filters = dict()
        last_results = dict()

        if not instances:
            raise ValueError("Kalman filter must be initialized with instances.")

        # TODO: make arg optional and use algorithm to find best nodes to track

        # instances = [inst for inst_list in tracked_instances_lists for inst in inst_list]

        for inst in instances:
            point_coords = inst.points_array[self.node_indices, 0:2].flatten()
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

        self.kalman_filters = filters
        self.tracks = track_list
        self.last_results = last_results
        self.last_frame_for_track = dict()

    def replace_track(self, old_track: Track):
        """
        Replaces track identity tied to a Kalman filter.

        This is used when there's a significant gap in the tracking so we're
        no longer confident that the filter is tracking the same identity as
        before.
        """
        new_track = Track(spawned_on=-1, name=old_track.name)
        self.kalman_filters[new_track] = self.kalman_filters.pop(old_track)
        self.tracks[self.tracks.index(old_track)] = new_track
        if old_track in self.last_results:
            self.last_results[new_track] = self.last_results.pop(old_track)

    def track_frame(
        self, untracked_instances: List[PredictedInstance], frame_idx: int
    ) -> List[PredictedInstance]:
        """
        Tracks instances from single frame using Kalman filters.

        Args:
            untracked_instances: List of instances from frame.
            frame_idx: Frame index, used for track spawn frame index and for
                determining if we've had a large gap in tracking.

        Returns:
            None; updates tracks on instances in place.
        """
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
            return untracked_instances

        # Only count best matches which are sufficiently better than next
        # best match (by threshold determined from data).
        min_dist_from_expected_location = float(np.nanmin(cost_matrix))
        cost_matrix = remove_second_bests_from_cost_matrix(
            cost_matrix, thresh=min_dist_from_expected_location
        )

        # Make function which determines whether instances are "too close"
        # for one of the instances to use its second choice match.
        # "too close" is determined by the same threshold used for
        # making sure best match is sufficiently better than second best,
        # i.e., the minimum (mean point) distance between any instance
        # and any of the expected coordinates (which correspond to tracks).
        too_close_funct = self.get_too_close_checking_function(
            untracked_instances, dist_thresh=min_dist_from_expected_location
        )

        # Match instances to tracks based on similarity matrix.
        matches = get_track_instance_matches(
            cost_matrix,
            instances=untracked_instances,
            tracks=self.tracks,
            are_too_close_function=too_close_funct,
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
            self.last_frame_for_track[match.track] = frame_idx

            # When tracks are reset during a gap we don't know when the
            # track will be used first so we set spawn to -1 so we can
            # set it correctly when the reset track is first matched.
            if match.track.spawned_on < 0:
                match.track.spawned_on = int(frame_idx)

        # Check how many tracks have a gap since they were last matched
        tracks_with_gap = self.tracks_with_gap(frame_idx)
        # If multiple tracks, then we want to start new tracks for each.
        if len(tracks_with_gap) > 1:
            for track in tracks_with_gap:
                self.replace_track(track)
                self.last_frame_for_track.pop(track)

        return untracked_instances

    def tracks_with_gap(self, frame_idx):
        return [
            track
            for track, last_frame_idx in self.last_frame_for_track.items()
            if (frame_idx - last_frame_idx) > self.reset_gap_size
        ]

    @property
    def last_frame_with_tracks(self):
        return max(self.last_frame_for_track.values(), default=0)

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

        if not self.node_indices:
            raise ValueError("Kalman tracker must have node_indices set.")

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

    def get_too_close_checking_function(
        self, instances: List[InstanceType], dist_thresh: float
    ) -> Callable:
        """ "
        Returns a function which determines if two instances are too close.

        Args:
            instances: Function should take pairs of instances in this list.
            dist_thresh: Pairs of instances with mean point distance less
                than this threshold count as "too close".

        Returns:
            Function with signature (Instance, Instance) -> bool.
        """

        mean_distance_lookup = self.get_mean_instance_distances(instances)

        def too_close_funct(inst_a: Instance, inst_b: Instance) -> bool:
            if (inst_a, inst_b) in mean_distance_lookup:
                return mean_distance_lookup[(inst_a, inst_b)] < dist_thresh
            else:
                return mean_distance_lookup[(inst_b, inst_a)] < dist_thresh

        return too_close_funct

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

    def get_mean_instance_distances(
        self, instances: List[PredictedInstance]
    ) -> Dict[Tuple[Instance, Instance], float]:
        """Returns minimum mean distance between instances compared pairwise."""
        inst_points = dict()
        for inst in instances:
            inst_points[inst], _ = self.get_instance_points_weight(inst)

        def pair_mean_dist(inst_a, inst_b):
            d = np.absolute(inst_points[inst_a] - inst_points[inst_b])
            return np.nanmean(d) if not np.all(np.isnan(d)) else np.nan

        return {
            (inst_a, inst_b): pair_mean_dist(inst_a, inst_b)
            for inst_a, inst_b in itertools.combinations(instances, 2)
        }

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
    cost_matrix: np.ndarray,  # [instance, track] match cost
    instances: List[InstanceType],  # rows in cost matrix
    tracks: List[Track],  # columns in cost matrix
    are_too_close_function: Callable,
) -> List[Match]:
    """
    Matches track identities (from filters) to instances in frame.

    Algorithm is modified greedy matching.

    Standard greedy matching:
    0. Start with list of all possible matches, sorted by ascending cost.
    1. Find the match with minimum cost.
    2. Use this match.
    3. Remove other matches from list with same row or same column.
    4. Go back to (1).

    The algorithm implemented here replaces step (2) with this:

    2'. If the instance for the match would have preferred a track which
        was already matched by another instance, then only use the match
        under consideration now if these instances aren't "too close".

    Whenever the greedy matching would assign an instance to a track that
    wasn't its first choice (i.e., another instance was a better match for
    the track that would have been our current instance's first choice),
    then we make sure that the two instances aren't too close together.

    The upshot is that if two nearby instances are competing for the same
    track, then the looser won't be given its second choice (since it's more
    likely to be a redundant instance), but if nearby instances both match
    best to distinct tracks, both matches are used.

    What counts as "too close" is determined by the `are_too_close_function`
    argument, which should have this signature:

        are_too_close_function(Instance, Instance) -> bool
    """

    # For each track, determine which instance was the best match in isolation.
    first_choice_matches_by_track = match_dict_from_match_function(
        cost_matrix=cost_matrix,
        row_items=instances,
        column_items=tracks,
        match_function=first_choice_matching,
    )

    # Find the best (greedy) set of compatible matches.
    greedy_matches = matches_from_match_tuples(
        match_tuples_from_match_function(
            cost_matrix=cost_matrix,
            row_items=instances,
            column_items=tracks,
            match_function=greedy_matching,
        )
    )

    good_matches = []
    for match in greedy_matches:

        # Was the matched track the first-choice match for any instance?
        if match.track in first_choice_matches_by_track:

            # Which instance was the best match (in isolation) for this track?
            competing_instance = first_choice_matches_by_track[match.track]

            # Was it a different instance that the instance we're considering
            # a match now (presumably because this instance may have had its
            # first choice taken by another instance).
            if match.instance != competing_instance:

                # The current match instance is distinct from the instance
                # which was the *best* (isolated) match for this track.
                # Check if instances are too close for this match to be valid.
                if are_too_close_function(match.instance, competing_instance):
                    continue

        good_matches.append(match)

    return good_matches


def match_dict_from_match_function(
    cost_matrix: np.ndarray,
    row_items: List[Any],
    column_items: List[Any],
    match_function: Callable,
    key_by_column: bool = True,
) -> Dict[Any, Any]:
    """
    Dict keys are from column (tracks), values are from row (instances).

    If multiple rows (instances) match on the same column (track), then
    dict will just contain the best match.
    """

    match_dict = dict()
    match_cost_dict = dict()

    for i, j in match_function(cost_matrix):
        match_cost = cost_matrix[i, j]
        if np.isfinite(match_cost):

            if key_by_column:
                key, val = column_items[j], row_items[i]
            else:
                val, key = column_items[j], row_items[i]

            if key not in match_dict or match_cost < match_cost_dict[key]:
                match_dict[key] = val
                match_cost_dict[key] = match_cost

    return match_dict


def match_tuples_from_match_function(
    cost_matrix: np.ndarray,
    row_items: List[Any],
    column_items: List[Any],
    match_function,
) -> List[Tuple[Any, Any, float]]:
    return [
        (row_items[i], column_items[j], cost_matrix[i, j])
        for (i, j) in match_function(cost_matrix)
        if np.isfinite(cost_matrix[i, j])
    ]


def matches_from_match_tuples(
    match_tuples: List[Tuple[Instance, Track, float]]
) -> List[Match]:
    return [
        Match(instance=inst, track=track, score=score)
        for (inst, track, score) in match_tuples
    ]


def remove_second_bests_from_cost_matrix(
    cost_matrix: np.ndarray,
    thresh: float,
    invalid_val: float = np.nan,
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
        cost_matrix: Cost matrix for matching, lower means better match.
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

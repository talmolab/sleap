import numpy as np

import sleap.nn.tracker.components
import sleap.nn.tracker.kalman as k
from sleap.nn.tracker.components import greedy_matching


def test_first_choice_matching():
    instances = ["instance a", "instance b"]
    tracks = ["track a", "track b"]

    # columns are tracks
    # rows are instances
    cost_matrix = np.array([[10, 150], [50, 100]])

    match_tuples = k.match_tuples_from_match_function(
        cost_matrix=cost_matrix,
        row_items=instances,
        column_items=tracks,
        match_function=sleap.nn.tracker.components.first_choice_matching,
    )

    assert len(match_tuples) == 2
    assert ("instance a", "track a", 10) in match_tuples
    assert ("instance b", "track a", 50) in match_tuples

    match_by_track = k.match_dict_from_match_function(
        cost_matrix=cost_matrix,
        row_items=instances,
        column_items=tracks,
        match_function=sleap.nn.tracker.components.first_choice_matching,
    )

    assert len(match_by_track) == 1
    assert match_by_track["track a"] == "instance a"

    match_by_instance = k.match_dict_from_match_function(
        cost_matrix=cost_matrix,
        row_items=instances,
        column_items=tracks,
        match_function=sleap.nn.tracker.components.first_choice_matching,
        key_by_column=False,
    )

    assert len(match_by_instance) == 2
    assert match_by_instance["instance a"] == "track a"
    assert match_by_instance["instance b"] == "track a"

    # another cost matrix
    # make sure we get *best* match for each track, regardless of row order
    cost_matrix = np.array(
        [
            [50, 100],
            [10, 150],
        ]
    )
    match_by_track = k.match_dict_from_match_function(
        cost_matrix=cost_matrix,
        row_items=instances,
        column_items=tracks,
        match_function=sleap.nn.tracker.components.first_choice_matching,
    )

    assert len(match_by_track) == 1
    assert match_by_track["track a"] == "instance b"


def test_greedy_matching():
    instances = ["instance a", "instance b"]
    tracks = ["track a", "track b"]

    # columns are tracks
    # rows are instances
    cost_matrix = np.array([[10, 200], [75, 150]])

    matches = k.matches_from_match_tuples(
        k.match_tuples_from_match_function(
            cost_matrix=cost_matrix,
            row_items=instances,
            column_items=tracks,
            match_function=greedy_matching,
        )
    )

    assert len(matches) == 2

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance a"
    assert matches[0].score == 10

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance b"
    assert matches[1].score == 150


def test_track_instance_matches():
    instances = ["instance a", "instance b"]
    tracks = ["track a", "track b"]

    # columns are tracks
    # rows are instances
    cost_matrix = np.array([[10, 200], [75, 150]])

    matches = k.get_track_instance_matches(
        cost_matrix=cost_matrix,
        instances=instances,
        tracks=tracks,
        are_too_close_function=lambda x, y: True,
    )

    # instance b would prefer track a but gets bumped to track b
    # since there's no competition for track b, the "too close" check
    # isn't applied.

    assert len(matches) == 2

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance a"
    assert matches[0].score == 10

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance b"
    assert matches[1].score == 150

    # another cost matrix
    # best match is instance a -> track a
    # next match is instance b -> track b
    # but instance b would prefer track a
    cost_matrix = np.array(
        [
            [10, 100],
            [50, 150],
        ]
    )

    matches = k.get_track_instance_matches(
        cost_matrix=cost_matrix,
        instances=instances,
        tracks=tracks,
        are_too_close_function=lambda x, y: True,
    )

    assert len(matches) == 2

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance a"
    assert matches[0].score == 10

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance b"
    assert matches[1].score == 150

    # best match is instance b -> track a (cost 10)
    # next match is instance a -> track b (cost 100)
    # each instance gets its first choice so "too close" check shouldn't apply
    cost_matrix = np.array(
        [
            [50, 100],
            [10, 150],
        ]
    )

    matches = k.get_track_instance_matches(
        cost_matrix=cost_matrix,
        instances=instances,
        tracks=tracks,
        are_too_close_function=lambda x, y: True,
    )

    assert len(matches) == 2

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance b"
    assert matches[0].score == 10

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance a"
    assert matches[1].score == 100

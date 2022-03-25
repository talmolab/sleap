import pytest
import numpy as np
import tensorflow as tf
import sleap
from numpy.testing import assert_array_equal, assert_allclose

from sleap.nn.paf_grouping import (
    get_connection_candidates,
    make_line_subs,
    get_paf_lines,
    compute_distance_penalty,
    score_paf_lines,
    score_paf_lines_batch,
    match_candidates_sample,
    match_candidates_batch,
    group_instances_sample,
    group_instances_batch,
    EdgeType,
    EdgeConnection,
    PeakID,
    toposort_edges,
    assign_connections_to_instances,
)

sleap.nn.system.use_cpu_only()


def test_get_connection_candidates():
    peak_channel_inds_sample = [0, 0, 0, 1, 1, 2]
    skeleton_edges = [[0, 1], [1, 2], [2, 3]]
    n_nodes = 4

    edge_inds, edge_peak_inds = get_connection_candidates(
        peak_channel_inds_sample, skeleton_edges, n_nodes
    )

    assert_array_equal(edge_inds, [0, 0, 0, 0, 0, 0, 1, 1])

    assert_array_equal(
        edge_peak_inds, [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [4, 5]]
    )


def test_make_line_subs():
    peaks_sample = tf.constant([[0, 0], [4, 8]], tf.float32)
    edge_peak_inds = tf.constant([[0, 1]], tf.int32)
    edge_inds = tf.constant([0], tf.int32)

    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points=3, pafs_stride=2
    )
    assert_array_equal(
        line_subs,
        [[[[0, 0, 0], [0, 0, 1]], [[2, 1, 0], [2, 1, 1]], [[4, 2, 0], [4, 2, 1]]]],
    )


def test_paf_lines():
    # pafs_sample = tf.reshape(tf.range(6 * 4 * 2), [6, 4, 2])
    pafs_sample = tf.cast(tf.reshape(tf.range(6 * 4 * 2), [6, 4, 2]), tf.float32)
    peaks_sample = tf.constant([[0, 0], [4, 8]], tf.float32)
    edge_peak_inds = tf.constant([[0, 1]], tf.int32)
    edge_inds = tf.constant([0], tf.int32)
    paf_lines = get_paf_lines(
        pafs_sample,
        peaks_sample,
        edge_peak_inds,
        edge_inds,
        n_line_points=3,
        pafs_stride=2,
    )
    assert_array_equal(paf_lines, [[[0, 1], [18, 19], [36, 37]]])


def test_score_paf_lines():
    pafs_sample = tf.cast(tf.reshape(tf.range(6 * 4 * 2), [6, 4, 2]), tf.float32)
    peaks_sample = tf.constant([[0, 0], [4, 8]], tf.float32)
    edge_peak_inds = tf.constant([[0, 1]], tf.int32)
    edge_inds = tf.constant([0], tf.int32)
    paf_lines = get_paf_lines(
        pafs_sample,
        peaks_sample,
        edge_peak_inds,
        edge_inds,
        n_line_points=3,
        pafs_stride=2,
    )

    scores = score_paf_lines(paf_lines, peaks_sample, edge_peak_inds, max_edge_length=2)
    assert_allclose(scores, [24.27], atol=1e-2)


def test_compute_distance_penalty():
    penalties = compute_distance_penalty(
        tf.cast([1, 2, 3, 4], tf.float32), max_edge_length=2
    )
    assert_allclose(penalties, [0, 0, 2 / 3 - 1, 2 / 4 - 1], atol=1e-6)

    penalties = compute_distance_penalty(
        tf.cast([1, 2, 3, 4], tf.float32), max_edge_length=2, dist_penalty_weight=2
    )
    assert_allclose(penalties, [0, 0, -0.6666666, -1], atol=1e-6)


def test_score_paf_lines_batch():
    pafs = tf.cast(tf.reshape(tf.range(6 * 4 * 2), [1, 6, 4, 2]), tf.float32)
    peaks = tf.constant([[[0, 0], [4, 8]]], tf.float32)
    peak_channel_inds = tf.constant([[0, 1]], tf.int32)
    skeleton_edges = tf.constant([[0, 1], [1, 2], [2, 3]], tf.int32)
    n_line_points = 3
    pafs_stride = 2
    max_edge_length_ratio = 2 / 12
    dist_penalty_weight = 1.0
    n_nodes = 4

    edge_inds, edge_peak_inds, line_scores = score_paf_lines_batch(
        pafs,
        peaks,
        peak_channel_inds,
        skeleton_edges,
        n_line_points,
        pafs_stride,
        max_edge_length_ratio,
        dist_penalty_weight,
        n_nodes,
    )
    assert_array_equal(edge_inds.to_list(), [[0]])
    assert_array_equal(edge_peak_inds.to_list(), [[[0, 1]]])
    assert_allclose(line_scores.to_list(), [[24.27]], atol=1e-2)


def test_match_candidates_sample():
    edge_inds_sample = tf.constant([0, 0])
    edge_peak_inds_sample = tf.constant([[0, 1], [2, 1]])
    line_scores_sample = tf.constant([-0.5, 1.0])
    n_edges = 1

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = match_candidates_sample(
        edge_inds_sample, edge_peak_inds_sample, line_scores_sample, n_edges
    )

    src_peak_inds_k, _ = tf.unique(edge_peak_inds_sample[:, 0])
    dst_peak_inds_k, _ = tf.unique(edge_peak_inds_sample[:, 1])

    assert_array_equal(match_edge_inds, [0])
    assert_array_equal(match_src_peak_inds, [1])
    assert_array_equal(match_dst_peak_inds, [0])
    assert_array_equal(match_line_scores, [1.0])
    assert tf.gather(src_peak_inds_k, match_src_peak_inds)[0] == 2
    assert tf.gather(dst_peak_inds_k, match_dst_peak_inds)[0] == 1


def test_match_candidates_batch():
    row_ids = tf.constant([0, 0], dtype=tf.int32)
    edge_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([0, 0], dtype=tf.int32), row_ids
    )
    edge_peak_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([[0, 1], [2, 1]], dtype=tf.int32), row_ids
    )
    line_scores = tf.RaggedTensor.from_value_rowids(
        tf.constant([-0.5, 1.0], dtype=tf.float32), row_ids
    )
    n_edges = 1

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = match_candidates_batch(edge_inds, edge_peak_inds, line_scores, n_edges)

    assert isinstance(match_edge_inds, tf.RaggedTensor)
    assert isinstance(match_src_peak_inds, tf.RaggedTensor)
    assert isinstance(match_dst_peak_inds, tf.RaggedTensor)
    assert isinstance(match_line_scores, tf.RaggedTensor)
    assert_array_equal(match_edge_inds.flat_values, [0])
    assert_array_equal(match_src_peak_inds.flat_values, [1])
    assert_array_equal(match_dst_peak_inds.flat_values, [0])
    assert_array_equal(match_line_scores.flat_values, [1.0])


def test_group_instances_sample():
    peaks_sample = tf.reshape(tf.range(5 * 2, dtype=tf.float32), [5, 2])
    peak_scores_sample = tf.range(5, dtype=tf.float32)
    peak_channel_inds_sample = tf.constant([0, 1, 2, 0, 1], tf.int32)
    match_edge_inds_sample = tf.constant([0, 1, 0], tf.int32)
    match_src_peak_inds_sample = tf.constant([0, 0, 1], tf.int32)
    match_dst_peak_inds_sample = tf.constant([0, 0, 1], tf.int32)
    match_line_scores_sample = tf.ones([3], dtype=tf.float32)
    n_nodes = 3
    sorted_edge_inds = tf.constant((0, 1), dtype=tf.int32)
    edge_types = [EdgeType(0, 1), EdgeType(1, 2)]
    min_instance_peaks = 0

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = group_instances_sample(
        peaks_sample,
        peak_scores_sample,
        peak_channel_inds_sample,
        match_edge_inds_sample,
        match_src_peak_inds_sample,
        match_dst_peak_inds_sample,
        match_line_scores_sample,
        n_nodes,
        sorted_edge_inds,
        edge_types,
        min_instance_peaks,
    )

    assert_array_equal(
        predicted_instances,
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [
                [6.0, 7.0],
                [8.0, 9.0],
                [np.nan, np.nan],
            ],
        ],
    )
    assert_array_equal(predicted_peak_scores, [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])
    assert_array_equal(predicted_instance_scores, [2.0, 1.0])


def test_group_instances_batch():
    row_ids = tf.zeros([5], dtype=tf.int32)
    peaks = tf.RaggedTensor.from_value_rowids(
        tf.reshape(tf.range(5 * 2, dtype=tf.float32), [5, 2]), row_ids
    )
    peak_scores = tf.RaggedTensor.from_value_rowids(
        tf.range(5, dtype=tf.float32), row_ids
    )
    peak_channel_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([0, 1, 2, 0, 1], tf.int32), row_ids
    )
    row_ids_edges = tf.zeros([3], dtype=tf.int32)
    match_edge_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([0, 1, 0], tf.int32), row_ids_edges
    )
    match_src_peak_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([0, 0, 1], tf.int32), row_ids_edges
    )
    match_dst_peak_inds = tf.RaggedTensor.from_value_rowids(
        tf.constant([0, 0, 1], tf.int32), row_ids_edges
    )
    match_line_scores = tf.RaggedTensor.from_value_rowids(
        tf.ones([3], dtype=tf.float32), row_ids_edges
    )
    n_nodes = 3
    sorted_edge_inds = (0, 1)
    edge_types = [EdgeType(0, 1), EdgeType(1, 2)]
    min_instance_peaks = 0

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = group_instances_batch(
        peaks,
        peak_scores,
        peak_channel_inds,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
        n_nodes,
        sorted_edge_inds,
        edge_types,
        min_instance_peaks,
    )

    assert isinstance(predicted_instances, tf.RaggedTensor)
    assert isinstance(predicted_peak_scores, tf.RaggedTensor)
    assert isinstance(predicted_instance_scores, tf.RaggedTensor)

    assert_array_equal(
        predicted_instances.flat_values,
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [
                [6.0, 7.0],
                [8.0, 9.0],
                [np.nan, np.nan],
            ],
        ],
    )
    assert_array_equal(
        predicted_peak_scores.flat_values, [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]]
    )
    assert_array_equal(predicted_instance_scores.flat_values, [2.0, 1.0])


def test_toposort_edges():
    edge_inds = [
        (5, 7),
        (5, 8),
        (5, 9),
        (5, 6),
        (5, 11),
        (5, 12),
        (1, 0),
        (1, 3),
        (1, 2),
        (1, 10),
        (1, 13),
        (1, 14),
        (4, 5),
        (4, 1),
    ]

    edge_types = [EdgeType(src_node, dst_node) for src_node, dst_node in edge_inds]

    sorted_edge_inds = toposort_edges(edge_types)
    assert sorted_edge_inds == (12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    edge_inds = [
        (1, 4),
        (1, 5),
        (6, 8),
        (6, 7),
        (6, 9),
        (9, 10),
        (1, 0),
        (1, 3),
        (1, 2),
        (6, 1),
    ]
    edge_types = [EdgeType(src_node, dst_node) for src_node, dst_node in edge_inds]
    sorted_edge_inds = toposort_edges(edge_types)
    assert sorted_edge_inds == (2, 3, 4, 9, 5, 0, 1, 6, 7, 8)


def test_assign_connections_to_instances():
    connections = {
        EdgeType(src_node_ind=5, dst_node_ind=7): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0465653)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=8): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0607507)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=9): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9563284)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=6): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=1, score=0.5797864)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=11): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9892818)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=12): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.7557168)
        ],
        EdgeType(src_node_ind=1, dst_node_ind=0): [],
        EdgeType(src_node_ind=1, dst_node_ind=3): [],
        EdgeType(src_node_ind=1, dst_node_ind=2): [],
        EdgeType(src_node_ind=1, dst_node_ind=10): [],
        EdgeType(src_node_ind=1, dst_node_ind=13): [],
        EdgeType(src_node_ind=1, dst_node_ind=14): [],
        EdgeType(src_node_ind=4, dst_node_ind=5): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9735552)
        ],
        EdgeType(src_node_ind=4, dst_node_ind=1): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.31536198)
        ],
    }
    instance_assignments = assign_connections_to_instances(
        connections,
        min_instance_peaks=0,
        n_nodes=15,
    )
    assert instance_assignments == {
        PeakID(node_ind=5, peak_ind=0): 0,
        PeakID(node_ind=7, peak_ind=0): 0,
        PeakID(node_ind=8, peak_ind=0): 0,
        PeakID(node_ind=9, peak_ind=0): 0,
        PeakID(node_ind=6, peak_ind=1): 0,
        PeakID(node_ind=11, peak_ind=0): 0,
        PeakID(node_ind=12, peak_ind=0): 0,
        PeakID(node_ind=4, peak_ind=0): 1,
        PeakID(node_ind=1, peak_ind=0): 1,
    }

    # Now in topological order:
    edge_types = list(connections.keys())
    sorted_edge_inds = toposort_edges(edge_types)
    instance_assignments = assign_connections_to_instances(
        {
            edge_types[edge_ind]: connections[edge_types[edge_ind]]
            for edge_ind in sorted_edge_inds
        },
        min_instance_peaks=0,
        n_nodes=15,
    )
    assert all(x == 0 for x in instance_assignments.values())

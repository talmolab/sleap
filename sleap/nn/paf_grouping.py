"""This module provides a set of utilities for grouping peaks based on PAFs.

Part affinity fields (PAFs) are a representation used to resolve the peak grouping
problem for multi-instance pose estimation [1].

They are a convenient way to represent directed graphs with support in image space. For
each edge, a PAF can be represented by an image with two channels, corresponding to the
x and y components of a unit vector pointing along the direction of the underlying
directed graph formed by the connections of the landmarks belonging to an instance.

Given a pair of putatively connected landmarks, the agreement between the line segment
that connects them and the PAF vectors found at the coordinates along the same line can
be used as a measure of "connectedness". These scores can then be used to guide the
instance-wise grouping of landmarks.

This image space representation is particularly useful as it is amenable to neural
network-based prediction from unlabeled images.

A high-level API for grouping based on PAFs is provided through the `PAFScorer` class.

References:
    .. [1] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D
       Pose Estimation using Part Affinity Fields. In _CVPR_, 2017.
"""

import attr
from typing import Dict, List, Union, Tuple, Text
import tensorflow as tf
import numpy as np
import networkx as nx
from sleap.nn.utils import tf_linear_sum_assignment
from sleap.nn.config import MultiInstanceConfig


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PeakID:
    """Indices to uniquely identify a single peak.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        node_ind: Index of the node type (channel) of the peak.
        peak_ind: Index of the peak within its node type.
    """

    node_ind: int
    peak_ind: int


@attr.s(auto_attribs=True, slots=True, frozen=True)
class EdgeType:
    """Indices to uniquely identify a single edge type.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        src_node_ind: Index of the source node type within the skeleton edges.
        dst_node_ind: Index of the destination node type within the skeleton edges.
    """

    src_node_ind: int
    dst_node_ind: int


@attr.s(auto_attribs=True, slots=True)
class EdgeConnection:
    """Indices to specify a matched connection between two peaks.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        src_peak_ind: Index of the source peak within all peaks.
        dst_peak_ind: Index of the destination peak within all peaks.
        score: Score of the match.
    """

    src_peak_ind: int
    dst_peak_ind: int
    score: float


def get_connection_candidates(
    peak_channel_inds_sample: tf.Tensor, skeleton_edges: tf.Tensor, n_nodes: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find the indices of all the possible connections formed by the detected peaks.

    Args:
        peak_channel_inds_sample: The channel indices of the peaks found in a sample.
            This is a `tf.Tensor` of shape `(n_peaks,)` and dtype `tf.int32` that is
            used to represent a detected peak by its channel/node index in the skeleton.
        skeleton_edges: The indices of the nodes that form the skeleton graph as a
            `tf.Tensor` of shape `(n_edges, 2)` and dtype `tf.int32` where each row
            corresponds to the source and destination node indices.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.

    Returns:
        A tuple of `(edge_inds, edge_peak_inds)`.

        `edge_inds` is a `tf.Tensor` of shape `(n_candidates,)` indicating the indices
        of the edge that each of the candidate connections belongs to.

        `edge_peak_inds` is a `tf.Tensor` of shape `(n_candidates, 2)` with the indices
        of the peaks that form the source and destination of each candidate connection.
        This indexes into the input `peak_channel_inds_sample`.
    """
    peak_inds = tf.argsort(peak_channel_inds_sample)
    node_inds = tf.gather(peak_channel_inds_sample, peak_inds)

    node_grouped_peak_inds = tf.RaggedTensor.from_value_rowids(
        peak_inds, node_inds, nrows=n_nodes
    )  # (n_nodes, (n_peaks_k))
    edge_grouped_peak_inds = tf.gather(
        node_grouped_peak_inds, skeleton_edges
    )  # (n_edges, (n_src_peaks), (n_dst_peaks))

    n_skeleton_edges = tf.shape(skeleton_edges)[0]
    edge_inds = tf.TensorArray(
        tf.int32,
        size=n_skeleton_edges,
        infer_shape=False,
        element_shape=tf.TensorShape([None]),
    )  # (n_skeleton_edges, (n_src * n_dst))
    edge_peak_inds = tf.TensorArray(
        tf.int32,
        size=n_skeleton_edges,
        infer_shape=False,
        element_shape=tf.TensorShape([None, 2]),
    )  # (n_skeleton_edges, (n_src * n_dst), 2)

    for k in range(n_skeleton_edges):
        sd = edge_grouped_peak_inds[k]

        s, d = tf.meshgrid(sd[0], sd[1], indexing="ij")
        sd = tf.reshape(tf.stack([s, d], axis=2), [-1, 2])

        edge_inds = edge_inds.write(k, tf.tile([k], [tf.shape(sd)[0]]))
        edge_peak_inds = edge_peak_inds.write(k, sd)

    edge_inds = edge_inds.concat()
    edge_peak_inds = edge_peak_inds.concat()

    return edge_inds, edge_peak_inds


def make_line_subs(
    peaks_sample: tf.Tensor,
    edge_peak_inds: tf.Tensor,
    edge_inds: tf.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> tf.Tensor:
    """Create the lines between candidate connections for evaluating the PAFs.

    Args:
        peaks_sample: The detected peaks in a sample as a `tf.Tensor` of shape
            `(n_peaks, 2)` and dtype `tf.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `tf.Tensor` of shape `(n_candidates, 2)` and dtype `tf.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `tf.Tensor` of shape `(n_candidates,)` and dtype `tf.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The line subscripts as a `tf.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `tf.int32`. These subscripts can
        be used directly with `tf.gather_nd` to pull out the PAF values at the lines.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        The subscripts are interpolated via nearest neighbor, so multiple fractional
        coordinates may map on to the same pixel if the line is short.

    See also: get_connection_candidates
    """
    src_peaks = tf.gather(peaks_sample, edge_peak_inds[:, 0])
    dst_peaks = tf.gather(peaks_sample, edge_peak_inds[:, 1])
    n_candidates = tf.shape(src_peaks)[0]

    XY = tf.linspace(src_peaks, dst_peaks, n_line_points, axis=2)
    XY = tf.cast(
        tf.round(XY / pafs_stride), tf.int32
    )  # (n_candidates, 2, n_line_points)  # dim 1 is [x, y]
    XY = tf.gather(XY, [1, 0], axis=1)  # dim 1 is [row, col]
    # TODO: clip coords to size of pafs tensor?

    line_subs = tf.concat(
        [
            XY,
            tf.broadcast_to(
                tf.reshape(edge_inds, [-1, 1, 1]), [n_candidates, 1, n_line_points]
            ),
        ],
        axis=1,
    )
    line_subs = tf.transpose(
        line_subs, [0, 2, 1]
    )  # (n_candidates, n_line_points, 3) -- last dim is [row, col, edge_ind]

    line_subs = tf.stack(
        [
            line_subs * tf.reshape([1, 1, 2], [1, 1, 3]),
            line_subs * tf.reshape([1, 1, 2], [1, 1, 3])
            + tf.reshape([0, 0, 1], [1, 1, 3]),
        ],
        axis=2,
    )  # (n_candidates, n_line_points, 2, 3)
    # The last dim is [row, col, edge_ind], but for both PAF (x and y) edge channels.

    return line_subs


def get_paf_lines(
    pafs_sample: tf.Tensor,
    peaks_sample: tf.Tensor,
    edge_peak_inds: tf.Tensor,
    edge_inds: tf.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> tf.Tensor:
    """Gets the PAF values at the lines formed between all detected peaks in a sample.

    Args:
        pafs_sample: The PAFs for the sample as a `tf.Tensor` of shape
            `(height, width, 2 * n_edges)`.
        peaks_sample: The detected peaks in a sample as a `tf.Tensor` of shape
            `(n_peaks, 2)` and dtype `tf.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `tf.Tensor` of shape `(n_candidates, 2)` and dtype `tf.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `tf.Tensor` of shape `(n_candidates,)` and dtype `tf.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The PAF vectors at all of the line points as a `tf.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `tf.int32`. These subscripts can
        be used directly with `tf.gather_nd` to pull out the PAF values at the lines.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        If only the subscripts are needed, use `make_line_subs()` to generate the lines
        without retrieving the PAF vector at the line points.

    See also: get_connection_candidates, make_line_subs, score_paf_lines
    """
    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points, pafs_stride
    )
    lines = tf.gather_nd(pafs_sample, line_subs)
    return lines


def compute_distance_penalty(
    spatial_vec_lengths: tf.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> tf.Tensor:
    """Compute the distance penalty component of the PAF line integral score.

    Args:
        spatial_vec_lengths: Euclidean distance between candidate source and
            destination points as a `tf.float32` tensor of any shape (typically
            `(n_candidates, 1)`).
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The distance penalty for each candidate as a `tf.float32` tensor of the same
        shape as `spatial_vec_lengths`.

        The penalty will be 0 (when below the threshold) and -1 as the distance
        approaches infinity. This is then scaled by the `dist_penalty_weight`.

    Notes:
        The penalty is computed from the distances scaled by the max length:

        ```
        if distance <= max_edge_length:
            penalty = 0
        else:
            penalty = (max_edge_length / distance) - 1
        ```

        For example, if the max length is 10 and the distance is 20, then the penalty
        will be: `(10 / 20) - 1 == 0.5 - 1 == -0.5`.

    See also: score_paf_lines
    """
    return (
        tf.math.minimum((max_edge_length / spatial_vec_lengths) - 1, 0)
        * dist_penalty_weight
    )  # < 0 = longer than max


def score_paf_lines(
    paf_lines_sample: tf.Tensor,
    peaks_sample: tf.Tensor,
    edge_peak_inds_sample: tf.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> tf.Tensor:
    """Compute the connectivity score for each PAF line in a sample.

    Args:
        paf_lines_sample: The PAF vectors evaluated at the lines formed between
            candidate conncetions as a `tf.Tensor` of shape
            `(n_candidates, n_line_points, 2, 3)` dtype `tf.int32`. This can be
            generated by `get_paf_lines()`.
        peaks_sample: The detected peaks in a sample as a `tf.Tensor` of shape
            `(n_peaks, 2)` and dtype `tf.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale.
        edge_peak_inds_sample: A `tf.Tensor` of shape `(n_candidates, 2)` and dtype
            `tf.int32` with the indices of the peaks that form the source and
            destination of each candidate connection. This indexes into the input
            `peaks_sample`. Can be generated using `get_connection_candidates()`.
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The line scores as a `tf.Tensor` of shape `(n_candidates,)` and dtype
        `tf.float32`. Each score value is the average dot product between the PAFs and
        the normalized displacement vector between source and destination peaks.

        Scores range from roughly -1.5 to 1.0, where larger values indicate a better
        connectivity score for the candidate. Values can be larger or smaller due to
        prediction error.

    Notes:
        This function operates on a single sample (frame). For batches of multiple
        frames, use `score_paf_lines_batch()`.

    See also: get_paf_lines, score_paf_lines_batch, compute_distance_penalty
    """
    # Pull out points.
    src_peaks = tf.gather(
        peaks_sample, edge_peak_inds_sample[:, 0], axis=0
    )  # (n_candidates, 2)
    dst_peaks = tf.gather(
        peaks_sample, edge_peak_inds_sample[:, 1], axis=0
    )  # (n_candidates, 2)

    # Compute normalized spatial displacement vector
    spatial_vecs = dst_peaks - src_peaks
    spatial_vec_lengths = tf.norm(
        spatial_vecs, axis=1, keepdims=True
    )  # (n_candidates, 1)
    spatial_vecs /= spatial_vec_lengths  # (n_candidates, 2)

    # Compute similarity scores
    line_scores = tf.squeeze(
        paf_lines_sample @ tf.expand_dims(spatial_vecs, axis=2), axis=-1
    )  # (n_candidates, n_line_points)

    # Compute distance penalties
    dist_penalties = tf.squeeze(
        compute_distance_penalty(
            spatial_vec_lengths,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        ),
        axis=1,
    )  # (n_candidates,)

    # Compute average line scores with distance penalty.
    mean_line_scores = tf.reduce_mean(line_scores, axis=1)
    penalized_line_scores = mean_line_scores + dist_penalties  # (n_candidates,)

    return penalized_line_scores


def score_paf_lines_batch(
    pafs: tf.Tensor,
    peaks: tf.Tensor,
    peak_channel_inds: tf.RaggedTensor,
    skeleton_edges: tf.Tensor,
    n_line_points: int,
    pafs_stride: int,
    max_edge_length_ratio: float,
    dist_penalty_weight: float,
    n_nodes: int,
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Create and score PAF lines formed between connection candidates.

    Args:
        pafs: The batch of part affinity fields as a `tf.Tensor` of shape
            `(n_samples, height, width, 2 * n_edges)` and type `tf.float32`.
        peaks: The coordinates of the peaks grouped by sample as a `tf.RaggedTensor` of
            shape `(n_samples, (n_peaks), 2)`.
        peak_channel_inds: The channel (node) that each peak in `peaks` corresponds to
            as a `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype
            `tf.int32`.
        skeleton_edges: The indices of the nodes that form the skeleton graph as a
            `tf.Tensor` of shape `(n_edges, 2)` and dtype `tf.int32` where each row
            corresponds to the source and destination node indices.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks` will be divided by
            this value to adjust the indexing into the `pafs` tensor.
        max_edge_length_ratio: The maximum expected length of a connected pair of points
            in relative image units. Candidate connections above this length will be
            penalized during matching.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.

    Returns:
        A tuple of `(edge_inds, edge_peak_inds, line_scores)` with the connections and
        their scores based on the PAFs.

        `edge_inds`: Sample-grouped indices of the edge in the skeleton that each
        connection corresponds to as `tf.RaggedTensor` of shape
        `(n_samples, (n_candidates))` and dtype `tf.int32`.

        `edge_peak_inds`: Sample-grouped indices of the peaks that form each connection
        as a `tf.RaggedTensor` of shape `(n_samples, (n_candidates), 2)` and dtype
        `tf.int32`. The last axis corresponds to the `[source, destination]` peak
        indices. These index into the input `peak_channel_inds`.

        `line_scores`: Sample-grouped scores for each candidate connection as
        `tf.RaggedTensor` of shape `(n_samples, (n_candidates))` and dtype `tf.float32`.

    Notes:
        This function handles the looping over samples in the batch and applies:

        1. `get_connection_candidates()`: Find peaks that form connections.
        2. `get_paf_lines()`: Retrieve PAF vectors for each line.
        3. `score_paf_lines()`: Compute connectivity score for each candidate.

    See also: get_connection_candidates, get_paf_lines, score_paf_lines
    """
    max_edge_length = (
        max_edge_length_ratio
        * tf.cast(tf.reduce_max(tf.shape(pafs[0])), tf.float32)
        * pafs_stride
    )

    n_samples = tf.shape(pafs)[0]
    edge_inds = tf.TensorArray(
        size=n_samples,
        infer_shape=False,
        element_shape=tf.TensorShape([None]),
        dtype=tf.int32,
    )
    edge_peak_inds = tf.TensorArray(
        size=n_samples,
        infer_shape=False,
        element_shape=tf.TensorShape([None, 2]),
        dtype=tf.int32,
    )
    line_scores = tf.TensorArray(
        size=n_samples,
        infer_shape=False,
        element_shape=tf.TensorShape([None]),
        dtype=tf.float32,
    )
    sample_inds = tf.TensorArray(
        size=n_samples,
        infer_shape=False,
        element_shape=tf.TensorShape([None]),
        dtype=tf.int32,
    )

    for sample in range(n_samples):
        pafs_sample = pafs[sample]
        peaks_sample = peaks[sample]
        peak_channel_inds_sample = peak_channel_inds[sample]

        edge_inds_sample, edge_peak_inds_sample = get_connection_candidates(
            peak_channel_inds_sample, skeleton_edges, n_nodes
        )
        paf_lines_sample = get_paf_lines(
            pafs_sample,
            peaks_sample,
            edge_peak_inds_sample,
            edge_inds_sample,
            n_line_points,
            pafs_stride,
        )
        line_scores_sample = score_paf_lines(
            paf_lines_sample,
            peaks_sample,
            edge_peak_inds_sample,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        )
        n_candidates = tf.shape(edge_peak_inds_sample)[0]

        edge_inds = edge_inds.write(sample, edge_inds_sample)
        edge_peak_inds = edge_peak_inds.write(sample, edge_peak_inds_sample)
        line_scores = line_scores.write(sample, line_scores_sample)
        sample_inds = sample_inds.write(sample, tf.repeat([sample], [n_candidates]))

    edge_inds = edge_inds.concat()
    edge_peak_inds = edge_peak_inds.concat()
    line_scores = line_scores.concat()
    sample_inds = sample_inds.concat()

    edge_inds = tf.RaggedTensor.from_value_rowids(
        edge_inds, sample_inds, nrows=n_samples
    )
    edge_peak_inds = tf.RaggedTensor.from_value_rowids(
        edge_peak_inds, sample_inds, nrows=n_samples
    )
    line_scores = tf.RaggedTensor.from_value_rowids(
        line_scores, sample_inds, nrows=n_samples
    )

    return (
        edge_inds,
        edge_peak_inds,
        line_scores,
    )


def match_candidates_sample(
    edge_inds_sample: tf.Tensor,
    edge_peak_inds_sample: tf.Tensor,
    line_scores_sample: tf.Tensor,
    n_edges: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Match candidate connections for a sample based on PAF scores.

    Args:
        edge_inds_sample: A `tf.Tensor` of shape `(n_candidates,)` and dtype `tf.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to for the sample. Can be generated using
            `get_connection_candidates()`.
        edge_peak_inds_sample: A `tf.Tensor` of shape `(n_candidates, 2)` and dtype
            `tf.int32` with the indices of the peaks that form the source and
            destination of each candidate connection. Can be generated using
            `get_connection_candidates()`.
        line_scores_sample: Scores for each candidate connection in the sample as a
            `tf.Tensor` of shape `(n_candidates,)` and dtype `tf.float32`. Can be
            generated using `score_paf_lines()`.
        n_edges: A scalar `int` denoting the number of edges in the skeleton.

    Returns:
        The connection peaks for each edge matched based on score as tuple of
        `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

        `match_edge_inds`: Indices of the skeleton edge that each connection corresponds
        to as a `tf.Tensor` of shape `(n_connections,)` and dtype `tf.int32`.

        `match_src_peak_inds`: Indices of the source peaks that form each connection
        as a `tf.Tensor` of shape `(n_connections,)` and dtype `tf.int32`. Important:
        These indices correspond to the edge-grouped peaks, not the set of all peaks in
        the sample.

        `match_dst_peak_inds`: Indices of the destination peaks that form each
        connection as a `tf.Tensor` of shape `(n_connections,)` and dtype `tf.int32`.
        Important: These indices correspond to the edge-grouped peaks, not the set of
        all peaks in the sample.

        `match_line_scores`: PAF line scores of the matched connections as a `tf.Tensor`
        of shape `(n_connections,)` and dtype `tf.float32`.

    Notes:
        The matching is performed using the Munkres algorithm implemented in
        `scipy.optimize.linear_sum_assignment()` which is wrapped in
        `tf_linear_sum_assignment()` for execution within a graph.

    See also: match_candidates_batch
    """
    match_edge_inds = tf.TensorArray(
        tf.int32, size=n_edges, infer_shape=False, element_shape=[None]
    )
    match_src_peak_inds = tf.TensorArray(
        tf.int32, size=n_edges, infer_shape=False, element_shape=[None]
    )
    match_dst_peak_inds = tf.TensorArray(
        tf.int32, size=n_edges, infer_shape=False, element_shape=[None]
    )
    match_line_scores = tf.TensorArray(
        tf.float32, size=n_edges, infer_shape=False, element_shape=[None]
    )

    for k in range(n_edges):

        is_edge_k = tf.squeeze(tf.where(edge_inds_sample == k), axis=1)
        edge_peak_inds_k = tf.gather(edge_peak_inds_sample, is_edge_k, axis=0)
        line_scores_k = tf.gather(line_scores_sample, is_edge_k, axis=0)

        # Get the unique peak indices
        src_peak_inds_k, _ = tf.unique(edge_peak_inds_k[:, 0])
        dst_peak_inds_k, _ = tf.unique(edge_peak_inds_k[:, 1])

        n_src = tf.shape(src_peak_inds_k)[0]
        n_dst = tf.shape(dst_peak_inds_k)[0]

        # Reshape line scores into cost matrix (n_src, n_dst)
        scores_matrix = tf.reshape(line_scores_k, [n_src, n_dst])

        # Replace NaNs with inf since linear_sum_assignment doesn't accept NaNs and flip
        # sign.
        cost_matrix = tf.where(
            condition=tf.math.is_nan(scores_matrix),
            x=tf.constant([np.inf]),
            y=-scores_matrix,
        )

        # Match
        match_src_inds, match_dst_inds = tf_linear_sum_assignment(cost_matrix)

        # Pull out matched scores.
        match_subs = tf.stack([match_src_inds, match_dst_inds], axis=1)
        match_line_scores_k = tf.gather_nd(scores_matrix, match_subs)

        # Get the peak indices for the matched points (these index into peaks_sample)
        # match_src_peak_inds_k = tf.gather(src_peak_inds_k, match_src_inds)
        # match_dst_peak_inds_k = tf.gather(dst_peak_inds_k, match_dst_inds)
        # These index into the edge-grouped peaks
        match_src_peak_inds_k = match_src_inds
        match_dst_peak_inds_k = match_dst_inds

        # Save
        match_edge_inds = match_edge_inds.write(
            k, tf.repeat([k], [tf.shape(match_src_peak_inds_k)[0]])
        )
        match_src_peak_inds = match_src_peak_inds.write(k, match_src_peak_inds_k)
        match_dst_peak_inds = match_dst_peak_inds.write(k, match_dst_peak_inds_k)
        match_line_scores = match_line_scores.write(k, match_line_scores_k)

    match_edge_inds = match_edge_inds.concat()
    match_src_peak_inds = match_src_peak_inds.concat()
    match_dst_peak_inds = match_dst_peak_inds.concat()
    match_line_scores = match_line_scores.concat()

    return (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )


def match_candidates_batch(
    edge_inds: tf.RaggedTensor,
    edge_peak_inds: tf.RaggedTensor,
    line_scores: tf.RaggedTensor,
    n_edges: int,
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Match candidate connections for a batch based on PAF scores.

    Args:
        edge_inds: Sample-grouped edge indices as a `tf.RaggedTensor` of shape
            `(n_samples, (n_candidates))` and dtype `tf.int32` indicating the indices
            of the edge that each of the candidate connections belongs to. Can be
            generated using `score_paf_lines_batch()`.
        edge_peak_inds: Sample-grouped indices of the peaks that form the source and
            destination of each candidate connection as a `tf.RaggedTensor` of shape
            `(n_samples, (n_candidates), 2)` and dtype `tf.int32`. Can be generated
            using `score_paf_lines_batch()`.
        line_scores: Sample-grouped scores for each candidate connection as a
            `tf.RaggedTensor` of shape `(n_samples, (n_candidates))` and dtype
            `tf.float32`. Can be generated using `score_paf_lines_batch()`.
        n_edges: A scalar `int` denoting the number of edges in the skeleton.

    Returns:
        The connection peaks for each edge matched based on score as tuple of
        `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

        `match_edge_inds`: Sample-grouped indices of the skeleton edge for each
        connection as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))` and
        dtype `tf.int32`.

        `match_src_peak_inds`: Sample-grouped indices of the source peaks that form each
        connection as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))` and
        dtype `tf.int32`. Important: These indices correspond to the edge-grouped peaks,
        not the set of all peaks in the sample.

        `match_dst_peak_inds`: Sample-grouped indices of the destination peaks that form
        each connection as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))`
        and dtype `tf.int32`. Important: These indices correspond to the edge-grouped
        peaks, not the set of all peaks in the sample.

        `match_line_scores`: Sample-grouped PAF line scores of the matched connections
        as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))` and dtype
        `tf.float32`.

    Notes:
        The matching is performed using the Munkres algorithm implemented in
        `scipy.optimize.linear_sum_assignment()` which is wrapped in
        `tf_linear_sum_assignment()` for execution within a graph.

    See also: match_candidates_sample, score_paf_lines_batch, group_instances_batch
    """
    n_samples = edge_inds.nrows()

    match_sample_inds = tf.TensorArray(
        tf.int32, size=n_samples, infer_shape=False, element_shape=[None]
    )
    match_edge_inds = tf.TensorArray(
        tf.int32, size=n_samples, infer_shape=False, element_shape=[None]
    )
    match_src_peak_inds = tf.TensorArray(
        tf.int32, size=n_samples, infer_shape=False, element_shape=[None]
    )
    match_dst_peak_inds = tf.TensorArray(
        tf.int32, size=n_samples, infer_shape=False, element_shape=[None]
    )
    match_line_scores = tf.TensorArray(
        tf.float32, size=n_samples, infer_shape=False, element_shape=[None]
    )

    for sample in range(n_samples):
        edge_inds_sample = edge_inds[sample]
        edge_peak_inds_sample = edge_peak_inds[sample]
        line_scores_sample = line_scores[sample]

        (
            match_edge_inds_sample,
            match_src_peak_inds_sample,
            match_dst_peak_inds_sample,
            match_line_scores_sample,
        ) = match_candidates_sample(
            edge_inds_sample,
            edge_peak_inds_sample,
            line_scores_sample,
            n_edges,
        )

        # Save
        match_sample_inds = match_sample_inds.write(
            sample, tf.repeat([sample], [tf.shape(match_edge_inds_sample)[0]])
        )
        match_edge_inds = match_edge_inds.write(sample, match_edge_inds_sample)
        match_src_peak_inds = match_src_peak_inds.write(
            sample, match_src_peak_inds_sample
        )
        match_dst_peak_inds = match_dst_peak_inds.write(
            sample, match_dst_peak_inds_sample
        )
        match_line_scores = match_line_scores.write(sample, match_line_scores_sample)

    match_sample_inds = match_sample_inds.concat()
    match_edge_inds = match_edge_inds.concat()
    match_src_peak_inds = match_src_peak_inds.concat()
    match_dst_peak_inds = match_dst_peak_inds.concat()
    match_line_scores = match_line_scores.concat()

    match_edge_inds = tf.RaggedTensor.from_value_rowids(
        match_edge_inds, match_sample_inds, nrows=n_samples
    )
    match_src_peak_inds = tf.RaggedTensor.from_value_rowids(
        match_src_peak_inds, match_sample_inds, nrows=n_samples
    )
    match_dst_peak_inds = tf.RaggedTensor.from_value_rowids(
        match_dst_peak_inds, match_sample_inds, nrows=n_samples
    )
    match_line_scores = tf.RaggedTensor.from_value_rowids(
        match_line_scores, match_sample_inds, nrows=n_samples
    )

    return (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )


def assign_connections_to_instances(
    connections: Dict[EdgeType, List[EdgeConnection]],
    min_instance_peaks: Union[int, float] = 0,
    n_nodes: int = None,
) -> Dict[PeakID, int]:
    """Assigns connected edges to instances via greedy graph partitioning.

    Args:
        connections: A dict that maps EdgeType to a list of EdgeConnections found
            through connection scoring. This can be generated by the
            filter_connection_candidates function.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a float in the
            range (0., 1.] is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an integer is provided, this is the
            absolute minimum number of peaks.
        n_nodes: Total node type count. Used to convert min_instance_peaks to an
            absolute number when a fraction is specified. If not provided, the node
            count is inferred from the unique node inds in connections.

    Returns:
        instance_assignments: A dict mapping PeakID to a unique instance ID specified
        as an integer.

        A PeakID is a tuple of (node_type_ind, peak_ind), where the peak_ind is the
        index or identifier specified in a EdgeConnection as a src_peak_ind or
        dst_peak_ind.

    Note:
        Instance IDs are not necessarily consecutive since some instances may be
        filtered out during the partitioning or filtering.

        This function expects connections from a single sample/frame!
    """
    # Grouping table that maps PeakID(node_ind, peak_ind) to an instance_id.
    instance_assignments = dict()

    # Loop through edge types.
    for edge_type, edge_connections in connections.items():

        # Loop through connections for the current edge.
        for connection in edge_connections:

            # Notation: specific peaks are identified by (node_ind, peak_ind).
            src_id = PeakID(edge_type.src_node_ind, connection.src_peak_ind)
            dst_id = PeakID(edge_type.dst_node_ind, connection.dst_peak_ind)

            # Get instance assignments for the connection peaks.
            src_instance = instance_assignments.get(src_id, None)
            dst_instance = instance_assignments.get(dst_id, None)

            if src_instance is None and dst_instance is None:
                # Case 1: Neither peak is assigned to an instance yet. We'll create a
                # new instance to hold both.
                new_instance = max(instance_assignments.values(), default=-1) + 1
                instance_assignments[src_id] = new_instance
                instance_assignments[dst_id] = new_instance

            elif src_instance is not None and dst_instance is None:
                # Case 2: The source peak is assigned already, but not the destination
                # peak. We'll assign the destination peak to the same instance as the
                # source.
                instance_assignments[dst_id] = src_instance

            elif src_instance is not None and dst_instance is not None:
                # Case 3: Both peaks have been assigned. We'll update the destination
                # peak to be a part of the source peak instance.
                instance_assignments[dst_id] = src_instance

                # We'll also check if they form disconnected subgraphs, in which case
                # we'll merge them by assigning all peaks belonging to the destination
                # peak's instance to the source peak's instance.
                src_instance_nodes = set(
                    peak_id.node_ind
                    for peak_id, instance in instance_assignments.items()
                    if instance == src_instance
                )
                dst_instance_nodes = set(
                    peak_id.node_ind
                    for peak_id, instance in instance_assignments.items()
                    if instance == dst_instance
                )

                if len(src_instance_nodes.intersection(dst_instance_nodes)) == 0:
                    for peak_id in instance_assignments:
                        if instance_assignments[peak_id] == dst_instance:
                            instance_assignments[peak_id] = src_instance

    if min_instance_peaks > 0:
        if isinstance(min_instance_peaks, float):

            if n_nodes is None:
                # Infer number of nodes if not specified.
                all_node_types = set()
                for edge_type in connections:
                    all_node_types.add(edge_type.src_node_ind)
                    all_node_types.add(edge_type.dst_node_ind)
                n_nodes = len(all_node_types)

            # Calculate minimum threshold.
            min_instance_peaks = int(min_instance_peaks * n_nodes)

        # Compute instance peak counts.
        instance_ids, instance_peak_counts = np.unique(
            list(instance_assignments.values()), return_counts=True
        )
        instance_peak_counts = {
            instance: peaks_count
            for instance, peaks_count in zip(instance_ids, instance_peak_counts)
        }

        # Filter out small instances.
        instance_assignments = {
            peak_id: instance
            for peak_id, instance in instance_assignments.items()
            if instance_peak_counts[instance] >= min_instance_peaks
        }

    return instance_assignments


def make_predicted_instances(
    peaks: np.array,
    peak_scores: np.array,
    connections: List[EdgeConnection],
    instance_assignments: Dict[PeakID, int],
) -> Tuple[np.array, np.array, np.array]:
    """Group peaks by assignments and accumulate scores.

    Args:
        peaks: Node-grouped peaks
        peak_scores: Node-grouped peak scores
        connections: `EdgeConnection`s grouped by edge type
        instance_assignments: `PeakID` to instance ID mapping

    Returns:
        Tuple of (predicted_instances, predicted_peak_scores, predicted_instance_scores)

        predicted_instances: (n_instances, n_nodes, 2) array
        predicted_peak_scores: (n_instances, n_nodes) array
        predicted_instance_scores: (n_instances,) array
    """
    # Ensure instance IDs are contiguous.
    instance_ids, instance_inds = np.unique(
        list(instance_assignments.values()), return_inverse=True
    )
    for peak_id, instance_ind in zip(instance_assignments.keys(), instance_inds):
        instance_assignments[peak_id] = instance_ind
    n_instances = len(instance_ids)

    # Compute instance scores as the sum of all edge scores.
    predicted_instance_scores = np.full((n_instances,), 0.0, dtype="float32")

    for edge_type, edge_connections in connections.items():
        # Loop over all connections for this edge type.
        for edge_connection in edge_connections:
            # Look up the source peak.
            src_peak_id = PeakID(
                node_ind=edge_type.src_node_ind, peak_ind=edge_connection.src_peak_ind
            )
            if src_peak_id in instance_assignments:
                # Add to the total instance score.
                instance_ind = instance_assignments[src_peak_id]
                predicted_instance_scores[instance_ind] += edge_connection.score

                # Sanity check: both peaks in the edge should have been assigned to the
                # same instance.
                dst_peak_id = PeakID(
                    node_ind=edge_type.dst_node_ind,
                    peak_ind=edge_connection.dst_peak_ind,
                )
                assert instance_ind == instance_assignments[dst_peak_id]

    # Fill out instances and peak scores.
    n_nodes = len(peaks)
    predicted_instances = np.full((n_instances, n_nodes, 2), np.nan, dtype="float32")
    predicted_peak_scores = np.full((n_instances, n_nodes), np.nan, dtype="float32")
    for peak_id, instance_ind in instance_assignments.items():
        predicted_instances[instance_ind, peak_id.node_ind, :] = peaks[
            peak_id.node_ind
        ][peak_id.peak_ind]
        predicted_peak_scores[instance_ind, peak_id.node_ind] = peak_scores[
            peak_id.node_ind
        ][peak_id.peak_ind]

    return predicted_instances, predicted_peak_scores, predicted_instance_scores


def group_instances_sample(
    peaks_sample: tf.Tensor,
    peak_scores_sample: tf.Tensor,
    peak_channel_inds_sample: tf.Tensor,
    match_edge_inds_sample: tf.Tensor,
    match_src_peak_inds_sample: tf.Tensor,
    match_dst_peak_inds_sample: tf.Tensor,
    match_line_scores_sample: tf.Tensor,
    n_nodes: int,
    sorted_edge_inds: Tuple[int],
    edge_types: List[EdgeType],
    min_instance_peaks: int,
    min_line_scores: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group matched connections into full instances for a single sample.

    Args:
        peaks_sample: The detected peaks in a sample as a `tf.Tensor` of shape
            `(n_peaks, 2)` and dtype `tf.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale.
        peak_scores_sample: The scores of the detected peaks in a sample as a
            `tf.Tensor` of shape `(n_peaks,)` and dtype `tf.float32`.
        peak_channel_inds_sample: The indices of the channel (node) that each detected
            peak is associated with as a `tf.Tensor` of shape `(n_peaks,)` and dtype
            `tf.int32`.
        match_edge_inds_sample: Indices of the skeleton edge that each connection
            corresponds to as a `tf.Tensor` of shape `(n_connections,)` and dtype
            `tf.int32`. This can be generated by `match_candidates_sample()`.
        match_src_peak_inds_sample: Indices of the source peaks that form each
            connection as a `tf.Tensor` of shape `(n_connections,)` and dtype
            `tf.int32`. Important: These indices correspond to the edge-grouped peaks,
            not the set of all peaks in the sample. This can be generated by
            `match_candidates_sample()`.
        match_dst_peak_inds_sample: Indices of the destination peaks that form each
            connection as a `tf.Tensor` of shape `(n_connections,)` and dtype
            `tf.int32`. Important: These indices correspond to the edge-grouped peaks,
            not the set of all peaks in the sample. This can be generated by
            `match_candidates_sample()`.
        match_line_scores_sample: PAF line scores of the matched connections as a
            `tf.Tensor` of shape `(n_connections,)` and dtype `tf.float32`. This can be
            generated by `match_candidates_sample()`.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).
        edge_types: A list of `EdgeType`s associated with the skeleton.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a `float` in the
            range `(0., 1.]` is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an `int` is provided, this is the
            absolute minimum number of peaks.
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs.

    Returns:
        A tuple of arrays with the grouped instances:

        `predicted_instances`: The grouped coordinates for each instance as an array of
        shape `(n_instances, n_nodes, 2)` and dtype `float32`. Missing peaks are
        represented by `np.NaN`s.

        `predicted_peak_scores`: The confidence map values for each peak as an array of
        `(n_instances, n_nodes)` and dtype `float32`.

        `predicted_instance_scores`: The grouping score for each instance as an array of
        shape `(n_instances,)` and dtype `float32`.

    Notes:
        This function is meant to be run as a `tf.py_function` within a graph (see
        `group_instances_batch()`).
    """
    if isinstance(peaks_sample, tf.Tensor):
        # Convert all the data to numpy arrays.
        peaks_sample = peaks_sample.numpy()
        peak_scores_sample = peak_scores_sample.numpy()
        peak_channel_inds_sample = peak_channel_inds_sample.numpy()
        match_edge_inds_sample = match_edge_inds_sample.numpy()
        match_src_peak_inds_sample = match_src_peak_inds_sample.numpy()
        match_dst_peak_inds_sample = match_dst_peak_inds_sample.numpy()
        match_line_scores_sample = match_line_scores_sample.numpy()
        sorted_edge_inds = sorted_edge_inds.numpy()

    # Filter out low scoring matches.
    is_valid_match = match_line_scores_sample >= min_line_scores
    match_edge_inds_sample = match_edge_inds_sample[is_valid_match]
    match_src_peak_inds_sample = match_src_peak_inds_sample[is_valid_match]
    match_dst_peak_inds_sample = match_dst_peak_inds_sample[is_valid_match]
    match_line_scores_sample = match_line_scores_sample[is_valid_match]

    # Group peaks by channel.
    peaks = []
    peak_scores = []
    for i in range(n_nodes):
        in_channel = peak_channel_inds_sample == i
        peaks.append(peaks_sample[in_channel])
        peak_scores.append(peak_scores_sample[in_channel])

    # Group connection data by edge in sorted order.
    # Note: This step is crucial since the instance assembly depends on the ordering
    # of the edges.
    connections = {}
    for edge_ind in sorted_edge_inds:
        in_edge = match_edge_inds_sample == edge_ind
        edge_type = edge_types[edge_ind]

        src_peak_inds = match_src_peak_inds_sample[in_edge]
        dst_peak_inds = match_dst_peak_inds_sample[in_edge]
        line_scores = match_line_scores_sample[in_edge]

        connections[edge_type] = [
            EdgeConnection(src, dst, score)
            for src, dst, score in zip(src_peak_inds, dst_peak_inds, line_scores)
        ]

    # Bipartite graph partitioning to group connections into instances.
    instance_assignments = assign_connections_to_instances(
        connections,
        min_instance_peaks=min_instance_peaks,
        n_nodes=n_nodes,
    )

    # Gather the data by instance.
    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = make_predicted_instances(peaks, peak_scores, connections, instance_assignments)

    return predicted_instances, predicted_peak_scores, predicted_instance_scores


def group_instances_batch(
    peaks: tf.RaggedTensor,
    peak_vals: tf.RaggedTensor,
    peak_channel_inds: tf.RaggedTensor,
    match_edge_inds: tf.RaggedTensor,
    match_src_peak_inds: tf.RaggedTensor,
    match_dst_peak_inds: tf.RaggedTensor,
    match_line_scores: tf.RaggedTensor,
    n_nodes: int,
    sorted_edge_inds: Tuple[int],
    edge_types: List[EdgeType],
    min_instance_peaks: int,
    min_line_scores: float = 0.25,
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Group matched connections into full instances for a batch.

    Args:
        peaks: The sample-grouped detected peaks in a batch as a `tf.RaggedTensor` of
            shape `(n_samples, (n_peaks), 2)` and dtype `tf.float32`. These should be
            `(x, y)` coordinates of each peak in the image scale.
        peak_vals: The sample-grouped scores of the detected peaks in a batch as a
            `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype `tf.float32`.
        peak_channel_inds: The sample-grouped indices of the channel (node) that each
            detected peak is associated with as a `tf.RaggedTensor` of shape
            `(n_samples, (n_peaks))` and dtype `tf.int32`.
        match_edge_inds: Sample-grouped indices of the skeleton edge that each
            connection corresponds to as a `tf.RaggedTensor` of shape
            `(n_samples, (n_connections))` and dtype `tf.int32`. This can be generated
            by `match_candidates_batch()`.
        match_src_peak_inds: Sample-grouped indices of the source peaks that form each
            connection as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))`
            and dtype `tf.int32`. Important: These indices correspond to the
            edge-grouped peaks, not the set of all peaks in each sample. This can be
            generated by `match_candidates_batch()`.
        match_dst_peak_inds: Sample-grouped indices of the destination peaks that form
            each connection as a `tf.RaggedTensor` of shape
            `(n_samples, (n_connections))` and dtype `tf.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample. This can be generated by `match_candidates_batch()`.
        match_line_scores: Sample-grouped PAF line scores of the matched connections as
            a `tf.RaggedTensor` of shape `(n_samples, (n_connections))` and dtype
            `tf.float32`. This can be generated by `match_candidates_batch()`.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).
        edge_types: A list of `EdgeType`s associated with the skeleton.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a `float` in the
            range `(0., 1.]` is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an `int` is provided, this is the
            absolute minimum number of peaks.
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs.

    Returns:
        A tuple of arrays with the grouped instances for the whole batch grouped by
        sample:

        `predicted_instances`: The sample- and instance-grouped coordinates for each
        instance as `tf.RaggedTensor` of shape `(n_samples, (n_instances), n_nodes, 2)`
        and dtype `tf.float32`. Missing peaks are represented by `NaN`s.

        `predicted_peak_scores`: The sample- and instance-grouped confidence map values
        for each peak as an array of `(n_samples, (n_instances), n_nodes)` and dtype
        `tf.float32`.

        `predicted_instance_scores`: The sample-grouped instance grouping score for each
        instance as an array of shape `(n_samples, (n_instances))` and dtype
        `tf.float32`.

    See also: match_candidates_batch, group_instances_sample
    """

    def _group_instances_sample(
        peaks_sample,
        peak_scores_sample,
        peak_channel_inds_sample,
        match_edge_inds_sample,
        match_src_peak_inds_sample,
        match_dst_peak_inds_sample,
        match_line_scores_sample,
        n_nodes,
        sorted_edge_inds,
        min_instance_peaks,
    ):
        """Helper to avoid passing `EdgeType`s to `tf.py_function`."""
        return group_instances_sample(
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
            min_line_scores=min_line_scores,
        )

    n_samples = peaks.nrows()

    sample_inds = tf.TensorArray(
        tf.int32, size=n_samples, infer_shape=False, element_shape=[None]
    )
    predicted_instances = tf.TensorArray(
        size=n_samples,
        dtype=tf.float32,
        infer_shape=False,
        element_shape=[None, n_nodes, 2],
    )
    predicted_peak_scores = tf.TensorArray(
        size=n_samples,
        dtype=tf.float32,
        infer_shape=False,
        element_shape=[None, n_nodes],
    )
    predicted_instance_scores = tf.TensorArray(
        size=n_samples, dtype=tf.float32, infer_shape=False, element_shape=[None]
    )

    for sample in range(n_samples):

        # Call sample-wise function in Eager mode.
        (
            predicted_instances_sample,
            predicted_peak_scores_sample,
            predicted_instance_scores_sample,
        ) = tf.py_function(
            _group_instances_sample,
            inp=[
                peaks[sample],
                peak_vals[sample],
                peak_channel_inds[sample],
                match_edge_inds[sample],
                match_src_peak_inds[sample],
                match_dst_peak_inds[sample],
                match_line_scores[sample],
                n_nodes,
                sorted_edge_inds,
                # edge_types, # not serializable!
                min_instance_peaks,
            ],
            Tout=[tf.float32, tf.float32, tf.float32],
        )

        sample_inds = sample_inds.write(
            sample, tf.repeat([sample], [tf.shape(predicted_instances_sample)[0]])
        )
        predicted_instances = predicted_instances.write(
            sample, predicted_instances_sample
        )
        predicted_peak_scores = predicted_peak_scores.write(
            sample, predicted_peak_scores_sample
        )
        predicted_instance_scores = predicted_instance_scores.write(
            sample, predicted_instance_scores_sample
        )

    sample_inds = sample_inds.concat()
    predicted_instances = predicted_instances.concat()
    predicted_peak_scores = predicted_peak_scores.concat()
    predicted_instance_scores = predicted_instance_scores.concat()

    predicted_instances = tf.RaggedTensor.from_value_rowids(
        predicted_instances, sample_inds, nrows=n_samples
    )
    predicted_peak_scores = tf.RaggedTensor.from_value_rowids(
        predicted_peak_scores, sample_inds, nrows=n_samples
    )
    predicted_instance_scores = tf.RaggedTensor.from_value_rowids(
        predicted_instance_scores, sample_inds, nrows=n_samples
    )

    return predicted_instances, predicted_peak_scores, predicted_instance_scores


def toposort_edges(edge_types: List[EdgeType]) -> Tuple[int]:
    """Find a topological ordering for a list of edge types.

    Args:
        edge_types: A list of `EdgeType` instances describing a skeleton.

    Returns:
        A tuple of indices specifying the topological order that the edge types should
        be accessed in during instance assembly (`assign_connections_to_instances`).

        This is important to ensure that instances are assembled starting at the root
        of the skeleton and moving down.

    See also: assign_connections_to_instances
    """
    edges = [
        (edge_type.src_node_ind, edge_type.dst_node_ind) for edge_type in edge_types
    ]
    dg = nx.DiGraph(edges)
    root_ind = next(nx.topological_sort(dg))
    sorted_edges = nx.bfs_edges(dg, root_ind)
    sorted_edge_inds = tuple([edges.index(edge) for edge in sorted_edges])
    return sorted_edge_inds


@attr.s(auto_attribs=True)
class PAFScorer:
    """Scoring pipeline based on part affinity fields.

    This class facilitates grouping of predicted peaks based on PAFs. It holds a set of
    common parameters that are used across different steps of the pipeline.

    Attributes:
        part_names: List of string node names in the skeleton.
        edges: List of (src_node, dst_node) names in the skeleton.
        pafs_stride: Output stride of the part affinity fields. This will be used to
            adjust the peak coordinates from full image to PAF subscripts.
        max_edge_length_ratio: The maximum expected length of a connected pair of points
            as a fraction of the image size. Candidate connections longer than this
            length will be penalized during matching.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.
        n_points: Number of points to sample along the line integral.
        min_instance_peaks: Minimum number of peaks the instance should have to be
            considered a real instance. Instances with fewer peaks than this will be
            discarded (useful for filtering spurious detections).
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs. Useful for rejecting spurious detections when
            there are no better ones.
        edge_inds: The edges of the skeleton defined as a list of (source, destination)
            tuples of node indices. This is created automatically on initialization.
        edge_types: A list of `EdgeType` instances representing the edges of the
            skeleton. This is created automatically on initialization.
        n_nodes: The number of nodes in the skeleton as a scalar `int`. This is created
            automatically on initialization.
        n_edges: The number of edges in the skeleton as a scalar `int`. This is created
            automatically on initialization.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).

    Notes:
        This class provides high level APIs for grouping peaks into instances using
        PAFs.

        The algorithm has three steps:

            1. Find all candidate connections between peaks and compute their matching
            score based on the PAFs.

            2. Match candidate connections using the connectivity score such that no
            peak is used in two connections of the same type.

            3. Group matched connections into complete instances.

        In general, the output from a peak finder (such as multi-peak confidence map
        prediction network) can be passed into `PAFScorer.predict()` to get back
        complete instances.

        For finer control over the grouping pipeline steps, use the instance methods in
        this class or the lower level functions in `sleap.nn.paf_grouping`.
    """

    part_names: List[Text]
    edges: List[Tuple[Text, Text]]
    pafs_stride: int
    max_edge_length_ratio: float = 0.25
    dist_penalty_weight: float = 1.0
    n_points: int = 10
    min_instance_peaks: Union[int, float] = 0
    min_line_scores: float = 0.25

    edge_inds: List[Tuple[int, int]] = attr.ib(init=False)
    edge_types: List[EdgeType] = attr.ib(init=False)
    n_nodes: int = attr.ib(init=False)
    n_edges: int = attr.ib(init=False)
    sorted_edge_inds: Tuple[int] = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Cache some computed attributes on initialization."""
        self.edge_inds = [
            (self.part_names.index(src), self.part_names.index(dst))
            for (src, dst) in self.edges
        ]
        self.edge_types = [
            EdgeType(src_node, dst_node) for src_node, dst_node in self.edge_inds
        ]

        self.n_nodes = len(self.part_names)
        self.n_edges = len(self.edges)
        self.sorted_edge_inds = toposort_edges(self.edge_types)

    @classmethod
    def from_config(
        cls,
        config: MultiInstanceConfig,
        max_edge_length_ratio: float = 0.25,
        dist_penalty_weight: float = 1.0,
        n_points: int = 10,
        min_instance_peaks: Union[int, float] = 0,
        min_line_scores: float = 0.25,
    ) -> "PAFScorer":
        """Initialize the PAF scorer from a `MultiInstanceConfig` head config.

        Args:
            config: `MultiInstanceConfig` from `cfg.model.heads.multi_instance`.
            max_edge_length_ratio: The maximum expected length of a connected pair of
                points as a fraction of the image size. Candidate connections longer
                than this length will be penalized during matching.
            dist_penalty_weight: A coefficient to scale weight of the distance penalty
                as a scalar float. Set to values greater than 1.0 to enforce the
                distance penalty more strictly.
            min_edge_score: Minimum score required to classify a connection as correct.
            n_points: Number of points to sample along the line integral.
            min_instance_peaks: Minimum number of peaks the instance should have to be
                considered a real instance. Instances with fewer peaks than this will be
                discarded (useful for filtering spurious detections).
            min_line_scores: Minimum line score (between -1 and 1) required to form a
                match between candidate point pairs. Useful for rejecting spurious
                detections when there are no better ones.

        Returns:
            The initialized instance of `PAFScorer`.
        """
        return cls(
            part_names=config.confmaps.part_names,
            edges=config.pafs.edges,
            pafs_stride=config.pafs.output_stride,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
        )

    def score_paf_lines(
        self, pafs: tf.Tensor, peaks: tf.Tensor, peak_channel_inds: tf.Tensor
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        """Create and score PAF lines formed between connection candidates.

        Args:
            pafs: The batch of part affinity fields as a `tf.Tensor` of shape
                `(n_samples, height, width, 2 * n_edges)` and type `tf.float32`.
            peaks: The coordinates of the peaks grouped by sample as a `tf.RaggedTensor`
                of shape `(n_samples, (n_peaks), 2)`.
            peak_channel_inds: The channel (node) that each peak in `peaks` corresponds
                to as a `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype
                `tf.int32`.

        Returns:
            A tuple of `(edge_inds, edge_peak_inds, line_scores)` with the connections
            and their scores based on the PAFs.

            `edge_inds`: Sample-grouped indices of the edge in the skeleton that each
            connection corresponds to as `tf.RaggedTensor` of shape
            `(n_samples, (n_candidates))` and dtype `tf.int32`.

            `edge_peak_inds`: Sample-grouped indices of the peaks that form each
            connection as a `tf.RaggedTensor` of shape `(n_samples, (n_candidates), 2)`
            and dtype `tf.int32`. The last axis corresponds to the
            `[source, destination]` peak indices. These index into the input
            `peak_channel_inds`.

            `line_scores`: Sample-grouped scores for each candidate connection as a
            `tf.RaggedTensor` of shape `(n_samples, (n_candidates))` and dtype
            `tf.float32`.

        Notes:
            This is a convenience wrapper for the standalone `score_paf_lines_batch()`.

        See also: score_paf_lines_batch
        """
        return score_paf_lines_batch(
            pafs,
            peaks,
            peak_channel_inds,
            self.edge_inds,
            self.n_points,
            self.pafs_stride,
            self.max_edge_length_ratio,
            self.dist_penalty_weight,
            self.n_nodes,
        )

    def match_candidates(
        self,
        edge_inds: tf.RaggedTensor,
        edge_peak_inds: tf.RaggedTensor,
        line_scores: tf.RaggedTensor,
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        """Match candidate connections for a batch based on PAF scores.

        Args:
            edge_inds: Sample-grouped edge indices as a `tf.RaggedTensor` of shape
                `(n_samples, (n_candidates))` and dtype `tf.int32` indicating the
                indices of the edge that each of the candidate connections belongs to.
                Can be generated using `PAFScorer.score_paf_lines()`.
            edge_peak_inds: Sample-grouped indices of the peaks that form the source and
                destination of each candidate connection as a `tf.RaggedTensor` of shape
                `(n_samples, (n_candidates), 2)` and dtype `tf.int32`. Can be generated
                using `PAFScorer.score_paf_lines()`.
            line_scores: Sample-grouped scores for each candidate connection as a
                `tf.RaggedTensor` of shape `(n_samples, (n_candidates))` and dtype
                `tf.float32`. Can be generated using `PAFScorer.score_paf_lines()`.

        Returns:
            The connection peaks for each edge matched based on score as tuple of
            `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

            `match_edge_inds`: Sample-grouped indices of the skeleton edge for each
            connection as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))`
            and dtype `tf.int32`.

            `match_src_peak_inds`: Sample-grouped indices of the source peaks that form
            each connection as a `tf.RaggedTensor` of shape
            `(n_samples, (n_connections))` and dtype `tf.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample.

            `match_dst_peak_inds`: Sample-grouped indices of the destination peaks that
            form each connection as a `tf.RaggedTensor` of shape
            `(n_samples, (n_connections))` and dtype `tf.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample.

            `match_line_scores`: Sample-grouped PAF line scores of the matched
            connections as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))`
            and dtype `tf.float32`.

        Notes:
            This is a convenience wrapper for the standalone `match_candidates_batch()`.

        See also: PAFScorer.score_paf_lines, match_candidates_batch
        """
        return match_candidates_batch(
            edge_inds, edge_peak_inds, line_scores, self.n_edges
        )

    def group_instances(
        self,
        peaks: tf.RaggedTensor,
        peak_vals: tf.RaggedTensor,
        peak_channel_inds: tf.RaggedTensor,
        match_edge_inds: tf.RaggedTensor,
        match_src_peak_inds: tf.RaggedTensor,
        match_dst_peak_inds: tf.RaggedTensor,
        match_line_scores: tf.RaggedTensor,
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        """Group matched connections into full instances for a batch.

        Args:
            peaks: The sample-grouped detected peaks in a batch as a `tf.RaggedTensor`
                of shape `(n_samples, (n_peaks), 2)` and dtype `tf.float32`. These
                should be `(x, y)` coordinates of each peak in the image scale.
            peak_vals: The sample-grouped scores of the detected peaks in a batch as a
                `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype
                `tf.float32`.
            peak_channel_inds: The sample-grouped indices of the channel (node) that
                each detected peak is associated with as a `tf.RaggedTensor` of shape
                `(n_samples, (n_peaks))` and dtype `tf.int32`.
            match_edge_inds: Sample-grouped indices of the skeleton edge that each
                connection corresponds to as a `tf.RaggedTensor` of shape
                `(n_samples, (n_connections))` and dtype `tf.int32`. This can be
                generated by `PAFScorer.match_candidates()`.
            match_src_peak_inds: Sample-grouped indices of the source peaks that form
                each connection as a `tf.RaggedTensor` of shape
                `(n_samples, (n_connections))` and dtype `tf.int32`. Important: These
                indices correspond to the edge-grouped peaks, not the set of all peaks
                in each sample. This can be generated by `PAFScorer.match_candidates()`.
            match_dst_peak_inds: Sample-grouped indices of the destination peaks that
                form each connection as a `tf.RaggedTensor` of shape
                `(n_samples, (n_connections))` and dtype `tf.int32`. Important: These
                indices correspond to the edge-grouped peaks, not the set of all peaks
                in the sample. This can be generated by `PAFScorer.match_candidates()`.
            match_line_scores: Sample-grouped PAF line scores of the matched connections
                as a `tf.RaggedTensor` of shape `(n_samples, (n_connections))` and dtype
                `tf.float32`. This can be generated by `PAFScorer.match_candidates()`.

        Returns:
            A tuple of arrays with the grouped instances for the whole batch grouped by
            sample:

            `predicted_instances`: The sample- and instance-grouped coordinates for each
            instance as `tf.RaggedTensor` of shape
            `(n_samples, (n_instances), n_nodes, 2)` and dtype `tf.float32`. Missing
            peaks are represented by `NaN`s.

            `predicted_peak_scores`: The sample- and instance-grouped confidence map
            values for each peak as an array of `(n_samples, (n_instances), n_nodes)`
            and dtype `tf.float32`.

            `predicted_instance_scores`: The sample-grouped instance grouping score for
            each instance as an array of shape `(n_samples, (n_instances))` and dtype
            `tf.float32`.

        Notes:
            This is a convenience wrapper for the standalone `group_instances_batch()`.

        See also: PAFScorer.match_candidates, group_instances_batch
        """
        return group_instances_batch(
            peaks,
            peak_vals,
            peak_channel_inds,
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
            self.n_nodes,
            self.sorted_edge_inds,
            self.edge_types,
            self.min_instance_peaks,
            min_line_scores=self.min_line_scores,
        )

    def predict(
        self,
        pafs: tf.Tensor,
        peaks: tf.RaggedTensor,
        peak_vals: tf.RaggedTensor,
        peak_channel_inds: tf.RaggedTensor,
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        """Group a batch of predicted peaks into full instance predictions using PAFs.

        Args:
            pafs: The batch of part affinity fields as a `tf.Tensor` of shape
                `(n_samples, height, width, 2 * n_edges)` and type `tf.float32`.
            peaks: The coordinates of the peaks grouped by sample as a `tf.RaggedTensor`
                of shape `(n_samples, (n_peaks), 2)`.
            peak_vals: The sample-grouped scores of the detected peaks in a batch as a
                `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype
                `tf.float32`.
            peak_channel_inds: The channel (node) that each peak in `peaks` corresponds
                to as a `tf.RaggedTensor` of shape `(n_samples, (n_peaks))` and dtype
                `tf.int32`.

        Returns:
            A tuple of arrays with the grouped instances for the whole batch grouped by
            sample:

            `predicted_instances`: The sample- and instance-grouped coordinates for each
            instance as `tf.RaggedTensor` of shape
            `(n_samples, (n_instances), n_nodes, 2)` and dtype `tf.float32`. Missing
            peaks are represented by `NaN`s.

            `predicted_peak_scores`: The sample- and instance-grouped confidence map
            values for each peak as an array of `(n_samples, (n_instances), n_nodes)`
            and dtype `tf.float32`.

            `predicted_instance_scores`: The sample-grouped instance grouping score for
            each instance as an array of shape `(n_samples, (n_instances))` and dtype
            `tf.float32`.

        Notes:
            This is a high level API for grouping peaks into instances using PAFs.

            See the `PAFScorer` class documentation for more details on the algorithm.

        See also:
            PAFScorer.score_paf_lines, PAFScorer.match_candidates,
            PAFScorer.group_instances
        """
        edge_inds, edge_peak_inds, line_scores = self.score_paf_lines(
            pafs, peaks, peak_channel_inds
        )
        (
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
        ) = self.match_candidates(edge_inds, edge_peak_inds, line_scores)
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
        ) = self.group_instances(
            peaks,
            peak_vals,
            peak_channel_inds,
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
        )
        return predicted_instances, predicted_peak_scores, predicted_instance_scores

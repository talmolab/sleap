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

References:
    .. [1] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D
       Pose Estimation using Part Affinity Fields. In _CVPR_, 2017.
"""

from collections import defaultdict
from typing import Dict, List, Union, Tuple
import attr
import tensorflow as tf
import numpy as np

from sleap.instance import PredictedPoint, PredictedInstance
from sleap.skeleton import Skeleton


@attr.s(auto_attribs=True, slots=True)
class Peak:
    x: float
    y: float
    score: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PeakID:
    node_ind: int
    peak_ind: int


@attr.s(auto_attribs=True, slots=True, frozen=True)
class EdgeType:
    src_node_ind: int
    dst_node_ind: int


@attr.s(auto_attribs=True, slots=True)
class EdgeConnection:
    src_peak_ind: int
    dst_peak_ind: int
    score: float


def make_peaks_table(peaks: np.ndarray, peak_vals: np.ndarray) -> Dict[int, List[Peak]]:
    """Creates a lookup table that maps node indices to peaks.

    Args:
        peaks: Numpy array of shape (n_peaks, 4) where subscripts of peak locations are
            specified in each row as [sample, row, col, channel]. The sample indices are
            ignored since this function expects to operate on single samples.
        peak_vals: Numpy array of shape (n_peaks,) corresponding to the values at the
            peaks. These are interpreted as scores for each peak.

    Returns:
        A dictionary mapping node indices (channel) to a list of Peak instances.
    """

    peaks_table = defaultdict(list)
    for (_, y, x, node_ind), peak_val in zip(peaks, peak_vals):
        peaks_table[int(node_ind)].append(Peak(x, y, peak_val))

    return peaks_table


def find_edge_connections(
    src_peaks: List[Peak],
    dst_peaks: List[Peak],
    paf_x: np.ndarray,
    paf_y: np.ndarray,
    paf_x_offset: float = 0.0,
    paf_y_offset: float = 0.0,
    paf_scale: float = 1.0,
    n_edge_samples: int = 10,
    min_edge_score: float = 0.05,
    min_samples_correct: float = 0.8,
    max_edge_length: float = 0.5,
) -> List[EdgeConnection]:
    """Finds valid connections for an edge by using PAF scoring.

    This function implements the core PAF algorithm described in [1].

    Args:
        src_peaks: A list of Peaks for the source node type.
        dst_peaks: A list of Peaks for the destination node type.
        paf_x: A numpy array of shape (height, width) corresponding to the x channel
            component of the PAFs associated with this edge type.
        paf_y: A numpy array of shape (height, width) corresponding to the y channel
            component of the PAFs associated with this edge type.
        paf_x_offset: The x coordinate of left side of the image region from which the
            PAFs were predicted. This must be specified when using region proposals in
            order to adjust the peak coordinates from absolute image coordinates to
            their location within the PAF array.
        paf_y_offset: The y coordinate of top side of the image region from which the
            PAFs were predicted. This must be specified when using region proposals in
            order to adjust the peak coordinates from absolute image coordinates to
            their location within the PAF array.
        paf_scale: The scaling factor of the image from which the PAFs were predicted.
            This must be specified when the output scale of the PAF inference model is
            not equal to 1.0, i.e., when the PAF arrays are at a different resolution
            than the image. This is needed in order to adjust the peak coordinates from
            absolute image coordinates to their location within the PAF array.
        n_edge_samples: Number of PAF points to sample along the line segment defined by
            the edge.
        min_edge_score: Minimum score to count a sampled point as correct.
        min_samples_correct: Fraction of samples that must be correct in order to
            consider a connection as valid.
        max_edge_length: Maximum length of edge before being penalized, expressed
            as a fraction of PAF height height (accounting for scale).

    Returns:
        A list of valid EdgeConnections.

    References:
        .. [1] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D
           Pose Estimation using Part Affinity Fields. In _CVPR_, 2017.
    """

    # Score each pair of candidates.
    edge_connection_candidates = []
    for src_peak_ind, src_peak in enumerate(src_peaks):
        for dst_peak_ind, dst_peak in enumerate(dst_peaks):

            # Compute spatial edge vector from peak coordinates.
            spatial_vec = np.array([dst_peak.x - src_peak.x, dst_peak.y - src_peak.y])
            spatial_vec_length = np.sqrt(np.sum(spatial_vec ** 2)) + 1e-9
            spatial_unit_vec = spatial_vec / spatial_vec_length

            # Sample the along the line integral on the PAF slices.
            line_x_inds = np.clip(
                np.round(
                    (
                        np.linspace(src_peak.x, dst_peak.x, num=n_edge_samples)
                        - paf_x_offset
                    )
                    * paf_scale
                ),
                0,
                paf_x.shape[1] - 1,
            ).astype(int)
            line_y_inds = np.clip(
                np.round(
                    (
                        np.linspace(src_peak.y, dst_peak.y, num=n_edge_samples)
                        - paf_y_offset
                    )
                    * paf_scale
                ),
                0,
                paf_x.shape[0] - 1,
            ).astype(int)
            line_paf = np.stack(
                [paf_x[line_y_inds, line_x_inds], paf_y[line_y_inds, line_x_inds]],
                axis=1,
            )

            # Compute scores.
            score_line = np.dot(line_paf, spatial_unit_vec)
            score_line_avg = score_line.mean()
            dist_penalty = min(
                0,
                (max_edge_length * paf_x.shape[0] / spatial_vec_length * paf_scale) - 1,
            )
            score_with_dist_penalty = score_line_avg + dist_penalty

            # Criterion 1: There are sufficient sampled points along the line integral
            # that are above the minimum threshold.
            fraction_correct = (score_line > min_edge_score).mean()
            enough_correct_samples = fraction_correct > min_samples_correct
            if not enough_correct_samples:
                continue

            # Criterion 2: Average score with length penalty is positive.
            positive_score = score_with_dist_penalty > 0
            if not positive_score:
                continue

            # Add the candidates as a connection candidate for the current edge.
            edge_connection_candidates.append(
                EdgeConnection(
                    src_peak_ind,
                    dst_peak_ind,
                    score_with_dist_penalty + src_peak.score + dst_peak.score,
                )
            )

    # Sort candidates by descending score.
    candidate_inds = np.argsort(
        [connection.score for connection in edge_connection_candidates]
    )[::-1]

    # Find maximum number of possible connections.
    max_connections = min(len(src_peaks), len(dst_peaks))

    # Keep connections by greedily assigning edges with unused nodes.
    edge_connections = []
    used_src_peaks = set()
    used_dst_peaks = set()
    for candidate_ind in candidate_inds:
        connection = edge_connection_candidates[candidate_ind]

        if (
            connection.src_peak_ind not in used_src_peaks
            and connection.dst_peak_ind not in used_dst_peaks
        ):

            edge_connections.append(connection)
            used_src_peaks.add(connection.src_peak_ind)
            used_dst_peaks.add(connection.dst_peak_ind)

            if len(edge_connections) >= max_connections:
                break

    return edge_connections


def connect_edges(
    peaks_table: Dict[int, List[Peak]],
    pafs: np.ndarray,
    edges: List[EdgeType],
    **kwargs
) -> Dict[EdgeType, List[EdgeConnection]]:
    """Connects peaks via PAF scoring and groups them by edge type.

    Args:
        peaks_table: A dictionary mapping node indices (channel) to a list of Peak
            instances. This can be generated by make_peaks_table.
        pafs: Rank-3 numpy array of PAFs with shape (height, width, n_edges * 2).
        edges: List of edge types whose order corresponds to the PAF channels.

        This function also supports keyword-only args that are passed through to
            find_edge_connections.

    Returns:
        The connected_edges, a table that maps EdgeType(src_node_ind, dst_node_ind)
        to a list of EdgeConnections found through edge-wise PAF matching.
    """

    connected_edges = defaultdict(list)
    for edge_ind, edge in enumerate(edges):

        # Pull out peak data for this edge.
        src_peaks = peaks_table[edge.src_node_ind]
        dst_peaks = peaks_table[edge.dst_node_ind]

        # Skip if no peaks detected for either node on this edge.
        if len(src_peaks) == 0 or len(dst_peaks) == 0:
            continue

        # Pull out PAF slices.
        paf_x = pafs[..., 2 * edge_ind]
        paf_y = pafs[..., (2 * edge_ind) + 1]

        # Find connections for this edge.
        connected_edges[edge] = find_edge_connections(
            src_peaks, dst_peaks, paf_x, paf_y, **kwargs
        )

    return connected_edges


def group_to_instances(
    connected_edges: Dict[EdgeType, List[EdgeConnection]],
    min_instance_peaks: Union[int, float] = 0,
    n_nodes: int = None,
) -> Dict[PeakID, int]:
    """Groups connected edges into instances via greedy graph partitioning.

    Args:
        connected_edges: A table that maps EdgeType(src_node_ind, dst_node_ind) to a
            list of EdgeConnections found through matching.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a float in the
            range (0., 1.] is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an integer is provided, this is the
            absolute minimum number of peaks.
        n_nodes: Total node type count. Used to convert min_instance_peaks to an
            absolute number when a fraction is specified. If not provided, the node
            count is inferred from the unique node inds in connected_edges.

    Returns:
        instance_assignments: a dict mapping each PeakID(node_ind, peak_ind) to a unique
        instance ID specified as an integer.

    Note:
        Instance IDs are not necessarily consecutive since some instances may be
        filtered out during the partitioning or filtering.
    """

    # Grouping table that maps PeakID(node_ind, peak_ind) to an instance_id.
    instance_assignments = dict()

    # Loop through edge types.
    for edge_type, edge_connections in connected_edges.items():

        # Loop through connections for the current edge.
        for connection in edge_connections:

            # Notation: specific peaks are identified by (node_ind, peak_ind).
            src_id = PeakID(edge_type.src_node_ind, connection.src_peak_ind)
            dst_id = PeakID(edge_type.dst_node_ind, connection.dst_peak_ind)

            # Get instance assignments for the connection peaks.
            src_instance = instance_assignments.get(src_id, None)
            dst_instance = instance_assignments.get(dst_id, None)

            if src_instance is None and dst_instance is None:
                # Case 1: Neither peak is assigned to an instance yet. We'll create a new
                # instance to hold both.
                new_instance = max(instance_assignments.values(), default=-1) + 1
                instance_assignments[src_id] = new_instance
                instance_assignments[dst_id] = new_instance

            elif src_instance is not None and dst_instance is None:
                # Case 2: The source peak is assigned already, but not the destination peak.
                # We'll assign the destination peak to the same instance as the source.
                instance_assignments[dst_id] = src_instance

            elif src_instance is not None and dst_instance is not None:
                # Case 3: Both peaks have been assigned. We'll update the destination peak
                # to be a part of the source peak instance.
                instance_assignments[dst_id] = src_instance

                # We'll also check if they form disconnected subgraphs, in which case we'll
                # merge them by assigning all peaks belonging to the destination peak's
                # instance to the source peak's instance.
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
                for edge_type in connected_edges:
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


def create_predicted_instances(
    peaks_table: Dict[int, List[Peak]],
    connected_edges: Dict[EdgeType, List[EdgeConnection]],
    instance_assignments: Dict[PeakID, int],
    skeleton: Skeleton,
) -> List[PredictedInstance]:
    """Assembles grouping associations to generate a list of PredictedInstances.

    This function effectively serves as postprocessing to all of the grouping logic
    implemented in the other methods of this module.

    Args:
        peaks_table: A dictionary mapping node indices (channel) to a list of Peak
            instances. This can be generated by make_peaks_table.
        connected_edges: A dictionary that maps EdgeType(src_node_ind, dst_node_ind)
            to a list of EdgeConnections found through edge-wise PAF matching. This can
            be generated by connect_edges.
        instance_assignments: A dictionary mapping each PeakID(node_ind, peak_ind) to a
            unique instance ID specified as an integer. This can be generated by
            group_to_instances.
        skeleton: The Skeleton object to use for associating to the resulting predicted
            instances.

    Returns:
        A list of PredictedInstances created from the detection and grouping data.
    """

    # Create predicted points and group by instance and node.
    instance_points = defaultdict(dict)
    for peak_id, instance_ind in instance_assignments.items():
        node_ind, peak_ind = attr.astuple(peak_id)

        # Get the original peak data.
        peak = peaks_table[node_ind][peak_ind]

        # Update instance with the peak as a PredictedPoint.
        node_name = skeleton.node_names[node_ind]
        instance_points[instance_ind][node_name] = PredictedPoint(**attr.asdict(peak))

    # Accumulate instance scores by looping through the connections.
    instance_scores = defaultdict(int)
    for edge, edge_connections in connected_edges.items():
        src_node_ind, dst_node_ind = attr.astuple(edge)

        for connection in edge_connections:

            # Check for existence of the source peak in the instances.
            peak_id = PeakID(src_node_ind, connection.src_peak_ind)
            if peak_id in instance_assignments:

                # Accumulate connection score.
                instance_ind = instance_assignments[peak_id]
                instance_scores[instance_ind] += connection.score

    # Create PredictedInstances.
    predicted_instances = []
    for instance_ind in instance_points:
        predicted_instances.append(
            PredictedInstance(
                skeleton=skeleton,
                points=instance_points[instance_ind],
                score=instance_scores[instance_ind],
            )
        )

    return predicted_instances

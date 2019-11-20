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
from typing import Dict, DefaultDict, List, Union, Tuple
import attr
import tensorflow as tf
import numpy as np

from sleap.nn import model
from sleap.nn import utils
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


def find_peak_pairs(
    peaks: np.ndarray, edge_types: List[EdgeType], min_pair_distance: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds all combinations of peak pairs that define valid connections for grouping.
    
    Args:
        peaks: Peak coordinates with shape (N, 4) in subscript form, where each row
            specifies a peak as (sample, row, col, channel).
        edge_types: List of EdgeType instances defining the directed graph.
        min_pair_distance: Minimum Euclidean distance between a pair of peaks in order
            to consider it valid. If nonzero, this filters out closely located points
            that would have no support under PAF integrals.
            
    Returns:
        A tuple of src_peak_inds, dst_peak_inds, edge_type_inds.
        
        These are all numpy array vectors of length n_pairs, denoting the indices
        associated with each pairing.
        
        src_peak_inds and dst_peak_inds are indices into the rows of the peaks matrix.
        
        edge_type_inds is an index into the edge_types list.
    """

    # Group peaks by sample.
    peak_inds = np.arange(len(peaks))
    peak_samples = peaks[:, 0]
    sample_grouped_peak_inds = utils.group_array(peak_inds, peak_samples)

    # Find peak pairs that define edge candidates within each sample.
    src_peak_inds = []
    dst_peak_inds = []
    edge_type_inds = []
    for sample_peak_inds in sample_grouped_peak_inds.values():

        # Group peaks in this sample by skeleton node (confmap channel).
        node_grouped_peak_inds = utils.group_array(
            sample_peak_inds, peaks[sample_peak_inds, 3]
        )

        # Find valid connections for each edge type.
        for edge_type_ind, edge_type in enumerate(edge_types):

            # Skip if no peaks detected for either node on this edge.
            if (
                edge_type.src_node_ind not in node_grouped_peak_inds
                or edge_type.dst_node_ind not in node_grouped_peak_inds
            ):
                continue

            # Pull out data for this edge type.
            edge_src_peak_inds = node_grouped_peak_inds[edge_type.src_node_ind]
            edge_dst_peak_inds = node_grouped_peak_inds[edge_type.dst_node_ind]

            # Compute distances between every possible pair of peaks.
            dists = utils.compute_pairwise_distances(
                peaks[edge_src_peak_inds, 1:3], peaks[edge_dst_peak_inds, 1:3]
            )

            # Find the pairs that are not too close.
            valid_src_sub_inds, valid_dst_sub_inds = np.nonzero(
                dists >= min_pair_distance
            )

            # Save valid pairs.
            for src_peak_ind, dst_peak_ind in zip(
                edge_src_peak_inds[valid_src_sub_inds],
                edge_dst_peak_inds[valid_dst_sub_inds],
            ):
                src_peak_inds.append(src_peak_ind)
                dst_peak_inds.append(dst_peak_ind)
                edge_type_inds.append(edge_type_ind)

    # Convert to numpy arrays and return.
    return np.array(src_peak_inds), np.array(dst_peak_inds), np.array(edge_type_inds)


def make_line_segments(
    src_peaks: np.ndarray,
    dst_peaks: np.ndarray,
    edge_type_inds: np.ndarray,
    n_edge_samples: int = 5,
) -> np.ndarray:
    """Interpolates coordinates along the line formed by each pair of peaks.
    
    Args:
        src_peaks: Peak coordinates with shape (n_pairs, 4) in subscript form.
        dst_peaks: Peak coordinates with shape (n_pairs, 4) in subscript form.
        edge_type_inds: Array of shape (n_pairs,) specifying the edge indices that each
            pair of peaks corresponds to.
        n_edge_samples: Number of points to sample between each pair of peaks.

    Returns:
        line_segments: A numpy array of shape (n_pairs * n_edge_samples, 4) such that
        each row specifies the coordinates of a point in a sampled line segment in the
        form:
            sample_ind, row_ind, col_ind, edge_type_ind = line_segments[i]
    """
    # Sample line segments along each pair of peaks.
    line_segments = np.linspace(
        src_peaks[:, 1:3],
        dst_peaks[:, 1:3],
        num=n_edge_samples,
        dtype="float32",
        axis=1,
    )  # (n_pairs, n_edge_samples, 2)

    # Discretize to the PAF image space.
    line_segments = np.round(line_segments).astype("int64")

    # Concatenate the sample and edge subscripts so the coordinates are fully specified.
    sample_inds = src_peaks[:, 0]
    line_segments = np.concatenate(
        [
            np.tile(
                sample_inds.astype("int64").reshape(-1, 1, 1), (1, n_edge_samples, 1)
            ),
            line_segments,
            np.tile(
                edge_type_inds.astype("int64").reshape(-1, 1, 1), (1, n_edge_samples, 1)
            ),
        ],
        axis=-1,
    )  # (n_pairs, n_edge_samples, 4)

    # Flatten into (n_pairs * n_edge_samples, 4).
    line_segments = line_segments.reshape(-1, 4)

    return line_segments


@tf.function(experimental_relax_shapes=True)
def gather_line_vectors(
    pafs: tf.Tensor, line_segments: np.ndarray, n_edge_samples: int
) -> tf.Tensor:
    """Gathers the PAF vectors along the points defined by line segments.
    
    This function is especially useful for sampling the values in the inferred PAFs
    without having to transfer the entire tensor back to the CPU after inference.
    
    Args:
        pafs: Part affinity fields output by a model with shape
            (samples, height, width, 2 * n_edge_types).
        line_segments: The (fully specified) subscripts that define the line segments
            between pairs of points in the form (sample, row, col, edge_type_ind). This
            can be generated by the make_line_segments function. This array has the
            shape: (n_pairs * n_edge_samples, 4).
        n_edge_samples: Number of samples that form each line segment. This should be
            the same value specified when generating the line_segments.
        
    Returns:
        paf_vals: a tensor of shape (n_pairs, n_edge_samples, 2), where each row
        contains the n_edge_samples vectors sampled along the line segment. The last
        dimension contains the y and x components of the PAF corresponding to the line
        segment formed by the pair, e.g., for the i-th pair of peaks:
            [[paf_y_0, paf_x_0], [paf_y_1, paf_x_1], ...] = paf_vals[i]
    """

    # Split into x and y components.
    pafs_x = pafs[:, :, :, ::2]
    pafs_y = pafs[:, :, :, 1::2]

    # Gather PAF values along line segments.
    paf_vals_x = tf.gather_nd(pafs_x, line_segments)
    paf_vals_y = tf.gather_nd(pafs_y, line_segments)

    # Stack into (n_pairs, n_edge_samples, 2).
    paf_vals = tf.stack(
        [
            tf.reshape(paf_vals_y, [-1, n_edge_samples]),
            tf.reshape(paf_vals_x, [-1, n_edge_samples]),
        ],
        axis=-1,
    )

    return paf_vals


def score_connection_candidates(
    src_peaks: np.ndarray,
    dst_peaks: np.ndarray,
    paf_vals: tf.Tensor,
    max_edge_length: float = 256.0,
    min_edge_score: float = 0.05,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the connection score for each line formed by pairs of peaks.
    
    Args:
        src_peaks: Peak coordinates with shape (n_pairs, 4) in subscript form.
        dst_peaks: Peak coordinates with shape (n_pairs, 4) in subscript form.
        paf_vals: PAF vectors sampled along the lines formed by each pair as a tensor
            of shape (n_pairs, n_edge_samples, 2), where the last axis specifies the
            (y, x) components of each PAF vector. These can be extracted by the
            gather_line_vectors function.
        max_edge_length: The maximum length of a potential connection in the same units
            as the coordinates in src_peaks and dst_peaks. Connections that form a line
            longer than this value are penalized.
        min_edge_score: The minimum connection score (before distance penalization)
            required in order to consider a connection as correct.
            
    Returns:
        A tuple of score_with_dist_penalty, fraction_correct.
        
        score_with_dist_penalty is a tensor float32 vector of length n_pairs, specifying
        the connection score for each pair of points. Higher values indicate a better
        connection score (max of 1.0).
        
        fraction_correct is a tensor float32 vector of length n_pairs, specifying the
        proportion of sampled points (n_edge_samples) that were above the min_edge_score
        threshold before applying the distance penalty.
        
    Note:
        The connection score is computed as follows:
        
        Given a pair of source and destination points (e.g., peak coordinates), let the
        directed line segment (i.e., Euclidean vector) formed between them define a
        potential connection from source to destination.
        
        The spatial vector for this connection is calculated as the relative offset
        (i.e., displacement) of the destination point relative to the source point.
        
        The unit spatial vector is the spatial vector divided by the length of the line,
        such that it has a magnitude (length) equal to 1.
        
        Given a particular PAF vector, sampled from the points along this line from the
        inferred PAFs, the connection score is simply the dot product between the unit
        spatial vector and the PAF vector.
        
        The overall connection score is then the average of the dot products for all
        points sampled from the line that forms the connection, possibly penalized for
        absolute distance.
        
        The connection score is a real-valued scalar in the range [-1, 1].
    """

    # Compute spatial vectors.
    spatial_vec = dst_peaks[:, 1:3] - src_peaks[:, 1:3]
    spatial_vec = tf.cast(spatial_vec, tf.float32)
    spatial_vec_length = tf.sqrt(tf.reduce_sum(spatial_vec ** 2, axis=1))
    spatial_vec /= tf.expand_dims(spatial_vec_length, axis=1)

    # Compute line scores for all connection pairs.
    line_scores = tf.squeeze(paf_vals @ tf.expand_dims(spatial_vec, axis=2), axis=-1)

    # Compute average line scores with distance penalty.
    dist_penalty = (tf.cast(max_edge_length, tf.float32) / spatial_vec_length) - 1
    score_with_dist_penalty = tf.reduce_mean(line_scores, axis=1) + tf.minimum(
        dist_penalty, 0
    )

    # Compute fraction of connections above threshold.
    fraction_correct = tf.reduce_mean(
        tf.cast(line_scores > min_edge_score, tf.float32), axis=1
    )

    return score_with_dist_penalty, fraction_correct


def filter_connection_candidates(
    src_peak_inds: np.ndarray,
    dst_peak_inds: np.ndarray,
    connection_scores: np.ndarray,
    edge_type_inds: np.ndarray,
    edge_types: List[EdgeType],
) -> Dict[EdgeType, List[EdgeConnection]]:
    """Greedily forms connections from candidate peak pairs without reusing peaks.

    This function creates EdgeConnections for each EdgeType by greedily assigning peak
    pairs by their connection scores in descending order, such that higher scoring
    connections are formed first.
    
    Args:
        src_peak_inds: A vector of n_pairs indices or unique identifiers for source
            peaks of each connection candidate. These should only correspond to
            connections within the same frame.
        dst_peak_inds: A vector of n_pairs indices or unique identifiers for destination
            peaks of each connection candidate. These should only correspond to
            connections within the same frame.
        connection_scores: A vector of n_pairs scores associated with each connection
            candidate. Connections will be formed in order of descending connection
            score. These should only correspond to connections within the same frame.
        edge_type_inds: A vector of n_pairs indices associated with each connection
            candidate indicating the edge type index within the edge_types list.
        edge_types: List of EdgeType instances defining the directed graph.
    
    Returns:
        connections, a dictionary mapping EdgeTypes to a list of valid EdgeConnections
        after filtering.
        
        An EdgeConnection is a tuple of (src_peak_ind, dst_peak_ind, connection_score),
        where the indices are a subset of the input src_peak_inds and dst_peak_inds.
        
        The output of this function can be provided as input to the
        assign_connections_to_instances function which will group the connections into
        distinct instances (i.e., disconnected subgraphs of the connections).

    Notes:
        This function should be applied to connection candidates within the same sample
        or frame!
    """

    # We'll group the filtered connections by edge type.
    connections = dict()

    # Group by edge type.
    candidate_inds = np.arange(len(src_peak_inds))
    edge_grouped_inds = utils.group_array(candidate_inds, edge_type_inds)
    for edge_type_ind, edge_candidate_inds in edge_grouped_inds.items():

        # Get data for this edge type.
        edge_connection_scores = connection_scores[edge_candidate_inds]
        edge_src_peak_inds = src_peak_inds[edge_candidate_inds]
        edge_dst_peak_inds = dst_peak_inds[edge_candidate_inds]

        # Sort edge connections by descending score.
        score_sorted_edge_sub_inds = np.argsort(edge_connection_scores)[::-1]

        # Keep connections by greedily assigning edges with unused nodes.
        edge_connections = []
        used_edge_src_peak_inds = set()
        used_edge_dst_peak_inds = set()
        for edge_sub_ind in score_sorted_edge_sub_inds:
            edge_src_peak_ind = edge_src_peak_inds[edge_sub_ind]
            edge_dst_peak_ind = edge_dst_peak_inds[edge_sub_ind]

            if (
                edge_src_peak_ind not in used_edge_src_peak_inds
                and edge_dst_peak_ind not in used_edge_dst_peak_inds
            ):

                # Create a new connection with the peak inds.
                edge_connections.append(
                    EdgeConnection(
                        edge_src_peak_ind,
                        edge_dst_peak_ind,
                        edge_connection_scores[edge_sub_ind],
                    )
                )

                # Keep track of the peak inds used in connections so far.
                used_edge_src_peak_inds.add(edge_src_peak_ind)
                used_edge_dst_peak_inds.add(edge_dst_peak_ind)

        # Save to the connections table.
        connections[edge_types[edge_type_ind]] = edge_connections

    return connections


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


def create_predicted_instances(
    peaks: np.ndarray,
    peak_vals: np.ndarray,
    connections: Dict[EdgeType, List[EdgeConnection]],
    instance_assignments: Dict[PeakID, int],
    skeleton: Skeleton,
) -> List[PredictedInstance]:
    """Creates PredictedInstances from a set of peaks and instance assignments.
    
    Args:
        peaks: Peak coordinates with shape (n_peaks, 4) in subscript form.
        peak_vals: Values at the peak coordinates, typically the confidence map score.
        connections: A dict that maps EdgeType to a list of EdgeConnections found
            through connection scoring. This can be generated by the
            filter_connection_candidates function.
        instance_assignments: A dict mapping PeakID to a unique instance ID specified
            as an integer. This can be generated by the assign_connections_to_instances
            function. The indices specified in the PeakIDs should index into peaks.
        skeleton: The Skeleton object to use for associating to the resulting predicted
            instances.
    
    Returns:
        A list of PredictedInstances after instance assignment.
        
        Each PredictedInstance will be composed of PredictedPoints with the score
        specified in peak_vals.
        
        The total instance score will be the sum of the scores for all connections
        within an instance, normalized by the total number of edges in the skeleton.
    """

    # Create predicted points and group by instance and node.
    instance_points = defaultdict(dict)
    for peak_id, instance_ind in instance_assignments.items():
        node_ind, peak_ind = attr.astuple(peak_id)

        # Update instance with the peak as a PredictedPoint.
        node_name = skeleton.node_names[node_ind]
        instance_points[instance_ind][node_name] = PredictedPoint(
            x=peaks[peak_ind, 2], y=peaks[peak_ind, 1], score=peak_vals[peak_ind],
        )

    # Accumulate instance scores by looping through the connections.
    instance_scores = defaultdict(int)
    for edge_type, edge_connections in connections.items():
        src_node_ind, dst_node_ind = attr.astuple(edge_type)

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
                score=instance_scores[instance_ind] / len(skeleton.edges),
            )
        )

    return predicted_instances


@attr.s(auto_attribs=True, eq=False)
class PAFGrouper:
    inference_model: model.InferenceModel
    batch_size: int = 8
    n_edge_samples: int = 5
    min_pair_distance: float = 1.0
    max_edge_length: float = 256.0
    min_edge_score: float = 0.05
    min_edge_samples_correct: float = 0.8
    min_instance_peaks: int = 2

    _paf_scale: float = attr.ib(init=False)
    _skeleton: Skeleton = attr.ib(init=False)
    _edge_types: List[EdgeType] = attr.ib(init=False)

    @property
    def paf_scale(self):
        return self._paf_scale

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def edge_types(self):
        return self._edge_types

    def __attrs_post_init__(self):
        self._paf_scale = self.inference_model.output_scale
        self._skeleton = self.inference_model.skeleton
        self._edge_types = [
            EdgeType(src_node_ind, dst_node_ind)
            for src_node_ind, dst_node_ind in self.skeleton.edge_inds
        ]

    def preproc(self, imgs):
        # Scale to model input size.
        imgs = utils.resize_imgs(
            imgs,
            self.inference_model.input_scale,
            common_divisor=2 ** self.inference_model.down_blocks,
        )

        # Convert to float32 and scale values to [0., 1.].
        imgs = utils.normalize_imgs(imgs)

        return imgs

    @tf.function
    def inference(self, imgs):
        # Model inference
        pafs = self.inference_model.keras_model(imgs)

        return pafs

    def batched_inference(self, imgs, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        pafs = utils.batched_call(self.inference, imgs, batch_size=batch_size)

        return pafs

    def postproc(self, paf_peaks, paf_peak_vals, pafs):

        # Find all possible connections formed by pairs of peak.
        src_peak_inds, dst_peak_inds, edge_type_inds = find_peak_pairs(
            paf_peaks, self.edge_types, self.min_pair_distance
        )

        # Interpolate between pairs to form the line segments.
        line_segments = make_line_segments(
            src_peaks=paf_peaks[src_peak_inds],
            dst_peaks=paf_peaks[dst_peak_inds],
            edge_type_inds=edge_type_inds,
            n_edge_samples=self.n_edge_samples,
        )

        # Gather the PAF vectors along each line segment.
        paf_vals = gather_line_vectors(pafs, line_segments, self.n_edge_samples)

        # Compute connection scores for each line segment.
        score_with_dist_penalty, fraction_correct = score_connection_candidates(
            src_peaks=paf_peaks[src_peak_inds],
            dst_peaks=paf_peaks[dst_peak_inds],
            paf_vals=paf_vals,
            max_edge_length=self.max_edge_length,
            min_edge_score=self.min_edge_score,
        )

        # Criterion 1: There are sufficient sampled points along the line integral
        # that are above the minimum threshold.
        enough_correct_samples = fraction_correct > self.min_edge_samples_correct

        # Criterion 2: Average score with length penalty is positive.
        positive_score = score_with_dist_penalty > 0

        # Find all valid connection candidates.
        valid_connection_inds = (
            tf.where(enough_correct_samples & positive_score).numpy().squeeze()
        )

        # Copy scores to CPU.
        score_with_dist_penalty = score_with_dist_penalty.numpy()

        # Assign connection candidates to instances by sample.
        samples = paf_peaks[src_peak_inds, 0].astype("int32")
        sample_grouped_valid_connection_inds = utils.group_array(
            valid_connection_inds, samples[valid_connection_inds]
        )
        sample_grouped_connections = dict()
        sample_grouped_instance_assignments = dict()
        for sample, connection_inds in sample_grouped_valid_connection_inds.items():

            # Include peak scores in the connection scores.
            connection_scores = (
                score_with_dist_penalty[connection_inds]
                + paf_peak_vals[src_peak_inds[connection_inds]]
                + paf_peak_vals[dst_peak_inds[connection_inds]]
            ) / 3.0

            # Assign peaks to unique connections for each edge type by score.
            connections = filter_connection_candidates(
                src_peak_inds=src_peak_inds[connection_inds],
                dst_peak_inds=dst_peak_inds[connection_inds],
                connection_scores=connection_scores,
                edge_type_inds=edge_type_inds[connection_inds],
                edge_types=self.edge_types,
            )

            # Now that peak pairs are disjoint with respect to edge type-wise bipartite
            # graphs, assign them to instances to form disjoint subgraphs.
            instance_assignments = assign_connections_to_instances(
                connections=connections,
                min_instance_peaks=self.min_instance_peaks,
                n_nodes=len(self.skeleton.nodes),
            )

            # Save results for this sample.
            sample_grouped_connections[sample] = connections
            sample_grouped_instance_assignments[sample] = instance_assignments

        return sample_grouped_connections, sample_grouped_instance_assignments

    def predict(self, imgs, peaks, peak_vals, bboxes=None, batch_size=None):

        # Do the heavy lifting.
        imgs = self.preproc(imgs)
        pafs = self.batched_inference(imgs, batch_size=batch_size)

        # We'll need to adjust the peaks, so let's keep the originals unmodified.
        paf_peaks = np.copy(peaks)

        if bboxes is not None:
            # Adjust peak coordinates to within-bounding box subscripts if needed.
            paf_peaks[:, 1:3] -= bboxes[:, 0:2]

        # Adjust peaks to the PAF output scale.
        paf_peaks *= np.array([[1, self.paf_scale, self.paf_scale, 1]])

        # Perform the PAF-based peak grouping.
        sample_grouped_connections, sample_grouped_instance_assignments = self.postproc(
            paf_peaks, peak_vals, pafs
        )

        # Generate predicted instances grouped by sample.
        sample_grouped_predicted_instances = dict()
        for sample in sample_grouped_connections:
            sample_grouped_predicted_instances[sample] = create_predicted_instances(
                peaks=peaks,
                peak_vals=peak_vals,
                connections=sample_grouped_connections[sample],
                instance_assignments=sample_grouped_instance_assignments[sample],
                skeleton=self.skeleton,
            )

        return sample_grouped_predicted_instances

    def predict_rps(self, region_proposal_set, region_peaks, batch_size=None):

        # Pull out peaks with patch-based indexing.
        patch_peaks = region_peaks.peaks_with_patch_inds

        # Predict PAFs and instances.
        patch_grouped_predicted_instances = self.predict(
            imgs=region_proposal_set.patches,
            peaks=patch_peaks,
            peak_vals=region_peaks.peak_vals,
            bboxes=region_proposal_set.bboxes[region_peaks.patch_inds],
            batch_size=batch_size,
        )

        # Regroup predicted instances by samples instead of patches.
        sample_grouped_predicted_instances = defaultdict(list)
        for patch_ind, predicted_instances in patch_grouped_predicted_instances.items():
            sample_ind = region_proposal_set.sample_inds[patch_ind]
            sample_grouped_predicted_instances[sample_ind].extend(predicted_instances)

        # TODO: Merge/suppress overlaps after regrouping?

        return sample_grouped_predicted_instances

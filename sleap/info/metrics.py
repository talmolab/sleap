"""
Module for producing prediction metrics for SLEAP datasets.
"""
from inspect import signature
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Callable, List, Optional, Union, Tuple

from sleap.instance import Instance, PredictedInstance
from sleap.io.dataset import Labels


def matched_instance_distances(
    labels_gt: Labels,
    labels_pr: Labels,
    match_lists_function: Callable,
    frame_range: Optional[range] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:

    """
    Distances between ground truth and predicted nodes over a set of frames.

    Args:
        labels_gt: the `Labels` object with ground truth data
        labels_pr: the `Labels` object with predicted data
        match_lists_function: function for determining corresponding instances
            Takes two lists of instances and returns "sorted" lists.
        frame_range (optional): range of frames for which to compare data
            If None, we compare every frame in labels_gt with corresponding
            frame in labels_pr.
    Returns:
        Tuple:
        * frame indices map: instance idx (for other matrices) -> frame idx
        * distance matrix: (instances * nodes)
        * ground truth points matrix: (instances * nodes * 2)
        * predicted points matrix: (instances * nodes * 2)
    """

    frame_idxs = []
    points_gt = []
    points_pr = []
    for lf_gt in labels_gt.find(labels_gt.videos[0]):
        frame_idx = lf_gt.frame_idx

        # Get instances from ground truth/predicted labels
        instances_gt = lf_gt.instances
        lfs_pr = labels_pr.find(labels_pr.videos[0], frame_idx=frame_idx)
        if len(lfs_pr):
            instances_pr = lfs_pr[0].instances
        else:
            instances_pr = []

        # Sort ground truth and predicted instances.
        # We'll then compare points between corresponding items in lists.
        # We can use different "match" functions depending on what we want.
        sorted_gt, sorted_pr = match_lists_function(instances_gt, instances_pr)

        # Convert lists of instances to (instances, nodes, 2) matrices.
        # This allows match_lists_function to return data as either
        # a list of Instances or a (instances, nodes, 2) matrix.
        if type(sorted_gt[0]) != np.ndarray:
            sorted_gt = list_points_array(sorted_gt)
        if type(sorted_pr[0]) != np.ndarray:
            sorted_pr = list_points_array(sorted_pr)

        points_gt.append(sorted_gt)
        points_pr.append(sorted_pr)
        frame_idxs.extend([frame_idx] * len(sorted_gt))

    # Convert arrays to numpy matrixes
    # instances * nodes * (x,y)
    points_gt = np.concatenate(points_gt)
    points_pr = np.concatenate(points_pr)

    # Calculate distances between corresponding nodes for all corresponding
    # ground truth and predicted instances.
    D = np.linalg.norm(points_gt - points_pr, axis=2)

    return frame_idxs, D, points_gt, points_pr


def match_instance_lists(
    instances_a: List[Union[Instance, PredictedInstance]],
    instances_b: List[Union[Instance, PredictedInstance]],
    cost_function: Callable,
) -> Tuple[
    List[Union[Instance, PredictedInstance]], List[Union[Instance, PredictedInstance]]
]:
    """Sorts two lists of Instances to find best overall correspondence
    for a given cost function (e.g., total distance between points)."""

    pairwise_distance_matrix = calculate_pairwise_cost(
        instances_a, instances_b, cost_function
    )
    match_a, match_b = linear_sum_assignment(pairwise_distance_matrix)

    sorted_a = list(map(lambda idx: instances_a[idx], match_a))
    sorted_b = list(map(lambda idx: instances_b[idx], match_b))
    return sorted_a, sorted_b


def calculate_pairwise_cost(
    instances_a: List[Union[Instance, PredictedInstance]],
    instances_b: List[Union[Instance, PredictedInstance]],
    cost_function: Callable,
) -> np.ndarray:
    """Calculate (a * b) matrix of pairwise costs using cost function."""

    matrix_size = (len(instances_a), len(instances_b))
    pairwise_cost_matrix = np.full(matrix_size, np.inf)
    for idx_a, inst_a in enumerate(instances_a):
        for idx_b, inst_b in enumerate(instances_b):

            # cost_function can either take a single input or two inputs
            # single input: ndarray of distances between corresponding nodes
            # two inputs: the pair of instances
            if len(signature(cost_function).parameters) == 1:
                point_dist_array = point_dist(inst_a, inst_b)
                cost = cost_function(point_dist_array)
            else:
                cost = cost_function(inst_a, inst_b)

            pairwise_cost_matrix[idx_a, idx_b] = cost
    return pairwise_cost_matrix


def match_instance_lists_nodewise(
    instances_a: List[Union[Instance, PredictedInstance]],
    instances_b: List[Union[Instance, PredictedInstance]],
    thresh: float = 5,
) -> Tuple[
    List[Union[Instance, PredictedInstance]], List[Union[Instance, PredictedInstance]]
]:
    """For each node for each instance in the first list, pairs it with the
    closest corresponding node from *any* instance in the second list."""

    node_count = len(instances_a[0].skeleton.nodes)
    b_points_arrays = list_points_array(instances_b)

    best_points_array = []

    for inst_a in instances_a:

        # Calculate distance from nodes in A to nodes for each B
        dist_array = []
        for inst_b in instances_b:
            dist_array.append(point_dist(inst_a, inst_b))
        dist_array = np.stack(dist_array)

        # Construct matrix with closest node from any B
        closest_point_array = np.full((node_count, 2), np.nan)
        for node_idx in range(node_count):

            # Make sure there's some prediction for this node
            if any(~np.isnan(dist_array[:, node_idx])):
                best_idx = np.nanargmin(dist_array[:, node_idx])

                # Ignore closest point if distance is beyond threshold
                if dist_array[best_idx, node_idx] <= thresh:
                    closest_point_array[node_idx] = b_points_arrays[best_idx, node_idx]

        # Add matrix of points to compare against ground truth instance
        best_points_array.append(closest_point_array)

    return instances_a, best_points_array


def point_dist(
    inst_a: Union[Instance, PredictedInstance],
    inst_b: Union[Instance, PredictedInstance],
) -> np.ndarray:
    """Given two instances, returns array of distances for corresponding nodes."""

    points_a = inst_a.points_array
    points_b = inst_b.points_array
    point_dist = np.linalg.norm(points_a - points_b, axis=1)
    return point_dist


def nodeless_point_dist(
    inst_a: Union[Instance, PredictedInstance],
    inst_b: Union[Instance, PredictedInstance],
) -> np.ndarray:
    """Given two instances, returns array of distances for closest points
    ignoring node identities."""

    matrix_size = (len(inst_a.skeleton.nodes), len(inst_b.skeleton.nodes))
    pairwise_distance_matrix = np.full(matrix_size, 0)

    points_a = inst_a.points_array
    points_b = inst_b.points_array

    # Calculate the distance between any pair of inst A and inst B points
    for idx_a in range(points_a.shape[0]):
        for idx_b in range(points_b.shape[0]):
            pair_distance = np.linalg.norm(points_a[idx_a] - points_b[idx_b])
            if not np.isnan(pair_distance):
                pairwise_distance_matrix[idx_a, idx_b] = pair_distance

    # Match A and B points to sum of distances
    match_a, match_b = linear_sum_assignment(pairwise_distance_matrix)

    # Sort points by this match and calculate overall distance
    sorted_points_a = points_a[match_a, :]
    sorted_points_b = points_b[match_b, :]
    point_dist = np.linalg.norm(points_a - points_b, axis=1)

    return point_dist


def compare_instance_lists(
    instances_a: List[Union[Instance, PredictedInstance]],
    instances_b: List[Union[Instance, PredictedInstance]],
) -> np.ndarray:
    """Given two lists of corresponding Instances, returns
    (instances * nodes) matrix of distances between corresponding nodes."""

    paired_points_array_distances = []
    for inst_a, inst_b in zip(instances_a, instances_b):
        paired_points_array_distances.append(point_dist(inst_a, inst_b))

    return np.stack(paired_points_array_distances)


def list_points_array(
    instances: List[Union[Instance, PredictedInstance]]
) -> np.ndarray:
    """Given list of Instances, returns (instances * nodes * 2) matrix."""
    points_arrays = list(map(lambda inst: inst.points_array, instances))
    return np.stack(points_arrays)


def point_match_count(dist_array: np.ndarray, thresh: float = 5) -> int:
    """Given an array of distances, returns number which are <= threshold."""
    return np.sum(dist_array[~np.isnan(dist_array)] <= thresh)


def point_nonmatch_count(dist_array: np.ndarray, thresh: float = 5) -> int:
    """Given an array of distances, returns number which are not <= threshold."""
    return dist_array.shape[0] - point_match_count(dist_array, thresh)


if __name__ == "__main__":

    labels_gt = Labels.load_json("tests/data/json_format_v1/centered_pair.json")
    labels_pr = Labels.load_json(
        "tests/data/json_format_v2/centered_pair_predictions.json"
    )

    # OPTION 1

    # Match each ground truth instance node to the closest corresponding node
    # from any predicted instance in the same frame.

    nodewise_matching_func = match_instance_lists_nodewise

    # OPTION 2

    # Match each ground truth instance to a distinct predicted instance:
    # We want to maximize the number of "matching" points between instances,
    # where "match" means the points are within some threshold distance.
    # Note that each sorted list will be as long as the shorted input list.

    instwise_matching_func = lambda gt_list, pr_list: match_instance_lists(
        gt_list, pr_list, point_nonmatch_count
    )

    # PICK THE FUNCTION

    inst_matching_func = nodewise_matching_func
    # inst_matching_func = instwise_matching_func

    # Calculate distances
    frame_idxs, D, points_gt, points_pr = matched_instance_distances(
        labels_gt, labels_pr, inst_matching_func
    )

    # Show mean difference for each node
    node_names = labels_gt.skeletons[0].node_names

    for node_idx, node_name in enumerate(node_names):
        mean_d = np.nanmean(D[..., node_idx])
        print(f"{node_name}\t\t{mean_d}")

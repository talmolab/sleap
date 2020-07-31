"""
Functions to align instances.

Usually you'll want to

1. find out the skeleton edge to use for aligning instances,
2. align all instances using this edge in the skeleton,
3. calculate mean/std for node locations of aligned instances.

For step (1), we use the most "stable" edge (smallest std in length) for the set
of instances which has a (mean) length above some threshold. Usually this will
be something like [head -> thorax], i.e., an edge between two body parts which
are relatively fixed relative to each other, and thus work well as an axis for
aligning all the instances.

Steps (2) and (3) are fairly straightforward: we calculate angle of the edge
found in step (1) for each instance, then rotate each instance accordingly,
then calculate mean/standard deviation for each node in the resulting matrix.

Note that all these functions are vectorized and work on matrices with shape
(instances, nodes, 2), where 2 corresponds to (x, y) for each node.

After we have a "mean" instance (i.e., an instance with all points at mean
of other, aligned instances), the "mean" instance can then itself be aligned
with another instance using the `align_instance_points` function. This is
useful so we can use "mean" instance to add "default" points to an instance
which doesn't yet have all points).


"""
from sleap import Labels, Instance
from typing import List, Tuple
import numpy as np


def get_stable_node_pairs(
    all_points_arrays: np.ndarray, node_names, min_dist: float = 0.0
):
    """Returns sorted list of node pairs with mean and standard dev distance."""

    # Calculate distance from each point to each other point within each instance
    intra_points = (
        all_points_arrays[:, :, np.newaxis, :] - all_points_arrays[:, np.newaxis, :, :]
    )
    intra_dist = np.linalg.norm(intra_points, axis=-1)

    # Find mean and standard deviation for distances between each pair of nodes
    inter_std = np.nanstd(intra_dist, axis=0)
    inter_mean = np.nanmean(intra_dist, axis=0)

    # Clear pairs with too small mean distance
    inter_std[inter_mean <= min_dist] = np.nan

    # Ravel so that we can sort along single dimension
    flat_inter_std = np.ravel(inter_std)
    flat_inter_mean = np.ravel(inter_mean)

    # Get indices for sort by standard deviation (asc)
    sorted_flat_inds = np.argsort(flat_inter_std)
    sorted_inds = np.stack(np.unravel_index(sorted_flat_inds, inter_std.shape), axis=1)

    # Take every other, since we'll get A->B and B->A for each pair
    sorted_inds = sorted_inds[::2]
    sorted_flat_inds = sorted_flat_inds[::2]

    # print(all_points_arrays.shape)
    # print(intra_points.shape)
    # print(intra_dist.shape)
    # print(inter_std.shape)
    # print(sorted_inds.shape)

    # Make sorted list of data to return
    results = []
    for inds, flat_idx in zip(sorted_inds, sorted_flat_inds):
        node_a, node_b = inds
        std, mean = flat_inter_std[flat_idx], flat_inter_mean[flat_idx]
        if mean <= min_dist:
            break
        results.append(dict(node_a=node_a, node_b=node_b, std=std, mean=mean))
    return results


def get_most_stable_node_pair(
    all_points_arrays: np.ndarray, min_dist: float = 0.0
) -> Tuple[int, int]:
    """Returns pair of nodes which are at stable distance (over min threshold)."""
    all_pairs = get_stable_node_pairs(all_points_arrays, min_dist)
    return all_pairs[0]["node_a"], all_pairs[0]["node_b"]


def align_instances(
    all_points_arrays: np.ndarray,
    node_a: int,
    node_b: int,
    rotate_on_node_a: bool = False,
) -> np.ndarray:
    """Rotates every instance so that line from node_a to node_b aligns."""

    # For each instance, calculate the angle between nodes A and B
    node_to_node_lines = (
        all_points_arrays[:, node_a, :] - all_points_arrays[:, node_b, :]
    )
    theta = np.arctan2(node_to_node_lines[:, 1], node_to_node_lines[:, 0])

    # Make rotation matrix for each instance based on this angle
    R = np.ndarray((len(theta), 2, 2))
    c, s = np.cos(theta), np.sin(theta)

    R[:, 0, 0] = c
    R[:, 1, 1] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s

    # Rotate each instance by taking dot product with its corresponding rotation
    rotated = np.einsum("aij,ajk->aik", all_points_arrays, R)

    if rotate_on_node_a:
        # Shift so that rotation is "around" node A
        node_a_pos = points[:, node_a, :][:, np.newaxis, :]

    else:
        # Shift so node A is at fixed position for every instance
        node_a_pos = rotated[:, node_a, :][:, np.newaxis, :]

    # Do the shift
    rotated -= node_a_pos

    return rotated


def align_instances_on_most_stable(
    all_points_arrays: np.ndarray, min_stable_dist: float = 4.0
) -> np.ndarray:
    """
    Gets most stable pair of nodes and aligned instances along these nodes.
    """
    node_a, node_b = get_most_stable_node_pair(
        all_points_arrays, min_dist=min_stable_dist
    )
    aligned = align_instances(all_points_arrays, node_a, node_b, rotate_on_node_a=False)
    return aligned


def get_mean_and_std_for_points(
    aligned_points_arrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean and standard deviation for every node given aligned points.
    """
    mean = np.nanmean(aligned_points_arrays, axis=0)
    stdev = np.nanstd(aligned_points_arrays, axis=0)

    return mean, stdev


def make_mean_instance(
    aligned_points_arrays: List[np.ndarray], std_thresh: int = 0
) -> Instance:
    mean, stdev = get_mean_and_std_for_points(aligned_points_arrays)

    # Remove points with standard deviation higher than threshold
    if std_thresh:
        mean[stdev > std_thresh] = np.nan

    from sleap import Instance
    from sleap.instance import Point

    OFFSET = 0  # FIXME

    new_instance = Instance(
        skeleton=labels.skeletons[0],
        points=[Point(p[0] + OFFSET, p[1] + OFFSET) for p in mean],
    )
    return new_instance


def align_instance_points(source_points_array, target_points_array):
    """Transforms source for best fit on to target."""

    # Find (furthest) pair of points in target to use for alignment
    pairwise_distances = np.linalg.norm(
        target_points_array[:, np.newaxis, :] - target_points_array[np.newaxis, :, :],
        axis=-1,
    )
    node_a, node_b = np.unravel_index(
        np.nanargmax(pairwise_distances), shape=pairwise_distances.shape
    )

    # Align source to target
    source_line = source_points_array[node_a] - source_points_array[node_b]
    target_line = target_points_array[node_a] - target_points_array[node_b]

    source_theta = np.arctan2(source_line[1], source_line[0])
    target_theta = np.arctan2(target_line[1], target_line[0])

    rotation_theta = source_theta - target_theta
    c, s = np.cos(rotation_theta), np.sin(rotation_theta)
    R = np.array([[c, -s], [s, c]])

    rotated = source_points_array.dot(R)

    # Shift source to minimize total point different from target
    target_row_mask = ~np.isnan(target_points_array)[:, 0]
    shift = np.mean(
        rotated[target_row_mask] - target_points_array[target_row_mask], axis=0
    )
    rotated -= shift

    return rotated


def get_instances_points(instances: List[Instance]) -> np.ndarray:
    """Returns single (instance, node, 2) matrix with points for all instances."""
    return np.stack([inst.points_array for inst in instances])


def get_template_points_array(instances: List[Instance]) -> np.ndarray:
    """Returns mean of aligned points for instances."""
    points = get_instances_points(instances)

    node_a, node_b = get_most_stable_node_pair(points, min_dist=4.0)

    aligned = align_instances(points, node_a=node_a, node_b=node_b)
    points_mean, points_std = get_mean_and_std_for_points(aligned)
    return points_mean


if __name__ == "__main__":
    # filename = "tests/data/json_format_v2/centered_pair_predictions.json"
    # filename = "/Volumes/fileset-mmurthy/shruthi/code/sleap_expts/preds/screen_all.5pts_tmp_augment_200122/191210_102108_18159112_rig3_2.preds.h5"
    filename = "/Volumes/fileset-mmurthy/talmo/wt_gold_labeling/100919.sleap_wt_gold.13pt_init.n=288.junyu.h5"

    labels = Labels.load_file(filename)

    points = get_instances_points(labels.instances())
    get_stable_node_pairs(points, np.array(labels.skeletons[0].node_names))

    # import time
    #
    # t0 = time.time()
    # labels.add_instance(
    #     frame=labels.find_first(video=labels.videos[0]),
    #     instance=make_mean_instance(align_instances(points, 12, 0))
    # )
    # print(labels.find_first(video=labels.videos[0]))
    # print("time", time.time() - t0)
    #
    # Labels.save_file(labels, "mean.h5")

    # R = np.array(((c, -s), (s, c)))
    # a_rotated = a.dot(R)
    # a_rotated += a[0] - a_rotated[0]

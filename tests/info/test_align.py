from sleap.info import align
import numpy as np
from sleap_io import Skeleton
from sleap_io.model.instance import Instance
from sleap.sleap_io_adaptors.lf_labels_utils import instances


def test_get_instances_points(centered_pair_labels):
    x = align.get_instances_points(instances(centered_pair_labels))

    instance_count = len(centered_pair_labels.labeled_frames) * 2
    node_count = len(centered_pair_labels.skeletons[0].nodes)

    assert x.shape == (instance_count, node_count, 2)


def test_align_points():
    points = np.array(
        [
            [[10, 10], [10, 20], [10, 30]],
            [[10, 10], [20, 10], [34, 10]],
            [[30, 30], [31, 40], [30, 58]],
        ]
    )

    node_pair = align.get_most_stable_node_pair(points)

    assert len(node_pair) == 2
    assert min(node_pair) == 0
    assert max(node_pair) == 1

    aligned = align.align_instances(points, 0, 1)

    assert aligned.shape == points.shape

    # First nodes should align perfectly
    assert all(aligned[0, 0, :] == aligned[1, 0, :])
    assert all(aligned[0, 0, :] == aligned[2, 0, :])

    # Second nodes won't perfectly align
    assert not all(aligned[0, 1, :] == aligned[1, 1, :])

    mean, std = align.get_mean_and_std_for_points(aligned)

    assert all(mean[0] == [0, 0])
    assert all(std[0] == [0, 0])

    assert np.allclose(mean[1], [-10, 0], atol=0.1)
    assert np.allclose(mean[2], [-24, -1], atol=0.1)


def test_get_template_points_array_single_node():
    """Single-node skeleton: short-circuit to nan-mean without raising.

    Regression test for #2718 — `get_most_stable_node_pair` previously
    crashed with IndexError for skeletons with fewer than 2 nodes, which
    broke creating new instances after the first labeled one in the GUI.
    """
    skeleton = Skeleton(nodes=["centroid"])
    instance_pts = [
        Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton),
        Instance.from_numpy(np.array([[12.0, 24.0]]), skeleton=skeleton),
        Instance.from_numpy(np.array([[14.0, 22.0]]), skeleton=skeleton),
    ]

    out = align.get_template_points_array(instance_pts)

    assert out.shape == (1, 2)
    assert np.allclose(out, [[12.0, 22.0]])


def test_get_most_stable_node_pair_empty_returns_zero():
    """All-coincident points → no stable pair → (0, 0) instead of IndexError."""
    points = np.zeros((3, 2, 2))  # 3 instances, 2 nodes, all at origin
    node_a, node_b = align.get_most_stable_node_pair(points, min_dist=4.0)
    assert (node_a, node_b) == (0, 0)

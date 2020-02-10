from sleap.info import align
import numpy as np


def test_get_instances_points(centered_pair_labels):
    x = align.get_instances_points(centered_pair_labels.instances())

    instance_count = len(centered_pair_labels.labels) * 2
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

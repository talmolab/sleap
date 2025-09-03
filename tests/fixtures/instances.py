import pytest

from sleap_io.model.instance import Instance, PredictedInstance


@pytest.fixture
def instances(skeleton, centered_pair_vid):
    # Generate some instances
    NUM_INSTANCES = 500

    instances = []
    for i in range(NUM_INSTANCES):
        # Create numpy points array for the instance
        import numpy as np

        points_array = np.array(
            [
                [i * 1, i * 2],
                [10 + i * 1, 10 + i * 2],
                [20 + i * 1, 20 + i * 2],
                [float("nan"), float("nan")],
            ],
            dtype=float,
        )

        instance = Instance.from_numpy(points_array, skeleton=skeleton)
        instances.append(instance)

    return instances


@pytest.fixture
def predicted_instances(instances):
    return [
        PredictedInstance.from_numpy(i.points["xy"], skeleton=i.skeleton, score=1.0)
        for i in instances
    ]


@pytest.fixture
def multi_skel_instances(skeleton, stickman):
    """
    Setup some instances that reference multiple skeletons
    """

    # Generate some instances
    NUM_INSTANCES = 500

    import numpy as np

    instances = []

    # First skeleton instances
    for i in range(NUM_INSTANCES):
        points_array = np.array(
            [
                [i * 1, i * 2],
                [10 + i * 1, 10 + i * 2],
                [20 + i * 1, 20 + i * 2],
                [float("nan"), float("nan")],  # thorax NaN entry for testing
            ],
            dtype=float,
        )

        instance = Instance.from_numpy(points_array, skeleton=skeleton)
        instances.append(instance)

    # Setup some instances of the stick man on the same frames
    for i in range(NUM_INSTANCES):
        # Stickman skeleton typically has different nodes
        stickman_points = np.array(
            [
                [i * 10, i * 20],  # head
                [100 + i * 1, 100 + i * 2],  # body
                [200 + i * 1, 200 + i * 2],  # left-arm
            ],
            dtype=float,
        )

        instance = Instance.from_numpy(stickman_points, skeleton=stickman)
        instances.append(instance)

    return instances

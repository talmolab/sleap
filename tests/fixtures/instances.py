import pytest

from sleap.instance import Instance, Point, PredictedInstance


@pytest.fixture
def instances(skeleton):

    # Generate some instances
    NUM_INSTANCES = 500

    instances = []
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=skeleton)
        instance["head"] = Point(i * 1, i * 2)
        instance["left-wing"] = Point(10 + i * 1, 10 + i * 2)
        instance["right-wing"] = Point(20 + i * 1, 20 + i * 2)

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        instances.append(instance)

    return instances


@pytest.fixture
def predicted_instances(instances):
    return [PredictedInstance.from_instance(i, 1.0) for i in instances]


@pytest.fixture
def multi_skel_instances(skeleton, stickman):
    """
    Setup some instances that reference multiple skeletons
    """

    # Generate some instances
    NUM_INSTANCES = 500

    instances = []
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=skeleton, video=None, frame_idx=i)
        instance["head"] = Point(i * 1, i * 2)
        instance["left-wing"] = Point(10 + i * 1, 10 + i * 2)
        instance["right-wing"] = Point(20 + i * 1, 20 + i * 2)

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        instances.append(instance)

    # Setup some instances of the stick man on the same frames
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=stickman, video=None, frame_idx=i)
        instance["head"] = Point(i * 10, i * 20)
        instance["body"] = Point(100 + i * 1, 100 + i * 2)
        instance["left-arm"] = Point(200 + i * 1, 200 + i * 2)

        instances.append(instance)

    return instances

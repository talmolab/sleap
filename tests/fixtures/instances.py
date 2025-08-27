import pytest

from sleap_io.model.instance import Instance, PredictedInstance


@pytest.fixture
def instances(skeleton, centered_pair_vid):
    # Generate some instances
    NUM_INSTANCES = 500

    instances = []
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=skeleton)
        instance["head"] = ([i * 1, i * 2], True, False)  # (xy, visible, complete)
        instance["left-wing"] = ([10 + i * 1, 10 + i * 2], True, False)  # (xy, visible, complete)
        instance["right-wing"] = ([20 + i * 1, 20 + i * 2], True, False)  # (xy, visible, complete)

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        # Add a LabeledFrame

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
        instance["head"] = ([i * 1, i * 2], True, False)  # (xy, visible, complete)
        instance["left-wing"] = ([10 + i * 1, 10 + i * 2], True, False)  # (xy, visible, complete)
        instance["right-wing"] = ([20 + i * 1, 20 + i * 2], True, False)  # (xy, visible, complete)

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        instances.append(instance)

    # Setup some instances of the stick man on the same frames
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=stickman, video=None, frame_idx=i)
        instance["head"] = ([i * 10, i * 20], True, False)  # (xy, visible, complete)
        instance["body"] = ([100 + i * 1, 100 + i * 2], True, False)  # (xy, visible, complete)
        instance["left-arm"] = ([200 + i * 1, 200 + i * 2], True, False)  # (xy, visible, complete)

        instances.append(instance)

    return instances

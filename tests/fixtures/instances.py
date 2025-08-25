import pytest

from sleap.instance import LabeledFrame
from sleap_io.model.instance import Instance, PredictedInstance


@pytest.fixture
def instances(skeleton, centered_pair_vid):
    # Generate some instances
    NUM_INSTANCES = 500

    video = centered_pair_vid
    instances = []
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=skeleton)
        instance["head"] = [i * 1, i * 2, True, True]
        instance["left-wing"] = [10 + i * 1, 10 + i * 2, True, True]
        instance["right-wing"] = [20 + i * 1, 20 + i * 2, True, True]

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        # Add a LabeledFrame
        labeled_frame = LabeledFrame(video=video, frame_idx=i, instances=[instance])
        instance.frame = labeled_frame

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
        instance["head"] = [i * 1, i * 2]
        instance["left-wing"] = [10 + i * 1, 10 + i * 2]
        instance["right-wing"] = [20 + i * 1, 20 + i * 2]

        # Lets make an NaN entry to test skip_nan as well
        instance["thorax"]

        instances.append(instance)

    # Setup some instances of the stick man on the same frames
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=stickman, video=None, frame_idx=i)
        instance["head"] = [i * 10, i * 20, True, True]
        instance["body"] = [100 + i * 1, 100 + i * 2, True, True]
        instance["left-arm"] = [200 + i * 1, 200 + i * 2, True, True]

        instances.append(instance)

    return instances

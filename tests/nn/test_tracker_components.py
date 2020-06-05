import pytest
import numpy as np

from sleap.nn.tracking import Tracker, cull_instances, nms_fast, nms_instances

from sleap.instance import PredictedInstance
from sleap.skeleton import Skeleton


@pytest.mark.parametrize("tracker", ["simple", "flow"])
@pytest.mark.parametrize("similarity", ["instance", "iou", "centroid"])
@pytest.mark.parametrize("match", ["greedy", "hungarian"])
@pytest.mark.parametrize("count", [0, 2])
def test_tracker_by_name(tracker, similarity, match, count):
    t = Tracker.make_tracker_by_name(
        "flow", "instance", "greedy", clean_instance_count=2
    )
    t.track([])
    t.final_pass([])


def test_cull_instances(centered_pair_predictions):
    frames = centered_pair_predictions.labeled_frames[352:360]
    cull_instances(frames=frames, instance_count=2)

    for frame in frames:
        assert len(frame.instances) == 2

    frames = centered_pair_predictions.labeled_frames[:5]
    cull_instances(frames=frames, instance_count=1)

    for frame in frames:
        assert len(frame.instances) == 1


def test_nms():
    boxes = np.array(
        [[10, 10, 20, 20], [10, 10, 15, 15], [30, 30, 40, 40], [32, 32, 42, 42]]
    )
    scores = np.array([1, 0.3, 1, 0.5])

    picks = nms_fast(boxes, scores, iou_threshold=0.5)
    assert sorted(picks) == [0, 2]


def test_nms_with_target():
    boxes = np.array(
        [[10, 10, 20, 20], [10, 10, 15, 15], [30, 30, 40, 40], [32, 32, 42, 42]]
    )
    # Box 1 is suppressed and has lowest score
    scores = np.array([1, 0.3, 1, 0.5])
    picks = nms_fast(boxes, scores, iou_threshold=0.5, target_count=3)
    assert sorted(picks) == [0, 2, 3]

    # Box 3 is suppressed and has lowest score
    scores = np.array([1, 0.5, 1, 0.3])
    picks = nms_fast(boxes, scores, iou_threshold=0.5, target_count=3)
    assert sorted(picks) == [0, 1, 2]


def test_nms_instances_to_remove():
    skeleton = Skeleton()
    skeleton.add_nodes(("a", "b"))

    instances = []

    inst = PredictedInstance(skeleton=skeleton)
    inst["a"].x = 10
    inst["a"].y = 10
    inst["b"].x = 20
    inst["b"].y = 20
    inst.score = 1
    instances.append(inst)

    inst = PredictedInstance(skeleton=skeleton)
    inst["a"].x = 10
    inst["a"].y = 10
    inst["b"].x = 15
    inst["b"].y = 15
    inst.score = 0.3
    instances.append(inst)

    inst = PredictedInstance(skeleton=skeleton)
    inst["a"].x = 30
    inst["a"].y = 30
    inst["b"].x = 40
    inst["b"].y = 40
    inst.score = 1
    instances.append(inst)

    inst = PredictedInstance(skeleton=skeleton)
    inst["a"].x = 32
    inst["a"].y = 32
    inst["b"].x = 42
    inst["b"].y = 42
    inst.score = 0.5
    instances.append(inst)

    to_keep, to_remove = nms_instances(instances, iou_threshold=0.5, target_count=3)

    assert len(to_remove) == 1
    assert to_remove[0].matches(instances[1])

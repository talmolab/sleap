import pytest
from sleap.nn.tracking import Tracker


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

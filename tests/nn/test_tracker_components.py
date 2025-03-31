import numpy as np
import pytest

from sleap.instance import LabeledFrame, PredictedInstance
from sleap.io.dataset import Labels
from sleap.nn.tracker.components import (
    FrameMatches,
    cull_instances,
    cull_frame_instances,
    greedy_matching,
    nms_fast,
    nms_instances,
)
from sleap.nn.tracking import Tracker
from sleap.skeleton import Skeleton


@pytest.mark.parametrize(
    "tracker", ["simple", "flow", "simplemaxtracks", "flowmaxtracks"]
)
@pytest.mark.parametrize("similarity", ["instance", "iou", "centroid"])
@pytest.mark.parametrize("match", ["greedy", "hungarian"])
@pytest.mark.parametrize("count", [0, 2])
def test_tracker_by_name(tracker, similarity, match, count):
    t = Tracker.make_tracker_by_name(
        tracker=tracker, similarity=similarity, match=match, clean_instance_count=count
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


def test_cull_frame_instances_no_target(centered_pair_predictions: Labels):
    labels = centered_pair_predictions
    video = labels.video
    labeled_frame: LabeledFrame = labels.find_last(video=video, frame_idx=1098)

    # There will never be an IOU greater than 1, so expect all instances back.
    assert len(labeled_frame.instances) == 3
    cull_frame_instances(
        instances_list=labeled_frame.instances, general_iou_threshold=1
    )
    assert len(labeled_frame.instances) == 3

    # There is an instance with an IOU of 1 though, so expect 2 instances back.
    assert len(labeled_frame.instances) == 3
    cull_frame_instances(
        instances_list=labeled_frame.instances, general_iou_threshold=0.999999999999999
    )
    assert len(labeled_frame.instances) == 2

    # Test with Tracker

    tracker: Tracker = Tracker.make_tracker_by_name(
        pre_cull_general_iou_threshold=0.999999999999999,
    )
    assert tracker.pre_cull_function is not None

    # There is also an instance with an IOU of 0.67, so expect 1 instance back.
    assert len(labeled_frame.instances) == 2
    tracker: Tracker = Tracker.make_tracker_by_name(
        pre_cull_general_iou_threshold=0.6,
    )
    assert tracker.pre_cull_function is not None
    tracker.pre_cull_function(inst_list=labeled_frame.instances)
    assert len(labeled_frame.instances) == 1


def test_cull_frame_instances_with_target(centered_pair_predictions: Labels):
    labels = centered_pair_predictions
    video = labels.video
    labeled_frame: LabeledFrame = labels.find_last(video=video, frame_idx=1098)

    # Target count equal to the number of instances. Expect all instances back.
    target_count = 3

    # No IOU threshold.
    assert len(labeled_frame.instances) == target_count
    cull_frame_instances(instances_list=labeled_frame.instances, instance_count=3)
    assert len(labeled_frame.instances) == target_count

    # With IOU threshold.
    assert len(labeled_frame.instances) == target_count
    cull_frame_instances(
        instances_list=labeled_frame.instances,
        instance_count=target_count,
        iou_threshold=0.0,
    )
    assert len(labeled_frame.instances) == target_count

    # Target count less than the number of instances. Expect target count instances back

    # Without IOU.
    target_count = 2
    assert len(labeled_frame.instances) == 3
    cull_frame_instances(
        instances_list=labeled_frame.instances, instance_count=target_count
    )
    assert len(labeled_frame.instances) == target_count

    # With IOU.
    target_count = 1
    assert len(labeled_frame.instances) == 2
    cull_frame_instances(
        instances_list=labeled_frame.instances,
        instance_count=target_count,
        iou_threshold=0.0,
    )
    assert len(labeled_frame.instances) == target_count

    # Test with both target count and general IOU threshold. Switching frames and using
    # Tracker.

    labeled_frame: LabeledFrame = labels.find_last(video=video, frame_idx=1095)
    tracker: Tracker = Tracker.make_tracker_by_name(target_instance_count=target_count)
    assert tracker.pre_cull_function is None

    # No instances removed.

    target_count = 4
    general_iou_threshold = 1
    tracker: Tracker = Tracker.make_tracker_by_name(
        target_instance_count=target_count,
        pre_cull_general_iou_threshold=general_iou_threshold,
    )
    assert tracker.pre_cull_function is not None

    # Without non-general IOU.
    assert len(labeled_frame.instances) == target_count
    tracker.pre_cull_function(inst_list=labeled_frame.instances)
    assert len(labeled_frame.instances) == target_count

    # With non-general IOU.
    iou_threshold = 0.0
    assert len(labeled_frame.instances) == target_count
    tracker: Tracker = Tracker.make_tracker_by_name(
        target_instance_count=target_count,
        pre_cull_iou_threshold=iou_threshold,
        pre_cull_general_iou_threshold=general_iou_threshold,
    )
    assert tracker.pre_cull_function is not None
    tracker.pre_cull_function(inst_list=labeled_frame.instances)
    assert len(labeled_frame.instances) == target_count

    # Instance removed via general IOU.
    target_count = 4
    general_iou_threshold = 0.999999999999999
    assert len(labeled_frame.instances) == 4
    tracker: Tracker = Tracker.make_tracker_by_name(
        target_instance_count=target_count,
        pre_cull_general_iou_threshold=general_iou_threshold,
    )
    assert tracker.pre_cull_function is not None
    tracker.pre_cull_function(inst_list=labeled_frame.instances)
    assert len(labeled_frame.instances) == target_count - 1

    # Instance removed via non-general IOU.
    target_count = 2
    iou_threshold = 0.0
    assert len(labeled_frame.instances) == 3
    tracker: Tracker = Tracker.make_tracker_by_name(
        target_instance_count=target_count,
        pre_cull_to_target=True,
        pre_cull_iou_threshold=iou_threshold,
    )
    assert tracker.pre_cull_function is not None
    tracker.pre_cull_function(inst_list=labeled_frame.instances)
    assert len(labeled_frame.instances) == target_count


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


def test_frame_match_object():
    instances = ["instance a", "instance b"]
    tracks = ["track a", "track b"]

    # columns are tracks
    # rows are instances
    cost_matrix = np.array(
        [
            [10, 200],  # instance a will match track a
            [75, 150],
        ]  # instance b will match track b, its second choice
    )

    frame_matches = FrameMatches.from_cost_matrix(
        cost_matrix=cost_matrix,
        instances=instances,
        tracks=tracks,
        matching_function=greedy_matching,
    )

    assert not frame_matches.has_only_first_choice_matches

    matches = frame_matches.matches

    assert len(matches) == 2

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance a"
    assert matches[0].score == -10

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance b"
    assert matches[1].score == -150

    # columns are tracks
    # rows are instances
    cost_matrix = np.array(
        [
            [10, 200],  # instance a will match track a
            [150, 75],
        ]  # instance b will match track b, now its first choice
    )

    frame_matches = FrameMatches.from_cost_matrix(
        cost_matrix=cost_matrix,
        instances=instances,
        tracks=tracks,
        matching_function=greedy_matching,
    )

    assert frame_matches.has_only_first_choice_matches

    assert matches[0].track == "track a"
    assert matches[0].instance == "instance a"

    assert matches[1].track == "track b"
    assert matches[1].instance == "instance b"


def make_insts(trx):
    skel = Skeleton.from_names_and_edge_inds(
        ["A", "B", "C"], edge_inds=[[0, 1], [1, 2]]
    )

    def make_inst(x, y):
        pts = np.array([[-0.1, -0.1], [0.0, 0.0], [0.1, 0.1]]) + np.array([[x, y]])
        return PredictedInstance.from_numpy(pts, [1, 1, 1], 1, skel)

    insts = []
    for frame in trx:
        insts_frame = []
        for x, y in frame:
            insts_frame.append(make_inst(x, y))
        insts.append(insts_frame)
    return insts


def test_max_tracking_large_gap_single_track():
    # Track 2 instances with gap > window size
    preds = make_insts(
        [
            [
                (0, 0),
                (0, 1),
            ],
            [
                (0.1, 0),
                (0.1, 1),
            ],
            [
                (0.2, 0),
                (0.2, 1),
            ],
            [
                (0.3, 0),
            ],
            [
                (0.4, 0),
            ],
            [
                (0.5, 0),
                (0.5, 1),
            ],
            [
                (0.6, 0),
                (0.6, 1),
            ],
        ]
    )

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        # tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        # max_tracks=2,
        # max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 3

    tracker = Tracker.make_tracker_by_name(
        # tracker="simple",
        tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        max_tracks=2,
        max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2


def test_max_tracking_small_gap_on_both_tracks():
    # Test 2 instances with both tracks with gap > window size
    preds = make_insts(
        [
            [
                (0, 0),
                (0, 1),
            ],
            [
                (0.1, 0),
                (0.1, 1),
            ],
            [
                (0.2, 0),
                (0.2, 1),
            ],
            [],
            [],
            [
                (0.5, 0),
                (0.5, 1),
            ],
            [
                (0.6, 0),
                (0.6, 1),
            ],
        ]
    )

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        # tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        # max_tracks=2,
        # max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 4

    tracker = Tracker.make_tracker_by_name(
        # tracker="simple",
        tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        max_tracks=2,
        max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2


def test_max_tracking_extra_detections():
    # Test having more than 2 detected instances in a frame
    preds = make_insts(
        [
            [
                (0, 0),
                (0, 1),
            ],
            [
                (0.1, 0),
                (0.1, 1),
            ],
            [
                (0.2, 0),
                (0.2, 1),
            ],
            [
                (0.3, 0),
            ],
            [
                (0.4, 0),
            ],
            [
                (0.5, 0),
                (0.5, 1),
            ],
            [
                (0.6, 0),
                (0.6, 1),
                (0.6, 0.5),
            ],
        ]
    )

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        # tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        # max_tracks=2,
        # max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 4

    tracker = Tracker.make_tracker_by_name(
        # tracker="simple",
        tracker="simplemaxtracks",
        match="hungarian",
        track_window=2,
        max_tracks=2,
        max_tracking=True,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2

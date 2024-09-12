import pytest
import numpy as np

from sleap.nn.tracking import Tracker
from sleap.nn.tracker.components import (
    nms_instances,
    nms_fast,
    cull_instances,
    FrameMatches,
    greedy_matching,
)
from sleap.io.dataset import Labels

from sleap.instance import PredictedInstance
from sleap.skeleton import Skeleton


def run_tracker_by_name(frames=None, img_scale: float = 0, **kwargs):
    # Create tracker
    t = Tracker.make_tracker_by_name(verbosity="none", **kwargs)
    # Update img_scale
    if img_scale:
        if hasattr(t, "candidate_maker") and hasattr(t.candidate_maker, "img_scale"):
            t.candidate_maker.img_scale = img_scale
        else:
            # Do not even run tracking as it can be slow
            pytest.skip("img_scale is not defined for this tracker")
            return

    # Run tracking
    new_frames = t.run_tracker(frames or [])
    assert len(new_frames) == len(frames)


@pytest.mark.parametrize("tracker", ["simple", "flow"])
@pytest.mark.parametrize("similarity", ["instance", "iou", "centroid"])
@pytest.mark.parametrize("match", ["greedy", "hungarian"])
@pytest.mark.parametrize("img_scale", [0, 1, 0.25])
@pytest.mark.parametrize("count", [0, 2])
def test_tracker_by_name(
    centered_pair_predictions_sorted,
    tracker,
    similarity,
    match,
    img_scale,
    count,
):
    # This is slow, so limit to 5 time points
    frames = centered_pair_predictions_sorted[:5]

    run_tracker_by_name(
        frames=frames,
        tracker=tracker,
        similarity=similarity,
        match=match,
        img_scale=img_scale,
        max_tracks=count,
    )


@pytest.mark.parametrize("tracker", ["simple", "flow"])
@pytest.mark.parametrize("oks_score_weighting", ["True", "False"])
@pytest.mark.parametrize("oks_normalization", ["all", "ref", "union"])
def test_oks_tracker_by_name(
    centered_pair_predictions_sorted,
    tracker,
    oks_score_weighting,
    oks_normalization,
):
    # This is slow, so limit to 5 time points
    frames = centered_pair_predictions_sorted[:5]

    run_tracker_by_name(
        frames=frames,
        tracker=tracker,
        similarity="object_keypoint",
        matching="greedy",
        oks_score_weighting=oks_score_weighting,
        oks_normalization=oks_normalization,
        max_tracks=2,
    )


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


def test_max_tracks_large_gap_single_track():
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
        match="hungarian",
        track_window=2,
        max_tracks=-1,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 3

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        match="hungarian",
        track_window=2,
        max_tracks=2,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2


def test_max_tracks_small_gap_on_both_tracks():
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
        match="hungarian",
        track_window=2,
        max_tracks=-1,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 4

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        match="hungarian",
        track_window=2,
        max_tracks=2,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2


def test_max_tracks_extra_detections():
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
        match="hungarian",
        track_window=2,
        max_tracks=-1,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 4

    tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        match="hungarian",
        track_window=2,
        max_tracks=2,
    )

    tracked = []
    for insts in preds:
        tracked_insts = tracker.track(insts)
        tracked.append(tracked_insts)
    all_tracks = list(set([inst.track for frame in tracked for inst in frame]))

    assert len(all_tracks) == 2

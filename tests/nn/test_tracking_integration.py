import inspect
import operator
import os
import time

import sleap
from sleap.nn.inference import main as inference_cli
import sleap.nn.tracker.components
from sleap.io.dataset import Labels, LabeledFrame


def test_simple_tracker(tmpdir, centered_pair_predictions_slp_path):
    cli = (
        "--tracking.tracker simple "
        "--frames 200-300 "
        f"-o {tmpdir}/simpletracks.slp "
        f"{centered_pair_predictions_slp_path}"
    )
    inference_cli(cli.split(" "))

    labels = sleap.load_file(f"{tmpdir}/simpletracks.slp")
    assert len(labels.tracks) == 8


def test_simplemax_tracker(tmpdir, centered_pair_predictions_slp_path):
    cli = (
        "--tracking.tracker simplemaxtracks "
        "--tracking.max_tracking 1 --tracking.max_tracks 2 "
        "--frames 200-300 "
        f"-o {tmpdir}/simplemaxtracks.slp "
        f"{centered_pair_predictions_slp_path}"
    )
    inference_cli(cli.split(" "))

    labels = sleap.load_file(f"{tmpdir}/simplemaxtracks.slp")
    assert len(labels.tracks) == 2


# TODO: Refactor the below things into a real test suite.
# running an equivalent to `make_ground_truth` is done as a test in tests/nn/test_tracker_components.py


def make_ground_truth(frames, tracker, gt_filename):
    t0 = time.time()
    new_labels = tracker.run_tracker(frames, verbosity="none")
    print(f"{gt_filename}\t{len(tracker.spawned_tracks)}\t{time.time()-t0}")
    Labels.save_file(new_labels, gt_filename)


def compare_ground_truth(frames, tracker, gt_filename):
    t0 = time.time()
    new_labels = tracker.run_tracker(frames, verbosity="none")
    print(f"{gt_filename}\t{time.time() - t0}")

    does_match = check_tracks(new_labels, gt_filename)

    print(f"{gt_filename}\t{time.time() - t0}")
    print(f"{gt_filename}\t\t{does_match}")


def check_tracks(labels, gt_filename, limit=None):
    gt_lfs = Labels.load_file(gt_filename).labeled_frames
    lfs = labels.labeled_frames

    if limit:
        gt_lfs = gt_lfs[limit]
        lfs = lfs[limit]

    for lf, gt_lf in zip(lfs, gt_lfs):
        for inst, gt_inst in zip(lf, gt_lf):
            if inst.track is None and gt_inst.track is None:
                continue
            elif inst.track is None or gt_inst.track is None:
                print(lf.frame_idx, "None mismatch")
                return False
            elif inst.track.name != gt_inst.track.name:
                print(lf.frame_idx, inst.track.name, gt_inst.track.name)
                return False
    return True


def main(f, dir):
    filename = "tests/data/json_format_v2/centered_pair_predictions.json"

    # gt_filename = "tests/data/json_format_v2/centered_pair_predictions.json"
    # filename = "/Users/tabris/Documents/pni/tracking/000000.mp4.predictions.UDenseNet-ish.2_4_6_8.best_val.centroid_tracker.h5"
    labels = Labels.load_file(
        filename, video_search=Labels.make_video_callback([os.path.dirname(filename)])
    )

    trackers = dict(
        simple=sleap.nn.tracker.simple.SimpleTracker,
        flow=sleap.nn.tracker.flow.FlowTracker,
        simplemaxtracks=sleap.nn.tracker.SimpleMaxTracker,
        flowmaxtracks=sleap.nn.tracker.FlowMaxTracker,
    )
    matchers = dict(
        hungarian=sleap.nn.tracker.components.hungarian_matching,
        greedy=sleap.nn.tracker.components.greedy_matching,
    )
    similarities = dict(
        instance=sleap.nn.tracker.components.instance_similarity,
        centroid=sleap.nn.tracker.components.centroid_distance,
        iou=sleap.nn.tracker.components.instance_iou,
    )
    scales = (
        1,
        0.25,
    )

    def make_tracker(
        tracker_name, matcher_name, sim_name, max_tracks, max_tracking=False, scale=0
    ):
        if tracker_name == "simplemaxtracks" or tracker_name == "flowmaxtracks":
            tracker = trackers[tracker_name](
                matching_function=matchers[matcher_name],
                similarity_function=similarities[sim_name],
                max_tracks=max_tracks,
                max_tracking=max_tracking,
            )
        else:
            tracker = trackers[tracker_name](
                matching_function=matchers[matcher_name],
                similarity_function=similarities[sim_name],
            )
        if scale:
            tracker.candidate_maker.img_scale = scale
        return tracker

    def make_filename(tracker_name, matcher_name, sim_name, scale=0):
        return os.path.join(
            dir,
            f"{tracker_name}_{int(scale * 100)}_{matcher_name}_{sim_name}.h5",
        )

    def make_tracker_and_filename(*args, **kwargs):
        tracker = make_tracker(*args, **kwargs)
        filename = make_filename(*args, **kwargs)
        return tracker, filename

    frames = sorted(
        labels.labeled_frames, key=operator.attrgetter("frame_idx")
    )  # [:100]

    for tracker_name in trackers.keys():
        for matcher_name in matchers.keys():
            for sim_name in similarities.keys():
                if tracker_name == "flow":
                    # If this tracker supports scale, try multiple scales
                    for scale in scales:
                        tracker, gt_filename = make_tracker_and_filename(
                            tracker_name=tracker_name,
                            matcher_name=matcher_name,
                            sim_name=sim_name,
                            scale=scale,
                        )
                        f(frames, tracker, gt_filename)
                elif tracker_name == "flowmaxtracks":
                    # If this tracker supports scale, try multiple scales
                    for scale in scales:
                        tracker, gt_filename = make_tracker_and_filename(
                            tracker_name=tracker_name,
                            matcher_name=matcher_name,
                            sim_name=sim_name,
                            max_tracks=2,
                            max_tracking=True,
                            scale=scale,
                        )
                        f(frames, tracker, gt_filename)
                elif tracker_name == "simplemaxtracks":
                    tracker, gt_filename = make_tracker_and_filename(
                        tracker_name=tracker_name,
                        matcher_name=matcher_name,
                        sim_name=sim_name,
                        max_tracks=2,
                        max_tracking=True,
                        scale=0,
                    )
                    f(frames, tracker, gt_filename)
                else:
                    tracker, gt_filename = make_tracker_and_filename(
                        tracker_name=tracker_name,
                        matcher_name=matcher_name,
                        sim_name=sim_name,
                        scale=0,
                    )
                    f(frames, tracker, gt_filename)


if __name__ == "__main__":
    main(compare_ground_truth, "tracked/")

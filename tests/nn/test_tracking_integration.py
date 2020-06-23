import inspect
import operator
import os
import time

import sleap.nn.tracker.components
from sleap.io.dataset import Labels, LabeledFrame


def make_ground_truth(frames, tracker, gt_filename):
    t0 = time.time()
    new_labels = run_tracker(frames, tracker)
    print(f"{gt_filename}\t{len(tracker.spawned_tracks)}\t{time.time()-t0}")
    Labels.save_file(new_labels, gt_filename)


def compare_ground_truth(frames, tracker, gt_filename):
    t0 = time.time()
    new_labels = run_tracker(frames, tracker)
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


def run_tracker(frames, tracker):
    sig = inspect.signature(tracker.track)
    takes_img = "img" in sig.parameters

    # t0 = time.time()

    new_lfs = []

    # Run tracking on every frame
    for lf in frames:

        # Clear the tracks
        for inst in lf.instances:
            inst.track = None

        track_args = dict(untracked_instances=lf.instances)
        if takes_img:
            track_args["img"] = lf.video[lf.frame_idx]
        else:
            track_args["img"] = None

        new_lf = LabeledFrame(
            frame_idx=lf.frame_idx,
            video=lf.video,
            instances=tracker.track(**track_args),
        )
        new_lfs.append(new_lf)

        # if lf.frame_idx % 100 == 0: print(lf.frame_idx, time.time()-t0)

    # print(time.time() - t0)

    new_labels = Labels()
    new_labels.extend(new_lfs)
    return new_labels


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

    def make_tracker(tracker_name, matcher_name, sim_name, scale=0):
        tracker = trackers[tracker_name](
            matching_function=matchers[matcher_name],
            similarity_function=similarities[sim_name],
        )
        if scale:
            tracker.candidate_maker.img_scale = scale
        return tracker

    def make_filename(tracker_name, matcher_name, sim_name, scale=0):
        return f"{dir}{tracker_name}_{int(scale * 100)}_{matcher_name}_{sim_name}.h5"

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

import inspect
import operator
import os
import time
import pytest
import sleap
from sleap.nn.inference import main as inference_cli
import sleap.nn.tracker.components
from sleap.io.dataset import Labels, LabeledFrame


similarity_args = [
    "instance",
    "normalized_instance",
    "object_keypoint",
    "centroid",
    "iou",
]
match_args = ["hungarian", "greedy"]


@pytest.mark.parametrize(
    "tracker_name", ["simple", "simplemaxtracks", "flow", "flowmaxtracks"]
)
@pytest.mark.parametrize("similarity", similarity_args)
@pytest.mark.parametrize("match", match_args)
def test_kalman_tracker(
    tmpdir, centered_pair_predictions_slp_path, tracker_name, similarity, match
):

    if tracker_name == "flow" or tracker_name == "flowmaxtracks":
        # Expecting ValueError for "flow" or "flowmaxtracks" due to Kalman filter requiring a simple tracker
        with pytest.raises(
            ValueError,
            match="Kalman filter requires simple tracker for initial tracking.",
        ):
            cli = (
                f"--tracking.tracker {tracker_name} "
                "--tracking.max_tracking 1 --tracking.max_tracks 2 "
                f"--tracking.similarity {similarity} "
                f"--tracking.match {match} "
                "--tracking.track_window 5 "
                "--tracking.kf_init_frame_count 10 "
                "--tracking.kf_node_indices 0,1 "
                f"-o {tmpdir}/{tracker_name}.slp "
                f"{centered_pair_predictions_slp_path}"
            )
            inference_cli(cli.split(" "))
    else:
        # For simple or simplemaxtracks, continue with other tests
        # Check for ValueError when similarity is "normalized_instance"
        if similarity == "normalized_instance":
            with pytest.raises(
                ValueError,
                match="Kalman filter does not support normalized_instance_similarity.",
            ):
                cli = (
                    f"--tracking.tracker {tracker_name} "
                    "--tracking.max_tracking 1 --tracking.max_tracks 2 "
                    f"--tracking.similarity {similarity} "
                    f"--tracking.match {match} "
                    "--tracking.track_window 5 "
                    "--tracking.kf_init_frame_count 10 "
                    "--tracking.kf_node_indices 0,1 "
                    f"-o {tmpdir}/{tracker_name}.slp "
                    f"{centered_pair_predictions_slp_path}"
                )
                inference_cli(cli.split(" "))
            return

        # Check for ValueError when kf_node_indices is None which is the default
        with pytest.raises(
            ValueError,
            match="Kalman filter requires node indices for instance tracking.",
        ):
            cli = (
                f"--tracking.tracker {tracker_name} "
                "--tracking.max_tracking 1 --tracking.max_tracks 2 "
                f"--tracking.similarity {similarity} "
                f"--tracking.match {match} "
                "--tracking.track_window 5 "
                "--tracking.kf_init_frame_count 10 "
                f"-o {tmpdir}/{tracker_name}.slp "
                f"{centered_pair_predictions_slp_path}"
            )
            inference_cli(cli.split(" "))

        # Test for missing max_tracks and target_instance_count with kf_init_frame_count
        with pytest.raises(
            ValueError,
            match="Kalman filter requires max tracks or target instance count.",
        ):
            cli = (
                f"--tracking.tracker {tracker_name} "
                f"--tracking.similarity {similarity} "
                f"--tracking.match {match} "
                "--tracking.track_window 5 "
                "--tracking.kf_init_frame_count 10 "
                "--tracking.kf_node_indices 0,1 "
                f"-o {tmpdir}/{tracker_name}.slp "
                f"{centered_pair_predictions_slp_path}"
            )
            inference_cli(cli.split(" "))

        # Test with target_instance_count and without max_tracks
        cli = (
            f"--tracking.tracker {tracker_name} "
            f"--tracking.similarity {similarity} "
            f"--tracking.match {match} "
            "--tracking.track_window 5 "
            "--tracking.kf_init_frame_count 10 "
            "--tracking.kf_node_indices 0,1 "
            "--tracking.target_instance_count 2 "
            f"-o {tmpdir}/{tracker_name}_target_instance_count.slp "
            f"{centered_pair_predictions_slp_path}"
        )
        inference_cli(cli.split(" "))

        labels = sleap.load_file(f"{tmpdir}/{tracker_name}_target_instance_count.slp")
        assert len(labels.tracks) == 2

        # Test with target_instance_count and with max_tracks
        cli = (
            f"--tracking.tracker {tracker_name} "
            "--tracking.max_tracking 1 --tracking.max_tracks 2 "
            f"--tracking.similarity {similarity} "
            f"--tracking.match {match} "
            "--tracking.track_window 5 "
            "--tracking.kf_init_frame_count 10 "
            "--tracking.kf_node_indices 0,1 "
            "--tracking.target_instance_count 2 "
            f"-o {tmpdir}/{tracker_name}_max_tracks_target_instance_count.slp "
            f"{centered_pair_predictions_slp_path}"
        )
        inference_cli(cli.split(" "))

        labels = sleap.load_file(
            f"{tmpdir}/{tracker_name}_max_tracks_target_instance_count.slp"
        )
        assert len(labels.tracks) == 2

        # Test with "--tracking.pre_cull_iou_threshold", "0.8"
        cli = (
            f"--tracking.tracker {tracker_name} "
            "--tracking.max_tracking 1 --tracking.max_tracks 2 "
            f"--tracking.similarity {similarity} "
            f"--tracking.match {match} "
            "--tracking.track_window 5 "
            "--tracking.kf_init_frame_count 10 "
            "--tracking.kf_node_indices 0,1 "
            "--tracking.target_instance_count 2 "
            "--tracking.pre_cull_iou_threshold 0.8 "
            f"-o {tmpdir}/{tracker_name}_max_tracks_target_instance_count_iou.slp "
            f"{centered_pair_predictions_slp_path}"
        )
        inference_cli(cli.split(" "))

        labels = sleap.load_file(
            f"{tmpdir}/{tracker_name}_max_tracks_target_instance_count_iou.slp"
        )
        assert len(labels.tracks) == 2

        # Test with "--tracking.pre_cull_to_target", "1"
        cli = (
            f"--tracking.tracker {tracker_name} "
            "--tracking.max_tracking 1 --tracking.max_tracks 2 "
            f"--tracking.similarity {similarity} "
            f"--tracking.match {match} "
            "--tracking.track_window 5 "
            "--tracking.kf_init_frame_count 10 "
            "--tracking.kf_node_indices 0,1 "
            "--tracking.target_instance_count 2 "
            "--tracking.pre_cull_to_target 1 "
            f"-o {tmpdir}/{tracker_name}_max_tracks_target_instance_count_to_target.slp "
            f"{centered_pair_predictions_slp_path}"
        )
        inference_cli(cli.split(" "))
        labels = sleap.load_file(
            f"{tmpdir}/{tracker_name}_max_tracks_target_instance_count_to_target.slp"
        )
        assert len(labels.tracks) == 2

        # Test with 'tracking.post_connect_single_breaks': 0
        cli = (
            f"--tracking.tracker {tracker_name} "
            "--tracking.max_tracking 1 --tracking.max_tracks 2 "
            f"--tracking.similarity {similarity} "
            f"--tracking.match {match} "
            "--tracking.track_window 5 "
            "--tracking.kf_init_frame_count 10 "
            "--tracking.kf_node_indices 0,1 "
            "--tracking.target_instance_count 2 "
            "--tracking.post_connect_single_breaks 0 "
            f"-o {tmpdir}/{tracker_name}_max_tracks_target_instance_count_single_breaks.slp "
            f"{centered_pair_predictions_slp_path}"
        )
        inference_cli(cli.split(" "))
        labels = sleap.load_file(
            f"{tmpdir}/{tracker_name}_max_tracks_target_instance_count_single_breaks.slp"
        )
        assert len(labels.tracks) == 2


def test_simple_tracker(tmpdir, centered_pair_predictions_slp_path):
    cli = (
        "--tracking.tracker simple "
        "--frames 200-300 "
        f"-o {tmpdir}/simpletracks.slp "
        f"{centered_pair_predictions_slp_path}"
    )
    inference_cli(cli.split(" "))

    labels = sleap.load_file(f"{tmpdir}/simpletracks.slp")
    assert len(labels.tracks) == 27


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
            instances=tracker.track(**track_args, img_hw=lf.image.shape[-3:-1]),
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
        normalized_instance=sleap.nn.tracker.components.normalized_instance_similarity,
        object_keypoint=sleap.nn.tracker.components.factory_object_keypoint_similarity(),
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

"""End-to-end tests of GUI-built inference/tracking commands against sleap-nn.

These run the real `sleap predict` CLI (sleap-nn) on the command the GUI builds, then
merge the results back the way the GUI does, so they catch flags sleap-nn rejects or
interprets differently from what the dialog shows.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap.gui.learning.runners import InferenceTask, VideoItemForInference

CLIP_PREDICTIONS = "tests/data/tracks/clip.predictions.slp"
ROBOT_MODEL = "tests/data/models/minimal_robot.UNet.single_instance"


def make_tracking_project(tmp_path, n_nodes=2, n_frames=40, alternate_order=False):
    """Build an untracked project from the 2-animal clip predictions.

    Adds one user instance on frame 5 and suggestions on frames 10, 20 and 30.

    Args:
        tmp_path: Directory to save the project in.
        n_nodes: Number of skeleton nodes to keep (1 makes a single-node skeleton).
        n_frames: Number of leading frames to keep.
        alternate_order: Reverse the instance order on odd frames, so a tracker that
            matches arbitrarily (rather than by position) swaps identities.

    Returns:
        Tuple of (project path, source labels with ground-truth tracks).
    """
    src = sio.load_slp(CLIP_PREDICTIONS)
    video = src.videos[0]
    skeleton = sio.Skeleton([node.name for node in src.skeletons[0].nodes[:n_nodes]])

    lfs = []
    for lf in src.labeled_frames:
        if lf.frame_idx >= n_frames:
            continue
        insts = [
            sio.PredictedInstance.from_numpy(
                inst.numpy()[:n_nodes],
                skeleton=skeleton,
                score=1.0,
                point_scores=np.ones(n_nodes),
            )
            for inst in lf.predicted_instances
        ]
        if alternate_order and lf.frame_idx % 2:
            insts = insts[::-1]
        lfs.append(
            sio.LabeledFrame(video=video, frame_idx=lf.frame_idx, instances=insts)
        )

    labels = sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skeleton])
    lf5 = labels.find(video, 5)[0]
    lf5.instances.append(
        sio.Instance.from_numpy(lf5.predicted_instances[0].numpy(), skeleton=skeleton)
    )
    labels.suggestions = [
        sio.SuggestionFrame(video=video, frame_idx=i) for i in (10, 20, 30)
    ]

    path = str(tmp_path / "project.slp")
    sio.save_slp(labels, path)
    return path, src


def tracking_params(**overrides):
    """GUI inference params for the simple tracker with the dialog's defaults."""
    params = {
        "tracking.tracker": "simple",
        "tracking.match": "greedy",
        "tracking.track_window": 5,
        "tracking.max_tracks": None,
        "tracking.post_connect_single_breaks": 0,
        "tracking.robust": 1.0,  # "Use max (non-robust)"
        "tracking.similarity": "oks",
        "_prediction_mode": "add",
    }
    params.update(overrides)
    return params


def run_tracking_only(path, frames=None, **param_overrides):
    """Run GUI tracking-only through `sleap predict` and merge into the project."""
    labels = sio.load_slp(path)
    task = InferenceTask(
        trained_job_paths=[],
        inference_params=tracking_params(**param_overrides),
        labels=labels,
        labels_filename=path,
    )
    item = VideoItemForInference(
        video=labels.videos[0], frames=frames, labels_path=path, video_idx=0
    )
    _, ret = task.predict_subprocess(item, append_results=True)
    assert ret == "success"
    n_result_frames = len(task.results)
    task.merge_results()
    return labels, n_result_frames


def count_id_switches(labels, src):
    """Count frame-to-frame identity switches against the source tracks."""
    switches = 0
    prev = None
    for lf in sorted(labels.labeled_frames, key=lambda lf: lf.frame_idx):
        src_lf = src.find(src.videos[0], lf.frame_idx)[0]
        cur = {}
        for inst in lf.predicted_instances:
            if inst.track is None:
                continue
            nearest = min(
                src_lf.predicted_instances,
                key=lambda s: np.nansum((s.numpy()[0] - inst.numpy()[0]) ** 2),
            )
            cur[inst.track.name] = nearest.track.name
        if prev is not None:
            switches += sum(1 for k, v in cur.items() if k in prev and prev[k] != v)
        prev = cur
    return switches


def test_default_tracking_flags_run_in_sleap_nn(tmp_path):
    """Default GUI tracking (max reduction, OKS) runs and tracks both animals."""
    path, src = make_tracking_project(tmp_path)

    labels, _ = run_tracking_only(path)

    assert len(labels.tracks) == 2
    # sleap-nn tracks the user instance on a user-labeled frame (frame 5) in place
    # of that frame's predictions, so only those predictions stay untracked.
    untracked = {
        lf.frame_idx
        for lf in labels
        if any(inst.track is None for inst in lf.predicted_instances)
    }
    assert untracked <= {5}
    assert count_id_switches(labels, src) == 0


def test_single_node_oks_tracks_by_centroid(tmp_path):
    """OKS on a single-node skeleton falls back to centroid distance in sleap-nn.

    OKS of a one-node skeleton is 0 for any motion, so with the OKS flags passed
    explicitly the tracker matched arbitrarily and swapped identities whenever the
    instance order changed between frames.
    """
    path, src = make_tracking_project(tmp_path, n_nodes=1, alternate_order=True)

    labels, _ = run_tracking_only(path, **{"tracking.similarity": "oks"})

    assert len(labels.tracks) == 2
    assert count_id_switches(labels, src) == 0


@pytest.mark.parametrize(
    "mode_params",
    [
        {"_prediction_mode": "add"},
        {"_prediction_mode": "replace"},
        {"_prediction_mode": "add", "_clear_all_first": True},
    ],
    ids=["keep", "replace", "clear-all"],
)
def test_tracking_only_merge_does_not_duplicate(tmp_path, mode_params):
    """Retracking replaces predictions in every mode instead of duplicating them."""
    path, _ = make_tracking_project(tmp_path)
    before = sio.load_slp(path)
    n_preds_before = sum(len(lf.predicted_instances) for lf in before)

    labels, _ = run_tracking_only(path, **mode_params)

    lf5 = labels.find(labels.videos[0], 5)[0]
    assert len(lf5.user_instances) == 1
    assert len(lf5.predicted_instances) == 2
    assert sum(len(lf.predicted_instances) for lf in labels) == n_preds_before
    assert sum(len(lf.user_instances) for lf in labels) == 1


def test_tracking_only_explicit_frames_scope(tmp_path):
    """Explicit `--frames` (how tracking-only targets suggestions) scopes retracking."""
    path, _ = make_tracking_project(tmp_path)

    labels, n_result_frames = run_tracking_only(path, frames=[10, 20, 30])

    assert n_result_frames == 3
    tracked = {
        lf.frame_idx
        for lf in labels
        if any(inst.track is not None for inst in lf.predicted_instances)
    }
    assert tracked == {10, 20, 30}


def test_accelerator_device_runs_model_inference(tmp_path):
    """The Accelerator choice reaches sleap-nn as a `--device` it accepts."""
    labels = sio.load_slp(f"{ROBOT_MODEL}/labels_gt.train.slp")
    labels.replace_filenames(new_filenames=["tests/data/videos/small_robot.mp4"])
    path = str(tmp_path / "robot.slp")
    sio.save_slp(labels, path)
    labels = sio.load_slp(path)

    task = InferenceTask(
        trained_job_paths=[f"{ROBOT_MODEL}/training_config.json"],
        inference_params={"trainer_config.trainer_accelerator": "cpu"},
        labels=labels,
        labels_filename=path,
    )
    item = VideoItemForInference(
        video=labels.videos[0], frames=[0, 1, 2], labels_path=path, video_idx=0
    )
    cli_args, _ = task.make_predict_cli_call(item, output_path=str(tmp_path / "o.slp"))
    assert cli_args[cli_args.index("--device") + 1] == "cpu"

    _, ret = task.predict_subprocess(item, append_results=True)

    assert ret == "success"
    assert sorted(lf.frame_idx for lf in task.results) == [0, 1, 2]

"""Adaptors for legacy CLI commands."""

import click
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import logging

import sleap_io as sio
from sleap_nn.training.model_trainer import ModelTrainer

# from sleap_nn.track import main as track
from sleap_nn.predict import run_inference as predict
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.evaluation import Evaluator
from sleap.sleap_io_adaptors.lf_labels_utils import load_labels_video_search

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("training_job_path", required=True)
@click.argument("labels_path", required=False)
@click.option(
    "--video-paths",
    "video_paths",
    help=(
        "List of paths for finding videos in case paths inside labels file are not "
        "accessible."
    ),
)
@click.option(
    "--val_labels",
    "--val",
    "val_labels",
    help=(
        "Path to labels file to use for validation. If specified, overrides the path "
        "specified in the training job config."
    ),
)
@click.option(
    "--test_labels",
    "--test",
    "test_labels",
    help=(
        "Path to labels file to use for test. If specified, overrides the path "
        "specified in the training job config."
    ),
)
@click.option(
    "--base_checkpoint",
    "base_checkpoint",
    help=(
        "Path to base checkpoint (directory containing best_model.h5) to resume "
        "training from."
    ),
)
@click.option(
    "--save_viz",
    "save_viz",
    is_flag=True,
    help=(
        "Enable saving of prediction visualizations to the run folder if not already "
        "specified in the training job config."
    ),
)
@click.option(
    "--keep_viz",
    "keep_viz",
    is_flag=True,
    help=(
        "Keep prediction visualization images in the run folder after training if "
        "--save_viz is enabled."
    ),
)
@click.option(
    "--zmq",
    "zmq",
    is_flag=True,
    help=(
        "Enable ZMQ logging (for GUI) if not already specified in the training job "
        "config."
    ),
)
@click.option(
    "--run_name",
    "run_name",
    help=("Run name to use when saving file, overrides other run name settings."),
)
@click.option("--prefix", "prefix", help="Prefix to prepend to run name.")
@click.option("--suffix", "suffix", help="Suffix to append to run name.")
@click.option(
    "--cpu",
    "cpu",
    is_flag=True,
    help="Run training only on CPU. If not specified, will use available GPU.",
)
@click.option(
    "--first-gpu",
    "first_gpu",
    is_flag=True,
    help="Run training on the first GPU, if available.",
)
@click.option(
    "--last-gpu",
    "last_gpu",
    is_flag=True,
    help="Run training on the last GPU, if available.",
)
@click.option(
    "--gpu",
    "gpu",
    help=(
        "Run training on the i-th GPU on the system. If 'auto', run on the GPU with "
        "the highest percentage of available memory."
    ),
)
def train_command(
    training_job_path,
    labels_path,
    video_paths,
    val_labels,
    test_labels,
    base_checkpoint,
    save_viz,
    keep_viz,
    zmq,
    run_name,
    prefix,
    suffix,
    cpu,
    first_gpu,
    last_gpu,
    gpu,
):
    """Train a SLEAP model with the specified configuration."""
    # Convert click arguments to the format expected by the training function
    if training_job_path.endswith(".json"):
        config = TrainingJobConfig.load_sleap_config(training_job_path)
    elif training_job_path.endswith((".yaml", ".yml")):
        config = OmegaConf.load(training_job_path)

    if labels_path is not None:
        config.data_config.train_labels_path = [labels_path]

    if video_paths:
        video_paths = video_paths.split(",")
    if val_labels is not None:
        config.data_config.val_labels_path = [val_labels]
    if test_labels is not None:
        config.data_config.test_file_path = [test_labels]
    if base_checkpoint is not None:
        config.model_config.pretrained_backbone_weights = base_checkpoint
        config.model_config.pretrained_head_weights = base_checkpoint
    if save_viz is not None and save_viz:
        config.trainer_config.visualize_preds_during_training = True
    if keep_viz is not None and keep_viz:
        config.trainer_config.keep_viz = True
    if zmq is not None and not zmq:
        config.trainer_config.zmq.publish_port = None
        config.trainer_config.zmq.controller_port = None
    if run_name is not None:
        config.trainer_config.run_name = run_name
    if prefix is not None:
        config.trainer_config.run_name = prefix + config.trainer_config.run_name
    if suffix is not None:
        config.trainer_config.run_name = config.trainer_config.run_name + suffix
    if cpu is not None and cpu:
        config.trainer_config.trainer_accelerator = "cpu"
    if first_gpu:
        config.trainer_config.trainer_accelerator = "gpu"
        config.trainer_config.trainer_device_indices = [0]
    if last_gpu:
        config.trainer_config.trainer_accelerator = "gpu"
        config.trainer_config.trainer_device_indices = [-1]
    if gpu:
        config.trainer_config.trainer_accelerator = "gpu"
        config.trainer_config.trainer_device_indices = [gpu] if gpu != "auto" else None

    # Call the original training function with the arguments
    if video_paths is not None:
        train = load_labels_video_search(labels_path, video_paths)
        val = (
            load_labels_video_search(val_labels, video_paths)
            if val_labels is not None
            else None
        )
    else:
        train = sio.load_slp(labels_path)
        val = sio.load_slp(val_labels) if val_labels is not None else None

    start_train_time = time.time()
    start_timestamp = str(datetime.now())
    logger.info(f"Started training at: {start_timestamp}")

    trainer = ModelTrainer.get_model_trainer_from_config(
        config=config,
        train_labels=[train],
        val_labels=[val] if val is not None else None,
    )
    trainer.train()

    finish_timestamp = str(datetime.now())
    total_elapsed = time.time() - start_train_time
    logger.info(f"Finished training at: {finish_timestamp}")
    logger.info(f"Total training time: {total_elapsed} secs")

    rank = trainer.trainer.global_rank if trainer.trainer is not None else -1

    logger.info(f"Training Config: {OmegaConf.to_yaml(trainer.config)}")

    if rank in [0, -1]:
        # run inference on val dataset
        if trainer.config.trainer_config.save_ckpt:
            data_paths = {}
            for index, path in enumerate(trainer.config.data_config.train_labels_path):
                ckpt_path = (
                    Path(trainer.config.trainer_config.ckpt_dir)
                    / trainer.config.trainer_config.run_name
                ).as_posix()
                logger.info(f"Training labels path for index {index}: {ckpt_path}")
                data_paths[f"train_{index}"] = (
                    Path(trainer.config.trainer_config.ckpt_dir)
                    / trainer.config.trainer_config.run_name
                    / f"labels_train_gt_{index}.slp"
                ).as_posix()
                data_paths[f"val_{index}"] = (
                    Path(trainer.config.trainer_config.ckpt_dir)
                    / trainer.config.trainer_config.run_name
                    / f"labels_val_gt_{index}.slp"
                ).as_posix()

            if (
                OmegaConf.select(config, "data_config.test_file_path", default=None)
                is not None
            ):
                data_paths["test"] = config.data_config.test_file_path

            for d_name, path in data_paths.items():
                labels = sio.load_slp(path)

                pred_labels = predict(
                    data_path=path,
                    model_paths=[
                        Path(trainer.config.trainer_config.ckpt_dir)
                        / trainer.config.trainer_config.run_name
                    ],
                    peak_threshold=0.2,
                    make_labels=True,
                    device=trainer.trainer.strategy.root_device,
                    output_path=Path(trainer.config.trainer_config.ckpt_dir)
                    / trainer.config.trainer_config.run_name
                    / f"pred_{d_name}.slp",
                    ensure_rgb=config.data_config.preprocessing.ensure_rgb,
                    ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
                )

                if not len(pred_labels):
                    logger.info(
                        f"Skipping eval on `{d_name}` dataset as there are no labeled "
                        f"frames..."
                    )
                    continue  # skip if there are no labeled frames

                evaluator = Evaluator(
                    ground_truth_instances=labels, predicted_instances=pred_labels
                )
                metrics = evaluator.evaluate()
                np.savez(
                    (
                        Path(trainer.config.trainer_config.ckpt_dir)
                        / trainer.config.trainer_config.run_name
                        / f"{d_name}_pred_metrics.npz"
                    ).as_posix(),
                    **metrics,
                )

                logger.info(f"---------Evaluation on `{d_name}` dataset---------")
                logger.info(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']}")
                logger.info(f"Average distance: {metrics['distance_metrics']['avg']}")
                logger.info(f"p90 dist: {metrics['distance_metrics']['p90']}")
                logger.info(f"p50 dist: {metrics['distance_metrics']['p50']}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("data_path", required=True)
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    help=(
        "Path to trained model directory (with training_config json/yaml). "
        "Multiple models can be specified, each preceded by --model."
    ),
)
@click.option(
    "--frames",
    "frames",
    help=(
        "List of frames to predict when running on a video. Can be specified as a "
        "comma separated list (e.g. 1,2,3) or a range separated by hyphen (e.g., 1-3, "
        "for 1,2,3). If not provided, defaults to predicting on the entire video."
    ),
)
@click.option(
    "--only-labeled-frames",
    "only_labeled_frames",
    is_flag=True,
    help=(
        "Only run inference on user labeled frames when running on labels dataset. "
        "This is useful for generating predictions to compare against ground truth."
    ),
)
@click.option(
    "--only-suggested-frames",
    "only_suggested_frames",
    is_flag=True,
    help=(
        "Only run inference on unlabeled suggested frames when running on labels "
        "dataset. This is useful for generating predictions for initialization during "
        "labeling."
    ),
)
@click.option(
    "-o",
    "--output",
    "output",
    help=(
        "The output filename or directory path to use for the predicted data. If not "
        "provided, defaults to '[data_path].predictions.slp'."
    ),
)
@click.option("--video.dataset", "video_dataset", help="The dataset for HDF5 videos.")
@click.option(
    "--video.input_format",
    "video_input_format",
    help="The input_format for HDF5 videos.",
)
@click.option(
    "--video.index",
    "video_index",
    type=int,
    help=(
        "Integer index of video in .slp file to predict on. To be used with an .slp "
        "path as an alternative to specifying the video path."
    ),
)
@click.option(
    "--cpu",
    "cpu",
    is_flag=True,
    help="Run inference only on CPU. If not specified, will use available GPU.",
)
@click.option(
    "--first-gpu",
    "first_gpu",
    is_flag=True,
    help="Run inference on the first GPU, if available.",
)
@click.option(
    "--last-gpu",
    "last_gpu",
    is_flag=True,
    help="Run inference on the last GPU, if available.",
)
@click.option(
    "--gpu",
    "gpu",
    help=(
        "Run training on the i-th GPU on the system. If 'auto', run on the GPU with "
        "the highest percentage of available memory."
    ),
)
@click.option(
    "--max_edge_length_ratio",
    "max_edge_length_ratio",
    type=float,
    help=(
        "The maximum expected length of a connected pair of points as a fraction of "
        "the "
        "image size. Candidate connections longer than this length will be penalized "
        "during matching. Only applies to bottom-up (PAF) models."
    ),
)
@click.option(
    "--dist_penalty_weight",
    "dist_penalty_weight",
    type=float,
    help=(
        "A coefficient to scale weight of the distance penalty. Set to values greater "
        "than 1.0 to enforce the distance penalty more strictly. Only applies to "
        "bottom-up (PAF) models."
    ),
)
@click.option(
    "--batch_size",
    "batch_size",
    type=int,
    help=(
        "Number of frames to predict at a time. Larger values result in faster "
        "inference speeds, but require more memory."
    ),
)
@click.option(
    "--open-in-gui",
    "open_in_gui",
    is_flag=True,
    help="Open the resulting predictions in the GUI when finished.",
)
@click.option(
    "--peak_threshold",
    "peak_threshold",
    type=float,
    help="Minimum confidence map value to consider a peak as valid.",
)
@click.option(
    "-n",
    "--max_instances",
    "max_instances",
    type=int,
    help=(
        "Limit maximum number of instances in multi-instance models. Not available for "
        "ID models. Defaults to None."
    ),
)
@click.option(
    "--tracking.tracker",
    "tracking_tracker",
    help="Options: simple, flow, simplemaxtracks, flowmaxtracks, None (default: None)",
)
@click.option(
    "--tracking.max_tracking",
    "tracking_max_tracking",
    type=int,
    help="If 1 (True) then the tracker will cap the max number of tracks. 0 (False)",
)
@click.option(
    "--tracking.max_tracks",
    "tracking_max_tracks",
    type=int,
    help="Maximum number of tracks to be tracked by the tracker. (default: None)",
)
@click.option(
    "--tracking.post_connect_single_breaks",
    "tracking_post_connect_single_breaks",
    type=int,
    help=(
        "If non-zero and target_instance_count is also non-zero, then connect track "
        "breaks when exactly one track is lost and exactly one track is spawned in "
        "frame. (default: 0)"
    ),
)
@click.option(
    "--tracking.similarity",
    "tracking_similarity",
    help=(
        "Options: instance, normalized_instance, object_keypoint, centroid, iou "
        "(default: instance)"
    ),
)
@click.option(
    "--tracking.match",
    "tracking_match",
    help="Options: hungarian, greedy (default: greedy)",
)
@click.option(
    "--tracking.robust",
    "tracking_robust",
    type=float,
    help=(
        "Robust quantile of similarity score for instance matching. If equal to 1, "
        "keep the max similarity score (non-robust). (default: 1)"
    ),
)
@click.option(
    "--tracking.track_window",
    "tracking_track_window",
    type=int,
    help="How many frames back to look for matches (default: 5)",
)
@click.option(
    "--tracking.min_new_track_points",
    "tracking_min_new_track_points",
    type=int,
    help="Minimum number of instance points for spawning new track (default: 0)",
)
@click.option(
    "--tracking.min_match_points",
    "tracking_min_match_points",
    type=int,
    help="Minimum points for match candidates (default: 0)",
)
@click.option(
    "--tracking.img_scale",
    "tracking_img_scale",
    type=float,
    help="For optical-flow: Image scale (default: 1.0)",
)
@click.option(
    "--tracking.of_window_size",
    "tracking_of_window_size",
    type=int,
    help=(
        "For optical-flow: Optical flow window size to consider at each pyramid "
        "(default: 21)"
    ),
)
@click.option(
    "--tracking.of_max_levels",
    "tracking_of_max_levels",
    type=int,
    help="For optical-flow: Number of pyramid scale levels to consider (default: 3)",
)
def track_command(
    data_path,
    models,
    frames,
    only_labeled_frames,
    only_suggested_frames,
    output,
    video_dataset,
    video_input_format,
    video_index,
    cpu,
    first_gpu,
    last_gpu,
    gpu,
    max_edge_length_ratio,
    dist_penalty_weight,
    batch_size,
    open_in_gui,
    peak_threshold,
    max_instances,
    tracking_tracker,
    tracking_max_tracking,
    tracking_max_tracks,
    tracking_post_connect_single_breaks,
    tracking_similarity,
    tracking_match,
    tracking_robust,
    tracking_track_window,
    tracking_min_new_track_points,
    tracking_min_match_points,
    tracking_img_scale,
    tracking_of_window_size,
    tracking_of_max_levels,
):
    """Track instances in video data using trained SLEAP models."""
    import torch

    # Build kwargs for the tracking function
    kwargs = {}

    if models is not None:
        kwargs["model_paths"] = models
    if frames is not None:
        kwargs["frames"] = frames
    if only_labeled_frames is not None and only_labeled_frames:
        kwargs["only_labeled_frames"] = True
    if only_suggested_frames is not None and only_suggested_frames:
        kwargs["only_suggested_frames"] = True
    kwargs["output_path"] = output
    if output is None:
        kwargs["output_path"] = f"{data_path}.predictions.slp"
    if video_dataset is not None:
        kwargs["video_dataset"] = video_dataset
    if video_input_format is not None:
        kwargs["video_input_format"] = video_input_format
    if video_index is not None:
        kwargs["video_index"] = video_index
    if cpu is not None and cpu:
        kwargs["device"] = "cpu"
    if torch.cuda.is_available():
        if first_gpu:
            kwargs["device"] = "cuda:0"
        if last_gpu:
            n_gpus = torch.cuda.device_count()
            kwargs["device"] = f"cuda:{n_gpus - 1}"
        if gpu:
            kwargs["device"] = f"cuda:{gpu}" if gpu != "auto" else "cuda"
    if max_edge_length_ratio is not None:
        kwargs["max_edge_length_ratio"] = max_edge_length_ratio
    if dist_penalty_weight is not None:
        kwargs["dist_penalty_weight"] = dist_penalty_weight
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    # if open_in_gui:
    #     kwargs['open_in_gui'] = True #TODO
    if peak_threshold is not None:
        kwargs["peak_threshold"] = peak_threshold
    if max_instances is not None:
        kwargs["max_instances"] = max_instances

    if tracking_tracker:
        if "flow" in tracking_tracker:
            kwargs["use_flow"] = True

    # Check max_tracking flag int value for True/False
    if tracking_max_tracking is not None and tracking_max_tracking == 1:
        kwargs["candidates_method"] = "local_queues"
        kwargs["max_tracks"] = tracking_max_tracks

    if tracking_similarity is not None:
        if tracking_similarity == "oks":
            kwargs["features"] = "keypoints"
            kwargs["scoring_method"] = "oks"
        elif tracking_similarity == "centroids":
            kwargs["features"] = "centroids"
            kwargs["scoring_method"] = "euclidean_dist"
        elif tracking_similarity == "iou":
            kwargs["features"] = "bboxes"
            kwargs["scoring_method"] = "iou"
    if (
        tracking_post_connect_single_breaks is not None
        and tracking_post_connect_single_breaks
    ):
        kwargs["post_connect_single_breaks"] = tracking_post_connect_single_breaks
    if tracking_match is not None:
        kwargs["track_matching_method"] = tracking_match
    if tracking_robust is not None:
        kwargs["robust_best_instance"] = tracking_robust
    if tracking_track_window is not None:
        kwargs["tracking_window_size"] = tracking_track_window
    if tracking_min_new_track_points is not None:
        kwargs["min_new_track_points"] = tracking_min_new_track_points
    if tracking_min_match_points is not None:
        kwargs["min_match_points"] = tracking_min_match_points
    if tracking_img_scale is not None:
        kwargs["of_img_scale"] = tracking_img_scale
    if tracking_of_window_size is not None:
        kwargs["of_window_size"] = tracking_of_window_size
    if tracking_of_max_levels is not None:
        kwargs["of_max_levels"] = tracking_of_max_levels

    # # Call the original tracking function with kwargs
    predict(data_path=data_path, **kwargs)

    if open_in_gui:
        import subprocess

        # Launch SLEAP GUI with the output file
        subprocess.run(["sleap-label", str(kwargs.get("output_path", ""))])

"""CLI commands for SLEAP-NN training and inference."""

import logging
from pathlib import Path

import click
from click import Command

logger = logging.getLogger(__name__)


class TrainCommand(Command):
    """Custom command class that overrides help behavior for train command."""

    def format_help(self, ctx, formatter):
        """Override the help formatting to show custom training help."""
        show_training_help()


def show_training_help():
    """Display training help information."""
    help_text = """
sleap-nn-train — Train SLEAP models from a config YAML file.

Usage:
  sleap-nn-train --config-dir <dir> --config-name <name> [overrides]

Common overrides:
  trainer_config.max_epochs=100
  trainer_config.batch_size=32

Examples:
  Start new run:
    sleap-nn-train --config-dir /path/to/config_dir/ --config-name myrun
  Resume 20 more epochs:
    sleap-nn-train --config-dir /path/to/config_dir/ --config-name myrun \\
      trainer_config.resume_ckpt_path=<path/to/ckpt> \\
      trainer_config.max_epochs=20

For a detailed list of all available config options, please refer to https://nn.sleap.ai/config/.
"""
    click.echo(help_text)


@click.command(cls=TrainCommand)
@click.option("--config-name", "-c", type=str, help="Configuration file name")
@click.option(
    "--config-dir", "-d", type=str, default=".", help="Configuration directory path"
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train(config_name, config_dir, overrides):
    """Run training workflow with Hydra config overrides.

    (From `sleap-nn`: `sleap-nn train`)

    Examples:
        sleap-nn-train --config-name myconfig --config-dir /path/to/config_dir/
        sleap-nn-train -c myconfig -d /path/to/config_dir/ trainer_config.max_epochs=100
        sleap-nn-train -c myconfig -d /path/to/config_dir/ +experiment=new_model
    """
    # Import sleap-nn modules inside function
    try:
        from sleap_nn.train import run_training
        from omegaconf import OmegaConf
        import hydra

        # Show help if no config name provided
        if not config_name:
            show_training_help()
            return

        # Initialize Hydra manually
        # resolve the path to the config directory (hydra expects absolute path)
        config_dir = Path(config_dir).resolve().as_posix()
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            # Compose config with overrides
            cfg = hydra.compose(config_name=config_name, overrides=list(overrides))

            # Validate config
            if not hasattr(cfg, "model_config") or not cfg.model_config:
                click.echo(
                    "No model config found! Use `sleap-nn-train --help` for more info."
                )
                raise click.Abort()

            print("Input config:")
            print("\n" + OmegaConf.to_yaml(cfg))
            run_training(cfg)

    except ImportError:
        logger.error(
            "sleap-nn is not installed. This appears to be a GUI-only installation. "
            "To enable training, please install SLEAP with the 'nn' dependency. "
            "See the installation guide: https://docs.sleap.ai/latest/installation/"
        )


@click.command()
@click.option(
    "--data_path",
    "-i",
    type=str,
    required=True,
    help="Path to data to predict on. Can be a labels (.slp) file or video format.",
)
@click.option(
    "--model_paths",
    "-m",
    multiple=True,
    help="Path to trained model directory (with training_config.json). "
    "Multiple models can be specified, each preceded by --model_paths.",
)
@click.option(
    "--output_path",
    "-o",
    type=str,
    default=None,
    help="The output filename for predicted data. "
    "If not provided, defaults to '[data_path].slp'.",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="auto",
    help="Device on which torch.Tensor will be allocated. "
    "One of ('cpu', 'cuda', 'mps', 'auto'). Default: 'auto' "
    "(based on available backend). If `cuda` is available, "
    "you could also use `cuda:0` to specify the device.",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    default=4,
    help="Number of frames to predict at a time. "
    "Larger values result in faster inference speeds, but require more memory.",
)
@click.option(
    "--tracking",
    "-t",
    is_flag=True,
    default=False,
    help="If True, runs tracking on the predicted instances.",
)
@click.option(
    "-n",
    "--max_instances",
    type=int,
    default=None,
    help="Limit maximum number of instances in multi-instance models. "
    "Not available for ID models. Defaults to None.",
)
@click.option(
    "--backbone_ckpt_path",
    type=str,
    default=None,
    help="To run inference on any `.ckpt` other than `best.ckpt` from the "
    "`model_paths` dir, the path to the `.ckpt` file should be passed here.",
)
@click.option(
    "--head_ckpt_path",
    type=str,
    default=None,
    help="Path to `.ckpt` file if a different set of head layer weights are to be used."
    "If `None`, the `best.ckpt` from `model_paths` dir is used "
    "(or the ckpt from `backbone_ckpt_path` if provided).",
)
@click.option(
    "--max_height",
    type=int,
    default=None,
    help="Maximum height the image should be padded to. "
    "If not provided, the values from the training config are used.",
)
@click.option(
    "--max_width",
    type=int,
    default=None,
    help="Maximum width the image should be padded to. "
    "If not provided, the values from the training config are used.",
)
@click.option(
    "--input_scale",
    type=float,
    default=None,
    help="Scale factor to apply to the input image. "
    "If not provided, the values from the training config are used.",
)
@click.option(
    "--ensure_rgb",
    is_flag=True,
    default=False,
    help="True if the input image should have 3 channels (RGB image). "
    "If input has only one channel when this is set to `True`, then the images "
    "from single-channel is replicated along the channel axis. If the image has "
    "three channels and this is set to False, then we retain the three channels.",
)
@click.option(
    "--ensure_grayscale",
    is_flag=True,
    default=False,
    help="True if the input image should only have a single channel. "
    "If input has three channels (RGB) and this is set to True, then we convert "
    "the image to grayscale (single-channel) image. If the source image has only "
    "1 channel and this is set to False, then we retain the single channel input.",
)
@click.option(
    "--anchor_part",
    type=str,
    default=None,
    help="The node name to use as the anchor for the centroid. "
    "If not provided, the anchor part in the `training_config.yaml` is used.",
)
@click.option(
    "--only_labeled_frames",
    is_flag=True,
    default=False,
    help="Only run inference on user labeled frames when running on labels dataset. "
    "This is useful for generating predictions to compare against ground truth.",
)
@click.option(
    "--only_suggested_frames",
    is_flag=True,
    default=False,
    help="Only run inference on unlabeled suggested frames when running on"
    "labels dataset.This is useful for generating predictions for initialization.",
)
@click.option(
    "--exclude_user_labeled",
    is_flag=True,
    default=False,
    help="Skip frames that have user-labeled instances. Useful when predicting on "
    "entire video but skipping already-labeled frames.",
)
@click.option(
    "--no_empty_frames",
    is_flag=True,
    default=False,
    help=("Clear empty frames that did not have predictions before saving to output."),
)
@click.option(
    "--video_index",
    type=int,
    default=None,
    help="Integer index of video in .slp file to predict on. "
    "To be used with an .slp path as an alternative to specifying the video path.",
)
@click.option(
    "--video_dataset", type=str, default=None, help="The dataset for HDF5 videos."
)
@click.option(
    "--video_input_format",
    type=str,
    default="channels_last",
    help="The input_format for HDF5 videos.",
)
@click.option(
    "--frames",
    type=str,
    default="",
    help="List of frames to predict when running on a video. Can be specified as a "
    "comma separated list (e.g. 1,2,3) or a range separated by hyphen "
    "(e.g., 1-3, for 1,2,3). If not provided, defaults to predicting on entire "
    "video.",
)
@click.option(
    "--integral_patch_size",
    type=int,
    default=5,
    help="Size of patches to crop around each rough peak as an integer scalar. "
    "Default: 5.",
)
@click.option(
    "--max_edge_length_ratio",
    type=float,
    default=0.25,
    help="The maximum expected length of a connected pair of points as a fraction of "
    "the image size. Candidate connections longer than this length will be "
    "penalized during matching. Default: 0.25.",
)
@click.option(
    "--dist_penalty_weight",
    type=float,
    default=1.0,
    help="A coefficient to scale weight of the distance penalty as a scalar float. "
    "Set to values greater than 1.0 to enforce the distance penalty more strictly."
    "Default: 1.0.",
)
@click.option(
    "--n_points",
    type=int,
    default=10,
    help="Number of points to sample along the line integral. Default: 10.",
)
@click.option(
    "--min_instance_peaks",
    type=float,
    default=0,
    help="Min number of peaks instance should have to be considered a real instance."
    "Instances with fewer peaks than this will be discarded "
    "(useful for filtering spurious detections). Default: 0.",
)
@click.option(
    "--min_line_scores",
    type=float,
    default=0.25,
    help="Minimum line score (between -1 and 1) required to form a match between "
    "candidate point pairs. Useful for rejecting spurious detections when"
    "there are no better ones. Default: 0.25.",
)
@click.option(
    "--queue_maxsize",
    type=int,
    default=8,
    help="Maximum size of the frame buffer queue.",
)
@click.option(
    "--crop_size",
    type=int,
    default=None,
    help="Crop size. If not provided, the crop size from training_config.yaml is used.",
)
@click.option(
    "--peak_threshold",
    type=float,
    default=0.2,
    help="Minimum confidence map value to consider a peak as valid.",
)
@click.option(
    "--integral_refinement",
    type=str,
    default="integral",
    help="If `None`, returns the grid-aligned peaks with no refinement. "
    "If `'integral'`, peaks will be refined with integral regression. "
    "Default: 'integral'.",
)
@click.option(
    "--tracking_window_size",
    type=int,
    default=5,
    help="Number of frames to look for in the candidate instances to match with the "
    "current detections.",
)
@click.option(
    "--min_new_track_points",
    type=int,
    default=0,
    help="We won't spawn a new track for an instance with fewer than this many points.",
)
@click.option(
    "--candidates_method",
    type=str,
    default="fixed_window",
    help="Either of `fixed_window` or `local_queues`. In fixed window method, "
    "candidates from the last `window_size` frames. In local queues, last "
    "`window_size` instances for each track ID is considered for matching against "
    "the current detection.",
)
@click.option(
    "--min_match_points",
    type=int,
    default=0,
    help="Minimum non-NaN points for match candidates.",
)
@click.option(
    "--features",
    type=str,
    default="keypoints",
    help="Feature representation for the candidates to update current detections. "
    "One of [`keypoints`, `centroids`, `bboxes`, `image`].",
)
@click.option(
    "--scoring_method",
    type=str,
    default="oks",
    help="Method to compute association score between features from the current frame "
    "and previous tracks. options:[`oks`, `cosine_sim`, `iou`, `euclidean_dist`].",
)
@click.option(
    "--scoring_reduction",
    type=str,
    default="mean",
    help="Method to aggregate multiple scores if there are several detections"
    "associated with the same track. One of [`mean`, `max`, `robust_quantile`].",
)
@click.option(
    "--robust_best_instance",
    type=float,
    default=1.0,
    help="If the value is between 0 and 1 (excluded), use a robust quantile similarity "
    "score for the track. If the value is 1, use the max similarity (non-robust). "
    "For selecting a robust score, 0.95 is a good value.",
)
@click.option(
    "--track_matching_method",
    type=str,
    default="hungarian",
    help="Track matching algorithm. One of `hungarian`, `greedy`.",
)
@click.option(
    "--max_tracks",
    type=int,
    default=None,
    help="Maximum number of new tracks to be created to avoid redundant tracks. "
    "(only for local queues candidate)",
)
@click.option(
    "--use_flow",
    is_flag=True,
    default=False,
    help="If True, `FlowShiftTracker` is used, where the poses are matched using "
    "optical flow shifts.",
)
@click.option(
    "--of_img_scale",
    type=float,
    default=1.0,
    help="Factor to scale the images by when computing optical flow. Decrease this to "
    "increase performance at the cost of finer accuracy. Sometimes decreasing the "
    "image scale can improve performance with fast movements.",
)
@click.option(
    "--of_window_size",
    type=int,
    default=21,
    help="Optical flow window size to consider at each pyramid scale level.",
)
@click.option(
    "--of_max_levels",
    type=int,
    default=3,
    help="Number of pyramid scale levels to consider. This is different from the scale "
    "parameter, which determines the initial image scaling.",
)
@click.option(
    "--post_connect_single_breaks",
    is_flag=True,
    default=False,
    help="If True and `max_tracks` is not None with local queues candidate method, "
    "connects track breaks when exactly one track is lost and exactly"
    "one new track is spawned in the frame.",
)
@click.option(
    "--tracking_target_instance_count",
    type=int,
    default=0,
    help="Target number of instances to track per frame. (default: 0)",
)
@click.option(
    "--tracking_pre_cull_to_target",
    type=int,
    default=0,
    help=(
        "If non-zero and target_instance_count is also non-zero, then cull instances "
        "over target count per frame *before* tracking. (default: 0)"
    ),
)
@click.option(
    "--tracking_pre_cull_iou_threshold",
    type=float,
    default=0,
    help=(
        "If non-zero and pre_cull_to_target also set, then use IOU threshold to remove "
        "overlapping instances over count *before* tracking. (default: 0)"
    ),
)
@click.option(
    "--tracking_clean_instance_count",
    type=int,
    default=0,
    help="Target number of instances to clean *after* tracking. (default: 0)",
)
@click.option(
    "--tracking_clean_iou_threshold",
    type=float,
    default=0,
    help="IOU to use when culling instances *after* tracking. (default: 0)",
)
def track(**kwargs):
    """Run Inference and Tracking workflow.

    (From `sleap-nn`: `sleap-nn track`)
    """
    # Import sleap-nn modules inside function
    try:
        from sleap_nn.predict import run_inference, frame_list

        # Convert model_paths from tuple to list
        if "model_paths" in kwargs and kwargs["model_paths"]:
            kwargs["model_paths"] = list(kwargs["model_paths"])
        else:
            kwargs["model_paths"] = None

        # Convert frames string to list
        if "frames" in kwargs and kwargs["frames"]:
            kwargs["frames"] = frame_list(kwargs["frames"])
        else:
            kwargs["frames"] = None

        # Call the original function
        return run_inference(**kwargs)

    except ImportError:
        logger.error(
            "sleap-nn is not installed. This appears to be a GUI-only installation. "
            "To enable training, please install SLEAP with the 'nn' dependency. "
            "See the installation guide: https://docs.sleap.ai/latest/installation/"
        )

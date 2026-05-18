"""
Command line utility which prints data about labels file.
"""

import os
from sleap.sleap_io_adaptors.instance_utils import bounding_box
from sleap.sleap_io_adaptors.lf_labels_utils import get_labeled_frame_count
from sleap.sleap_io_adaptors.lf_labels_utils import labels_load_file
from sleap.util import show_sleap_nn_installation_message
from sleap.gui.learning.load_legacy_metrics import load_npz_extract_arrays


def describe_labels(data_path, verbose=False):
    from sleap.sleap_io_adaptors.lf_labels_utils import (
        make_video_callback,
    )

    video_callback = make_video_callback([os.path.dirname(data_path)])
    labels = labels_load_file(data_path, video_search=video_callback)

    print(f"Labeled frames: {len(labels)}")
    print(f"Tracks: {len(labels.tracks)}")

    print("Video files:")

    total_user_frames = 0

    for vid in labels.videos:
        print(f"  {vid.filename}")

        lfs = labels.find(vid)

        print(f"    labeled frames: {len(lfs)}")

        if not lfs:
            continue

        first_idx = min((lf.frame_idx for lf in lfs))
        last_idx = max((lf.frame_idx for lf in lfs))

        tracks = {inst.track for lf in lfs for inst in lf}
        concurrent_count = max((len(lf.instances) for lf in lfs))
        user_frames = get_labeled_frame_count(labels, vid, "user")

        total_user_frames += user_frames

        print(f"    labeled frames from {first_idx} to {last_idx}")
        print(f"    user labeled frames: {user_frames}")
        print(f"    tracks: {len(tracks)}")
        print(f"    max instances in frame: {concurrent_count}")

        if verbose:
            print()
            print("    labeled frames:              bounding box top left (x, y)")
            for lf in lfs:
                bb_cords = [
                    f"({bounding_box(inst)[0, 1]:.2f}, {bounding_box(inst)[0, 0]:.2f})"
                    f"{'^' if hasattr(inst, 'score') else ''}"
                    for inst in lf.instances
                ]
                pt_str = " ".join(bb_cords)
                print(
                    f"      frame {lf.frame_idx}: {len(lf.instances)} instances -> "
                    f"{pt_str}"
                )
            print()

    print(f"Total user labeled frames: {total_user_frames}")

    if labels.provenance:
        print()
        print("Provenance:")

        for key, value in labels.provenance.items():
            print(f"  {key}: {value}")


def describe_model(model_path, verbose=False):
    import numpy as np
    from omegaconf import OmegaConf

    print("=====")
    print("Model:", model_path)
    print("=====")

    def rel_path(x):
        return os.path.join(model_path, x)

    if "training_config.json" in os.listdir(model_path):
        try:
            from sleap_nn.config.training_job_config import TrainingJobConfig

            cfg = TrainingJobConfig.load_sleap_config(rel_path("training_config.json"))
        except ImportError:
            show_sleap_nn_installation_message()
            print(
                "sleap-nn is not installed. This appears to be a GUI-only installation."
                "To enable training, please install SLEAP with the 'nn' dependency."
                "See the installation guide: https://docs.sleap.ai/latest/installation/"
            )
            return
    elif "training_config.yaml" in os.listdir(model_path):
        cfg = OmegaConf.load(rel_path("training_config.yaml"))

    print("=====")
    print("Heads:")
    print("=====")
    print(cfg.model_config.head_configs)
    print("=====")
    print()

    print("=====")
    print("Backbone:")
    print("=====")
    print(cfg.model_config.backbone_config)
    print("=====")
    print()
    print()

    def describe_metrics(metrics, legacy):
        if legacy:
            if isinstance(metrics, str):
                metrics = load_npz_extract_arrays(metrics)

            p90 = metrics.get("metrics[0].dist.p90").item()
            p95 = metrics.get("metrics[0].dist.p95").item()
            p99 = metrics.get("metrics[0].dist.p99").item()
            print(f"Dist (90%/95%/99%): {p90} / {p95} / {p99}")

            oks_map = metrics.get("metrics[0].oks_voc.mAP").item()
            oks_mar = metrics.get("metrics[0].oks_voc.mAR").item()
            print(f"OKS VOC (mAP / mAR): {oks_map} / {oks_mar}")

            pck_min = metrics.get("metrics[0].pck.thresholds")[0]
            pck_max = metrics.get("metrics[0].pck.thresholds")[-1]
            mpck = metrics.get("metrics[0].pck.mPCK").item()
            print(f"PCK (mean {pck_min}-{pck_max} px): {mpck}")
        else:
            if isinstance(metrics, str):
                with np.load(metrics, allow_pickle=True) as data:
                    display_data = {
                        "dist.p99": data["distance_metrics"].item().get("p99"),
                        "dist.p95": data["distance_metrics"].item().get("p95"),
                        "dist.p90": data["distance_metrics"].item().get("p90"),
                        "oks_voc.mAP": data["voc_metrics"].item().get("oks_voc.mAP"),
                        "oks_voc.mAR": data["voc_metrics"].item().get("oks_voc.mAR"),
                        "pck.mPCK": data["pck_metrics"].item().get("mPCK"),
                        "pck.thresholds": data["pck_metrics"].item().get("thresholds"),
                    }

            print(
                f"Dist (90%/95%/99%): {display_data['dist.p90']} / "
                f"{display_data['dist.p95']} / {display_data['dist.p99']}"
            )
            print(
                f"OKS VOC (mAP / mAR): {display_data['oks_voc.mAP']} / "
                f"{display_data['oks_voc.mAR']}"
            )
            print(
                f"PCK (mean {display_data['pck.thresholds'][0]}-"
                f"{display_data['pck.thresholds'][-1]} px): {display_data['pck.mPCK']}"
            )

    def describe_dataset(split_name):
        # Check whether the checkpoint files are sleap_nn or legacy sleap
        # and load the labels accordingly.
        # Try new naming first, then legacy, then old sleap-nn naming
        labels = None
        gt_new = rel_path(f"labels_gt.{split_name}.0.slp")  # v0.1.0+
        gt_legacy = rel_path(f"labels_gt.{split_name}.slp")  # legacy
        gt_old = rel_path(f"labels_{split_name}_gt_0.slp")  # < v0.1.0
        if os.path.exists(gt_new):
            labels = labels_load_file(gt_new)
        elif os.path.exists(gt_legacy):
            labels = labels_load_file(gt_legacy)
        elif os.path.exists(gt_old):
            labels = labels_load_file(gt_old)

        if labels is not None:
            labeled_frames_user = [
                lf for lf in labels.labeled_frames if lf.has_user_instances
            ]
            user_instances = [
                inst for lf in labeled_frames_user for inst in lf.user_instances
            ]
            print(
                f"Frames: {len(labeled_frames_user)} / Instances: {len(user_instances)}"
            )

        # Try new naming first, then old sleap-nn naming, then legacy naming
        metrics_new = rel_path(f"metrics.{split_name}.0.npz")  # v0.1.0+
        metrics_old = rel_path(f"{split_name}_0_pred_metrics.npz")  # < v0.1.0
        metrics_legacy = rel_path(f"metrics.{split_name}.npz")  # legacy
        if os.path.exists(metrics_new):
            print("Metrics:")
            describe_metrics(metrics_new, legacy=False)
        elif os.path.exists(metrics_old):
            print("Metrics:")
            describe_metrics(metrics_old, legacy=False)
        elif os.path.exists(metrics_legacy):
            print("Metrics:")
            describe_metrics(metrics_legacy, legacy=True)

    print("=====")
    print("Training set:")
    print("=====")
    describe_dataset("train")
    print("=====")
    print()

    print("=====")
    print("Validation set:")
    print("=====")
    describe_dataset("val")
    print("=====")
    print()

    print("=====")
    print("Test set:")
    print("=====")
    describe_dataset("test")
    print("=====")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels file (.slp) or model folder")
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    if args.data_path.endswith(".slp"):
        describe_labels(args.data_path, verbose=args.verbose)

    elif os.path.isdir(args.data_path):
        if os.path.exists(
            os.path.join(args.data_path, "training_config.yaml")
        ) or os.path.exists(os.path.join(args.data_path, "training_config.json")):
            describe_model(args.data_path, verbose=args.verbose)


if __name__ == "__main__":
    main()

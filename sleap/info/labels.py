"""
Command line utility which prints data about labels file.
"""
import os


def describe_labels(data_path, verbose=False):
    from sleap.io.dataset import Labels

    video_callback = Labels.make_video_callback([os.path.dirname(data_path)])
    labels = Labels.load_file(data_path, video_search=video_callback)

    print(f"Labeled frames: {len(labels)}")
    print(f"Tracks: {len(labels.tracks)}")

    print(f"Video files:")

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
        user_frames = labels.get_labeled_frame_count(vid, "user")

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
                    f"({inst.bounding_box[0]:.2f}, {inst.bounding_box[1]:.2f}){'^' if hasattr(inst, 'score') else ''}"
                    for inst in lf.instances
                ]
                pt_str = " ".join(bb_cords)
                print(
                    f"      frame {lf.frame_idx}: {len(lf.instances)} instances -> {pt_str}"
                )
            print()

    print(f"Total user labeled frames: {total_user_frames}")

    if labels.provenance:
        print()
        print(f"Provenance:")

        for key, value in labels.provenance.items():
            print(f"  {key}: {value}")


def describe_model(model_path, verbose=False):
    import sleap
    import numpy as np

    print("=====")
    print("Model:", model_path)
    print("=====")

    rel_path = lambda x: os.path.join(model_path, x)

    initial_cfg = sleap.load_config(rel_path("initial_config.json"))
    cfg = sleap.load_config(rel_path("training_config.json"))

    print("=====")
    print("Heads:")
    print("=====")
    print(cfg.model.heads)
    print("=====")
    print()

    print("=====")
    print("Backbone:")
    print("=====")
    print(cfg.model.backbone)
    print("=====")
    print()
    print()

    def describe_metrics(metrics):
        if isinstance(metrics, str):
            metrics = np.load(metrics, allow_pickle=True)["metrics"].tolist()

        print(
            f"Dist (90%/95%/99%): {metrics['dist.p90']} / {metrics['dist.p95']} / {metrics['dist.p99']}"
        )
        print(
            f"OKS VOC (mAP / mAR): {metrics['oks_voc.mAP']} / {metrics['oks_voc.mAR']}"
        )
        print(
            f"PCK (mean {metrics['pck.thresholds'][0]}-{metrics['pck.thresholds'][-1]} px): {metrics['pck.mPCK']}"
        )

    def describe_dataset(split_name):
        if os.path.exists(rel_path(f"labels_gt.{split_name}.slp")):
            labels = sleap.load_file(rel_path(f"labels_gt.{split_name}.slp"))
            print(
                f"Frames: {len(labels.user_labeled_frames)} / Instances: {len(labels.user_instances)}"
            )

        if os.path.exists(rel_path(f"metrics.{split_name}.npz")):
            print("Metrics:")
            describe_metrics(rel_path(f"metrics.{split_name}.npz"))

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
        if os.path.exists(os.path.join(args.data_path, "training_config.json")):
            describe_model(args.data_path, verbose=args.verbose)


if __name__ == "__main__":
    main()

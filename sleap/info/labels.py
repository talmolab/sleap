"""
Command line utility which prints data about labels file.
"""
import os

from sleap.io.dataset import Labels


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    video_callback = Labels.make_video_callback([os.path.dirname(args.data_path)])
    labels = Labels.load_file(args.data_path, video_search=video_callback)

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

        if args.verbose:
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


if __name__ == "__main__":
    main()

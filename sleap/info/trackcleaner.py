"""
Command line utility for merging breaks in the predicted tracks.

Method:
1. You specify how many instances there should be in each frame.
2. The lowest scoring instances beyond this limit are deleting from each frame.
3. Going frame by frame, any time there's exactly one missing track and exactly
   one new track, we merge the new track into the missing track.

You should review the results to check for "swaps". This can be done using the
velocity threshold suggestion method.
"""

import operator

from sleap import Labels


def fit_tracks(filename, instance_count: int):
    labels = Labels.load_file(filename)

    video = labels.videos[0]

    lf_inst_list = []
    # Find all instances contained in selected area
    for lf in labels.find(video):
        if len(lf.predicted_instances) > instance_count:
            # Get all but the instance_count many instances with the highest score
            extra_instances = sorted(
                lf.predicted_instances, key=operator.attrgetter("score")
            )[:-instance_count]
            lf_inst_list.extend([(lf, inst) for inst in extra_instances])

    # Remove instances over per frame threshold
    for lf, inst in lf_inst_list:
        labels.remove_instance(lf, inst)

    # Move instances in new tracks into tracks that disappeared on previous frame
    fix_track_map = dict()
    last_good_frame_tracks = {inst.track for inst in labels.find_first(video).instances}
    for lf in labels.find(video):
        frame_tracks = {inst.track for inst in lf.instances}

        tracks_fixed_before = frame_tracks.intersection(set(fix_track_map.keys()))
        if tracks_fixed_before:
            for inst in lf.instances:
                if (
                    inst.track in fix_track_map
                    and fix_track_map[inst.track] not in frame_tracks
                ):
                    inst.track = fix_track_map[inst.track]
                    frame_tracks = {inst.track for inst in lf.instances}

        extra_tracks = frame_tracks - last_good_frame_tracks
        missing_tracks = last_good_frame_tracks - frame_tracks

        # print(f"{lf.frame_idx}: {extra_tracks} -> {missing_tracks}")
        if len(extra_tracks) == 1 and len(missing_tracks) == 1:
            for inst in lf.instances:
                if inst.track in extra_tracks:
                    # labels.track_set_instance(lf, inst, missing_tracks.pop())
                    old_track = inst.track
                    new_track = missing_tracks.pop()
                    fix_track_map[old_track] = new_track
                    inst.track = new_track
                    # print(f"{lf.frame_idx}: {old_track} -> {new_track} FIXED")
                    break
            frame_tracks = {inst.track for inst in lf.instances}
        else:
            if len(frame_tracks) == instance_count:
                last_good_frame_tracks = frame_tracks

    # Rebuild list of tracks
    labels.tracks = list(
        {
            instance.track
            for frame in labels
            for instance in frame.instances
            if instance.track
        }
    )

    labels.tracks.sort(key=operator.attrgetter("spawned_on", "name"))

    # Save new file
    save_filename = filename
    save_filename = save_filename.replace(".h5", ".cleaned.h5")
    save_filename = save_filename.replace(".json", ".cleaned.json")
    Labels.save_file(labels, save_filename)

    print(f"Saved: {save_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument(
        "-c",
        "--instance_count",
        type=int,
        default=2,
        help="Count of instances to keep in each frame",
    )

    args = parser.parse_args()

    # print(args)

    fit_tracks(filename=args.data_path, instance_count=args.instance_count)

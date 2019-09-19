import os
import re
import h5py as h5
import numpy as np

from sleap.io.dataset import Labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    args = parser.parse_args()

    def video_callback(video_list, new_paths=[os.path.dirname(args.data_path)]):
        # Check each video
        for video_item in video_list:
            if "backend" in video_item and "filename" in video_item["backend"]:
                current_filename = video_item["backend"]["filename"]
                # check if we can find video
                if not os.path.exists(current_filename):
                    is_found = False

                    current_basename = os.path.basename(current_filename)
                    # handle unix, windows, or mixed paths
                    if current_basename.find("/") > -1:
                        current_basename = current_basename.split("/")[-1]
                    if current_basename.find("\\") > -1:
                        current_basename = current_basename.split("\\")[-1]

                    # First see if we can find the file in another directory,
                    # and if not, prompt the user to find the file.

                    # We'll check in the current working directory, and if the user has
                    # already found any missing videos, check in the directory of those.
                    for path_dir in new_paths:
                        check_path = os.path.join(path_dir, current_basename)
                        if os.path.exists(check_path):
                            # we found the file in a different directory
                            video_item["backend"]["filename"] = check_path
                            is_found = True
                            break

    labels = Labels.load_file(args.data_path, video_callback=video_callback)

    frame_count = len(labels)
    track_count = len(labels.tracks)
    track_names = [np.string_(track.name) for track in labels.tracks]
    node_count = len(labels.skeletons[0].nodes)

    frame_idxs = [lf.frame_idx for lf in labels]
    frame_idxs.sort()

    # Desired MATLAB format:
    # "track_occupancy"     tracks * frames
    # "tracks"              frames * nodes * 2 * tracks
    # "track_names"         tracks

    occupancy_matrix = np.zeros((track_count, frame_count), dtype=np.uint8)
    prediction_matrix = np.full((frame_count, node_count, 2, track_count), np.nan, dtype=float)
    
    for lf, inst in [(lf, inst) for lf in labels for inst in lf.instances]:
        frame_i = frame_idxs.index(lf.frame_idx)
        track_i = labels.tracks.index(inst.track)

        occupancy_matrix[track_i, frame_i] = 1

        inst_points = inst.visible_points_array
        prediction_matrix[frame_i, ..., track_i] = inst_points

    occupied_track_mask = np.sum(occupancy_matrix, axis=1) > 0
#     print(track_names[occupied_track_mask])

    # Ignore unoccupied tracks
    if(np.sum(~occupied_track_mask)):
        print(f"ignoring {np.sum(~occupied_track_mask)} empty tracks")
        occupancy_matrix = occupancy_matrix[occupied_track_mask]
        prediction_matrix = prediction_matrix[...,occupied_track_mask]
        track_names = [track_names[i] for i in range(len(track_names)) if occupied_track_mask[i]]

    print(f"track_occupancy: {occupancy_matrix.shape}")
    print(f"tracks: {prediction_matrix.shape}")

    output_filename = re.sub("(\.json(\.zip)?|\.h5)$", "", args.data_path)
    output_filename = output_filename + ".tracking.h5"

    with h5.File(output_filename, "w") as f:
        # We have to transpose the arrays since MATLAB expects column-major
        ds = f.create_dataset("track_names", data=track_names)
        ds = f.create_dataset(
                "track_occupancy", data=np.transpose(occupancy_matrix),
                compression="gzip", compression_opts=9)
        ds = f.create_dataset(
                "tracks", data=np.transpose(prediction_matrix),
                compression="gzip", compression_opts=9)

    print(f"Saved as {output_filename}")
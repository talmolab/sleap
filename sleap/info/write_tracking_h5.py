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

    video_callback = Labels.make_video_callback([os.path.dirname(args.data_path)])
    labels = Labels.load_file(args.data_path, video_callback=video_callback)

    track_count = len(labels.tracks)
    track_names = [np.string_(track.name) for track in labels.tracks]
    node_count = len(labels.skeletons[0].nodes)

    frame_idxs = [lf.frame_idx for lf in labels]
    frame_idxs.sort()
    frame_count = frame_idxs[-1] - frame_idxs[0] + 1 # count should include unlabeled frames

    # Desired MATLAB format:
    # "track_occupancy"     tracks * frames
    # "tracks"              frames * nodes * 2 * tracks
    # "track_names"         tracks

    occupancy_matrix = np.zeros((track_count, frame_count), dtype=np.uint8)
    prediction_matrix = np.full((frame_count, node_count, 2, track_count), np.nan, dtype=float)
    
    for lf, inst in [(lf, inst) for lf in labels for inst in lf.instances]:
        frame_i = lf.frame_idx - frame_idxs[0]
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
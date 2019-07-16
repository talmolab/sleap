import re
import h5py as h5
import numpy as np

from sleap.io.dataset import Labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    args = parser.parse_args()

    labels = Labels.load_json(args.data_path)

    frame_count = len(labels)
    track_count = len(labels.tracks)
    node_count = len(labels.skeletons[0].nodes)

    frame_idxs = [lf.frame_idx for lf in labels]
    frame_idxs.sort()

    # Desired MATLAB format:
    # "track_occupancy"     tracks * frames
    # "tracks"              frames * nodes * 2 * tracks

    occupancy_matrix = np.zeros((track_count, frame_count), dtype=np.uint8)
    prediction_matrix = np.full((frame_count, node_count, 2, track_count), np.nan, dtype=float)
    
    for lf, inst in [(lf, inst) for lf in labels for inst in lf.instances]:
        frame_i = frame_idxs.index(lf.frame_idx)
        track_i = labels.tracks.index(inst.track)

        occupancy_matrix[track_i, frame_i] = 1

        inst_points = inst.points_array(invisible_as_nan=True)
        prediction_matrix[frame_i, ..., track_i] = inst_points
        
    print(f"track_occupancy: {occupancy_matrix.shape}")
    print(f"tracks: {prediction_matrix.shape}")

    output_filename = re.sub("\.json(\.zip)?", "", args.data_path)
    output_filename = output_filename + ".tracking.h5"

    with h5.File(output_filename, "w") as f:
        # We have to transpose the arrays since MATLAB expects column-major
        ds = f.create_dataset("track_occupancy", data=np.transpose(occupancy_matrix))
        ds = f.create_dataset("tracks", data=np.transpose(prediction_matrix))

    print(f"Saved as {output_filename}")
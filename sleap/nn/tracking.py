import sys
import argparse
import multiprocessing
import os

import numpy as np
import cv2

from time import time

from scipy.io import loadmat, savemat

from sleap.nn.inference import find_all_peaks, FlowShiftTracker, get_inference_model
from sleap.nn.paf_inference import match_peaks_paf_par
from sleap.util import usable_cpu_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to video file")
    parser.add_argument("confmap_model_path", help="Path to saved confmap model")
    parser.add_argument("paf_model_path", help="Path to saved PAF model")
    parser.add_argument("skeleton_path", help="Path to skeleton MAT file")
    args = parser.parse_args()

    data_path = args.data_path
    confmap_model_path = args.confmap_model_path
    paf_model_path = args.paf_model_path
    save_path = data_path + ".paf_tracking.mat"
    skeleton_path = args.skeleton_path
    inference_batch_size = 4 # frames per inference batch (GPU memory limited)
    read_chunk_size = 512
    nms_min_thresh = 0.3
    nms_sigma = 3
    min_score_to_node_ratio = 0.2 #0.4
    min_score_midpts = 0.05
    min_score_integral = 0.6 #0.8
    add_last_edge = True # False
    flow_window = 15 # frames
    save_every = 3 # chunks
    params = dict(
        data_path=data_path,
        confmap_model_path=confmap_model_path,
        paf_model_path=paf_model_path,
        save_path=save_path,
        skeleton_path=skeleton_path,
        read_chunk_size=read_chunk_size,
        inference_batch_size=inference_batch_size,
        nms_min_thresh=nms_min_thresh,
        flow_window=flow_window,
        nms_sigma=nms_sigma,
        save_every=save_every,
        )

    # Load skeleton
    skeleton = loadmat(skeleton_path)
    skeleton["nodes"] = skeleton["nodes"][0][0] # convert to scalar
    skeleton["edges"] = skeleton["edges"] - 1 # convert to 0-based indexing
    print("Skeleton (%d nodes):" % skeleton["nodes"])
    print("  %s" % str(skeleton["edges"]))

    # Load model
    model = get_inference_model(confmap_model_path, paf_model_path)
    _, h, w, c = model.input_shape
    model_channels = c
    print("Loaded models:")
    print("  confmap:", confmap_model_path)
    print("  paf:", paf_model_path)
    print("  Input shape: %d x %d x %d" % (h, w, c))

    # Initialize video
    vid = cv2.VideoCapture(data_path)
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    scale = h / vid_h
    print("Opened video:")
    print("  Path:", data_path)
    print("  Frames: %d" % num_frames)
    print("  Frame shape: %d x %d" % (vid_h, vid_w))
    print("  Scale: %f" % scale)

    # Initialize tracking
    tracker = FlowShiftTracker(window=flow_window)

    # Initialize parallel pool
    pool = multiprocessing.Pool(processes=usable_cpu_count())

    # Fix the number of threads for OpenCV
    cv2.setNumThreads(usable_cpu_count())

    # Process chunk-by-chunk!
    t0_start = time()
    matched_instances = []
    match_scores = []
    matched_peak_vals = []
    num_chunks = int(np.ceil(num_frames / read_chunk_size))
    for chunk in range(num_chunks):
        print("Processing chunk %d/%d:" % (chunk+1, num_chunks))
        t0_chunk = time()
        # Calculate how many frames to read
        # num_chunk_frames = min(read_chunk_size, num_frames - int(vid.get(cv2.CAP_PROP_POS_FRAMES)))

        # Read images
        t0 = time()
        mov = []
        while True:
            ret, I = vid.read()

            # No more frames left
            if not ret:
                break

            # Preprocess frame
            if model_channels == 1:
                I = I[:,:,0]
            I = cv2.resize(I, (w, h))
            mov.append(I)

            if len(mov) >= read_chunk_size:
                break

        # Merge and add singleton dimension
        mov = np.stack(mov, axis=0)
        if model_channels == 1:
            mov = mov[...,None]
        else:
            mov = mov[...,::-1]

        print("  Read %d frames [%.1fs]" % (len(mov), time() - t0))

        # Run inference
        t0 = time()
        confmaps, pafs = model.predict(mov.astype("float32")/255, batch_size=inference_batch_size)
        print("  Inferred confmaps and PAFs [%.1fs]" % (time() - t0))

        # Find peaks
        t0 = time()
        peaks, peak_vals = find_all_peaks(confmaps, min_thresh=nms_min_thresh, sigma=nms_sigma)
        print("  Found peaks [%.1fs]" % (time() - t0))

        # Match peaks via PAFs
        t0 = time()
        # instances, scores = match_peaks_paf(peaks, peak_vals, pafs, skeleton)
        instances, scores, peak_vals = match_peaks_paf_par(peaks, peak_vals, pafs, skeleton,
            min_score_to_node_ratio=min_score_to_node_ratio, min_score_midpts=min_score_midpts, min_score_integral=min_score_integral, add_last_edge=add_last_edge, pool=pool)
        print("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

        # # Adjust for input scale
        # for i in range(len(instances)):
        #     for j in range(len(instances[i])):
        #         instances[i][j] = instances[i][j] / scale

        # Track
        t0 = time()
        tracker.track(mov, instances)
        print("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

        # Save
        matched_instances.extend(instances)
        match_scores.extend(scores)
        matched_peak_vals.extend(peak_vals)

        if chunk % save_every == 0 or chunk == (num_chunks-1):
            t0 = time()
            savemat(save_path, dict(params=params, skeleton=skeleton,
                matched_instances=matched_instances, match_scores=match_scores, matched_peak_vals=matched_peak_vals, scale=scale,
                uids=tracker.uids, tracked_instances=tracker.generate_tracks(matched_instances), flow_assignment_costs=tracker.flow_assignment_costs,
                ), do_compression=True)
            print("  Saved to: %s [%.1fs]" % (save_path, time() - t0))

        elapsed = time() - t0_chunk
        total_elapsed = time() - t0_start
        fps = len(matched_instances) / total_elapsed
        frames_left = num_frames - len(matched_instances)
        print("  Finished chunk [%.1fs / %.1f FPS / ETA: %.1f min]" % (elapsed, fps, (frames_left / fps) / 60))

        sys.stdout.flush()

    print("Total: %.1f min" % (total_elapsed / 60))


if __name__ == "__main__":
    main()

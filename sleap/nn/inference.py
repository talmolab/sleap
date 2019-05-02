import sys
import multiprocessing

import numpy as np
import cv2
import keras
import tensorflow as tf

from time import time

from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat, loadmat
from keras.utils import multi_gpu_model


def get_inference_model(confmap_model_path: str, paf_model_path: str) -> keras.Model:
    """ Loads and merges confmap and PAF models into one. """

    # Load
    confmap_model = keras.models.load_model(confmap_model_path)
    paf_model = keras.models.load_model(paf_model_path)

    # Single input
    new_input = confmap_model.input

    # Rename to prevent layer naming conflict
    confmap_model.name = "confmap_" + confmap_model.name
    paf_model.name = "paf_" + paf_model.name
    for i in range(len(confmap_model.layers)):
        confmap_model.layers[i].name = "confmap_" + confmap_model.layers[i].name
    for i in range(len(paf_model.layers)):
        paf_model.layers[i].name = "paf_" + paf_model.layers[i].name

    # Get rid of first layer
    confmap_model.layers.pop(0)
    paf_model.layers.pop(0)

    # Combine models with tuple output
    model = keras.Model(new_input, [confmap_model(new_input), paf_model(new_input)])

    model = multi_gpu_model(model, gpus=4)

    return model

def impeaksnms(I, min_thresh=0.3, sigma=3, return_val=True):
    """ Find peaks via non-maximum suppresion. """

    # Threshold
    if min_thresh is not None:
        I[I < min_thresh] = 0

    # Blur
    if sigma is not None:
        I = gaussian_filter(I, sigma=sigma, mode="constant", cval=0, truncate=8)

    # Maximum filter
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    m = maximum_filter(I, footprint=kernel, mode="constant", cval=0)

    # Convert to points
    r, c = np.nonzero(I > m)
    pts = np.stack((c, r), axis=1)

    # Return
    if return_val:
        vals = np.array([I[pt[1],pt[0]] for pt in pts])
        return pts, vals
    else:
        return pts


def impeaksnms_cv(I, min_thresh=0.3, sigma=3, return_val=True):
    """ Find peaks via non-maximum suppresion using OpenCV. """

    # Threshold
    if min_thresh is not None:
        I[I < min_thresh] = 0

    # Blur
    if sigma is not None:
        I = cv2.GaussianBlur(I, (9,9), sigma)

    # Maximum filter
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]]).astype("uint8")
    m = cv2.dilate(I, kernel)

    # Convert to points
    r, c = np.nonzero(I > m)
    pts = np.stack((c, r), axis=1)

    # Return
    if return_val:
        vals = np.array([I[pt[1],pt[0]] for pt in pts])
        return pts.astype("float32"), vals
    else:
        return pts.astype("float32")

def find_all_peaks(confmaps, min_thresh=0.3, sigma=3):
    """ Finds peaks for all frames/channels in a stack of confidence maps """
    peaks = []
    peak_vals = []
    for confmap in confmaps:
        peaks_i = []
        peak_vals_i = []
        for i in range(confmap.shape[-1]):
            # peak, val = impeaksnms(confmap[...,i], min_thresh=min_thresh, sigma=sigma, return_val=True)
            peak, val = impeaksnms_cv(confmap[...,i], min_thresh=min_thresh, sigma=sigma, return_val=True)
            peaks_i.append(peak)
            peak_vals_i.append(val)
        peaks.append(peaks_i)
        peak_vals.append(peak_vals_i)

    return peaks, peak_vals



class FlowShiftTracker:
    """ Tracks unique individual identities from a set of matched instances in each frame.

    This will track identities by computing the optical flow from each point in each
    reference frame to the current frame and apply this shift to their coordinates. Points
    in the current frame will be matched at the instance-level with previous instances to
    identify the same individual across time.
    """
    def __init__(self, window=10, of_win_size=(21,21), of_max_level=3, of_max_count=30, of_epsilon=0.01):
        """ Initializes configuration and states for tracker.

        Args:
            window: number of previous frames to use as reference when computing flow shift
            of_win_size: window size for optical flow computation (see cv2.calcOpticalFlowPyrLK)
            of_max_level: number of pyramid levels (see cv2.calcOpticalFlowPyrLK)
            of_max_count: max iterations of optimization before stopping (see cv2.calcOpticalFlowPyrLK)
            of_epsilon: minimum shift of search window befor stopping (see cv2.calcOpticalFlowPyrLK)
        """
        self.window = window
        self.of_params = dict(
            winSize=of_win_size,
            maxLevel=of_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, of_max_count, of_epsilon)
        )

        self.nodes = None # gets filled in when we get the first frame
        self.uids = []
        self.flow_assignment_costs = []
        self.frames_refs = []
        self.instances_refs = []
        self.max_uid = -1
        self.t0 = 0

    def track(self, frames, instances):
        """ Track a batch of frames.

        Args:
            frames: images in THWC format
            instances: a list of matched instances
        """

        for t in range(len(instances)):
            # No instance in frame t? -> skip
            if len(instances[t]) == 0:
                self.uids.append([])
                self.flow_assignment_costs.append([])
                continue

            # Initialize instances in the first frame
            if self.max_uid < 0:
                self.uids.append(np.arange(len(instances[t])))
                self.flow_assignment_costs.append(np.nan)
                self.max_uid = max(self.uids[t])

                self.frames_refs.append(frames[t])
                self.instances_refs.append(instances[t])

                continue


            # Initialize node count
            if self.nodes is None:
                self.nodes = instances[t][0].shape[0]

            # Get data and instances at current frame
            frame_t = frames[t].copy()
            instances_t = instances[t]

            # Get reference window data
#             t_refs = np.arange(max(t - self.window, 0), t, dtype=int)
            t_refs = np.arange(start=-1, stop=max(-self.window, -len(self.frames_refs))-1, step=-1, dtype=int)
            frames_refs = [self.frames_refs[t_w] for t_w in t_refs]
            instances_refs = [self.instances_refs[t_w] for t_w in t_refs]
            uids_refs = [self.uids[t_w] for t_w in t_refs]
            cost_refs = []

            # Compute costs for window of ref frames via flow shifting
            for frame_ref, instances_ref in zip(frames_refs, instances_refs):
                # Merge instances into single point set
                pts_ref = np.concatenate(instances_ref, axis=0)

                # Compute optical flow (note: pts_ref must be float32)
                pts_ref_of, status, err = cv2.calcOpticalFlowPyrLK(frame_ref, frame_t, pts_ref.astype("float32"), None, **self.of_params)

                # TODO: NaN out status == 0 points that were not found?

                # Split flow-shifted points into instances
                sections_ref = np.cumsum([len(x) for x in instances_ref])[:-1]
                instances_ref_of = np.split(pts_ref_of, sections_ref, axis=0)

                # Compute cost
                cost_ref = np.zeros((len(instances_ref_of), len(instances_t), self.nodes)) * np.nan
                for i in range(len(instances_ref_of)):
                    for j in range(len(instances_t)):
                        cost_ref[i,j,:] = np.sqrt(np.sum((instances_ref_of[i] - instances_t[j]) ** 2, axis=1))

                cost_refs.append(cost_ref)

            # Aggregate costs per unique instance
            all_uids_ref = np.unique(np.concatenate(uids_refs,axis=0))
            all_costs_ref = []
            for uid_i in all_uids_ref:
                # |ref instances with uid_i| x |instances_t| x nodes
                costs_i = [cost[uid == uid_i] for cost, uid in zip(cost_refs, uids_refs)]
                costs_i = np.concatenate(costs_i, axis=0)

                # Reduce across ref instances and nodes
                costs_i = -np.nansum(1 / (costs_i + 1e-10), axis=(0, 2))

                all_costs_ref.append(costs_i)

            # |instances_t| x |all_uids_ref|
            all_costs_ref = np.stack(all_costs_ref, axis=1)

            # Compute assignments for each instance_t
            row_ind, col_ind = linear_sum_assignment(all_costs_ref)
            flow_assignment_cost = all_costs_ref[row_ind, col_ind].sum()

            # Assign existing unique IDs
            uids_t = np.zeros(len(instances_t)) * np.nan
            for r, c in zip(row_ind, col_ind):
                uids_t[r] = all_uids_ref[c]

            # Create new unique IDs if needed
            for r in np.where(np.isnan(uids_t))[0]:
                uids_t[r] = self.max_uid + 1
                self.max_uid += 1

            # Save
            self.uids.append(uids_t)
            self.flow_assignment_costs.append(flow_assignment_cost)
            self.frames_refs.append(frame_t)
            self.instances_refs.append(instances_t)

            # Keep only window buffer of frames/instances
            while len(self.frames_refs) > self.window:
                self.frames_refs.pop(0)
                self.instances_refs.pop(0)

    def generate_tracks(self, instances):
        """ Generates full coordinates tracks from stored UIDs.

        Args:
            instances: list of matched instances per frame

        Returns:
            tracked_instances: list of tracked instance timeseries
        """
        tracked_instances = []
        all_uids = np.arange(self.max_uid+1)
        for uid in all_uids:
            track = np.zeros((len(self.uids), self.nodes, 2)) * np.nan
            for t in range(len(self.uids)):
                if uid in self.uids[t]:
                    instances_t = np.stack(instances[t], axis=0)
                    instance_uid = instances_t[np.where(self.uids[t] == uid)[0]]
                    track[t] = instance_uid
            tracked_instances.append(track)

        return tracked_instances


if __name__ == "__main__":
    # Params
    # data_path = "W:/rebekah/20181204_102606/processed.h5.mp4"
    data_path = sys.argv[1]
    # confmap_model_path = "models/190131_192028_training.scale=0.50,sigma=10,unet,confmaps_n=42/newest_model.h5"
    confmap_model_path = "models/190203_172137_training.scale=0.50,sigma=10,unet,confmaps_n=81/newest_model.h5"
    # paf_model_path = "models/190131_200654_training.scale=0.50,sigma=10,leap_cnn,pafs_n=42/newest_model.h5"
    paf_model_path = "models/190204_015357_training.scale=0.50,sigma=10,unet,pafs_n=81/newest_model.h5"
    save_path = data_path + ".paf_tracking.mat"
    skeleton_path = "skeleton_thorax_root.mat"
    inference_batch_size = 32 # frames per inference batch (GPU memory limited)
    read_chunk_size = inference_batch_size * 10 # frames to process at a time (CPU memory limited)
    nms_min_thresh = 0.3
    nms_sigma = 3
    save_every = 50
    params = dict(
        data_path=data_path,
        confmap_model_path=confmap_model_path,
        paf_model_path=paf_model_path,
        save_path=save_path,
        skeleton_path=skeleton_path,
        read_chunk_size=read_chunk_size,
        inference_batch_size=inference_batch_size,
        nms_min_thresh=nms_min_thresh,
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
    print("Loaded models:")
    print("  confmap:", confmap_model_path)
    print("  paf:", paf_model_path)
    print("  Input shape: %d x %d x %d" % (h, w, c))

    # Initialize video
    vid = cv2.VideoCapture(data_path)
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = h / vid_h
    print("Opened video:")
    print("  Path:", data_path)
    print("  Frames: %d" % num_frames)
    print("  Frame shape: %d x %d" % (h, w))
    print("  Scale: %f" % scale)

    # Process chunk-by-chunk!
    t0_start = time()
    matched_instances = []
    match_scores = []
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
            I = I[:,:,0]
            I = cv2.resize(I, (h, w))
            mov.append(I)

            if len(mov) >= read_chunk_size:
                break

        # Merge and add singleton dimension
        mov = np.stack(mov, axis=0)
        mov = mov[...,None]
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
        instances, scores = match_peaks_paf(peaks, peak_vals, pafs, skeleton)
        print("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

        # Adjust for input scale
        for i in range(len(instances)):
            for j in range(len(instances[i])):
                instances[i][j] = instances[i][j] / scale

        # Save
        t0 = time()
        matched_instances.extend(instances)
        match_scores.extend(scores)
        if chunk % save_every == 0 or chunk == (num_chunks-1):
            savemat(save_path, dict(params=params, skeleton=skeleton, matched_instances=matched_instances, match_scores=match_scores))
            print("  Saved to: %s [%.1fs]" % (save_path, time() - t0))

        elapsed = time() - t0_chunk
        total_elapsed = time() - t0_start
        fps = len(matched_instances) / total_elapsed
        frames_left = num_frames - len(matched_instances)
        print("  Finished chunk [%.1fs / %.1f FPS/ ETA: %.1f min]" % (elapsed, fps, (frames_left / fps) / 60))

    print("Total: %.1f min" % (total_elapsed / 60))

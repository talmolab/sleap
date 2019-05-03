import sys
import argparse
import multiprocessing
import logging
logger = logging.getLogger(__name__)

import numpy as np
import cv2
import attr
import keras
import h5py as h5

from typing import Union
from time import time

from scipy.io import loadmat, savemat
from scipy.optimize import linear_sum_assignment

from sleap.io.video import Video
from sleap.skeleton import Skeleton
from sleap.nn.inference import find_all_peaks, get_inference_model, match_peaks_paf_par
from sleap.util import usable_cpu_count, save_dict_to_hdf5

# from sleap.instance import Track, Instance, ShiftedInstance, Tracks
from sleap.instance import Track, ShiftedInstance, Tracks
from sleap.instance import InstanceArray as Instance



class FlowShiftTracker():
    def __init__(self, window=10, of_win_size=(21,21), of_max_level=3, of_max_count=30, of_epsilon=0.01, img_scale=1.0, verbosity=0):
        self.window = window
        self.of_params = dict(
            winSize=of_win_size,
            maxLevel=of_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, of_max_count, of_epsilon)
        )
        self.img_scale = np.array(img_scale).astype("float32")
        self.verbosity = verbosity

        self.last_img = None
        self.tracks = Tracks()

    def process(self, imgs, frame_idxs, matched_instances, skeleton):
        """ Flow shift track a batch of frames with matched instances """

        for img_idx, t in enumerate(frame_idxs):
            if len(self.tracks.tracks) == 0:
                for i, pts in enumerate(matched_instances[img_idx]):
                    self.tracks.add_instance(Instance(points=pts / self.img_scale, frame_idx=t, track=Track(spawned_on=t, name=f"{i}")))
                if self.verbosity > 0: print(f"[t = {t}] Created {len(self.tracks.tracks)} initial tracks")
                self.last_img = imgs[img_idx].copy()
                continue
                
            # Get all points in reference frame
            instances_ref = self.tracks.get_frame_instances(t - 1, max_shift=self.window - 1)
            pts_ref = [instance.points for instance in instances_ref]
            if self.verbosity > 0: print(f"[t = {t}] Using {len(instances_ref)} refs back to t = {min([instance.frame_idx for instance in instances_ref] + [instance.source.frame_idx for instance in instances_ref if isinstance(instance, ShiftedInstance)])} ")

            # Flow shift
            pts_fs, status, err = cv2.calcOpticalFlowPyrLK(self.last_img, imgs[img_idx], (np.concatenate(pts_ref, axis=0)).astype("float32") * self.img_scale, None, **self.of_params)
            self.last_img = imgs[img_idx].copy()

            # Split by instance
            sections = np.cumsum([len(x) for x in pts_ref])[:-1]
            pts_fs = np.split(pts_fs / self.img_scale, sections, axis=0)
            status = np.split(status, sections, axis=0)
            err = np.split(err, sections, axis=0)

            # Store shifted instances with metadata
            shifted_instances = [ShiftedInstance(parent=ref, points=pts, frame_idx=t) for ref, pts, found in zip(instances_ref, pts_fs, status) if np.sum(found) > 0]
            self.tracks.add_instances(shifted_instances)

            if len(matched_instances[img_idx]) == 0:
                if self.verbosity > 0: print(f"[t = {t}] No matched instances to assign to tracks")
                continue

            # Reduce distances by track
            unassigned_pts = np.stack(matched_instances[img_idx], axis=0) / self.img_scale # instances x nodes x 2
            shifted_tracks = list({instance.track for instance in shifted_instances})
            if self.verbosity > 0: print(f"[t = {t}] Flow shift matching {len(unassigned_pts)} instances to {len(shifted_tracks)} ref tracks")
            cost_matrix = np.full((len(unassigned_pts), len(shifted_tracks)), np.nan)
            for i, track in enumerate(shifted_tracks):
                # Get shifted points for current track
                track_pts = np.stack([instance.points for instance in shifted_instances if instance.track == track], axis=0) # track_instances x nodes x 2

                # Compute pairwise distances between points
                distances = np.sqrt(np.sum((np.expand_dims(unassigned_pts, axis=1) - np.expand_dims(track_pts, axis=0)) ** 2, axis=-1)) # unassigned_instances x track_instances x nodes

                # Reduce over nodes and instances
                distances = -np.nansum(np.exp(-distances), axis=(1, 2))

                # Save
                cost_matrix[:, i] = distances

            # Hungarian matching
            assigned_ind, track_ind = linear_sum_assignment(cost_matrix)

            # Save assigned instances
            for i, j in zip(assigned_ind, track_ind):
                self.tracks.add_instance(
                    Instance(points=unassigned_pts[i], track=shifted_tracks[j], frame_idx=t)
                )
                if self.verbosity > 0: print(f"[t = {t}] Assigned instance {i} to existing track {shifted_tracks[j].name} (cost = {cost_matrix[i,j]})")

            # Spawn new tracks for unassigned instances
            for i, pts in enumerate(unassigned_pts):
                if i in assigned_ind: continue
                instance = Instance(points=pts, track=Track(spawned_on=t, name=f"{len(self.tracks.tracks)}"), frame_idx=t)
                self.tracks.add_instance(instance)
                if self.verbosity > 0: print(f"[t = {t}] Assigned remaining instance {i} to newly spawned track {instance.track.name} (best cost = {cost_matrix[i,:].min()})")

                
    def occupancy(self):
        """ Compute occupancy matrix """
        num_frames = max(self.tracks.instances.keys()) + 1
        occ = np.zeros((len(self.tracks.tracks), int(num_frames)), dtype="bool")
        for t in range(int(num_frames)):
            instances = self.tracks.get_frame_instances(t)
            instances = [instance for instance in instances if isinstance(instance, Instance)]
            for instance in instances:
                occ[self.tracks.tracks.index(instance.track),t] = True

        return occ

    def generate_tracks(self):
        """ Serializes tracking data into a dict """
        # return attr.asdict(self.tracks) # grr, doesn't work with savemat

        num_tracks = len(self.tracks.tracks)
        num_frames = int(max(self.tracks.instances.keys()) + 1)
        num_nodes = len(self.tracks.instances[0][0].points)

        instance_tracks = np.full((num_frames, num_nodes, 2, num_tracks), np.nan)
        for t in range(num_frames):
            instances = self.tracks.get_frame_instances(t)
            instances = [instance for instance in instances if isinstance(instance, Instance)]

            for instance in instances:
                instance_tracks[t, :, :, self.tracks.tracks.index(instance.track)] = instance.points

        return instance_tracks

    def generate_shifted_data(self):
        """ Generate arrays with all shifted instance data """

        shifted_instances = [y for x in self.tracks.instances.values() for y in x if isinstance(y, ShiftedInstance)]

        track_id = np.array([self.tracks.tracks.index(instance.track) for instance in shifted_instances])
        frame_idx = np.array([instance.frame_idx for instance in shifted_instances])
        frame_idx_source = np.array([instance.source.frame_idx for instance in shifted_instances])
        points = np.stack([instance.points for instance in shifted_instances], axis=0)

        return track_id, frame_idx, frame_idx_source, points



@attr.s(auto_attribs=True)
class Predictor:
    """
    The Predictor class takes a trained sLEAP model and runs
    the complete inference pipeline from confidence map/part affinity field
    inference, non-maximum suppression peak finding, paf part matching, to tracking.

    Args:
        model: A trained keras model used for confidence map and paf inference. FIXME: Should this be a keras model or a sLEAP model class
        skeleton: The skeleton(s) to use for prediction. FIXME. This should be stored with the model I think
        inference_batch_size: Frames per inference batch (GPU memory limited)
        read_chunk_size: How many frames to read into CPU memory at a time (CPU memory limited)
        nms_min_thresh: A threshold of non-max suppression peak finding in confidence maps. All
        values below this minimum threshold will be set to zero before peak finding algorithm
        is run.
        nms_sigma: Gaussian blur is applied to confidence maps before non-max supression peak
        finding occurs. This is the standard deviation of the kernel applied to the image.
        min_score_to_node_ratio: FIXME
        min_score_midpts: FIXME
        min_score_integral: FIXME
        add_last_edge: FIXME

    """

    model: keras.Model = attr.ib()
    skeleton: Skeleton = attr.ib()
    inference_batch_size: int = 4
    read_chunk_size: int = 512
    save_frequency: int = 30 # chunks
    nms_min_thresh = 0.3
    nms_sigma = 3
    min_score_to_node_ratio: float = 0.2
    min_score_midpts: float = 0.05
    min_score_integral: float = 0.6
    add_last_edge: bool = True
    flow_window: int = 15
    save_shifted_instances: bool = True

    def run(self, input_video: Union[str, Video], output_path: str):
        """
        Run the entire inference pipeline on an input video or file object.

        Args:
            input_video: Either a video object or video filename.
            output_path: The output path to save the results.

        Returns:
            None
        """

        # Load model
        _, h, w, c = self.model.input_shape
        model_channels = c
        logger.info("Loaded models:")
        logger.info("  Input shape: %d x %d x %d" % (h, w, c))

        # Open the video if we need it.
        try:
            input_video.get_frame(0)
            vid = input_video
        except AttributeError:
            vid = Video.from_filename(input_video)

        num_frames = vid.num_frames
        vid_h = vid.shape[1]
        vid_w = vid.shape[2]
        scale = h / vid_h
        logger.info("Opened video:")
        logger.info("  Source: " + str(vid.backend))
        logger.info("  Frames: %d" % num_frames)
        logger.info("  Frame shape: %d x %d" % (vid_h, vid_w))
        logger.info("  Scale: %f" % scale)

        # Initialize tracking
        tracker = FlowShiftTracker(window=self.flow_window)

        # Initialize parallel pool
        pool = multiprocessing.Pool(processes=usable_cpu_count())

        # Fix the number of threads for OpenCV, not that we are using
        # anything in OpenCV that is actually multi-threaded but maybe
        # we will down the line.
        cv2.setNumThreads(usable_cpu_count())

        # Process chunk-by-chunk!
        t0_start = time()
        matched_instances = []
        match_scores = []
        matched_peak_vals = []
        num_chunks = int(np.ceil(num_frames / self.read_chunk_size))
        frame_idx = 0
        for chunk in range(num_chunks):
            logger.info("Processing chunk %d/%d:" % (chunk + 1, num_chunks))
            t0_chunk = time()
            # Calculate how many frames to read
            # num_chunk_frames = min(read_chunk_size, num_frames - int(vid.get(cv2.CAP_PROP_POS_FRAMES)))

            # Read the next batch of images
            t0 = time()

            # Read the next chunk of frames
            frame_end = frame_idx + self.read_chunk_size
            if frame_end > vid.num_frames:
                frame_end = vid.num_frames

            mov = vid[frame_idx:frame_end]

            frames_idx = np.arange(frame_idx, frame_idx + len(mov))
            frame_idx += len(mov)

            # Preprocess the frames
            if model_channels == 1:
                mov = mov[:, :, :, 0]

            # Resize the frames to the model input size
            for i in range(mov.shape[0]):
                mov[i, :, :] = cv2.resize(mov[i, :, :], (w, h))

            # Add back singleton dimension
            if model_channels == 1:
                mov = mov[..., None]
            else:
                # TODO: figure out when/if this is necessary for RGB videos
                mov = mov[..., ::-1]

            logger.info("  Read %d frames [%.1fs]" % (len(mov), time() - t0))

            # Run inference
            t0 = time()
            confmaps, pafs = self.model.predict(mov.astype("float32") / 255, batch_size=self.inference_batch_size)
            logger.info("  Inferred confmaps and PAFs [%.1fs]" % (time() - t0))

            # Find peaks
            t0 = time()
            peaks, peak_vals = find_all_peaks(confmaps, min_thresh=self.nms_min_thresh, sigma=self.nms_sigma)
            logger.info("  Found peaks [%.1fs]" % (time() - t0))

            # Match peaks via PAFs
            t0 = time()
            # instances, scores = match_peaks_paf(peaks, peak_vals, pafs, skeleton)
            instances, scores, peak_vals = match_peaks_paf_par(peaks, peak_vals, pafs, self.skeleton,
                                                               min_score_to_node_ratio=self.min_score_to_node_ratio,
                                                               min_score_midpts=self.min_score_midpts,
                                                               min_score_integral=self.min_score_integral,
                                                               add_last_edge=self.add_last_edge, pool=pool)
            logger.info("  Matched peaks via PAFs [%.1fs]" % (time() - t0))

            # Track
            t0 = time()
            tracker.process(mov, frames_idx, instances, self.skeleton)
            logger.info("  Tracked IDs via flow shift [%.1fs]" % (time() - t0))

            # Save
            matched_instances.extend(instances)
            match_scores.extend(scores)
            matched_peak_vals.extend(peak_vals)

            # Get the parameters used for this inference.
            params = attr.asdict(self, filter=lambda attr, value: attr.name not in ["model", "skeleton"])
            print(params)

            if chunk % self.save_frequency == 0 or chunk == (num_chunks - 1):
                t0 = time()

                save_dict = dict(params=params,
                     matched_instances=matched_instances, match_scores=match_scores,
                     matched_peak_vals=matched_peak_vals, scale=scale,
                     uids=tracker.uids, tracked_instances=tracker.generate_tracks(matched_instances),
                    flow_assignment_costs=tracker.flow_assignment_costs)

                if self.save_shifted_instances:
                    shifted_track_id, shifted_frame_idx, shifted_frame_idx_source, shifted_points = tracker.generate_shifted_data()
                    save_dict.update(dict(
                        shifted_track_id=shifted_track_id,
                        shifted_frame_idx=shifted_frame_idx,
                        shifted_frame_idx_source=shifted_frame_idx_source,
                        shifted_points=shifted_points, ))

                with h5.File(output_path, 'w') as f:
                    save_dict_to_hdf5(f, '/', save_dict)

                    # Save the skeleton as well, in JSON to the HDF5
                    self.skeleton.save_hdf5(f)

                logger.info("  Saved to: %s [%.1fs]" % (output_path, time() - t0))

            elapsed = time() - t0_chunk
            total_elapsed = time() - t0_start
            fps = len(matched_instances) / total_elapsed
            frames_left = num_frames - len(matched_instances)
            logger.info("  Finished chunk [%.1fs / %.1f FPS / ETA: %.1f min]" % (elapsed, fps, (frames_left / fps) / 60))

            sys.stdout.flush()

        logger.info("Total: %.1f min" % (total_elapsed / 60))


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
    save_path = data_path + ".paf_tracking.h5"
    skeleton_path = args.skeleton_path

    # Load the model
    model = get_inference_model(confmap_model_path, paf_model_path)

    # Load the skeleton(s)
    skeleton = Skeleton.load_json(skeleton_path)
    logger.info(f"Skeleton (name={skeleton.name}, {len(skeleton.nodes)} nodes):")

    # Create a predictor to do the work.
    predictor = Predictor(model=model, skeleton=skeleton)

    # Run the inference pipeline
    predictor.run(input_video=data_path, output_path=save_path)


if __name__ == "__main__":
    main()

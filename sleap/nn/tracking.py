import functools
import logging

from attr import __init__

logger = logging.getLogger(__name__)

from typing import List, Tuple, Dict, Union

import numpy as np
import h5py as h5
import cv2
import attr

from scipy.optimize import linear_sum_assignment

from sleap.instance import Instance, Track, LabeledFrame
from sleap.io.dataset import Labels


@attr.s(cmp=False, slots=True)
class ShiftedInstance:
    """
    During tracking, optical flow shifted instances are represented to help track instances
    across frames. This class encapsulates an Instance object that has been optical flow
    shifted.

    Args:
        parent: The Instance that this optical flow shifted instance is derived from.
    """
    parent: Union[Instance, 'ShiftedInstance'] = attr.ib()
    frame: Union[LabeledFrame, None] = attr.ib()
    points: np.ndarray = attr.ib()

    @property
    @functools.lru_cache()
    def source(self) -> 'Instance':
        """
        Recursively discover root instance to a chain of flow shifted instances.

        Returns:
            The root InstanceArray of a flow shifted instance.
        """
        if isinstance(self.parent, Instance):
            return self.parent
        else:
            return self.parent.source

    @property
    def track(self) -> Track:
        """
        Get the track object for root flow shifted instance.

        Returns:
            The track object of the root flow shifted instance.
        """
        return self.source.track

    @property
    def frame_idx(self) -> int:
        """
        A convenience method to return the frame index this instance is
        assigned to last.

        Returns:
            The frame index.
        """
        return self.frame.frame_idx

    def get_points_array(self, *args, **kwargs):
        """
        Return the ShiftedInstance as a numpy array. ShiftedInstance stores its
        points as an array always, unlike the Instance class. This method provides
        and identical interface to the points_array method inf Instance.

        Returns:
            The instances points as a Nx2 numpy array. Where N is the number of
            skeleton nodes. Each row represents a point.
        """
        return self.points

@attr.s(slots=True)
class Tracks:
    instances: Dict[int, list] = attr.ib(default=attr.Factory(dict))
    tracks: List[Track] = attr.ib(factory=list)
    last_known_instance: Dict[Track, Instance] = attr.ib(factory=dict)

    def get_frame_instances(self, frame_idx: int, max_shift=None):

        instances = self.instances.get(frame_idx, [])

        # Filter
        if max_shift is not None:
            instances = [instance for instance in instances if isinstance(instance, Instance) or (
                        isinstance(instance, ShiftedInstance) and (
                            (frame_idx - instance.source.frame_idx) <= max_shift))]

        return instances

    def add_instance(self, instance: Union[Instance, 'ShiftedInstance'], frame_index: int):
        frame_instances = self.instances.get(frame_index, [])
        frame_instances.append(instance)
        self.instances[frame_index] = frame_instances
        if instance.track not in self.tracks:
            self.tracks.append(instance.track)

    def add_instances(self, instances: list, frame_index: int):
        for instance in instances:
            self.add_instance(instance, frame_index)

    def get_last_known(self, curr_frame_index: int = None, max_shift: int = None):
        if curr_frame_index is None:
            return list(self.last_known_instance.values())
        else:
            if max_shift is None:
                return [i for i in self.last_known_instance.values()
                        if i.track == curr_frame_index]
            else:
                return [i for i in self.last_known_instance.values()
                        if (curr_frame_index-i.frame_idx) < max_shift]

    def update_track_last_known(self, frame: LabeledFrame, max_shift: int = None):
        for i in frame.instances:
            assert i.track is not None
            self.last_known_instance[i.track] = i

        # Remove tracks from the dict that have exceeded the max_shift horizon
        if max_shift is not None:
            del_tracks = [track
                          for track, instance in self.last_known_instance.items()
                          if (frame.frame_idx-instance.frame_idx) > max_shift]
            for key in del_tracks:
                del self.last_known_instance[key]


@attr.s(auto_attribs=True)
class FlowShiftTracker:
    """
    The FlowShiftTracker class represents and interface to the flow shift
    tracking algorithm. This algorithm allows tracking matched instances
    of animals/objects across multiple frames in a video.

    Args:
        window: The number of frames to look back into for instance id tracking.
        of_win_size: The dimensions in pixels of the window to apply optical flow.
        of_max_level: The number of Gaussian pyramid levels to run optical flow on.
        of_max_count: The maximum amount of iterations for optical flow.
        of_epsilon: The error tolerance for optical flow convergence.
        img_scale: The image scale factor to apply to images.
        verbosity: The verbosity of logging for this module, the higher the level,
        the more informative.
        tracks: A Tracks object that stores tracks (animal instances through time/frames)
    """

    window: int = 10
    of_win_size: Tuple = (21,21)
    of_max_level: int = 3
    of_max_count: int = 30
    of_epsilon: float = 0.01
    img_scale: float = 1.0
    verbosity: int = 0
    tracks: Tracks = attr.ib(default=attr.Factory(Tracks))

    def __attrs_post_init__(self):
        self.tracks = Tracks()
        self.last_img = None
        self.last_frame_index = None

    def _fix_img(self, img: np.ndarray):
        # Drop single channel dimension and convert to uint8 in [0, 255] range
        curr_img = (np.squeeze(img)*255).astype(np.uint8)
        np.clip(curr_img, 0, 255)

        # If we still have 3 dimensions the image is color, need to convert
        # to grayscale for optical flow.
        if len(curr_img.shape) == 3:
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

        return curr_img

    def process(self,
                imgs: np.ndarray,
                labeled_frames: List[LabeledFrame]):
        """
        Flow shift track a batch of frames with matched instances for each frame represented as
        a list of LabeledFrame's.

        Args:
            imgs: A 4D array containing a batch of image frames to run tracking on.
            labeled_frames: A list of labeled frames containing matched instances from
            the inference pipeline. This list must be the same length as imgs. img[i] and
            matched_instances[i] correspond to the same frame.

        Returns:
            None
        """

        # Labels.save_json(Labels(labeled_frames), filename='/tigress/dmturner/sleap/inference_debug/tracking_bug3.json',
        #                  save_frame_data=True)

        # Convert img_scale to an array for use later
        img_scale = np.array(self.img_scale).astype("float32")

        # Set logging to DEBUG if the user has set verbosity > 0
        curr_log_level = logger.getEffectiveLevel()
        if self.verbosity > 0:
            logger.setLevel(logging.DEBUG)

        # Go through each labeled frame and track all the instances
        # present.
        t = self.last_frame_index
        for img_idx, frame in enumerate(labeled_frames):

            # Update the data structures in Tracks that keep the last
            # known instance for each track. Do this for the last frame and
            # skip on the first frame.
            if img_idx > 0:
                self.tracks.update_track_last_known(labeled_frames[img_idx-1], max_shift=None)

            # Copy the actual frame index for this labeled frame, we will
            # use this a lot.
            self.last_frame_index = t
            t = frame.frame_idx

            instances_pts = [i.get_points_array() for i in frame.instances]

            # If we do not have any active tracks, we will spawn one for each
            # matched instance and continue to the next frame.
            if len(self.tracks.tracks) == 0:
                if len(frame.instances) > 0:
                    for i, instance in enumerate(frame.instances):
                        instance.track = Track(spawned_on=t, name=f"{i}")
                        self.tracks.add_instance(instance, frame_index=t)

                    logger.debug(f"[t = {t}] Created {len(self.tracks.tracks)} initial tracks")

                self.last_img = self._fix_img(imgs[img_idx].copy())

                continue

            # Get all points in reference frame
            instances_ref = self.tracks.get_frame_instances(self.last_frame_index, max_shift=self.window - 1)
            pts_ref = [instance.get_points_array() for instance in instances_ref]

            tmp = min([instance.frame_idx for instance in instances_ref] +
                      [instance.source.frame_idx for instance in instances_ref
                       if isinstance(instance, ShiftedInstance)])
            logger.debug(f"[t = {t}] Using {len(instances_ref)} refs back to t = {tmp}")

            curr_img = self._fix_img(imgs[img_idx].copy())

            pts_fs, status, err = \
                cv2.calcOpticalFlowPyrLK(self.last_img, curr_img,
                                         (np.concatenate(pts_ref, axis=0)).astype("float32"),
                                          None, winSize=self.of_win_size,
                                          maxLevel=self.of_max_level,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                    self.of_max_count, self.of_epsilon))
            self.last_img = curr_img

            # Split by instance
            sections = np.cumsum([len(x) for x in pts_ref])[:-1]
            pts_fs = np.split(pts_fs, sections, axis=0)
            status = np.split(status, sections, axis=0)
            status_sum = [np.sum(x) for x in status]
            err = np.split(err, sections, axis=0)

            # Store shifted instances with metadata
            shifted_instances = [ShiftedInstance(parent=ref, points=pts, frame=frame)
                                 for ref, pts, found in zip(instances_ref, pts_fs, status)
                                 if np.sum(found) > 0]

            # Get the track present in the shifted instances
            shifted_tracks = list({instance.track for instance in shifted_instances})

            last_known = self.tracks.get_last_known(curr_frame_index=t, max_shift=self.window)
            alive_tracks = {i.track for i in last_known}

            # If we didn't get any shifted instances from the reference frame, use the last
            # know positions for each track.
            if len(shifted_instances) == 0:
                logger.debug(f"[t = {t}] Optical flow failed, using last known positions for each track.")
                shifted_instances = self.tracks.get_last_known()
                shifted_tracks = list({instance.track for instance in shifted_instances})
            else:
                # We might have got some shifted instances, but make sure we aren't missing any
                # tracks
                for track in alive_tracks:
                    if track in shifted_tracks:
                        continue
                    shifted_tracks.append(track)
                    shifted_instances.append(self.tracks.last_known_instance[track])

            self.tracks.add_instances(shifted_instances, frame_index=t)

            if len(frame.instances) == 0:
                logger.debug(f"[t = {t}] No matched instances to assign to tracks")
                continue

            # Reduce distances by track
            unassigned_pts = np.stack(instances_pts, axis=0) # instances x nodes x 2
            logger.debug(f"[t = {t}] Flow shift matching {len(unassigned_pts)} "
                         f"instances to {len(shifted_tracks)} ref tracks")

            cost_matrix = np.full((len(unassigned_pts), len(shifted_tracks)), np.nan)
            for i, track in enumerate(shifted_tracks):
                # Get shifted points for current track
                track_pts = np.stack([instance.get_points_array()
                                      for instance in shifted_instances
                                      if instance.track == track], axis=0) # track_instances x nodes x 2

                # Compute pairwise distances between points
                distances = np.sqrt(np.sum((np.expand_dims(unassigned_pts / img_scale, axis=1) -
                                            np.expand_dims(track_pts, axis=0)) ** 2,
                                           axis=-1)) # unassigned_instances x track_instances x nodes

                # Reduce over nodes and instances
                distances = -np.nansum(np.exp(-distances), axis=(1, 2))

                # Save
                cost_matrix[:, i] = distances

            # Hungarian matching
            assigned_ind, track_ind = linear_sum_assignment(cost_matrix)

            # Save assigned instances
            for i, j in zip(assigned_ind, track_ind):
                frame.instances[i].track = shifted_tracks[j]
                self.tracks.add_instance(frame.instances[i], frame_index=t)

                logger.debug(f"[t = {t}] Assigned instance {i} to existing track "
                             f"{shifted_tracks[j].name} (cost = {cost_matrix[i,j]})")

            # Spawn new tracks for unassigned instances
            for i, pts in enumerate(unassigned_pts):
                if i in assigned_ind: continue
                instance = frame.instances[i]
                instance.track = Track(spawned_on=t, name=f"{len(self.tracks.tracks)}")
                self.tracks.add_instance(instance, frame_index=t)
                logger.debug(f"[t = {t}] Assigned remaining instance {i} to newly "
                             f"spawned track {instance.track.name} "
                             f"(best cost = {cost_matrix[i,:].min()})")

        # Update the last know data structures for the last frame.
        self.tracks.update_track_last_known(labeled_frames[img_idx - 1], max_shift=None)

        # Reset the logging level
        logger.setLevel(curr_log_level)

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

        shifted_instances = [y for x in self.tracks.instances.values()
                             for y in x if isinstance(y, ShiftedInstance)]

        track_id = np.array([self.tracks.tracks.index(instance.track) for instance in shifted_instances])
        frame_idx = np.array([instance.frame_idx for instance in shifted_instances])
        frame_idx_source = np.array([instance.source.frame_idx for instance in shifted_instances])
        points = np.stack([instance.points for instance in shifted_instances], axis=0)

        return track_id, frame_idx, frame_idx_source, points

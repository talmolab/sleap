import functools
import logging

from attr import __init__

logger = logging.getLogger(__name__)

from typing import List, Tuple, Dict, Union

import numpy as np
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

    def points_array(self, *args, **kwargs):
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

    def get_frame_instances(self, frame_idx: int, max_shift=None):

        instances = self.instances.get(frame_idx, [])

        # Filter
        if max_shift is not None:
            instances = [instance for instance in instances if isinstance(instance, Instance) or (
                        isinstance(instance, ShiftedInstance) and (
                            (frame_idx - instance.source.frame_idx) <= max_shift))]

        return instances

    def add_instance(self, instance: Union[Instance, 'ShiftedInstance']):
        frame_instances = self.instances.get(instance.frame_idx, [])
        frame_instances.append(instance)
        self.instances[instance.frame_idx] = frame_instances
        if instance.track not in self.tracks:
            self.tracks.append(instance.track)

    def add_instances(self, instances: list):
        for instance in instances:
            self.add_instance(instance)


@attr.s(auto_attribs=True)
class FlowShiftTracker:
    """
    The FlowShiftTracker class represents and interface to the flow shift
    tracking algorithm. This algorithm allows tracking matched instances
    of animals/objects across multiple frames in a video.

    Args:
        window: TODO
        of_win_size: TODO
        of_max_level: TODO
        of_max_count: TODO
        of_epsilon: TODO
        img_scale: TODO
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

        # Convert img_scale to an array for use later
        img_scale = np.array(self.img_scale).astype("float32")

        # Go through each labeled frame and track all the instances
        # present.
        for img_idx, frame in enumerate(labeled_frames):

            # Copy the actual frame index for this labeled frame, we will
            # use this a lot.
            t = frame.frame_idx

            instances_pts = [i.points_array(cached=True) for i in frame.instances]

            # If we do not have any active tracks, we will spawn one for each
            # matched instance and continue to the next frame.
            if len(self.tracks.tracks) == 0:
                for i, instance in enumerate(frame.instances):
                    instance.track = Track(spawned_on=t, name=f"{i}")
                    self.tracks.add_instance(instance)

                if self.verbosity > 0:
                    logger.info(f"[t = {t}] Created {len(self.tracks.tracks)} initial tracks")
                self.last_img = np.squeeze(imgs[img_idx].copy())

                # If we still have 3 dimensions the image is color, need to convert
                # to grayscale for optical flow.
                if len(self.last_img.shape) == 3:
                    self.last_img = cv2.cvtColor(self.last_img, cv2.COLOR_RGB2GRAY)

                continue

            # Get all points in reference frame
            instances_ref = self.tracks.get_frame_instances(t - 1, max_shift=self.window - 1)
            pts_ref = [instance.points_array(cached=True) for instance in instances_ref]
            if self.verbosity > 0:
                tmp = min([instance.frame_idx for instance in instances_ref] +
                          [instance.source.frame_idx for instance in instances_ref
                           if isinstance(instance, ShiftedInstance)])
                logger.info(f"[t = {t}] Using {len(instances_ref)} refs back to t = {tmp}")

            curr_img = np.squeeze(imgs[img_idx].copy())

            # If we still have 3 dimensions the image is color, need to convert
            # to grayscale for optical flow.
            if len(curr_img.shape) == 3:
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

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
            err = np.split(err, sections, axis=0)

            # Store shifted instances with metadata
            shifted_instances = [ShiftedInstance(parent=ref, points=pts, frame=frame)
                                 for ref, pts, found in zip(instances_ref, pts_fs, status)
                                 if np.sum(found) > 0]
            self.tracks.add_instances(shifted_instances)

            if len(frame.instances) == 0:
                if self.verbosity > 0:
                    logger.info(f"[t = {t}] No matched instances to assign to tracks")
                continue

            # Reduce distances by track
            unassigned_pts = np.stack(instances_pts, axis=0) # instances x nodes x 2
            shifted_tracks = list({instance.track for instance in shifted_instances})
            if self.verbosity > 0:
                logger.info(f"[t = {t}] Flow shift matching {len(unassigned_pts)} "
                             f"instances to {len(shifted_tracks)} ref tracks")

            cost_matrix = np.full((len(unassigned_pts), len(shifted_tracks)), np.nan)
            for i, track in enumerate(shifted_tracks):
                # Get shifted points for current track
                track_pts = np.stack([instance.points_array(cached=True)
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
                self.tracks.add_instance(frame.instances[i])

                if self.verbosity > 0:
                    logger.info(f"[t = {t}] Assigned instance {i} to existing track "
                                 f"{shifted_tracks[j].name} (cost = {cost_matrix[i,j]})")

            # Spawn new tracks for unassigned instances
            for i, pts in enumerate(unassigned_pts):
                if i in assigned_ind: continue
                instance = frame.instances[i]
                instance.track = Track(spawned_on=t, name=f"{len(self.tracks.tracks)}")
                self.tracks.add_instance(instance)
                if self.verbosity > 0:
                    logger.info(f"[t = {t}] Assigned remaining instance {i} to newly "
                                 f"spawned track {instance.track.name} "
                                 f"(best cost = {cost_matrix[i,:].min()})")

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

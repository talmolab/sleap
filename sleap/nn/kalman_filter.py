from collections import defaultdict
from typing import List

import attr
import numpy as np
import pykalman
from numpy import ma
from pykalman import KalmanFilter

from sleap.instance import Track


@attr.s(auto_attribs=True)
class TrackKalman:
    kalman_filters: List[pykalman.KalmanFilter]
    state_means: List[list]
    state_covariances: List[list]
    tracks: List[str]
    frame_tracking: List[list]
    instance_count: int
    node_indices: List[int]  # indices of rows for points to use

    @classmethod
    def initialize(cls, frames: List["LabeledFrame"], instance_count: int, node_indices: List[int]) -> "TrackKalman":
        frame_array_dict = defaultdict(list)

        kalman_filter_list = []
        state_means_list = []
        state_covariances_list = []
        track_list = []
        frame_tracking_list = []

        instances = [inst for lf in frames for inst in lf.instances]

        if not instances:
            raise ValueError("Kalman filter must be initialized with instances.")

        # TODO: make arg optional and use algorithm to find best nodes to track

        for inst in instances:
            point_coords = inst.points_array[node_indices, 0:2].flatten()
            frame_array_dict[inst.track.name].append(point_coords)

        for track_name, frame_array in frame_array_dict.items():

            frame_array = ma.asarray(frame_array)
            frame_array = ma.masked_invalid(frame_array)

            initial_frame = frame_array[0]
            initial_frame_size = initial_frame.size
            initial_state_means = [0] * (initial_frame_size * 2)

            for coord_idx, coord_value in enumerate(initial_frame.flatten()):
                initial_state_means[(coord_idx * 2)] = coord_value

            transition_matrix = []

            for coord_idx in range(0, initial_frame_size):
                transition_matrix.append(
                    [
                        int(x in [(coord_idx * 2), (coord_idx * 2) + 1])
                        for x in range(initial_frame_size * 2)
                    ]
                )

                transition_matrix.append(
                    [
                        int(x == ((coord_idx * 2) + 1))
                        for x in range(initial_frame_size * 2)
                    ]
                )

            observation_matrix = []

            for coord_idx in range(0, initial_frame_size):
                observation_matrix.append(
                    [int(x == (coord_idx * 2)) for x in range(initial_frame_size * 2)]
                )

            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_means,
            )

            kf = kf.em(frame_array, n_iter=20)

            state_means, state_covariances = kf.filter(frame_array)

            kalman_filter_list.append(kf)
            state_means_list.append(list(state_means))
            state_covariances_list.append(list(state_covariances))
            track_list.append(track_name)
            frame_tracking_list.append([track_name] * len(frames))

        return cls(
            kalman_filters=kalman_filter_list,
            state_means=state_means_list,
            state_covariances=state_covariances_list,
            tracks=track_list,
            frame_tracking=frame_tracking_list,
            instance_count=instance_count,
            node_indices=node_indices,
        )

    def track_frames(self, frames):
        pa_row_idxs = self.node_indices

        for lf in frames:
            smallest_distance_mean = [None] * len(self.kalman_filters)
            smallest_distance_track = [None] * len(self.kalman_filters)
            smallest_distance_points_array = [None] * len(self.kalman_filters)

            def set_smallest(idx, dist_mean, instance):
                points = instance.points_array[pa_row_idxs, 0:2]  # x, y

                smallest_distance_mean[idx] = dist_mean
                smallest_distance_track[idx] = instance.track.name
                smallest_distance_points_array[idx] = points.flatten()

            def clear_smallest(idx, clear_mean=True):
                if clear_mean:
                    smallest_distance_mean[idx] = None
                smallest_distance_points_array[idx] = ma.masked
                smallest_distance_track[idx] = None

            # Update each Kalman filter, one per tracked identity
            for kalman_idx, kalman_filter in enumerate(self.kalman_filters):

                exp_mean, exp_covariance = kalman_filter.filter_update(
                    self.state_means[kalman_idx][-1],
                    self.state_covariances[kalman_idx][-1],
                    ma.masked,
                )

                exp_coord_means = np.array(exp_mean[::2])

                for inst in lf.instances:
                    if inst.score <= 0.30:
                        if not smallest_distance_track[kalman_idx]:
                            smallest_distance_points_array[kalman_idx] = ma.masked
                        continue

                    point_array = inst.get_points_array(
                        copy=True, invisible_as_nan=True, full=True
                    )

                    # row of full points array: x, y, visible, complete, score
                    inst_points = point_array[pa_row_idxs, 0:2].flatten()
                    weights = point_array[pa_row_idxs, 4].flatten().repeat(2)
                    distances = abs(exp_coord_means - inst_points)

                    if all(np.isnan(distances)):
                        if not smallest_distance_track[kalman_idx]:
                            smallest_distance_points_array[kalman_idx] = ma.masked
                        continue

                    distances = ma.MaskedArray(distances, mask=np.isnan(distances))
                    current_distance_mean = ma.average(distances, weights=weights)

                    if smallest_distance_mean[kalman_idx] is None or current_distance_mean < smallest_distance_mean[kalman_idx]:
                        set_smallest(kalman_idx, current_distance_mean, inst)

            # single instance and (...?)
            if (
                len(lf.instances) == 1
                and smallest_distance_track[0]
                and smallest_distance_track[0] == smallest_distance_track[1]
            ):

                inst_difference = abs(smallest_distance_mean[0] - smallest_distance_mean[1])
                inst_function = min(smallest_distance_mean)

                if (inst_difference / inst_function) <= 1:
                    # clear all
                    for i in range(len(smallest_distance_track)):
                        clear_smallest(i, clear_mean=False)

            # one distinct track and (...?)
            if len(set(smallest_distance_track)) == 1 and smallest_distance_track[0]:

                if smallest_distance_mean[0] < smallest_distance_mean[1]:
                    # one instance
                    if len(lf.instances) == 1:
                        clear_smallest(1)

                    # two instances
                    elif len(lf.instances) == 2:
                        for inst in lf.instances:
                            if inst.track.name != smallest_distance_track[1]:
                                set_smallest(1, dist_mean=None, instance=inst)
                                break

                    # more than two instances
                    else:
                        clear_smallest(1)

                else:
                    # one instance in frame
                    if len(lf.instances) == 1:
                        clear_smallest(0)

                    # two instances
                    elif len(lf.instances) == 2:
                        for inst in lf.instances:
                            if inst.track.name != smallest_distance_track[0]:
                                set_smallest(0, dist_mean=None, instance=inst)
                                break

                    # more than two instances
                    else:
                        clear_smallest(0)

            # distinct tracks
            if (
                smallest_distance_track[0] != smallest_distance_track[1]
                and smallest_distance_track[0]
                and smallest_distance_track[1]
                and smallest_distance_mean[0]
                and smallest_distance_mean[1]
            ):

                point_difference = abs(smallest_distance_points_array[0] - smallest_distance_points_array[1])

                if not all(np.isnan(point_difference)):
                    point_difference_mean = np.nanmean(point_difference)

                    is_diff_smaller = all((point_difference_mean < dist for dist in smallest_distance_mean))
                    if is_diff_smaller:

                        # TODO: find id with smallest mean, clear the others?

                        # id 0 has larger mean
                        if smallest_distance_mean[0] > smallest_distance_mean[1]:
                            clear_smallest(0, clear_mean=False)

                        # id 0 has larger mean
                        elif smallest_distance_mean[1] > smallest_distance_mean[0]:
                            clear_smallest(1, clear_mean=False)

            # Now do the actual track assignments (?)
            for smallest_idx in range(len(smallest_distance_track)):
                if ma.is_masked(smallest_distance_points_array[smallest_idx]):
                    last_mean = self.state_means[smallest_idx][-1]
                    last_covariance = self.state_covariances[smallest_idx][-1]
                    last_track = None

                else:
                    points_array = ma.asarray(smallest_distance_points_array[smallest_idx])
                    points_array = ma.masked_invalid(points_array)

                    last_mean, last_covariance = self.kalman_filters[smallest_idx].filter_update(
                        self.state_means[smallest_idx][-1],
                        self.state_covariances[smallest_idx][-1],
                        points_array,
                    )
                    last_track = smallest_distance_track[smallest_idx]

                self.state_means[smallest_idx].append(last_mean)
                self.state_covariances[smallest_idx].append(last_covariance)
                self.frame_tracking[smallest_idx].append(last_track)

    def get_tracking_array(self):
        return np.asarray(self.frame_tracking)


def filter_frames(
    frames: List["LabeledFrame"],
    instance_count: int,
    node_indices: List[int],
    keep_non_tracked: bool = False,
    init_len: int = 10,
):
    """
    Attempts to track N instances using a Kalman Filter.

    Args:
        frames: The list of `LabeldFrame` objects with predictions.
        instance_count: The number of expected instances per frame.
        node_indices: Indices of nodes to use for tracking.
        keep_non_tracked: Bool if non-tracked frames should be kept. False
            by default.
        init_len: The number of frames that should be used to initialize 
            the Kalman filter.

    Returns:
        None; modifies frames in place.
    """

    # Initialize the filter
    kalman_filter = TrackKalman.initialize(frames[:init_len], instance_count, node_indices)

    # Run the filter, frame by frame
    kalman_filter.track_frames(frames[init_len:])

    # Assign the tracking array
    tracking_array = kalman_filter.get_tracking_array()

    # Create list to store the initial tracks
    initial_tracks = [None] * tracking_array.shape[0]

    # Loop the frames
    for frame_idx, lf in enumerate(frames):

        # Assign the predicted tracks
        predicted_track_names = tracking_array[:, frame_idx]

        # Check if this is the first frame
        if frame_idx == 0:

            # Loop the frame tracking data
            for track_idx in range(len(predicted_track_names)):

                # Assign the initial track name
                initial_track_name = predicted_track_names[track_idx]

                # Loop the instances
                for inst in lf.instances:

                    # Check if current instance is the correct initial track
                    if inst.track.name == initial_track_name:
                        initial_tracks[track_idx] = inst.track

        # Create list of instances that were not tracked
        non_tracked_instances = []

        for inst in lf.instances:
            # Check if current instances is on the predicted track
            if inst.track.name not in predicted_track_names:
                non_tracked_instances.append(inst)

        # Clear track if we want to keep non-tracked instances
        if keep_non_tracked:
            for inst in non_tracked_instances:
                inst.track = None

        # Otherwise remove the non-tracked instances
        else:
            for non_tracked_instance in non_tracked_instances:
                lf.instances.remove(non_tracked_instance)

        # Update tracks by matching on name (FIXME)
        for inst in lf.instances:

            # Loop the frame tracking data
            for initial_track, predicted_track_name in zip(
                initial_tracks, predicted_track_names
            ):

                # Check if current instances is on the predicted track
                if inst.track and inst.track.name == predicted_track_name:
                    # Update the track assignment
                    inst.track = initial_track

                    # Break if assigned, to avoid another assignment
                    break

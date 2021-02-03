"""
Adaptor for reading DeepPoseKit datasets (HDF5).
"""

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle

from sleap.instance import Instance, LabeledFrame, Point, Track

from sleap import Labels, Video, Skeleton

import numpy as np
import pandas as pd


class LabelsDeepPoseKitAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "h5"

    @property
    def all_exts(self):
        return ["h5", "hdf5"]

    @property
    def name(self):
        return "DeepPoseKit Dataset HDF5"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_hdf5:
            return False
        if not hasattr(file.file, "pose"):
            return False
        return True

    def can_write_filename(self, filename: str):
        return False

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return False

    @classmethod
    def read(
        cls,
        file: FileHandle,
        video_path: str,
        skeleton_path: str,
        *args,
        **kwargs,
    ) -> Labels:
        f = file.file

        video = Video.from_filename(video_path)
        skeleton_data = pd.read_csv(skeleton_path, header=0)

        skeleton = Skeleton()
        skeleton.add_nodes(skeleton_data["name"])
        nodes = skeleton.nodes

        for name, parent, swap in skeleton_data.itertuples(index=False, name=None):
            if parent is not np.nan:
                skeleton.add_edge(parent, name)

        lfs = []

        pose_matrix = f["pose"][:]

        track_count, frame_count, node_count, _ = pose_matrix.shape

        tracks = [Track(0, f"Track {i}") for i in range(track_count)]
        for frame_idx in range(frame_count):
            lf_instances = []
            for track_idx in range(track_count):
                points_array = pose_matrix[track_idx, frame_idx, :, :]
                points = dict()
                for p in range(len(points_array)):
                    x, y, score = points_array[p]
                    points[nodes[p]] = Point(x, y)  # TODO: score

                inst = Instance(
                    skeleton=skeleton, track=tracks[track_idx], points=points
                )
                lf_instances.append(inst)
            lfs.append(LabeledFrame(video, frame_idx=frame_idx, instances=lf_instances))

        return Labels(labeled_frames=lfs)

import numpy as np

from typing import Union

from sleap import Labels, Video, Skeleton
from sleap.instance import PredictedInstance, LabeledFrame, Track

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class SleapAnalysisAdaptor(Adaptor):
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
        return "SLEAP Analysis HDF5"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_hdf5:
            return False
        if "track_occupancy" not in file.file:
            return False
        return True

    def can_write_filename(self, filename: str):
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    @classmethod
    def read(
        cls, file: FileHandle, video: Union[Video, str], *args, **kwargs,
    ) -> Labels:
        connect_adj_nodes = False

        if video is None:
            raise ValueError("Cannot read analysis hdf5 if no video specified.")

        if not isinstance(video, Video):
            video = Video.from_filename(video)

        f = file.file
        tracks_matrix = f["tracks"][:].T
        track_names_list = f["track_names"][:].T
        node_names_list = f["node_names"][:].T

        # shape: frames * nodes * 2 * tracks
        frame_count, node_count, _, track_count = tracks_matrix.shape

        tracks = [Track(0, track_name.decode()) for track_name in track_names_list]

        skeleton = Skeleton()
        last_node_name = None
        for node_name in node_names_list:
            node_name = node_name.decode()
            skeleton.add_node(node_name)
            if connect_adj_nodes and last_node_name:
                skeleton.add_edge(last_node_name, node_name)
            last_node_name = node_name

        frames = []
        for frame_idx in range(frame_count):
            instances = []
            for track_idx in range(track_count):
                points = tracks_matrix[frame_idx, ..., track_idx]
                if not np.all(np.isnan(points)):
                    point_scores = np.ones(len(points))
                    # make everything a PredictedInstance since the usual use
                    # case is to export predictions for analysis
                    instances.append(
                        PredictedInstance.from_arrays(
                            points=points,
                            point_confidences=point_scores,
                            skeleton=skeleton,
                            track=tracks[track_idx],
                            instance_score=1,
                        )
                    )
            if instances:
                frames.append(
                    LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
                )

        return Labels(labeled_frames=frames)

    @classmethod
    def write(cls, filename: str, source_object: Labels):
        from sleap.info.write_tracking_h5 import main as write_analysis

        write_analysis(source_object, output_path=filename, all_frames=True)

"""
Adaptor to read (not write) LEAP MATLAB data files.

This attempts to find videos. If they cannot automatically be found and the
`gui` param is True, then the user will be prompted to find the videos.
"""
import os

import scipy.io as sio

from sleap import Labels, Video, Skeleton
from sleap.gui.dialogs.missingfiles import MissingFilesDialog
from sleap.instance import (
    Instance,
    LabeledFrame,
    Point,
)
from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class LabelsLeapMatlabAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "mat"

    @property
    def all_exts(self):
        return ["mat"]

    @property
    def name(self):
        return "LEAP Matlab dataset"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        # if "boxPath" not in file.file:
        #     return False
        return True

    def can_write_filename(self, filename: str):
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return False

    @classmethod
    def read(
        cls,
        file: FileHandle,
        gui: bool = True,
        *args,
        **kwargs,
    ):
        filename = file.filename

        mat_contents = sio.loadmat(filename)

        box_path = cls._unwrap_mat_scalar(mat_contents["boxPath"])

        # If the video file isn't found, try in the same dir as the mat file
        if not os.path.exists(box_path):
            file_dir = os.path.dirname(filename)
            box_path_name = box_path.split("\\")[-1]  # assume windows path
            box_path = os.path.join(file_dir, box_path_name)

        if not os.path.exists(box_path):
            if gui:
                video_paths = [box_path]
                missing = [True]
                okay = MissingFilesDialog(video_paths, missing).exec_()

                if not okay or missing[0]:
                    return

                box_path = video_paths[0]
            else:
                # Ignore missing videos if not loading from gui
                box_path = ""

        if os.path.exists(box_path):
            vid = Video.from_hdf5(
                dataset="box", filename=box_path, input_format="channels_first"
            )
        else:
            vid = None

        nodes_ = mat_contents["skeleton"]["nodes"]
        edges_ = mat_contents["skeleton"]["edges"]
        points_ = mat_contents["positions"]

        edges_ = edges_ - 1  # convert matlab 1-indexing to python 0-indexing

        nodes = cls._unwrap_mat_array(nodes_)
        edges = cls._unwrap_mat_array(edges_)

        nodes = list(map(str, nodes))  # convert np._str to str

        sk = Skeleton(name=filename)
        sk.add_nodes(nodes)
        for edge in edges:
            sk.add_edge(source=nodes[edge[0]], destination=nodes[edge[1]])

        labeled_frames = []
        node_count, _, frame_count = points_.shape

        for i in range(frame_count):
            new_inst = Instance(skeleton=sk)
            for node_idx, node in enumerate(nodes):
                x = points_[node_idx][0][i]
                y = points_[node_idx][1][i]
                new_inst[node] = Point(x, y)
            if len(new_inst.points):
                new_frame = LabeledFrame(video=vid, frame_idx=i)
                new_frame.instances = (new_inst,)
                labeled_frames.append(new_frame)

        labels = Labels(labeled_frames=labeled_frames, videos=[vid], skeletons=[sk])

        return labels

    @classmethod
    def _unwrap_mat_scalar(cls, a):
        """Extract single value from nested MATLAB file data."""
        if a.shape == (1,):
            return cls._unwrap_mat_scalar(a[0])
        else:
            return a

    @classmethod
    def _unwrap_mat_array(cls, a):
        """Extract list of values from nested MATLAB file data."""
        b = a[0][0]
        c = [cls._unwrap_mat_scalar(x) for x in b]
        return c

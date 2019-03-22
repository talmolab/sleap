""" Temporary classes while full implementations are not working """

import os
import numpy as np
import pandas as pd
import h5py

from sleap.skeleton import Skeleton
from sleap.io.video import HDF5Video
from sleap.instance import Point, Instance
from sleap.io.dataset import Dataset

class GeneratedLabels():
    def __init__(self, file_path, imgs_dset="/box", matlab_indexing=True):
        scale = 1.0
        with h5py.File(file_path, "r") as f:
            # Auto-detect scale
            if "scale" in f[imgs_dset].attrs:
                scale = float(f[imgs_dset].attrs["scale"])

            # Load skeleton
            # skeleton = Skeleton.load_hdf5(f["skeleton"])
            node_names = f["skeleton"].attrs["nodeNames"].decode().split()
            edges = f["skeleton"]["edges"][:].T.astype("uint8")
            if matlab_indexing:
                edges -= 1

            # Create class
            skeleton = Skeleton()
            skeleton.add_nodes(node_names)
            for (src, dst) in edges:
                skeleton.add_edge(node_names[src], node_names[dst])
            
            # Load frames and points
            frames = {k: f["frames"][k][:].flatten() for k in ["videoId", "frameIdx"]}
            points = {k: f["points"][k][:].flatten() for k in ["id", "frameIdx", "instanceId", "x", "y", "node", "visible"]}

            # Adjust points
            if matlab_indexing:
                points["x"] -= 1
                points["y"] -= 1
                points["node"] = points["node"].astype("uint8") - 1
            points["x"] *= scale
            points["y"] *= scale
            points["visible"] = points["visible"].astype("bool")

        self.file_path = file_path
        self.imgs_dset = imgs_dset
        self.video = HDF5Video(file_path, imgs_dset, input_format="channels_first")
        self.scale = scale
        self.matlab_indexing = matlab_indexing
        self.skeleton = skeleton
        self.frames = pd.DataFrame(frames)
        self.points = pd.DataFrame(points)

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        """ Informal string representation (for print or format) """
        return f"{type(self).__name__} (n = {len(self)})"

    def __repr__(self):
        """ Formal string representation (Python code-like) """
        return str(self)

    def get_frame_instances(self, idx):
        is_in_frame = self.points["frameIdx"] == self.frames["frameIdx"][idx]
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(self.points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (self.points["instanceId"] == instance_id)
            instance_points = {self.skeleton.node_names[n]: Point(x, y, visible=v) for x, y, n, v in
                                            zip(*[self.points[k][is_instance] for k in ["x", "y", "node", "visible"]])}

            instance = Instance(skeleton=self.skeleton, video=self.video, frame_idx=idx, points=instance_points)
            instances.append(instance)

        return instances

if __name__ == "__main__":
    data_path = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    labels = Dataset(data_path)

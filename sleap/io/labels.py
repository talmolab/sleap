""" Temporary classes while full implementations are not working """

import os
import numpy as np
import pandas as pd
import h5py
import json
# from scipy.io import loadmat, savemat

from sleap.skeleton import Skeleton
from sleap.io.video import HDF5Video
from sleap.instance import Point, Instance

class Labels():
    def __init__(self, data_path=None, skeleton=None):
        if data_path is None:
            self.empty_dataset()
        elif isinstance(data_path, str):
            if data_path.endswith(".json"):
                self.load_json(data_path)
            elif data_path.endswith(".h5"):
                pass


    def empty_dataset(self):
        self.videos = pd.DataFrame(columns=['id','filepath','format','dataset','width','height','channels','frames','dtype'])
        self.instances = pd.DataFrame(columns=['id','videoId','frameIdx','complete','trackId'])
        self.points = pd.DataFrame(columns=['id','videoId','frameIdx','instanceId','x','y','node','visible'])
        self.predicted_instances = pd.DataFrame(columns=['id','videoId','frameIdx','trackId','matching_score','tracking_score'])
        self.predicted_points = pd.DataFrame(columns=['id','videoId','frameIdx','instanceId','x','y','node','visible','confidence'])
        
    def load_json(self, data_path, adjust_matlab_indexing=True):
        data = json.loads(open(data_path).read())

        self.videos = pd.DataFrame(data["videos"])
        self.instances = pd.DataFrame(data["instances"])
        self.points = pd.DataFrame(data["points"])
        self.predicted_instances = pd.DataFrame(data["predicted_instances"])
        self.predicted_points = pd.DataFrame(data["predicted_points"])

        if adjust_matlab_indexing:
            self.instances.frameIdx -= 1
            self.points.frameIdx -= 1
            self.predicted_instances.frameIdx -= 1
            self.predicted_points.frameIdx -= 1

            self.points.node -= 1
            self.predicted_points.node -= 1

            self.points.x -= 1
            self.predicted_points.x -= 1

            self.points.y -= 1
            self.predicted_points.y -= 1

        self.skeleton = Skeleton()
        self.skeleton.add_nodes(data["skeleton"]["nodeNames"])
        edges = data["skeleton"]["edges"]
        if adjust_matlab_indexing:
            edges = np.array(edges) - 1
        for (src, dst) in edges:
                self.skeleton.add_edge(self.skeleton.node_names[src], self.skeleton.node_names[dst])


    def __len__(self):
        return self.instances.groupby(["videoId", "frameIdx"]).ngroups

    def get_frame_instances(self, video_id, frame_idx):
        is_in_frame = (self.points["videoId"] == video_id) & (self.points["frameIdx"] == frame_idx)
        if not is_in_frame.any():
            return []

        instances = []
        frame_instance_ids = np.unique(self.points["instanceId"][is_in_frame])
        for i, instance_id in enumerate(frame_instance_ids):
            is_instance = is_in_frame & (self.points["instanceId"] == instance_id)
            instance_points = {self.skeleton.node_names[n]: Point(x, y, visible=v) for x, y, n, v in
                                            zip(*[self.points[k][is_instance] for k in ["x", "y", "node", "visible"]])}

            instance = Instance(skeleton=self.skeleton, instance_id=instance_id, points=instance_points)
            instances.append(instance)

        return instances



if __name__ == "__main__":
    # data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.mat"
    # data = loadmat(data_path)

    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = Labels(data_path)
    print(len(labels))
""" Data generation routines """

from sleap.io.labels import Labels
# from sleap.instance import Instance, Point
# from sleap.skeleton import Skeleton

import numpy as np

def generate_confidence_maps(labels:Labels, sigma=5.0, scale=1.0, output_size=None):
    vid = labels.videos.iloc[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    xv = np.linspace(0, full_size[0] - 1, output_size[0], dtype="float32")
    yv = np.linspace(0, full_size[1] - 1, output_size[1], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    num_channels = len(labels.skeleton.node_names)
    keys = list(labels.instances.groupby(["videoId","frameIdx"]).groups.keys())
    points = []
    confmaps = np.zeros((len(keys), XX.shape[0], XX.shape[1], num_channels), dtype="float32")
    for i, (videoId, frameIdx) in enumerate(keys):

        valid_points = labels.points[(labels.points.videoId == videoId) & (labels.points.frameIdx == frameIdx) & labels.points.visible]
        points.append(valid_points)

        for point in valid_points.itertuples():
            confmap = np.exp(-((YY - point.y) ** 2 + (XX - point.x) ** 2) / (2 * sigma ** 2))
            confmaps[i, :, :, point.node] = np.maximum(confmaps[i, :, :, point.node], confmap)

    return confmaps, keys, points


if __name__ == "__main__":
    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = Labels(data_path)
    confmaps, keys, points = generate_confidence_maps(labels)
    print(confmaps.shape)
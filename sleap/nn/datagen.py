""" Data generation routines """

import os
import numpy as np

from sleap.io.labels import Labels
from sleap.io.video import HDF5Video, MediaVideo
# from sleap.instance import Instance, Point
# from sleap.skeleton import Skeleton

def generate_images(labels:Labels, scale=1.0, output_size=None):
    vid = labels.videos.iloc[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    keys = list(labels.instances.groupby(["videoId","frameIdx"]).groups.keys())
    imgs = []
    for i, (videoId, frameIdx) in enumerate(keys):
        vid = labels.videos[labels.videos.id == videoId].iloc[0]

        if vid.format == "media":
            video = MediaVideo(vid.filepath, grayscale=vid.channels == 1)

        imgs.append(video[frameIdx])

    imgs = np.concatenate(imgs, axis=0).astype("float32") / 255

    return imgs, keys


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
    if not os.path.exists(data_path):
        data_path = "D:/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = Labels(data_path)
    imgs, keys = generate_images(labels)
    confmaps, _keys, points = generate_confidence_maps(labels)

    import matplotlib.pyplot as plt

    cmap = np.array([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ]).astype("float32") / 255

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(imgs[0].squeeze())

    plt.subplot(1,2,2)
    for i in range(confmaps.shape[-1]):
        col = cmap[i % len(cmap)]
        I = confmaps[0][...,i][...,None] * col[None][None]
        I = np.concatenate((I, confmaps[0][...,i][...,None]), axis=-1)
        plt.imshow(I)

    plt.show()

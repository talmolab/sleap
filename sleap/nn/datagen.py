""" Data generation routines """

import os
import numpy as np

from sleap.io.dataset import Labels, load_labels_json_old
from sleap.io.video import Video
# from sleap.instance import Instance, Point
# from sleap.skeleton import Skeleton

def generate_images(labels:Labels, scale=1.0, output_size=None):

    vid = labels.videos[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    imgs = []
    for labeled_frame in labels.labels:
        img = labeled_frame.video[labeled_frame.frame_idx]
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)

    if imgs.dtype == "uint8":
        imgs = imgs.astype("float32") / 255

    return imgs


def generate_confidence_maps(labels:Labels, sigma=5.0, scale=1.0, output_size=None):
    vid = labels.videos[0]
    skeleton = labels.labels[0].instances[0].skeleton

    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    xv = np.linspace(0, full_size[0] - 1, output_size[0], dtype="float32")
    yv = np.linspace(0, full_size[1] - 1, output_size[1], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    num_channels = len(skeleton.nodes)
    confmaps = np.zeros((len(labels.labels), XX.shape[0], XX.shape[1], num_channels), dtype="float32")
    for i, labeled_frame in enumerate(labels.labels):
        for instance in labeled_frame.instances:
            for node in skeleton.nodes:
                if instance[node].visible:
                    j = skeleton.node_to_index(node)
                    confmaps[i, :, :, j] = np.maximum(
                        confmaps[i, :, :, j],
                        np.exp(-((YY - instance[node].y) ** 2 + (XX - instance[node].x) ** 2) / (2 * sigma ** 2))
                        )

    return confmaps


if __name__ == "__main__":
    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        data_path = "D:/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = load_labels_json_old(data_path)

    imgs = generate_images(labels)
    print(imgs.shape)
    print(imgs.dtype)
    print(np.ptp(imgs))

    confmaps = generate_confidence_maps(labels)
    print(confmaps.shape)
    print(confmaps.dtype)
    print(np.ptp(confmaps))

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

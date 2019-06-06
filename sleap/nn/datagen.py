""" Data generation routines """

import numpy as np
import math
from sleap.io.dataset import Labels

def generate_images(labels:Labels, scale=1.0, output_size=None):

    vid = labels.videos[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    imgs = []
    for labeled_frame in labels:
        img = labeled_frame.video[labeled_frame.frame_idx]
        # TODO: resizing
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)

    # TODO: more options for normalization
    if imgs.dtype == "uint8":
        imgs = imgs.astype("float32") / 255

    return imgs


def generate_confidence_maps(labels:Labels, sigma=5.0, scale=1.0, output_size=None):

    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    full_size = (vid.height, vid.width)
    if output_size is None:
        output_size = (vid.height // (1/scale), vid.width // (1/scale))

    # TODO: throw warning for truncation errors
    full_size = tuple(map(int, full_size))
    output_size = tuple(map(int, output_size))

    # Pre-allocate coordinate grid
    xv = np.linspace(0, full_size[1] - 1, output_size[1], dtype="float32")
    yv = np.linspace(0, full_size[0] - 1, output_size[0], dtype="float32")
    XX, YY = np.meshgrid(xv, yv)

    # TODO: speed up through broadcasting?
    num_frames = len(labels)
    num_channels = len(skeleton.nodes)
    confmaps = np.zeros((num_frames, output_size[0], output_size[1], num_channels), dtype="float32")
    for i, labeled_frame in enumerate(labels):
        for instance in labeled_frame.instances:
            for node in skeleton.nodes:
                if instance[node].visible and not math.isnan(instance[node].y) and not math.isnan(instance[node].y):
                    j = skeleton.node_to_index(node.name)
                    confmaps[i, :, :, j] = np.maximum(
                        confmaps[i, :, :, j],
                        np.exp(-((YY - instance[node].y) ** 2 + (XX - instance[node].x) ** 2) / (2 * sigma ** 2))
                        )

    return confmaps


def raster_pafs(arr, c, x0, y0, x1, y1, sigma=5):
    delta_x, delta_y = x1 - x0, y1 - y0

    edge_len = (delta_x ** 2 + delta_y ** 2) ** .5
    edge_x = delta_x / edge_len
    edge_y = delta_y / edge_len

    perp_x0 = x0 + (edge_y * sigma)
    perp_y0 = y0 - (edge_x * sigma)
    perp_x1 = x0 - (edge_y * sigma)
    perp_y1 = y0 + (edge_x * sigma)

    # perp 0 -> perp 0 + delta -> perp 1 + delta -> perp 1 -> perp 0
    xx = perp_x0, perp_x0 + delta_x, perp_x1 + delta_x, perp_x1
    yy = perp_y0, perp_y0 + delta_y, perp_y1 + delta_y, perp_y1

    from skimage.draw import polygon

    points_y, points_x = polygon(yy, xx, (arr.shape[0], arr.shape[1]))

    for x, y in zip(points_x, points_y):
        # print(x,y)
        arr[y, x, c] += edge_x
        arr[y, x, c + 1] += edge_y


def get_labels_edge_points_list(labels):
    return [(frame_idx, [instance_edge_points(instance) for instance in labeled_frame.instances]) for
            frame_idx, labeled_frame in enumerate(labels)]


def instance_edge_points(instance):
    skeleton = instance.skeleton
    points = [(edge_idx, (edge_points_tuples(instance, src_node, dst_node)))
              for edge_idx, (src_node, dst_node)
              in enumerate(skeleton.edges)
              if points_are_present(instance, src_node, dst_node)]
    return points


def edge_points_tuples(instance, src_node, dst_node):
    x0, y0 = instance[src_node].x, instance[src_node].y
    x1, y1 = instance[dst_node].x, instance[dst_node].y
    # (instance[src_node].x, instance[src_node].y), (instance[dst_node].x, instance[dst_node].y)
    # return np.array((x0, y0)), np.array((x1, y1))
    return (x0, y0), (x1, y1)


def points_are_present(instance, src_node, dst_node):
    if src_node in instance and dst_node in instance and instance[src_node].visible and instance[dst_node].visible:
        return True
    else:
        return False


def generate_pafs(labels: Labels, sigma=5.0, scale=1.0, output_size=None):
    # TODO: multi-skeleton support
    skeleton = labels.skeletons[0]

    vid = labels.videos[0]
    if output_size is None:
        output_size = (vid.height // (1 / scale), vid.width // (1 / scale))

    # TODO: throw warning for truncation errors
    output_size = tuple(map(int, output_size))

    # Pre-allocate output array
    num_frames = len(labels)
    num_channels = len(skeleton.edges) * 2

    pafs = np.zeros((num_frames, output_size[0], output_size[1], num_channels), dtype="float32")

    for frame_idx, frame_instance_edges in get_labels_edge_points_list(labels):
        for inst_edges in frame_instance_edges:
            for c, (src, dst) in inst_edges:
                raster_pafs(pafs[frame_idx], c * 2, *src, *dst, sigma)

    # Clip PAFs to valid range (in-place)
    np.clip(pafs, -1.0, 1.0, out=pafs)

    return pafs




if __name__ == "__main__":
    import os

    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        data_path = "tests/data/json_format_v2/minimal_instance.json"

    labels = Labels.load_json(data_path)

    imgs = generate_images(labels)
    print("--imgs--")
    print(imgs.shape)
    print(imgs.dtype)
    print(np.ptp(imgs))

    from PySide2 import QtWidgets
    from sleap.io.video import Video
    from sleap.gui.video import QtVideoPlayer
    from sleap.gui.confmapsplot import ConfMapsPlot
    from sleap.gui.quiverplot import MultiQuiverPlot

    app = QtWidgets.QApplication([])
    vid = Video.from_numpy(imgs * 255)
    conf_window = QtVideoPlayer(video=vid)
    conf_window.setWindowTitle("confmaps")
    conf_window.show()

    confmaps = generate_confidence_maps(labels)
    print("--confmaps--")
    print(confmaps.shape)
    print(confmaps.dtype)
    print(np.ptp(confmaps))

    def plot_confmaps(parent, item_idx):
        frame_conf_map = ConfMapsPlot(confmaps[parent.frame_idx,...])
        conf_window.view.scene.addItem(frame_conf_map)

    conf_window.changedPlot.connect(plot_confmaps)
    conf_window.plot()

    pafs = generate_pafs(labels)
    print("--pafs--")
    print(pafs.shape)
    print(pafs.dtype)
    print(np.ptp(pafs))

    demo_pafs(pafs, video)

    app.exec_()

#     import matplotlib.pyplot as plt
# 
#     cmap = np.array([
#         [0,   114,   189],
#         [217,  83,    25],
#         [237, 177,    32],
#         [126,  47,   142],
#         [119, 172,    48],
#         [77,  190,   238],
#         [162,  20,    47],
#         ]).astype("float32") / 255
# 
#     idx = 0
#     img = imgs[idx]
#     confmap = confmaps[idx]
#     paf = pafs[idx]
# 
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(img.squeeze(), cmap="gray")
# 
#     plt.subplot(1,3,2)
#     plt.imshow(img.squeeze(), cmap="gray")
#     for i in range(confmap.shape[-1]):
#         col = cmap[i % len(cmap)]
#         I = confmap[...,i][...,None] * col[None][None]
#         I = np.concatenate((I, confmap[...,i][...,None]), axis=-1) # add alpha
#         plt.imshow(I)
# 
#     # Warning: VERY SLOW to plot these vector fields in matplotlib
#     # Reimplement in Qt with polyline or 3 lines?
#     plt.subplot(1,3,3)
#     plt.imshow(img.squeeze(), cmap="gray")
#     for e in range(paf.shape[-1] // 2):
#         col = cmap[e % len(cmap)]
#         paf_x = paf[..., e * 2]
#         paf_y = paf[..., (e * 2) + 1]
#         plt.quiver(np.arange(paf.shape[1]), np.arange(paf.shape[0]), paf_x, paf_y, color=col,
#             angles="xy", scale=1, scale_units="xy")
# 
#     plt.show()

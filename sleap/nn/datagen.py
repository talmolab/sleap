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


def generate_pafs(labels:Labels, sigma=5.0, scale=1.0, output_size=None):
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
    coord_grid = np.stack((XX, YY), axis=0)

    # Pre-allocate output array
    num_frames = len(labels)
    channel_edge_idx = np.repeat(np.arange(len(skeleton.edges)), 2)
    num_channels = len(channel_edge_idx)
    pafs = np.zeros((num_frames, output_size[0], output_size[1], num_channels), dtype="float32")

    for i, labeled_frame in enumerate(labels):
        for instance in labeled_frame:
            for e, (src_node, dst_node) in enumerate(skeleton.edges):
                # Ignore edge if either node is not present or visible in the instance
                if not (src_node in instance and dst_node in instance): continue
                if not (instance[src_node].visible and instance[dst_node].visible): continue

                # Pull out coordinates
                src = np.array((instance[src_node].x, instance[src_node].y))
                dst = np.array((instance[dst_node].x, instance[dst_node].y))

                # Compute distance between points
                edge_length = np.linalg.norm(dst - src)

                # Skip if points are on the same coordinate
                if edge_length == 0: continue

                # Compute unit vectors (i.e., norm == magnitude == 1)
                edge_vec = (dst - src) / edge_length # along the edge
                edge_perp_vec = np.array((-edge_vec[1], edge_vec[0])) # orthogonal to edge

                # Compute coordinate grid relative to the source point
                rel_grid = coord_grid - src[:, None, None] # [X, Y] x height x width

                # Compute signed distance along the edge at each grid point
                edge_dist_grid = np.sum(edge_vec[:, None, None] * rel_grid, axis=0) # height x width

                # Compute absolute distance perpendicular to the edge at each grid point
                edge_perp_dist_grid = np.abs(np.sum(edge_perp_vec[:, None, None] * rel_grid, axis=0))

                # Compute mask for edge PAF based on edge distances
                paf_mask = np.logical_and.reduce((
                    edge_dist_grid >= -sigma, # -sigma units before src along the edge
                    edge_dist_grid <= (edge_length + sigma), # +sigma units after dst along the edge
                    edge_perp_dist_grid <= sigma, # sigma units perpendicular from edge
                    ), dtype="float32")

                # Create PAF by placing the edge direction vector at each masked coordinate
                paf = np.moveaxis(edge_vec[:, None, None], 0, 2) * paf_mask[:, :, None]

                # Accumulate
                pafs[i][:, :, channel_edge_idx == e] += paf

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

    window_pafs = QtVideoPlayer(video=vid)
    window_pafs.setWindowTitle("pafs")
    window_pafs.show()

    def plot_fields(parent, i):
        aff_fields_item = MultiQuiverPlot(pafs[parent.frame_idx,...], show=None, decimation=1)
        window_pafs.view.scene.addItem(aff_fields_item)

    window_pafs.changedPlot.connect(plot_fields)
    window_pafs.plot()

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

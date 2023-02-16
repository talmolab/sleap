"""Visualization and plotting utilities."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import Union, Tuple, Optional, Text
from sleap import Instance
from io import BytesIO
from PIL import Image


def imgfig(
    size: Union[float, Tuple] = 6, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Create a tight figure for image plotting.

    Args:
        size: Scalar or 2-tuple specifying the (width, height) of the figure in inches.
            If scalar, will assume equal width and height.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    fig = plt.figure(figsize=(scale * size[0], scale * size[1]), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def plot_img(
    img: np.ndarray, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Plot an image in a tight figure."""
    if hasattr(img, "numpy"):
        img = img.numpy()

    if img.shape[0] == 1:
        # Squeeze out batch singleton dimension.
        img = img.squeeze(axis=0)

    # Check if image is grayscale (single channel).
    grayscale = img.shape[-1] == 1
    if grayscale:
        # Squeeze out singleton channel.
        img = img.squeeze(axis=-1)

    # Normalize the range of pixel values.
    img_min = img.min()
    img_max = img.max()
    if img_min < 0.0 or img_max > 1.0:
        img = (img - img_min) / (img_max - img_min)

    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )

    ax = fig.gca()
    ax.imshow(
        img,
        cmap="gray" if grayscale else None,
        origin="upper",
        extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
    )
    return fig


def plot_confmaps(confmaps: np.ndarray, output_scale: float = 1.0):
    """Plot confidence maps reduced over channels."""
    ax = plt.gca()
    return ax.imshow(
        np.squeeze(confmaps.max(axis=-1)),
        alpha=0.5,
        origin="upper",
        vmin=0,
        vmax=1,
        extent=[
            -0.5,
            confmaps.shape[1] / output_scale - 0.5,
            confmaps.shape[0] / output_scale - 0.5,
            -0.5,
        ],
    )


def plot_peaks(
    pts_gt: np.ndarray, pts_pr: Optional[np.ndarray] = None, paired: bool = False
):
    """Plot ground truth and detected peaks."""
    handles = []
    ax = plt.gca()
    if paired and pts_pr is not None:
        for p_gt, p_pr in zip(pts_gt, pts_pr):
            handles.append(
                ax.plot([p_gt[0], p_pr[0]], [p_gt[1], p_pr[1]], "r-", alpha=0.5, lw=2)
            )
    if pts_pr is not None:
        handles.append(
            ax.plot(
                pts_gt[..., 0].ravel(),
                pts_gt[..., 1].ravel(),
                "g.",
                alpha=0.7,
                ms=10,
                mew=1,
                mec="w",
            )
        )
        handles.append(
            ax.plot(pts_pr[:, 0], pts_pr[:, 1], "r.", alpha=0.7, ms=10, mew=1, mec="w")
        )
    else:
        cmap = sns.color_palette("tab20")
        for i, pt in enumerate(pts_gt):
            handles.append(
                ax.plot(
                    pt[0],
                    pt[1],
                    ".",
                    alpha=0.7,
                    ms=15,
                    mew=1,
                    mfc=cmap[i % len(cmap)],
                    mec="w",
                )
            )
    return handles


def plot_pafs(
    pafs: np.ndarray,
    output_scale: float = 1.0,
    stride: int = 1,
    scale: float = 4.0,
    width: float = 1.0,
    cmap: Optional[Text] = None,
):
    """Quiver plot for a single frame of pafs."""
    if cmap is None:
        cmap = sns.color_palette("tab20")

    pafs = pafs.reshape((pafs.shape[0], pafs.shape[1], -1, 2))

    h_quivers = []
    for k in range(pafs.shape[-2]):
        pafs_k = pafs[..., k, :]  # rank 3
        pafs_k = pafs_k[::stride, ::stride, :]

        h_quivers_k = plt.quiver(
            np.linspace(
                0,
                (stride * pafs_k.shape[1] / output_scale),
                pafs_k.shape[1],
                endpoint=False,
            ),
            np.linspace(
                0,
                (stride * pafs_k.shape[0] / output_scale),
                pafs_k.shape[0],
                endpoint=False,
            ),
            pafs_k[..., 0],
            pafs_k[..., 1],
            angles="xy",
            pivot="mid",
            units="xy",
            scale_units="xy",
            scale=1.0 / scale,
            color=cmap[k % len(cmap)],
            minlength=0.1,
            width=width,
            alpha=0.8,
        )
        h_quivers.append(h_quivers_k)

    return h_quivers


def plot_instance(
    instance,
    skeleton=None,
    cmap=None,
    color_by_node=False,
    lw=2,
    ms=10,
    bbox=None,
    scale=1.0,
    **kwargs,
):
    """Plot a single instance with edge coloring."""
    if cmap is None:
        cmap = sns.color_palette("tab20")

    if skeleton is None and hasattr(instance, "skeleton"):
        skeleton = instance.skeleton

    if skeleton is None:
        color_by_node = True
    else:
        if len(skeleton.edges) == 0:
            color_by_node = True

    if hasattr(instance, "numpy"):
        inst_pts = instance.numpy()
    else:
        inst_pts = instance

    h_lines = []
    if color_by_node:
        for k, (x, y) in enumerate(inst_pts):
            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(x, y, ".", ms=ms, c=cmap[k % len(cmap)], **kwargs)
            h_lines.append(h_lines_k)

    else:
        for k, (src_node, dst_node) in enumerate(skeleton.edges):
            src_pt = instance.points_array[instance.skeleton.node_to_index(src_node)]
            dst_pt = instance.points_array[instance.skeleton.node_to_index(dst_node)]

            x = np.array([src_pt[0], dst_pt[0]])
            y = np.array([src_pt[1], dst_pt[1]])

            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(
                x, y, ".-", ms=ms, lw=lw, c=cmap[k % len(cmap)], **kwargs
            )

            h_lines.append(h_lines_k)

    return h_lines


def plot_instances(
    instances, skeleton=None, cmap=None, color_by_track=False, tracks=None, **kwargs
):
    """Plot a list of instances with identity coloring."""

    if cmap is None:
        cmap = sns.color_palette("tab10")

    if color_by_track and tracks is None:
        # Infer tracks for ordering if not provided.
        tracks = set()
        for instance in instances:
            tracks.add(instance.track)

        # Sort by spawned frame.
        tracks = sorted(list(tracks), key=lambda track: track.name)

    h_lines = []
    for i, instance in enumerate(instances):
        if color_by_track:
            if instance.track is None:
                raise ValueError(
                    "Instances must have a set track when coloring by track."
                )

            if instance.track not in tracks:
                raise ValueError("Instance has a track not found in specified tracks.")

            color = cmap[tracks.index(instance.track) % len(cmap)]

        else:
            # Color by identity (order in list).
            color = cmap[i % len(cmap)]

        h_lines_i = plot_instance(instance, skeleton=skeleton, cmap=[color], **kwargs)
        h_lines.append(h_lines_i)

    return h_lines


def plot_bbox(bbox, **kwargs):
    if hasattr(bbox, "bounding_box"):
        bbox = bbox.bounding_box
    y1, x1, y2, x2 = bbox
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "-", **kwargs)


def generate_skeleton_preview_image(
    instance: Instance, square_bb: bool = True, thumbnail_size=(128, 128)
) -> bytes:
    """Generate preview image for skeleton based on given instance.

    Args:
        instance: A `sleap.Instance` object for which to generate the preview image from.
        square_bb: A boolean flag for whether or not the preview image should be a square image
        thumbnail_size: A tuple of (w,h) for what the size of the thumbnail image should be

    Returns:
        A byte string encoding of the preview image.
    """

    def get_square_bounding_box(bb):
        """Convert rectangular bounding box to square bounding box.

        Args:
            bb: A tuple representing a bounding box in `sleap.Instance.bounding_box`
                with the format [y1, x1, y2, x2]

        Returns:
            A square bounding box in `PIL.Image.crop()` with the format [x1, y1, x2, y2]
        """

        y1, x1, y2, x2 = bb

        # Get side lengths
        dist_x = x2 - x1
        dist_y = y2 - y1

        mid_x = x1 + dist_x / 2
        mid_y = y1 + dist_y / 2

        # Get max side length to use as square side length
        max_dist = max(dist_x, dist_y)

        # Get new coordinates
        new_x1 = mid_x - max_dist / 2
        new_x2 = mid_x + max_dist / 2
        new_y1 = mid_y - max_dist / 2
        new_y2 = mid_y + max_dist / 2

        assert new_x2 - new_x1 == new_y2 - new_y1, ValueError(
            f"{new_x2-new_x1} != {new_y2-new_y1}"
        )
        return (new_x1, new_y1, new_x2, new_y2)

    if square_bb:
        x1, y1, x2, y2 = get_square_bounding_box(instance.bounding_box)
    else:
        y1, x1, y2, x2 = instance.bounding_box
    bb = [x1, y1, x2, y2]
    bb = [coor - 20 if idx < 2 else coor + 20 for idx, coor in enumerate(bb)]

    frame = plot_img(instance.video.get_frame(instance.frame_idx))

    # Custom formula for scaling line width and marker size based on bounding box size.
    max_dim = max(abs(y1 - y2), abs(x1 - x2))
    ms = int(max_dim / 7)
    lw = int(max_dim / 30)
    skeleton = plot_instance(
        instance, skeleton=instance.skeleton, lw=lw, ms=ms, color_by_node=False
    )

    fig = skeleton[0][0].figure
    ax = fig.gca()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    fig.set(facecolor="white", frameon=False)

    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", facecolor="white")
    im = Image.open(img_buf)
    im = im.crop(bb)
    im.thumbnail(thumbnail_size)

    img_stream = BytesIO()
    im.save(img_stream, format="png")
    img_bytes = img_stream.getvalue()  # image in binary format
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

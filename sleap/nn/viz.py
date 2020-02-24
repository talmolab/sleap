"""Visualization and plotting utilities."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple


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
    if not isinstance(size, [tuple, list]):
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
    if img.shape[0] == 1:
        img = img.squeeze(axis=0)
    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )
    fig.gca().imshow(
        img,
        cmap="gray" if img.shape[-1] == 1 else None,
        origin="lower",
        extent=[-0.5, img.shape[1] - 0.5, -0.5, img.shape[0] - 0.5],
    )
    return fig


def plot_confmaps(confmaps: np.ndarray, output_scale: float = 1.0):
    """Plot confidence maps reduced over channels."""
    ax = plt.gca()
    return ax.imshow(
        np.squeeze(cms.max(axis=-1)),
        alpha=0.5,
        origin="lower",
        extent=[
            -0.5,
            cms.shape[1] / output_scale - 0.5,
            -0.5,
            cms.shape[0] / output_scale - 0.5,
        ],
    )


def plot_peaks(pts_gt: np.ndarray, pts_pr: np.ndarray):
    """Plot ground truth and detected peaks."""
    handles = []
    ax = plt.gca()
    if pts_gt.shape == pts_pr.shape:
        for p_gt, p_pr in zip(pts_gt, pts_pr):
            handles.append(
                ax.plot([p_gt[0], p_pr[0]], [p_gt[1], p_pr[1]], "r-", alpha=0.5, lw=2)
            )
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
    return handles


def plot_pafs(
    pafs: np.ndarray, output_scale=1.0, stride=1, scale=4.0, width=1.0, cmap=None
):
    """Quiver plot for a single frame of pafs."""
    if cmap is None:
        cmap = sns.color_palette("tab20")

    if pafs.shape[-1] != 2:
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

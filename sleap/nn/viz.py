"""Visualization and plotting utilities."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    pts_gt: np.ndarray, pts_pr: np.ndarray | None = None, paired: bool = False
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
    cmap: str | None = None,
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


def plot_bbox(bbox, **kwargs):
    if hasattr(bbox, "bounding_box"):
        bbox = bbox.bounding_box
    y1, x1, y2, x2 = bbox
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "-", **kwargs)

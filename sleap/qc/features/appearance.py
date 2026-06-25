"""Appearance-outlier detection: points placed on occluders / the wrong object.

This module implements detector (e): "points placed on occluders / wrong
object" -- e.g. a keypoint dragged onto white cotton bedding sitting over a dark
mouse, or onto a cage bar. Unlike the geometry-only detectors, the *geometry*
of such an error can look perfectly plausible (the point is in a sensible place
for the pose); what gives it away is the **image content** under the node. We
therefore build a small per-node appearance model from the labeled data and
flag nodes whose local image patch does not look like that node usually looks.

How it works
------------
At fit time, for every *visible* node of every (clean) labeled instance we cut a
``patch_size x patch_size`` window centred on the node's ``(x, y)`` in its frame
and reduce it to a compact **appearance descriptor**: per-channel mean and
standard deviation of the patch intensities (a length-2 vector for grayscale
``(H, W, 1)`` frames, length-6 for RGB). Mean captures brightness ("white cotton
bedding" vs "dark fur"), std captures local texture/contrast. We accumulate
these descriptors per node and fit a robust Gaussian (median + regularized
covariance), giving a per-node squared **Mahalanobis** distance. Nodes with
fewer than ``min_samples`` descriptors are left unmodeled (no opinion).

At score time, for each visible node we cut the same descriptor and compute its
Mahalanobis distance to that node's learned model, then squash it to ``[0, 1]``
against a per-node distance scale learned at fit time (so "normal" appearance
sits near 0 and clear outliers approach 1). The per-instance score is the
**max** over nodes -- one badly-misplaced node is enough to warrant review.

This is a NON-GMM channel detector (default-OFF / experimental): it produces a
per-instance ``appearance_outlier_score`` in ``[0, 1]`` that the integration
layer stores in :attr:`QCResults.channel_scores` under the ``"appearance"``
channel, exactly like the missing-node detector.

Honest scope / limitations
---------------------------
- The descriptor is intentionally tiny (mean + std per channel). It catches
  gross "this node is on visually-different stuff" errors (bright bedding vs
  dark fur, a node on the cage floor vs on the animal). It will NOT distinguish
  two regions that have the same brightness/texture but are semantically
  different (e.g. two similarly-grey body parts).
- It learns the *project's own* appearance statistics. If a node is
  systematically mislabeled onto the same wrong surface across the whole
  dataset, that wrong surface becomes "normal" and is not flagged.
- Patches near the image border are clipped to the valid region, so an
  edge node is described by fewer pixels but is still modeled/scored.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# Minimum number of patch samples required before a node is modeled at all.
DEFAULT_MIN_SAMPLES = 20

# Floor added to the covariance diagonal (in descriptor units, ~intensity^2)
# to keep it invertible when a node's patches are nearly constant.
_COV_REG = 1.0

# Multiple of the learned median Mahalanobis distance that maps to a score of
# ~0.5 (the half-saturation point of the squashing function). Larger = more
# permissive (a node must be more anomalous to score high).
_DIST_SCALE_FACTOR = 3.0

# Floor on the per-node distance scale so a node whose training patches are
# essentially identical (median distance ~0) does not produce an infinite /
# hair-trigger score. In Mahalanobis (whitened) units.
_MIN_DIST_SCALE = 1.0


def extract_patch_descriptor(
    frame: np.ndarray,
    x: float,
    y: float,
    patch_size: int = 7,
) -> Optional[np.ndarray]:
    """Cut a patch around ``(x, y)`` and reduce it to an appearance descriptor.

    The descriptor is the per-channel **mean** followed by the per-channel
    **standard deviation** of the patch intensities. For a grayscale frame
    ``(H, W, 1)`` (or ``(H, W)``) this is a length-2 vector ``[mean, std]``; for
    an RGB frame ``(H, W, 3)`` it is length-6 ``[mean_r, mean_g, mean_b, std_r,
    std_g, std_b]``. Patches that fall partly outside the image are clipped to
    the valid region (an edge node is described by fewer pixels).

    Args:
        frame: Decoded image as ``(H, W)``, ``(H, W, 1)`` or ``(H, W, C)``
            ndarray (typically ``uint8``). Decoding is the caller's job so the
            heavy video I/O stays out of unit tests.
        x: Node x-coordinate (column) in pixels. May be fractional; rounded to
            the nearest pixel for the patch centre.
        y: Node y-coordinate (row) in pixels.
        patch_size: Side length (pixels) of the square window. Values < 1 are
            treated as 1. Defaults to 7.

    Returns:
        A 1-D ``float64`` descriptor of length ``2 * C``, or ``None`` if the
        frame is unusable (empty / not 2-D-or-3-D), the coordinate is NaN, or
        the clipped patch is empty (centre fully outside the image).
    """
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[:, :, None]
    if frame.ndim != 3 or frame.size == 0:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    height, width = frame.shape[0], frame.shape[1]
    if height == 0 or width == 0:
        return None

    half = max(int(patch_size), 1) // 2
    cx = int(round(float(x)))
    cy = int(round(float(y)))

    # Clip the window to the valid image region so edge nodes still yield a
    # (smaller) patch instead of an out-of-bounds error.
    r0 = max(cy - half, 0)
    r1 = min(cy + half + 1, height)
    c0 = max(cx - half, 0)
    c1 = min(cx + half + 1, width)
    if r1 <= r0 or c1 <= c0:
        # Centre is fully outside the image -> nothing to describe.
        return None

    patch = frame[r0:r1, c0:c1, :].astype(np.float64)
    # Reduce over the spatial axes, keeping per-channel stats.
    flat = patch.reshape(-1, patch.shape[2])
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    return np.concatenate([means, stds]).astype(np.float64)


def _patches_to_descriptors(
    patches: Iterable[np.ndarray], patch_size: int
) -> list[np.ndarray]:
    """Reduce already-cut patches to descriptors (helper for tests/integration).

    Each patch is treated as a tiny image and run through the same per-channel
    mean+std reduction as :func:`extract_patch_descriptor`. ``None`` / empty
    patches are skipped.
    """
    out: list[np.ndarray] = []
    for patch in patches:
        if patch is None:
            continue
        patch = np.asarray(patch)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        if patch.ndim != 3 or patch.size == 0:
            continue
        flat = patch.reshape(-1, patch.shape[2]).astype(np.float64)
        out.append(
            np.concatenate([flat.mean(axis=0), flat.std(axis=0)]).astype(np.float64)
        )
    return out


def _fit_node_model(descriptors: np.ndarray) -> Optional[dict]:
    """Fit a robust Gaussian + distance scale for one node's descriptors.

    Uses the median as a robust centre and a regularized covariance (with a
    small diagonal floor) so the squared Mahalanobis distance stays finite even
    for near-constant patches. The learned ``dist_scale`` is the median
    training Mahalanobis distance, used later to squash test distances to
    ``[0, 1]``.

    Args:
        descriptors: ``(n_samples, n_features)`` array of node descriptors.

    Returns:
        Per-node model dict, or ``None`` if there are too few samples.
    """
    descriptors = np.asarray(descriptors, dtype=np.float64)
    n_samples, n_features = descriptors.shape
    if n_samples < 2:
        return None

    center = np.median(descriptors, axis=0)
    cov = np.cov(descriptors, rowvar=False)
    cov = np.atleast_2d(cov)
    # Regularize: keep the matrix invertible / well-conditioned even when a
    # feature is (near-)constant across the training patches.
    cov = cov + _COV_REG * np.eye(n_features)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    # Per-node distance scale: typical (median) Mahalanobis distance of the
    # training descriptors themselves. Floored so identical patches don't give
    # a hair-trigger score.
    diffs = descriptors - center
    d2 = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    d2 = np.clip(d2, 0.0, None)
    train_dist = np.sqrt(d2)
    dist_scale = float(max(np.median(train_dist), _MIN_DIST_SCALE))

    return {
        "center": center,
        "inv_cov": inv_cov,
        "dist_scale": dist_scale,
        "n_samples": int(n_samples),
    }


def _mahalanobis_distance(descriptor: np.ndarray, node_model: dict) -> float:
    """Squared-root Mahalanobis distance of ``descriptor`` to a node model."""
    diff = np.asarray(descriptor, dtype=np.float64) - node_model["center"]
    d2 = float(diff @ node_model["inv_cov"] @ diff)
    if not np.isfinite(d2) or d2 < 0.0:
        d2 = 0.0
    return float(np.sqrt(d2))


def _squash(distance: float, dist_scale: float) -> float:
    """Map a non-negative distance to ``[0, 1)`` with a smooth saturation.

    Uses ``d / (d + s)`` where ``s = _DIST_SCALE_FACTOR * dist_scale`` is the
    half-saturation distance: a node at the typical (training-median) distance
    scores well below 0.5, while a node many scales away approaches 1.
    """
    if not np.isfinite(distance) or distance <= 0.0:
        return 0.0
    s = max(_DIST_SCALE_FACTOR * dist_scale, 1e-6)
    return float(distance / (distance + s))


def fit_appearance(
    frames_and_instances: Iterable[tuple],
    n_nodes: int,
    patch_size: int = 7,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """Learn a per-node appearance model from labeled instances.

    For each ``(frame, points)`` pair and each *visible* node, cut a
    ``patch_size x patch_size`` patch at the node's ``(x, y)`` and reduce it to
    an appearance descriptor (per-channel mean + std). Descriptors are pooled
    per node and a robust Gaussian is fit per node that has at least
    ``min_samples`` samples; nodes below that are left unmodeled.

    Decoding is the caller's responsibility: pass already-decoded frame arrays
    (``(H, W)``, ``(H, W, 1)`` or ``(H, W, C)``) so the heavy video I/O stays
    out of unit tests. Pairs whose frame is ``None`` (e.g. undecodable) are
    skipped; NaN / out-of-frame nodes are skipped per node.

    Args:
        frames_and_instances: Iterable of ``(frame, points)`` tuples, where
            ``frame`` is a decoded image (or ``None`` to skip) and ``points`` is
            an ``(n_nodes, 2)`` array with NaN for invisible nodes.
        n_nodes: Number of skeleton nodes (length of each ``points`` row dim).
        patch_size: Side length of the square appearance patch. Defaults to 7.
        min_samples: Minimum patch samples for a node to be modeled. Defaults to
            :data:`DEFAULT_MIN_SAMPLES` (20).

    Returns:
        A model dict with:

        - ``"node_models"``: ``dict[int, dict]`` mapping modeled node index to
          its per-node model (``center``, ``inv_cov``, ``dist_scale``,
          ``n_samples``). Unmodeled nodes are absent.
        - ``"node_sample_counts"``: ``dict[int, int]`` mapping every node index
          seen to how many descriptors were collected (modeled or not). Useful
          for diagnostics / explaining why a node has no opinion.
        - ``"n_nodes"``: number of skeleton nodes.
        - ``"patch_size"``: the patch size used (so scoring can default to it).
        - ``"min_samples"``: the threshold used.
        - ``"descriptor_dim"``: descriptor length seen (``2 * C``), or ``None``
          if no descriptors were collected.
    """
    n_nodes = int(n_nodes)
    patch_size = max(int(patch_size), 1)
    per_node: list[list[np.ndarray]] = [[] for _ in range(n_nodes)]
    descriptor_dim: Optional[int] = None

    for pair in frames_and_instances:
        if pair is None:
            continue
        frame, points = pair
        if frame is None or points is None:
            continue
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            continue

        n = min(points.shape[0], n_nodes)
        for node_idx in range(n):
            x, y = points[node_idx, 0], points[node_idx, 1]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            desc = extract_patch_descriptor(frame, x, y, patch_size)
            if desc is None:
                continue
            # Descriptor length is fixed by channel count; guard against mixing
            # grayscale and RGB frames in one fit (keep the first seen shape).
            if descriptor_dim is None:
                descriptor_dim = desc.shape[0]
            elif desc.shape[0] != descriptor_dim:
                continue
            per_node[node_idx].append(desc)

    node_models: dict[int, dict] = {}
    node_sample_counts: dict[int, int] = {}
    for node_idx in range(n_nodes):
        samples = per_node[node_idx]
        node_sample_counts[node_idx] = len(samples)
        if len(samples) < min_samples:
            continue
        model = _fit_node_model(np.asarray(samples, dtype=np.float64))
        if model is not None:
            node_models[node_idx] = model

    return {
        "node_models": node_models,
        "node_sample_counts": node_sample_counts,
        "n_nodes": n_nodes,
        "patch_size": patch_size,
        "min_samples": int(min_samples),
        "descriptor_dim": descriptor_dim,
    }


def score_appearance(
    frame: np.ndarray,
    points: np.ndarray,
    model: dict,
    patch_size: Optional[int] = None,
) -> dict:
    """Score an instance for appearance outliers against a fitted model.

    For each *visible* node that the model has an opinion on, cut its patch,
    compute the Mahalanobis distance of its descriptor to the node's learned
    model, and squash it to ``[0, 1]``. The per-instance score is the max over
    scored nodes (one clearly-misplaced node is enough to flag the instance).

    Args:
        frame: Decoded image (``(H, W)``, ``(H, W, 1)`` or ``(H, W, C)``), or
            ``None`` if the frame could not be decoded (returns a zero score).
        points: ``(n_nodes, 2)`` array with NaN for invisible nodes.
        model: Model dict from :func:`fit_appearance`.
        patch_size: Patch side length. If ``None``, uses the size stored in the
            model (the one used at fit time).

    Returns:
        Dictionary with:

        - ``"appearance_outlier_score"``: ``float`` in ``[0, 1]``. Max squashed
          Mahalanobis distance over scored nodes (0.0 if none could be scored).
          High = a node sits on visually-anomalous pixels.
        - ``"worst_node"``: ``int`` node index of the highest-scoring node, or
          ``-1`` if no node was scored.
        - ``"node_scores"``: ``dict[int, float]`` mapping each scored node index
          to its individual ``[0, 1]`` outlier score.
    """
    empty = {
        "appearance_outlier_score": 0.0,
        "worst_node": -1,
        "node_scores": {},
    }
    if model is None:
        return empty
    node_models = model.get("node_models", {})
    if not node_models:
        return empty
    if frame is None or points is None:
        return empty

    if patch_size is None:
        patch_size = model.get("patch_size", 7)
    patch_size = max(int(patch_size), 1)
    descriptor_dim = model.get("descriptor_dim")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return empty

    node_scores: dict[int, float] = {}
    n = points.shape[0]
    for node_idx, node_model in node_models.items():
        if node_idx >= n:
            continue
        x, y = points[node_idx, 0], points[node_idx, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        desc = extract_patch_descriptor(frame, x, y, patch_size)
        if desc is None:
            continue
        # Skip if the channel count (descriptor length) does not match the
        # model -- scoring an RGB frame against a grayscale model is undefined.
        if descriptor_dim is not None and desc.shape[0] != descriptor_dim:
            continue
        if desc.shape[0] != node_model["center"].shape[0]:
            continue
        dist = _mahalanobis_distance(desc, node_model)
        node_scores[int(node_idx)] = _squash(dist, node_model["dist_scale"])

    if not node_scores:
        return empty

    worst_node = max(node_scores, key=node_scores.get)
    score = float(np.clip(node_scores[worst_node], 0.0, 1.0))
    return {
        "appearance_outlier_score": score,
        "worst_node": int(worst_node),
        "node_scores": node_scores,
    }

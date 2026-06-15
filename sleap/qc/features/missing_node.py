"""Missing-node detection: labelable points left unlabeled.

This module implements a *geometry/visibility-only* heuristic (Tier-1) for
detector (f): "labelable points left unlabeled". The idea is to flag instances
that are missing a node which their *peers* (instances where the same
co-visible nodes are present) usually keep visible.

The detector reuses the co-visibility statistics learned by
:class:`sleap.qc.features.visibility.VisibilityModel`: the integration layer
fits that model on the labeled dataset and passes the learned
``co_visibility_matrix`` (``P(node_j visible | node_i visible)``) here. This
module itself is a pure function and learns nothing.

Honest scope / limitations:
    This only catches *outlier* drops -- an individual instance that is missing
    a node its co-visible peers keep. It does NOT catch dataset-wide systematic
    under-labeling (e.g. nobody in the project ever labels the tail tip). When a
    node is rarely labeled, the co-visibility column for that node is small for
    everyone, so its expected probability stays low and it is never flagged.
    Detecting systematic under-labeling requires a model-based (Tier-2)
    approach that compares against learned appearance/geometry rather than only
    the project's own visibility statistics.
"""

from __future__ import annotations

import numpy as np


def score_missing_nodes(
    visibility_mask: np.ndarray,
    co_visibility_matrix: np.ndarray,
    edges: list[tuple[int, int]],
    threshold: float = 0.9,
    require_neighbors_visible: bool = False,
) -> dict:
    """Score whether an instance is missing nodes its peers usually keep.

    For each *invisible* node ``k`` we estimate the probability that it
    *should* be visible given the nodes that are actually present::

        p_expected[k] = mean over visible nodes i of co_visibility_matrix[i, k]

    where ``co_visibility_matrix[i, k] = P(node k visible | node i visible)``
    (the convention used by :class:`VisibilityModel`). A node is flagged as
    suspiciously-missing when ``p_expected[k] >= threshold`` -- i.e. nodes that
    co-occur with the visible nodes nearly always also bring ``k`` along, yet
    ``k`` is absent here.

    Optionally (``require_neighbors_visible=True``) a node is only flagged when
    *all* of its skeleton neighbors (from ``edges``) are visible, which makes
    the heuristic stricter: an isolated missing node surrounded by present
    neighbors is a much stronger signal of an accidental drop than a node on
    the boundary of an occluded region.

    Args:
        visibility_mask: ``(n_nodes,)`` boolean (or boolean-like) array. ``True``
            = node visible/labeled, ``False`` = invisible/missing. Non-boolean
            input is coerced with :func:`numpy.asarray(..., dtype=bool)`.
        co_visibility_matrix: ``(n_nodes, n_nodes)`` array where entry
            ``[i, j] = P(node j visible | node i visible)``, as learned by
            :meth:`VisibilityModel.fit`.
        edges: List of ``(src, dst)`` node-index pairs describing the skeleton
            graph. Only used when ``require_neighbors_visible`` is set. May be
            empty.
        threshold: Minimum expected visibility probability for a missing node to
            be flagged as suspicious. Defaults to ``0.9``.
        require_neighbors_visible: If ``True``, only flag a missing node when all
            of its skeleton neighbors are visible. Defaults to ``False``.

    Returns:
        Dictionary with:

        - ``missing_node_score``: ``float`` in ``[0, 1]``. The maximum expected
          visibility probability over all invisible nodes (0.0 if no node is
          invisible). High values mean at least one missing node is one the
          peers almost always keep.
        - ``suspicious_nodes``: ``list[int]`` of node indices that were flagged
          (``p_expected >= threshold`` and, if requested, all-neighbors-visible),
          sorted ascending. Useful for naming the nodes in an explanation.
        - ``n_suspicious``: ``int`` count of flagged nodes.
    """
    visibility_mask = np.asarray(visibility_mask, dtype=bool).ravel()
    co_visibility_matrix = np.asarray(co_visibility_matrix, dtype=float)
    n_nodes = visibility_mask.shape[0]

    # Guard against an empty skeleton or a co-visibility matrix that does not
    # match the mask -- there is nothing meaningful to flag in either case.
    if n_nodes == 0 or co_visibility_matrix.shape != (n_nodes, n_nodes):
        return {
            "missing_node_score": 0.0,
            "suspicious_nodes": [],
            "n_suspicious": 0,
        }

    visible_indices = np.where(visibility_mask)[0]
    invisible_indices = np.where(~visibility_mask)[0]

    # All nodes visible (nothing missing) or no visible node to condition on
    # (we have no evidence about what *should* be present). Either way: score 0.
    if len(invisible_indices) == 0 or len(visible_indices) == 0:
        return {
            "missing_node_score": 0.0,
            "suspicious_nodes": [],
            "n_suspicious": 0,
        }

    # Build neighbor adjacency only if we need it (avoids work in the common
    # require_neighbors_visible=False path).
    neighbors: list[list[int]] = []
    if require_neighbors_visible:
        neighbors = [[] for _ in range(n_nodes)]
        for src, dst in edges:
            # Ignore edges that reference nodes outside this skeleton.
            if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                neighbors[src].append(dst)
                neighbors[dst].append(src)

    # Expected visibility for each invisible node: mean over visible nodes i of
    # P(node k visible | node i visible) = column k restricted to visible rows.
    # NaN-guard: a co-visibility column may contain NaN if VisibilityModel ever
    # saw a node that was never visible; nanmean keeps the estimate finite and
    # treats such entries as "no evidence" rather than poisoning the mean.
    suspicious: list[int] = []
    p_expected_values: list[float] = []
    for k in invisible_indices:
        col = co_visibility_matrix[visible_indices, k]
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            # No usable evidence for this node; cannot be suspicious.
            p_expected = 0.0
        else:
            p_expected = float(np.mean(finite))

        p_expected_values.append(p_expected)

        if p_expected < threshold:
            continue

        if require_neighbors_visible:
            node_neighbors = neighbors[k]
            # A node with no neighbors cannot have "all neighbors visible" in a
            # meaningful sense; skip it under the stricter rule.
            if not node_neighbors:
                continue
            if not all(visibility_mask[n] for n in node_neighbors):
                continue

        suspicious.append(int(k))

    missing_node_score = float(np.max(p_expected_values)) if p_expected_values else 0.0
    # Clamp defensively in case of tiny floating-point overshoot above 1.0.
    missing_node_score = float(np.clip(missing_node_score, 0.0, 1.0))

    return {
        "missing_node_score": missing_node_score,
        "suspicious_nodes": sorted(suspicious),
        "n_suspicious": len(suspicious),
    }

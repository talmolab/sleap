"""Pose-split (chimera) features.

A *chimera* is a single labeled instance whose keypoints actually span two
different animals: e.g. the head/thorax of animal A connected to the abdomen of
animal B. This typically shows up as a skeleton that is internally consistent in
two tight clusters joined by a single, abnormally stretched "bridging" edge.

This module provides a pure function :func:`compute_pose_split` that scores how
chimera-like a single pose is. It is deliberately argument-driven (points,
adjacency, learned edge-length stats) so it can be unit-tested in isolation and
so the detector's already-computed baseline statistics can be reused.

The primary signal is graph-based:

1. On the subgraph induced by the *visible* nodes, find the connected bridging
   edge whose normalized length z-score ``z = (len - mean) / std`` is largest.
2. Cut that edge. If the visible subgraph splits cleanly into two components,
   measure how *balanced* the split is (``split_ratio = smaller / total``).
3. Measure how far apart the two clusters sit relative to their own internal
   spread (``gap_ratio``). A real chimera has two compact clusters separated by
   a wide gap.

A high ``split_score`` requires *all three* (large bridging z, balanced split,
large gap). This gating is what keeps it from firing on a normal pose or on a
partially-occluded single animal (whose visible subgraph is merely disconnected
without an abnormally long bridging edge).

For skeletons where the graph signal is weak or unavailable (star skeletons,
very short skeletons, or missing adjacency), a skeleton-agnostic 2-means
bimodality :func:`compute_pose_split_fallback` is used instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Minimum number of visible nodes required for a meaningful split analysis.
# With fewer points a "split" is not distinguishable from a normal small pose.
MIN_VISIBLE_NODES = 4

# Floors to keep ratios finite/stable.
_EPS = 1e-9


def _visible_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean (N_nodes,) mask of visible (non-NaN) nodes."""
    return ~np.isnan(points).any(axis=1)


def _cluster_spread(points: np.ndarray, indices: np.ndarray) -> float:
    """Mean distance of a cluster's points to their centroid.

    Args:
        points: (N_nodes, 2) coordinate array (may contain NaN).
        indices: 1-D array of node indices belonging to the cluster.

    Returns:
        Mean radial spread of the cluster. ``0.0`` for a single-point cluster.
    """
    pts = points[indices]
    if len(pts) < 2:
        return 0.0
    centroid = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - centroid, axis=1)))


def _components_after_cut(
    nodes: list[int],
    edges: list[tuple[int, int]],
    cut_edge: tuple[int, int],
) -> list[set[int]]:
    """Connected components of a node set after removing one edge.

    A tiny union-find over the visible subgraph (so we don't require networkx
    here and stay dependency-light, matching the pure-function contract).

    Args:
        nodes: Node indices present in the subgraph.
        edges: Undirected edges (i, j) within the subgraph.
        cut_edge: The edge to remove, as a sorted (i, j) tuple.

    Returns:
        List of components, each a set of node indices.
    """
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        if tuple(sorted((i, j))) == cut_edge:
            continue
        union(i, j)

    groups: dict[int, set[int]] = {}
    for n in nodes:
        groups.setdefault(find(n), set()).add(n)
    return list(groups.values())


def compute_pose_split_fallback(
    points: np.ndarray,
    min_visible: int = MIN_VISIBLE_NODES,
) -> dict[str, float]:
    """Skeleton-agnostic 2-means bimodality fallback.

    Used for star/short skeletons or when adjacency is unavailable. Splits the
    visible nodes into two clusters with 2-means and reports a silhouette-like
    separation score. This intentionally ignores skeleton edges, so it cannot
    use a bridging-edge z-score; the gate therefore relies entirely on geometry
    (balanced split + wide gap).

    Args:
        points: (N_nodes, 2) coordinate array (NaN for invisible nodes).
        min_visible: Minimum visible nodes required to attempt a split.

    Returns:
        Dictionary with keys ``split_score``, ``split_ratio``, ``gap_ratio``
        (all floats, ``split_score >= 0``).
    """
    zero = {"split_score": 0.0, "split_ratio": 0.0, "gap_ratio": 0.0}

    vis = _visible_mask(points)
    vis_idx = np.where(vis)[0]
    n_vis = len(vis_idx)
    if n_vis < min_visible:
        return dict(zero)

    coords = points[vis_idx]

    # 2-means via KMeans (deterministic seeding for test stability).
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(coords)
    except Exception:
        return dict(zero)

    idx_a = vis_idx[labels == 0]
    idx_b = vis_idx[labels == 1]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return dict(zero)

    n_small = min(len(idx_a), len(idx_b))
    split_ratio = n_small / n_vis

    pts_a = points[idx_a]
    pts_b = points[idx_b]

    # Bimodality via a boundary-gap test rather than a centroid-distance test.
    # A *uniformly* spread pose (e.g. an evenly spaced line) trivially splits in
    # half with a large centroid distance, so centroid distance alone gives
    # false positives. Instead compare the gap *across the cluster boundary*
    # (nearest pair of points from opposite clusters) to the typical *within*
    # cluster spacing. For two genuine blobs the boundary gap dominates; for a
    # uniform spread they are comparable.
    cross = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
    boundary_gap = float(cross.min())

    within = _typical_within_spacing(pts_a, pts_b)
    gap_ratio = boundary_gap / max(within, _EPS)

    # Silhouette-like separation: how much closer points are to their own
    # cluster than to the other. ~1 for well-separated blobs, ~0 for a uniform
    # spread that was split arbitrarily.
    silhouette = _silhouette_like(pts_a, pts_b)

    split_score = _combine_score(
        bridge_z=gap_ratio,  # No edge z available; use gap as the strength term.
        split_ratio=split_ratio,
        gap_ratio=gap_ratio,
        is_fallback=True,
        silhouette=silhouette,
    )

    return {
        "split_score": float(split_score),
        "split_ratio": float(split_ratio),
        "gap_ratio": float(gap_ratio),
    }


def _typical_within_spacing(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Typical nearest-neighbor spacing within clusters.

    For each cluster with >= 2 points, take each point's distance to its
    nearest same-cluster neighbor; the typical (median) of those is the
    "within" scale. Used to normalize the boundary gap.

    Args:
        pts_a: (Na, 2) points of cluster A.
        pts_b: (Nb, 2) points of cluster B.

    Returns:
        Median within-cluster nearest-neighbor distance (>= 0).
    """
    nn_dists: list[float] = []
    for pts in (pts_a, pts_b):
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn_dists.extend(d.min(axis=1).tolist())
    if not nn_dists:
        return 0.0
    return float(np.median(nn_dists))


def _silhouette_like(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Mean silhouette coefficient for a 2-cluster split (simplified).

    For each point, ``s = (b - a) / max(a, b)`` where ``a`` is the mean
    distance to other points in its own cluster and ``b`` is the mean distance
    to the other cluster. Ranges in [-1, 1]; ~1 means tight, well-separated
    clusters; ~0 means the split is arbitrary (uniform data).

    Args:
        pts_a: (Na, 2) points of cluster A.
        pts_b: (Nb, 2) points of cluster B.

    Returns:
        Mean silhouette coefficient over all points.
    """
    scores: list[float] = []
    for own, other in ((pts_a, pts_b), (pts_b, pts_a)):
        for k in range(len(own)):
            p = own[k]
            if len(own) > 1:
                a = float(np.sum(np.linalg.norm(own - p, axis=1)) / (len(own) - 1))
            else:
                a = 0.0
            b = float(np.mean(np.linalg.norm(other - p, axis=1)))
            denom = max(a, b)
            scores.append((b - a) / denom if denom > _EPS else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _balance_factor(split_ratio: float) -> float:
    """Map split_ratio to a 0..1 balance weight peaking in [0.3, 0.5].

    A genuine chimera divides the keypoints into two non-trivial groups. A
    ratio near 0.5 is perfectly balanced; near 0 means one stray node (the
    occlusion / mislabel-of-a-single-node case, which we do *not* want to flag
    as a chimera). The factor is 0 below ~0.18 and ramps to 1 by ~0.3.

    Args:
        split_ratio: smaller_cluster_size / total_visible (in [0, 0.5]).

    Returns:
        Weight in [0, 1].
    """
    lo, hi = 0.18, 0.30
    if split_ratio <= lo:
        return 0.0
    if split_ratio >= hi:
        return 1.0
    return (split_ratio - lo) / (hi - lo)


# Minimum mean-silhouette for the fallback to treat a split as truly bimodal.
# A uniformly spread pose split in half scores well below this.
_FALLBACK_SILHOUETTE_FLOOR = 0.5


def _combine_score(
    bridge_z: float,
    split_ratio: float,
    gap_ratio: float,
    is_fallback: bool = False,
    silhouette: float = 1.0,
) -> float:
    """Combine the sub-signals into a gated split score.

    The score is multiplicative so that *all* conditions must hold: a large
    bridging stretch, a balanced split, and a wide gap. Any one being small
    drives the score toward zero.

    Args:
        bridge_z: Normalized stretch of the bridging edge (or boundary-gap ratio
            in the fallback path). Clamped at 0.
        split_ratio: smaller / total visible nodes.
        gap_ratio: graph path: inter-cluster distance / max intra-cluster
            spread. Fallback path: boundary gap / within-cluster spacing.
        is_fallback: Whether this is the geometry-only fallback path.
        silhouette: Mean silhouette coefficient (fallback only); used to gate
            out arbitrary splits of uniformly spread points.

    Returns:
        Non-negative split score.
    """
    bridge_term = max(bridge_z, 0.0)
    balance = _balance_factor(split_ratio)

    # Gap must clear a floor before it contributes (two clusters whose gap is
    # comparable to their own internal scale are just one diffuse animal).
    gap_floor = 1.5
    gap_term = max(gap_ratio - gap_floor, 0.0)

    if is_fallback:
        # Geometry-only: additionally require a genuinely bimodal split so we do
        # not fire on uniformly spread points (which 2-means always halves with
        # a large centroid gap but a near-zero silhouette).
        if silhouette < _FALLBACK_SILHOUETTE_FLOOR:
            return 0.0
        return float(balance * gap_term)

    return float(bridge_term * balance * gap_term)


def compute_pose_split(
    points: np.ndarray,
    adjacency: Optional[dict[int, list[int]]],
    edge_means: dict[tuple[int, int], float],
    edge_stds: dict[tuple[int, int], float],
    min_visible: int = MIN_VISIBLE_NODES,
) -> dict[str, float]:
    """Score how *chimera-like* a single pose is (one instance, two animals).

    Operates on the subgraph induced by the visible nodes:

    1. **Bridging edge.** Among visible edges with learned length stats, find
       the one with the largest normalized stretch ``z = (len - mean) / std``.
    2. **Balanced split.** Remove that edge; if the visible subgraph splits into
       exactly two components, ``split_ratio = smaller / total`` measures
       balance (ideal ~0.3-0.5).
    3. **Gap.** ``gap_ratio = ||centroidA - centroidB|| / max(spread_A, spread_B)``
       measures how far the clusters sit relative to their own internal spread.

    The returned ``split_score`` is the *gated product* of these signals, so it
    is only high when a long bridging edge separates two balanced, well-spaced
    clusters. This is what prevents false positives on:

    - a **normal pose** (no bridging edge with a large z; small gap_ratio),
    - a **partially occluded** single animal (the visible subgraph may be
      disconnected, but there is no abnormally stretched bridging edge joining
      two balanced clusters), and
    - two genuinely close, correctly-merged animals (small gap relative to
      spread).

    If adjacency is unavailable, or the skeleton is too star-like / short for a
    bridging edge to be meaningful, falls back to
    :func:`compute_pose_split_fallback` (2-means bimodality).

    Args:
        points: (N_nodes, 2) coordinate array (NaN for invisible nodes).
        adjacency: Maps node_index -> list of neighbor indices (skeleton
            edges). Pass ``SkeletonAnalyzer.get_adjacency()``. If ``None``, the
            geometry-only fallback is used.
        edge_means: Maps sorted ``(i, j)`` edge tuples -> learned mean edge
            length (reuse ``DatasetStats.edge_means``).
        edge_stds: Maps sorted ``(i, j)`` edge tuples -> learned edge-length std
            (reuse ``DatasetStats.edge_stds``). Values should be floored away
            from zero by the caller (the baseline extractor already does this).
        min_visible: Minimum visible nodes required to attempt a split.

    Returns:
        Dictionary with:

        - ``split_score``: non-negative chimera score (0 = not a chimera; the
          integration layer thresholds this, e.g. on the training distribution).
        - ``split_ratio``: smaller-cluster fraction of visible nodes (0..0.5).
        - ``gap_ratio``: inter-cluster distance / max intra-cluster spread.
    """
    zero = {"split_score": 0.0, "split_ratio": 0.0, "gap_ratio": 0.0}

    vis = _visible_mask(points)
    vis_idx = set(np.where(vis)[0].tolist())
    n_vis = len(vis_idx)

    # Too few visible nodes -> not analyzable.
    if n_vis < min_visible:
        return dict(zero)

    # No adjacency -> geometry-only fallback.
    if not adjacency:
        return compute_pose_split_fallback(points, min_visible=min_visible)

    # Build the visible subgraph's edge list with z-scores, deduplicating
    # undirected edges via sorted tuples.
    visible_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    best_z = -np.inf
    best_edge: Optional[tuple[int, int]] = None

    for node, neighbors in adjacency.items():
        if node not in vis_idx:
            continue
        for nb in neighbors:
            if nb not in vis_idx:
                continue
            key = tuple(sorted((int(node), int(nb))))
            if key in seen:
                continue
            seen.add(key)
            visible_edges.append(key)

            mean = edge_means.get(key)
            std = edge_stds.get(key)
            if mean is None or std is None:
                continue
            length = float(np.linalg.norm(points[key[1]] - points[key[0]]))
            z = (length - mean) / max(std, _EPS)
            if z > best_z:
                best_z = z
                best_edge = key

    # Decide whether the graph signal is usable. A star skeleton (a hub with
    # many leaves) cannot be split into two balanced clusters by cutting one
    # edge, and very few internal edges also make the bridging test unreliable.
    # In those cases, defer to the geometry-only fallback.
    if (
        best_edge is None
        or len(visible_edges) < 2
        or _is_star_like(visible_edges, vis_idx)
    ):
        return compute_pose_split_fallback(points, min_visible=min_visible)

    nodes_list = sorted(vis_idx)
    components = _components_after_cut(nodes_list, visible_edges, best_edge)

    # We require the cut to produce exactly two components (a clean bridge). If
    # the subgraph was already disconnected (occlusion) the cut yields >2
    # components; if the edge was redundant (a cycle) it yields 1. Either way it
    # is not a clean chimera bridge.
    if len(components) != 2:
        return dict(zero)

    comp_a, comp_b = components
    idx_a = np.array(sorted(comp_a))
    idx_b = np.array(sorted(comp_b))

    n_small = min(len(idx_a), len(idx_b))
    split_ratio = n_small / n_vis

    centroid_a = points[idx_a].mean(axis=0)
    centroid_b = points[idx_b].mean(axis=0)
    gap = float(np.linalg.norm(centroid_a - centroid_b))

    spread_a = _cluster_spread(points, idx_a)
    spread_b = _cluster_spread(points, idx_b)
    max_spread = max(spread_a, spread_b)
    gap_ratio = gap / max(max_spread, _EPS)

    split_score = _combine_score(
        bridge_z=best_z,
        split_ratio=split_ratio,
        gap_ratio=gap_ratio,
        is_fallback=False,
    )

    return {
        "split_score": float(split_score),
        "split_ratio": float(split_ratio),
        "gap_ratio": float(gap_ratio),
    }


def _is_star_like(edges: list[tuple[int, int]], vis_idx: set[int]) -> bool:
    """Heuristic: is the visible subgraph a star (one hub, many leaves)?

    A star cannot be split into two balanced clusters by removing a single
    edge, so the bridging-edge test is meaningless and we should fall back to
    geometry. We call the subgraph star-like when a single node touches almost
    all edges.

    Args:
        edges: Undirected edges within the visible subgraph.
        vis_idx: Set of visible node indices.

    Returns:
        True if the subgraph looks like a star.
    """
    if len(vis_idx) < 4 or len(edges) == 0:
        return False
    degree: dict[int, int] = {}
    for i, j in edges:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    max_deg = max(degree.values())
    # If one hub touches (n_visible - 1) edges, it's a pure star. Allow one
    # extra edge of slack for near-stars.
    return max_deg >= (len(vis_idx) - 1) and max_deg >= len(edges) - 1

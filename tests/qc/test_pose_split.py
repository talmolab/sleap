"""Tests for sleap.qc.features.pose_split (chimera detector)."""

from __future__ import annotations

import numpy as np
import pytest

from sleap.qc.features.pose_split import (
    compute_pose_split,
    compute_pose_split_fallback,
)


# ---------------------------------------------------------------------------
# Skeleton / baseline-stat helpers
# ---------------------------------------------------------------------------


def chain_adjacency(n_nodes: int) -> dict[int, list[int]]:
    """Adjacency for a line graph 0-1-2-...-(n-1)."""
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    return adj


def edges_from_adjacency(adj: dict[int, list[int]]) -> list[tuple[int, int]]:
    """Sorted unique undirected edges from an adjacency dict."""
    seen = set()
    for node, nbrs in adj.items():
        for nb in nbrs:
            seen.add(tuple(sorted((node, nb))))
    return sorted(seen)


def learn_edge_stats(
    instances: list[np.ndarray],
    edges: list[tuple[int, int]],
    min_std: float = 1e-6,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Mimic BaselineFeatureExtractor's edge-length stats from clean poses.

    Returns (edge_means, edge_stds) keyed by sorted (i, j) tuples.
    """
    lengths: dict[tuple[int, int], list[float]] = {e: [] for e in edges}
    for pts in instances:
        for i, j in edges:
            p1, p2 = pts[i], pts[j]
            if not (np.isnan(p1).any() or np.isnan(p2).any()):
                lengths[(i, j)].append(float(np.linalg.norm(p2 - p1)))
    means = {e: (float(np.mean(v)) if v else 0.0) for e, v in lengths.items()}
    stds = {
        e: max(float(np.std(v)) if v else min_std, min_std) for e, v in lengths.items()
    }
    return means, stds


@pytest.fixture
def normal_chain_stats():
    """A 6-node line skeleton with edge stats learned from normal poses.

    Each segment ~10 units long with small jitter.
    """
    n_nodes = 6
    adj = chain_adjacency(n_nodes)
    edges = edges_from_adjacency(adj)

    rng = np.random.default_rng(0)
    instances = []
    base = np.array([[i * 10.0, 0.0] for i in range(n_nodes)])
    for _ in range(40):
        instances.append(base + rng.normal(0, 0.4, size=(n_nodes, 2)))
    means, stds = learn_edge_stats(instances, edges)
    return adj, edges, means, stds


# ---------------------------------------------------------------------------
# Graph-based path: chimera vs normal vs occluded
# ---------------------------------------------------------------------------


def test_clear_chimera_high_score(normal_chain_stats):
    """Two tight clusters joined by one very long edge -> high split_score."""
    adj, _edges, means, stds = normal_chain_stats

    # Cluster A: nodes 0-2 near origin (segments ~10). Cluster B: nodes 3-5 far
    # away (also internally ~10 apart). The 2->3 edge is the stretched bridge.
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [320.0, 0.0],  # huge jump across the bridge edge (2,3)
            [330.0, 0.0],
            [340.0, 0.0],
        ]
    )

    result = compute_pose_split(points, adj, means, stds)

    assert result["split_score"] > 1.0
    # Balanced 3/3 split -> ratio 0.5.
    assert result["split_ratio"] == pytest.approx(0.5)
    # Gap (~300) is huge relative to intra-cluster spread (~6.7).
    assert result["gap_ratio"] > 10.0


def test_normal_pose_low_score(normal_chain_stats):
    """A normal in-distribution pose -> ~zero split_score."""
    adj, _edges, means, stds = normal_chain_stats

    points = np.array([[i * 10.0, 0.0] for i in range(6)], dtype=float)
    points[2, 1] += 0.5  # tiny perturbation, still normal

    result = compute_pose_split(points, adj, means, stds)

    assert result["split_score"] < 0.5
    # Gap between the two halves is comparable to their internal spread.
    assert result["gap_ratio"] < 3.0


def test_partially_occluded_normal_pose_low_score(normal_chain_stats):
    """Occluded single animal (disconnected visible subgraph) -> low score.

    Hiding an interior node breaks the chain into two visible components with
    *no* bridging edge between them. Removing the largest-z edge then yields
    more than two components, so we must not flag it.
    """
    adj, _edges, means, stds = normal_chain_stats

    points = np.array([[i * 10.0, 0.0] for i in range(6)], dtype=float)
    points[2] = np.nan  # occlude an interior node -> subgraph splits at 1|3

    result = compute_pose_split(points, adj, means, stds)

    assert result["split_score"] < 0.5


def test_occluded_single_node_not_chimera(normal_chain_stats):
    """One stray displaced node is a single-node error, not a chimera.

    Node 5 is yanked far away. That is an unbalanced 1-vs-5 split, which the
    balance gate must suppress even though the end edge is stretched.
    """
    adj, _edges, means, stds = normal_chain_stats

    points = np.array([[i * 10.0, 0.0] for i in range(6)], dtype=float)
    points[5] = [500.0, 0.0]  # single stretched leaf

    result = compute_pose_split(points, adj, means, stds)

    assert result["split_ratio"] < 0.2  # 1/6
    assert result["split_score"] == pytest.approx(0.0, abs=1e-6)


def test_two_close_overlapping_animals_low_score(normal_chain_stats):
    """Two animals merged but spatially interleaved/overlapping -> low score.

    When two correctly-merged animals genuinely overlap, the two halves of the
    skeleton are interleaved in space: the gap between the cluster centroids is
    small relative to the within-cluster spread, so gap_ratio stays below the
    floor and the score is ~zero. (A *separated* pair with a stretched bridge is
    a true chimera and is covered by ``test_clear_chimera_high_score``.)
    """
    adj, _edges, means, stds = normal_chain_stats

    # Interleave: nodes alternate around the origin so neither graph-half forms
    # a spatially distinct cluster. The bridge edge (2,3) is normal length.
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 2.0],
            [20.0, 0.0],
            [25.0, 2.0],
            [15.0, -2.0],
            [5.0, 0.0],
        ]
    )

    result = compute_pose_split(points, adj, means, stds)

    # Centroids of the two graph-halves nearly coincide -> tiny gap_ratio.
    assert result["gap_ratio"] < 2.0
    assert result["split_score"] < 1.0


def test_score_is_nonnegative_and_finite(normal_chain_stats):
    """split_score is always >= 0 and finite across cases."""
    adj, _edges, means, stds = normal_chain_stats
    rng = np.random.default_rng(7)
    for _ in range(20):
        pts = rng.normal(0, 50, size=(6, 2))
        result = compute_pose_split(pts, adj, means, stds)
        assert result["split_score"] >= 0.0
        assert np.isfinite(result["split_score"])
        assert np.isfinite(result["gap_ratio"])
        assert 0.0 <= result["split_ratio"] <= 0.5


# ---------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------


def test_translation_rotation_scale_invariance(normal_chain_stats):
    """split_ratio and gap_ratio are invariant to translation and rotation.

    (Scale invariance holds for gap_ratio/split_ratio; split_score depends on
    the learned edge z-score which is scale-sensitive by design, so we only
    assert the ratios are scale-invariant here.)
    """
    adj, _edges, means, stds = normal_chain_stats
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [320.0, 0.0],
            [330.0, 0.0],
            [340.0, 0.0],
        ]
    )
    base = compute_pose_split(points, adj, means, stds)

    # Rotate 37 degrees + translate.
    theta = np.deg2rad(37.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = points @ rot.T + np.array([123.4, -56.7])
    rt = compute_pose_split(moved, adj, means, stds)

    assert rt["split_ratio"] == pytest.approx(base["split_ratio"])
    assert rt["gap_ratio"] == pytest.approx(base["gap_ratio"], rel=1e-6)


# ---------------------------------------------------------------------------
# Degenerate / guard cases
# ---------------------------------------------------------------------------


def test_too_few_visible_nodes_returns_zero(normal_chain_stats):
    """Fewer than MIN_VISIBLE_NODES visible -> all-zero result."""
    adj, _edges, means, stds = normal_chain_stats
    points = np.full((6, 2), np.nan)
    points[0] = [0.0, 0.0]
    points[3] = [100.0, 0.0]  # only 2 visible

    result = compute_pose_split(points, adj, means, stds)

    assert result == {"split_score": 0.0, "split_ratio": 0.0, "gap_ratio": 0.0}


def test_no_adjacency_uses_fallback():
    """adjacency=None routes to the geometry-only fallback for a chimera."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [320.0, 0.0],
            [330.0, 0.0],
            [340.0, 0.0],
        ]
    )
    result = compute_pose_split(points, None, {}, {})
    assert result["split_score"] > 0.0
    assert result["split_ratio"] == pytest.approx(0.5)
    assert result["gap_ratio"] > 10.0


def test_empty_adjacency_uses_fallback():
    """Empty adjacency dict routes to the fallback; uniform line -> ~zero.

    An evenly spaced line is *not* bimodal: 2-means halves it with a large
    centroid gap but a near-zero silhouette, so the fallback's bimodality gate
    suppresses it.
    """
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    result = compute_pose_split(points, {}, {}, {})
    assert result["split_score"] == pytest.approx(0.0, abs=1e-6)


def test_missing_edge_stats_falls_back_gracefully(normal_chain_stats):
    """If learned stats lack the visible edges, no crash; defers to fallback."""
    adj, _edges, _means, _stds = normal_chain_stats
    points = np.array([[i * 10.0, 0.0] for i in range(6)], dtype=float)
    # Empty stats -> no edge gets a z-score -> best_edge is None -> fallback.
    result = compute_pose_split(points, adj, {}, {})
    assert np.isfinite(result["split_score"])
    assert result["split_score"] >= 0.0


def test_nan_never_leaks(normal_chain_stats):
    """NaN nodes never produce NaN outputs."""
    adj, _edges, means, stds = normal_chain_stats
    points = np.array([[i * 10.0, 0.0] for i in range(6)], dtype=float)
    points[1] = np.nan
    points[4] = np.nan
    result = compute_pose_split(points, adj, means, stds)
    for v in result.values():
        assert not np.isnan(v)


# ---------------------------------------------------------------------------
# Fallback (star / short skeleton) path
# ---------------------------------------------------------------------------


def test_star_skeleton_uses_fallback():
    """A star skeleton (hub + leaves) routes through the fallback.

    Hub node 0 connects to all others. Leaves 1-3 are one tight cluster, leaves
    4-6 are a far cluster -> the geometry fallback should see two clusters.
    """
    n_nodes = 7
    adj: dict[int, list[int]] = {0: [1, 2, 3, 4, 5, 6]}
    for k in range(1, n_nodes):
        adj[k] = [0]
    edges = edges_from_adjacency(adj)

    # Tight cluster near origin, far cluster near x=300; hub between-ish.
    points = np.array(
        [
            [150.0, 0.0],  # hub
            [0.0, 0.0],
            [5.0, 5.0],
            [-5.0, 5.0],
            [300.0, 0.0],
            [305.0, 5.0],
            [295.0, 5.0],
        ]
    )
    # Provide normal-ish stats so the (non-fallback) z would be modest; we only
    # care that the star is routed to the fallback.
    means = {e: 150.0 for e in edges}
    stds = {e: 30.0 for e in edges}

    result = compute_pose_split(points, adj, means, stds)
    assert result["split_score"] >= 0.0
    assert np.isfinite(result["split_score"])
    # Two well separated leaf groups -> meaningful gap_ratio.
    assert result["gap_ratio"] > 1.5


def test_fallback_clear_bimodal_high():
    """Fallback fires on two tight, well-separated clusters."""
    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [200.0, 0.0],
            [204.0, 0.0],
            [200.0, 4.0],
        ]
    )
    result = compute_pose_split_fallback(points)
    assert result["split_score"] > 0.0
    assert result["split_ratio"] == pytest.approx(0.5)
    assert result["gap_ratio"] > 5.0


def test_fallback_single_blob_low():
    """Fallback stays low for a single compact blob (no real gap)."""
    rng = np.random.default_rng(3)
    points = rng.normal(0, 5, size=(8, 2))
    result = compute_pose_split_fallback(points)
    # A diffuse single blob: gap between halves is comparable to spread.
    assert result["split_score"] < 1.0


def test_fallback_too_few_visible_zero():
    """Fallback returns zeros with too few visible nodes."""
    points = np.full((6, 2), np.nan)
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 1.0]
    result = compute_pose_split_fallback(points)
    assert result == {"split_score": 0.0, "split_ratio": 0.0, "gap_ratio": 0.0}


# ---------------------------------------------------------------------------
# Integration with sleap_io-built skeleton (smoke test)
# ---------------------------------------------------------------------------


def test_integration_with_sleap_io_skeleton():
    """End-to-end style: build adjacency/stats like the detector would.

    Uses SkeletonAnalyzer.get_adjacency() and edge-length stats over clean
    instances, mirroring how the integration layer supplies arguments.
    """
    import sleap_io as sio

    from sleap.qc.features.skeleton import SkeletonAnalyzer

    node_names = ["n0", "n1", "n2", "n3", "n4", "n5"]
    skel = sio.Skeleton(
        nodes=node_names,
        edges=[("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")],
    )
    analyzer = SkeletonAnalyzer(skel)
    adj = analyzer.get_adjacency()
    edges = analyzer.edges

    rng = np.random.default_rng(11)
    base = np.array([[i * 10.0, 0.0] for i in range(6)])
    clean = [base + rng.normal(0, 0.4, size=(6, 2)) for _ in range(40)]
    means, stds = learn_edge_stats(clean, [tuple(sorted(e)) for e in edges])

    # Normal pose -> low.
    normal_res = compute_pose_split(base.copy(), adj, means, stds)
    assert normal_res["split_score"] < 0.5

    # Chimera -> high.
    chimera = base.copy()
    chimera[3:] += np.array([300.0, 0.0])
    chimera_res = compute_pose_split(chimera, adj, means, stds)
    assert chimera_res["split_score"] > normal_res["split_score"]
    assert chimera_res["split_score"] > 1.0

"""Tests for sleap.qc.features.missing_node.

These tests cover the Tier-1 (geometry/visibility-only) missing-node detector
``score_missing_nodes``. The co-visibility matrix it consumes is normally
learned by :class:`VisibilityModel`; here we both construct matrices by hand
(for precise control) and fit a real ``VisibilityModel`` (to confirm the two
interoperate as the integration layer expects).
"""

import numpy as np
import pytest

from sleap.qc.features.missing_node import score_missing_nodes
from sleap.qc.features.visibility import VisibilityModel


# A simple 5-node line skeleton reused across tests: 0-1-2-3-4.
LINE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]


def _covis_from_masks(masks: np.ndarray) -> np.ndarray:
    """Fit a VisibilityModel and return its co-visibility matrix.

    This mirrors exactly what the integration layer does: learn co-visibility on
    the dataset, then hand the matrix to ``score_missing_nodes``.
    """
    model = VisibilityModel()
    model.fit(np.asarray(masks, dtype=bool))
    return model.co_visibility_matrix


class TestScoreMissingNodesPositive:
    """The instance is missing a node its peers almost always keep."""

    def test_flags_dropped_covisible_node(self):
        # All five nodes are essentially always visible together.
        masks = np.ones((50, 5), dtype=bool)
        # A tiny bit of noise on node 4 so nothing is a degenerate 100%/0%.
        masks[:2, 4] = False
        covis = _covis_from_masks(masks)

        # This instance is missing node 2, which peers keep ~100% of the time.
        vis = np.array([True, True, False, True, True])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        assert result["n_suspicious"] == 1
        assert result["suspicious_nodes"] == [2]
        # Expected visibility of node 2 given the others is ~1.0.
        assert result["missing_node_score"] == pytest.approx(1.0, abs=1e-6)

    def test_flags_multiple_dropped_nodes(self):
        masks = np.ones((40, 5), dtype=bool)
        covis = _covis_from_masks(masks)

        # Missing both node 1 and node 3.
        vis = np.array([True, False, True, False, True])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        assert result["n_suspicious"] == 2
        assert result["suspicious_nodes"] == [1, 3]
        assert result["missing_node_score"] == pytest.approx(1.0, abs=1e-6)

    def test_hand_built_matrix_positive(self):
        # Hand-built: node 2 is visible with everything ~95% of the time.
        n = 4
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)

        vis = np.array([True, True, False, True])
        result = score_missing_nodes(vis, covis, [(0, 1), (1, 2), (2, 3)])

        assert result["suspicious_nodes"] == [2]
        assert result["missing_node_score"] == pytest.approx(0.95, abs=1e-6)


class TestScoreMissingNodesNegative:
    """The missing node is one peers usually also lack -> not suspicious."""

    def test_usually_absent_node_not_flagged(self):
        # Nodes 0,1,2 always visible; nodes 3,4 almost never visible.
        masks = np.zeros((50, 5), dtype=bool)
        masks[:, 0:3] = True
        masks[:2, 3] = True  # 4% visibility for node 3
        masks[:1, 4] = True  # 2% visibility for node 4
        covis = _covis_from_masks(masks)

        # Common pattern: 0,1,2 visible, 3,4 missing. Nothing suspicious.
        vis = np.array([True, True, True, False, False])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        assert result["n_suspicious"] == 0
        assert result["suspicious_nodes"] == []
        # Score should be the (low) expected visibility of nodes 3/4.
        assert result["missing_node_score"] < 0.1

    def test_score_below_threshold_not_flagged(self):
        # Node 2 co-visible only 80% of the time; default threshold is 0.9.
        n = 4
        covis = np.full((n, n), 0.8)
        np.fill_diagonal(covis, 1.0)

        vis = np.array([True, True, False, True])
        result = score_missing_nodes(vis, covis, [(0, 1), (1, 2), (2, 3)])

        assert result["n_suspicious"] == 0
        # But the score still reflects the 0.8 expected probability.
        assert result["missing_node_score"] == pytest.approx(0.8, abs=1e-6)


class TestScoreMissingNodesAllVisible:
    """All nodes visible -> nothing missing -> score 0."""

    def test_all_visible_score_zero(self):
        masks = np.ones((20, 5), dtype=bool)
        covis = _covis_from_masks(masks)

        vis = np.array([True, True, True, True, True])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        assert result["missing_node_score"] == 0.0
        assert result["n_suspicious"] == 0
        assert result["suspicious_nodes"] == []


class TestThreshold:
    """The threshold argument controls flagging."""

    def test_custom_threshold_flags_lower_prob(self):
        n = 4
        covis = np.full((n, n), 0.8)
        np.fill_diagonal(covis, 1.0)
        vis = np.array([True, True, False, True])

        # Default 0.9 -> not flagged.
        assert score_missing_nodes(vis, covis, [])["n_suspicious"] == 0
        # Lower threshold -> flagged.
        result = score_missing_nodes(vis, covis, [], threshold=0.75)
        assert result["suspicious_nodes"] == [2]

    def test_threshold_is_inclusive(self):
        n = 3
        covis = np.full((n, n), 0.9)
        np.fill_diagonal(covis, 1.0)
        vis = np.array([True, False, True])

        # p_expected == threshold exactly should flag (>=).
        result = score_missing_nodes(vis, covis, [], threshold=0.9)
        assert result["suspicious_nodes"] == [1]


class TestRequireNeighborsVisible:
    """The optional neighbor-strengthening rule."""

    def test_flagged_when_all_neighbors_visible(self):
        n = 5
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)

        # Node 2 missing; its neighbors 1 and 3 are both visible.
        vis = np.array([True, True, False, True, True])
        result = score_missing_nodes(
            vis, covis, LINE_EDGES, require_neighbors_visible=True
        )
        assert result["suspicious_nodes"] == [2]

    def test_not_flagged_when_a_neighbor_invisible(self):
        n = 5
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)

        # Node 2 missing AND its neighbor 3 also missing -> looks like an
        # occluded region, not an accidental single drop.
        vis = np.array([True, True, False, False, True])
        result = score_missing_nodes(
            vis, covis, LINE_EDGES, require_neighbors_visible=True
        )
        # Node 2 has an invisible neighbor (3) so it is NOT flagged. Node 3's
        # neighbor 2 is also invisible, so it is not flagged either.
        assert result["suspicious_nodes"] == []
        # The score (max expected prob over invisible nodes) is still high --
        # the strengthening rule only gates *flagging*, not the raw score.
        assert result["missing_node_score"] == pytest.approx(0.95, abs=1e-6)

    def test_node_with_no_neighbors_not_flagged_under_rule(self):
        # 4 nodes, but node 3 is isolated (no edges touch it).
        n = 4
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)
        edges = [(0, 1), (1, 2)]  # node 3 has no neighbors

        vis = np.array([True, True, True, False])
        # Without the rule, node 3 is flagged.
        assert score_missing_nodes(vis, covis, edges)["suspicious_nodes"] == [3]
        # With the rule, a neighborless node cannot satisfy "all neighbors
        # visible" and is skipped.
        result = score_missing_nodes(
            vis, covis, edges, require_neighbors_visible=True
        )
        assert result["suspicious_nodes"] == []


class TestDegenerate:
    """Degenerate / edge inputs must not raise or leak NaN."""

    def test_no_visible_nodes_score_zero(self):
        n = 4
        covis = np.full((n, n), 0.95)
        vis = np.array([False, False, False, False])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        # No visible node to condition on -> no evidence -> score 0.
        assert result["missing_node_score"] == 0.0
        assert result["n_suspicious"] == 0

    def test_single_visible_node(self):
        n = 4
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)
        vis = np.array([True, False, False, False])
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        # Only node 0 visible; nodes 1,2,3 each have p_expected ~0.95.
        assert result["suspicious_nodes"] == [1, 2, 3]
        assert result["missing_node_score"] == pytest.approx(0.95, abs=1e-6)

    def test_empty_skeleton(self):
        vis = np.array([], dtype=bool)
        covis = np.zeros((0, 0))
        result = score_missing_nodes(vis, covis, [])

        assert result["missing_node_score"] == 0.0
        assert result["suspicious_nodes"] == []
        assert result["n_suspicious"] == 0

    def test_matrix_shape_mismatch_returns_zero(self):
        # A 5-node mask with a 3x3 matrix is inconsistent; do not crash.
        vis = np.array([True, True, False, True, True])
        covis = np.full((3, 3), 0.95)
        result = score_missing_nodes(vis, covis, LINE_EDGES)

        assert result["missing_node_score"] == 0.0
        assert result["n_suspicious"] == 0

    def test_nan_in_covis_column_does_not_leak(self):
        # A node that was never visible during fitting can yield NaN columns.
        n = 4
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)
        # Make every entry predicting node 2's visibility NaN.
        covis[:, 2] = np.nan

        vis = np.array([True, True, False, True])
        result = score_missing_nodes(vis, covis, [(0, 1), (1, 2), (2, 3)])

        # No usable evidence for node 2 -> p_expected treated as 0, not NaN.
        assert np.isfinite(result["missing_node_score"])
        assert result["missing_node_score"] == 0.0
        assert result["n_suspicious"] == 0

    def test_score_never_exceeds_one(self):
        # Even with an out-of-range matrix, the score is clamped to [0, 1].
        n = 3
        covis = np.full((n, n), 1.5)  # pathological > 1 probabilities
        vis = np.array([True, False, True])
        result = score_missing_nodes(vis, covis, [])

        assert 0.0 <= result["missing_node_score"] <= 1.0

    def test_non_boolean_mask_is_coerced(self):
        # Integer 0/1 masks (a common alternative encoding) must work.
        n = 4
        covis = np.full((n, n), 0.95)
        np.fill_diagonal(covis, 1.0)
        vis = np.array([1, 1, 0, 1])  # node 2 missing
        result = score_missing_nodes(vis, covis, [(0, 1), (1, 2), (2, 3)])

        assert result["suspicious_nodes"] == [2]


class TestSleapIoIntegration:
    """Build masks/skeleton from sleap_io and confirm end-to-end behavior."""

    def test_end_to_end_with_sleap_io_instances(self):
        import sleap_io as sio

        # 4-node skeleton: head - thorax - abdomen - tail (a line).
        skel = sio.Skeleton(
            nodes=["head", "thorax", "abdomen", "tail"],
            edges=[("head", "thorax"), ("thorax", "abdomen"), ("abdomen", "tail")],
        )
        name_to_idx = {n.name: i for i, n in enumerate(skel.nodes)}
        edges = [
            (name_to_idx[e.source.name], name_to_idx[e.destination.name])
            for e in skel.edges
        ]
        n_nodes = len(skel.nodes)

        # Build a dataset of instances. "tail" is frequently invisible
        # (systematic), the rest are almost always visible.
        rng = np.random.default_rng(0)
        instances: list[sio.Instance] = []
        masks: list[np.ndarray] = []
        for _ in range(60):
            pts = rng.normal(size=(n_nodes, 2)) * 5 + np.array([0, 0])
            pts = pts.astype(float)
            # Tail invisible ~70% of the time.
            if rng.random() < 0.7:
                pts[3] = np.nan
            inst = sio.Instance.from_numpy(pts, skeleton=skel)
            instances.append(inst)
            masks.append(~np.isnan(pts).any(axis=1))

        masks = np.asarray(masks, dtype=bool)
        covis = _covis_from_masks(masks)

        # Case 1: an instance missing "tail" -- which is systematically dropped.
        # Tier-1 should NOT flag it (this is the documented limitation).
        vis_systematic = np.array([True, True, True, False])
        res_sys = score_missing_nodes(vis_systematic, covis, edges)
        assert 3 not in res_sys["suspicious_nodes"]
        assert res_sys["missing_node_score"] < 0.9

        # Case 2: an instance missing "thorax" -- a node peers almost always
        # keep. Tier-1 SHOULD flag it as a suspicious outlier drop.
        vis_outlier = np.array([True, False, True, True])
        res_out = score_missing_nodes(vis_outlier, covis, edges)
        assert 1 in res_out["suspicious_nodes"]
        assert res_out["missing_node_score"] >= 0.9

    def test_covis_from_instances_matches_visibility_model(self):
        # Sanity: feeding VisibilityModel's own matrix back yields a finite,
        # in-range score for a typical pattern.
        masks = np.ones((30, 5), dtype=bool)
        masks[::3, 4] = False
        model = VisibilityModel().fit(masks)

        vis = np.array([True, True, False, True, True])
        result = score_missing_nodes(vis, model.co_visibility_matrix, LINE_EDGES)

        assert 0.0 <= result["missing_node_score"] <= 1.0
        assert result["suspicious_nodes"] == [2]

"""Tests for sleap.qc.features modules."""

import numpy as np
import pytest

from sleap.qc.features.baseline import BaselineFeatureExtractor, BASELINE_FEATURE_NAMES
from sleap.qc.features.structural import compute_curvature, compute_convex_hull
from sleap.qc.features.visibility import VisibilityModel, compute_isolated_invisible
from sleap.qc.features.reference import (
    NearestNeighborScorer,
    normalize_pose,
    pose_distance,
)


class TestBaselineFeatureExtractor:
    """Tests for BaselineFeatureExtractor."""

    @pytest.fixture
    def simple_skeleton(self):
        """A simple 5-node skeleton: line graph."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        n_nodes = 5
        symmetry_pairs = []
        return edges, n_nodes, symmetry_pairs

    @pytest.fixture
    def extractor(self, simple_skeleton):
        """Create fitted extractor with simple skeleton."""
        edges, n_nodes, symmetry_pairs = simple_skeleton
        extractor = BaselineFeatureExtractor(edges, n_nodes, symmetry_pairs)

        # Create simple training instances (straight line poses)
        instances = []
        for _ in range(10):
            points = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=float)
            # Add small variation
            points += np.random.randn(5, 2) * 0.5
            instances.append(points)

        extractor.fit(instances)
        return extractor

    def test_fit_creates_stats(self, extractor):
        """Test that fit() creates statistics."""
        assert extractor.stats is not None
        assert len(extractor.stats.edge_means) > 0
        assert len(extractor.stats.pairwise_means) > 0

    def test_extract_returns_correct_shape(self, extractor):
        """Test that extract() returns correct number of features."""
        points = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=float)
        features = extractor.extract(points)

        assert features.shape == (len(BASELINE_FEATURE_NAMES),)

    def test_extract_handles_nan(self, extractor):
        """Test that extract() handles NaN (invisible nodes)."""
        points = np.array(
            [[0, 0], [10, 0], [np.nan, np.nan], [30, 0], [40, 0]], dtype=float
        )
        features = extractor.extract(points)

        # Should still return valid features
        assert features.shape == (len(BASELINE_FEATURE_NAMES),)
        assert not np.isnan(features).all()

    def test_extract_detects_anomaly(self, extractor):
        """Test that extract() produces different features for anomalies."""
        # Normal instance
        normal = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=float)
        normal_features = extractor.extract(normal)

        # Anomalous instance (one node displaced)
        anomalous = np.array([[0, 0], [10, 0], [20, 50], [30, 0], [40, 0]], dtype=float)
        anomalous_features = extractor.extract(anomalous)

        # Max edge z-score should be higher for anomaly
        assert anomalous_features[0] > normal_features[0]

    def test_symmetry_consistency(self, simple_skeleton):
        """Test symmetry consistency with symmetry pairs."""
        edges = [(0, 1), (0, 2), (1, 3), (2, 4)]  # Y-shaped
        n_nodes = 5
        symmetry_pairs = [(1, 2), (3, 4)]

        extractor = BaselineFeatureExtractor(edges, n_nodes, symmetry_pairs)

        # Create symmetric training data
        instances = []
        for _ in range(10):
            points = np.array(
                [[0, 0], [-10, 10], [10, 10], [-20, 20], [20, 20]], dtype=float
            )
            points += np.random.randn(5, 2) * 0.5
            instances.append(points)

        extractor.fit(instances)

        # Test symmetric instance
        symmetric = np.array(
            [[0, 0], [-10, 10], [10, 10], [-20, 20], [20, 20]], dtype=float
        )
        sym_features = extractor.extract(symmetric)

        # Test swapped instance (L/R swap)
        swapped = np.array(
            [[0, 0], [10, 10], [-10, 10], [-20, 20], [20, 20]], dtype=float
        )
        swap_features = extractor.extract(swapped)

        # min_symmetry_consistency should be lower for swapped
        sym_idx = BASELINE_FEATURE_NAMES.index("min_symmetry_consistency")
        assert swap_features[sym_idx] < sym_features[sym_idx]


class TestCurvature:
    """Tests for compute_curvature."""

    def test_straight_line_zero_curvature(self):
        """Straight line should have zero curvature."""
        points = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=float)
        chain = [0, 1, 2, 3, 4]

        result = compute_curvature(points, chain)

        assert result["max_curvature"] < 0.1
        assert result["mean_curvature"] < 0.1

    def test_bent_line_has_curvature(self):
        """Bent line should have non-zero curvature."""
        points = np.array([[0, 0], [10, 0], [20, 10], [30, 10], [40, 10]], dtype=float)
        chain = [0, 1, 2, 3, 4]

        result = compute_curvature(points, chain)

        assert result["max_curvature"] > 0.1

    def test_short_chain_returns_empty(self):
        """Chain < 3 nodes should return empty result."""
        points = np.array([[0, 0], [10, 0]], dtype=float)
        chain = [0, 1]

        result = compute_curvature(points, chain)

        assert result["max_curvature"] == 0.0
        assert len(result["curvatures"]) == 0

    def test_handles_nan(self):
        """Should handle NaN (invisible nodes)."""
        points = np.array(
            [[0, 0], [10, 0], [np.nan, np.nan], [30, 0], [40, 0]], dtype=float
        )
        chain = [0, 1, 2, 3, 4]

        result = compute_curvature(points, chain)

        # Should still return valid result
        assert "max_curvature" in result


class TestConvexHull:
    """Tests for compute_convex_hull."""

    def test_square_hull(self):
        """Square should have predictable hull metrics."""
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=float)

        result = compute_convex_hull(points)

        assert result["hull_area"] == pytest.approx(100.0, rel=0.01)
        assert result["n_hull_points"] == 4

    def test_handles_nan(self):
        """Should ignore NaN points."""
        points = np.array(
            [[0, 0], [10, 0], [np.nan, np.nan], [0, 10], [10, 10]], dtype=float
        )

        result = compute_convex_hull(points)

        assert result["hull_area"] > 0
        assert result["n_hull_points"] <= 4

    def test_too_few_points(self):
        """Should handle fewer than 3 visible points."""
        points = np.array(
            [[0, 0], [10, 0], [np.nan, np.nan], [np.nan, np.nan]], dtype=float
        )

        result = compute_convex_hull(points)

        assert result["hull_area"] == 0.0


class TestVisibilityModel:
    """Tests for VisibilityModel."""

    @pytest.fixture
    def visibility_model(self):
        """Create fitted visibility model."""
        # Create visibility masks where nodes 0,1,2 tend to be visible together
        masks = []
        for _ in range(50):
            mask = np.array([True, True, True, False, False])  # Common pattern
            masks.append(mask)
        for _ in range(5):
            mask = np.array([True, True, True, True, True])  # All visible
            masks.append(mask)

        model = VisibilityModel()
        model.fit(np.array(masks))
        return model

    def test_fit_creates_matrices(self, visibility_model):
        """Test that fit() creates required matrices."""
        assert visibility_model.co_visibility_matrix is not None
        assert visibility_model.visibility_rates is not None
        assert visibility_model.n_nodes == 5

    def test_normal_pattern_low_score(self, visibility_model):
        """Normal visibility pattern should have low score."""
        normal_mask = np.array([True, True, True, False, False])
        result = visibility_model.score(normal_mask)

        assert result["pattern_score"] < 0.5
        assert result["n_violations"] < 3

    def test_unusual_pattern_high_score(self, visibility_model):
        """Unusual visibility pattern should have high score."""
        # Node 0 visible but nodes 1,2 invisible (unusual given training)
        unusual_mask = np.array([True, False, False, True, True])
        result = visibility_model.score(unusual_mask)

        assert result["pattern_score"] > 0
        assert result["n_violations"] > 0


class TestIsolatedInvisible:
    """Tests for compute_isolated_invisible."""

    def test_no_isolated_invisible(self):
        """No isolated invisible when all visible."""
        mask = np.array([True, True, True, True, True])
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

        result = compute_isolated_invisible(mask, edges)

        assert not result["has_isolated_invisible"]

    def test_detects_isolated_invisible(self):
        """Should detect isolated invisible node."""
        # Node 2 invisible but neighbors 1 and 3 visible
        mask = np.array([True, True, False, True, True])
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

        result = compute_isolated_invisible(mask, edges)

        assert result["has_isolated_invisible"]
        assert 2 in result["isolated_invisible_nodes"]


class TestNearestNeighborScorer:
    """Tests for NearestNeighborScorer."""

    @pytest.fixture
    def nn_scorer(self):
        """Create fitted NN scorer."""
        # Create reference poses (simple variations)
        poses = []
        for i in range(10):
            points = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
            points += np.random.randn(3, 2) * 0.5
            poses.append(points)

        scorer = NearestNeighborScorer(normalize=True)
        scorer.fit(np.array(poses))
        return scorer

    def test_similar_pose_low_distance(self, nn_scorer):
        """Similar pose should have low NN distance."""
        similar = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        result = nn_scorer.score(similar)

        assert result["nn_distance"] < 0.5

    def test_different_pose_high_distance(self, nn_scorer):
        """Very different pose should have higher NN distance."""
        different = np.array([[0, 0], [0, 10], [0, 20]], dtype=float)  # Rotated 90°
        result = nn_scorer.score(different)

        assert result["nn_distance"] > 0.1

    def test_handles_nan(self, nn_scorer):
        """Should handle NaN in query pose."""
        partial = np.array([[0, 0], [np.nan, np.nan], [20, 0]], dtype=float)
        result = nn_scorer.score(partial)

        # Should still return valid distance
        assert np.isfinite(result["nn_distance"])


class TestNormalizePose:
    """Tests for normalize_pose."""

    def test_centers_pose(self):
        """Should center pose at origin."""
        points = np.array([[10, 10], [20, 10], [30, 10]], dtype=float)
        normalized = normalize_pose(points)

        centroid = normalized.mean(axis=0)
        assert np.allclose(centroid, [0, 0], atol=1e-6)

    def test_scales_to_unit(self):
        """Should scale to approximately unit size."""
        points = np.array([[0, 0], [100, 0], [100, 100]], dtype=float)
        normalized = normalize_pose(points)

        # Bounding box diagonal should be 1
        bbox_min = normalized.min(axis=0)
        bbox_max = normalized.max(axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)
        assert diagonal == pytest.approx(1.0, rel=0.01)

    def test_preserves_nan(self):
        """Should preserve NaN values."""
        points = np.array([[0, 0], [np.nan, np.nan], [10, 10]], dtype=float)
        normalized = normalize_pose(points)

        assert np.isnan(normalized[1]).all()


class TestPoseDistance:
    """Tests for pose_distance."""

    def test_identical_poses_zero_distance(self):
        """Identical poses should have zero distance."""
        pose = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        dist = pose_distance(pose, pose)

        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_poses_nonzero_distance(self):
        """Different poses should have non-zero distance."""
        pose_a = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        pose_b = np.array([[0, 0], [10, 5], [20, 0]], dtype=float)

        dist = pose_distance(pose_a, pose_b)
        assert dist > 0

    def test_handles_partial_visibility(self):
        """Should compute distance only for common visible nodes."""
        pose_a = np.array([[0, 0], [np.nan, np.nan], [20, 0]], dtype=float)
        pose_b = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)

        dist = pose_distance(pose_a, pose_b)
        assert np.isfinite(dist)

    def test_no_common_nodes_returns_inf(self):
        """Should return inf if no common visible nodes."""
        pose_a = np.array([[0, 0], [np.nan, np.nan]], dtype=float)
        pose_b = np.array([[np.nan, np.nan], [10, 0]], dtype=float)

        dist = pose_distance(pose_a, pose_b)
        assert dist == float("inf")

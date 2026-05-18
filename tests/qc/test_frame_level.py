"""Tests for sleap.qc.frame_level module."""

import numpy as np
import pytest

from sleap.qc.frame_level import (
    InstanceCountChecker,
    compute_instance_iou,
    compute_node_overlap,
    detect_duplicates,
)


class TestInstanceCountChecker:
    """Tests for InstanceCountChecker."""

    @pytest.fixture
    def count_checker(self):
        """Create fitted instance count checker."""
        # Typically 2 instances per frame
        frame_counts = [2, 2, 2, 2, 2, 2, 1, 2, 2, 2]
        video_ids = ["vid1"] * 10

        checker = InstanceCountChecker(per_video=True)
        checker.fit(frame_counts, video_ids)
        return checker

    def test_fit_computes_expected(self, count_checker):
        """Test that fit() computes expected counts."""
        assert count_checker.global_expected == 2.0
        assert "vid1" in count_checker.expected_counts
        assert count_checker.expected_counts["vid1"] == 2.0

    def test_normal_count_not_incomplete(self, count_checker):
        """Normal count should not be flagged as incomplete."""
        result = count_checker.check(2, "vid1")

        assert not result["is_incomplete"]
        assert result["expected_count"] == 2.0
        assert result["actual_count"] == 2

    def test_low_count_is_incomplete(self, count_checker):
        """Low count should be flagged as incomplete."""
        result = count_checker.check(1, "vid1")

        assert result["is_incomplete"]
        assert result["count_difference"] == -1

    def test_high_count_not_incomplete(self, count_checker):
        """High count should not be flagged as incomplete."""
        result = count_checker.check(3, "vid1")

        assert not result["is_incomplete"]
        assert result["count_difference"] == 1

    def test_uses_global_for_unknown_video(self, count_checker):
        """Should use global count for unknown video."""
        result = count_checker.check(1, "unknown_video")

        assert result["is_incomplete"]
        assert result["expected_count"] == 2.0

    def test_per_video_counts(self):
        """Test per-video expected counts."""
        frame_counts = [1, 1, 1, 3, 3, 3]
        video_ids = ["vid1", "vid1", "vid1", "vid2", "vid2", "vid2"]

        checker = InstanceCountChecker(per_video=True)
        checker.fit(frame_counts, video_ids)

        assert checker.expected_counts["vid1"] == 1.0
        assert checker.expected_counts["vid2"] == 3.0

        # Check for vid1 (expects 1)
        result1 = checker.check(1, "vid1")
        assert not result1["is_incomplete"]

        # Check for vid2 (expects 3)
        result2 = checker.check(1, "vid2")
        assert result2["is_incomplete"]


class TestComputeInstanceIOU:
    """Tests for compute_instance_iou."""

    def test_identical_instances_iou_one(self):
        """Identical instances should have IOU = 1."""
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        iou = compute_instance_iou(points, points)

        assert iou == pytest.approx(1.0, rel=0.01)

    def test_non_overlapping_iou_zero(self):
        """Non-overlapping instances should have IOU = 0."""
        points_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        points_b = np.array(
            [[100, 100], [110, 100], [110, 110], [100, 110]], dtype=float
        )

        iou = compute_instance_iou(points_a, points_b)

        assert iou == 0.0

    def test_partial_overlap_intermediate_iou(self):
        """Partially overlapping instances should have 0 < IOU < 1."""
        points_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        points_b = np.array([[5, 0], [15, 0], [15, 10], [5, 10]], dtype=float)

        iou = compute_instance_iou(points_a, points_b)

        assert 0 < iou < 1

    def test_handles_nan(self):
        """Should handle NaN (invisible nodes)."""
        points_a = np.array([[0, 0], [10, 0], [np.nan, np.nan], [0, 10]], dtype=float)
        points_b = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        iou = compute_instance_iou(points_a, points_b)

        # Should still compute IOU from visible points
        assert iou >= 0

    def test_too_few_visible_points(self):
        """Should return 0 if fewer than 2 visible points."""
        points_a = np.array([[0, 0], [np.nan, np.nan]], dtype=float)
        points_b = np.array([[0, 0], [10, 0]], dtype=float)

        iou = compute_instance_iou(points_a, points_b)

        assert iou == 0.0


class TestComputeNodeOverlap:
    """Tests for compute_node_overlap."""

    def test_identical_instances_full_overlap(self):
        """Identical instances should have 100% node overlap."""
        points = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)

        result = compute_node_overlap(points, points, distance_threshold=1.0)

        assert result["overlap_ratio"] == 1.0
        assert result["mean_distance"] == pytest.approx(0.0, abs=1e-6)

    def test_close_instances_high_overlap(self):
        """Close instances should have high overlap ratio."""
        points_a = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        points_b = np.array([[1, 0], [11, 0], [21, 0]], dtype=float)  # Shifted by 1

        result = compute_node_overlap(points_a, points_b, distance_threshold=5.0)

        assert result["overlap_ratio"] == 1.0
        assert result["mean_distance"] == pytest.approx(1.0, abs=0.1)

    def test_far_instances_low_overlap(self):
        """Far instances should have low overlap ratio."""
        points_a = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        points_b = np.array([[100, 0], [110, 0], [120, 0]], dtype=float)

        result = compute_node_overlap(points_a, points_b, distance_threshold=10.0)

        assert result["overlap_ratio"] == 0.0

    def test_handles_nan(self):
        """Should only compare commonly visible nodes."""
        points_a = np.array([[0, 0], [np.nan, np.nan], [20, 0]], dtype=float)
        points_b = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)

        result = compute_node_overlap(points_a, points_b, distance_threshold=5.0)

        # Only 2 common nodes (0 and 2)
        assert len(result["common_nodes"]) == 2
        assert 0 in result["common_nodes"]
        assert 2 in result["common_nodes"]

    def test_no_common_nodes(self):
        """Should handle no common visible nodes."""
        points_a = np.array([[0, 0], [np.nan, np.nan]], dtype=float)
        points_b = np.array([[np.nan, np.nan], [10, 0]], dtype=float)

        result = compute_node_overlap(points_a, points_b)

        assert len(result["common_nodes"]) == 0
        assert result["overlap_ratio"] == 0.0


class TestDetectDuplicates:
    """Tests for detect_duplicates."""

    def test_no_duplicates_in_distinct_instances(self):
        """Distinct instances should not be flagged as duplicates."""
        instances = [
            np.array([[0, 0], [10, 0], [20, 0]], dtype=float),
            np.array([[100, 100], [110, 100], [120, 100]], dtype=float),
        ]

        duplicates = detect_duplicates(instances)

        assert len(duplicates) == 0

    def test_detects_identical_duplicates(self):
        """Should detect identical instances as duplicates."""
        points = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        instances = [points, points.copy()]

        duplicates = detect_duplicates(instances)

        assert len(duplicates) == 1
        assert duplicates[0]["index_a"] == 0
        assert duplicates[0]["index_b"] == 1

    def test_detects_by_iou(self):
        """Should detect duplicates by high IOU."""
        instances = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
            np.array([[1, 1], [11, 1], [11, 11], [1, 11]], dtype=float),  # Slight shift
        ]

        duplicates = detect_duplicates(instances, iou_threshold=0.7)

        assert len(duplicates) == 1
        # May trigger via iou or node_overlap (both are valid duplicate reasons)
        assert duplicates[0]["reason"] in ("iou", "node_overlap")

    def test_detects_by_node_overlap(self):
        """Should detect duplicates by node overlap even with lower IOU."""
        instances = [
            np.array([[0, 0], [10, 0], [20, 0]], dtype=float),
            np.array([[1, 0], [11, 0], [21, 0]], dtype=float),  # Close nodes
        ]

        duplicates = detect_duplicates(
            instances,
            iou_threshold=0.95,  # High threshold - won't trigger
            node_distance_threshold=5.0,
            node_overlap_ratio=0.8,
        )

        assert len(duplicates) == 1
        assert duplicates[0]["reason"] == "node_overlap"

    def test_multiple_duplicates(self):
        """Should detect multiple duplicate pairs."""
        points = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        instances = [points, points.copy(), points.copy()]

        duplicates = detect_duplicates(instances)

        # Should detect (0,1), (0,2), (1,2)
        assert len(duplicates) == 3

    def test_handles_nan(self):
        """Should handle instances with NaN values."""
        instances = [
            np.array([[0, 0], [10, 0], [np.nan, np.nan]], dtype=float),
            np.array([[0, 0], [10, 0], [20, 0]], dtype=float),
        ]

        # Should not raise
        duplicates = detect_duplicates(instances)

        # May or may not be detected as duplicate depending on thresholds
        assert isinstance(duplicates, list)

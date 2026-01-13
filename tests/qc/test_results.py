"""Tests for sleap.qc.results module."""

import pytest

from sleap.qc.results import QCResults, QCFlag, FrameQC, InstanceKey, FrameKey


class TestInstanceKey:
    """Tests for InstanceKey."""

    def test_creation(self):
        """Test creating an InstanceKey."""
        key = InstanceKey(0, 10, 2)

        assert key.video_idx == 0
        assert key.frame_idx == 10
        assert key.instance_idx == 2

    def test_hashable(self):
        """Test that InstanceKey is hashable (usable as dict key)."""
        key1 = InstanceKey(0, 10, 2)
        key2 = InstanceKey(0, 10, 2)
        key3 = InstanceKey(0, 10, 3)

        d = {key1: "test"}
        assert d[key2] == "test"
        assert key3 not in d


class TestFrameKey:
    """Tests for FrameKey."""

    def test_creation(self):
        """Test creating a FrameKey."""
        key = FrameKey(0, 10)

        assert key.video_idx == 0
        assert key.frame_idx == 10


class TestFrameQC:
    """Tests for FrameQC."""

    def test_default_values(self):
        """Test default FrameQC values."""
        frame_qc = FrameQC()

        assert not frame_qc.is_incomplete
        assert frame_qc.expected_instance_count == 0
        assert frame_qc.actual_instance_count == 0
        assert frame_qc.duplicate_pairs == []

    def test_with_values(self):
        """Test FrameQC with values."""
        frame_qc = FrameQC(
            is_incomplete=True,
            expected_instance_count=2,
            actual_instance_count=1,
            duplicate_pairs=[(0, 1)],
            duplicate_reasons=["iou"],
        )

        assert frame_qc.is_incomplete
        assert frame_qc.expected_instance_count == 2
        assert len(frame_qc.duplicate_pairs) == 1


class TestQCFlag:
    """Tests for QCFlag."""

    def test_creation(self):
        """Test creating a QCFlag."""
        key = InstanceKey(0, 10, 2)
        flag = QCFlag(
            instance_key=key,
            score=0.85,
            confidence="high",
            top_issue="Unusual edge length",
            feature_contributions={"max_edge_zscore": 5.0},
            explanation="Test explanation",
        )

        assert flag.score == 0.85
        assert flag.confidence == "high"
        assert flag.video_idx == 0
        assert flag.frame_idx == 10
        assert flag.instance_idx == 2


class TestQCResults:
    """Tests for QCResults."""

    @pytest.fixture
    def results(self):
        """Create QCResults with sample data."""
        results = QCResults(feature_names=["max_edge_zscore", "mean_edge_zscore"])

        # Add some instance scores
        key1 = InstanceKey(0, 0, 0)
        key2 = InstanceKey(0, 0, 1)
        key3 = InstanceKey(0, 1, 0)

        results.instance_scores = {
            key1: 0.3,  # Normal
            key2: 0.8,  # Should be flagged
            key3: 0.95,  # Should be flagged
        }
        results.feature_contributions = {
            key1: {"max_edge_zscore": 1.0, "mean_edge_zscore": 0.5},
            key2: {"max_edge_zscore": 4.0, "mean_edge_zscore": 2.0},
            key3: {"max_edge_zscore": 6.0, "mean_edge_zscore": 3.0},
        }

        # Add frame results
        frame_key = FrameKey(0, 0)
        results.frame_results[frame_key] = FrameQC(
            is_incomplete=False,
            expected_instance_count=2,
            actual_instance_count=2,
        )

        return results

    def test_get_flagged_returns_list(self, results):
        """Test get_flagged returns list of QCFlag."""
        flagged = results.get_flagged(threshold=0.7)

        assert isinstance(flagged, list)
        assert len(flagged) == 2  # key2 and key3
        assert all(isinstance(f, QCFlag) for f in flagged)

    def test_get_flagged_sorted_by_score(self, results):
        """Test get_flagged returns flags sorted by score descending."""
        flagged = results.get_flagged(threshold=0.7)

        assert flagged[0].score == 0.95
        assert flagged[1].score == 0.8

    def test_get_flagged_threshold(self, results):
        """Test get_flagged respects threshold."""
        # High threshold - only highest score
        flagged_high = results.get_flagged(threshold=0.9)
        assert len(flagged_high) == 1

        # Low threshold - all flagged
        flagged_low = results.get_flagged(threshold=0.2)
        assert len(flagged_low) == 3

    def test_get_flagged_includes_explanation(self, results):
        """Test that flagged instances include explanations."""
        flagged = results.get_flagged(threshold=0.7)

        assert flagged[0].explanation is not None
        assert "Anomaly score" in flagged[0].explanation
        assert flagged[0].top_issue is not None

    def test_get_frame_issues(self, results):
        """Test get_frame_issues."""
        # Add a frame with issues
        frame_key = FrameKey(0, 5)
        results.frame_results[frame_key] = FrameQC(
            is_incomplete=True,
            expected_instance_count=2,
            actual_instance_count=1,
        )

        issues = results.get_frame_issues()

        assert len(issues) == 1
        assert issues[0][0] == frame_key
        assert issues[0][1].is_incomplete

    def test_get_explanation(self, results):
        """Test get_explanation for specific instance."""
        key = InstanceKey(0, 0, 1)  # High score instance
        explanation = results.get_explanation(key)

        assert "Anomaly score: 0.80" in explanation
        assert "max_edge_zscore" in explanation

    def test_get_explanation_unknown_instance(self, results):
        """Test get_explanation for unknown instance."""
        unknown_key = InstanceKey(99, 99, 99)
        explanation = results.get_explanation(unknown_key)

        assert "not found" in explanation

    def test_to_dataframe(self, results):
        """Test export to DataFrame."""
        df = results.to_dataframe()

        assert len(df) == 3
        assert "video_idx" in df.columns
        assert "frame_idx" in df.columns
        assert "instance_idx" in df.columns
        assert "score" in df.columns
        assert "confidence" in df.columns
        assert "top_issue" in df.columns
        assert "max_edge_zscore" in df.columns

    def test_confidence_levels(self, results):
        """Test confidence level assignment."""
        flagged = results.get_flagged(threshold=0.0)

        confidences = {f.score: f.confidence for f in flagged}

        # Score 0.95 -> high (> 0.8)
        assert confidences[0.95] == "high"
        # Score 0.8 -> medium (== 0.8, not > 0.8)
        assert confidences[0.8] == "medium"
        # Score 0.3 -> low (< 0.5)
        assert confidences[0.3] == "low"

    def test_infer_top_issue(self, results):
        """Test top issue inference from feature contributions."""
        flagged = results.get_flagged(threshold=0.7)

        # Highest contribution is max_edge_zscore
        assert "edge length" in flagged[0].top_issue.lower()

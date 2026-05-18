"""Tests for sleap.qc.gmm module."""

import numpy as np
import pytest

from sleap.qc.gmm import GMMDetector, ZScoreDetector


class TestGMMDetector:
    """Tests for GMMDetector."""

    @pytest.fixture
    def gmm_detector(self):
        """Create fitted GMM detector."""
        # Create clean feature matrix
        n_samples = 100
        n_features = 10

        # Normal distribution centered at 0
        features = np.random.randn(n_samples, n_features) * 0.5

        detector = GMMDetector(n_components=3, percentile_threshold=5.0)
        detector.fit(features)
        return detector

    def test_fit_creates_model(self, gmm_detector):
        """Test that fit() creates model and scaler."""
        assert gmm_detector.model is not None
        assert gmm_detector.scaler is not None
        assert gmm_detector.log_likelihood_threshold is not None

    def test_score_returns_dict(self, gmm_detector):
        """Test that score() returns expected dictionary."""
        features = np.random.randn(10)
        result = gmm_detector.score(features)

        assert "log_likelihood" in result
        assert "is_anomaly" in result
        assert "normalized_score" in result
        assert "component_probs" in result

    def test_normal_sample_low_score(self, gmm_detector):
        """Normal samples should have low anomaly scores."""
        # Use mean of training distribution (should be very normal)
        normal = np.zeros(10)  # Center of distribution
        result = gmm_detector.score(normal)

        # Score should be relatively low (not an extreme outlier)
        # With percentile-based scoring, center of distribution should be ~0.5
        assert result["normalized_score"] < 0.9

    def test_outlier_high_score(self, gmm_detector):
        """Outliers should have high anomaly scores."""
        # Sample far from training distribution
        outlier = np.ones(10) * 10
        result = gmm_detector.score(outlier)

        assert result["normalized_score"] > 0.5

    def test_score_batch_returns_array(self, gmm_detector):
        """Test that score_batch() returns array of scores."""
        features = np.random.randn(20, 10)
        scores = gmm_detector.score_batch(features)

        assert scores.shape == (20,)
        assert not np.isnan(scores).all()

    def test_handles_nan_in_score(self, gmm_detector):
        """Test handling of NaN in score()."""
        features_with_nan = np.array([1.0, np.nan, 0.5] + [0.0] * 7)
        result = gmm_detector.score(features_with_nan)

        assert np.isnan(result["log_likelihood"])
        assert np.isnan(result["normalized_score"])

    def test_handles_nan_in_batch(self, gmm_detector):
        """Test handling of NaN rows in score_batch()."""
        features = np.random.randn(5, 10)
        features[2, 3] = np.nan  # One row has NaN

        scores = gmm_detector.score_batch(features)

        assert np.isnan(scores[2])
        assert not np.isnan(scores[0])

    def test_small_sample_reduces_components(self):
        """Test that GMM reduces components for small samples."""
        # Only 15 samples - should reduce n_components
        features = np.random.randn(15, 5)

        detector = GMMDetector(n_components=5)
        detector.fit(features)

        # n_components should be at most n_samples // 10 = 1
        assert detector.model.n_components <= 2

    def test_normalized_score_range(self, gmm_detector):
        """Test that normalized scores are in [0, 1]."""
        features = np.random.randn(50, 10)
        scores = gmm_detector.score_batch(features)

        valid_scores = scores[~np.isnan(scores)]
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 1).all()


class TestZScoreDetector:
    """Tests for ZScoreDetector (fallback detector)."""

    @pytest.fixture
    def zscore_detector(self):
        """Create fitted z-score detector."""
        features = np.random.randn(50, 5) * 2  # std=2

        detector = ZScoreDetector(threshold=3.0)
        detector.fit(features)
        return detector

    def test_fit_computes_stats(self, zscore_detector):
        """Test that fit() computes mean and std."""
        assert zscore_detector.means is not None
        assert zscore_detector.stds is not None
        assert len(zscore_detector.means) == 5

    def test_normal_sample_low_score(self, zscore_detector):
        """Normal samples should have low scores."""
        # Sample with z < 3
        normal = np.zeros(5)
        scores = zscore_detector.score_batch(normal.reshape(1, -1))

        assert scores[0] < 0.5

    def test_outlier_high_score(self, zscore_detector):
        """Outliers should have high scores."""
        # Sample with z >> 3
        outlier = np.ones(5) * 20  # Way above mean
        scores = zscore_detector.score_batch(outlier.reshape(1, -1))

        assert scores[0] > 0.5

    def test_score_batch_shape(self, zscore_detector):
        """Test output shape."""
        features = np.random.randn(10, 5)
        scores = zscore_detector.score_batch(features)

        assert scores.shape == (10,)

    def test_handles_nan(self, zscore_detector):
        """Test handling of NaN."""
        features = np.array([[1.0, np.nan, 0.5, 0.0, 0.0]])
        scores = zscore_detector.score_batch(features)

        assert np.isnan(scores[0])

    def test_prevents_division_by_zero(self):
        """Test that zero std is handled."""
        # Create features with constant column
        features = np.ones((10, 3))
        features[:, 0] = np.arange(10)  # Only first column varies

        detector = ZScoreDetector()
        detector.fit(features)

        # Should not have zero std (should be clipped to 1e-6)
        assert (detector.stds >= 1e-6).all()

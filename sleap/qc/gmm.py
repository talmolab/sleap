"""Gaussian Mixture Model for anomaly detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMDetector:
    """Anomaly detection using GMM likelihood.

    Fits a mixture model to clean data and flags low-likelihood instances.

    Attributes:
        n_components: Number of Gaussian components.
        covariance_type: Covariance type for GMM.
        percentile_threshold: Percentile below which instances are anomalies.
        model: Fitted GaussianMixture model.
        scaler: Fitted StandardScaler for feature normalization.
        log_likelihood_threshold: Threshold for anomaly detection.
        feature_names: Optional list of feature names.
    """

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = "full",
        percentile_threshold: float = 5.0,
    ):
        """Initialize detector.

        Args:
            n_components: Number of Gaussian components.
            covariance_type: Covariance type ("full", "tied", "diag", "spherical").
            percentile_threshold: Percentile below which instances are anomalies.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.percentile_threshold = percentile_threshold

        self.model: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.log_likelihood_threshold: Optional[float] = None
        self.train_log_likelihoods: Optional[np.ndarray] = None  # For percentiles
        self.feature_names: Optional[list[str]] = None

    def fit(
        self,
        feature_matrix: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "GMMDetector":
        """Fit the GMM on clean data.

        Args:
            feature_matrix: (N_instances, N_features) array of clean instances.
            feature_names: Optional list of feature names.

        Returns:
            Self for chaining.
        """
        self.feature_names = feature_names

        # Remove instances with NaN
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        valid_features = feature_matrix[valid_mask]

        if len(valid_features) < 2:
            raise ValueError("Need at least 2 valid instances to fit GMM")

        # Determine number of components (at most n_samples // 10)
        n_components = min(self.n_components, len(valid_features) // 10)
        if n_components < 1:
            n_components = 1

        # Standardize features
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(valid_features)

        # Fit GMM
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        self.model.fit(scaled)

        # Compute threshold and store training log-likelihoods for percentile scoring
        self.train_log_likelihoods = self.model.score_samples(scaled)
        self.log_likelihood_threshold = np.percentile(
            self.train_log_likelihoods, self.percentile_threshold
        )

        return self

    def score(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Score a single instance.

        Args:
            feature_vector: (N_features,) array.

        Returns:
            Dictionary with:
            - log_likelihood: log-likelihood under the model
            - is_anomaly: True if below threshold
            - normalized_score: score normalized to ~0-1 (higher = more anomalous)
            - component_probs: probability of belonging to each component
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if np.isnan(feature_vector).any():
            return {
                "log_likelihood": np.nan,
                "is_anomaly": False,
                "normalized_score": np.nan,
                "component_probs": [],
            }

        # Scale and score
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        log_likelihood = self.model.score_samples(scaled)[0]
        component_probs = self.model.predict_proba(scaled)[0]

        is_anomaly = log_likelihood < self.log_likelihood_threshold

        # Normalize: convert log-likelihood to percentile-based score
        # Score = 1 - percentile (so low log_ll = high score = anomalous)
        percentile = (self.train_log_likelihoods < log_likelihood).sum() / len(
            self.train_log_likelihoods
        )
        normalized = 1.0 - percentile

        return {
            "log_likelihood": float(log_likelihood),
            "is_anomaly": is_anomaly,
            "normalized_score": float(normalized),
            "component_probs": component_probs.tolist(),
        }

    def score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score multiple instances.

        Args:
            feature_matrix: (N_instances, N_features) array.

        Returns:
            (N_instances,) array of normalized anomaly scores (0-1).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = np.full(len(feature_matrix), np.nan)

        # Find valid instances
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        if valid_mask.sum() == 0:
            return scores

        valid_features = feature_matrix[valid_mask]
        scaled = self.scaler.transform(valid_features)

        log_likelihoods = self.model.score_samples(scaled)

        # Normalize to 0-1 (higher = more anomalous) using percentile-based scoring
        # For each log_ll, compute its percentile in training distribution
        # Score = 1 - percentile (so low log_ll = high score)
        normalized = np.zeros(len(log_likelihoods))
        for i, ll in enumerate(log_likelihoods):
            percentile = (self.train_log_likelihoods < ll).sum() / len(
                self.train_log_likelihoods
            )
            normalized[i] = 1.0 - percentile

        scores[valid_mask] = normalized
        return scores


class ZScoreDetector:
    """Fallback detector using simple z-score thresholding.

    Used when there are too few samples for GMM fitting.
    """

    def __init__(self, threshold: float = 3.0):
        """Initialize detector.

        Args:
            threshold: Z-score threshold for flagging.
        """
        self.threshold = threshold
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, feature_matrix: np.ndarray) -> "ZScoreDetector":
        """Compute mean and std from reference data.

        Args:
            feature_matrix: (N_instances, N_features) array.

        Returns:
            Self for chaining.
        """
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        valid_features = feature_matrix[valid_mask]

        self.means = np.mean(valid_features, axis=0)
        self.stds = np.std(valid_features, axis=0)
        # Avoid division by zero
        self.stds = np.maximum(self.stds, 1e-6)

        return self

    def score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score instances by max z-score.

        Args:
            feature_matrix: (N_instances, N_features) array.

        Returns:
            (N_instances,) array of normalized anomaly scores (0-1).
        """
        if self.means is None or self.stds is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = np.full(len(feature_matrix), np.nan)
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)

        if valid_mask.sum() == 0:
            return scores

        valid_features = feature_matrix[valid_mask]

        # Compute z-scores
        zscores = np.abs((valid_features - self.means) / self.stds)
        max_zscores = np.max(zscores, axis=1)

        # Normalize to 0-1 using sigmoid around threshold
        diff = max_zscores - self.threshold
        normalized = 1 / (1 + np.exp(-diff))

        scores[valid_mask] = normalized
        return scores

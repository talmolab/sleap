"""Result classes for Label QC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    import pandas as pd


class InstanceKey(NamedTuple):
    """Key to identify a specific instance in a Labels object."""

    video_idx: int
    frame_idx: int
    instance_idx: int


class FrameKey(NamedTuple):
    """Key to identify a specific frame in a Labels object."""

    video_idx: int
    frame_idx: int


@dataclass
class FrameQC:
    """Quality check results for a single frame."""

    is_incomplete: bool = False
    expected_instance_count: int = 0
    actual_instance_count: int = 0
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    duplicate_reasons: list[str] = field(default_factory=list)


@dataclass
class QCFlag:
    """Single flagged instance with explanation."""

    instance_key: InstanceKey
    score: float
    confidence: str  # "low", "medium", "high"
    top_issue: str
    feature_contributions: dict[str, float]
    explanation: str

    @property
    def video_idx(self) -> int:
        """Video index."""
        return self.instance_key.video_idx

    @property
    def frame_idx(self) -> int:
        """Frame index."""
        return self.instance_key.frame_idx

    @property
    def instance_idx(self) -> int:
        """Instance index within the frame."""
        return self.instance_key.instance_idx


@dataclass
class QCResults:
    """Container for all QC results.

    Attributes:
        instance_scores: Mapping from instance key to anomaly score (0-1).
        frame_results: Mapping from frame key to frame-level QC results.
        feature_contributions: Mapping from instance key to per-feature scores.
        feature_names: List of feature names used.
    """

    instance_scores: dict[InstanceKey, float] = field(default_factory=dict)
    frame_results: dict[FrameKey, FrameQC] = field(default_factory=dict)
    feature_contributions: dict[InstanceKey, dict[str, float]] = field(
        default_factory=dict
    )
    feature_names: list[str] = field(default_factory=list)

    def get_flagged(self, threshold: float = 0.7) -> list[QCFlag]:
        """Get instances flagged above threshold.

        Args:
            threshold: Score threshold (0-1). Instances with scores >= threshold
                are flagged.

        Returns:
            List of QCFlag objects, sorted by score descending.
        """
        flagged = []
        for key, score in self.instance_scores.items():
            if score >= threshold:
                contributions = self.feature_contributions.get(key, {})
                top_issue = self._infer_top_issue(contributions)
                confidence = self._get_confidence(score, contributions)
                explanation = self._generate_explanation(
                    score, top_issue, contributions
                )

                flagged.append(
                    QCFlag(
                        instance_key=key,
                        score=score,
                        confidence=confidence,
                        top_issue=top_issue,
                        feature_contributions=contributions,
                        explanation=explanation,
                    )
                )

        # Sort by score descending
        flagged.sort(key=lambda f: f.score, reverse=True)
        return flagged

    def get_frame_issues(self) -> list[tuple[FrameKey, FrameQC]]:
        """Get frames with issues (incomplete or duplicates)."""
        issues = []
        for key, frame_qc in self.frame_results.items():
            if frame_qc.is_incomplete or frame_qc.duplicate_pairs:
                issues.append((key, frame_qc))
        return issues

    def get_explanation(self, instance_key: InstanceKey) -> str:
        """Get human-readable explanation for instance."""
        score = self.instance_scores.get(instance_key)
        if score is None:
            return "Instance not found in results."

        contributions = self.feature_contributions.get(instance_key, {})
        top_issue = self._infer_top_issue(contributions)
        return self._generate_explanation(score, top_issue, contributions)

    def to_dataframe(self) -> "pd.DataFrame":
        """Export results as DataFrame.

        Returns:
            DataFrame with columns: video_idx, frame_idx, instance_idx, score,
            confidence, top_issue, and one column per feature.
        """
        import pandas as pd

        rows = []
        for key, score in self.instance_scores.items():
            contributions = self.feature_contributions.get(key, {})
            row = {
                "video_idx": key.video_idx,
                "frame_idx": key.frame_idx,
                "instance_idx": key.instance_idx,
                "score": score,
                "confidence": self._get_confidence(score, contributions),
                "top_issue": self._infer_top_issue(contributions),
            }
            row.update(contributions)
            rows.append(row)

        return pd.DataFrame(rows)

    def _infer_top_issue(self, contributions: dict[str, float]) -> str:
        """Infer the most likely issue from feature contributions.

        Normalizes contributions to comparable scales before finding the
        dominant feature, since z-score features (~0-5) and raw distance
        features (~0-100+) have different magnitudes.
        """
        if not contributions:
            return "Unknown"

        # Normalize contributions to comparable scales
        # Z-score features are already ~0-5 range, raw features need scaling
        scale_factors = {
            # Raw distance features - scale to ~0-5 range
            "max_centroid_distance": 30.0,
            "centroid_distance_std": 10.0,
            "nn_distance": 10.0,
            # Curvature is typically 0-3
            "max_curvature": 1.0,
            "curvature_std": 1.0,
            # Rate features (0-1 range) - scale up to be comparable
            "visibility_rate": 0.3,
            "visibility_pattern_score": 0.3,
            "has_isolated_invisible": 0.3,
            # Symmetry: only meaningful if skeleton has symmetry defined
            # Value of 1.0 usually means no symmetry info, so scale down
            "min_symmetry_consistency": 5.0,
        }

        normalized = {}
        for feat, val in contributions.items():
            scale = scale_factors.get(feat, 1.0)
            # Skip features with default/uninformative values
            if feat == "min_symmetry_consistency" and val == 1.0:
                normalized[feat] = 0.0  # Ignore if no symmetry data
            else:
                normalized[feat] = val / scale

        # Find the feature with highest normalized contribution
        top_feature = max(normalized, key=normalized.get)

        # Map feature names to issue descriptions
        issue_map = {
            "max_edge_zscore": "Unusual edge length",
            "mean_edge_zscore": "Unusual proportions",
            "max_angle_zscore": "Unusual joint angle",
            "mean_angle_zscore": "Unusual pose structure",
            "max_pairwise_zscore": "Unusual node spacing",
            "mean_pairwise_zscore": "Unusual scale",
            "bbox_area_zscore": "Unusual scale",
            "max_centroid_distance": "Isolated node",
            "centroid_distance_std": "Inconsistent spacing",
            "min_symmetry_consistency": "Likely L/R swap",
            "visibility_rate": "Unusual visibility",
            "has_isolated_invisible": "Isolated invisible node",
            "visibility_pattern_score": "Unusual visibility pattern",
            "nn_distance": "Unusual pose shape",
            "max_curvature": "Unusual curvature",
            "hull_area_zscore": "Unusual pose extent",
        }

        return issue_map.get(top_feature, f"High {top_feature}")

    def _get_confidence(self, score: float, contributions: dict[str, float]) -> str:
        """Determine confidence level."""
        if score > 0.8:
            return "high"
        elif score > 0.5:
            return "medium"
        return "low"

    def _generate_explanation(
        self, score: float, top_issue: str, contributions: dict[str, float]
    ) -> str:
        """Generate human-readable explanation."""
        lines = [f"Anomaly score: {score:.2f}", f"Primary issue: {top_issue}"]

        if contributions:
            # Get top 3 contributing features
            sorted_features = sorted(
                contributions.items(), key=lambda x: x[1], reverse=True
            )[:3]
            lines.append("Top contributing features:")
            for feature, value in sorted_features:
                lines.append(f"  - {feature}: {value:.3f}")

        return "\n".join(lines)

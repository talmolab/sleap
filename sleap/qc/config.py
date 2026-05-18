"""Configuration for Label QC detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class QCConfig:
    """Configuration for QC detector.

    Attributes:
        use_gmm: Whether to use GMM-based anomaly detection.
        use_curvature: Whether to compute curvature features.
            If "auto", enables when skeleton has chains >= 5 nodes.
        use_symmetry: Whether to compute symmetry features.
            If "auto", enables when skeleton has symmetry pairs defined.
        use_anatomical: Whether to compute anatomical features (signed angles).
        instance_threshold: Threshold for flagging instances (0-1).
            Higher = fewer flags, lower = more flags.
        frame_threshold: Threshold for frame-level checks.
        duplicate_iou_threshold: IOU threshold for duplicate detection.
        duplicate_node_overlap_ratio: Node overlap ratio for partial duplicates.
        gmm_n_components: Number of GMM components.
        gmm_min_samples: Minimum samples required for GMM fitting.
            Below this, falls back to z-score thresholding.
        gmm_percentile_threshold: Percentile below which instances are anomalies.
        auto_calibrate: Whether to auto-calibrate threshold from data.
        calibration_percentile: Percentile for auto-calibration.
    """

    # Feature selection
    use_gmm: bool = True
    use_curvature: Literal["auto"] | bool = "auto"
    use_symmetry: Literal["auto"] | bool = "auto"
    use_anatomical: bool = False

    # Thresholds (validated in v4 investigation)
    instance_threshold: float = 0.7  # Default: balanced
    frame_threshold: float = 0.5
    duplicate_iou_threshold: float = 0.5
    duplicate_node_overlap_ratio: float = 0.8
    duplicate_node_distance_threshold: float = 10.0

    # GMM settings
    gmm_n_components: int = 5
    gmm_min_samples: int = 50
    gmm_percentile_threshold: float = 5.0

    # Calibration
    auto_calibrate: bool = True
    calibration_percentile: float = 95.0

    def should_use_curvature(self, max_chain_length: int) -> bool:
        """Determine if curvature features should be used."""
        if isinstance(self.use_curvature, bool):
            return self.use_curvature
        # Auto mode: enable for chains >= 5 nodes
        return max_chain_length >= 5

    def should_use_symmetry(self, has_symmetry: bool) -> bool:
        """Determine if symmetry features should be used."""
        if isinstance(self.use_symmetry, bool):
            return self.use_symmetry
        # Auto mode: enable if skeleton has symmetry pairs
        return has_symmetry

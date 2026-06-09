"""Configuration for Label QC detector."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        use_chirality: Whether to compute the whole-instance left/right
            mirror-flip (chirality) feature. If "auto", enables when the
            skeleton has symmetry pairs (defined or inferred by name). Reliable
            detector, default-ON.
        use_split_detection: Whether to compute the pose-split (chimera) feature
            that flags a single instance spanning two animals. Reliable
            detector, default-ON.
        use_duplicate_score: Whether to fold the complementary split-duplicate
            signal into frame-level duplicate detection. Reliable detector,
            default-ON.
        use_chain_ordering: Whether to compute the keypoint chain-ordering
            feature (wrong order along an ordered chain). If "auto", enables
            when the longest chain has >= 4 nodes. Experimental, default-OFF.
        use_missing_node_check: Whether to run the missing-node check (a node a
            instance's peers usually keep is absent). Experimental, default-OFF.
        instance_threshold: Threshold for flagging instances (0-1).
            Higher = fewer flags, lower = more flags.
        frame_threshold: Threshold for frame-level checks.
        duplicate_iou_threshold: IOU threshold for duplicate detection.
        duplicate_node_overlap_ratio: Node overlap ratio for partial duplicates.
        chirality_flip_threshold: ``chirality_wrong_fraction`` at/above which an
            instance is force-flagged as a whole-instance L/R flip.
        duplicate_score_threshold: Combined duplicate-score at/above which a
            pair is flagged as a duplicate at the frame level.
        chain_turn_angle_deg: Per-interior-node turning angle (degrees) above
            which a chain node counts as an ordering inversion.
        order_inversion_threshold: ``order_inversion_rate`` at/above which an
            instance is force-flagged as having wrong keypoint order.
        missing_node_prob_threshold: Minimum expected-visibility probability for
            a missing node to be flagged as suspicious.
        ordered_chains: User-defined ordered chains as lists of node *names*
            (ground truth ordering for the chain-ordering detector). Empty =
            fall back to auto-detected skeleton chains.
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
    # New detectors: reliable ones default-ON, experimental ones default-OFF.
    use_chirality: Literal["auto"] | bool = "auto"  # (c)
    use_split_detection: bool = True  # (d)
    use_duplicate_score: bool = True  # (a)
    use_chain_ordering: Literal["auto"] | bool = False  # (b) experimental
    use_missing_node_check: bool = False  # (f, Tier-1) experimental

    # Thresholds (validated in v4 investigation)
    instance_threshold: float = 0.7  # Default: balanced
    frame_threshold: float = 0.5
    duplicate_iou_threshold: float = 0.5
    duplicate_node_overlap_ratio: float = 0.8
    duplicate_node_distance_threshold: float = 10.0

    # New-detector thresholds.
    chirality_flip_threshold: float = 0.5
    duplicate_score_threshold: float = 0.5
    chain_turn_angle_deg: float = 60.0
    order_inversion_threshold: float = 0.3
    missing_node_prob_threshold: float = 0.9

    # User-defined ordered chains (lists of node NAMES) for chain-ordering.
    ordered_chains: list = field(default_factory=list)

    # GMM settings
    gmm_n_components: int = 5
    gmm_min_samples: int = 50
    gmm_percentile_threshold: float = 5.0

    # Calibration (reserved: NOT consumed anywhere yet — no auto-calibration is
    # implemented. Flagging uses the fixed instance_threshold / GUI slider value.)
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

    def should_use_chirality(self, has_symmetry: bool) -> bool:
        """Determine if the chirality (L/R mirror-flip) feature should be used."""
        if isinstance(self.use_chirality, bool):
            return self.use_chirality
        # Auto mode: enable if skeleton has symmetry pairs (defined or inferred).
        return has_symmetry

    def should_use_chain_ordering(self, max_chain_length: int) -> bool:
        """Determine if the chain-ordering feature should be used."""
        if isinstance(self.use_chain_ordering, bool):
            return self.use_chain_ordering
        # Auto mode: enable for chains >= 4 nodes (need an interior turning angle).
        return max_chain_length >= 4

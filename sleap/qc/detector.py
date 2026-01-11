"""Main Label QC Detector class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from sleap.qc.config import QCConfig
from sleap.qc.features.baseline import BaselineFeatureExtractor, BASELINE_FEATURE_NAMES
from sleap.qc.features.skeleton import SkeletonAnalyzer
from sleap.qc.features.structural import compute_curvature, compute_convex_hull
from sleap.qc.features.visibility import VisibilityModel
from sleap.qc.features.reference import NearestNeighborScorer
from sleap.qc.frame_level import (
    InstanceCountChecker,
    detect_duplicates,
)
from sleap.qc.gmm import GMMDetector, ZScoreDetector
from sleap.qc.results import QCResults, FrameQC, InstanceKey, FrameKey

if TYPE_CHECKING:
    import sleap_io as sio


# Additional feature names for v3 features
V3_FEATURE_NAMES = [
    "max_curvature",
    "curvature_std",
    "visibility_pattern_score",
    "nn_distance",
    "hull_area",
    "hull_compactness",
]


class LabelQCDetector:
    """Main detection interface for Label QC.

    This class provides the primary API for detecting annotation errors
    in pose labeling data.

    Example:
        detector = LabelQCDetector()
        detector.fit(labels)
        results = detector.score(labels)
        flagged = results.get_flagged(threshold=0.7)

    Attributes:
        config: Configuration for the detector.
        skeleton_analyzer: Analyzer for skeleton properties.
        baseline_extractor: Baseline feature extractor.
        gmm_detector: GMM-based anomaly detector.
        zscore_detector: Fallback z-score detector.
        visibility_model: Visibility pattern model.
        nn_scorer: Nearest neighbor scorer.
        instance_count_checker: Frame-level instance count checker.
        use_gmm: Whether GMM is being used (vs fallback).
        feature_names: Combined list of feature names.
    """

    def __init__(self, config: Optional[QCConfig] = None):
        """Initialize detector with optional config.

        Args:
            config: Configuration for the detector. If None, uses defaults.
        """
        self.config = config or QCConfig()

        # These will be set during fit()
        self.skeleton_analyzer: Optional[SkeletonAnalyzer] = None
        self.baseline_extractor: Optional[BaselineFeatureExtractor] = None
        self.gmm_detector: Optional[GMMDetector] = None
        self.zscore_detector: Optional[ZScoreDetector] = None
        self.visibility_model: Optional[VisibilityModel] = None
        self.nn_scorer: Optional[NearestNeighborScorer] = None
        self.instance_count_checker: Optional[InstanceCountChecker] = None

        self.use_gmm: bool = True
        self.feature_names: list[str] = []

        # Cache for computed statistics
        self._hull_stats: Optional[dict] = None

    def fit(self, labels: "sio.Labels") -> "LabelQCDetector":
        """Fit detector on labels (uses user-labeled instances).

        Args:
            labels: Labels object containing annotated instances.

        Returns:
            Self for chaining.
        """
        if not labels.skeletons:
            raise ValueError("Labels must have at least one skeleton")

        skeleton = labels.skeletons[0]
        self.skeleton_analyzer = SkeletonAnalyzer(skeleton)

        # Collect all instances as arrays
        instances = self._collect_instances(labels)
        if len(instances) == 0:
            raise ValueError("No instances found in labels")

        # Fit baseline feature extractor
        self.baseline_extractor = BaselineFeatureExtractor(
            edges=self.skeleton_analyzer.edges,
            n_nodes=self.skeleton_analyzer.n_nodes,
            symmetry_pairs=self.skeleton_analyzer.symmetry_pairs,
        )
        self.baseline_extractor.fit(instances)

        # Fit visibility model
        visibility_masks = self._get_visibility_masks(instances)
        self.visibility_model = VisibilityModel()
        self.visibility_model.fit(visibility_masks)

        # Fit NN scorer
        self.nn_scorer = NearestNeighborScorer(normalize=True)
        self.nn_scorer.fit(np.array(instances))

        # Compute leave-one-out NN distances for training
        # (so training features are comparable to test features)
        self._training_nn_distances = self._compute_loo_nn_distances(instances)

        # Compute hull statistics for z-scoring
        hull_areas = []
        for inst in instances:
            hull = compute_convex_hull(inst)
            if hull["hull_area"] > 0:
                hull_areas.append(hull["hull_area"])
        self._hull_stats = {
            "mean": np.mean(hull_areas) if hull_areas else 1.0,
            "std": np.std(hull_areas) if hull_areas else 1.0,
        }

        # Build feature matrix (use LOO NN distances for training)
        self.feature_names = self._get_feature_names()  # Set first, needed by extract
        feature_matrix = self._extract_all_features(instances, use_loo_nn=True)

        # Decide between GMM and fallback
        n_samples = len(instances)
        if n_samples >= self.config.gmm_min_samples and self.config.use_gmm:
            self.use_gmm = True
            self.gmm_detector = GMMDetector(
                n_components=self.config.gmm_n_components,
                percentile_threshold=self.config.gmm_percentile_threshold,
            )
            self.gmm_detector.fit(feature_matrix, self.feature_names)
        else:
            self.use_gmm = False
            self.zscore_detector = ZScoreDetector(threshold=3.0)
            self.zscore_detector.fit(feature_matrix)

        # Fit instance count checker
        frame_counts, video_ids = self._collect_frame_counts(labels)
        self.instance_count_checker = InstanceCountChecker(per_video=True)
        self.instance_count_checker.fit(frame_counts, video_ids)

        return self

    def score(self, labels: "sio.Labels") -> QCResults:
        """Score all instances and return results.

        Args:
            labels: Labels object to score.

        Returns:
            QCResults containing instance scores, frame results, and
            feature contributions.
        """
        if self.baseline_extractor is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        results = QCResults(feature_names=self.feature_names)

        # Score all instances
        for video_idx, video in enumerate(labels.videos):
            video_id = video.filename if video.filename else str(video_idx)
            labeled_frames = [lf for lf in labels if lf.video == video]

            for lf in labeled_frames:
                frame_idx = lf.frame_idx

                # Collect instances for this frame
                frame_instances = []
                for inst_idx, inst in enumerate(lf.instances):
                    points = self._instance_to_array(inst)
                    frame_instances.append(points)

                    # Score instance
                    key = InstanceKey(video_idx, frame_idx, inst_idx)
                    features = self._extract_features(points)
                    score, contributions = self._score_instance(features)

                    results.instance_scores[key] = score
                    results.feature_contributions[key] = contributions

                # Frame-level checks
                frame_key = FrameKey(video_idx, frame_idx)
                frame_qc = self._check_frame(frame_instances, video_id)
                results.frame_results[frame_key] = frame_qc

        return results

    def flag(
        self, labels: "sio.Labels", threshold: Optional[float] = None
    ) -> list:
        """Return list of flagged instances above threshold.

        Args:
            labels: Labels object to check.
            threshold: Score threshold. If None, uses config default.

        Returns:
            List of QCFlag objects.
        """
        threshold = threshold or self.config.instance_threshold
        results = self.score(labels)
        return results.get_flagged(threshold)

    def _collect_instances(self, labels: "sio.Labels") -> list[np.ndarray]:
        """Collect all instances as numpy arrays."""
        instances = []
        for lf in labels:
            for inst in lf.instances:
                points = self._instance_to_array(inst)
                instances.append(points)
        return instances

    def _instance_to_array(self, instance: "sio.Instance") -> np.ndarray:
        """Convert instance to (n_nodes, 2) array.

        Uses Instance.numpy() which returns invisible points as NaN.
        """
        return instance.numpy(invisible_as_nan=True)

    def _get_visibility_masks(
        self, instances: list[np.ndarray]
    ) -> np.ndarray:
        """Get visibility masks for all instances."""
        masks = []
        for inst in instances:
            mask = ~np.isnan(inst).any(axis=1)
            masks.append(mask)
        return np.array(masks)

    def _extract_features(self, points: np.ndarray) -> np.ndarray:
        """Extract combined feature vector for a single instance."""
        # Baseline features
        baseline = self.baseline_extractor.extract(points)

        # V3 features
        v3_features = []

        # Curvature
        if self.config.should_use_curvature(
            self.skeleton_analyzer.max_chain_length
        ):
            chains = self.skeleton_analyzer.get_curvature_chains()
            if chains:
                curv = compute_curvature(points, chains[0])
                v3_features.extend([curv["max_curvature"], curv["curvature_std"]])
            else:
                v3_features.extend([0.0, 0.0])
        else:
            v3_features.extend([0.0, 0.0])

        # Visibility pattern
        vis_mask = ~np.isnan(points).any(axis=1)
        vis_result = self.visibility_model.score(vis_mask)
        v3_features.append(vis_result["pattern_score"])

        # NN distance
        nn_result = self.nn_scorer.score(points)
        v3_features.append(nn_result["nn_distance"])

        # Hull features
        hull = compute_convex_hull(points)
        hull_area_z = (
            (hull["hull_area"] - self._hull_stats["mean"])
            / max(self._hull_stats["std"], 1e-6)
        )
        v3_features.extend([hull_area_z, hull["compactness"]])

        return np.concatenate([baseline, np.array(v3_features)])

    def _extract_all_features(
        self, instances: list[np.ndarray], use_loo_nn: bool = False
    ) -> np.ndarray:
        """Extract features for all instances.

        Args:
            instances: List of pose arrays.
            use_loo_nn: If True, use leave-one-out NN distances (for training).
        """
        features = []
        for i, inst in enumerate(instances):
            feat = self._extract_features(inst)
            # Replace NN distance with LOO version during training
            if use_loo_nn and hasattr(self, "_training_nn_distances"):
                nn_dist_idx = self.feature_names.index("nn_distance") if self.feature_names else 15
                feat[nn_dist_idx] = self._training_nn_distances[i]
            features.append(feat)
        return np.array(features)

    def _compute_loo_nn_distances(
        self, instances: list[np.ndarray]
    ) -> list[float]:
        """Compute leave-one-out nearest neighbor distances.

        For each instance, finds distance to nearest OTHER instance.
        """
        from sleap.qc.features.reference import normalize_pose, pose_distance

        n = len(instances)
        normalized = [normalize_pose(inst) for inst in instances]
        loo_distances = []

        for i in range(n):
            min_dist = float("inf")
            for j in range(n):
                if i == j:
                    continue
                dist = pose_distance(normalized[i], normalized[j], method="euclidean")
                if dist < min_dist:
                    min_dist = dist
            loo_distances.append(min_dist if np.isfinite(min_dist) else 0.0)

        return loo_distances

    def _get_feature_names(self) -> list[str]:
        """Get combined feature names."""
        return BASELINE_FEATURE_NAMES + V3_FEATURE_NAMES

    def _score_instance(
        self, features: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Score an instance and return contributions."""
        # Handle NaN in features
        features_clean = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.use_gmm:
            result = self.gmm_detector.score(features_clean)
            score = result["normalized_score"]
        else:
            scores = self.zscore_detector.score_batch(
                features_clean.reshape(1, -1)
            )
            score = scores[0] if len(scores) > 0 else 0.0

        # Build contributions dict
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(features[i]) if i < len(features) else 0.0

        return float(score) if np.isfinite(score) else 0.0, contributions

    def _check_frame(
        self, instances: list[np.ndarray], video_id: str
    ) -> FrameQC:
        """Check frame-level quality."""
        frame_qc = FrameQC()

        # Instance count check
        count_result = self.instance_count_checker.check(
            len(instances), video_id
        )
        frame_qc.is_incomplete = count_result["is_incomplete"]
        frame_qc.expected_instance_count = int(count_result["expected_count"])
        frame_qc.actual_instance_count = len(instances)

        # Duplicate detection
        if len(instances) >= 2:
            duplicates = detect_duplicates(
                instances,
                iou_threshold=self.config.duplicate_iou_threshold,
                node_distance_threshold=self.config.duplicate_node_distance_threshold,
                node_overlap_ratio=self.config.duplicate_node_overlap_ratio,
            )
            for dup in duplicates:
                frame_qc.duplicate_pairs.append((dup["index_a"], dup["index_b"]))
                frame_qc.duplicate_reasons.append(dup["reason"])

        return frame_qc

    def _collect_frame_counts(
        self, labels: "sio.Labels"
    ) -> tuple[list[int], list[str]]:
        """Collect instance counts per frame."""
        counts = []
        video_ids = []
        for video_idx, video in enumerate(labels.videos):
            video_id = video.filename if video.filename else str(video_idx)
            labeled_frames = [lf for lf in labels if lf.video == video]

            for lf in labeled_frames:
                counts.append(len(lf.instances))
                video_ids.append(video_id)

        return counts, video_ids

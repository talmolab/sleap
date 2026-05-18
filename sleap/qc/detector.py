"""Main Label QC Detector class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

import numpy as np

from sleap.qc.config import QCConfig
from sleap.qc.features.baseline import BaselineFeatureExtractor, BASELINE_FEATURE_NAMES
from sleap.qc.features.skeleton import SkeletonAnalyzer
from sleap.qc.features.structural import compute_curvature, compute_convex_hull
from sleap.qc.features.visibility import VisibilityModel
from sleap.qc.features.reference import NearestNeighborScorer, normalize_pose
from sleap.qc.frame_level import (
    InstanceCountChecker,
    detect_duplicates,
)
from sleap.qc.gmm import GMMDetector, ZScoreDetector
from sleap.qc.results import QCResults, FrameQC, InstanceKey, FrameKey

if TYPE_CHECKING:
    import sleap_io as sio

# Progress callback type: (step_name, progress_fraction, detail_message)
ProgressCallback = Callable[[str, float, Optional[str]], None]


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

    def fit(
        self,
        labels: "sio.Labels",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "LabelQCDetector":
        """Fit detector on labels (uses user-labeled instances).

        Args:
            labels: Labels object containing annotated instances.
            progress_callback: Optional callback for progress updates.
                Called with (step_name, progress_fraction, detail_message).

        Returns:
            Self for chaining.
        """

        def _report(step: str, progress: float, detail: str = None):
            if progress_callback:
                progress_callback(step, progress, detail)

        if not labels.skeletons:
            raise ValueError("Labels must have at least one skeleton")

        skeleton = labels.skeletons[0]
        self.skeleton_analyzer = SkeletonAnalyzer(skeleton)

        # Collect all instances as arrays
        _report("Collecting instances", 0.0, None)
        instances = self._collect_instances(labels)
        if len(instances) == 0:
            raise ValueError("No instances found in labels")
        _report("Collecting instances", 0.05, f"{len(instances)} instances")

        # Fit baseline feature extractor
        _report("Fitting feature extractors", 0.05, "Baseline features")
        self.baseline_extractor = BaselineFeatureExtractor(
            edges=self.skeleton_analyzer.edges,
            n_nodes=self.skeleton_analyzer.n_nodes,
            symmetry_pairs=self.skeleton_analyzer.symmetry_pairs,
        )
        self.baseline_extractor.fit(instances)

        # Fit visibility model
        _report("Fitting feature extractors", 0.08, "Visibility model")
        visibility_masks = self._get_visibility_masks(instances)
        self.visibility_model = VisibilityModel()
        self.visibility_model.fit(visibility_masks)

        # Fit NN scorer
        _report("Fitting feature extractors", 0.10, "Nearest neighbor scorer")
        self.nn_scorer = NearestNeighborScorer(normalize=True)
        self.nn_scorer.fit(np.array(instances))

        # Compute leave-one-out NN distances for training using fast KD-tree method
        # (so training features are comparable to test features)
        _report("Computing nearest neighbors", 0.12, "Building KD-tree")
        self._training_nn_distances = self._compute_loo_nn_distances_fast(instances)
        _report("Computing nearest neighbors", 0.15, "Done")

        # Compute hull statistics for z-scoring
        _report("Computing hull statistics", 0.15, None)
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
        _report("Extracting features", 0.20, f"0/{len(instances)}")
        self.feature_names = self._get_feature_names()  # Set first, needed by extract
        feature_matrix = self._extract_all_features(
            instances, use_loo_nn=True, progress_callback=progress_callback
        )

        # Decide between GMM and fallback
        n_samples = len(instances)
        if n_samples >= self.config.gmm_min_samples and self.config.use_gmm:
            _report("Fitting detection model", 0.70, "GMM with EM algorithm")
            self.use_gmm = True
            self.gmm_detector = GMMDetector(
                n_components=self.config.gmm_n_components,
                percentile_threshold=self.config.gmm_percentile_threshold,
            )
            self.gmm_detector.fit(feature_matrix, self.feature_names)
        else:
            _report("Fitting detection model", 0.70, "Z-score fallback")
            self.use_gmm = False
            self.zscore_detector = ZScoreDetector(threshold=3.0)
            self.zscore_detector.fit(feature_matrix)
        _report("Fitting detection model", 0.75, "Done")

        # Fit instance count checker
        _report("Fitting frame-level checkers", 0.75, None)
        frame_counts, video_ids = self._collect_frame_counts(labels)
        self.instance_count_checker = InstanceCountChecker(per_video=True)
        self.instance_count_checker.fit(frame_counts, video_ids)
        _report("Fitting complete", 0.80, None)

        return self

    def score(
        self,
        labels: "sio.Labels",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> QCResults:
        """Score all instances and return results.

        Args:
            labels: Labels object to score.
            progress_callback: Optional callback for progress updates.
                Called with (step_name, progress_fraction, detail_message).

        Returns:
            QCResults containing instance scores, frame results, and
            feature contributions.
        """

        def _report(step: str, progress: float, detail: str = None):
            if progress_callback:
                progress_callback(step, progress, detail)

        if self.baseline_extractor is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        results = QCResults(feature_names=self.feature_names)

        # Count total instances for progress
        total_instances = sum(len(lf.instances) for lf in labels)
        instance_count = 0

        # Score all instances
        _report("Scoring instances", 0.80, f"0/{total_instances}")
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

                    # Progress update (every 500 instances)
                    instance_count += 1
                    if instance_count % 500 == 0:
                        progress = 0.80 + 0.18 * (instance_count / total_instances)
                        msg = f"{instance_count}/{total_instances}"
                        _report("Scoring instances", progress, msg)

                # Frame-level checks
                frame_key = FrameKey(video_idx, frame_idx)
                frame_qc = self._check_frame(frame_instances, video_id)
                results.frame_results[frame_key] = frame_qc

        _report("Complete", 1.0, f"{instance_count} instances scored")
        return results

    def flag(self, labels: "sio.Labels", threshold: Optional[float] = None) -> list:
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

        Uses Instance.numpy() which returns invisible points as NaN by default.
        Feature extractors handle NaN values by skipping them in computations.
        """
        return instance.numpy()

    def _get_visibility_masks(self, instances: list[np.ndarray]) -> np.ndarray:
        """Get visibility masks for all instances."""
        masks = []
        for inst in instances:
            mask = ~np.isnan(inst).any(axis=1)
            masks.append(mask)
        return np.array(masks)

    def _extract_features(
        self, points: np.ndarray, nn_distance: Optional[float] = None
    ) -> np.ndarray:
        """Extract combined feature vector for a single instance.

        Args:
            points: (N_nodes, 2) array of coordinates.
            nn_distance: Optional precomputed NN distance (skips slow NN query).
        """
        # Baseline features
        baseline = self.baseline_extractor.extract(points)

        # V3 features
        v3_features = []

        # Curvature
        if self.config.should_use_curvature(self.skeleton_analyzer.max_chain_length):
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

        # NN distance (use precomputed if available)
        if nn_distance is not None:
            v3_features.append(nn_distance)
        else:
            nn_result = self.nn_scorer.score(points)
            v3_features.append(nn_result["nn_distance"])

        # Hull features
        hull = compute_convex_hull(points)
        hull_area_z = (hull["hull_area"] - self._hull_stats["mean"]) / max(
            self._hull_stats["std"], 1e-6
        )
        v3_features.extend([hull_area_z, hull["compactness"]])

        return np.concatenate([baseline, np.array(v3_features)])

    def _extract_all_features(
        self,
        instances: list[np.ndarray],
        use_loo_nn: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        """Extract features for all instances.

        Uses batch NN scoring for O(n log n) performance instead of O(n²).

        Args:
            instances: List of pose arrays.
            use_loo_nn: If True, use leave-one-out NN distances (for training).
            progress_callback: Optional callback for progress updates.
        """

        def _report(step: str, progress: float, detail: str = None):
            if progress_callback:
                progress_callback(step, progress, detail)

        n = len(instances)

        # Pre-compute all NN distances in batch (fast KD-tree query)
        if use_loo_nn and hasattr(self, "_training_nn_distances"):
            # Use precomputed LOO distances for training
            nn_distances = self._training_nn_distances
        else:
            # Batch query for scoring (not LOO)
            _report("Computing NN distances", 0.20, f"Batch query for {n} instances")
            nn_distances = self.nn_scorer.score_batch(np.array(instances))

        # Extract features with precomputed NN distances
        features = []
        for i, inst in enumerate(instances):
            feat = self._extract_features(inst, nn_distance=nn_distances[i])
            features.append(feat)

            # Progress update (every 1000 instances)
            if (i + 1) % 1000 == 0:
                progress = 0.20 + 0.50 * ((i + 1) / n)
                _report("Extracting features", progress, f"{i + 1}/{n}")

        return np.array(features)

    def _compute_loo_nn_distances_fast(
        self, instances: list[np.ndarray]
    ) -> list[float]:
        """Compute leave-one-out nearest neighbor distances using KD-tree.

        Uses sklearn's NearestNeighbors with k=2 to efficiently find
        each instance's nearest neighbor (excluding itself).

        This is O(n log n) vs O(n^2) for the naive approach.

        For each instance, finds distance to nearest OTHER instance.

        Args:
            instances: List of (n_nodes, 2) pose arrays.

        Returns:
            List of LOO NN distances.
        """
        from sklearn.neighbors import NearestNeighbors

        # Normalize poses
        normalized = [normalize_pose(inst) for inst in instances]

        # Flatten and impute NaN with 0 for KD-tree
        # (NaN handling is approximate but maintains rank ordering)
        flattened = []
        for norm in normalized:
            flat = norm.flatten()
            flat = np.nan_to_num(flat, nan=0.0)
            flattened.append(flat)
        X = np.array(flattened)

        # Use KD-tree with k=2 (self + nearest other)
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # distances[:,0] is distance to self (0)
        # distances[:,1] is distance to nearest neighbor
        return distances[:, 1].tolist()

    def _compute_loo_nn_distances(self, instances: list[np.ndarray]) -> list[float]:
        """Compute leave-one-out nearest neighbor distances (naive O(n^2)).

        For each instance, finds distance to nearest OTHER instance.

        Note: For datasets > 1000 instances, use _compute_loo_nn_distances_fast
        instead which uses KD-tree for O(n log n) performance.
        """
        from sleap.qc.features.reference import pose_distance

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

    def _score_instance(self, features: np.ndarray) -> tuple[float, dict[str, float]]:
        """Score an instance and return contributions."""
        # Handle NaN in features
        features_clean = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.use_gmm:
            result = self.gmm_detector.score(features_clean)
            score = result["normalized_score"]
        else:
            scores = self.zscore_detector.score_batch(features_clean.reshape(1, -1))
            score = scores[0] if len(scores) > 0 else 0.0

        # Build contributions dict
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(features[i]) if i < len(features) else 0.0

        return float(score) if np.isfinite(score) else 0.0, contributions

    def _check_frame(self, instances: list[np.ndarray], video_id: str) -> FrameQC:
        """Check frame-level quality."""
        frame_qc = FrameQC()

        # Instance count check
        count_result = self.instance_count_checker.check(len(instances), video_id)
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

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
from sleap.qc.features.chirality import (
    fit_chirality,
    compute_chirality,
    infer_symmetry_pairs_by_name,
)
from sleap.qc.features.pose_split import compute_pose_split
from sleap.qc.features.ordering import compute_chain_ordering, resolve_chains
from sleap.qc.features.missing_node import score_missing_nodes
from sleap.qc.features.appearance import fit_appearance, score_appearance
from sleap.qc.insample_prediction import run_insample_prediction
from sleap.qc.frame_level import (
    InstanceCountChecker,
    check_negative_frame,
    detect_duplicates,
)
from sleap.qc.gmm import GMMDetector, ZScoreDetector
from sleap.qc.results import QCResults, FrameQC, InstanceKey, FrameKey

if TYPE_CHECKING:
    import sleap_io as sio

# Progress callback type: (step_name, progress_fraction, detail_message)
ProgressCallback = Callable[[str, float, Optional[str]], None]


# Additional feature names for v3 features.
#
# IMPORTANT: the order here is load-bearing. ``_extract_features`` appends the
# per-instance feature values in exactly this order, and ``_score_instance``
# maps contributions back to names positionally. Any change here MUST be
# mirrored by the append order in ``_extract_features`` or fit/score widths and
# the contribution mapping will diverge.
V3_FEATURE_NAMES = [
    "max_curvature",
    "curvature_std",
    "visibility_pattern_score",
    "nn_distance",
    "hull_area_zscore",
    "hull_compactness",
    # B1 detectors (appended after hull_compactness, in this exact order):
    "chirality_wrong_fraction",  # (c) whole-instance L/R flip
    "pose_split_score",  # (d) chimera / pose-split (log1p-compressed)
    "order_inversion_rate",  # (b) chain ordering
    "chain_intersection_count",  # (b) chain ordering
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

        # B1 detector fit-time state (set in fit(), consumed in _extract_features
        # and score()). Initialized empty so _extract_features is safe even if
        # called before fit() sets them.
        self._chirality_model: Optional[dict] = None
        self._symmetry_pairs: list[tuple[int, int]] = []
        self._axis_nodes: Optional[tuple[int, int]] = None
        self._ordering_chains: list[list[int]] = []
        self._adjacency: Optional[dict[int, list[int]]] = None
        self._co_visibility: Optional[np.ndarray] = None

        # B2 appearance-channel fit-time state (set in fit() when
        # use_appearance is on, consumed in score()). None = no appearance model.
        self._appearance_model: Optional[dict] = None

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

        # B1 fit-time setup. These MUST exist before _extract_all_features runs,
        # since _extract_features reads them while building the feature matrix.
        _report("Fitting feature extractors", 0.16, "B1 detectors")
        sa = self.skeleton_analyzer
        self._symmetry_pairs = list(sa.symmetry_pairs) or infer_symmetry_pairs_by_name(
            sa.node_names
        )
        # Body axis for chirality must run along MIDLINE (non-symmetric) nodes.
        # The skeleton's longest graph path can end at a side node (e.g. a
        # Haunch_left leaf), which biases the axis to one side and makes the
        # signed-side chirality meaningless (-> false whole-instance-flip flags).
        # Restrict the spine to non-symmetric nodes for the axis.
        _sym_idxs = {i for pair in self._symmetry_pairs for i in pair}
        _midline = [i for i in sa.spine if i not in _sym_idxs]
        if len(_midline) >= 2:
            self._axis_nodes = (_midline[0], _midline[-1])
        elif len(sa.spine) >= 2:
            self._axis_nodes = (sa.spine[0], sa.spine[-1])
        else:
            self._axis_nodes = None
        self._adjacency = sa.get_adjacency()
        self._ordering_chains = resolve_chains(
            sa.node_names, self.config.ordered_chains or None, sa.get_curvature_chains()
        )
        self._co_visibility = self.visibility_model.co_visibility_matrix
        if self.config.should_use_chirality(len(self._symmetry_pairs) >= 1):
            self._chirality_model = fit_chirality(
                instances, self._symmetry_pairs, self._axis_nodes
            )

        # B2 appearance channel (experimental, default-OFF): build a per-node
        # appearance model from the labeled frames. Guarded by use_appearance so
        # the default path never touches (potentially expensive) video decoding.
        # Each labeled frame is decoded ONCE; undecodable frames are skipped.
        if self.config.use_appearance:
            _report("Fitting feature extractors", 0.18, "Appearance model")
            appearance_pairs = []
            for video in labels.videos:
                for lf in [lf for lf in labels if lf.video == video]:
                    try:
                        frame = video[lf.frame_idx]
                    except Exception:
                        continue
                    for inst in lf.user_instances:
                        appearance_pairs.append(
                            (frame, inst.numpy(invisible_as_nan=True))
                        )
            self._appearance_model = fit_appearance(
                appearance_pairs,
                n_nodes=self.skeleton_analyzer.n_nodes,
                patch_size=self.config.appearance_patch_size,
                min_samples=self.config.appearance_min_samples,
            )

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
        total_instances = sum(len(lf.user_instances) for lf in labels)
        instance_count = 0

        # Score all instances
        _report("Scoring instances", 0.80, f"0/{total_instances}")
        for video_idx, video in enumerate(labels.videos):
            video_id = self._video_id(video, video_idx)
            labeled_frames = [lf for lf in labels if lf.video == video]

            for lf in labeled_frames:
                frame_idx = lf.frame_idx

                # Decode the frame ONCE per labeled frame for the appearance
                # channel (experimental). Hoisted out of the instance loop so a
                # frame is never decoded more than once; undecodable -> None.
                appearance_frame = None
                if self.config.use_appearance and self._appearance_model is not None:
                    try:
                        appearance_frame = lf.video[frame_idx]
                    except Exception:
                        appearance_frame = None

                # Collect instances for this frame
                frame_instances = []
                for inst_idx, inst in enumerate(lf.user_instances):
                    points = self._instance_to_array(inst)
                    frame_instances.append(points)

                    # Score instance
                    key = InstanceKey(video_idx, frame_idx, inst_idx)
                    features = self._extract_features(points)
                    score, contributions = self._score_instance(features)

                    # Pop the forced-issue marker before contributions are
                    # stored, so feature_contributions stays pure floats.
                    forced_issue = contributions.pop("_forced_top_issue", None)

                    results.instance_scores[key] = score
                    results.feature_contributions[key] = contributions

                    if forced_issue is not None:
                        results.forced_issues[key] = forced_issue

                    # Missing-node channel (experimental): scored separately from
                    # the GMM and merged in QCResults.get_flagged.
                    if (
                        self.config.use_missing_node_check
                        and self._co_visibility is not None
                    ):
                        _vmask = ~np.isnan(points).any(axis=1)
                        _mn = score_missing_nodes(
                            _vmask,
                            self._co_visibility,
                            self.skeleton_analyzer.edges,
                            threshold=self.config.missing_node_prob_threshold,
                        )
                        if _mn["missing_node_score"] > 0:
                            results.channel_scores.setdefault("missing_node", {})[
                                key
                            ] = _mn["missing_node_score"]

                    # Appearance channel (experimental): scored against the
                    # per-node appearance model using the once-decoded frame.
                    if (
                        self.config.use_appearance
                        and self._appearance_model is not None
                        and appearance_frame is not None
                    ):
                        _ap = score_appearance(
                            appearance_frame, points, self._appearance_model
                        )
                        if _ap["appearance_outlier_score"] > 0:
                            results.channel_scores.setdefault("appearance", {})[key] = (
                                _ap["appearance_outlier_score"]
                            )

                    # Progress update (every 500 instances)
                    instance_count += 1
                    if instance_count % 500 == 0:
                        progress = 0.80 + 0.18 * (instance_count / total_instances)
                        msg = f"{instance_count}/{total_instances}"
                        _report("Scoring instances", progress, msg)

                # Frame-level checks
                frame_key = FrameKey(video_idx, frame_idx)
                frame_qc = self._check_frame(
                    frame_instances, video_id, is_negative=lf.is_negative
                )
                results.frame_results[frame_key] = frame_qc

        # In-sample model-prediction channel (experimental, Tier-2 missing-node):
        # ONE batched inference over ALL labeled frames, run after the per-instance
        # loop completes. run_insample_prediction self-skips (returns an empty
        # instance_scores) when the model path is falsy, so guarding only on
        # use_insample_prediction is safe and avoids real inference by default.
        if self.config.use_insample_prediction:
            out = run_insample_prediction(
                labels,
                model_path=self.config.insample_model_path or "",
                peak_threshold=self.config.insample_peak_threshold,
                min_confidence=self.config.insample_min_confidence,
                device=self.config.insample_device,
                progress_callback=progress_callback,
            )
            for (v_idx, f_idx, i_idx), s in out["instance_scores"].items():
                results.channel_scores.setdefault("prediction", {})[
                    InstanceKey(v_idx, f_idx, i_idx)
                ] = s

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
            for inst in lf.user_instances:
                points = self._instance_to_array(inst)
                instances.append(points)
        return instances

    def _instance_to_array(self, instance: "sio.Instance") -> np.ndarray:
        """Convert instance to (n_nodes, 2) array with invisible points as NaN.

        Explicitly passes ``invisible_as_nan=True`` instead of relying on the
        sleap-io default. Invisible (``visible=False``) nodes must never
        contribute their stored coordinates to QC geometry features: those
        coordinates are display-only placeholders (the GUI has to draw an
        invisible node *somewhere*), and older sleap-io versions defaulted to
        returning them, which leaked far-off invisible-node coordinates into
        the edge/angle/distance/hull statistics (see #2753).

        Feature extractors treat NaN as "missing" and skip those nodes, while
        the downstream visibility mask (``~np.isnan(...)``) still records that
        the node is invisible, so the visibility-pattern features keep working.
        """
        return instance.numpy(invisible_as_nan=True)

    @staticmethod
    def _video_id(video: "sio.Video", video_idx: int) -> str:
        """Return a stable, hashable identifier for a video.

        ``Video.filename`` is a list of paths for image-sequence backends
        (e.g. ``ImageVideo``, as produced by CVAT/COCO imports). A list is
        unhashable, so it cannot be used as a dict key for the per-video
        grouping in the frame-level checks. Fall back to the video index,
        which is unique and stable across ``fit``/``score``.

        Args:
            video: The video to identify.
            video_idx: Index of the video within ``labels.videos``.

        Returns:
            The filename when it is a non-empty string, otherwise the video
            index as a string.
        """
        filename = getattr(video, "filename", None)
        if isinstance(filename, str) and filename:
            return filename
        return str(video_idx)

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

        # --- B1 detectors. Each block ALWAYS appends a fixed number of values
        # (emitting 0.0 defaults when its flag is off), so the feature-vector
        # width is identical at fit and score time. The append order MUST match
        # V3_FEATURE_NAMES exactly. ---

        # (c) chirality / whole-instance L/R flip
        if self._chirality_model is not None:
            v3_features.append(
                compute_chirality(
                    points,
                    self._symmetry_pairs,
                    self._axis_nodes,
                    self._chirality_model,
                )["chirality_wrong_fraction"]
            )
        else:
            v3_features.append(0.0)

        # (d) chimera / pose-split — log1p to tame the unbounded dynamic range
        # before the GMM
        if self.config.use_split_detection:
            _ps = compute_pose_split(
                points,
                self._adjacency,
                self.baseline_extractor.stats.edge_means,
                self.baseline_extractor.stats.edge_stds,
            )["split_score"]
            v3_features.append(float(np.log1p(max(_ps, 0.0))))
        else:
            v3_features.append(0.0)

        # (b) chain ordering (experimental)
        if (
            self.config.should_use_chain_ordering(
                self.skeleton_analyzer.max_chain_length
            )
            and self._ordering_chains
        ):
            _ord = compute_chain_ordering(
                points,
                self._ordering_chains,
                max_turn_angle=np.deg2rad(self.config.chain_turn_angle_deg),
            )
            v3_features.extend(
                [_ord["order_inversion_rate"], float(_ord["chain_intersection_count"])]
            )
        else:
            v3_features.extend([0.0, 0.0])

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

        score = float(score) if np.isfinite(score) else 0.0

        # Build contributions dict (raw feature values keyed by name).
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(features[i]) if i < len(features) else 0.0

        # Raise-only hard-rule overrides. These never lower the GMM score; they
        # only force it up (and record a human-readable issue) when an
        # unambiguous structural error is present. The chimera (d) detector gets
        # NO hard rule for now — it relies on its GMM feature
        # (pose_split_score), which is why there is no clause for it here.
        forced = None
        if (
            self._chirality_model is not None
            and contributions.get("chirality_wrong_fraction", 0.0)
            >= self.config.chirality_flip_threshold
        ):
            forced = (
                max(0.9, contributions["chirality_wrong_fraction"]),
                "Whole-instance L/R flip",
            )
        elif self.config.should_use_chain_ordering(
            self.skeleton_analyzer.max_chain_length
        ) and (
            contributions.get("chain_intersection_count", 0.0) >= 1
            or contributions.get("order_inversion_rate", 0.0)
            >= self.config.order_inversion_threshold
        ):
            forced = (0.9, "Wrong keypoint order along chain")

        if forced is not None:
            score = max(score, forced[0])
            contributions["_forced_top_issue"] = forced[1]

        return score, contributions

    def _check_frame(
        self,
        instances: list[np.ndarray],
        video_id: str,
        is_negative: bool = False,
    ) -> FrameQC:
        """Check frame-level quality."""
        frame_qc = FrameQC()

        # Instance count check
        count_result = self.instance_count_checker.check(len(instances), video_id)
        frame_qc.is_incomplete = count_result["is_incomplete"]
        frame_qc.expected_instance_count = int(count_result["expected_count"])
        frame_qc.actual_instance_count = len(instances)

        # Negative (background) frames should have no instances.
        frame_qc.is_negative_with_instances = check_negative_frame(
            is_negative, len(instances)
        )

        # Duplicate detection
        if len(instances) >= 2:
            if self.config.use_duplicate_score:
                duplicates = detect_duplicates(
                    instances,
                    iou_threshold=self.config.duplicate_iou_threshold,
                    node_distance_threshold=(
                        self.config.duplicate_node_distance_threshold
                    ),
                    node_overlap_ratio=self.config.duplicate_node_overlap_ratio,
                    edge_means=self.baseline_extractor.stats.edge_means,
                    duplicate_score_threshold=self.config.duplicate_score_threshold,
                )
            else:
                # Keep current behavior: IOU + node-overlap only. An
                # unreachable score threshold (> the clamped [0, 1] max) keeps
                # the always-computed split-duplicate signal from ever firing.
                duplicates = detect_duplicates(
                    instances,
                    iou_threshold=self.config.duplicate_iou_threshold,
                    node_distance_threshold=(
                        self.config.duplicate_node_distance_threshold
                    ),
                    node_overlap_ratio=self.config.duplicate_node_overlap_ratio,
                    duplicate_score_threshold=float("inf"),
                )
            for dup in duplicates:
                frame_qc.duplicate_pairs.append((dup["index_a"], dup["index_b"]))
                frame_qc.duplicate_reasons.append(dup["reason"])
                frame_qc.duplicate_scores.append(dup.get("duplicate_score", 1.0))

        return frame_qc

    def _collect_frame_counts(
        self, labels: "sio.Labels"
    ) -> tuple[list[int], list[str]]:
        """Collect instance counts per frame."""
        counts = []
        video_ids = []
        for video_idx, video in enumerate(labels.videos):
            video_id = self._video_id(video, video_idx)
            labeled_frames = [lf for lf in labels if lf.video == video]

            for lf in labeled_frames:
                counts.append(len(lf.user_instances))
                video_ids.append(video_id)

        return counts, video_ids

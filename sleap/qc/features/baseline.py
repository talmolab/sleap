"""Baseline feature extraction (v2 features)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Feature names for the baseline feature vector
BASELINE_FEATURE_NAMES = [
    "max_edge_zscore",
    "mean_edge_zscore",
    "max_angle_zscore",
    "mean_angle_zscore",
    "max_pairwise_zscore",
    "mean_pairwise_zscore",
    "bbox_area_zscore",
    "max_centroid_distance",
    "centroid_distance_std",
    "min_symmetry_consistency",
    "visibility_rate",
    "has_isolated_invisible",
]


@dataclass
class DatasetStats:
    """Statistics computed from reference (clean) instances."""

    edge_means: dict[tuple[int, int], float]
    edge_stds: dict[tuple[int, int], float]
    pairwise_means: dict[tuple[int, int], float]
    pairwise_stds: dict[tuple[int, int], float]
    angle_means: dict[tuple[int, int, int], float]
    angle_stds: dict[tuple[int, int, int], float]
    bbox_area_mean: float
    bbox_area_std: float


class BaselineFeatureExtractor:
    """Extract baseline (v2) features from pose instances.

    Features include:
    - Edge length z-scores
    - Joint angle z-scores
    - Pairwise distance z-scores
    - Bounding box area z-score
    - Node isolation (centroid distance)
    - Symmetry consistency
    - Visibility features
    """

    def __init__(
        self,
        edges: list[tuple[int, int]],
        n_nodes: int,
        symmetry_pairs: Optional[list[tuple[int, int]]] = None,
    ):
        """Initialize extractor.

        Args:
            edges: List of edge tuples (src_idx, dst_idx).
            n_nodes: Number of nodes in skeleton.
            symmetry_pairs: Optional list of symmetric node pairs.
        """
        self.edges = edges
        self.n_nodes = n_nodes
        self.symmetry_pairs = symmetry_pairs or []
        self.stats: Optional[DatasetStats] = None
        self._adjacency: Optional[dict[int, list[int]]] = None

    def fit(self, instances: list[np.ndarray]) -> "BaselineFeatureExtractor":
        """Compute statistics from reference instances.

        Args:
            instances: List of (n_nodes, 2) arrays of pose coordinates.
                NaN values indicate invisible nodes.

        Returns:
            Self for chaining.
        """
        # Build adjacency
        self._adjacency = {i: [] for i in range(self.n_nodes)}
        for src, dst in self.edges:
            self._adjacency[src].append(dst)
            self._adjacency[dst].append(src)

        # Collect edge lengths
        edge_lengths: dict[tuple[int, int], list[float]] = {
            tuple(sorted(e)): [] for e in self.edges
        }
        for points in instances:
            for src, dst in self.edges:
                p1, p2 = points[src], points[dst]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    key = tuple(sorted([src, dst]))
                    edge_lengths[key].append(np.linalg.norm(p2 - p1))

        # Collect pairwise distances
        pairwise_dists: dict[tuple[int, int], list[float]] = {}
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                pairwise_dists[(i, j)] = []

        for points in instances:
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    p1, p2 = points[i], points[j]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        pairwise_dists[(i, j)].append(np.linalg.norm(p2 - p1))

        # Collect joint angles
        angle_values: dict[tuple[int, int, int], list[float]] = {}
        for center, neighbors in self._adjacency.items():
            if len(neighbors) < 2:
                continue
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    key = (center, min(n1, n2), max(n1, n2))
                    angle_values[key] = []

        for points in instances:
            for center, neighbors in self._adjacency.items():
                if len(neighbors) < 2:
                    continue
                pc = points[center]
                if np.isnan(pc).any():
                    continue

                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i + 1 :]:
                        p1 = points[n1]
                        p2 = points[n2]
                        if np.isnan(p1).any() or np.isnan(p2).any():
                            continue

                        v1 = p1 - pc
                        v2 = p2 - pc
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 < 1e-6 or norm2 < 1e-6:
                            continue

                        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                        angle = np.arccos(cos_angle)

                        key = (center, min(n1, n2), max(n1, n2))
                        angle_values[key].append(angle)

        # Collect bbox areas
        bbox_areas = []
        for points in instances:
            visible = points[~np.isnan(points[:, 0])]
            if len(visible) >= 2:
                min_xy = visible.min(axis=0)
                max_xy = visible.max(axis=0)
                area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
                bbox_areas.append(area)

        # Compute statistics
        def safe_mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        def safe_std(values: list[float], min_val: float = 1e-6) -> float:
            return max(float(np.std(values)) if values else min_val, min_val)

        self.stats = DatasetStats(
            edge_means={k: safe_mean(v) for k, v in edge_lengths.items()},
            edge_stds={k: safe_std(v) for k, v in edge_lengths.items()},
            pairwise_means={k: safe_mean(v) for k, v in pairwise_dists.items()},
            pairwise_stds={k: safe_std(v) for k, v in pairwise_dists.items()},
            angle_means={k: safe_mean(v) for k, v in angle_values.items()},
            angle_stds={k: safe_std(v) for k, v in angle_values.items()},
            bbox_area_mean=safe_mean(bbox_areas),
            bbox_area_std=safe_std(bbox_areas),
        )

        return self

    def extract(self, points: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single instance.

        Args:
            points: (n_nodes, 2) array of pose coordinates.

        Returns:
            (n_features,) feature vector.
        """
        if self.stats is None:
            raise ValueError("Must call fit() before extract()")

        # Edge features
        edge_zscores = self._compute_edge_zscores(points)
        max_edge_z = (
            float(np.nanmax(np.abs(edge_zscores))) if len(edge_zscores) > 0 else 0.0
        )
        mean_edge_z = (
            float(np.nanmean(np.abs(edge_zscores))) if len(edge_zscores) > 0 else 0.0
        )

        # Angle features
        angle_zscores = self._compute_angle_zscores(points)
        max_angle_z = (
            float(np.nanmax(np.abs(angle_zscores))) if len(angle_zscores) > 0 else 0.0
        )
        mean_angle_z = (
            float(np.nanmean(np.abs(angle_zscores))) if len(angle_zscores) > 0 else 0.0
        )

        # Pairwise features
        pairwise_zscores = self._compute_pairwise_zscores(points)
        max_pw_z = (
            float(np.nanmax(np.abs(pairwise_zscores)))
            if len(pairwise_zscores) > 0
            else 0.0
        )
        mean_pw_z = (
            float(np.nanmean(np.abs(pairwise_zscores)))
            if len(pairwise_zscores) > 0
            else 0.0
        )

        # Bbox features
        bbox_z = self._compute_bbox_zscore(points)

        # Isolation features
        centroid_dists = self._compute_centroid_distances(points)
        max_cent_dist = (
            float(np.nanmax(centroid_dists)) if len(centroid_dists) > 0 else 0.0
        )
        cent_dist_std = (
            float(np.nanstd(centroid_dists)) if len(centroid_dists) > 0 else 0.0
        )

        # Symmetry features
        min_sym = self._compute_symmetry_consistency(points)

        # Visibility features
        vis_rate, has_isolated_inv = self._compute_visibility_features(points)

        return np.array(
            [
                max_edge_z,
                mean_edge_z,
                max_angle_z,
                mean_angle_z,
                max_pw_z,
                mean_pw_z,
                bbox_z,
                max_cent_dist,
                cent_dist_std,
                min_sym,
                vis_rate,
                1.0 if has_isolated_inv else 0.0,
            ]
        )

    def _compute_edge_zscores(self, points: np.ndarray) -> np.ndarray:
        """Compute edge length z-scores."""
        zscores = []
        for src, dst in self.edges:
            p1, p2 = points[src], points[dst]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue

            length = np.linalg.norm(p2 - p1)
            key = tuple(sorted([src, dst]))
            if key in self.stats.edge_means:
                z = (length - self.stats.edge_means[key]) / self.stats.edge_stds[key]
                zscores.append(z)

        return np.array(zscores)

    def _compute_angle_zscores(self, points: np.ndarray) -> np.ndarray:
        """Compute joint angle z-scores."""
        zscores = []
        for center, neighbors in self._adjacency.items():
            if len(neighbors) < 2:
                continue
            pc = points[center]
            if np.isnan(pc).any():
                continue

            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    p1 = points[n1]
                    p2 = points[n2]
                    if np.isnan(p1).any() or np.isnan(p2).any():
                        continue

                    v1 = p1 - pc
                    v2 = p2 - pc
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 < 1e-6 or norm2 < 1e-6:
                        continue

                    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                    key = (center, min(n1, n2), max(n1, n2))
                    if key in self.stats.angle_means:
                        z = (
                            angle - self.stats.angle_means[key]
                        ) / self.stats.angle_stds[key]
                        zscores.append(z)

        return np.array(zscores)

    def _compute_pairwise_zscores(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance z-scores."""
        zscores = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                p1, p2 = points[i], points[j]
                if np.isnan(p1).any() or np.isnan(p2).any():
                    continue

                dist = np.linalg.norm(p2 - p1)
                key = (i, j)
                if key in self.stats.pairwise_means:
                    z = (
                        dist - self.stats.pairwise_means[key]
                    ) / self.stats.pairwise_stds[key]
                    zscores.append(z)

        return np.array(zscores)

    def _compute_bbox_zscore(self, points: np.ndarray) -> float:
        """Compute bounding box area z-score."""
        visible = points[~np.isnan(points[:, 0])]
        if len(visible) < 2:
            return 0.0

        min_xy = visible.min(axis=0)
        max_xy = visible.max(axis=0)
        area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
        return (area - self.stats.bbox_area_mean) / self.stats.bbox_area_std

    def _compute_centroid_distances(self, points: np.ndarray) -> np.ndarray:
        """Compute distance from each node to centroid."""
        visible_mask = ~np.isnan(points[:, 0])
        if visible_mask.sum() < 2:
            return np.array([])

        visible_pts = points[visible_mask]
        centroid = visible_pts.mean(axis=0)

        distances = []
        for i in range(self.n_nodes):
            if visible_mask[i]:
                distances.append(np.linalg.norm(points[i] - centroid))

        return np.array(distances)

    def _compute_symmetry_consistency(self, points: np.ndarray) -> float:
        """Compute minimum symmetry consistency score."""
        if len(self.symmetry_pairs) < 2:
            return 1.0  # No symmetry, assume consistent

        consistency_scores = []
        for i, (l1, r1) in enumerate(self.symmetry_pairs):
            p_l1 = points[l1]
            p_r1 = points[r1]
            if np.isnan(p_l1).any() or np.isnan(p_r1).any():
                continue

            consistent_count = 0
            total_count = 0

            for j, (l2, r2) in enumerate(self.symmetry_pairs):
                if i == j:
                    continue
                p_l2 = points[l2]
                p_r2 = points[r2]
                if np.isnan(p_l2).any() or np.isnan(p_r2).any():
                    continue

                dist_ll = np.linalg.norm(p_l1 - p_l2)
                dist_lr = np.linalg.norm(p_l1 - p_r2)
                ratio = dist_ll / max(dist_lr, 1e-6)

                if ratio < 0.9:
                    consistent_count += 1
                elif ratio <= 1.1:
                    consistent_count += 0.5
                total_count += 1

            if total_count > 0:
                consistency_scores.append(consistent_count / total_count)

        return float(np.min(consistency_scores)) if consistency_scores else 1.0

    def _compute_visibility_features(self, points: np.ndarray) -> tuple[float, bool]:
        """Compute visibility rate and isolated invisible flag."""
        visible_mask = ~np.isnan(points[:, 0])
        vis_rate = visible_mask.sum() / self.n_nodes

        # Check for isolated invisible nodes
        has_isolated_inv = False
        for i in range(self.n_nodes):
            if visible_mask[i]:
                continue
            neighbors = self._adjacency[i]
            if neighbors and all(visible_mask[n] for n in neighbors):
                has_isolated_inv = True
                break

        return vis_rate, has_isolated_inv

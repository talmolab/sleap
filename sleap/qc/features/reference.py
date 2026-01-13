"""Reference-based features: nearest neighbor distance."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_pose(points: np.ndarray) -> np.ndarray:
    """Normalize pose to unit scale and center.

    Args:
        points: (N_nodes, 2) array of coordinates (may contain NaN).

    Returns:
        Normalized points array (NaN preserved).
    """
    visible_mask = ~np.isnan(points).any(axis=1)
    if visible_mask.sum() < 2:
        return points.copy()

    visible_points = points[visible_mask]

    # Center
    centroid = visible_points.mean(axis=0)

    # Scale by bounding box diagonal
    bbox_min = visible_points.min(axis=0)
    bbox_max = visible_points.max(axis=0)
    scale = np.linalg.norm(bbox_max - bbox_min)
    if scale < 1e-6:
        scale = 1.0

    normalized = (points - centroid) / scale
    return normalized


def pose_distance(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    method: str = "euclidean",
) -> float:
    """Compute distance between two poses.

    Args:
        pose_a: (N_nodes, 2) array.
        pose_b: (N_nodes, 2) array.
        method: Distance method ("euclidean", "procrustes").

    Returns:
        Distance value (lower = more similar).
    """
    # Find commonly visible nodes
    visible_a = ~np.isnan(pose_a).any(axis=1)
    visible_b = ~np.isnan(pose_b).any(axis=1)
    common = visible_a & visible_b

    if common.sum() < 2:
        return float("inf")

    pts_a = pose_a[common]
    pts_b = pose_b[common]

    if method == "euclidean":
        return float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1)))

    elif method == "procrustes":
        from scipy.spatial import procrustes

        try:
            _, _, disparity = procrustes(pts_a, pts_b)
            return float(disparity)
        except ValueError:
            return float("inf")

    else:
        raise ValueError(f"Unknown method: {method}")


class NearestNeighborScorer:
    """Score instances by distance to nearest neighbor in reference set.

    Uses KD-tree for efficient O(log n) nearest neighbor queries.

    Attributes:
        normalize: Whether to normalize poses before comparison.
        method: Distance method ("euclidean" or "procrustes").
        reference_poses: Stored reference poses after fitting.
    """

    def __init__(self, normalize: bool = True, method: str = "euclidean"):
        """Initialize scorer.

        Args:
            normalize: Whether to normalize poses before comparison.
            method: Distance method.
        """
        self.normalize = normalize
        self.method = method
        self.reference_poses: Optional[np.ndarray] = None
        self._kdtree = None
        self._flattened_refs: Optional[np.ndarray] = None

    def fit(self, poses: np.ndarray) -> "NearestNeighborScorer":
        """Store reference poses and build KD-tree for fast queries.

        Args:
            poses: (N_instances, N_nodes, 2) array of reference poses.

        Returns:
            Self for chaining.
        """
        from sklearn.neighbors import NearestNeighbors

        if self.normalize:
            self.reference_poses = np.array([normalize_pose(p) for p in poses])
        else:
            self.reference_poses = poses.copy()

        # Build KD-tree for fast queries (euclidean method only)
        if self.method == "euclidean":
            # Flatten poses and impute NaN with 0 (0 is near center after norm)
            self._flattened_refs = np.array(
                [np.nan_to_num(p.flatten(), nan=0.0) for p in self.reference_poses]
            )
            self._kdtree = NearestNeighbors(
                n_neighbors=1, algorithm="auto", metric="euclidean"
            )
            self._kdtree.fit(self._flattened_refs)

        return self

    def score(self, pose: np.ndarray) -> dict[str, float]:
        """Score a pose by distance to nearest neighbor.

        Uses KD-tree for fast O(log n) queries when available.

        Args:
            pose: (N_nodes, 2) array.

        Returns:
            Dictionary with:
            - nn_distance: distance to nearest neighbor
            - nn_index: index of nearest neighbor
            - mean_distance: mean distance to all references (only for non-KD-tree)
        """
        if self.reference_poses is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.normalize:
            query = normalize_pose(pose)
        else:
            query = pose

        # Fast path: use KD-tree for euclidean distance
        if self._kdtree is not None:
            query_flat = np.nan_to_num(query.flatten(), nan=0.0).reshape(1, -1)
            distances, indices = self._kdtree.kneighbors(query_flat)
            return {
                "nn_distance": float(distances[0, 0]),
                "nn_index": int(indices[0, 0]),
                "mean_distance": float(distances[0, 0]),  # Approximate
            }

        # Slow path: iterate over all references (for procrustes)
        distances = []
        for ref_pose in self.reference_poses:
            dist = pose_distance(query, ref_pose, method=self.method)
            distances.append(dist)

        distances = np.array(distances)
        valid_distances = distances[np.isfinite(distances)]

        if len(valid_distances) == 0:
            return {
                "nn_distance": float("inf"),
                "nn_index": -1,
                "mean_distance": float("inf"),
            }

        nn_idx = int(np.argmin(distances))
        return {
            "nn_distance": float(distances[nn_idx]),
            "nn_index": nn_idx,
            "mean_distance": float(np.mean(valid_distances)),
        }

    def score_batch(self, poses: np.ndarray) -> np.ndarray:
        """Score multiple poses efficiently using KD-tree.

        Args:
            poses: (N_instances, N_nodes, 2) array of poses to score.

        Returns:
            (N_instances,) array of nearest neighbor distances.
        """
        if self.reference_poses is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self._kdtree is None:
            # Fall back to individual scoring
            return np.array([self.score(p)["nn_distance"] for p in poses])

        # Normalize and flatten all poses
        if self.normalize:
            normalized = np.array([normalize_pose(p) for p in poses])
        else:
            normalized = poses

        flattened = np.array([np.nan_to_num(p.flatten(), nan=0.0) for p in normalized])

        # Batch KD-tree query
        distances, _ = self._kdtree.kneighbors(flattened)
        return distances[:, 0]

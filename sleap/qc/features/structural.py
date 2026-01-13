"""Structural features: curvature, convex hull."""

from __future__ import annotations

import numpy as np


def compute_curvature(
    points: np.ndarray,
    chain: list[int],
) -> dict[str, float]:
    """Compute curvature along a chain of nodes (e.g., spine).

    Curvature at each interior node is computed from the angle formed
    by adjacent edges. High curvature = sharp bend.

    Args:
        points: (N, 2) array of node coordinates.
        chain: Ordered list of node indices forming a chain.

    Returns:
        Dictionary with:
        - curvatures: array of curvature values at each interior node
        - max_curvature: maximum absolute curvature
        - mean_curvature: mean absolute curvature
        - curvature_std: standard deviation of curvature
        - sign_changes: number of curvature sign changes (wiggliness)
    """
    if len(chain) < 3:
        return {
            "curvatures": np.array([]),
            "max_curvature": 0.0,
            "mean_curvature": 0.0,
            "curvature_std": 0.0,
            "sign_changes": 0,
        }

    curvatures = []
    for i in range(1, len(chain) - 1):
        prev_idx, curr_idx, next_idx = chain[i - 1], chain[i], chain[i + 1]

        # Skip if any node is invisible
        if (
            np.isnan(points[prev_idx]).any()
            or np.isnan(points[curr_idx]).any()
            or np.isnan(points[next_idx]).any()
        ):
            curvatures.append(np.nan)
            continue

        # Vectors from current to neighbors
        v1 = points[prev_idx] - points[curr_idx]
        v2 = points[next_idx] - points[curr_idx]

        # Angle between vectors (curvature proxy)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            curvatures.append(np.nan)
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        # Curvature = pi - angle (0 = straight, pi = folded back)
        curvature = np.pi - angle

        # Signed curvature (cross product sign)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        curvature = curvature * np.sign(cross) if cross != 0 else curvature

        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    valid_curvatures = curvatures[~np.isnan(curvatures)]

    # Count sign changes
    sign_changes = 0
    if len(valid_curvatures) > 1:
        signs = np.sign(valid_curvatures)
        sign_changes = int(np.sum(signs[1:] != signs[:-1]))

    return {
        "curvatures": curvatures,
        "max_curvature": (
            float(np.max(np.abs(valid_curvatures)))
            if len(valid_curvatures) > 0
            else 0.0
        ),
        "mean_curvature": (
            float(np.mean(np.abs(valid_curvatures)))
            if len(valid_curvatures) > 0
            else 0.0
        ),
        "curvature_std": (
            float(np.std(valid_curvatures)) if len(valid_curvatures) > 0 else 0.0
        ),
        "sign_changes": sign_changes,
    }


def compute_convex_hull(
    points: np.ndarray,
) -> dict[str, float]:
    """Compute convex hull metrics for pose compactness.

    Args:
        points: (N, 2) array of node coordinates (NaN for invisible).

    Returns:
        Dictionary with:
        - hull_area: area of convex hull
        - hull_perimeter: perimeter of convex hull
        - hull_aspect_ratio: width/height of hull bounding box
        - compactness: 4*pi*area / perimeter^2 (1 = circle)
        - n_hull_points: number of points on hull
    """
    from scipy.spatial import ConvexHull

    # Filter to visible points
    visible_mask = ~np.isnan(points).any(axis=1)
    visible_points = points[visible_mask]

    if len(visible_points) < 3:
        return {
            "hull_area": 0.0,
            "hull_perimeter": 0.0,
            "hull_aspect_ratio": 1.0,
            "compactness": 0.0,
            "n_hull_points": len(visible_points),
        }

    try:
        hull = ConvexHull(visible_points)
        area = hull.volume  # In 2D, volume = area
        perimeter = hull.area  # In 2D, area = perimeter

        # Aspect ratio from bounding box
        hull_points = visible_points[hull.vertices]
        min_pt = hull_points.min(axis=0)
        max_pt = hull_points.max(axis=0)
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        aspect_ratio = width / height if height > 0 else 1.0

        # Compactness (isoperimetric quotient)
        compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0

        return {
            "hull_area": float(area),
            "hull_perimeter": float(perimeter),
            "hull_aspect_ratio": float(aspect_ratio),
            "compactness": float(compactness),
            "n_hull_points": len(hull.vertices),
        }

    except Exception:
        # Hull computation can fail for degenerate cases
        return {
            "hull_area": 0.0,
            "hull_perimeter": 0.0,
            "hull_aspect_ratio": 1.0,
            "compactness": 0.0,
            "n_hull_points": 0,
        }

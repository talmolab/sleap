"""Frame-level quality checks: instance count, duplicate detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


class InstanceCountChecker:
    """Detect frames with unusual instance counts (incomplete annotation).

    Attributes:
        per_video: Whether to compute expected counts per video.
        expected_counts: Video-specific expected counts.
        global_expected: Global expected count.
    """

    def __init__(self, per_video: bool = True):
        """Initialize checker.

        Args:
            per_video: If True, compute expected counts per video.
        """
        self.per_video = per_video
        self.expected_counts: dict[str, float] = {}
        self.global_expected: float = 0.0

    def fit(
        self,
        frame_counts: list[int],
        video_ids: Optional[list[str]] = None,
    ) -> "InstanceCountChecker":
        """Learn expected instance counts.

        Args:
            frame_counts: List of instance counts per frame.
            video_ids: Optional list of video IDs per frame.

        Returns:
            Self for chaining.
        """
        frame_counts_arr = np.array(frame_counts)
        self.global_expected = float(np.median(frame_counts_arr))

        if self.per_video and video_ids is not None:
            video_counts: dict[str, list[int]] = defaultdict(list)
            for count, vid in zip(frame_counts, video_ids):
                video_counts[vid].append(count)

            for vid, counts in video_counts.items():
                self.expected_counts[vid] = float(np.median(counts))

        return self

    def check(
        self,
        instance_count: int,
        video_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Check if a frame's instance count is unusual.

        Args:
            instance_count: Number of instances in frame.
            video_id: Optional video ID for per-video comparison.

        Returns:
            Dictionary with:
            - is_incomplete: True if fewer instances than expected
            - expected_count: expected count for this video
            - actual_count: actual count
            - count_difference: actual - expected
        """
        if self.per_video and video_id and video_id in self.expected_counts:
            expected = self.expected_counts[video_id]
        else:
            expected = self.global_expected

        difference = instance_count - expected
        is_incomplete = instance_count < expected

        return {
            "is_incomplete": is_incomplete,
            "expected_count": expected,
            "actual_count": instance_count,
            "count_difference": difference,
        }


def compute_instance_iou(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """Compute IOU between two instances based on bounding boxes.

    Args:
        points_a: (N_nodes, 2) array for instance A (NaN for invisible).
        points_b: (N_nodes, 2) array for instance B.

    Returns:
        IOU value (0-1).
    """
    # Get visible points
    visible_a = points_a[~np.isnan(points_a).any(axis=1)]
    visible_b = points_b[~np.isnan(points_b).any(axis=1)]

    if len(visible_a) < 2 or len(visible_b) < 2:
        return 0.0

    try:
        # Bounding boxes
        min_a = visible_a.min(axis=0)
        max_a = visible_a.max(axis=0)
        min_b = visible_b.min(axis=0)
        max_b = visible_b.max(axis=0)

        # Intersection
        inter_min = np.maximum(min_a, min_b)
        inter_max = np.minimum(max_a, max_b)
        inter_dims = np.maximum(0, inter_max - inter_min)
        intersection = inter_dims[0] * inter_dims[1]

        # Union
        area_a = (max_a[0] - min_a[0]) * (max_a[1] - min_a[1])
        area_b = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])
        union = area_a + area_b - intersection

        return float(intersection / union) if union > 0 else 0.0

    except Exception:
        return 0.0


def compute_node_overlap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    distance_threshold: float = 10.0,
) -> dict[str, object]:
    """Compute node-wise overlap for partial duplicate detection.

    Args:
        points_a: (N_nodes, 2) array for instance A.
        points_b: (N_nodes, 2) array for instance B.
        distance_threshold: Max distance to consider nodes as overlapping.

    Returns:
        Dictionary with:
        - common_nodes: list of node indices visible in both
        - overlapping_nodes: list of nodes within threshold
        - mean_distance: mean distance at common nodes
        - overlap_ratio: overlapping_nodes / common_nodes
    """
    # Find commonly visible nodes
    visible_a = ~np.isnan(points_a).any(axis=1)
    visible_b = ~np.isnan(points_b).any(axis=1)
    common_mask = visible_a & visible_b
    common_nodes = np.where(common_mask)[0].tolist()

    if len(common_nodes) == 0:
        return {
            "common_nodes": [],
            "overlapping_nodes": [],
            "mean_distance": float("inf"),
            "overlap_ratio": 0.0,
        }

    # Compute distances at common nodes
    distances = []
    overlapping = []
    for node in common_nodes:
        dist = np.linalg.norm(points_a[node] - points_b[node])
        distances.append(dist)
        if dist < distance_threshold:
            overlapping.append(node)

    return {
        "common_nodes": common_nodes,
        "overlapping_nodes": overlapping,
        "mean_distance": float(np.mean(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "overlap_ratio": len(overlapping) / len(common_nodes),
    }


def detect_duplicates(
    instances: list[np.ndarray],
    iou_threshold: float = 0.5,
    node_distance_threshold: float = 10.0,
    node_overlap_ratio: float = 0.8,
) -> list[dict]:
    """Detect duplicate instances in a frame.

    Uses both IOU and node-wise overlap to catch partial duplicates.

    Args:
        instances: List of (N_nodes, 2) arrays.
        iou_threshold: IOU above this = duplicate.
        node_distance_threshold: Distance for node overlap.
        node_overlap_ratio: Min overlap ratio to flag as duplicate.

    Returns:
        List of duplicate pair dictionaries with:
        - index_a, index_b: instance indices
        - iou: IOU value
        - node_overlap: node overlap info
        - reason: "iou" or "node_overlap"
    """
    duplicates = []
    n_instances = len(instances)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            iou = compute_instance_iou(instances[i], instances[j])
            node_overlap = compute_node_overlap(
                instances[i], instances[j], node_distance_threshold
            )

            is_duplicate = False
            reason = None

            # Check IOU
            if iou > iou_threshold:
                is_duplicate = True
                reason = "iou"

            # Check node overlap (catches partial duplicates)
            elif (
                len(node_overlap["common_nodes"]) >= 2
                and node_overlap["overlap_ratio"] > node_overlap_ratio
            ):
                is_duplicate = True
                reason = "node_overlap"

            if is_duplicate:
                duplicates.append(
                    {
                        "index_a": i,
                        "index_b": j,
                        "iou": iou,
                        "node_overlap": node_overlap,
                        "reason": reason,
                    }
                )

    return duplicates

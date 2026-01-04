"""
Instance size computation and data structures for bounding boxes.

This module provides utilities for analyzing the distribution of instance
sizes across a dataset, including rotation augmentation effects. Useful for:
- Determining optimal crop sizes for centered-instance models
- Identifying outlier instances that may be annotation errors
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sleap_io as sio


@dataclass
class InstanceSizeInfo:
    """Data class holding size info for a single instance.

    Attributes:
        video_idx: Index of the video in labels.videos.
        frame_idx: Frame index within the video.
        instance_idx: Index of the instance within the labeled frame.
        raw_width: Bounding box width before rotation.
        raw_height: Bounding box height before rotation.
        raw_size: Maximum of width and height (square bounding box size).
    """

    video_idx: int
    frame_idx: int
    instance_idx: int
    raw_width: float
    raw_height: float
    raw_size: float

    def get_rotated_size(self, max_angle_degrees: float) -> float:
        """Calculate size needed for rotation augmentation range +/-max_angle.

        When rotation augmentation samples uniformly from [-max_angle, +max_angle],
        we need a size large enough to contain the instance at ANY angle in that range.

        For a rectangle of size (w, h), rotated by angle theta:
            new_width = w*|cos(theta)| + h*|sin(theta)|
            new_height = w*|sin(theta)| + h*|cos(theta)|

        The worst case (largest size needed) depends on the aspect ratio:
        - For most shapes (aspect ratio > 0.41), worst case is at 45 degrees
        - For very elongated shapes, worst case may be at 0 or 90 degrees

        For +/-180 (full rotation), we check 45 as the typical worst case.
        For +/-theta where theta < 45, we check theta as the boundary.

        Args:
            max_angle_degrees: Maximum rotation angle in degrees. The augmentation
                range is assumed to be [-max_angle, +max_angle].

        Returns:
            The size needed to contain the instance at any rotation
            within the specified range.
        """
        if max_angle_degrees == 0:
            return self.raw_size

        max_angle = min(abs(max_angle_degrees), 90)  # Symmetry beyond 90

        # Angles to check for worst case:
        # - 0 (original orientation)
        # - max_angle (boundary of range)
        # - 45 (worst case for most shapes, if within range)
        angles_to_check = {0, max_angle}
        if max_angle >= 45:
            angles_to_check.add(45)

        max_size = 0.0
        for angle in angles_to_check:
            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Calculate bounding box after rotation
            new_width = self.raw_width * cos_t + self.raw_height * sin_t
            new_height = self.raw_width * sin_t + self.raw_height * cos_t

            size = max(new_width, new_height)
            max_size = max(max_size, size)

        return max_size


def compute_instance_sizes(
    labels: "sio.Labels",
    user_instances_only: bool = True,
) -> List[InstanceSizeInfo]:
    """Compute sizes for all instances in labels.

    Iterates through all labeled frames and computes the bounding box
    dimensions for each instance. By default, only user-labeled instances
    are included (not predicted instances).

    Args:
        labels: A sleap_io.Labels object containing labeled frames.
        user_instances_only: If True (default), only include user-labeled
            instances, not predicted instances.

    Returns:
        List of InstanceSizeInfo for each non-empty instance.
    """
    results: List[InstanceSizeInfo] = []

    for lf in labels:
        video_idx = labels.videos.index(lf.video) if lf.video in labels.videos else 0

        # Choose which instances to iterate based on user_instances_only flag
        if user_instances_only:
            instances = lf.user_instances if hasattr(lf, "user_instances") else []
        else:
            instances = lf.instances

        for inst_idx, inst in enumerate(instances):
            if inst.is_empty:
                continue

            pts = inst.numpy()

            # Calculate bounding box dimensions
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            # Filter out NaN values
            valid_x = x_coords[~np.isnan(x_coords)]
            valid_y = y_coords[~np.isnan(y_coords)]

            if len(valid_x) < 2 or len(valid_y) < 2:
                continue

            width = float(np.max(valid_x) - np.min(valid_x))
            height = float(np.max(valid_y) - np.min(valid_y))

            results.append(
                InstanceSizeInfo(
                    video_idx=video_idx,
                    frame_idx=lf.frame_idx,
                    instance_idx=inst_idx,
                    raw_width=width,
                    raw_height=height,
                    raw_size=max(width, height),
                )
            )

    return results

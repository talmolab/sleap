"""Helper functions for `sleap_io.Instance` objects."""

from typing import Tuple, Optional, List
import numpy as np

from sleap_io.model.skeleton import Node
from sleap_io.model.instance import (
    Instance,
    PredictedInstance,
    PointsArray,
)


def node_points(instance) -> List[Tuple[Node, np.ndarray]]:
    """
    Return a list of (node, point) tuples for all labeled points.

    Args:
        instance: sleap_io Instance object

    Returns:
        List of (Node, point_data) tuples where point_data is [x, y, visible, complete]
    """
    # Get all nodes from the skeleton
    skeleton_nodes = list(instance.skeleton.nodes)

    # Get all points data
    points_data = instance.points

    # Create mapping of node names to points
    node_points = []

    # Create mapping
    for node_idx in range(len(points_data)):
        node = skeleton_nodes[node_idx]
        # Find the point data for this node
        point_data = points_data[node_idx]

        # Convert to [x, y, visible, complete] format

        node_points.append((node, point_data))

    return node_points


def get_nodes_from_instance(instance: Instance) -> Tuple[Node, ...]:
    """Return nodes that have been labelled (non-nan) for this instance."""
    node_names = instance.points["name"]

    labeled_nodes = []
    for i, (node_name, point_data) in enumerate(zip(node_names, instance.points)):
        # Check if the point has valid coordinates (not NaN)
        if not np.isnan(point_data["xy"][0]) and not np.isnan(point_data["xy"][1]):
            # Check if the node exists in the skeleton
            if node_name in instance.skeleton.node_names:
                labeled_nodes.append(node_name)

    return tuple(labeled_nodes)


def bounding_box(instance: Instance):
    """Return bounding box containing all points in `[y1, x1, y2, x2]` format."""
    points = instance.points["xy"]
    if np.isnan(points).all():
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    bbox = np.concatenate(
        [[np.nanmin(points, axis=0)[::-1]], [np.nanmax(points, axis=0)[::-1]]]
    )
    return bbox


def fill_missing(
    instance: Instance, max_x: Optional[float] = None, max_y: Optional[float] = None
):
    """Add points for skeleton nodes that are missing in the instance.

    This is useful when modifying the skeleton so the nodes appear in the GUI.

    Args:
        instance: sleap_io Instance object
        max_x: If specified, make sure points are not added outside of valid range.
        max_y: If specified, make sure points are not added outside of valid range.

    Returns:
        Modified instance with missing points filled
    """
    # Get current bounding box
    bbox = bounding_box(instance)  # [[min_x, min_y], [max_x, max_y]]
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    y1, x1 = np.nanmax([y1, 0]), np.nanmax([x1, 0])

    if max_x is not None:
        x2 = np.nanmin([x2, max_x])
    if max_y is not None:
        y2 = np.nanmin([y2, max_y])

    # Build a new full points array aligned to the skeleton order.
    skeleton_nodes = list(instance.skeleton.nodes)
    current_names = list(instance.points["name"]) if len(instance.points) else []
    w, h = y2 - y1, x2 - x1

    def _rand_point():
        off = np.array([w, h]) * np.random.rand(2)
        x, y = off + np.array([x1, y1])
        y, x = max(y, 0), max(x, 0)
        if max_x is not None:
            x = min(x, max_x)
        if max_y is not None:
            y = min(y, max_y)
        return x, y

    # Prepare a raw input array for all nodes, then convert to PointsArray
    input_array = np.empty(
        len(skeleton_nodes),
        dtype=[
            ("xy", "<f8", (2,)),
            ("visible", "bool"),
            ("complete", "bool"),
            ("name", "O"),
        ],
    )

    for i, node in enumerate(skeleton_nodes):
        name = node.name
        if name in current_names:
            # Copy existing if present and not NaN; otherwise generate new
            existing_idx = current_names.index(name)
            pt = instance.points[existing_idx]
            xy = pt["xy"]
            if not (np.isnan(xy[0]) or np.isnan(xy[1])):
                input_array[i] = (
                    np.array(xy, dtype=np.float64),
                    pt["visible"],
                    pt["complete"],
                    name,
                )
            else:
                input_array[i] = (_rand_point(), False, False, name)
        else:
            input_array[i] = (_rand_point(), False, False, name)

    # Replace points with an array sized to the skeleton
    instance.points = PointsArray.from_array(input_array)

    # Create new instance with filled points
    if hasattr(instance, "score"):  # PredictedInstance
        new_instance = PredictedInstance(
            points=instance.points,
            skeleton=instance.skeleton,
            track=instance.track,
            score=instance.score,
            tracking_score=instance.tracking_score,
            from_predicted=instance.from_predicted,
        )
    else:  # Instance
        new_instance = Instance(
            points=instance.points,
            skeleton=instance.skeleton,
            track=instance.track,
            tracking_score=instance.tracking_score,
            from_predicted=instance.from_predicted,
        )

    return new_instance


# Instance/PredictedInstance API Compatibility Functions
# These functions provide backward compatibility with legacy SLEAP Instance API


def instance_get_points_array(instance):
    """Get points array for backward compatibility.

    This provides backward compatibility for the missing points_array attribute.
    Maps instance.points_array to instance.numpy() or instance.points["xy"].
    """
    # Try using numpy() method first (preferred)
    if hasattr(instance, "numpy") and callable(instance.numpy):
        return instance.numpy()
    else:
        # Fallback to extracting xy coordinates from points
        return instance.points["xy"]


def instance_get_scores(instance):
    """Get point-wise scores for backward compatibility.

    This provides backward compatibility for the missing scores attribute.
    Maps instance.scores to point-wise scores from instance.points["score"]
    if available,
    or falls back to instance.score for PredictedInstance.
    """
    if hasattr(instance, "points") and "score" in instance.points.dtype.names:
        # Return point-wise scores
        return instance.points["score"]
    elif hasattr(instance, "score"):
        # Fallback to instance-level score for PredictedInstance
        return instance.score
    else:
        # No scores available
        return None


def predicted_instance_from_numpy_compat(
    points, skeleton, point_confidences=None, instance_score=None, track=None, **kwargs
):
    """Create PredictedInstance from numpy array with backward compatibility.

    This provides backward compatibility for PredictedInstance.from_numpy() method
    signature changes. Maps old parameter names to new ones.
    """
    # Convert old parameter names to new sleap-io API
    from sleap_io.model.instance import PredictedInstance

    # Handle parameter mapping
    if point_confidences is not None:
        # In sleap-io, point scores are stored in the points structure
        # We need to create a proper points dict/array
        points_dict = {}
        for i, node in enumerate(skeleton.nodes):
            x, y = points[i] if len(points) > i else (np.nan, np.nan)
            score = (
                point_confidences[i]
                if point_confidences is not None and len(point_confidences) > i
                else 1.0
            )
            points_dict[node.name] = (
                x,
                y,
                score,
                True,
                True,
            )  # x, y, score, visible, complete

        return PredictedInstance(
            points=points_dict,
            skeleton=skeleton,
            track=track,
            score=instance_score if instance_score is not None else 1.0,
            **kwargs,
        )
    else:
        # Simple case - just use points directly
        points_dict = {}
        for i, node in enumerate(skeleton.nodes):
            if i < len(points):
                x, y = points[i]
                points_dict[node.name] = (
                    x,
                    y,
                    1.0,
                    True,
                    True,
                )  # x, y, score, visible, complete

        return PredictedInstance(
            points=points_dict,
            skeleton=skeleton,
            track=track,
            score=instance_score if instance_score is not None else 1.0,
            **kwargs,
        )


def instance_same_pose_as_compat(instance1, instance2):
    """Compare poses between instances for backward compatibility.

    This provides a safe way to compare instance poses that handles
    array comparison ambiguity issues.
    """
    try:
        # Try the native same_pose_as method first
        if hasattr(instance1, "same_pose_as"):
            return instance1.same_pose_as(instance2)
        else:
            # Fallback to numpy array comparison
            points1 = instance_get_points_array(instance1)
            points2 = instance_get_points_array(instance2)
            return np.array_equal(points1, points2, equal_nan=True)
    except ValueError:
        # Handle "truth value is ambiguous" errors
        points1 = instance_get_points_array(instance1)
        points2 = instance_get_points_array(instance2)
        return np.array_equal(points1, points2, equal_nan=True)


def get_centroid(instance: Instance) -> np.ndarray:
    """Return instance centroid as an array of `(x, y)` coordinates.

    Notes:
        This computes the centroid as the median of the visible points.
    """
    points = instance.points["xy"]
    centroid = np.nanmedian(points, axis=0)
    return centroid

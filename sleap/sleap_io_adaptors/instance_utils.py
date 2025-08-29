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

    w, h = y2 - y1, x2 - x1

    # Build a new full points array aligned to the skeleton order.
    skeleton_nodes = list(instance.skeleton.nodes)
    current_names = list(instance.points["name"]) if len(instance.points) else []

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

    # Helper to generate a bounded random point inside bbox
    def _rand_point():
        off = np.array([w, h]) * np.random.rand(2)
        x, y = off + np.array([x1, y1])
        y, x = max(y, 0), max(x, 0)
        if max_x is not None:
            x = min(x, max_x)
        if max_y is not None:
            y = min(y, max_y)
        return np.array([x, y], dtype=np.float64)

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


def _is_node_nan(instance, node_name: str) -> bool:
    """Check if a node has NaN coordinates."""
    try:
        node_idx = list(instance.points["name"]).index(node_name)
        point_data = instance.points[node_idx]
        return np.isnan(point_data["xy"][0]) or np.isnan(point_data["xy"][1])
    except ValueError:
        return True  # Node not found

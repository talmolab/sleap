"""Helper functions for `sleap_io.Skeleton` objects."""
from typing import TYPE_CHECKING, Tuple, Optional, List
import numpy as np

if TYPE_CHECKING:
    from sleap.instance import Instance
from sleap_io.model.skeleton import Node
from sleap_io.model.instance import Instance, PredictedInstance

def nodes_points(instance) -> List[Tuple[Node, np.ndarray]]:
    """
    Return a list of (node, point) tuples for all labeled points.

    Args:
        instance: sleap_io Instance object

    Returns:
        List of (Node, point_data) tuples where point_data is [x, y, visible, complete]
    """

    # Get all nodes from the skeleton
    skeleton_nodes = list(instance.skeleton.nodes.values())

    # Get all points data
    points_data = instance.points

    # Create mapping of node names to points
    node_points = []

    # Get all node names from points
    point_node_names = points_data["name"]

    # Get valid points (not NaN)
    valid_mask = ~(np.isnan(points_data["xy"][:, 0]) | np.isnan(points_data["xy"][:, 1]))
    valid_points = points_data[valid_mask]
    valid_node_names = point_node_names[valid_mask]

    # Create mapping
    for node in skeleton_nodes:
        if node.name in valid_node_names:
            # Find the point data for this node
            node_idx = list(valid_node_names).index(node.name)
            point_data = valid_points[node_idx]

            # Convert to [x, y, visible, complete] format
            point_array = np.array([
                point_data["xy"][0],  # x
                point_data["xy"][1],  # y
                point_data["visible"],
                point_data["complete"]
            ])
            node_points.append((node, point_array))

    return node_points

def get_nodes_from_instance(instance: Instance) -> Tuple[Node, ...]:
    """Return nodes that have been labelled (non-nan) for this instance."""
    node_names = instance.points["name"]

    labeled_nodes = []
    for i, (node_name, point_data) in enumerate(zip(node_names, instance.points)):
        # Check if the point has valid coordinates (not NaN)
        if not np.isnan(point_data["xy"][0]) and not np.isnan(point_data["xy"][1]):
            # Check if the node exists in the skeleton
            if node_name in instance.skeleton.nodes:
                labeled_nodes.append(instance.skeleton.nodes[node_name])

    return tuple(labeled_nodes)


def fill_missing(instance: Instance, max_x: Optional[float] = None, max_y: Optional[float] = None):
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
    bbox = instance.bounding_box() # [[min_x, min_y], [max_x, max_y]]
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    y1, x1 = np.nanmax([y1, 0]), np.nanmax([x1, 0])

    if max_x is not None:
        x2 = np.nanmin([x2, max_x])
    if max_y is not None:
        y2 = np.nanmin([y2, max_y])

    w, h = y2 - y1, x2 - x1

    # Get current node names from points
    current_node_names = set(instance.points["name"])

    # Find missing nodes
    missing_nodes = []
    for node in instance.skeleton.nodes:
        if node.name not in current_node_names or _is_node_nan(instance, node.name):
            missing_nodes.append(node)

    if not missing_nodes:
        return instance

    # Create new points array with missing nodes
    new_points = np.empty(len(missing_nodes), dtype=instance.points.dtype)

    for i, node in enumerate(missing_nodes):
        # Generate random position within bounding box
        off = np.array([w, h]) * np.random.rand(2)
        x, y = off + np.array([x1, y1])

        # Clamp to bounds
        y, x = max(y, 0), max(x, 0)
        if max_x is not None:
            x = min(x, max_x)
        if max_y is not None:
            y = min(y, max_y)

        # Set point data
        new_points[i]["xy"] = [x, y]
        new_points[i]["visible"] = False
        new_points[i]["complete"] = False
        new_points[i]["name"] = node.name

    # Combine existing and new points
    combined_points = np.append(instance.points, new_points)

    # Create new instance with filled points
    if hasattr(instance, 'score'):  # PredictedInstance
        new_instance = PredictedInstance(
            points=combined_points,
            skeleton=instance.skeleton,
            track=instance.track,
            score=instance.score,
            tracking_score=instance.tracking_score,
            from_predicted=instance.from_predicted
        )
    else:  # Instance
        new_instance = Instance(
            points=combined_points,
            skeleton=instance.skeleton,
            track=instance.track,
            tracking_score=instance.tracking_score,
            from_predicted=instance.from_predicted
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

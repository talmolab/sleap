"""Helper functions for `sleap_io.Instance` objects."""

from typing import TYPE_CHECKING, Tuple, Optional, List, Union, Dict
import numpy as np

from sleap_io.model.skeleton import Node, Skeleton
from sleap_io.model.instance import (
    Instance,
    PredictedInstance,
    PointsArray,
    PredictedPointsArray,
    Track,
)
import attr
import cattr


def make_instance_cattr() -> cattr.Converter:
    """Create a cattr converter for Lists of Instances/PredictedInstances.

    This is required because cattrs doesn't automatically detect the class when the
    attributes of one class are a subset of another.

    Returns:
        A cattr converter with hooks registered for structuring and unstructuring
        `Instance` and `PredictedInstance` objects.
    """
    converter = cattr.Converter()

    #### UNSTRUCTURE HOOKS

    # JSON dump cant handle NumPy bools so convert them. These are present
    # in Point/PredictedPoint objects now since they are actually custom numpy dtypes.
    converter.register_unstructure_hook(np.bool_, bool)

    converter.register_unstructure_hook(PointsArray, lambda x: None)
    converter.register_unstructure_hook(PredictedPointsArray, lambda x: None)

    def unstructure_instance(x: Instance):
        # Unstructure everything but the points array, nodes, and frame attribute
        d = {
            field.name: converter.unstructure(x.__getattribute__(field.name))
            for field in attr.fields(x.__class__)
            if field.name not in ["_points", "_nodes", "frame"]
        }

        # Replace the point array with a dict
        d["_points"] = converter.unstructure({k: v for k, v in x.nodes_points})

        return d

    converter.register_unstructure_hook(Instance, unstructure_instance)
    converter.register_unstructure_hook(PredictedInstance, unstructure_instance)

    # Function to determine object type for objects being structured.
    def structure_instances_list(x, type):
        inst_list = []
        for inst_data in x:
            inst = structure_instance(inst_data, type)
            inst_list.append(inst)

        return inst_list

    def structure_instance(inst_data, type):
        """Structure hook for Instance and PredictedInstance objects."""
        from_predicted = None

        if "score" in inst_data.keys():
            inst = converter.structure(inst_data, PredictedInstance)
        else:
            if (
                "from_predicted" in inst_data
                and inst_data["from_predicted"] is not None
            ):
                from_predicted = converter.structure(
                    inst_data["from_predicted"], PredictedInstance
                )
                # Remove the from_predicted key. We'll add it back afterwards.
                inst_data["from_predicted"] = None

            # Structure the instance data, then add the from_predicted attribute.
            inst = converter.structure(inst_data, Instance)
            inst.from_predicted = from_predicted
        return inst

    converter.register_structure_hook(
        Union[List[Instance], List[PredictedInstance]], structure_instances_list
    )

    return converter


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
    for node_idx, node in enumerate(skeleton_nodes):
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
    if hasattr(instance, "score"):  # PredictedInstance
        new_instance = PredictedInstance(
            points=combined_points,
            skeleton=instance.skeleton,
            track=instance.track,
            score=instance.score,
            tracking_score=instance.tracking_score,
            from_predicted=instance.from_predicted,
        )
    else:  # Instance
        new_instance = Instance(
            points=combined_points,
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

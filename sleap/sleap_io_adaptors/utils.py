"""Helper functions for `sleap_io.Skeleton` objects."""
from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from sleap.instance import Instance
from PIL import Image
from sleap.util import plot_img, plot_instance
from sleap_io.model.skeleton import Skeleton, Node
from sleap_io.model.instance import Instance
from io import BytesIO
import base64
import matplotlib.pyplot as plt

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

def find_node(skeleton: Skeleton, node_name: str) -> Node:
    """Find node in skeleton by name of node."""
    for node in skeleton.nodes:
        if node.name == node_name:
            return node
    return None


def get_symmetry_node(skeleton: Skeleton, node_name: str) -> str:
    """Get symmetry node name for given node name."""
    for symmetry in skeleton.symmetries:
        if node_name in [n.name for n in symmetry.nodes]:
            # Return the other node in the symmetry pair
            return next((n.name for n in symmetry.nodes if n.name != node_name), None)
    return None

"""Helper functions for `sleap_io.Skeleton` objects."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sleap.instance import Instance
from PIL import Image
from sleap.util import plot_img, plot_instance
from sleap_io.model.skeleton import Skeleton, Node
from io import BytesIO
import base64
import matplotlib.pyplot as plt

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

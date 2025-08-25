"""Helper functions for `sleap_io.Skeleton` objects."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from sleap_io.model.skeleton import Skeleton, Node, NodeOrIndex

def find_node(skeleton: Skeleton, node_name: str) -> Node:
    """Find node in skeleton by name of node."""
    for node in skeleton.nodes:
        if node.name == node_name:
            return node
    return None

def node_to_index(skeleton: Skeleton, node: NodeOrIndex) -> int:
        """Return the index of the node, accepts either `Node` or name.
        Args:
            node: The name of the node or the Node object.
        Raises:
            ValueError if node cannot be found in skeleton.
        Returns:
            The index of the node in the graph.
        """
        node_list = skeleton.node_names
        if isinstance(node, Node):
            node = node.name
        if node in node_list:
            return node_list.index(node)
        else:
            raise ValueError(f"Node {node} not found in skeleton.")

def edge_to_index(skeleton: Skeleton,
                    source: NodeOrIndex,
                    destination: NodeOrIndex) -> int:
        """Return the index of edge from source to destination."""
        source = source.name if isinstance(source, Node) else source
        destination = destination.name if isinstance(destination, Node) else destination
        for edge_idx, edge in enumerate(skeleton.edges):
            if edge.source.name == source and edge.destination.name == destination:
                return edge_idx
        return -1


def get_symmetry_node(skeleton: Skeleton, node_name: str) -> str:
    """Get symmetry node name for given node name."""
    for symmetry in skeleton.symmetries:
        if node_name in [n.name for n in symmetry.nodes]:
            # Return the other node in the symmetry pair
            return next((n.name for n in symmetry.nodes if n.name != node_name), None)
    return None

def delete_symmetry(skeleton: Skeleton, node1: str, node2: str):
    """Delete symmetry for given node name."""
    for symmetry_idx, symmetry in enumerate(skeleton.symmetries):
        node_names = [n.name for n in symmetry.nodes]
        if node1 in node_names and node2 in node_names:
            skeleton.symmetries.pop(symmetry_idx)
    return skeleton

def delete_edge(skeleton: Skeleton, source: str, destination: str):
        """Delete an edge between two nodes.
        Args:
            skeleton: The skeleton to delete the edge from.
            source: The name of the source node.
            destination: The name of the destination node.
        Raises:
            ValueError: If skeleton does not have either source node,
                destination node, or edge between them.
        Returns:
            Skeleton: The skeleton with the edge deleted.
        """
        if isinstance(source, Node):
            source = source.name

        if isinstance(destination, Node):
            destination = destination.name

        if source is None or source not in skeleton.node_names:
            raise ValueError(
                "Skeleton does not have source node named ({})".format(source)
            )

        if destination is None or destination not in skeleton.node_names:
            raise ValueError(
                "Skeleton does not have destination node named ({})".format(destination)
            )

        for edge_idx, edge in enumerate(skeleton.edges):
            if edge.source.name == source and edge.destination.name == destination:
                skeleton.edges.pop(edge_idx)
                return skeleton

        raise ValueError(
            "Skeleton has no edge between ({}) and ({}).".format(source, destination)
        )

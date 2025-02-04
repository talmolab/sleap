"""
Implementation of skeleton data structure and API.

This module implements and API for creating animal skeletons. The goal
is to provide a common interface for defining the parts of the animal,
their connection to each other, and needed meta-data.
"""

import base64
import copy
import json
import operator
from enum import Enum
from io import BytesIO
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple, Union

import attr
import cattr
import h5py
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from PIL import Image
from scipy.io import loadmat

NodeRef = Union[str, "Node"]
H5FileRef = Union[str, h5py.File]


class EdgeType(Enum):
    """Type of edge in the skeleton graph.

    The skeleton graph can store different types of edges to represent
    different things. All edges must specify one or more of the
    following types:

        * BODY - these edges represent connections between parts or
          landmarks.
        * SYMMETRY - these edges represent symmetrical relationships
          between parts (e.g. left and right arms)
    """

    BODY = 1
    SYMMETRY = 2


@attr.s(auto_attribs=True, slots=True, eq=False, order=False)
class Node:
    """This class represents node in the skeleton graph, i.e., a body part.

    Note: Nodes can exist without being part of a skeleton.

    Attributes:
        name: String name of the node.
        weight: Weight of the node (not currently used).
    """

    name: str
    weight: float = 1.0

    @staticmethod
    def from_names(name_list: str) -> List["Node"]:
        """Convert list of node names to list of nodes objects."""
        nodes = []
        for name in name_list:
            nodes.append(Node(name))
        return nodes

    @classmethod
    def as_node(cls, node: NodeRef) -> "Node":
        """Convert given `node` to `Node` object (if not already)."""
        return node if isinstance(node, cls) else cls(node)

    def matches(self, other: "Node") -> bool:
        """
        Check whether all attributes match between two nodes.

        Args:
            other: The `Node` to compare to this one.

        Returns:
            True if all attributes match, False otherwise.
        """
        return other.name == self.name and other.weight == self.weight


class SkeletonDecoder:
    """Replace jsonpickle.decode with our own decoder.

    This function will decode the following from jsonpickle's encoded format:

    `Node` objects from
        {
            "py/object": "sleap.skeleton.Node",
            "py/state": { "py/tuple": ["thorax1", 1.0] }
        }
    to `Node(name="thorax1", weight=1.0)`

    `EdgeType` objects from
        {
            "py/reduce": [
                { "py/type": "sleap.skeleton.EdgeType" },
                { "py/tuple": [1] }
            ]
        }
    to `EdgeType(1)`

    `bytes` from
        {
            "py/b64": "aVZC..."
        }
    to `b"iVBO..."`

    and any repeated objects from
        {
            "py/id": 1
        }
    to the object with the same reconstruction id (from top to bottom).
    """

    def __init__(self):
        self.decoded_objects: List[Union[Node, EdgeType]] = []

    def _decode_id(self, id: int) -> Union[Node, EdgeType]:
        """Decode the object with the given `py/id` value of `id`.

        Args:
            id: The `py/id` value to decode (1-indexed).
            objects: The dictionary of objects that have already been decoded.

        Returns:
            The object with the given `py/id` value.
        """
        return self.decoded_objects[id - 1]

    @staticmethod
    def _decode_state(state: dict) -> Node:
        """Reconstruct the `Node` object from 'py/state' key in the serialized nx_graph.

        We support states in either dictionary or tuple format:
        {
            "py/state": { "py/tuple": ["thorax1", 1.0] }
        }
        or
        {
            "py/state": {"name": "thorax1", "weight": 1.0}
        }

        Args:
            state: The state to decode, i.e. state = dict["py/state"]

        Returns:
            The `Node` object reconstructed from the state.
        """

        if "py/tuple" in state:
            return Node(*state["py/tuple"])

        return Node(**state)

    @staticmethod
    def _decode_object_dict(object_dict) -> Node:
        """Decode dict containing `py/object` key in the serialized nx_graph.

        Args:
            object_dict: The dict to decode, i.e.
                object_dict = {"py/object": ..., "py/state":...}

        Raises:
            ValueError: If object_dict does not have 'py/object' and 'py/state' keys.
            ValueError: If object_dict['py/object'] is not 'sleap.skeleton.Node'.

        Returns:
            The decoded `Node` object.
        """

        if object_dict["py/object"] != "sleap.skeleton.Node":
            raise ValueError("Only 'sleap.skeleton.Node' objects are supported.")

        node: Node = SkeletonDecoder._decode_state(state=object_dict["py/state"])
        return node

    def _decode_node(self, encoded_node: dict) -> Node:
        """Decode an item believed to be an encoded `Node` object.

        Also updates the list of decoded objects.

        Args:
            encoded_node: The encoded node to decode.

        Returns:
            The decoded node and the updated list of decoded objects.
        """

        if isinstance(encoded_node, int):
            # Using index mapping to replace the object (load from Labels)
            return encoded_node
        elif "py/object" in encoded_node:
            decoded_node: Node = SkeletonDecoder._decode_object_dict(encoded_node)
            self.decoded_objects.append(decoded_node)
        elif "py/id" in encoded_node:
            decoded_node: Node = self._decode_id(encoded_node["py/id"])

        return decoded_node

    def _decode_nodes(self, encoded_nodes: List[dict]) -> List[Dict[str, Node]]:
        """Decode the 'nodes' key in the serialized nx_graph.

        The encoded_nodes is a list of dictionary of two types:
            - A dictionary with 'py/object' and 'py/state' keys.
            - A dictionary with 'py/id' key.

        Args:
            encoded_nodes: The list of encoded nodes to decode.

        Returns:
            The decoded nodes.
        """

        decoded_nodes: List[Dict[str, Node]] = []
        for e_node_dict in encoded_nodes:
            e_node = e_node_dict["id"]
            d_node = self._decode_node(e_node)
            decoded_nodes.append({"id": d_node})

        return decoded_nodes

    def _decode_reduce_dict(self, reduce_dict: Dict[str, List[dict]]) -> EdgeType:
        """Decode the 'reduce' key in the serialized nx_graph.

        The reduce_dict is a dictionary in the following format:
        {
            "py/reduce": [
                { "py/type": "sleap.skeleton.EdgeType" },
                { "py/tuple": [1] }
            ]
        }

        Args:
            reduce_dict: The dictionary to decode i.e. reduce_dict = {"py/reduce": ...}

        Returns:
            The decoded `EdgeType` object.
        """

        reduce_list = reduce_dict["py/reduce"]
        has_py_type = has_py_tuple = False
        for reduce_item in reduce_list:
            if reduce_item is None:
                # Sometimes the reduce list has None values, skip them
                continue
            if (
                "py/type" in reduce_item
                and reduce_item["py/type"] == "sleap.skeleton.EdgeType"
            ):
                has_py_type = True
            elif "py/tuple" in reduce_item:
                edge_type: int = reduce_item["py/tuple"][0]
                has_py_tuple = True

        if not has_py_type or not has_py_tuple:
            raise ValueError(
                "Only 'sleap.skeleton.EdgeType' objects are supported. "
                "The 'py/reduce' list must have dictionaries with 'py/type' and "
                "'py/tuple' keys."
                f"\n\tHas py/type: {has_py_type}\n\tHas py/tuple: {has_py_tuple}"
            )

        edge = EdgeType(edge_type)
        self.decoded_objects.append(edge)

        return edge

    def _decode_edge_type(self, encoded_edge_type: dict) -> EdgeType:
        """Decode the 'type' key in the serialized nx_graph.

        Args:
            encoded_edge_type: a dictionary with either 'py/id' or 'py/reduce' key.

        Returns:
            The decoded `EdgeType` object.
        """

        if "py/reduce" in encoded_edge_type:
            edge_type = self._decode_reduce_dict(encoded_edge_type)
        else:
            # Expect a "py/id" instead of "py/reduce"
            edge_type = self._decode_id(encoded_edge_type["py/id"])
        return edge_type

    def _decode_links(
        self, links: List[dict]
    ) -> List[Dict[str, Union[int, Node, EdgeType]]]:
        """Decode the 'links' key in the serialized nx_graph.

        The links are the edges in the graph and will have the following keys:
            - source: The source node of the edge.
            - target: The destination node of the edge.
            - type: The type of the edge (e.g. BODY, SYMMETRY).
        and more.

        Args:
            encoded_links: The list of encoded links to decode.
        """

        for link in links:
            for key, value in link.items():
                if key == "source":
                    link[key] = self._decode_node(value)
                elif key == "target":
                    link[key] = self._decode_node(value)
                elif key == "type":
                    link[key] = self._decode_edge_type(value)

        return links

    @staticmethod
    def decode_preview_image(
        img_b64: bytes, return_bytes: bool = False
    ) -> Union[Image.Image, bytes]:
        """Decode a skeleton preview image byte string representation to a `PIL.Image`

        Args:
            img_b64: a byte string representation of a skeleton preview image
            return_bytes: whether to return the decoded image as bytes

        Returns:
            Either a PIL.Image of the skeleton preview image or the decoded image as bytes
            (if `return_bytes` is True).
        """
        bytes = base64.b64decode(img_b64)
        if return_bytes:
            return bytes

        buffer = BytesIO(bytes)
        img = Image.open(buffer)
        return img

    def _decode(self, json_str: str):
        dicts = json.loads(json_str)

        # Enforce same format across template and non-template skeletons
        if "nx_graph" not in dicts:
            # Non-template skeletons use the dicts as the "nx_graph"
            dicts = {"nx_graph": dicts}

        # Decode the graph
        nx_graph = dicts["nx_graph"]

        self.decoded_objects = []  # Reset the decoded objects incase reusing decoder
        for key, value in nx_graph.items():
            if key == "nodes":
                nx_graph[key] = self._decode_nodes(value)
            elif key == "links":
                nx_graph[key] = self._decode_links(value)

        # Decode the preview image (if it exists)
        preview_image = dicts.get("preview_image", None)
        if preview_image is not None:
            dicts["preview_image"] = SkeletonDecoder.decode_preview_image(
                preview_image["py/b64"], return_bytes=True
            )

        return dicts

    @classmethod
    def decode(cls, json_str: str) -> Dict:
        """Decode the given json string into a dictionary.

        Returns:
            A dict with `Node`s, `EdgeType`s, and `bytes` decoded/reconstructed.
        """
        decoder = cls()
        return decoder._decode(json_str)


class SkeletonEncoder:
    """Replace jsonpickle.encode with our own encoder.

    The input is a dictionary containing python objects that need to be encoded as
    JSON strings. The output is a JSON string that represents the input dictionary.

    `Node(name='neck', weight=1.0)` =>
        {
            "py/object": "sleap.Skeleton.Node",
            "py/state": {"py/tuple" ["neck", 1.0]}
        }

    `<EdgeType.BODY: 1>` =>
        {"py/reduce": [
            {"py/type": "sleap.Skeleton.EdgeType"},
            {"py/tuple": [1] }
            ]
        }`

    Where `name` and `weight` are the attributes of the `Node` class; weight is always 1.0.
    `EdgeType` is an enum with values `BODY = 1` and `SYMMETRY = 2`.

    See sleap.skeleton.Node and sleap.skeleton.EdgeType.

    If the object has been "seen" before, it will not be encoded as the full JSON string
    but referenced by its `py/id`, which starts at 1 and indexes the objects in the
    order they are seen so that the second time the first object is used, it will be
    referenced as `{"py/id": 1}`.
    """

    def __init__(self):
        """Initializes a SkeletonEncoder instance."""
        # Maps object id to py/id
        self._encoded_objects: Dict[int, int] = {}

    @classmethod
    def encode(cls, data: Dict[str, Any]) -> str:
        """Encodes the input dictionary as a JSON string.

        Args:
            data: The data to encode.

        Returns:
            json_str: The JSON string representation of the data.
        """

        # This is required for backwards compatibility with SLEAP <=1.3.4
        sorted_data = cls._recursively_sort_dict(data)

        encoder = cls()
        encoded_data = encoder._encode(sorted_data)
        json_str = json.dumps(encoded_data)
        return json_str

    @staticmethod
    def _recursively_sort_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sorts the dictionary by keys."""
        sorted_dict = dict(sorted(dictionary.items()))
        for key, value in sorted_dict.items():
            if isinstance(value, dict):
                sorted_dict[key] = SkeletonEncoder._recursively_sort_dict(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        sorted_dict[key][i] = SkeletonEncoder._recursively_sort_dict(
                            item
                        )
        return sorted_dict

    def _encode(self, obj: Any) -> Any:
        """Recursively encodes the input object.

        Args:
            obj: The object to encode. Can be a dictionary, list, Node, EdgeType or
                primitive data type.

        Returns:
            The encoded object as a dictionary.
        """
        if isinstance(obj, dict):
            encoded_obj = {}
            for key, value in obj.items():
                if key == "links":
                    encoded_obj[key] = self._encode_links(value)
                else:
                    encoded_obj[key] = self._encode(value)
            return encoded_obj
        elif isinstance(obj, list):
            return [self._encode(v) for v in obj]
        elif isinstance(obj, EdgeType):
            return self._encode_edge_type(obj)
        elif isinstance(obj, Node):
            return self._encode_node(obj)
        else:
            return obj  # Primitive data types

    def _encode_links(self, links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encodes the list of links (edges) in the skeleton graph.

        Args:
            links: A list of dictionaries, each representing an edge in the graph.

        Returns:
            A list of encoded edge dictionaries with keys ordered as specified.
        """
        encoded_links = []
        for link in links:
            # Use a regular dict (insertion order preserved in Python 3.7+)
            encoded_link = {}

            for key, value in link.items():
                if key in ("source", "target"):
                    encoded_link[key] = self._encode_node(value)
                elif key == "type":
                    encoded_link[key] = self._encode_edge_type(value)
                else:
                    encoded_link[key] = self._encode(value)
            encoded_links.append(encoded_link)

        return encoded_links

    def _encode_node(self, node: Union["Node", int]) -> Dict[str, Any]:
        """Encodes a Node object.

        Args:
            node: The Node object to encode or integer index. The latter requires that
                the class has the `idx_to_node` attribute set.

        Returns:
            The encoded `Node` object as a dictionary.
        """
        if isinstance(node, int):
            # We sometimes have the node object already replaced by its index (when
            # `node_to_idx` is provided). In this case, the node is already encoded.
            return node

        # Check if object has been encoded before
        first_encoding = self._is_first_encoding(node)
        py_id = self._get_or_assign_id(node, first_encoding)
        if first_encoding:
            # Full encoding
            return {
                "py/object": "sleap.skeleton.Node",
                "py/state": {"py/tuple": [node.name, node.weight]},
            }
        else:
            # Reference by py/id
            return {"py/id": py_id}

    def _encode_edge_type(self, edge_type: "EdgeType") -> Dict[str, Any]:
        """Encodes an EdgeType object.

        Args:
            edge_type: The EdgeType object to encode. Either `EdgeType.BODY` or
                `EdgeType.SYMMETRY` enum with values 1 and 2 respectively.

        Returns:
            The encoded EdgeType object as a dictionary.
        """
        # Check if object has been encoded before
        first_encoding = self._is_first_encoding(edge_type)
        py_id = self._get_or_assign_id(edge_type, first_encoding)
        if first_encoding:
            # Full encoding
            return {
                "py/reduce": [
                    {"py/type": "sleap.skeleton.EdgeType"},
                    {"py/tuple": [edge_type.value]},
                ]
            }
        else:
            # Reference by py/id
            return {"py/id": py_id}

    def _get_or_assign_id(self, obj: Any, first_encoding: bool) -> int:
        """Gets or assigns a py/id for the object.

        Args:
            The object to get or assign a py/id for.

        Returns:
            The py/id assigned to the object.
        """
        # Object id is unique for each object in the current session
        obj_id = id(obj)
        # Assign a py/id to the object if it hasn't been assigned one yet
        if first_encoding:
            py_id = len(self._encoded_objects) + 1  # py/id starts at 1
            # Assign the py/id to the object and store it in _encoded_objects
            self._encoded_objects[obj_id] = py_id
        return self._encoded_objects[obj_id]

    def _is_first_encoding(self, obj: Any) -> bool:
        """Checks if the object is being encoded for the first time.

        Args:
            obj: The object to check.

        Returns:
            True if this is the first encoding of the object, False otherwise.
        """
        obj_id = id(obj)
        first_time = obj_id not in self._encoded_objects
        return first_time


class Skeleton:
    """The main object for representing animal skeletons.

    The skeleton represents the constituent parts of the animal whose
    pose is being estimated.

    Attributes:
        _skeleton_idx: An index variable used to give skeletons a default name that
            should be unique across all skeletons.
        preview_image: A byte string containing an encoded preview image for the
            skeleton. Used only for templates.
        description: A text description of the skeleton. Used only for templates.
        _is_template: Whether this skeleton is a template. Used only for templates.
    """

    _skeleton_idx = count(0)
    preview_image: Optional[bytes] = None
    description: Optional[str] = None
    _is_template: bool = False

    def __init__(self, name: str = None):
        """Initialize an empty skeleton object.

        Skeleton objects, once created, can be modified by adding nodes
        and edges.

        Args:
            name: A name for this skeleton.
        """
        # If no skeleton was create, try to create a unique name for this Skeleton.
        if name is None or not isinstance(name, str) or not name:
            name = "Skeleton-" + str(next(self._skeleton_idx))

        # Since networkx does not keep edges in the order we insert them we need
        # to keep track of how many edges have been inserted so we can number them
        # as they are inserted and sort them by this numbering when the edge list
        # is returned.
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph(name=name, num_edges_inserted=0)

    def __repr__(self) -> str:
        """Return full description of the skeleton."""
        return (
            f"Skeleton(name='{self.name}', "
            f"description='{self.description}', "
            f"nodes={self.node_names}, "
            f"edges={self.edge_names}, "
            f"symmetries={self.symmetry_names}"
            ")"
        )

    def __str__(self) -> str:
        """Return short readable description of the skeleton."""
        description = self.description
        nodes = ", ".join(self.node_names)
        edges = ", ".join([f"{s}->{d}" for (s, d) in self.edge_names])
        symm = ", ".join([f"{s}<->{d}" for (s, d) in self.symmetry_names])
        return (
            "Skeleton("
            f"description={description}, "
            f"nodes=[{nodes}], "
            f"edges=[{edges}], "
            f"symmetries=[{symm}]"
            ")"
        )

    def matches(self, other: "Skeleton") -> bool:
        """Compare this `Skeleton` to another, ignoring name and node identities.

        Args:
            other: The other skeleton.

        Returns:
            `True` if the skeleton graphs are isomorphic and node names.
        """

        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the graphs are iso-morphic
        is_isomorphic = nx.is_isomorphic(
            self._graph, other._graph, node_match=dict_match
        )

        if not is_isomorphic:
            return False

        # Now check that the nodes have the same labels and order.
        for node1, node2 in zip(self._graph.nodes, other._graph.nodes):
            if node1.name != node2.name:
                return False

        return True

    @property
    def is_template(self) -> bool:
        """Return whether this skeleton is a template.

        If is_template is True, then the preview image and description are saved.
        If is_template is False, then the preview image and description are not saved.

        Only provided template skeletons are considered templates. To save a new
        template skeleton, change this to True before saving.
        """
        return self._is_template

    @is_template.setter
    def is_template(self, value: bool):
        """Set whether this skeleton is a template."""

        self._is_template = False
        if value and ((self.preview_image is None) or (self.description is None)):
            raise ValueError(
                "For a skeleton to be a template, it must have both a preview image "
                "and description. Checkout `generate_skeleton_preview_image` to "
                "generate a preview image."
            )

        self._is_template = value

    @property
    def is_arborescence(self) -> bool:
        """Return whether this skeleton graph forms an arborescence."""
        return nx.algorithms.tree.recognition.is_arborescence(self.graph)

    @property
    def in_degree_over_one(self) -> List[Node]:
        return [node for node, in_degree in self.graph.in_degree if in_degree > 1]

    @property
    def root_nodes(self) -> List[Node]:
        return [node for node, in_degree in self.graph.in_degree if in_degree == 0]

    @property
    def cycles(self) -> List[List[Node]]:
        return list(nx.algorithms.simple_cycles(self.graph))

    @property
    def graph(self):
        """Return a view on the subgraph of body nodes and edges for skeleton."""

        def edge_filter_fn(src, dst, edge_key):
            edge_data = self._graph.get_edge_data(src, dst, edge_key)
            return edge_data["type"] == EdgeType.BODY

        return nx.subgraph_view(self._graph, filter_edge=edge_filter_fn)

    @property
    def graph_symmetry(self):
        """Return subgraph of symmetric edges for skeleton."""

        def edge_filter_fn(src, dst, edge_key):
            edge_data = self._graph.get_edge_data(src, dst, edge_key)
            return edge_data["type"] == EdgeType.SYMMETRY

        return nx.subgraph_view(self._graph, filter_edge=edge_filter_fn)

    @staticmethod
    def find_unique_nodes(skeletons: List["Skeleton"]) -> List[Node]:
        """Find all unique nodes from a list of skeletons.

        Args:
            skeletons: The list of skeletons.

        Returns:
            A list of unique `Node` objects.
        """
        return list({node for skeleton in skeletons for node in skeleton.nodes})

    @staticmethod
    def make_cattr(idx_to_node: Dict[int, Node] = None) -> cattr.Converter:
        """Make cattr.Convert() for `Skeleton`.

        Make a cattr.Converter() that registers structure/unstructure
        hooks for Skeleton objects to handle serialization of skeletons.

        Args:
            idx_to_node: A dict that maps node index to Node objects.

        Returns:
            A cattr.Converter() instance for skeleton serialization
            and deserialization.
        """
        node_to_idx = (
            {node: idx for idx, node in idx_to_node.items()}
            if idx_to_node is not None
            else None
        )

        _cattr = cattr.Converter()
        _cattr.register_unstructure_hook(
            Skeleton, lambda x: Skeleton.to_dict(x, node_to_idx)
        )
        _cattr.register_structure_hook(
            Skeleton, lambda x, cls: Skeleton.from_dict(x, idx_to_node)
        )
        return _cattr

    @property
    def name(self) -> str:
        """Get the name of the skeleton.

        Returns:
            A string representing the name of the skeleton.
        """
        return self._graph.name

    @name.setter
    def name(self, name: str):
        """Set skeleton name (no-op).

        A skeleton object cannot change its name.

        This property is immutable because it is used to hash skeletons.
        If you want to rename a Skeleton you must use the class
        method :code:`rename_skeleton`: ::

           >>> new_skeleton = Skeleton.rename_skeleton(
           >>>     skeleton=old_skeleton, name="New Name")

        Args:
            name: The name of the Skeleton.

        Raises:
            NotImplementedError: Error is always raised.
        """
        raise NotImplementedError(
            "Cannot change Skeleton name, it is immutable since "
            "it is used for hashing. Create a copy of the skeleton "
            "with new name using "
            f"new_skeleton = Skeleton.rename(skeleton, '{name}'))"
        )

    @classmethod
    def rename_skeleton(cls, skeleton: "Skeleton", name: str) -> "Skeleton":
        """Make copy of skeleton with new name.

        This property is immutable because it is used to hash skeletons.
        If you want to rename a Skeleton you must use this class method. ::

           >>> new_skeleton = Skeleton.rename_skeleton(
           >>>     skeleton=old_skeleton, name="New Name")

        Args:
            skeleton: The skeleton to copy.
            name: The name of the new skeleton.

        Returns:
            A new deep copied skeleton with the changed name.
        """
        new_skeleton = cls(name)
        new_skeleton._graph = copy.deepcopy(skeleton._graph)
        new_skeleton._graph.name = name
        return new_skeleton

    @property
    def nodes(self) -> List[Node]:
        """Get a list of :class:`Node`s.

        Returns:
            A list of :class:`Node`s
        """
        return list(self._graph.nodes)

    @property
    def node_names(self) -> List[str]:
        """Get a list of node names.

        Returns:
            A list of node names.
        """
        return [node.name for node in self.nodes]

    @property
    def edges(self) -> List[Tuple[Node, Node]]:
        """Get a list of edge tuples.

        Returns:
            list of (src_node, dst_node)
        """
        edge_list = [
            (d["edge_insert_idx"], src, dst)
            for src, dst, key, d in self._graph.edges(keys=True, data=True)
            if d["type"] == EdgeType.BODY
        ]

        # We don't want to return the edge list in the order it is stored. We
        # want to use the insertion order. Sort by the insertion index for each
        # edge then drop it from the edge list.
        edge_list = [(src, dst) for _, src, dst in sorted(edge_list)]

        return edge_list

    @property
    def edge_names(self) -> List[Tuple[str, str]]:
        """Get a list of edge name tuples.

        Returns:
            list of (src_node.name, dst_node.name)
        """
        edge_list = [
            (d["edge_insert_idx"], src.name, dst.name)
            for src, dst, key, d in self._graph.edges(keys=True, data=True)
            if d["type"] == EdgeType.BODY
        ]

        # We don't want to return the edge list in the order it is stored. We
        # want to use the insertion order. Sort by the insertion index for each
        # edge then drop it from the edge list.
        edge_list = [(src, dst) for _, src, dst in sorted(edge_list)]

        return [(src.name, dst.name) for src, dst in self.edges]

    @property
    def edge_inds(self) -> List[Tuple[int, int]]:
        """Get a list of edges as node indices.

        Returns:
            A list of (src_node_ind, dst_node_ind), where indices are subscripts into
            the Skeleton.nodes list.
        """
        return [
            (self.nodes.index(src_node), self.nodes.index(dst_node))
            for src_node, dst_node in self.edges
        ]

    @property
    def edges_full(self) -> List[Tuple[Node, Node, Any, Any]]:
        """Get a list of edge tuples with keys and attributes.

        Returns:
            list of (src_node, dst_node, key, attributes)
        """
        return [
            (src, dst, key, attr)
            for src, dst, key, attr in self._graph.edges(keys=True, data=True)
            if attr["type"] == EdgeType.BODY
        ]

    @property
    def symmetries(self) -> List[Tuple[Node, Node]]:
        """Get a list of all symmetries without duplicates.

        Returns:
            list of (node1, node2)
        """
        # Find all symmetric edges
        symmetries = [
            (src, dst)
            for src, dst, key, edge_type in self._graph.edges(keys=True, data="type")
            if edge_type == EdgeType.SYMMETRY
        ]
        # Get rid of duplicates
        symmetries = list(
            set([tuple(sorted(e, key=operator.attrgetter("name"))) for e in symmetries])
        )
        return symmetries

    @property
    def symmetry_names(self) -> List[Tuple[str, str]]:
        """List of symmetry edges as tuples of node names."""
        return [(s.name, d.name) for (s, d) in self.symmetries]

    @property
    def symmetries_full(self) -> List[Tuple[Node, Node, Any, Any]]:
        """Get a list of all symmetries with keys and attributes.

        Note: The returned list will contain duplicates (node1, node2)
        and (node2, node1).

        Returns:
            list of (node1, node2, key, attr)
        """
        # Find all symmetric edges
        return [
            (src, dst, key, attr)
            for src, dst, key, attr in self._graph.edges(keys=True, data=True)
            if attr["type"] == EdgeType.SYMMETRY
        ]

    @property
    def symmetric_inds(self) -> np.ndarray:
        """Return the symmetric nodes as an array of indices."""
        return np.array(
            [
                [self.nodes.index(node1), self.nodes.index(node2)]
                for node1, node2 in self.symmetries
            ]
        )

    def node_to_index(self, node: NodeRef) -> int:
        """Return the index of the node, accepts either `Node` or name.

        Args:
            node: The name of the node or the Node object.

        Raises:
            ValueError if node cannot be found in skeleton.

        Returns:
            The index of the node in the graph.
        """
        node_list = list(self._graph.nodes)
        try:
            return node_list.index(node)
        except ValueError:
            return node_list.index(self.find_node(node))

    def edge_to_index(self, source: NodeRef, destination: NodeRef) -> int:
        """Return the index of edge from source to destination."""
        source = self.find_node(source)
        destination = self.find_node(destination)
        edge = (source, destination)
        if edge in self.edges:
            return self.edges.index(edge)

        return -1

    def add_node(self, name: str):
        """Add a node representing an animal part to the skeleton.

        Args:
            name: The name of the node to add to the skeleton.
                This name must be unique within the skeleton.

        Raises:
            ValueError: If name is not unique.
        """
        if not isinstance(name, str):
            raise TypeError("Cannot add nodes to the skeleton that are not str")

        if name in self.node_names:
            raise ValueError("Skeleton already has a node named ({})".format(name))

        self._graph.add_node(Node(name))

    def add_nodes(self, name_list: List[str]):
        """Add a list of nodes representing animal parts to the skeleton.

        Args:
            name_list: List of strings representing the nodes.
        """
        for node in name_list:
            self.add_node(node)

    def delete_node(self, name: str):
        """Remove a node from the skeleton.

        The method removes a node from the skeleton and any edge that is
        connected to it.

        Args:
            name: The name of the node to remove

        Raises:
            ValueError: If node cannot be found.

        Returns:
            None
        """
        try:
            node = self.find_node(name)
            self._graph.remove_node(node)
        except nx.NetworkXError:
            raise ValueError(
                "The node named ({}) does not exist, cannot remove it.".format(name)
            )

    def find_node(self, name: NodeRef) -> Node:
        """Find node in skeleton by name of node.

        Args:
            name: The name of the :class:`Node` (or a :class:`Node`)

        Returns:
            `Node`, or None if no match found
        """
        if isinstance(name, Node):
            name = name.name

        nodes = [node for node in self.nodes if node.name == name]

        if len(nodes) == 1:
            return nodes[0]

        if len(nodes) > 1:
            raise ValueError("Found multiple nodes named ({}).".format(name))

        return None

    def find_neighbors(self, node: NodeRef) -> List[Node]:
        """Find nodes that are predecessors or successors from a node.

        Args:
            node: Name or `Node` instance.

        Returns:
            A list of `Node` objects that are neighbors to the node.
        """
        node = self.find_node(node)
        return list(self.graph.predecessors(node)) + list(self.graph.successors(node))

    def add_edge(self, source: str, destination: str):
        """Add an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.
        Raises:
            ValueError: If source or destination nodes cannot be found,
                or if edge already exists between those nodes.

        Returns:
            None.
        """
        if isinstance(source, Node):
            source_node = source
            source = source_node.name
        else:
            source_node = self.find_node(source)

        if isinstance(destination, Node):
            destination_node = destination
            destination = destination_node.name
        else:
            destination_node = self.find_node(destination)

        if source_node is None:
            raise ValueError(
                "Skeleton does not have source node named ({})".format(source)
            )

        if destination_node is None:
            raise ValueError(
                "Skeleton does not have destination node named ({})".format(destination)
            )

        if self._graph.has_edge(source_node, destination_node):
            raise ValueError(
                "Skeleton already has an edge between ({}) and ({}).".format(
                    source, destination
                )
            )

        self._graph.add_edge(
            source_node,
            destination_node,
            type=EdgeType.BODY,
            edge_insert_idx=self._graph.graph["num_edges_inserted"],
        )
        self._graph.graph["num_edges_inserted"] = (
            self._graph.graph["num_edges_inserted"] + 1
        )

    def delete_edge(self, source: str, destination: str):
        """Delete an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

        Raises:
            ValueError: If skeleton does not have either source node,
                destination node, or edge between them.

        Returns:
            None
        """
        if isinstance(source, Node):
            source_node = source
            source = source_node.name
        else:
            source_node = self.find_node(source)

        if isinstance(destination, Node):
            destination_node = destination
            destination = destination_node.name
        else:
            destination_node = self.find_node(destination)

        if source_node is None:
            raise ValueError(
                "Skeleton does not have source node named ({})".format(source)
            )

        if destination_node is None:
            raise ValueError(
                "Skeleton does not have destination node named ({})".format(destination)
            )

        if not self._graph.has_edge(source_node, destination_node):
            raise ValueError(
                "Skeleton has no edge between ({}) and ({}).".format(
                    source, destination
                )
            )

        self._graph.remove_edge(source_node, destination_node)

    def clear_edges(self):
        """Delete all edges in skeleton."""
        for src, dst in self.edges:
            self.delete_edge(src, dst)

    def add_symmetry(self, node1: str, node2: str):
        """Specify that two parts (nodes) in skeleton are symmetrical.

        Certain parts of an animal body can be related as symmetrical
        parts in a pair. For example, left and right hands of a person.

        Args:
            node1: The name of the first part in the symmetric pair
            node2: The name of the second part in the symmetric pair

        Raises:
            ValueError: If node1 and node2 match, or if there is already
                a symmetry between them.

        Returns:
            None

        """
        node1_node, node2_node = self.find_node(node1), self.find_node(node2)

        # We will represent symmetric pairs in the skeleton via additional edges in the
        # _graph. These edges will have a special attribute signifying they are not part
        # of the skeleton itself

        if node1 == node2:
            raise ValueError("Cannot add symmetry to the same node.")

        if self.get_symmetry(node1) is not None:
            raise ValueError(
                f"{node1} is already symmetric with {self.get_symmetry(node1)}."
            )

        if self.get_symmetry(node2) is not None:
            raise ValueError(
                f"{node2} is already symmetric with {self.get_symmetry(node2)}."
            )

        self._graph.add_edge(node1_node, node2_node, type=EdgeType.SYMMETRY)
        self._graph.add_edge(node2_node, node1_node, type=EdgeType.SYMMETRY)

    def delete_symmetry(self, node1: NodeRef, node2: NodeRef):
        """Delete a previously established symmetry between two nodes.

        Args:
            node1: One node (by `Node` object or name) in symmetric pair.
            node2: Other node in symmetric pair.

        Raises:
            ValueError: If there's no symmetry between node1 and node2.

        Returns:
            None
        """
        node1_node = self.find_node(node1)
        node2_node = self.find_node(node2)

        if (
            self.get_symmetry(node1) != node2_node
            or self.get_symmetry(node2) != node1_node
        ):
            raise ValueError(f"Nodes {node1}, {node2} are not symmetric.")

        edges = [
            (src, dst, key)
            for src, dst, key, edge_type in self._graph.edges(
                [node1_node, node2_node], keys=True, data="type"
            )
            if edge_type == EdgeType.SYMMETRY
        ]
        self._graph.remove_edges_from(edges)

    def get_symmetry(self, node: NodeRef) -> Optional[Node]:
        """Return the node symmetric with the specified node.

        Args:
            node: Node (by `Node` object or name) to query.

        Raises:
            ValueError: If node has more than one symmetry.

        Returns:
            The symmetric :class:`Node`, None if no symmetry.
        """
        node_node = self.find_node(node)

        symmetry = [
            dst
            for src, dst, edge_type in self._graph.edges(node_node, data="type")
            if edge_type == EdgeType.SYMMETRY
        ]

        if len(symmetry) == 0:
            return None
        elif len(symmetry) == 1:
            return symmetry[0]
        else:
            raise ValueError(f"{node} has more than one symmetry.")

    def get_symmetry_name(self, node: NodeRef) -> Optional[str]:
        """Return the name of the node symmetric with the specified node.

        Args:
            node: Node (by `Node` object or name) to query.

        Returns:
            Name of symmetric node, None if no symmetry.
        """
        symmetric_node = self.get_symmetry(node)
        return None if symmetric_node is None else symmetric_node.name

    def __getitem__(self, node_name: str) -> dict:
        """Retrieve the node data associated with skeleton node.

        Args:
            node_name: The name from which to retrieve data.

        Raises:
            ValueError: If node cannot be found.

        Returns:
            A dictionary of data associated with this node.

        """
        node = self.find_node(node_name)
        if node is None:
            raise ValueError(f"Skeleton does not have node named '{node_name}'.")

        return self._graph.nodes.data()[node]

    def __contains__(self, node_name: str) -> bool:
        """Check if specified node exists in skeleton.

        Args:
            node_name: the node name to query

        Returns:
            True if node is in the skeleton.
        """
        return self.has_node(node_name)

    def __len__(self) -> int:
        """Return the number of nodes in the skeleton."""
        return len(self.nodes)

    def relabel_node(self, old_name: str, new_name: str):
        """Relabel a single node to a new name.

        Args:
            old_name: The old name of the node.
            new_name: The new name of the node.

        Returns:
            None
        """
        self.relabel_nodes({old_name: new_name})

    def relabel_nodes(self, mapping: Dict[str, str]):
        """Relabel the nodes of the skeleton.

        Args:
            mapping: A dictionary with the old labels as keys and new
                labels as values. A partial mapping is allowed.

        Raises:
            ValueError: If node already present with one of the new names.

        Returns:
            None
        """
        existing_nodes = self.nodes
        for old_name, new_name in mapping.items():
            if self.has_node(new_name):
                raise ValueError("Cannot relabel a node to an existing name.")
            node = self.find_node(old_name)
            if node is not None:
                node.name = new_name

        # self._graph = nx.relabel_nodes(G=self._graph, mapping=mapping)

    def has_node(self, name: str) -> bool:
        """
        Check whether the skeleton has a node.

        Args:
            name: The name of the node to check for.

        Returns:
            True for yes, False for no.

        """
        return name in self.node_names

    def has_nodes(self, names: Iterable[str]) -> bool:
        """
        Check whether the skeleton has a list of nodes.

        Args:
            names: The list names of the nodes to check for.

        Returns:
            True for yes, False for no.

        """
        current_node_names = self.node_names
        for name in names:
            if name not in current_node_names:
                return False

        return True

    def has_edge(self, source_name: str, dest_name: str) -> bool:
        """
        Check whether the skeleton has an edge.

        Args:
            source_name: The name of the source node for the edge.
            dest_name: The name of the destination node for the edge.

        Returns:
            True is yes, False if no.

        """
        source_node, destination_node = (
            self.find_node(source_name),
            self.find_node(dest_name),
        )
        return self._graph.has_edge(source_node, destination_node)

    @staticmethod
    def to_dict(obj: "Skeleton", node_to_idx: Optional[Dict[Node, int]] = None) -> Dict:
        """
        Convert skeleton to dict; used for saving as JSON.

        Args:
            obj: the :object:`Skeleton` to convert
            node_to_idx: optional dict which maps :class:`Node`sto index
                in some list. This is used when saving
                :class:`Labels`where we want to serialize the
                :class:`Nodes` outside the :class:`Skeleton` object.
                If given, then we replace each :class:`Node` with
                specified index before converting :class:`Skeleton`.
                Otherwise, we convert :class:`Node` objects with the rest of
                the :class:`Skeleton`.
        Returns:
            dict with data from skeleton
        """

        # This is a weird hack to serialize the whole _graph into a dict.
        # I use the underlying to_json and parse it.
        return json.loads(obj.to_json(node_to_idx=node_to_idx))

    @classmethod
    def from_dict(cls, d: Dict, node_to_idx: Dict[Node, int] = None) -> "Skeleton":
        """
        Create skeleton from dict; used for loading from JSON.

        Args:
            d: the dict from which to deserialize
            node_to_idx: optional dict which maps :class:`Node`sto index
                in some list. This is used when saving
                :class:`Labels`where we want to serialize the
                :class:`Nodes` outside the :class:`Skeleton` object.
                If given, then we replace each :class:`Node` with
                specified index before converting :class:`Skeleton`.
                Otherwise, we convert :class:`Node` objects with the rest of
                the :class:`Skeleton`.

        Returns:
            :class:`Skeleton`.

        """
        return Skeleton.from_json(json.dumps(d), node_to_idx)

    @classmethod
    def from_names_and_edge_inds(
        cls, node_names: List[Text], edge_inds: List[Tuple[int, int]] = None
    ) -> "Skeleton":
        """Create skeleton from a list of node names and edge indices.

        Args:
            node_names: List of strings defining the nodes.
            edge_inds: List of tuples in the form (src_node_ind, dst_node_ind). If not
                specified, the resulting skeleton will have no edges.

        Returns:
            The instantiated skeleton.
        """

        skeleton = cls()
        skeleton.add_nodes(node_names)
        if edge_inds is not None:
            for src, dst in edge_inds:
                skeleton.add_edge(node_names[src], node_names[dst])
        return skeleton

    def to_json(self, node_to_idx: Optional[Dict[Node, int]] = None) -> str:
        """Convert the :class:`Skeleton` to a JSON representation.

        Args:
            node_to_idx: optional dict which maps :class:`Node`sto index
                in some list. This is used when saving
                :class:`Labels`where we want to serialize the
                :class:`Nodes` outside the :class:`Skeleton` object.
                If given, then we replace each :class:`Node` with
                specified index before converting :class:`Skeleton`.
                Otherwise, we convert :class:`Node` objects with the rest of
                the :class:`Skeleton`.

        Returns:
            A string containing the JSON representation of the skeleton.
        """

        if node_to_idx is not None:
            # Map Nodes to int
            indexed_node_graph = nx.relabel_nodes(G=self._graph, mapping=node_to_idx)
        else:
            # Keep graph nodes as Node objects
            indexed_node_graph = self._graph

        # Encode to JSON
        graph = json_graph.node_link_data(indexed_node_graph)

        # SLEAP v1.3.0 added `description` and `preview_image` to `Skeleton`, but saving
        # these fields breaks data format compatibility. Currently, these are only
        # added in our custom template skeletons. To ensure backwards data format
        # compatibilty of user data, we only save these fields if they are not None.
        if self.is_template:
            data = {
                "nx_graph": graph,
                "description": self.description,
                "preview_image": self.preview_image,
            }
        else:
            data = graph

        json_str = SkeletonEncoder.encode(data)

        return json_str

    def save_json(self, filename: str, node_to_idx: Optional[Dict[Node, int]] = None):
        """
        Save the :class:`Skeleton` as JSON file.

        Output the complete skeleton to a file in JSON format.

        Args:
            filename: The filename to save the JSON to.
            node_to_idx: optional dict which maps :class:`Node`sto index
                in some list. This is used when saving
                :class:`Labels`where we want to serialize the
                :class:`Nodes` outside the :class:`Skeleton` object.
                If given, then we replace each :class:`Node` with
                specified index before converting :class:`Skeleton`.
                Otherwise, we convert :class:`Node` objects with the rest of
                the :class:`Skeleton`.

        Returns:
            None
        """

        json_str = self.to_json(node_to_idx)

        with open(filename, "w") as file:
            file.write(json_str)

    @classmethod
    def from_json(
        cls, json_str: str, idx_to_node: Dict[int, Node] = None
    ) -> "Skeleton":
        """Instantiate :class:`Skeleton` from JSON string.

        Args:
            json_str: The JSON encoded Skeleton.
            idx_to_node: optional dict which maps an int (indexing a
                list of :class:`Node` objects) to the already
                deserialized :class:`Node`.
                This should invert `node_to_idx` we used when saving.
                If not given, then we'll assume each :class:`Node` was
                left in the :class:`Skeleton` when it was saved.

        Returns:
            An instance of the `Skeleton` object decoded from the JSON.
        """
        dicts: dict = SkeletonDecoder.decode(json_str)
        nx_graph = dicts.get("nx_graph", dicts)
        graph = json_graph.node_link_graph(nx_graph)

        # Replace graph node indices with corresponding nodes from node_map
        if idx_to_node is not None:
            graph = nx.relabel_nodes(G=graph, mapping=idx_to_node)

        skeleton = Skeleton()
        skeleton._graph = graph
        skeleton.description = dicts.get("description", None)
        skeleton.preview_image = dicts.get("preview_image", None)

        return skeleton

    @classmethod
    def load_json(
        cls, filename: str, idx_to_node: Dict[int, Node] = None
    ) -> "Skeleton":
        """Load a skeleton from a JSON file.

        This method will load the Skeleton from JSON file saved
        with; :meth:`~Skeleton.save_json`

        Args:
            filename: The file that contains the JSON.
            idx_to_node: optional dict which maps an int (indexing a
                list of :class:`Node` objects) to the already
                deserialized :class:`Node`.
                This should invert `node_to_idx` we used when saving.
                If not given, then we'll assume each :class:`Node` was
                left in the :class:`Skeleton` when it was saved.

        Returns:
            The `Skeleton` object stored in the JSON filename.

        """

        with open(filename, "r") as file:
            skeleton = cls.from_json(file.read(), idx_to_node)

        return skeleton

    @classmethod
    def load_hdf5(cls, file: H5FileRef, name: str) -> List["Skeleton"]:
        """
        Load a specific skeleton (by name) from the HDF5 file.

        Args:
            file: The file name or open h5py.File
            name: The name of the skeleton.

        Returns:
            The specified `Skeleton` instance stored in the HDF5 file.
        """
        if isinstance(file, str):
            with h5py.File(file, "r") as _file:
                skeletons = cls._load_hdf5(_file)  # Load all skeletons
        else:
            skeletons = cls._load_hdf5(file)

        return skeletons[name]

    @classmethod
    def load_all_hdf5(
        cls, file: H5FileRef, return_dict: bool = False
    ) -> Union[List["Skeleton"], Dict[str, "Skeleton"]]:
        """
        Load all skeletons found in the HDF5 file.

        Args:
            file: The file name or open h5py.File
            return_dict: Whether the the return value should be a dict
                where the keys are skeleton names and values the
                corresponding skeleton. If False, then method will
                return just a list of the skeletons.

        Returns:
            The skeleton instances stored in the HDF5 file.
            Either in List or Dict form.
        """
        if isinstance(file, str):
            with h5py.File(file, "r") as _file:
                skeletons = cls._load_hdf5(_file)  # Load all skeletons
        else:
            skeletons = cls._load_hdf5(file)

        if return_dict:
            return skeletons

        return list(skeletons.values())

    @classmethod
    def _load_hdf5(cls, file: h5py.File):

        skeletons = {}
        for name, json_str in file["skeleton"].attrs.items():
            skeletons[name] = cls.from_json(json_str)

        return skeletons

    @classmethod
    def save_all_hdf5(self, file: H5FileRef, skeletons: List["Skeleton"]):
        """
        Convenience method to save a list of skeletons to HDF5 file.

        Skeletons are saved as attributes of a /skeleton group in the
        file.

        Args:
            file: The filename or the open h5py.File object.
            skeletons: The list of skeletons to save.

        Raises:
            ValueError: If multiple skeletons have the same name.

        Returns:
            None
        """

        # Make sure no skeleton has the same name
        unique_names = {s.name for s in skeletons}

        if len(unique_names) != len(skeletons):
            raise ValueError("Cannot save multiple Skeleton's with the same name.")

        for skeleton in skeletons:
            skeleton.save_hdf5(file)

    def save_hdf5(self, file: H5FileRef):
        """
        Wrapper for HDF5 saving which takes either filename or h5py.File.

        Args:
            file: can be filename (string) or `h5py.File` object

        Returns:
            None
        """

        if isinstance(file, str):
            with h5py.File(file, "a") as _file:
                self._save_hdf5(_file)
        else:
            self._save_hdf5(file)

    def _save_hdf5(self, file: h5py.File):
        """
        Actual implementation of HDF5 saving.

        Args:
            file: The open h5py.File to write the skeleton data to.

        Returns:
            None
        """

        # All skeleton will be put as sub-groups in the skeleton group
        if "skeleton" not in file:
            all_sk_group = file.create_group("skeleton", track_order=True)
        else:
            all_sk_group = file.require_group("skeleton")

        # Write the dataset to JSON string, then store it in a string
        # attribute
        all_sk_group.attrs[self.name] = np.string_(self.to_json())

    @classmethod
    def load_mat(cls, filename: str) -> "Skeleton":
        """
        Load the skeleton from a Matlab MAT file.

        This is to support backwards compatibility with old LEAP
        MATLAB code and datasets.

        Args:
            filename: The name of the skeleton file

        Returns:
            An instance of the skeleton.
        """

        # Lets create a skeleton object, use the filename for the name since old LEAP
        # skeletons did not have names.
        skeleton = cls(name=filename)

        skel_mat = loadmat(filename)
        skel_mat["nodes"] = skel_mat["nodes"][0][0]  # convert to scalar
        skel_mat["edges"] = skel_mat["edges"] - 1  # convert to 0-based indexing

        node_names = skel_mat["nodeNames"]
        node_names = [str(n[0][0]) for n in node_names]
        skeleton.add_nodes(node_names)
        for k in range(len(skel_mat["edges"])):
            edge = skel_mat["edges"][k]
            skeleton.add_edge(
                source=node_names[edge[0]], destination=node_names[edge[1]]
            )

        return skeleton

    def __hash__(self):
        """
        Construct a hash from skeleton id.
        """
        return id(self)


cattr.register_unstructure_hook(Skeleton, lambda skeleton: Skeleton.to_dict(skeleton))
cattr.register_structure_hook(Skeleton, lambda dicts, cls: Skeleton.from_dict(dicts))

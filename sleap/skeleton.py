"""Implementation of skeleton data structure and API.

This module implements and API for creating animal skeleton's in LEAP. The goal
is to provide a common interface for defining the parts of the animal, their
connection to each other, and needed meta-data.

"""

import attr
import cattr
import numpy as np
import jsonpickle
import json
import networkx as nx
import h5py as h5
import copy

from enum import Enum
from itertools import count
from typing import Iterable, Union, List, Dict

from networkx.readwrite import json_graph
from scipy.io import loadmat, savemat


class EdgeType(Enum):
    """
    The skeleton graph can store different types of edges to represent
    different things. All edges must specify one or more of the following types.

        * BODY - these edges represent connections between parts or landmarks.
        * SYMMETRY - these edges represent symmetrical relationships between
        parts (e.g. left and right arms)
    """
    BODY = 1
    SYMMETRY = 2


@attr.s(auto_attribs=True, slots=True, cmp=False)
class Node:
    """
    The class :class:`Node` represents a potential skeleton node.
    (But note that nodes can exist without being part of a skeleton.)
    """

    name: str
    weight: float = 1.

    @staticmethod
    def from_names(name_list: str):
        nodes = []
        for name in name_list:
            nodes.append(Node(name))
        return nodes

    @classmethod
    def as_node(cls, node):
        return node if isinstance(node, cls) else cls(node)

    def matches(self, other):
        """
        Check whether all attributes match between two nodes.

        Args:
            other: The node to compare to this one.

        Returns:
            True if all attributes match, False otherwise.
        """
        return other.name == self.name and other.weight == self.weight


class Skeleton:
    """The main object for representing animal skeletons in LEAP.

    The skeleton represents the constituent parts of the animal whose pose
    is being estimated.

    """

    """
    A index variable used to give skeletons a default name that attemtpts to be
    unique across all skeletons. 
    """
    _skeleton_idx = count(0)

    def __init__(self, name: str = None):
        """Initialize an empty skeleton object.

        Skeleton objects, once they are created can be modified by adding nodes and edges.

        Args:
            name: A name for this skeleton.
        """

        # If no skeleton was create, try to create a unique name for this Skeleton.
        if name is None or type(name) is not str or len(name) == 0:
            name = "Skeleton-" + str(next(self._skeleton_idx))


        # Since networkx does not keep edges in the order we insert them we need
        # to keep track of how many edges have been inserted so we can number them
        # as they are inserted and sort them by this numbering when the edge list
        # is returned.
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph(name=name, num_edges_inserted=0)

    def matches(self, other: 'Skeleton'):
        """
        Compare this `Skeleton` to another, ignoring skeleton name and
        the identities of the `Node` objects in each graph.

        Args:
            other: The other skeleton.

        Returns:
            True if match, False otherwise.
        """
        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the graphs are iso-morphic
        is_isomorphic = nx.is_isomorphic(self._graph, other._graph, node_match=dict_match)

        if not is_isomorphic:
            return False

        # Now check that the nodes have the same labels and order. They can have
        # different weights I guess?!
        for node1, node2 in zip(self._graph.nodes, other._graph.nodes):
            if node1.name != node2.name:
                return False

        # Check if the two graphs are equal
        return True

    @property
    def graph(self):
        edges = [(src, dst, key) for src, dst, key, edge_type in self._graph.edges(keys=True, data="type") if edge_type == EdgeType.BODY]
        # TODO: properly induce subgraph for MultiDiGraph
        #   Currently, NetworkX will just return the nodes in the subgraph. 
        #   See: https://stackoverflow.com/questions/16150557/networkxcreating-a-subgraph-induced-from-edges
        return self._graph.edge_subgraph(edges)

    @property
    def graph_symmetry(self):
        edges = [(src, dst, key) for src, dst, key, edge_type in self._graph.edges(keys=True, data="type") if edge_type == EdgeType.SYMMETRY]
        return self._graph.edge_subgraph(edges)

    @staticmethod
    def find_unique_nodes(skeletons: List['Skeleton']):
        """
        Given list of skeletons, return a list of unique node objects across all skeletons.

        Args:
            skeletons: The list of skeletons.

        Returns:
            A list of unique node objects.
        """
        return list({node for skeleton in skeletons for node in skeleton.nodes})

    @staticmethod
    def make_cattr(idx_to_node: Dict[int, Node] = None):
        """
        Create a cattr.Converter() that registers structure and unstructure hooks for
        Skeleton objects that handle serialization of skeletons objects.

        Args:
            idx_to_node: A dict that maps node index to Node objects.

        Returns:
            A cattr.Converter() instance ready for skeleton serialization and deserialization.
        """
        node_to_idx = {node:idx for idx,node in idx_to_node.items()} if idx_to_node is not None else None

        _cattr = cattr.Converter()
        _cattr.register_unstructure_hook(Skeleton, lambda x: Skeleton.to_dict(x, node_to_idx))
        _cattr.register_structure_hook(Skeleton, lambda x,type: Skeleton.from_dict(x, idx_to_node))
        return _cattr

    @property
    def name(self):
        """Get the name of the skeleton.

        Returns:
            A string representing the name of the skeleton.
        """
        return self._graph.name

    @name.setter
    def name(self, name: str):
        """
        A skeleton object cannot change its name. This property is immutable because it is
        used to hash skeletons. If you want to rename a Skeleton you must use the class
        method :code:`rename_skeleton`:

        >>> new_skeleton = Skeleton.rename_skeleton(skeleton=old_skeleton, name="New Name")

        Args:
            name: The name of the Skeleton.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Cannot change Skeleton name, it is immutable since " +
                                  "it is used for hashing. Create a copy of the skeleton " +
                                  "with new name using " +
                                  f"new_skeleton = Skeleton.rename(skeleton, '{name}'))")

    @classmethod
    def rename_skeleton(cls, skeleton: 'Skeleton', name: str) -> 'Skeleton':
        """
        A skeleton object cannot change its name. This property is immutable because it is
        used to hash skeletons. If you want to rename a Skeleton you must use this classmethod.

        >>> new_skeleton = Skeleton.rename_skeleton(skeleton=old_skeleton, name="New Name")

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
    def nodes(self):
        """Get a list of :class:`Node`s.

        Returns:
            A list of :class:`Node`s
        """
        return list(self._graph.nodes)

    @property
    def node_names(self):
        """Get a list of node names.

        Returns:
            A list of node names.
        """
        return [node.name for node in self.nodes]

    @property
    def edges(self):
        """Get a list of edge tuples.

        Returns:
            list of (src_node, dst_node)
        """
        edge_list = [(d['edge_insert_idx'], src, dst)
                     for src, dst, key, d in self._graph.edges(keys=True, data=True)
                     if d['type'] == EdgeType.BODY]

        # We don't want to return the edge list in the order it is stored. We
        # want to use the insertion order. Sort by the insertion index for each
        # edge then drop it from the edge list.
        edge_list = [(src, dst) for _, src, dst in sorted(edge_list)]

        return edge_list

    @property
    def edge_names(self):
        """Get a list of edge name tuples.

        Returns:
            list of (src_node.name, dst_node.name)
        """
        edge_list = [(d['edge_insert_idx'], src.name, dst.name)
                     for src, dst, key, d in self._graph.edges(keys=True, data=True)
                     if d['type'] == EdgeType.BODY]

        # We don't want to return the edge list in the order it is stored. We
        # want to use the insertion order. Sort by the insertion index for each
        # edge then drop it from the edge list.
        edge_list = [(src, dst) for _, src, dst in sorted(edge_list)]

        return [(src.name, dst.name) for src, dst in self.edges]

    @property
    def edges_full(self):
        """Get a list of edge tuples with keys and attributes.

        Returns:
            list of (src_node, dst_node, key, attributes)
        """
        return [(src, dst, key, attr) for src, dst, key, attr in self._graph.edges(keys=True, data=True) if attr["type"] == EdgeType.BODY]

    @property
    def symmetries(self):
        """Get a list of all symmetries without duplicates.

        Returns:
            list of (node1, node2)
        """
        # Find all symmetric edges
        symmetries = [(src, dst) for src, dst, key, edge_type in self._graph.edges(keys=True, data="type") if edge_type == EdgeType.SYMMETRY]
        # Get rid of duplicates
        symmetries = list(set([tuple(set(e)) for e in symmetries]))
        return symmetries

    @property
    def symmetries_full(self):
        """Get a list of all symmetries with keys and attributes.

        Note: The returned list will contain duplicates (node1, node2) and (node2, node1).

        Returns:
            list of (node1, node2, key, attr)
        """
        # Find all symmetric edges
        return [(src, dst, key, attr) for src, dst, key, attr in self._graph.edges(keys=True, data=True) if attr["type"] == EdgeType.SYMMETRY]

    def node_to_index(self, node: Union[str, Node]):
        """
        Return the index of the node, accepts either a node or string name of a Node.

        Args:
            node: The name of the node or the Node object.

        Returns:
            The index of the node in the graph.
        """
        node_list = list(self._graph.nodes)
        try:
            return node_list.index(node)
        except ValueError:
            return node_list.index(self.find_node(node))

    def add_node(self, name: str):
        """Add a node representing an animal part to the skeleton.

        Args:
            name: The name of the node to add to the skeleton. This name must be unique within the skeleton.

        Returns:
            None
        """
        if type(name) is not str:
            raise TypeError("Cannot add nodes to the skeleton that are not str")

        if name in self.node_names:
            raise ValueError("Skeleton already has a node named ({})".format(name))

        self._graph.add_node(Node(name))

    def add_nodes(self, name_list: list):
        """
        Add a list of nodes representing animal parts to the skeleton.

        Args:
            name_list: List of strings representing the nodes.

        Returns:
            None
        """
        for node in name_list:
            self.add_node(node)

    def delete_node(self, name: str):
        """Remove a node from the skeleton.

        The method removes a node from the skeleton and any edge that is connected to it.

        Args:
            name: The name of the edge to remove

        Returns:
            None
        """
        try:
            node = self.find_node(name)
            self._graph.remove_node(node)
        except nx.NetworkXError:
            raise ValueError("The node named ({}) does not exist, cannot remove it.".format(name))

    def find_node(self, name: str):
        """Find node in skeleton by name of node.

        Args:
            name: The name of the :class:`Node` (or a :class:`Node`)

        Returns:
            Node, or None if no match found
        """
        if isinstance(name, Node):
            name = name.name
        nodes = [node for node in self.nodes if node.name == name]
        if len(nodes) == 1:
            return nodes[0]
        elif len(nodes) > 1:
            raise ValueError("Found multiple nodes named ({}).".format(name))
        elif len(nodes) == 0:
            return None

    def add_edge(self, source: str, destination: str):
        """Add an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

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
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if destination_node is None:
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if self._graph.has_edge(source_node, destination_node):
            raise ValueError("Skeleton already has an edge between ({}) and ({}).".format(source, destination))

        self._graph.add_edge(source_node, destination_node, type = EdgeType.BODY,
                             edge_insert_idx = self._graph.graph['num_edges_inserted'])
        self._graph.graph['num_edges_inserted'] = self._graph.graph['num_edges_inserted'] + 1

    def delete_edge(self, source: str, destination: str):
        """Delete an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

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
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if destination_node is None:
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if not self._graph.has_edge(source_node, destination_node):
            raise ValueError("Skeleton has no edge between ({}) and ({}).".format(source, destination))

        self._graph.remove_edge(source_node, destination_node)

    def add_symmetry(self, node1:str, node2: str):
        """Specify that two parts (nodes) in the skeleton are symmetrical.

        Certain parts of an animal body can be related as symmetrical parts in a pair. For example,
        the left and right hands of a person.

        Args:
            node1: The name of the first part in the symmetric pair
            node2: The name of the second part in the symmetric pair

        Returns:
            None

        """
        node1_node, node2_node = self.find_node(node1), self.find_node(node2)

        # We will represent symmetric pairs in the skeleton via additional edges in the _graph
        # These edges will have a special attribute signifying they are not part of the skeleton itself

        if node1 == node2:
            raise ValueError("Cannot add symmetry to the same node.")

        if self.get_symmetry(node1) is not None:
            raise ValueError(f"{node1} is already symmetric with {self.get_symmetry(node1)}.")

        if self.get_symmetry(node2) is not None:
            raise ValueError(f"{node2} is already symmetric with {self.get_symmetry(node2)}.")

        self._graph.add_edge(node1_node, node2_node, type=EdgeType.SYMMETRY)
        self._graph.add_edge(node2_node, node1_node, type=EdgeType.SYMMETRY)

    def delete_symmetry(self, node1:str, node2: str):
        """Deletes a previously established symmetry relationship between two nodes.

        Args:
            node1: The name of the first part in the symmetric pair
            node2: The name of the second part in the symmetric pair

        Returns:
            None
        """
        node1_node, node1_node = self.find_node(node1), self.find_node(node2)

        if self.get_symmetry(node1) != node2 or self.get_symmetry(node2) != node1:
            raise ValueError(f"Nodes {node1}, {node2} are not symmetric.")

        edges = [(src, dst, key) for src, dst, key, edge_type in self._graph.edges([node1_node, node2_node], keys=True, data="type") if edge_type == EdgeType.SYMMETRY]
        self._graph.remove_edges_from(edges)

    def get_symmetry(self, node:str):
        """ Returns the node symmetric with the specified node.

        Args:
            node: The name of the node to query.

        Returns:
            The symmetric :class:`Node`, None if no symmetry
        """
        node_node = self.find_node(node)

        symmetry = [dst for src, dst, edge_type in self._graph.edges(node_node, data="type") if edge_type == EdgeType.SYMMETRY]

        if len(symmetry) == 0:
            return None
        elif len(symmetry) == 1:
            return symmetry[0]
        else:
            raise ValueError(f"{node} has more than one symmetry.")

    def get_symmetry_name(self, node:str):
        """ Returns the name of the node symmetric with the specified node.

        Args:
            node: The name of the node to query.

        Returns:
            name of symmetric node, None if no symmetry
        """
        symmetric_node = self.get_symmetry(node)
        return None if symmetric_node is None else symmetric_node.name

    def __getitem__(self, node_name: str) -> dict:
        """
        Retrieves the node data associated with Skeleton node.

        Args:
            node_name: The name from which to retrieve data.

        Returns:
            A dictionary of data associated with this node.

        """
        node = self.find_node(node_name)
        if node is None:
            raise ValueError(f"Skeleton does not have node named '{node_name}'.")

        return self._graph.nodes.data()[node]

    def __contains__(self, node_name: str) -> bool:
        """
        Checks if specified node exists in skeleton.

        Args:
            node_name: the node name to query

        Returns:
            True if node is in the skeleton.
        """
        return self.has_node(node_name)

    def relabel_node(self, old_name: str, new_name: str):
        """
        Relabel a single node to a new name.

        Args:
            old_name: The old name of the node.
            new_name: The new name of the node.

        Returns:
            None
        """
        self.relabel_nodes({old_name: new_name})

    def relabel_nodes(self, mapping:dict):
        """
        Relabel the nodes of the skeleton.

        Args:
            mapping: A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.

        Returns:
            None
        """
        existing_nodes = self.nodes
        for k, v in mapping.items():
            if self.has_node(v):
                raise ValueError("Cannot relabel a node to an existing name.")
            node = self.find_node(k)
            if node is not None:
                node.name = v

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
            name: The list names of the nodes to check for.

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
        source_node, destination_node = self.find_node(source_name), self.find_node(dest_name) 
        return self._graph.has_edge(source_node, destination_node)

    @staticmethod
    def to_dict(obj: 'Skeleton', node_to_idx: Dict[Node, int] = None):

        # This is a weird hack to serialize the whole _graph into a dict.
        # I use the underlying to_json and parse it.
        return json.loads(obj.to_json(node_to_idx))

    @classmethod
    def from_dict(cls, d: Dict, node_to_idx: Dict[Node, int] = None):
        return Skeleton.from_json(json.dumps(d), node_to_idx)

    def to_json(self, node_to_idx: Dict[Node, int] = None) -> str:
        """
        Convert the skeleton to a JSON representation.

        Args:
            node_to_idx (optional): Map for converting `Node` nodes to int

        Returns:
            A string containing the JSON representation of the Skeleton.
        """
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
        if node_to_idx is not None:
            indexed_node_graph = nx.relabel_nodes(G=self._graph, mapping=node_to_idx) # map nodes to int
        else:
            indexed_node_graph = self._graph

        # Encode to JSON
        json_str = jsonpickle.encode(json_graph.node_link_data(indexed_node_graph))

        return json_str

    def save_json(self, filename: str, node_to_idx: Dict[Node, int] = None):
        """Save the skeleton as JSON file.

           Output the complete skeleton to a file in JSON format.

           Args:
               filename: The filename to save the JSON to.
               node_to_idx (optional): Map for converting `Node` nodes to int

            Returns:
                None
           """

        json_str = self.to_json(node_to_idx)

        with open(filename, 'w') as file:
            file.write(json_str)

    @classmethod
    def from_json(cls, json_str: str, idx_to_node: Dict[int, Node] = None):
        """
        Parse a JSON string containing the Skeleton object and create an instance from it.

        Args:
            json_str: The JSON encoded Skeleton.
            idx_to_node (optional): Map for converting int node in json back to corresponding `Node`.

        Returns:
            An instance of the Skeleton object decoded from the JSON.
        """
        graph = json_graph.node_link_graph(jsonpickle.decode(json_str))

        # Replace graph node indices with corresponding nodes from node_map
        if idx_to_node is not None:
            graph = nx.relabel_nodes(G=graph, mapping=idx_to_node)

        skeleton = Skeleton()
        skeleton._graph = graph

        return skeleton

    @classmethod
    def load_json(cls, filename: str, idx_to_node: Dict[int, Node] = None):
        """Load a skeleton from a JSON file.

        This method will load the Skeleton from JSON file saved with; :meth:`~Skeleton.save_json`

        Args:
            filename: The file that contains the JSON specifying the skeleton.
            idx_to_node (optional): Map for converting int node in json back to corresponding `Node`.

        Returns:
            The Skeleton object stored in the JSON filename.

        """

        with open(filename, 'r') as file:
            skeleton = Skeleton.from_json(file.read(), idx_to_node)

        return skeleton

    @classmethod
    def load_hdf5(cls, file: Union[str, h5.File], name: str):
        """
        Load a specific skeleton (by name) from the HDF5 file.

        Args:
            file: The file name or open h5.File
            name: The name of the skeleton.

        Returns:
            The skeleton instance stored in the HDF5 file.
        """
        if type(file) is str:
            with h5.File(file) as _file:
                skeletons = Skeleton._load_hdf5(_file) # Load all skeletons
        else:
            skeletons = Skeleton._load_hdf5(file)

        return skeletons[name]

    @classmethod
    def load_all_hdf5(cls, file: Union[str, h5.File],
                      return_dict: bool = False) -> Union[List['Skeleton'], Dict[str, 'Skeleton']]:
        """
        Load all skeletons found in the HDF5 file.

        Args:
            file: The file name or open h5.File
            return_dict: True if the the return value should be a dict where the
            keys are skeleton names and values the corresponding skeleton. False
            if the return should just be a list of the skeletons.

        Returns:
            The skeleton instances stored in the HDF5 file. Either in List or Dict form.
        """
        if type(file) is str:
            with h5.File(file) as _file:
                skeletons = Skeleton._load_hdf5(_file) # Load all skeletons
        else:
            skeletons = Skeleton._load_hdf5(file)

        if return_dict:
            return skeletons
        else:
            return list(skeletons.values())

    @classmethod
    def _load_hdf5(cls, file: h5.File):

        skeletons = {}
        for name, json_str in file['skeleton'].attrs.items():
            skeletons[name] = Skeleton.from_json(json_str)

        return skeletons

    def save_hdf5(self, file: Union[str, h5.File]):
        if type(file) is str:
            with h5.File(file) as _file:
                self._save_hdf5(_file)
        else:
            self._save_hdf5(file)

    @classmethod
    def save_all_hdf5(self, file: Union[str, h5.File], skeletons: List['Skeleton']):
        """
        Convenience method to save a list of skeletons to HDF5 file. Skeletons are saved
        as attributes of a /skeleton group in the file.

        Args:
            file: The filename or the open h5.File object.
            skeletons: The list of skeletons to save.

        Returns:
            None
        """

        # Make sure no skeleton has the same name
        unique_names = {s.name for s in skeletons}

        if len(unique_names) != len(skeletons):
            raise ValueError("Cannot save multiple Skeleton's with the same name.")

        for skeleton in skeletons:
            skeleton.save_hdf5(file)

    def _save_hdf5(self, file: h5.File):
        """
        Actual implementation of HDF5 saving.

        Args:
            file: The open h5.File to write the skeleton data too.

        Returns:
            None
        """

        # All skeleton will be put as sub-groups in the skeleton group
        if 'skeleton' not in file:
            all_sk_group = file.create_group('skeleton', track_order=True)
        else:
            all_sk_group = file.require_group('skeleton')

        # Write the dataset to JSON string, then store it in a string
        # attribute
        all_sk_group.attrs[self.name] = np.string_(self.to_json())

    @classmethod
    def load_mat(cls, filename: str):
        """
        Load the skeleton from a Matlab MAT file. This is to support backwards
        compatibility with old LEAP MATLAB code and datasets.

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
        skel_mat["edges"] = skel_mat["edges"] - 1    # convert to 0-based indexing

        node_names = skel_mat['nodeNames']
        node_names = [str(n[0][0]) for n in node_names]
        skeleton.add_nodes(node_names)
        for k in range(len(skel_mat["edges"])):
            edge = skel_mat["edges"][k]
            skeleton.add_edge(source=node_names[edge[0]], destination=node_names[edge[1]])

        return skeleton

    def __str__(self):
        return "%s(name=%r)" % (self.__class__.__name__, self.name)

    def __hash__(self):
        """
        Construct a hash from skeleton id.
        """
        return id(self)

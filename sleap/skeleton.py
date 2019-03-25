"""Implementation of skeleton data structure and API.

This module implements and API for creating animal skeleton's in LEAP. The goal
is to provide a common interface for defining the parts of the animal, their
connection to each other, and needed meta-data.

"""

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


class Skeleton:
    """The main object for representing animal skeletons in LEAP.

    The skeleton represents the constituent parts of the animal whose pose
    is being estimated.

    """

    """
    A index variable used to give skeletons a default name that attemtpts to be
    unique across all skeletons. Will be non-
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
            name = "Skeleton-" + str(self._skeleton_idx)

        self._graph: nx.MultiDiGraph = nx.MultiDiGraph(name=name)

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
    def make_cattr():
        _cattr = cattr.Converter()
        _cattr.register_unstructure_hook(Skeleton, Skeleton.to_dict)
        _cattr.register_structure_hook(Skeleton, lambda x,type: Skeleton.from_dict(x))
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
        """Get a list of node names.

        Returns:
            A list of strings with the node names.
        """
        return list(self._graph.nodes)

    @property
    def edges(self):
        """Get a list of edge tuples.

        Returns:
            list of (src_node, dst_node)
        """
        return [(src, dst) for src, dst, key, edge_type in self._graph.edges(keys=True, data="type") if edge_type == EdgeType.BODY]
    
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

    def node_to_index(self, node_name: str):
        """
        Return the index of the node with name node_name.

        Args:
            node_name: The name of the node.

        Returns:
            The index of the node in the graph.
        """
        return list(self._graph.nodes()).index(node_name)

    def add_node(self, name: str):
        """Add a node representing an animal part to the skeleton.

        Args:
            name: The name of the node to add to the skeleton. This name must be unique within the skeleton.

        Returns:
            None

        """
        if self._graph.has_node(name):
            raise ValueError("Skeleton already has a node named ({})".format(name))

        self._graph.add_node(name)

    def add_nodes(self, name_list: list):
        """
        Add a list of nodes representing animal parts to the skeleton.

        Args:
            name_list: List of strings representing the nodes.

        Returns:
            None
        """
        for name in name_list:
            self.add_node(name)

    def delete_node(self, name: str):
        """Remove a node from the skeleton.

        The method removes a node from the skeleton and any edge that is connected to it.

        Args:
            name: The name of the edge to remove

        Returns:
            None

        """
        try:
            self._graph.remove_node(name)
        except nx.NetworkXError:
            raise ValueError("The node named ({}) does not exist, cannot remove it.".format(name))

    def add_edge(self, source: str, destination: str):
        """Add an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

        Returns:
            None

        """

        if not self._graph.has_node(source):
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if not self._graph.has_node(destination):
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if self._graph.has_edge(source, destination):
            raise ValueError("Skeleton already has an edge between ({}) and ({}).".format(source, destination))

        self._graph.add_edge(source, destination, type = EdgeType.BODY)

    def delete_edge(self, source: str, destination: str):
        """Delete an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

        Returns:
            None
        """
        if not self._graph.has_node(source):
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if not self._graph.has_node(destination):
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if not self._graph.has_edge(source, destination):
            raise ValueError("Skeleton has no edge between ({}) and ({}).".format(source, destination))

        self._graph.remove_edge(source, destination)

    def add_symmetry(self, node1:str, node2:str):
        """Specify that two parts (nodes) in the skeleton are symmetrical.

        Certain parts of an animal body can be related as symmetrical parts in a pair. For example,
        the left and right hands of a person.

        Args:
            node1: The name of the first part in the symmetric pair
            node2: The name of the second part in the symmetric pair

        Returns:
            None

        """

        # We will represent symmetric pairs in the skeleton via additional edges in the _graph
        # These edges will have a special attribute signifying they are not part of the skeleton itself

        if node1 == node2:
            raise ValueError("Cannot add symmetry to the same node.")

        if self.get_symmetry(node1) is not None:
            raise ValueError(f"{node1} is already symmetric with {self.get_symmetry(node1)}.")

        if self.get_symmetry(node2) is not None:
            raise ValueError(f"{node2} is already symmetric with {self.get_symmetry(node2)}.")

        self._graph.add_edge(node1, node2, type=EdgeType.SYMMETRY)
        self._graph.add_edge(node2, node1, type=EdgeType.SYMMETRY)

    def delete_symmetry(self, node1:str, node2: str):
        """Deletes a previously established symmetry relationship between two nodes.

        Args:
            node1: The name of the first part in the symmetric pair
            node2: The name of the second part in the symmetric pair

        Returns:
            None
        """
        if self.get_symmetry(node1) != node2 or self.get_symmetry(node2) != node1:
            raise ValueError(f"Nodes {node1}, {node2} are not symmetric.")

        edges = [(src, dst, key) for src, dst, key, edge_type in self._graph.edges([node1, node2], keys=True, data="type") if edge_type == EdgeType.SYMMETRY]
        self._graph.remove_edges_from(edges)

    def get_symmetry(self, node:str):
        """ Returns the node symmetric with the specified node.

        Args:
            node: The name of the node to query.

        Returns:
            name of symmetric node, None if no symmetry
        """
        symmetry = [dst for src, dst, edge_type in self._graph.edges(node, data="type") if edge_type == EdgeType.SYMMETRY]

        if len(symmetry) == 0:
            return None
        elif len(symmetry) == 1:
            return symmetry[0]
        else:
            raise ValueError(f"{node} has more than one symmetry.")


    def __getitem__(self, node_name:str) -> dict:
        """
        Retrieves a the node data associated with Skeleton node.

        Args:
            node_name: The name from which to retrieve data.

        Returns:
            A dictionary of data associated with this node.

        """
        if not self._graph.has_node(node_name):
            raise ValueError("Skeleton does not have source node named ({})".format(node_name))

        return self._graph.nodes.data()[node_name]

    def relabel_node(self, old_name: str, new_name: str):
        """
        Relable a single node to a new name.

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
            if self._graph.has_node(v):
                raise ValueError("Cannot relabel a node to an existing name.")

        self._graph = nx.relabel_nodes(G=self._graph, mapping=mapping)

    def has_node(self, name: str) -> bool:
        """
        Check whether the skeleton has a node.

        Args:
            name: The name of the node to check for.

        Returns:
            True for yes, False for no.

        """
        return self._graph.has_node(name)

    def has_nodes(self, names: Iterable[str]) -> bool:
        """
        Check whether the skeleton has a list of nodes.

        Args:
            name: The list names of the nodes to check for.

        Returns:
            True for yes, False for no.

        """
        for name in names:
            if not self._graph.has_node(name):
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
        return self._graph.has_edge(source_name, dest_name)

    @staticmethod
    def to_dict(obj: 'Skeleton'):

        # This is a weird hack to serialize the whole _graph into a dict.
        # I use the underlying to_json and parse it.
        return json.loads(obj.to_json())

    @classmethod
    def from_dict(cls, d: Dict):
        return Skeleton.from_json(json.dumps(d))

    def to_json(self) -> str:
        """
        Convert the skeleton to a JSON representation.

        Returns:
            A string containing the JSON representation of the Skeleton.
        """
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)

        # Encode to JSON
        json_str = jsonpickle.encode(json_graph.node_link_data(self._graph))

        return json_str

    def save_json(self, filename: str):
        """Save the skeleton as JSON file.

           Output the complete skeleton to a file in JSON format.

           Args:
               filename: The filename to save the JSON to.

            Returns:
                None

           """

        json_str = self.to_json()

        with open(filename, 'w') as file:
            file.write(json_str)

    @classmethod
    def from_json(cls, json_str: str):
        """
        Parse a JSON string containing the Skeleton object and create an instance from it.

        Args:
            json_str: The JSON encoded Skeleton.

        Returns:
            An instance of the Skeleton object decoded from the JSON.
        """
        graph = json_graph.node_link_graph(jsonpickle.decode(json_str))
        skeleton = Skeleton()
        skeleton._graph = graph

        return skeleton

    @classmethod
    def load_json(cls, filename: str):
        """Load a skeleton from a JSON file.

        This method will load the Skeleton from JSON file saved with; :meth:`~Skeleton.save_json`

        Args:
            filename: The file that contains the JSON specifying the skeleton.

        Returns:
            The Skeleton object stored in the JSON file.

        """

        with open(filename, 'r') as file:
            skeleton = Skeleton.from_json(file.read())

        return skeleton

    @classmethod
    def load_hdf5(cls, file: Union[str, h5.File], name: str):
        """
        Load a specific skeleton (by name) from the HDF5 file.

        Args:
            file: The file name or open h5.File
            name: The name of the skeleton.

        Returns:
            The skeleton intance stored in the HDF5 file.
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
            The skeleton intances stored in the HDF5 file. Either in List or Dict form.
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
            file: The file name or the open h5.File object.
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

    def __str__(self):
        return "%s(name=%r)" % (self.__class__.__name__, self.name)

    def __eq__(self, other: 'Skeleton'):

        # First check names, duh!
        if other.name != self.name:
            return False

        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the graphs are iso-morphic
        is_isomorphic = nx.is_isomorphic(self._graph, other._graph, node_match=dict_match)

        if not is_isomorphic:
            return False

        # Now check that the nodes have the same labels
        for node in self._graph.nodes:
            if node not in other._graph:
                return False

        # FIXME: Skeletons still might not be exactly equal, isomorph with labels swapped.

        # Check if the two graphs are equal
        return True

    def __hash__(self):
        """
        Construct a hash from skeleton name, which we force to be immutable so hashes
        will not change.
        """
        return hash(self.name)


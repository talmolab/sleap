"""Implementation of skeleton data structure and API.

This module implements and API for creating animal skeleton's in LEAP. The goal
is to provide a common interface for defining the parts of the animal, their
connection to each other, and needed meta-data.

"""

import numpy as np
import jsonpickle
import networkx as nx
import h5py as h5

from itertools import count
from typing import Iterable, Union, List, Dict

from networkx.readwrite import json_graph


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

        self.graph = nx.MultiDiGraph(name=name)

    @property
    def name(self):
        """
        Get the name of the skeleton.

        Returns:
            A string representing the name of the skeleton.
        """
        return self.graph.name

    @name.setter
    def name(self, name: str):
        """
        Set the name of the skeleton. Must be a valid string with length greater than 0.

        Args:
            name: The name of the Skeleton.

        Returns:
            None
        """
        if name is None or type(name) is not str or len(name) == 0:
            raise ValueError("A skeleton must have a valid string name.")
        self.graph.name = name

    def add_node(self, name: str):
        """Add a node representing an animal part to the skeleton.

        Args:
            name: The name of the node to add to the skeleton. This name must be unique within the skeleton.

        Returns:
            None

        """
        if self.graph.has_node(name):
            raise ValueError("Skeleton already has a node named ({})".format(name))

        self.graph.add_node(name)

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
            self.graph.remove_node(name)
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

        if not self.graph.has_node(source):
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if not self.graph.has_node(destination):
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if self.graph.has_edge(source, destination):
            raise ValueError("Skeleton already has an edge between ({}) and ({}).".format(source, destination))

        self.graph.add_edge(source, destination)

    def delete_edge(self, source: str, destination: str):
        """Delete an edge between two nodes.

        Args:
            source: The name of the source node.
            destination: The name of the destination node.

        Returns:
            None
        """
        if not self.graph.has_node(source):
            raise ValueError("Skeleton does not have source node named ({})".format(source))

        if not self.graph.has_node(destination):
            raise ValueError("Skeleton does not have destination node named ({})".format(destination))

        if not self.graph.has_edge(source, destination):
            raise ValueError("Skeleton has no edge between ({}) and ({}).".format(source, destination))

        self.graph.remove_edge(source, destination)

    def add_symmetry(self, node1:str, node2:str):
        """Specify that two parts (nodes) in the skeleton are symmetrical.

        Certain parts of an animal body can be related as symmetrical parts in a pair. For example,
        the left and right hands of a person.

        Args:
            node1: The name of the one part in the symmetric pair
            node2:  The name of the secondd part in the symmetric pair

        Returns:
            None

        """

        # We will represent symmetric pairs in the skeleton via additional edges in the graph
        # These edges will have a special attribute signifying they are not part of the skeleton it self
        self.graph.add_edge(node1, node2, symmetry=True)

    def __getitem__(self, node_name:str) -> dict:
        """
        Retrieves a the node data associated with Skeleton node.
        Args:
            node_name: The name from which to retrieve data.

        Returns:
            A dictionary of data associated with this node.

        """
        if not self.graph.has_node(node_name):
            raise ValueError("Skeleton does not have source node named ({})".format(node_name))

        return self.graph.nodes.data()[node_name]

    def relabel_nodes(self, mapping:dict):
        """
        Relabel the nodes of the skeleton.

        Args:
            mapping: A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.

        Returns:
            None

        """
        nx.relabel_nodes(G=self.graph, mapping=mapping, copy=False)

    def has_node(self, name: str) -> bool:
        """
        Check whether the skeleton has a node.

        Args:
            name: The name of the node to check for.

        Returns:
            True for yes, False for no.

        """
        return self.graph.has_node(name)

    def has_nodes(self, names: Iterable[str]) -> bool:
        """
        Check whether the skeleton has a list of nodes.

        Args:
            name: The list names of the nodes to check for.

        Returns:
            True for yes, False for no.

        """
        for name in names:
            if not self.graph.has_node(name):
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
        return self.graph.has_edge(source_name, dest_name)

    def to_json(self) -> str:
        """
        Convert the skeleton to a JSON representation.

        Returns:
            A string containing the JSON representation of the Skeleton.
        """
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)

        # Encode to JSON
        json_str = jsonpickle.encode(json_graph.node_link_data(self.graph))

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
        skeleton.graph = graph

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
        Actual implemetation of HDF5 saving.

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

    def __eq__(self, other: 'Skeleton'):

        # First check names, duh!
        if other.name != self.name:
            return False

        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the graphs are iso-morphic
        is_isomorphic = nx.is_isomorphic(self.graph, other.graph, node_match=dict_match)

        if not is_isomorphic:
            return False

        # Now check that the nodes have the same labels
        for node in self.graph.nodes:
            if node not in other.graph:
                return False

        # FIXME: Skeletons still might not be exactly equal, isomorph with labels swapped.

        # Check if the two graphs are equal
        return True

    def __str__(self):
        return "%s(name=%r)" % (self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(self.graph)


"""Implementation of skeleton data structure and API.

This module implements and API for creating animal skeleton's in LEAP. The goal
is to provide a common interface for defining the parts of the animal, their
connection to each other, and needed meta-data.

"""

import numpy as np
import jsonpickle
import networkx as nx
import h5py as h5

from networkx.readwrite import json_graph


class Skeleton:
    """The main object for representing animal skeletons in LEAP.

    The skeleton represents the constituent parts of the animal whose pose
    is being estimated.

    """
    def __init__(self, name: str = None):
        """Initialize an empty skeleton object.

        Skeleton objects, once they are created can be modified by adding nodes and edges.

        Args:
            name: A name for this skeleton.
        """
        self.graph = nx.MultiDiGraph(name=name)

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
        """Add an edge between two

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

    def save_json(self, filename: str):
        """Save the skeleton as JSON file.

           Output the complete skeleton to a file in JSON format.

           Args:
               filename: The filename to save the JSON to.

            Returns:
                None

           """

        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)

        # Encode to JSON
        json_str = jsonpickle.encode(json_graph.node_link_data(self.graph))

        with open(filename, 'w') as file:
            file.write(json_str)

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
            json_str = file.read()
            graph = json_graph.node_link_graph(jsonpickle.decode(json_str))
            skeleton = Skeleton()
            skeleton.graph = graph

        return skeleton

    @classmethod
    def load_hdf5(cls, h5_group: h5.Group):

        # Check to make sure the datasets exist in group
        if not 'nodes' in h5_group:
            raise ValueError("Could not find nodes dataset in skeleton group.")

        if not 'edges' in h5_group:
            raise ValueError("Could not find edges dataset in skeleton group.")

        # Lets first grab the attributes of the group, these should contain
        # the names of the nodes.
        node_names = h5_group.attrs.get("nodeNames", default=None)
        if node_names is None:
            raise ValueError("Couldn't not find nodeNames attribute in skeleton HDF5 group.")

        # Decode the byte string and split it by end lines, this is how the nodes names are stored
        node_names = node_names.decode('utf-8').split('\n')

        # Get the number of nodes
        num_nodes = np.asscalar(h5_group["nodes"][:])

        # Get the edges
        edges = h5_group["edges"][:].astype('int32')

        # Make sure number nodes is equal to the length of node_names
        if len(node_names) != num_nodes:
            raise ValueError("Length of skeleton nodeNames attribute does not equal number of nodes in nodes dataset.")

        # Perform some checks on the edge list


        # Lets make the skeleton object now
        skeleton = Skeleton()

        # Add the nodes
        for node in node_names:
            skeleton.add_node(name=node)

        # Add the edges
        for i in range(edges.shape[1]):
            skeleton.add_edge(source=node_names[edges[0,i]-1], destination=node_names[edges[1,i]-1])

        return skeleton


    def __eq__(self, other):

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


"""Implementation of skeleton data structure and API.

This module implements and API for creating animal skeleton's in LEAP. The goal
is to provide a common interface for defining the parts of the animal, their
connection to each other, and needed meta-data.

"""

import jsonpickle
import networkx as nx
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

    def __eq__(self, other):

        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the two graphs are equal
        return nx.is_isomorphic(self.graph, other.graph, node_match=dict_match)

    def __str__(self):
        return "%s(name=%r)" % (self.__class__.__name__, self.name)


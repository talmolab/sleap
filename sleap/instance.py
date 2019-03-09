"""

"""

import math

from typing import Dict


from sleap.skeleton import Skeleton
from sleap.io.video import Video



# This can probably be a namedtuple but has been made a full class just in case
# we need more complicated functionality later.
class Point:

    def __init__(self, x:float = math.nan, y:float = math.nan, visible:bool = True):
        """
        A very simple class to define a labelled point and any metadata associated with it.

        Args:
            x: The horizontal pixel location of the point within the image frame.
            y: The vertical pixel location of the point within the image frame.
            visible: Whether point is visible in the labelled image or not.
        """
        self.x = x
        self.y = y
        self.visible = visible

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.visible == other.visible

class Instance:

    def __init__(self, skeleton:Skeleton, video:Video, frame_idx:int, points: Dict[str, Point] = {}):
        """
        The class :class:`Instance` represents a labelled instance of skeleton on
        a particular frame of a particular video.

        Args:
            skeleton: The skeleton that this instance is associated with.
            video: The videos that the instance appears.
            frame_idx: The frame number of the video.
            points: A dictionary where keys are skeleton node names and values are _points.
        """
        self.skeleton = skeleton
        self.video = video
        self.frame_idx = frame_idx

        # Create a data structure to store a list of labelled _points for each node of this
        # skeleton.
        self._points = points

        self._validate_all_points()

    def _validate_all_points(self):
        """
        Function that makes sure all the _points defined for the skeleton are found in the skeleton.

        Returns:
            None

        Raises:
            ValueError: If a point is associated with a skeleton node name that doesn't exist.
        """
        for node_name in self._points.keys():
            if not self.skeleton.has_node(node_name):
                raise KeyError(f"There is no skeleton node named {node_name} in {self.skeleton}")

    def __getitem__(self, node):
        """
        Get the _points associated with particular skeleton node or list of skeleton nodes

        Args:
            node: A single node or list of nodes within the skeleton associated with this instance.

        Returns:
            A single point of list of _points related to the nodes provided as argument.

        """

        # If the node is a list of nodes, use get item recursively and return a list of _points.
        if type(node) is list:
            ret_list = []
            for n in node:
                ret_list.append(self.__getitem__(n))

            return ret_list

        if self.skeleton.has_node(node):
            if not node in self._points:
                self._points[node] = Point()

            return self._points[node]
        else:
            raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node named '{node}'")

    def __setitem__(self, node, value):

        # Make sure node and value, if either are lists, are of compatible size
        if type(node) is not list and type(value) is list and len(value) != 1:
            raise IndexError("Node list for indexing must be same length and value list.")

        if type(node) is list and type(value) is not list and len(node) != 1:
            raise IndexError("Node list for indexing must be same length and value list.")

        # If we are dealing with lists, do multiple assignment recursively, this should be ok because
        # skeletons and instances are small.
        if type(node) is list:
            for n, v in zip(node, value):
                self.__setitem__(n, v)
        else:
            if self.skeleton.has_node(node):
                self._points[node] = value
            else:
                raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node named '{node}'")

    def nodes(self):
        """
        Get the list of nodes that have been labelled for this instance.

        Returns:
            A list of nodes that have been labelled for this instance.

        """
        return self._points.keys()

    def nodes_points(self):
        """
        Return view object that displays a list of the instance's (node, point) tuple pairs
        for all labelled point.

        Returns:
            The instance's (node, point) tuple pairs for all labelled point.
        """
        return self._points.items()

    def points(self):
        """
        Return the list of labelled points, in order they were labelled.

        Returns:
            The list of labelled points, in order they were labelled.
        """
        return self._points.values()


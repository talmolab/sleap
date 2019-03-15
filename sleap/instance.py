"""

"""

import math
import shelve

import numpy as np
import h5py as h5
import pandas as pd

from typing import Dict, List, Union

from sleap.skeleton import Skeleton
from sleap.io.video import Video

import attr


# This can probably be a namedtuple but has been made a full class just in case
# we need more complicated functionality later.
@attr.s(auto_attribs=True)
class Point:
    """
    A very simple class to define a labelled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of the point within the image frame.
        y: The vertical pixel location of the point within the image frame.
        visible: Whether point is visible in the labelled image or not.
    """

    x: float = math.nan
    y: float = math.nan
    visible: bool = True

class Instance:

    def __init__(self, skeleton:Skeleton, video:Video, frame_idx:int, points: Dict[str, Point] = None):
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
        if points is None:
            self._points = {}
        else:
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

    @classmethod
    def to_pandas_df(cls, instances: Union['Instance', List['Instance']], skip_nan:bool = True) -> pd.DataFrame:
        """
        Given an instance or list of instances, generate a pandas DataFrame that contains
        all of the data in normalized form.
        Args:
            instances: A single instance or list of instances.
            skip_nan: Whether to drop points that have NaN values for x or y.

        Returns:
            A pandas DataFrame that contains all of the isntance's points level data
            in and normalized form. The columns of the DataFrame are:

            * id - A unique number for each row of the table.
            * instanceId - a unique id for each unique instance.
            * skeleton - the name of the skeleton that this point is a part of.
            * node - A string specifying the name of the skeleton node that this point value corresponds.
            * videoId - A string specifying the video that this instance is in.
            * frameIdx - The frame number of the video that this instance occurs on.
            * visible - Whether the point in this row for this instance is visible.
            * x - The horizontal pixel position of this node for this instance.
            * y - The vertical pixel position of this node for this instance.
        """

        # If this is a single instance, make it a list
        if type(instances) is Instance:
            instances = [instances]

        instance_ids = []
        nodes = []
        frames = []
        xs = []
        ys = []
        videos = []
        skeletons = []
        visibles = []

        # Extract all the data from each instance and its points
        for instance_id, instance in enumerate(instances):
            for (node, point) in instance.nodes_points():

                # Skip any NaN points if the user has asked for it.
                if skip_nan and (math.isnan(point.x) or math.isnan(point.y)):
                    continue

                instance_ids.append(instance_id)
                nodes.append(node)
                frames.append(instance.frame_idx)
                xs.append(point.x)
                ys.append(point.y)
                visibles.append(point.visible)
                videos.append(instance.video)
                skeletons.append(instance.skeleton)

        # Construct a pandas data frame from this list of instances
        df = pd.DataFrame.from_dict({
            'id': [i for i in range(len(instance_ids))],
            'instanceId': instance_ids,
            'skeleton': [s.name for s in skeletons],
            'node': nodes,
            'videoId': [str(video) for video in videos ],
            'frameIdx': frames,
            'visible': visibles,
            'x': xs,
            'y': ys
        })

        return df

    @classmethod
    def to_hdf5(cls, instances: Union['Instance',  List['Instance']],
                file: Union[str, h5.File], group: Union[str, h5.Group],
                skip_nan: bool = True):
        """
        Write the instance point level data to an HDF5 file and group. This
        function writes the data to an HDF5 group not a dataset. Each
        column of the data is a dataset. The datasets within the group
        will be all the same length (the total number of points across all
        instances). They are as follows:

            * id - A unique number for each row of the table.
            * instanceId - a unique id for each unique instance.
            * skeleton - the name of the skeleton that this point is a part of.
            * node - A string specifying the name of the skeleton node that this point value corresponds.
            * videoId - A string specifying the video that this instance is in.
            * frameIdx - The frame number of the video that this instance occurs on.
            * visible - Whether the point in this row for this instance is visible.
            * x - The horizontal pixel position of this node for this instance.
            * y - The vertical pixel position of this node for this instance.

        Args:
            instances: A single instance or list of instances.
            skip_nan: Whether to drop points that have NaN values for x or y.
            file:
            group:

        Returns:
            None
        """

        # First, lets get the instance data as a pandas data frame.
        df = cls.to_pandas_df(instances=instances, skip_nan=skip_nan)

        # Are we dealing with a string or an open h5.File object
        _file = file if isinstance(file, h5.File) else h5.File(file, mode="a")

        try:

            # If the group doesn't exists, create it, but do so with track order.
            # If it does exists, leave it be.
            if type(group) is str and group not in _file:
                _group = _file.create_group(group, track_order=True)
            elif type(group) is str:
                _group = _file[group]
            elif type(group) is h5.Group:
                _group = group

            # Right each column as a data frame.
            for col in df:
                vals = df[col].values
                if col in _group:
                    del _group[col]

                # If the column are objects (should be strings), convert to dtype=S, strings as per
                # h5py recommendations.
                if vals.dtype == np.dtype('O'):
                    dataset = _group.create_dataset(name=col, shape=vals.shape,
                                                    data=vals.astype(np.dtype('S')),
                                                    compression="gzip")
                else:
                    dataset = _group.create_dataset(name=col, shape=vals.shape,
                                                    data=vals, compression="gzip")

        except Exception as ex:

            # If the user passed a string, close things down, otherwise leave them open,
            # that is their job. Hopefully, they are in a context manager.
            if type(file) is str:
                _file.close()

            # Re-raise
            raise ex

        # If the user passed a string, close things down, otherwise leave them open,
        # that is their job.
        if type(file) is str:
            _file.close()

    # @classmethod
    # def load_instances_hdf5_group(cls, h5_group: h5.Group, skeleton: Skeleton) -> List['Instance']:
    #
    #     # Get the datasets
    #     x = h5_group['x']
    #     y = h5_group['y']
    #     frames = h5_group['frameIdx']
    #     visible = h5_group['visible']
    #     video = h5_group['videoId']
    #     node = h5_group['node']
    #     instance_id = h5_group['instanceId']


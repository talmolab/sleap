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
from sleap.util import attr_to_dtype

import attr

# This can probably be a namedtuple but has been made a full class just in case
# we need more complicated functionality later.
@attr.s(auto_attribs=True, slots=True)
class Point:
    """
    A very simple class to define a labelled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of the point within the image frame.
        y: The vertical pixel location of the point within the image frame.
        visible: Whether point is visible in the labelled image or not.
    """

    x: float = attr.ib(default=math.nan, converter=float)
    y: float = attr.ib(default=math.nan, converter=float)
    visible: bool = True

    def __str__(self):
        return f"({self.x}, {self.y})"

    @classmethod
    def dtype(cls):
        """
        Get the compound numpy dtype of a point. This is very important for
        serialization.

        Returns:
            The compound numpy dtype of the point
        """
        return attr_to_dtype(cls)

    def isnan(self):
        return math.isnan(self.x) or math.isnan(self.y)

@attr.s(auto_attribs=True, slots=True)
class Instance:
    """
    The class :class:`Instance` represents a labelled instance of skeleton

    Args:
        skeleton: The skeleton that this instance is associated with.
        points: A dictionary where keys are skeleton node names and values are _points.
    """

    skeleton: Skeleton
    _points: Dict[str, Point] = attr.ib(default=attr.Factory(dict))

    @_points.validator
    def _validate_all_points(self, attribute, points):
        """
        Function that makes sure all the _points defined for the skeleton are found in the skeleton.

        Returns:
            None

        Raises:
            ValueError: If a point is associated with a skeleton node name that doesn't exist.
        """
        for node_name in points.keys():
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

        # Lets construct a list of dicts which will be records for the pandas data frame
        records = []

        # Extract all the data from each instance and its points
        id = 0
        for instance_id, instance in enumerate(instances):

            # Get all the attributes from the instance except the points dict
            irecord = {'id': id, 'instance_id': instance_id,
                       **attr.asdict(instance, filter=lambda attr, value: attr.name != "_points")}

            # Convert the skeleton to it's name
            irecord['skeleton'] = irecord['skeleton'].name

            # FIXME: Do the same for the video

            for (node, point) in instance.nodes_points():

                # Skip any NaN points if the user has asked for it.
                if skip_nan and (math.isnan(point.x) or math.isnan(point.y)):
                    continue

                precord = {'node': node, **attr.asdict(point)}

                records.append({**irecord, **precord})

        id = id + 1

        # Construct a pandas data frame from this list of instances
        if len(records) == 1:
            df = pd.DataFrame.from_records(records, index=[0])
        else:
            df = pd.DataFrame.from_records(records)

        return df

    @classmethod
    def save_hdf5(cls, file: Union[str, h5.File],
                  instances: Union['Instance', List['Instance']],
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
            file: The HDF5 file to save the instance data to.
            instances: A single instance or list of instances.
            skip_nan: Whether to drop points that have NaN values for x or y.

        Returns:
            None
        """

        # Make it into a list of length one if needed.
        if type(instances) is Instance:
            instances = [instances]

        if type(file) is str:
            with h5.File(file) as _file:
                Instance._save_hdf5(file=_file, instances=instances, skip_nan=skip_nan)
        else:
            Instance._save_hdf5(file=file, instances=instances, skip_nan=skip_nan)

    @classmethod
    def _save_hdf5(cls, file: h5.File, instances: List['Instance'], skip_nan: bool = True):

        # Get all the unique skeleton objects in this list of instances
        # This is a set comprehension, slick python, but unreadable
        skeletons = {i.skeleton for i in instances}

        # First, lets save the skeletons to the file
        Skeleton.save_all_hdf5(file=file, skeletons=list(skeletons))

        # Second, lets get the instance data as a pandas data frame.
        df = cls.to_pandas_df(instances=instances, skip_nan=skip_nan)

        # If the group doesn't exists, create it, but do so with track order.
        # If it does exists, leave it be.
        if 'points' not in file:
            group = file.create_group('points', track_order=True)
        else:
            group = file['points']

        # Write each column as a data frame.
        for col in df:
            vals = df[col].values
            if col in group:
                del group[col]

            # If the column are objects (should be strings), convert to dtype=S, strings as per
            # h5py recommendations.
            if vals.dtype == np.dtype('O'):
                dtype = h5.special_dtype(vlen=str)
                group.create_dataset(name=col, shape=vals.shape,
                                     data=vals,
                                     dtype=dtype,
                                     compression="gzip")
            else:
                group.create_dataset(name=col, shape=vals.shape,
                                     data=vals, compression="gzip")

    @classmethod
    def load_hdf5(cls, file: Union[h5.File, str]) -> List['Instance']:
        """
        Load instance data from an HDF5 dataset.

        Args:
            file: The name of the HDF5 file or the open h5.File object.

        Returns:
            A list of Instance objects.
        """

        if type(file) is str:
            with h5.File(file) as _file:
                return Instance._load_hdf5(_file)
        else:
            return Instance._load_hdf5(file)

    @classmethod
    def _load_hdf5(self, file: h5.File):

        # First, get all the skeletons in the HDF5 file
        skeletons = Skeleton.load_all_hdf5(file=file, return_dict=True)

        if 'points' not in file:
            raise ValueError("No instance data found in dataset.")

        group = file['points']

        # Next get a dict that contains all the datasets for the instance
        # data.
        records = {}
        for key, dset in group.items():
            records[key] = dset[...]

        # Convert to a data frame.
        df = pd.DataFrame.from_dict(records)

        # Lets first create all the points, start by grabbing the Point columns, grab only
        # columns that exist. This is just in case we are reading an older form of the dataset
        # format and the fields don't line up.
        point_cols = [f.name for f in attr.fields(Point)]
        point_cols = list(filter(lambda x: x in group, point_cols))

        # Extract the points columns and convert dicts of keys and values.
        points = df[[*point_cols]].to_dict('records')

        # Convert to points dicts to points objects
        points = [Point(**args) for args in points]

        # Instance columns
        instance_cols = [f.name for f in attr.fields(Instance)]
        instance_cols = list(filter(lambda x: x in group, instance_cols))

        instance_records = df[[*instance_cols]].to_dict('records')

        # Convert skeletons references to skeleton objects
        for r in instance_records:
            r['skeleton'] = skeletons[r['skeleton']]

        instances: List[Instance] = []
        curr_id = -1 # Start with an invalid instance id so condition is tripped
        for idx, r in enumerate(instance_records):
            if curr_id == -1 or curr_id != df['instance_id'].values[idx]:
                curr_id = df['instance_id'].values[idx]
                curr_instance = Instance(**r)
                instances.append(curr_instance)

            # Add the point the instance
            curr_instance[df['node'].values[idx]] = points[idx]

        return instances

    def drop_nan_points(self):
        """
        Drop any points for the instance that are not completely specified.

        Returns:
            None
        """
        is_nan = []
        for n, p in self._points.items():
            if p.isnan():
                is_nan.append(n)

        # Remove them
        for n in is_nan:
            self._points.pop(n, None)

    @classmethod
    def drop_all_nan_points(cls, instances: List['Instance']):
        """
        Call drop_nan_points on a list of Instances.

        Args:
            instances: The list of instances to call drop_nan_points() on.

        Returns:
            None
        """
        for i in instances:
            i.drop_nan_points()

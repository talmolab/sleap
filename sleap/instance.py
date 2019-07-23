"""

"""

import math

import numpy as np
import h5py as h5
import pandas as pd

from typing import Dict, List, Optional, Union, Tuple

from attr import __init__

from sleap.skeleton import Skeleton, Node
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
    complete: bool = False

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
class PredictedPoint(Point):
    """
    A predicted point is an output of the inference procedure. It has all
    the properties of a labeled point with an accompanying score.

    Args:
        score: The point level prediction score.
    """
    score: float = attr.ib(default=0.0, converter=float)

    @classmethod
    def from_point(cls, point: Point, score: float = 0.0):
        """
        Create a PredictedPoint from a Point

        Args:
            point: The point to copy all data from.
            score: The score for this predicted point.

        Returns:
            A scored point based on the point passed in.
        """
        return cls(**{**attr.asdict(point), 'score': score})


@attr.s(slots=True, cmp=False)
class Track:
    """
    A track object is associated with a set of animal/object instances across multiple
    frames of video. This allows tracking of unique entities in the video over time and
    space.

    Args:
        spawned_on: The frame of the video that this track was spawned on.
        name: A name given to this track for identifying purposes.
    """
    spawned_on: int = attr.ib(converter=int)
    name: str = attr.ib(default="", converter=str)

    def matches(self, other: 'Track'):
        """
        Check if two tracks match by value.

        Args:
            other: The other track to check

        Returns:
            True if they match, False otherwise.
        """
        return attr.asdict(self) == attr.asdict(other)


# NOTE:
# Instance cannot be a slotted class at the moment. This is because it creates
# attributes _frame and _point_array_cache after init. These are private variables
# that are created in post init so they are not serialized.

@attr.s(auto_attribs=True)
class Instance:
    """
    The class :class:`Instance` represents a labelled instance of skeleton

    Args:
        skeleton: The skeleton that this instance is associated with.
        points: A dictionary where keys are skeleton node names and values are _points.
        track: An optional multi-frame object track associated with this instance.
            This allows individual animals/objects to be tracked across frames.
        from_predicted: The predicted instance (if any) that this was copied from.
    """

    skeleton: Skeleton
    track: Track = attr.ib(default=None)
    from_predicted: Optional['PredictedInstance'] = attr.ib(default=None)
    _points: Dict[Node, Union[Point, PredictedPoint]] = attr.ib(default=attr.Factory(dict))

    @from_predicted.validator
    def _validate_from_predicted_(self, attribute, from_predicted):
        if from_predicted is not None and type(from_predicted) != PredictedInstance:
            raise TypeError(f"Instance.from_predicted type must be PredictedInstance (not {type(from_predicted)})")

    @_points.validator
    def _validate_all_points(self, attribute, points):
        """
        Function that makes sure all the _points defined for the skeleton are found in the skeleton.

        Returns:
            None

        Raises:
            ValueError: If a point is associated with a skeleton node name that doesn't exist.
        """
        is_string_dict = set(map(type, self._points)) == {str}
        if is_string_dict:
            for node_name in points.keys():
                if not self.skeleton.has_node(node_name):
                    raise KeyError(f"There is no node named {node_name} in {self.skeleton}")

    def __attrs_post_init__(self):

        # If the points dict is non-empty, validate it.
        if self._points:
            # Check if the dict contains all strings
            is_string_dict = set(map(type, self._points)) == {str}

            # Check if the dict contains all Node objects
            is_node_dict = set(map(type, self._points)) == {Node}

            # If the user fed in a dict whose keys are strings, these are node names,
            # convert to node indices so we don't break references to skeleton nodes
            # if the node name is relabeled.
            if self._points and is_string_dict:
                self._points = {self.skeleton.find_node(name): point for name,point in self._points.items()}

            if not is_string_dict and not is_node_dict:
                raise ValueError("points dictionary must be keyed by either strings " +
                                 "(node names) or Nodes.")

        # Create here in post init so it is not serialized by cattrs, wish there was better way
        # This field keeps track of any labeled frame that this Instance has been associated with.
        self.frame: Union[LabeledFrame, None] = None

        # A variable used to cache a constructed points_array call to save on time. This is only
        # used when cached=True is passed to points_array
        self._points_array_cache: Union[np.array, None] = None

    def _node_to_index(self, node_name):
        """
        Helper method to get the index of a node from its name.

        Args:
            node_name: The name of the node.

        Returns:
            The index of the node on skeleton graph.
        """
        return self.skeleton.node_to_index(node_name)

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

        if isinstance(node, str):
            node = self.skeleton.find_node(node)
        if node in self.skeleton.nodes:
            if not node in self._points:
                self._points[node] = Point()

            return self._points[node]
        else:
            raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node '{node}'")

    def __contains__(self, node):
        """
        Returns True if this instance has a point with the specified node.

        Args:
            node: node name

        Returns:
            bool: True if the point with the node name specified has a point in this instance.
        """
        return node in self._points

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
            if isinstance(node,str):
                node = self.skeleton.find_node(node)

            if node in self.skeleton.nodes:
                self._points[node] = value
            else:
                raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node '{node}'")

    def __delitem__(self, node):
        """ Delete node key and points associated with that node. """
        # TODO: handle this case somehow?
        pass

    def matches(self, other):
        """
        Compare this `Instance` to another, modulo the particular `Node` objects.

        Args:
            other: The other instance.

        Returns:
            True if match, False otherwise.
        """
        if list(self.points()) != list(other.points()):
            return False

        if not self.skeleton.matches(other.skeleton):
            return False

        if self.track and other.track and not self.track.matches(other.track):
            return False

        if self.track and not other.track or not self.track and other.track:
            return False

        # Make sure the frame indices match
        if not self.frame_idx == other.frame_idx:
            return False

        return True

    @property
    def nodes(self):
        """
        Get the list of nodes that have been labelled for this instance.

        Returns:
            A list of nodes that have been labelled for this instance.

        """
        return tuple(self._points.keys())

    @property
    def nodes_points(self):
        """
        Return view object that displays a list of the instance's (node, point) tuple pairs
        for all labelled point.

        Returns:
            The instance's (node, point) tuple pairs for all labelled point.
        """
        names_to_points = {node: point for node, point in self._points.items()}
        return names_to_points.items()

    def points(self) -> Tuple:
        """
        Return the list of labelled points, in order they were labelled.

        Returns:
            The list of labelled points, in order they were labelled.
        """
        return tuple(self._points.values())

    def points_array(self, cached=False, invisible_as_nan=False) -> np.ndarray:
        """
        Return the instance's points in array form.

        Args:
            cached: If True, use a cached version of the points data if available,
            create cache if it doesn't exist. If False, recompute and cache each
            call.
        Returns:
            A Nx2 array containing x and y coordinates of each point
            as the rows of the array and N is the number of nodes in the skeleton.
            The order of the rows corresponds to the ordering of the skeleton nodes.
            Any skeleton node not defined will have NaNs present.
        """
        if not cached or (cached and self._points_array_cache is None):
            pts = np.ndarray((len(self.skeleton.nodes), 2))
            for i, n in enumerate(self.skeleton.nodes):
                p = self._points.get(n, Point())
                pts[i, 0] = p.x if p.visible or not invisible_as_nan else np.nan
                pts[i, 1] = p.y if p.visible or not invisible_as_nan else np.nan

            self._points_array_cache = pts

        return self._points_array_cache

    @property
    def centroid(self) -> np.ndarray:
        """Returns instance centroid as (x,y) numpy row vector."""
        points = self.points_array(invisible_as_nan=True)
        centroid = np.nanmedian(points, axis=0)
        return centroid

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

            # Get all the attributes from the instance except the points dict or from_predicted
            irecord = {'id': id, 'instance_id': instance_id,
                       **attr.asdict(instance, filter=lambda attr, value: attr.name not in ("_points", "from_predicted"))}

            # Convert the skeleton to it's name
            irecord['skeleton'] = irecord['skeleton'].name

            # FIXME: Do the same for the video

            for (node, point) in instance.nodes_points:

                # Skip any NaN points if the user has asked for it.
                if skip_nan and (math.isnan(point.x) or math.isnan(point.y)):
                    continue

                precord = {'node': node.name, **attr.asdict(point)} # FIXME: save other node attributes?

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

    @property
    def frame_idx(self) -> Union[None, int]:
        """
        Get the index of the frame that this instance was found on. This is a convenience
        method for Instance.frame.frame_idx.

        Returns:
            The frame number this instance was found on.
        """
        if self.frame is None:
            return None
        else:
            return self.frame.frame_idx


@attr.s(auto_attribs=True)
class PredictedInstance(Instance):
    """
    A predicted instance is an output of the inference procedure. It is
    the main output of the inference procedure.

    Args:
        score: The instance level prediction score.
    """
    score: float = attr.ib(default=0.0, converter=float)

    def __attrs_post_init__(self):
        super(PredictedInstance, self).__attrs_post_init__()
        if self.from_predicted is not None:
            raise ValueError("PredictedInstance should not have from_predicted.")

    @classmethod
    def from_instance(cls, instance: Instance, score):
        """
        Create a PredictedInstance from and Instance object. The fields are
        copied in a shallow manner with the exception of points. For each
        point in the instance an PredictedPoint is created with score set
        to default value.

        Args:
            instance: The Instance object to shallow copy data from.
            score: The score for this instance.

        Returns:
            A PredictedInstance for the given Instance.
        """
        kw_args = attr.asdict(instance, recurse=False)
        kw_args['points'] = {key: PredictedPoint.from_point(val)
                             for key, val in kw_args['_points'].items()}
        del kw_args['_points']
        kw_args['score'] = score
        return cls(**kw_args)


@attr.s(auto_attribs=True)
class LabeledFrame:
    video: Video = attr.ib()
    frame_idx: int = attr.ib(converter=int)
    _instances: Union[List[Instance], List[PredictedInstance]] = attr.ib(default=attr.Factory(list))

    def __attrs_post_init__(self):

        # Make sure all instances have a reference to this frame
        for instance in self.instances:
            instance.frame = self

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances.__getitem__(index)

    def index(self, value: Instance):
        return self.instances.index(value)

    def __delitem__(self, index):
        value = self.instances.__getitem__(index)

        self.instances.__delitem__(index)

        # Modify the instance to remove reference to this frame
        value.frame = None

    def insert(self, index, value: Instance):
        self.instances.insert(index, value)

        # Modify the instance to have a reference back to this frame
        value.frame = self

    def __setitem__(self, index, value: Instance):
        self.instances.__setitem__(index, value)

        # Modify the instance to have a reference back to this frame
        value.frame = self

    @property
    def instances(self):
        """
        A list of instances to associated with this frame.

        Returns:
            A list of instances to associated with this frame.
        """
        return self._instances

    @instances.setter
    def instances(self, instances: List[Instance]):
        """
        Set the list of instances assigned to this frame. Note: whenever an instance
        is associated with a LabeledFrame that Instance objects frame property will
        be overwritten to the LabeledFrame.

        Args:
            instances: A list of instances to associated with this frame.

        Returns:
            None
        """

        # Make sure to set the frame for each instance to this LabeledFrame
        for instance in instances:
            instance.frame = self

        self._instances = instances

    @property
    def user_instances(self):
        return [inst for inst in self._instances if type(inst) == Instance]

    @property
    def has_user_instances(self):
        return (len(self.user_instances) > 0)

    @property
    def unused_predictions(self):
        unused_predictions = []
        any_tracks = [inst.track for inst in self._instances if inst.track is not None]
        if len(any_tracks):
            # use tracks to determine which predicted instances have been used
            used_tracks = [inst.track for inst in self._instances
                           if type(inst) == Instance and inst.track is not None
                           ]
            unused_predictions = [inst for inst in self._instances
                                  if inst.track not in used_tracks
                                  and type(inst) == PredictedInstance
                                  ]

        else:
            # use from_predicted to determine which predicted instances have been used
            # TODO: should we always do this instead of using tracks?
            used_instances = [inst.from_predicted for inst in self._instances
                              if inst.from_predicted is not None]
            unused_predictions = [inst for inst in self._instances
                                  if type(inst) == PredictedInstance
                                  and inst not in used_instances]

        return unused_predictions

    @property
    def instances_to_show(self):
        """
        Return a list of instances associated with this frame, but excluding any
        predicted instances for which there's a corresponding regular instance.
        """
        unused_predictions = self.unused_predictions
        inst_to_show = [inst for inst in self._instances
                        if type(inst) == Instance or inst in unused_predictions]
        inst_to_show.sort(key=lambda inst: inst.track.spawned_on if inst.track is not None else math.inf)
        return inst_to_show

    @staticmethod
    def merge_frames(labeled_frames, video):
        frames_found = dict()
        # move instances into first frame with matching frame_idx
        for idx, lf in enumerate(labeled_frames):
            if lf.video == video:
                if lf.frame_idx in frames_found.keys():
                    # move instances
                    dst_idx = frames_found[lf.frame_idx]
                    labeled_frames[dst_idx].instances.extend(lf.instances)
                    lf.instances = []
                else:
                    # note first lf with this frame_idx
                    frames_found[lf.frame_idx] = idx
        # remove labeled frames with no instances
        labeled_frames = list(filter(lambda lf: len(lf.instances),
                                     labeled_frames))
        return labeled_frames
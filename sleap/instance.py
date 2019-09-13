"""

"""

import math

import numpy as np
import h5py as h5
import pandas as pd
import cattr

from typing import Dict, List, Optional, Union, Tuple

from numpy.lib.recfunctions import structured_to_unstructured

from sleap.skeleton import Skeleton, Node
from sleap.io.video import Video

import attr

try:
    from typing import ForwardRef
except:
    from typing import _ForwardRef as ForwardRef


class Point(np.record):
    """
    A very simple class to define a labelled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of the point within the image frame.
        y: The vertical pixel location of the point within the image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the a user labeler.
    """

    # Define the dtype from the point class attributes plus some
    # additional fields we will use to relate point to instances and
    # nodes.
    dtype = np.dtype(
        [('x', 'f8'),
         ('y', 'f8'),
         ('visible', '?'),
         ('complete', '?')])

    def __new__(cls, x: float = math.nan, y: float = math.nan,
                visible: bool = True, complete: bool = False):

        # HACK: This is a crazy way to instantiate at new Point but I can't figure
        # out how recarray does it. So I just use it to make matrix of size 1 and
        # index in to get the np.record/Point
        # All of this is a giant hack so that Point(x=2,y=3) works like expected.
        val = PointArray(1)
        val[0] = (x, y, visible, complete)
        val = val[0]

        # val.x = x
        # val.y = y
        # val.visible = visible
        # val.complete = complete

        return val

    def __str__(self):
        return f"({self.x}, {self.y})"

    def isnan(self):
        """
        Are either of the coordinates a NaN value.

        Returns:
            True if x or y is NaN, False otherwise.
        """
        return math.isnan(self.x) or math.isnan(self.y)


# This turns PredictedPoint into an attrs class. Defines comparators for
# us and generaly makes it behave better. Crazy that this works!
Point = attr.s(these={name: attr.ib()
                               for name in Point.dtype.names},
                        init=False)(Point)


class PredictedPoint(Point):
    """
    A predicted point is an output of the inference procedure. It has all
    the properties of a labeled point with an accompanying score.

    Args:
        x: The horizontal pixel location of the point within the image frame.
        y: The vertical pixel location of the point within the image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the a user labeler.
        score: The point level prediction score.
    """

    # Define the dtype from the point class attributes plus some
    # additional fields we will use to relate point to instances and
    # nodes.
    dtype = np.dtype(
        [('x', 'f8'),
         ('y', 'f8'),
         ('visible', '?'),
         ('complete', '?'),
         ('score', 'f8')])

    def __new__(cls, x: float = math.nan, y: float = math.nan,
                visible: bool = True, complete: bool = False,
                score: float = 0.0):

        # HACK: This is a crazy way to instantiate at new Point but I can't figure
        # out how recarray does it. So I just use it to make matrix of size 1 and
        # index in to get the np.record/Point
        # All of this is a giant hack so that Point(x=2,y=3) works like expected.
        val = PredictedPointArray(1)
        val[0] = (x, y, visible, complete, score)
        val = val[0]

        # val.x = x
        # val.y = y
        # val.visible = visible
        # val.complete = complete
        # val.score = score

        return val

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
        return cls(**{**Point.asdict(point), 'score': score})


# This turns PredictedPoint into an attrs class. Defines comparators for
# us and generaly makes it behave better. Crazy that this works!
PredictedPoint = attr.s(these={name: attr.ib()
                               for name in PredictedPoint.dtype.names},
                        init=False)(PredictedPoint)


class PointArray(np.recarray):
    """
    PointArray is a sub-class of numpy recarray which stores
    Point objects as records.
    """

    _record_type = Point

    def __new__(subtype, shape, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):

        dtype = subtype._record_type.dtype

        if dtype is not None:
            descr = np.dtype(dtype)
        else:
            descr = np.format_parser(formats, names, titles, aligned, byteorder)._descr

        if buf is None:
            self = np.ndarray.__new__(subtype, shape, (subtype._record_type, descr), order=order)
        else:
            self = np.ndarray.__new__(subtype, shape, (subtype._record_type, descr),
                                      buffer=buf, offset=offset,
                                      strides=strides, order=order)
        return self

    def __array_finalize__(self, obj):
        """
        Overide __array_finalize__ on recarray because it converting the dtype
        of any np.void subclass to np.record, we don't want this.
        """
        pass

    @classmethod
    def make_default(cls, size: int):
        """
        Construct a point array of specific size where each value in the array
        is assigned the default values for a Point.

        Args:
            size: The number of points to allocate.

        Returns:
            A point array with all elements set to Point()
        """
        p = cls(size)
        p[:] = cls._record_type()
        return p

    def __getitem__(self, indx):
        obj = super(np.recarray, self).__getitem__(indx)

        # copy behavior of getattr, except that here
        # we might also be returning a single element
        if isinstance(obj, np.ndarray):
            if obj.dtype.fields:
                obj = obj.view(type(self))
                #if issubclass(obj.dtype.type, numpy.void):
                #    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=np.ndarray)
        else:
            # return a single element
            return obj

    @classmethod
    def from_array(cls, a: 'PointArray'):
        """
        Convert a PointArray to a new PointArray
        (or child class, i.e., PredictedPointArray),
        use the default attribute values for new array.

        Args:
            a: The array to convert.

        Returns:
            A PredictedPointArray with the same points as a.
        """
        v = cls.make_default(len(a))

        for field in Point.dtype.names:
            v[field] = a[field]

        return v

class PredictedPointArray(PointArray):
    """
    PredictedPointArray is analogous to PointArray except for predicted
    points.
    """
    _record_type = PredictedPoint

    @classmethod
    def to_array(cls, a: 'PredictedPointArray'):
        """
        Convert a PredictedPointArray to a normal PointArray.

        Args:
            a: The array to convert.

        Returns:
            The converted array.
        """
        v = PointArray.make_default(len(a))

        for field in Point.dtype.names:
            v[field] = a[field]

        return v


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

@attr.s(cmp=False, slots=True)
class Instance:
    """
    The class :class:`Instance` represents a labelled instance of skeleton

    Args:
        skeleton: The skeleton that this instance is associated with.
        points: A dictionary where keys are skeleton node names and values are Point objects. Alternatively,
        a point array whose length and order matches skeleton.nodes
        track: An optional multi-frame object track associated with this instance.
            This allows individual animals/objects to be tracked across frames.
        from_predicted: The predicted instance (if any) that this was copied from.
        frame: A back reference to the LabeledFrame that this Instance belongs to.
        This field is set when Instances are added to LabeledFrame objects.
    """

    skeleton: Skeleton = attr.ib()
    track: Track = attr.ib(default=None)
    from_predicted: Optional['PredictedInstance'] = attr.ib(default=None)
    _points: PointArray = attr.ib(default=None)
    _nodes: List = attr.ib(default=None)
    frame: Union['LabeledFrame', None] = attr.ib(default=None)

    # The underlying Point array type that this instances point array should be.
    _point_array_type = PointArray

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
        if type(points) is dict:
            is_string_dict = set(map(type, points)) == {str}
            if is_string_dict:
                for node_name in points.keys():
                    if not self.skeleton.has_node(node_name):
                        raise KeyError(f"There is no node named {node_name} in {self.skeleton}")
        elif isinstance(points, PointArray):
            if len(points) != len(self.skeleton.nodes):
                raise ValueError("PointArray does not have the same number of rows as skeleton nodes.")

    def __attrs_post_init__(self):

        if not self.skeleton:
            raise ValueError("No skeleton set for Instance")

        # If the user did not pass a points list initialize a point array for future
        # points.
        if self._points is None or len(self._points) == 0:

            # Initialize an empty point array that is the size of the skeleton.
            self._points = self._point_array_type.make_default(len(self.skeleton.nodes))

        else:

            if type(self._points) is dict:
                parray = self._point_array_type.make_default(len(self.skeleton.nodes))
                Instance._points_dict_to_array(self._points, parray, self.skeleton)
                self._points = parray

        # Now that we've validated the points, cache the list of nodes
        # in the skeleton since the PointArray indexing will be linked
        # to this list even if nodes are removed from the skeleton.
        self._nodes = self.skeleton.nodes

    @staticmethod
    def _points_dict_to_array(points, parray, skeleton):

        # Check if the dict contains all strings
        is_string_dict = set(map(type, points)) == {str}

        # Check if the dict contains all Node objects
        is_node_dict = set(map(type, points)) == {Node}

        # If the user fed in a dict whose keys are strings, these are node names,
        # convert to node indices so we don't break references to skeleton nodes
        # if the node name is relabeled.
        if points and is_string_dict:
            points = {skeleton.find_node(name): point for name, point in points.items()}

        if not is_string_dict and not is_node_dict:
            raise ValueError("points dictionary must be keyed by either strings " +
                             "(node names) or Nodes.")

        # Get rid of the points dict and replace with equivalent point array.
        for node, point in points.items():
            # Convert PredictedPoint to Point if Instance
            if type(parray) == PointArray and type(point) == PredictedPoint:
                point = Point(x=point.x, y=point.y, visible=point.visible, complete=point.complete)
            try:
                parray[skeleton.node_to_index(node)] = point
                # parray[skeleton.node_to_index(node.name)] = point
            except:
                pass

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
        Get the Points associated with particular skeleton node or list of skeleton nodes

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

        try:
            node = self._node_to_index(node)
            return self._points[node]
        except ValueError:
            raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node '{node}'")

    def __contains__(self, node):
        """
        Returns True if this instance has a point with the specified node.

        Args:
            node: node name

        Returns:
            bool: True if the point with the node name specified has a point in this instance.
        """

        if type(node) is Node:
            node = node.name

        if node not in self.skeleton:
            return False

        node_idx = self._node_to_index(node)

        # If the points are nan, then they haven't been allocated.
        return not self._points[node_idx].isnan()

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
            try:
                node_idx = self._node_to_index(node)
                self._points[node_idx] = value
            except ValueError:
                raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node '{node}'")

    def __delitem__(self, node):
        """ Delete node key and points associated with that node. """
        try:
            node_idx = self._node_to_index(node)
            self._points[node_idx].x = math.nan
            self._points[node_idx].y = math.nan
        except ValueError:
            raise KeyError(f"The underlying skeleton ({self.skeleton}) has no node '{node}'")

    def matches(self, other):
        """
        Compare this `Instance` to another, modulo the particular `Node` objects.

        Args:
            other: The other instance.

        Returns:
            True if match, False otherwise.
        """
        if type(self) is not type(other):
            return False

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
            A tuple of nodes that have been labelled for this instance.

        """
        self.fix_array()
        return tuple(self._nodes[i] for i, point in enumerate(self._points)
            if not point.isnan() and self._nodes[i] in self.skeleton.nodes)

    @property
    def nodes_points(self):
        """
        Return view object that displays a list of the instance's (node, point) tuple pairs
        for all labelled point.

        Returns:
            The instance's (node, point) tuple pairs for all labelled point.
        """
        names_to_points = dict(zip(self.nodes, self.points()))
        return names_to_points.items()

    def points(self) -> Tuple[Point]:
        """
        Return the list of labelled points, in order they were labelled.

        Returns:
            The list of labelled points, in order they were labelled.
        """
        self.fix_array()
        return tuple(point for point in self._points if not point.isnan())

    def fix_array(self):
        """Fix points array after nodes have been added or removed."""

        # Check if cached skeleton nodes are different than current nodes
        if self._nodes != self.skeleton.nodes:
            # Create new PointArray (or PredictedPointArray)
            cls = type(self._points)
            new_array = cls.make_default(len(self.skeleton.nodes))

            # Add points into new array
            for i, node in enumerate(self._nodes):
                if node in self.skeleton.nodes:
                    new_array[self.skeleton.nodes.index(node)] = self._points[i]

            # Update points and nodes for this instance
            self._points = new_array
            self._nodes = self.skeleton.nodes

    def points_array(self, copy: bool = True,
                     invisible_as_nan: bool = False,
                     full: bool = False) -> np.ndarray:
        """
        Return the instance's points in array form.

        Args:
            copy: If True, the return a copy of the points array as an
            Nx2 ndarray where first column is x and second column is y.
            If False, return a view of the underlying recarray.
            invisible_as_nan: Should invisible points be marked as NaN.
            full: If True, return the raw underlying recarray with all attributes
            of the point, if not, return just the x and y coordinate. Assumes
            copy is False and invisible_as_nan is False.
        Returns:
            A Nx2 array containing x and y coordinates of each point
            as the rows of the array and N is the number of nodes in the skeleton.
            The order of the rows corresponds to the ordering of the skeleton nodes.
            Any skeleton node not defined will have NaNs present.
        """
        self.fix_array()

        if full:
            return self._points

        if not copy and not invisible_as_nan:
            return self._points[['x', 'y']]
        else:
            parray = structured_to_unstructured(self._points[['x', 'y']])

            if invisible_as_nan:
                parray[~self._points.visible] = math.nan

            return parray

    @property
    def centroid(self) -> np.ndarray:
        """Returns instance centroid as (x,y) numpy row vector."""
        points = self.points_array(invisible_as_nan=True)
        centroid = np.nanmedian(points, axis=0)
        return centroid

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


@attr.s(cmp=False, slots=True)
class PredictedInstance(Instance):
    """
    A predicted instance is an output of the inference procedure. It is
    the main output of the inference procedure.

    Args:
        score: The instance level prediction score.
    """
    score: float = attr.ib(default=0.0, converter=float)

    # The underlying Point array type that this instances point array should be.
    _point_array_type = PredictedPointArray

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
        kw_args = attr.asdict(instance, recurse=False, filter=lambda attr, value: attr.name not in ("_points", "_nodes"))
        kw_args['points'] = PredictedPointArray.from_array(instance._points)
        kw_args['score'] = score
        return cls(**kw_args)


def make_instance_cattr():
    """
    Create a cattr converter for handling Lists of Instances/PredictedInstances

    Returns:
        A cattr converter with hooks registered for structuring and unstructuring
        Instances.
    """

    converter = cattr.Converter()

    #### UNSTRUCTURE HOOKS

    # JSON dump cant handle NumPy bools so convert them. These are present
    # in Point/PredictedPoint objects now since they are actually custom numpy dtypes.
    converter.register_unstructure_hook(np.bool_, bool)

    converter.register_unstructure_hook(PointArray, lambda x: None)
    converter.register_unstructure_hook(PredictedPointArray, lambda x: None)
    def unstructure_instance(x: Instance):

        # Unstructure everything but the points array, nodes, and frame attribute
        d = {field.name: converter.unstructure(x.__getattribute__(field.name))
             for field in attr.fields(x.__class__)
             if field.name not in ['_points', '_nodes', 'frame']}

        # Replace the point array with a dict
        d['_points'] = converter.unstructure({k: v for k, v in x.nodes_points})

        return d

    converter.register_unstructure_hook(Instance, unstructure_instance)
    converter.register_unstructure_hook(PredictedInstance, unstructure_instance)

    ## STRUCTURE HOOKS

    def structure_points(x, type):
        if 'score' in x.keys():
            return cattr.structure(x, PredictedPoint)
        else:
            return cattr.structure(x, Point)

    converter.register_structure_hook(Union[Point, PredictedPoint], structure_points)

    def structure_instances_list(x, type):
        inst_list = []
        for inst_data in x:
            if 'score' in inst_data.keys():
                inst = converter.structure(inst_data, PredictedInstance)
            else:
                inst = converter.structure(inst_data, Instance)
            inst_list.append(inst)

        return inst_list

    converter.register_structure_hook(Union[List[Instance], List[PredictedInstance]],
                                            structure_instances_list)

    converter.register_structure_hook(ForwardRef('PredictedInstance'),
                                      lambda x, type: converter.structure(x, PredictedInstance))

    # We can register structure hooks for point arrays that do nothing
    # because Instance can have a dict of points passed to it in place of
    # a PointArray
    def structure_point_array(x, t):
        if x:
            point1 = x[list(x.keys())[0]]
            if 'score' in point1.keys():
                return converter.structure(x, Dict[Node, PredictedPoint])
            else:
                return converter.structure(x, Dict[Node, Point])
        else:
            return {}

    converter.register_structure_hook(PointArray, structure_point_array)
    converter.register_structure_hook(PredictedPointArray, structure_point_array)

    return converter


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

    def find(self, track=-1, user=False):
        instances = self.instances
        if user:
            instances = list(filter(lambda inst: type(inst) == Instance, instances))
        if track != -1: # use -1 since we want to accept None as possible value
            instances = list(filter(lambda inst: inst.track == track, instances))
        return instances

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


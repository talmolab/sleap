"""
Data structures for all labeled data contained with a SLEAP project.

The relationships between objects in this module:

* A `LabeledFrame` can contain zero or more `Instance`s
  (and `PredictedInstance` objects).

* `Instance` objects (and `PredictedInstance` objects) have `PointArray`
  (or `PredictedPointArray`).

* `Instance` (`PredictedInstance`) can be associated with a `Track`

* A `PointArray` (or `PredictedPointArray`) contains zero or more
  `Point` objects (or `PredictedPoint` objectss), ideally as many as
  there are in the associated :class:`Skeleton` although these can get
  out of sync if the skeleton is manipulated.
"""

import math

import numpy as np
import cattr

from copy import copy
from typing import Dict, List, Optional, Union, Tuple

from numpy.lib.recfunctions import structured_to_unstructured

import sleap
from sleap.skeleton import Skeleton, Node
from sleap.io.video import Video

import attr

try:
    from typing import ForwardRef
except:
    from typing import _ForwardRef as ForwardRef


class Point(np.record):
    """
    A labelled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of point within image frame.
        y: The vertical pixel location of point within image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the user labeler.
    """

    # Define the dtype from the point class attributes plus some
    # additional fields we will use to relate point to instances and
    # nodes.
    dtype = np.dtype([("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")])

    def __new__(
        cls,
        x: float = math.nan,
        y: float = math.nan,
        visible: bool = True,
        complete: bool = False,
    ) -> "Point":

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

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def isnan(self) -> bool:
        """
        Are either of the coordinates a NaN value.

        Returns:
            True if x or y is NaN, False otherwise.
        """
        return math.isnan(self.x) or math.isnan(self.y)

    def numpy() -> np.ndarray:
        """Return the point as a numpy array."""
        return np.array([self.x, self.y])


# This turns PredictedPoint into an attrs class. Defines comparators for
# us and generaly makes it behave better. Crazy that this works!
Point = attr.s(these={name: attr.ib() for name in Point.dtype.names}, init=False)(Point)


class PredictedPoint(Point):
    """
    A predicted point is an output of the inference procedure.

    It has all the properties of a labeled point, plus a score.

    Args:
        x: The horizontal pixel location of point within image frame.
        y: The vertical pixel location of point within image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the user labeler.
        score: The point-level prediction score.
    """

    # Define the dtype from the point class attributes plus some
    # additional fields we will use to relate point to instances and
    # nodes.
    dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("score", "f8")]
    )

    def __new__(
        cls,
        x: float = math.nan,
        y: float = math.nan,
        visible: bool = True,
        complete: bool = False,
        score: float = 0.0,
    ) -> "PredictedPoint":

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
    def from_point(cls, point: Point, score: float = 0.0) -> "PredictedPoint":
        """
        Create a PredictedPoint from a Point

        Args:
            point: The point to copy all data from.
            score: The score for this predicted point.

        Returns:
            A scored point based on the point passed in.
        """
        return cls(**{**Point.asdict(point), "score": score})


# This turns PredictedPoint into an attrs class. Defines comparators for
# us and generaly makes it behave better. Crazy that this works!
PredictedPoint = attr.s(
    these={name: attr.ib() for name in PredictedPoint.dtype.names}, init=False
)(PredictedPoint)


class PointArray(np.recarray):
    """
    PointArray is a sub-class of numpy recarray which stores
    Point objects as records.
    """

    _record_type = Point

    def __new__(
        subtype,
        shape,
        buf=None,
        offset=0,
        strides=None,
        formats=None,
        names=None,
        titles=None,
        byteorder=None,
        aligned=False,
        order="C",
    ) -> "PointArray":

        dtype = subtype._record_type.dtype

        if dtype is not None:
            descr = np.dtype(dtype)
        else:
            descr = np.format_parser(formats, names, titles, aligned, byteorder)._descr

        if buf is None:
            self = np.ndarray.__new__(
                subtype, shape, (subtype._record_type, descr), order=order
            )
        else:
            self = np.ndarray.__new__(
                subtype,
                shape,
                (subtype._record_type, descr),
                buffer=buf,
                offset=offset,
                strides=strides,
                order=order,
            )
        return self

    def __array_finalize__(self, obj):
        """
        Override :method:`np.recarray.__array_finalize__()`.

        Overide __array_finalize__ on recarray because it converting the
        dtype of any np.void subclass to np.record, we don't want this.
        """
        pass

    @classmethod
    def make_default(cls, size: int) -> "PointArray":
        """
        Construct a point array where points are all set to default.

        The constructed :class:`PointArray` will have specified size
        and each value in the array is assigned the default values for
        a :class:`Point``.

        Args:
            size: The number of points to allocate.

        Returns:
            A point array with all elements set to Point()
        """
        p = cls(size)
        p[:] = cls._record_type()
        return p

    def __getitem__(self, indx: int) -> "Point":
        """Get point by its index in the array."""
        obj = super(np.recarray, self).__getitem__(indx)

        # copy behavior of getattr, except that here
        # we might also be returning a single element
        if isinstance(obj, np.ndarray):
            if obj.dtype.fields:
                obj = obj.view(type(self))
                # if issubclass(obj.dtype.type, numpy.void):
                #    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=np.ndarray)
        else:
            # return a single element
            return obj

    @classmethod
    def from_array(cls, a: "PointArray") -> "PointArray":
        """
        Converts a :class:`PointArray` (or child) to a new instance.

        This will convert an object to the same type as itself,
        so a :class:`PredictedPointArray` will result in the same.

        Uses the default attribute values for new array.

        Args:
            a: The array to convert.

        Returns:
            A :class:`PointArray` or :class:`PredictedPointArray` with
            the same points as a.
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
    def to_array(cls, a: "PredictedPointArray") -> "PointArray":
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


@attr.s(slots=True, eq=False, order=False)
class Track:
    """
    A track object is associated with a set of animal/object instances
    across multiple frames of video. This allows tracking of unique
    entities in the video over time and space.

    Args:
        spawned_on: The video frame that this track was spawned on.
        name: A name given to this track for identifying purposes.
    """

    spawned_on: int = attr.ib(default=0, converter=int)
    name: str = attr.ib(default="", converter=str)

    def matches(self, other: "Track"):
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


@attr.s(eq=False, order=False, slots=True, repr=False, str=False)
class Instance:
    """This class represents a labeled instance.

    Args:
        skeleton: The skeleton that this instance is associated with.
        points: A dictionary where keys are skeleton node names and
            values are Point objects. Alternatively, a point array whose
            length and order matches skeleton.nodes.
        track: An optional multi-frame object track associated with
            this instance. This allows individual animals/objects to be
            tracked across frames.
        from_predicted: The predicted instance (if any) that this was
            copied from.
        frame: A back reference to the :class:`LabeledFrame` that this
            :class:`Instance` belongs to. This field is set when
            instances are added to :class:`LabeledFrame` objects.
    """

    skeleton: Skeleton = attr.ib()
    track: Track = attr.ib(default=None)
    from_predicted: Optional["PredictedInstance"] = attr.ib(default=None)
    _points: PointArray = attr.ib(default=None)
    _nodes: List = attr.ib(default=None)
    frame: Union["LabeledFrame", None] = attr.ib(default=None)

    # The underlying Point array type that this instances point array should be.
    _point_array_type = PointArray

    @from_predicted.validator
    def _validate_from_predicted_(
        self, attribute, from_predicted: Optional["PredictedInstance"]
    ):
        """Validation method called by attrs.

        Checks that from_predicted is None or :class:`PredictedInstance`

        Args:
            attribute: Attribute being validated; not used.
            from_predicted: Value being validated.

        Raises:
            TypeError: If from_predicted is anything other than None
                or a `PredictedInstance`.

        """
        if from_predicted is not None and type(from_predicted) != PredictedInstance:
            raise TypeError(
                f"Instance.from_predicted type must be PredictedInstance (not "
                "{type(from_predicted)})"
            )

    @_points.validator
    def _validate_all_points(self, attribute, points: Union[dict, PointArray]):
        """Validation method called by attrs.

        Checks that all the _points defined for the skeleton are found
        in the skeleton.

        Args:
            attribute: Attribute being validated; not used.
            points: Either dict of points or PointArray
                If dict, keys should be node names.

        Raises:
            ValueError: If a point is associated with a skeleton node
                name that doesn't exist.

        Returns:
            None
        """
        if type(points) is dict:
            is_string_dict = set(map(type, points)) == {str}
            if is_string_dict:
                for node_name in points.keys():
                    if not self.skeleton.has_node(node_name):
                        raise KeyError(
                            f"There is no node named {node_name} in {self.skeleton}"
                        )
        elif isinstance(points, PointArray):
            if len(points) != len(self.skeleton.nodes):
                raise ValueError(
                    "PointArray does not have the same number of rows as skeleton "
                    "nodes."
                )

    def __attrs_post_init__(self):
        """Method called by attrs after __init__().

        Initializes points if none were specified when creating object,
        caches list of nodes so what we can still find points in array
        if the `Skeleton` changes.

        Args:
            None

        Raises:
            ValueError: If object has no `Skeleton`.
        """
        if self.skeleton is None:
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
    def _points_dict_to_array(
        points: Dict[Union[str, Node], Point], parray: PointArray, skeleton: Skeleton
    ):
        """Set values in given :class:`PointsArray` from dictionary.

        Args:
            points: The dictionary of points. Keys can be either node
                names or :class:`Node`s, values are :class:`Point`s.
            parray: The :class:`PointsArray` which is being updated.
            skeleton: The :class:`Skeleton` which contains the nodes
                referenced in the dictionary of points.

        Raises:
            ValueError: If dictionary keys are not either all strings
                or all :class:`Node`s.
        """
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
            raise ValueError(
                "points dictionary must be keyed by either strings "
                + "(node names) or Nodes."
            )

        # Get rid of the points dict and replace with equivalent point array.
        for node, point in points.items():
            # Convert PredictedPoint to Point if Instance
            if type(parray) == PointArray and type(point) == PredictedPoint:
                point = Point(
                    x=point.x, y=point.y, visible=point.visible, complete=point.complete
                )
            try:
                parray[skeleton.node_to_index(node)] = point
                # parray[skeleton.node_to_index(node.name)] = point
            except:
                pass

    def _node_to_index(self, node: Union[str, Node]) -> int:
        """Helper method to get the index of a node from its name.

        Args:
            node: Node name or :class:`Node` object.

        Returns:
            The index of the node on skeleton graph.
        """
        return self.skeleton.node_to_index(node)

    def __getitem__(
        self,
        node: Union[List[Union[str, Node, int]], Union[str, Node, int], np.ndarray],
    ) -> Union[List[Point], Point, np.ndarray]:
        """Get the Points associated with particular skeleton node(s).

        Args:
            node: A single node or list of nodes within the skeleton
                associated with this instance.

        Raises:
            KeyError: If node cannot be found in skeleton.

        Returns:
            Either a single point (if a single node given), or
            a list of points (if a list of nodes given) corresponding
            to each node.

        """
        self._fix_array()
        # If the node is a list of nodes, use get item recursively and return a list of
        # _points.
        if isinstance(node, (list, tuple, np.ndarray)):
            pts = []
            for n in node:
                pts.append(self.__getitem__(n))

            if isinstance(node, np.ndarray):
                return np.array([[pt.x, pt.y] for pt in pts])
            else:
                return pts

        if isinstance(node, (Node, str)):
            try:
                node = self._node_to_index(node)
            except ValueError:
                raise KeyError(
                    f"The underlying skeleton ({self.skeleton}) has no node '{node}'"
                )
        return self._points[node]

    def __contains__(self, node: Union[str, Node, int]) -> bool:
        """Whether this instance has a point with the specified node.

        Args:
            node: Node name or :class:`Node` object.

        Returns:
            bool: True if the point with the node name specified has a
                point in this instance.
        """

        if isinstance(node, Node):
            node = node.name

        if isinstance(node, str):
            if node not in self.skeleton:
                return False
            node = self._node_to_index(node)

        # If the points are nan, then they haven't been allocated.
        return not self._points[node].isnan()

    def __setitem__(
        self,
        node: Union[List[Union[str, Node, int]], Union[str, Node, int], np.ndarray],
        value: Union[List[Point], Point, np.ndarray],
    ):
        """Set the point(s) for given node(s).

        Args:
            node: Either node (by name or `Node`) or list of nodes.
            value: Either `Point` or list of `Point`s.

        Raises:
            IndexError: If lengths of lists don't match, or if exactly
                one of the inputs is a list.
            KeyError: If skeleton does not have (one of) the node(s).
        """
        self._fix_array()
        # Make sure node and value, if either are lists, are of compatible size
        if isinstance(node, (list, np.ndarray)):
            if not isinstance(value, (list, np.ndarray)) or len(value) != len(node):
                raise IndexError(
                    "Node list for indexing must be same length and value list."
                )

            for n, v in zip(node, value):
                self.__setitem__(n, v)
        else:
            if isinstance(node, (Node, str)):
                try:
                    node_idx = self._node_to_index(node)
                except ValueError:
                    raise KeyError(
                        f"The skeleton ({self.skeleton}) has no node '{node}'."
                    )
            else:
                node_idx = node

            if not isinstance(value, Point):
                if hasattr(value, "__len__") and len(value) == 2:
                    value = Point(x=value[0], y=value[1])
                else:
                    raise ValueError(
                        "Instance point values must be (x, y) coordinates."
                    )
            self._points[node_idx] = value

    def __delitem__(self, node: Union[str, Node]):
        """Delete node key and points associated with that node.

        Args:
            node: Node name or :class:`Node` object.

        Raises:
            KeyError: If skeleton does not have the node.

        Returns:
            None
        """
        try:
            node_idx = self._node_to_index(node)
            self._points[node_idx].x = math.nan
            self._points[node_idx].y = math.nan
        except ValueError:
            raise KeyError(
                f"The underlying skeleton ({self.skeleton}) has no node '{node}'"
            )

    def __repr__(self) -> str:
        """Return string representation of this object."""
        pts = []
        for node, pt in self.nodes_points:
            pts.append(f"{node.name}: ({pt.x:.1f}, {pt.y:.1f})")
        pts = ", ".join(pts)

        return (
            "Instance("
            f"video={self.video}, "
            f"frame_idx={self.frame_idx}, "
            f"points=[{pts}], "
            f"track={self.track}"
            ")"
        )

    def matches(self, other: "Instance") -> bool:
        """Whether two instances match by value.

        Checks the types, points, track, and frame index.

        Args:
            other: The other :class:`Instance`.

        Returns:
            True if match, False otherwise.
        """
        if type(self) is not type(other):
            return False

        if list(self.points) != list(other.points):
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
    def nodes(self) -> Tuple[Node, ...]:
        """Return nodes that have been labelled for this instance."""
        self._fix_array()
        return tuple(
            self._nodes[i]
            for i, point in enumerate(self._points)
            if not point.isnan() and self._nodes[i] in self.skeleton.nodes
        )

    @property
    def nodes_points(self) -> List[Tuple[Node, Point]]:
        """Return a list of (node, point) tuples for all labeled points."""
        names_to_points = dict(zip(self.nodes, self.points))
        return names_to_points.items()

    @property
    def points(self) -> Tuple[Point, ...]:
        """Return a tuple of labelled points, in the order they were labelled."""
        self._fix_array()
        return tuple(point for point in self._points if not point.isnan())

    def _fix_array(self):
        """Fix PointArray after nodes have been added or removed.

        This updates the PointArray as required by comparing the cached
        list of nodes to the nodes in the `Skeleton` object (which may
        have changed).
        """
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

    def get_points_array(
        self, copy: bool = True, invisible_as_nan: bool = False, full: bool = False
    ) -> Union[np.ndarray, np.recarray]:
        """Return the instance's points in array form.

        Args:
            copy: If True, the return a copy of the points array as an ndarray.
                If False, return a view of the underlying recarray.
            invisible_as_nan: Should invisible points be marked as NaN.
                If copy is False, then invisible_as_nan is ignored since we
                don't want to set invisible points to NaNs in original data.
            full: If True, return all data for points. Otherwise, return just
                the x and y coordinates.

        Returns:
            Either a recarray (if copy is False) or an ndarray (if copy True).

            The order of the rows corresponds to the ordering of the skeleton
            nodes. Any skeleton node not defined will have NaNs present.

            Columns in recarray are accessed by name, e.g., ["x"], ["y"].

            Columns in ndarray are accessed by number. The order matches
            the order in `Point.dtype` or `PredictedPoint.dtype`.
        """
        self._fix_array()

        if not copy:
            if full:
                return self._points
            else:
                return self._points[["x", "y"]]
        else:
            if full:
                parray = structured_to_unstructured(self._points)
            else:
                parray = structured_to_unstructured(self._points[["x", "y"]])

            # Note that invisible_as_nan assumes copy is True.
            if invisible_as_nan:
                parray[~self._points.visible] = math.nan

            return parray

    def fill_missing(
        self, max_x: Optional[float] = None, max_y: Optional[float] = None
    ):
        """Add points for skeleton nodes that are missing in the instance.

        This is useful when modifying the skeleton so the nodes appears in the GUI.

        Args:
            max_x: If specified, make sure points are not added outside of valid range.
            max_y: If specified, make sure points are not added outside of valid range.
        """
        self._fix_array()
        y1, x1, y2, x2 = self.bounding_box
        y1, x1 = max(y1, 0), max(x1, 0)
        if max_x is not None:
            x2 = min(x2, max_x)
        if max_y is not None:
            y2 = min(y2, max_y)
        w, h = y2 - y1, x2 - x1

        for node in self.skeleton.nodes:
            if node not in self.nodes or self[node].isnan():
                off = np.array([w, h]) * np.random.rand(2)
                x, y = off + np.array([x1, y1])
                y, x = max(y, 0), max(x, 0)
                if max_x is not None:
                    x = min(x, max_x)
                if max_y is not None:
                    y = min(y, max_y)

                self[node] = Point(x=x, y=y, visible=False)

    @property
    def points_array(self) -> np.ndarray:
        """Return array of x and y coordinates for visible points.

        Row in array corresponds to order of points in skeleton. Invisible points will
        be denoted by NaNs.

        Returns:
            A numpy array of of shape `(n_nodes, 2)` point coordinates.
        """
        return self.get_points_array(invisible_as_nan=True)

    def numpy(self) -> np.ndarray:
        """Return the instance node coordinates as a numpy array.

        Alias for `points_array`.

        Returns:
            Array of shape `(n_nodes, 2)` of dtype `float32` containing the coordinates
            of the instance's nodes. Missing/not visible nodes will be replaced with
            `NaN`.
        """
        return self.points_array

    def transform_points(self, transformation_matrix):
        """Apply affine transformation matrix to points in the instance.

        Args:
            transformation_matrix: Affine transformation matrix as a numpy array of
                shape `(3, 3)`.
        """
        points = self.get_points_array(copy=True, full=False, invisible_as_nan=False)

        if transformation_matrix.shape[1] == 3:
            rotation = transformation_matrix[:, :2]
            translation = transformation_matrix[:, 2]

            transformed = points @ rotation.T + translation

        else:
            transformed = points @ transformation_matrix.T

        self._points["x"] = transformed[:, 0]
        self._points["y"] = transformed[:, 1]

    @property
    def centroid(self) -> np.ndarray:
        """Return instance centroid as an array of `(x, y)` coordinates

        Notes:
            This computes the centroid as the median of the visible points.
        """
        points = self.points_array
        centroid = np.nanmedian(points, axis=0)
        return centroid

    @property
    def bounding_box(self) -> np.ndarray:
        """Return bounding box containing all points in `[y1, x1, y2, x2]` format."""
        points = self.points_array
        if np.isnan(points).all():
            return np.array([np.nan, np.nan, np.nan, np.nan])
        bbox = np.concatenate(
            [np.nanmin(points, axis=0)[::-1], np.nanmax(points, axis=0)[::-1]]
        )
        return bbox

    @property
    def midpoint(self) -> np.ndarray:
        """Return the center of the bounding box of the instance points."""
        y1, x1, y2, x2 = self.bounding_box
        return np.array([(x2 - x1) / 2, (y2 - y1) / 2])

    @property
    def n_visible_points(self) -> int:
        """Return the number of visible points in this instance."""
        n = 0
        for p in self.points:
            if p.visible:
                n += 1
        return n

    def __len__(self) -> int:
        """Return the number of visible points in this instance."""
        return self.n_visible_points

    @property
    def video(self) -> Optional[Video]:
        """Return the video of the labeled frame this instance is associated with."""
        if self.frame is None:
            return None
        else:
            return self.frame.video

    @property
    def frame_idx(self) -> Optional[int]:
        """Return the index of the labeled frame this instance is associated with."""
        if self.frame is None:
            return None
        else:
            return self.frame.frame_idx

    @classmethod
    def from_pointsarray(
        cls, points: np.ndarray, skeleton: Skeleton, track: Optional[Track] = None
    ) -> "Instance":
        """Create an instance from an array of points.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in (x, y) coordinates of each node. Missing nodes
                should be represented as `NaN`.
            skeleton: A `sleap.Skeleton` instance with `n_nodes` nodes to associate with
                the instance.
            track: Optional `sleap.Track` object to associate with the instance.

        Returns:
            A new `Instance` object.
        """
        predicted_points = dict()
        for point, node_name in zip(points, skeleton.node_names):
            if np.isnan(point).any():
                continue

            predicted_points[node_name] = Point(x=point[0], y=point[1])

        return cls(points=predicted_points, skeleton=skeleton, track=track)

    @classmethod
    def from_numpy(
        cls, points: np.ndarray, skeleton: Skeleton, track: Optional[Track] = None
    ) -> "Instance":
        """Create an instance from a numpy array.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in (x, y) coordinates of each node. Missing nodes
                should be represented as `NaN`.
            skeleton: A `sleap.Skeleton` instance with `n_nodes` nodes to associate with
                the instance.
            track: Optional `sleap.Track` object to associate with the instance.

        Returns:
            A new `Instance` object.

        Notes:
            This is an alias for `Instance.from_pointsarray()`.
        """
        return cls.from_pointsarray(points, skeleton, track=track)

    def _merge_nodes_data(self, base_node: str, merge_node: str):
        """Copy point data from one node to another.

        Args:
            base_node: Name of node that will be merged into.
            merge_node: Name of node that will be removed after merge.

        Notes:
            This is used when merging skeleton nodes and should not be called directly.
        """
        base_pt = self[base_node]
        merge_pt = self[merge_node]
        if merge_pt.isnan():
            return
        if base_pt.isnan() or not base_pt.visible:
            base_pt.x = merge_pt.x
            base_pt.y = merge_pt.y
            base_pt.visible = merge_pt.visible
            base_pt.complete = merge_pt.complete
            if hasattr(base_pt, "score"):
                base_pt.score = merge_pt.score


@attr.s(eq=False, order=False, slots=True, repr=False, str=False)
class PredictedInstance(Instance):
    """
    A predicted instance is an output of the inference procedure.

    Args:
        score: The instance-level grouping prediction score.
        tracking_score: The instance-level track matching score.
    """

    score: float = attr.ib(default=0.0, converter=float)
    tracking_score: float = attr.ib(default=0.0, converter=float)

    # The underlying Point array type that this instances point array should be.
    _point_array_type = PredictedPointArray

    def __attrs_post_init__(self):
        super(PredictedInstance, self).__attrs_post_init__()

        if self.from_predicted is not None:
            raise ValueError("PredictedInstance should not have from_predicted.")

    def __repr__(self) -> str:
        """Return string representation of this object."""
        pts = []
        for node, pt in self.nodes_points:
            pts.append(f"{node.name}: ({pt.x:.1f}, {pt.y:.1f}, {pt.score:.2f})")
        pts = ", ".join(pts)

        return (
            "PredictedInstance("
            f"video={self.video}, "
            f"frame_idx={self.frame_idx}, "
            f"points=[{pts}], "
            f"score={self.score:.2f}, "
            f"track={self.track}, "
            f"tracking_score={self.tracking_score:.2f}"
            ")"
        )

    @property
    def points_and_scores_array(self) -> np.ndarray:
        """Return the instance points and scores as an array.

        This will be a `(n_nodes, 3)` array of `(x, y, score)` for each predicted point.

        Rows in the array correspond to the order of points in skeleton. Invisible
        points will be represented as NaNs.
        """
        pts = self.get_points_array(full=True, copy=True, invisible_as_nan=True)
        return pts[:, (0, 1, 4)]  # (x, y, score)

    @property
    def scores(self) -> np.ndarray:
        """Return point scores for each predicted node."""
        return self.points_and_scores_array[:, 2]

    @classmethod
    def from_instance(cls, instance: Instance, score: float) -> "PredictedInstance":
        """Create a `PredictedInstance` from an `Instance`.

        The fields are copied in a shallow manner with the exception of points. For each
        point in the instance a `PredictedPoint` is created with score set to default
        value.

        Args:
            instance: The `Instance` object to shallow copy data from.
            score: The score for this instance.

        Returns:
            A `PredictedInstance` for the given `Instance`.
        """
        kw_args = attr.asdict(
            instance,
            recurse=False,
            filter=lambda attr, value: attr.name not in ("_points", "_nodes"),
        )
        kw_args["points"] = PredictedPointArray.from_array(instance._points)
        kw_args["score"] = score
        return cls(**kw_args)

    @classmethod
    def from_arrays(
        cls,
        points: np.ndarray,
        point_confidences: np.ndarray,
        instance_score: float,
        skeleton: Skeleton,
        track: Optional[Track] = None,
    ) -> "PredictedInstance":
        """Create a predicted instance from data arrays.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in `(x, y)` coordinates of each node. Missing nodes
                should be represented as `NaN`.
            point_confidences: A numpy array of shape `(n_nodes,)` and dtype `float32`
                that contains the confidence/score of the points.
            instance_score: Scalar float representing the overall instance score, e.g.,
                the PAF grouping score.
            skeleton: A sleap.Skeleton instance with n_nodes nodes to associate with the
                predicted instance.
            track: Optional `sleap.Track` to associate with the instance.

        Returns:
            A new `PredictedInstance`.
        """
        predicted_points = dict()
        for point, confidence, node_name in zip(
            points, point_confidences, skeleton.node_names
        ):
            if np.isnan(point).any():
                continue

            predicted_points[node_name] = PredictedPoint(
                x=point[0], y=point[1], score=confidence
            )

        return cls(
            points=predicted_points,
            skeleton=skeleton,
            score=instance_score,
            track=track,
        )

    @classmethod
    def from_pointsarray(
        cls,
        points: np.ndarray,
        point_confidences: np.ndarray,
        instance_score: float,
        skeleton: Skeleton,
        track: Optional[Track] = None,
    ) -> "PredictedInstance":
        """Create a predicted instance from data arrays.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in `(x, y)` coordinates of each node. Missing nodes
                should be represented as `NaN`.
            point_confidences: A numpy array of shape `(n_nodes,)` and dtype `float32`
                that contains the confidence/score of the points.
            instance_score: Scalar float representing the overall instance score, e.g.,
                the PAF grouping score.
            skeleton: A sleap.Skeleton instance with n_nodes nodes to associate with the
                predicted instance.
            track: Optional `sleap.Track` to associate with the instance.

        Returns:
            A new `PredictedInstance`.
        """
        return cls.from_arrays(
            points, point_confidences, instance_score, skeleton, track=track
        )

    @classmethod
    def from_numpy(
        cls,
        points: np.ndarray,
        point_confidences: np.ndarray,
        instance_score: float,
        skeleton: Skeleton,
        track: Optional[Track] = None,
    ) -> "PredictedInstance":
        """Create a predicted instance from data arrays.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in `(x, y)` coordinates of each node. Missing nodes
                should be represented as `NaN`.
            point_confidences: A numpy array of shape `(n_nodes,)` and dtype `float32`
                that contains the confidence/score of the points.
            instance_score: Scalar float representing the overall instance score, e.g.,
                the PAF grouping score.
            skeleton: A sleap.Skeleton instance with n_nodes nodes to associate with the
                predicted instance.
            track: Optional `sleap.Track` to associate with the instance.

        Returns:
            A new `PredictedInstance`.
        """
        return cls.from_arrays(
            points, point_confidences, instance_score, skeleton, track=track
        )


def make_instance_cattr() -> cattr.Converter:
    """Create a cattr converter for Lists of Instances/PredictedInstances.

    This is required because cattrs doesn't automatically detect the class when the
    attributes of one class are a subset of another.

    Returns:
        A cattr converter with hooks registered for structuring and unstructuring
        `Instance` and `PredictedInstance` objects.
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
        d = {
            field.name: converter.unstructure(x.__getattribute__(field.name))
            for field in attr.fields(x.__class__)
            if field.name not in ["_points", "_nodes", "frame"]
        }

        # Replace the point array with a dict
        d["_points"] = converter.unstructure({k: v for k, v in x.nodes_points})

        return d

    converter.register_unstructure_hook(Instance, unstructure_instance)
    converter.register_unstructure_hook(PredictedInstance, unstructure_instance)

    ## STRUCTURE HOOKS

    def structure_points(x, type):
        if "score" in x.keys():
            return cattr.structure(x, PredictedPoint)
        else:
            return cattr.structure(x, Point)

    converter.register_structure_hook(Union[Point, PredictedPoint], structure_points)

    # Function to determine object type for objects being structured.
    def structure_instances_list(x, type):
        inst_list = []
        for inst_data in x:
            if "score" in inst_data.keys():
                inst = converter.structure(inst_data, PredictedInstance)
            else:
                inst = converter.structure(inst_data, Instance)
            inst_list.append(inst)

        return inst_list

    converter.register_structure_hook(
        Union[List[Instance], List[PredictedInstance]], structure_instances_list
    )

    converter.register_structure_hook(
        ForwardRef("PredictedInstance"),
        lambda x, type: converter.structure(x, PredictedInstance),
    )

    # We can register structure hooks for point arrays that do nothing
    # because Instance can have a dict of points passed to it in place of
    # a PointArray
    def structure_point_array(x, t):
        if x:
            point1 = x[list(x.keys())[0]]
            if "score" in point1.keys():
                return converter.structure(x, Dict[Node, PredictedPoint])
            else:
                return converter.structure(x, Dict[Node, Point])
        else:
            return {}

    converter.register_structure_hook(PointArray, structure_point_array)
    converter.register_structure_hook(PredictedPointArray, structure_point_array)

    return converter


@attr.s(auto_attribs=True, eq=False, repr=False, str=False)
class LabeledFrame:
    """Holds labeled data for a single frame of a video.

    Args:
        video: The :class:`Video` associated with this frame.
        frame_idx: The index of frame in video.
        instances: List of instances associated with the frame.
    """

    video: Video = attr.ib()
    frame_idx: int = attr.ib(converter=int)
    _instances: Union[List[Instance], List[PredictedInstance]] = attr.ib(
        default=attr.Factory(list)
    )

    def __attrs_post_init__(self):
        """Called by attrs.

        Updates :attribute:`Instance.frame` for each instance associated
        with this :class:`LabeledFrame`.
        """

        # Make sure all instances have a reference to this frame
        for instance in self.instances:
            instance.frame = self

    def __len__(self) -> int:
        """Return number of instances associated with frame."""
        return len(self.instances)

    def __getitem__(self, index) -> Instance:
        """Return instance (retrieved by index)."""
        return self.instances.__getitem__(index)

    def index(self, value: Instance) -> int:
        """Return index of given :class:`Instance`."""
        return self.instances.index(value)

    def __delitem__(self, index):
        """Remove instance (by index) from frame."""
        value = self.instances.__getitem__(index)

        self.instances.__delitem__(index)

        # Modify the instance to remove reference to this frame
        value.frame = None

    def __repr__(self) -> str:
        """Return a readable representation of the LabeledFrame."""
        return (
            f"LabeledFrame(video={type(self.video.backend).__name__}"
            f"('{self.video.filename}'), "
            f"frame_idx={self.frame_idx}, "
            f"instances={len(self.instances)})"
        )

    def insert(self, index: int, value: Instance):
        """Add instance to frame.

        Args:
            index: The index in list of frame instances where we should
                insert the new instance.
            value: The instance to associate with frame.

        Returns:
            None.
        """
        self.instances.insert(index, value)

        # Modify the instance to have a reference back to this frame
        value.frame = self

    def __setitem__(self, index, value: Instance):
        """Set nth instance in frame to the given instance.

        Args:
            index: The index of instance to replace with new instance.
            value: The new instance to associate with frame.

        Returns:
            None.
        """
        self.instances.__setitem__(index, value)

        # Modify the instance to have a reference back to this frame
        value.frame = self

    def find(
        self, track: Optional[Union[Track, int]] = -1, user: bool = False
    ) -> List[Instance]:
        """Retrieve instances (if any) matching specifications.

        Args:
            track: The :class:`Track` to match. Note that None will only
                match instances where :attribute:`Instance.track` is
                None. If track is -1, then we'll match any track.
            user: Whether to only match user (non-predicted) instances.

        Returns:
            List of instances.
        """
        instances = self.instances
        if user:
            instances = list(filter(lambda inst: type(inst) == Instance, instances))
        if track != -1:  # use -1 since we want to accept None as possible value
            instances = list(filter(lambda inst: inst.track == track, instances))
        return instances

    @property
    def instances(self) -> List[Instance]:
        """Return list of all instances associated with this frame."""
        return self._instances

    @instances.setter
    def instances(self, instances: List[Instance]):
        """Set the list of instances associated with this frame.

        Updates the `frame` attribute on each instance to the
        :class:`LabeledFrame` which will contain the instance.
        The list of instances replaces instances that were previously
        associated with frame.

        Args:
            instances: A list of instances associated with this frame.

        Returns:
            None
        """

        # Make sure to set the frame for each instance to this LabeledFrame
        for instance in instances:
            instance.frame = self

        self._instances = instances

    @property
    def user_instances(self) -> List[Instance]:
        """Return list of user instances associated with this frame."""
        return [inst for inst in self._instances if type(inst) == Instance]

    @property
    def training_instances(self) -> List[Instance]:
        """Return list of user instances with points for training."""
        return [
            inst
            for inst in self._instances
            if not isinstance(inst, PredictedInstance) and inst.n_visible_points
        ]

    @property
    def predicted_instances(self) -> List[PredictedInstance]:
        """Return list of predicted instances associated with frame."""
        return [inst for inst in self._instances if type(inst) == PredictedInstance]

    @property
    def tracked_instances(self) -> List[PredictedInstance]:
        """Return list of predicted instances with tracks associated with frame."""
        return [
            inst
            for inst in self._instances
            if type(inst) == PredictedInstance and inst.track is not None
        ]

    def remove_untracked(self):
        """Removes any instances without a track assignment."""
        self.instances = [inst for inst in self.instances if inst.track is not None]

    @property
    def has_user_instances(self) -> bool:
        """Return whether the frame contains any user instances."""
        for inst in self._instances:
            if type(inst) == Instance:
                return True
        return False

    @property
    def has_predicted_instances(self) -> bool:
        """Return whether the frame contains any predicted instances."""
        for inst in self._instances:
            if type(inst) == PredictedInstance:
                return True
        return False

    @property
    def has_tracked_instances(self) -> bool:
        """Return whether the frame contains any predicted instances with tracks."""
        for inst in self._instances:
            if type(inst) == PredictedInstance and inst.track is not None:
                return True
        return False

    @property
    def n_user_instances(self) -> int:
        """Return the number of user instances in the frame."""
        n = 0
        for inst in self._instances:
            if type(inst) == Instance:
                n += 1
        return n

    @property
    def n_predicted_instances(self) -> int:
        """Return the number of predicted instances in the frame."""
        n = 0
        for inst in self._instances:
            if type(inst) == PredictedInstance:
                n += 1
        return n

    @property
    def n_tracked_instances(self) -> int:
        """Return the number of predicted instances with tracks in the frame."""
        n = 0
        for inst in self._instances:
            if type(inst) == PredictedInstance and inst.track is not None:
                n += 1
        return n

    def remove_empty_instances(self):
        """Remove instances with no visible nodes from the labeled frame."""
        self.instances = [inst for inst in self.instances if inst.n_visible_points > 0]

    @property
    def unused_predictions(self) -> List[Instance]:
        """Return a list of "unused" :class:`PredictedInstance` objects in frame.

        This is all the :class:`PredictedInstance` objects which do not have
        a corresponding :class:`Instance` in the same track in frame.
        """
        unused_predictions = []
        any_tracks = [inst.track for inst in self._instances if inst.track is not None]
        if len(any_tracks):
            # use tracks to determine which predicted instances have been used
            used_tracks = [
                inst.track
                for inst in self._instances
                if type(inst) == Instance and inst.track is not None
            ]
            unused_predictions = [
                inst
                for inst in self._instances
                if inst.track not in used_tracks and type(inst) == PredictedInstance
            ]

        else:
            # use from_predicted to determine which predicted instances have been used
            # TODO: should we always do this instead of using tracks?
            used_instances = [
                inst.from_predicted
                for inst in self._instances
                if inst.from_predicted is not None
            ]
            unused_predictions = [
                inst
                for inst in self._instances
                if type(inst) == PredictedInstance and inst not in used_instances
            ]

        return unused_predictions

    @property
    def instances_to_show(self) -> List[Instance]:
        """Return a list of instances to show in GUI for this frame.

        This list will not include any predicted instances for which
        there's a corresponding regular instance.

        Returns:
            List of instances to show in GUI.
        """
        unused_predictions = self.unused_predictions
        inst_to_show = [
            inst
            for inst in self._instances
            if type(inst) == Instance or inst in unused_predictions
        ]
        inst_to_show.sort(
            key=lambda inst: inst.track.spawned_on
            if inst.track is not None
            else math.inf
        )
        return inst_to_show

    @staticmethod
    def merge_frames(
        labeled_frames: List["LabeledFrame"], video: "Video", remove_redundant=True
    ) -> List["LabeledFrame"]:
        """Return merged LabeledFrames for same video and frame index.

        Args:
            labeled_frames: List of :class:`LabeledFrame` objects to merge.
            video: The :class:`Video` for which to merge.
                This is specified so we don't have to check all frames when we
                already know which video has new labeled frames.
            remove_redundant: Whether to drop instances in the merged frames
                where there's a perfect match.

        Returns:
            The merged list of :class:`LabeledFrame`s.
        """
        redundant_count = 0
        frames_found = dict()
        # move instances into first frame with matching frame_idx
        for idx, lf in enumerate(labeled_frames):
            if lf.video == video:
                if lf.frame_idx in frames_found.keys():
                    # move instances
                    dst_idx = frames_found[lf.frame_idx]
                    if remove_redundant:
                        for new_inst in lf.instances:
                            redundant = False
                            for old_inst in labeled_frames[dst_idx].instances:
                                if new_inst.matches(old_inst):
                                    redundant = True
                                    if not hasattr(new_inst, "score"):
                                        redundant_count += 1
                                    break
                            if not redundant:
                                labeled_frames[dst_idx].instances.append(new_inst)
                    else:
                        labeled_frames[dst_idx].instances.extend(lf.instances)
                    lf.instances = []
                else:
                    # note first lf with this frame_idx
                    frames_found[lf.frame_idx] = idx
        # remove labeled frames with no instances
        labeled_frames = list(filter(lambda lf: len(lf.instances), labeled_frames))
        if redundant_count:
            print(f"skipped {redundant_count} redundant instances")
        return labeled_frames

    @classmethod
    def complex_merge_between(
        cls, base_labels: "Labels", new_frames: List["LabeledFrame"]
    ) -> Tuple[Dict[Video, Dict[int, List[Instance]]], List[Instance], List[Instance]]:
        """Merge data from new frames into a :class:`Labels` object.

        Everything that can be merged cleanly is merged, any conflicts
        are returned.

        Args:
            base_labels: The :class:`Labels` into which we are merging.
            new_frames: The list of :class:`LabeledFrame` objects from
                which we are merging.
        Returns:
            tuple of three items:
            * Dictionary, keys are :class:`Video`, values are
                dictionary in which keys are frame index (int)
                and value is list of :class:`Instance`s
            * list of conflicting :class:`Instance` objects from base
            * list of conflicting :class:`Instance` objects from new frames
        """
        merged = dict()
        extra_base = []
        extra_new = []

        for new_frame in new_frames:
            base_lfs = base_labels.find(new_frame.video, new_frame.frame_idx)
            merged_instances = None

            # If the base doesn't have a frame corresponding this new
            # frame, then it can be merged cleanly.
            if not base_lfs:
                base_labels.labeled_frames.append(new_frame)
                merged_instances = new_frame.instances
            else:
                # There's a corresponding frame in the base labels,
                # so try merging the data.
                (
                    merged_instances,
                    extra_base_frame,
                    extra_new_frame,
                ) = cls.complex_frame_merge(base_lfs[0], new_frame)
                if extra_base_frame:
                    extra_base.append(extra_base_frame)
                if extra_new_frame:
                    extra_new.append(extra_new_frame)

            if merged_instances:
                if new_frame.video not in merged:
                    merged[new_frame.video] = dict()
                merged[new_frame.video][new_frame.frame_idx] = merged_instances
        return merged, extra_base, extra_new

    @classmethod
    def complex_frame_merge(
        cls, base_frame: "LabeledFrame", new_frame: "LabeledFrame"
    ) -> Tuple[List[Instance], List[Instance], List[Instance]]:
        """Merge two frames, return conflicts if any.

        A conflict occurs when
        * each frame has Instances which don't perfectly match those
          in the other frame, or
        * each frame has PredictedInstances which don't perfectly match
          those in the other frame.

        Args:
            base_frame: The `LabeledFrame` into which we want to merge.
            new_frame: The `LabeledFrame` from which we want to merge.

        Returns:
            tuple of three items:
            * list of instances that were merged
            * list of conflicting instances from base
            * list of conflicting instances from new
        """
        merged_instances = []
        redundant_instances = []
        extra_base_instances = copy(base_frame.instances)
        extra_new_instances = []

        for new_inst in new_frame:
            redundant = False
            for base_inst in base_frame.instances:
                if new_inst.matches(base_inst):
                    base_inst.frame = None
                    extra_base_instances.remove(base_inst)
                    redundant_instances.append(base_inst)
                    redundant = True
                    continue
            if not redundant:
                new_inst.frame = None
                extra_new_instances.append(new_inst)

        conflict = False
        if extra_base_instances and extra_new_instances:
            base_predictions = list(
                filter(lambda inst: hasattr(inst, "score"), extra_base_instances)
            )
            new_predictions = list(
                filter(lambda inst: hasattr(inst, "score"), extra_new_instances)
            )

            base_has_nonpred = len(extra_base_instances) - len(base_predictions)
            new_has_nonpred = len(extra_new_instances) - len(new_predictions)

            # If they both have some predictions or they both have some
            # non-predictions, then there is a conflict.
            # (Otherwise it's not a conflict since we can cleanly merge
            # all the predicted instances with all the non-predicted.)
            if base_predictions and new_predictions:
                conflict = True
            elif base_has_nonpred and new_has_nonpred:
                conflict = True

        if conflict:
            # Conflict, so update base to just include non-conflicting
            # instances (perfect matches)
            base_frame.instances.clear()
            base_frame.instances.extend(redundant_instances)
        else:
            # No conflict, so include all instances in base
            base_frame.instances.extend(extra_new_instances)
            merged_instances = copy(extra_new_instances)
            extra_base_instances = []
            extra_new_instances = []

        # Construct frames to hold any conflicting instances
        extra_base = (
            cls(
                video=base_frame.video,
                frame_idx=base_frame.frame_idx,
                instances=extra_base_instances,
            )
            if extra_base_instances
            else None
        )

        extra_new = (
            cls(
                video=new_frame.video,
                frame_idx=new_frame.frame_idx,
                instances=extra_new_instances,
            )
            if extra_new_instances
            else None
        )

        return merged_instances, extra_base, extra_new

    @property
    def image(self) -> np.ndarray:
        """Return the image for this frame of shape (height, width, channels)."""
        return self.video.get_frame(self.frame_idx)

    def numpy(self) -> np.ndarray:
        """Return the instances as an array of shape (instances, nodes, 2)."""
        if len(self.instances) > 0:
            return np.stack([inst.numpy() for inst in self.instances], axis=0)
        else:
            return np.full((0, 0, 2), np.nan)

    def plot(self, image: bool = True, scale: float = 1.0):
        """Plot the frame with all instances.

        Args:
            image: If False, only the instances will be plotted without loading the
                original image.
            scale: Relative scaling for the figure.

        Notes:
            See `sleap.nn.viz.plot_img` and `sleap.nn.viz.plot_instances` for more
            plotting options.
        """
        if image:
            sleap.nn.viz.plot_img(self.image, scale=scale)
        sleap.nn.viz.plot_instances(self.instances)

    def plot_predicted(self, image: bool = True, scale: float = 1.0):
        """Plot the frame with all predicted instances.

        Args:
            image: If False, only the instances will be plotted without loading the
                original image.
            scale: Relative scaling for the figure.

        Notes:
            See `sleap.nn.viz.plot_img` and `sleap.nn.viz.plot_instances` for more
            plotting options.
        """
        if image:
            sleap.nn.viz.plot_img(self.image, scale=scale)
        sleap.nn.viz.plot_instances(
            self.predicted_instances,
            color_by_track=(len(self.predicted_instances) > 0)
            and (self.predicted_instances[0].track is not None),
        )

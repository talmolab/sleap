"""
Data table widgets and view models used in GUI app.

Typically you'll need to subclass :py:class:`GenericTableModel` for your data
(unless your data is already a list of dictionaries with keys matching
the columns of the table you want), but you can use :py:class:`GenericTableView`
as is. For example::

    videos_table = GenericTableView(
        state=self.state,
        row_name="video",
        is_activatable=True,
        model=VideosTableModel(items=self.labels.videos, context=self.commands),
        )

"""

from PySide2 import QtCore, QtWidgets, QtGui

import numpy as np
import os

from operator import itemgetter

from typing import Any, Callable, Dict, List, Optional, Type

from sleap.gui.state import GuiState
from sleap.gui.commands import CommandContext
from sleap.gui.color import ColorManager
from sleap.io.dataset import Labels
from sleap.instance import LabeledFrame, Instance
from sleap.skeleton import Skeleton


class GenericTableModel(QtCore.QAbstractTableModel):
    """
    Generic Qt table model to show a list of properties for some items.

    Typically this will be used as base class. Subclasses can implement methods:
        object_to_items: allows conversion from a single object to a list of
            items which correspond to rows of table. for example, a table
            which shows skeleton nodes could implement this method and return
            the list of nodes for skeleton.
        item_to_data: if each item isn't already a dictionary with keys for
            columns of table (i.e., `properties` attribute) and values to show
            in table, then use this method to convert each item to such a dict.

    Note that if you need to convert a single object to a list of dictionaries,
    you can implement both steps in `object_to_items` (and use the default
    implementation of `item_to_data` which doesn't do any conversion), or you
    can implement this in two steps using the two methods. It doesn't make
    much difference which you do.

    For editable table, you must implement `can_set` and `set_item` methods.

    Usually it's simplest to override `properties` in the subclass, rather
    than passing as an init arg.

    Args:
        properties: The list of property names (table columns).
        items: The list of items with said properties (rows).
        context: A command context (required for editable items).
    """

    properties = None
    show_row_numbers: bool = True

    def __init__(
        self,
        items: Optional[list] = None,
        properties: Optional[List[str]] = None,
        context: Optional[CommandContext] = None,
    ):
        super(GenericTableModel, self).__init__()
        self.properties = properties or self.properties or []
        self.context = context

        self.items = items

    def object_to_items(self, item_list):
        """Virtual method, convert object to list of items to show in rows."""
        return item_list

    @property
    def items(self):
        """Gets or sets list of items to show in table."""
        return self._data

    @items.setter
    def items(self, obj):
        if not obj:
            self._data = []
            return

        self.obj = obj
        item_list = self.object_to_items(obj)

        self.beginResetModel()
        if hasattr(self, "item_to_data"):
            self._data = []
            for item in item_list:
                item_data = self.item_to_data(obj, item)
                item_data["_original_item"] = item
                self._data.append(item_data)
        else:
            self._data = item_list
        self.endResetModel()

    @property
    def original_items(self):
        """
        Gets the original items (rather than the dictionary we build from it).
        """
        try:
            return [datum["_original_item"] for datum in self._data]
        except:
            return self._data

    def get_item_color(self, item: Any, key: str):
        """Virtual method, returns color for given item."""
        return None

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method, returns data to show in table."""
        if not index.isValid():
            return None

        idx = index.row()
        key = self.properties[index.column()]

        if idx >= self.rowCount():
            return None

        item = self.items[idx]
        if role == QtCore.Qt.DisplayRole:
            if isinstance(item, dict) and key in item:
                return item[key]

            if hasattr(item, key):
                return getattr(item, key)

        elif role == QtCore.Qt.ForegroundRole:
            return self.get_item_color(self.original_items[idx], key)

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=QtCore.Qt.EditRole):
        """Overrides Qt method, dispatch for settable properties."""
        if role == QtCore.Qt.EditRole:
            item, key = self.get_from_idx(index)

            if self.can_set(item, key):
                self.set_item(item, key, value)
                self.dataChanged.emit(index, index)
                return True

        return False

    def rowCount(self, parent=None):
        """Overrides Qt method, returns number of rows (items)."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Overrides Qt method, returns number of columns (attributes)."""
        return len(self.properties)

    def headerData(
        self, idx: int, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Overrides Qt method, returns column (attribute) names."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                col_str = str(self.properties[idx])
                # use title case if key is lowercase
                if col_str == col_str.lower():
                    return col_str.title()
                # otherwise leave case as is
                return col_str
            elif orientation == QtCore.Qt.Vertical:
                # Add 1 to the row index so that we index from 1 instead of 0
                if self.show_row_numbers:
                    return str(idx + 1)
                return None

        return None

    def sort(
        self,
        column_idx: int,
        order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder,
    ):
        """
        Sorts table by given column and order.

        Correctly sorts numeric string (i.e., "123.45") numerically rather
        than alphabetically. Has logic for correctly sorting video frames by
        video then frame index.
        """
        prop = self.properties[column_idx]
        reverse = order == QtCore.Qt.SortOrder.DescendingOrder

        sort_function = itemgetter(prop)
        if prop in ("video", "frame"):
            if "video" in self.properties and "frame" in self.properties:
                sort_function = itemgetter("video", "frame")

        def string_safe_sort(x):
            sort_val = sort_function(x)
            try:
                return float(sort_val)
            except ValueError:
                return -np.inf
            except TypeError:
                return sort_val

        self.beginResetModel()
        self._data.sort(key=string_safe_sort, reverse=reverse)
        self.endResetModel()

    def get_from_idx(self, index: QtCore.QModelIndex):
        """Gets item from QModelIndex."""
        if not index.isValid():
            return None, None
        item = self.original_items[index.row()]
        key = self.properties[index.column()]
        return item, key

    def can_set(self, item, key):
        """Virtual method, returns whether table cell is editable."""
        return False

    def set_item(self, item, key, value):
        """Virtual method, used to set value for item in table cell."""
        pass

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method, returns whether item is selectable etc."""
        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

        item, key = self.get_from_idx(index)
        if self.can_set(item, key):
            flags |= QtCore.Qt.ItemIsEditable
        return flags


class GenericTableView(QtWidgets.QTableView):
    """
    Qt table view for use with `GenericTableModel` (and subclasses).

    Uses the :py:class:`GuiState` object to keep track of which row/item is
    selected. If the `row_name` attribute is "foo", then a "foo_selected"
    state will be item corresponding to the currently selected row in table
    (and the table will select the row if this state is updated by something
    else). When `is_activatable` is True, then a "foo" state will also be
    set to the item when a row is activated--typically by being double-clicked.
    This state can then be used to trigger something else outside the table.

    Note that by default "selected_" is used for the state key, e.g.,
    "selected_foo", but you can set the `name_prefix` attribute/init arg if
    for some reason you need this to be different. For instance, the table
    of instances in the GUI sets this to "" so that the row for an instance
    is automatically selected when `state["instance"]` is set outside the table.
    """

    row_name: Optional[str] = None
    name_prefix: str = "selected_"
    is_activatable: bool = False
    is_sortable: bool = False

    def __init__(
        self,
        model: QtCore.QAbstractTableModel,
        state: GuiState = None,
        row_name: Optional[str] = None,
        name_prefix: Optional[str] = None,
        is_sortable: bool = False,
        is_activatable: bool = False,
    ):
        super(GenericTableView, self).__init__()

        self.state = state or GuiState()
        self.row_name = row_name or self.row_name
        self.name_prefix = name_prefix if name_prefix is not None else self.name_prefix
        self.is_sortable = is_sortable or self.is_sortable
        self.is_activatable = is_activatable or self.is_activatable

        self.setModel(model)

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSortingEnabled(self.is_sortable)

        self.doubleClicked.connect(self.activateSelected)
        if self.row_name:
            self.state.connect(self.name_prefix + self.row_name, self.selectRowItem)

    def selectionChanged(self, new, old):
        """Custom event handler."""
        super(GenericTableView, self).selectionChanged(new, old)

        if self.row_name:
            item = self.getSelectedRowItem()
            self.state[self.name_prefix + self.row_name] = item

    def activateSelected(self, *args):
        """Activate item currently selected in table.

        "Activate" means that the relevant :py:class:`GuiState` state variable
        is set to the currently selected item.
        """
        if self.is_activatable:
            self.state[self.row_name] = self.getSelectedRowItem()

    def selectRowItem(self, item: Any):
        """Select row corresponding to item.

        If the table model converts items to dictionaries (using `item_to_data`
        method), then `item` argument should be the original item, not the
        converted dict.
        """
        if not item:
            return

        idx = self.model().original_items.index(item)
        table_row_idx = self.model().createIndex(idx, 0)
        self.setCurrentIndex(table_row_idx)

        if self.row_name:
            self.state[self.name_prefix + self.row_name] = item

    def selectRow(self, idx: int):
        """Select row corresponding to index."""
        self.selectRowItem(self.model().original_items[idx])

    def getSelectedRowItem(self) -> Any:
        """Return item corresponding to currently selected row.

        Note that if the table model converts items to dictionaries (using
        `item_to_data` method), then returned item will be the original item,
        not the converted dict.
        """
        idx = self.currentIndex()
        if not idx.isValid():
            return None
        return self.model().original_items[idx.row()]


class VideosTableModel(GenericTableModel):
    properties = ("filename", "frames", "height", "width", "channels")

    def item_to_data(self, obj, item):
        return {key: getattr(item, key) for key in self.properties}


class SkeletonNodesTableModel(GenericTableModel):
    properties = ("name", "symmetry")

    def object_to_items(self, skeleton: Skeleton):
        """Converts given skeleton to list of nodes to show in table."""
        items = skeleton.nodes
        self.skeleton = skeleton
        return items

    def item_to_data(self, obj, item):
        return dict(name=item.name, symmetry=obj.get_symmetry_name(item.name))

    def can_set(self, item, key):
        return True

    def set_item(self, item, key, value):
        if key == "name" and value:
            self.context.setNodeName(skeleton=self.obj, node=item, name=value)
        elif key == "symmetry":
            self.context.setNodeSymmetry(skeleton=self.obj, node=item, symmetry=value)

    def get_item_color(self, item: Any, key: str):
        if self.skeleton:
            color = self.context.app.color_manager.get_item_color(
                item, parent_skeleton=self.skeleton
            )
            return QtGui.QColor(*color)


class SkeletonEdgesTableModel(GenericTableModel):
    """Table model for skeleton edges."""

    properties = ("source", "destination")

    def object_to_items(self, skeleton: Skeleton):
        items = []
        self.skeleton = skeleton
        if hasattr(skeleton, "edges"):
            items = [
                dict(source=edge[0].name, destination=edge[1].name)
                for edge in skeleton.edges
            ]
        return items

    def get_item_color(self, item: Any, key: str):
        if self.skeleton:
            edge_pair = (item["source"], item["destination"])
            color = self.context.app.color_manager.get_item_color(
                edge_pair, parent_skeleton=self.skeleton
            )
            return QtGui.QColor(*color)


class LabeledFrameTableModel(GenericTableModel):
    """Table model for listing instances in labeled frame.

    Allows editing track names.

    Args:
        labeled_frame: `LabeledFrame` to show
        labels: `Labels` datasource
    """

    properties = ("points", "track", "score", "skeleton")

    def object_to_items(self, labeled_frame: LabeledFrame):
        if not labeled_frame:
            return []
        return labeled_frame.instances_to_show

    def item_to_data(self, obj, item):
        instance = item

        points = f"{len(instance.nodes)}/{len(instance.skeleton.nodes)}"
        track_name = instance.track.name if instance.track else ""
        score = ""
        if hasattr(instance, "score"):
            score = str(round(instance.score, 2))

        return dict(
            points=points,
            track=track_name,
            score=score,
            skeleton=instance.skeleton.name,
        )

    def get_item_color(self, item: Any, key: str):
        if key == "track" and item.track is not None:
            track = item.track
            return QtGui.QColor(*self.context.app.color_manager.get_track_color(track))
        return None

    def can_set(self, item, key):
        if key == "track" and item.track is not None:
            return True

    def set_item(self, item, key, value):
        if key == "track":
            self.context.setTrackName(item.track, value)


class SuggestionsTableModel(GenericTableModel):
    properties = ("video", "frame", "group", "labeled", "mean score")

    def item_to_data(self, obj, item):
        labels = self.context.labels
        item_dict = dict()

        item_dict["SuggestionFrame"] = item

        video_string = (
            f"{labels.videos.index(item.video)+1}: "
            f"{os.path.basename(item.video.filename)}"
        )

        item_dict["group"] = str(item.group + 1) if item.group is not None else ""
        item_dict["group_int"] = item.group if item.group is not None else -1
        item_dict["video"] = video_string
        item_dict["frame"] = int(item.frame_idx) + 1  # start at frame 1 rather than 0

        # show how many labeled instances are in this frame
        lf = labels.get((item.video, item.frame_idx))
        val = 0 if lf is None else len(lf.user_instances)
        val = str(val) if val > 0 else ""
        item_dict["labeled"] = val

        # calculate score for frame
        scores = [
            inst.score
            for lf in labels.find(item.video, item.frame_idx)
            for inst in lf
            if hasattr(inst, "score")
        ]
        val = sum(scores) / len(scores) if scores else ""
        item_dict["mean score"] = val

        return item_dict

    def sort(self, column_idx: int, order: QtCore.Qt.SortOrder):
        """Sorts table by given column and order."""
        prop = self.properties[column_idx]
        reverse = order == QtCore.Qt.SortOrder.DescendingOrder

        if prop != "group":
            super(SuggestionsTableModel, self).sort(column_idx, order)
        else:

            if not reverse:
                # Use group_int (int) instead of group (str).
                self.beginResetModel()
                self._data.sort(key=itemgetter("group_int"))
                self.endResetModel()

            else:
                # Instead of a reverse sort order on groups, we'll interleave the
                # items so that we get the earliest item from each group, then the
                # second item from each group, and so on.

                # Make a decorated list of items with positions in group (plus the
                # secondary sort keys: group, video, and frame)
                self._data.sort(key=itemgetter("group_int"))
                decorated_data = []
                last_group = object()
                for item in self._data:
                    if last_group != item["group_int"]:
                        group_i = 0
                    decorated_data.append(
                        (group_i, item["group_int"], item["video"], item["frame"], item)
                    )
                    last_group = item["group_int"]
                    group_i += 1

                # Sort decorated list
                decorated_data.sort()

                # Undecorate the list and update table
                self.beginResetModel()
                self._data = [item for (*_, item) in decorated_data]
                self.endResetModel()

        # Update order in project (so order can be saved and affects what we
        # consider previous/next suggestion for navigation).
        resorted_suggestions = [item["SuggestionFrame"] for item in self._data]
        self.context.labels.set_suggestions(resorted_suggestions)


class SkeletonNodeModel(QtCore.QStringListModel):
    """
    String list model for source/destination nodes of edges.

    Args:
        skeleton: The skeleton for which to list nodes.
        src_node: If given, then we assume that this model is being used for
            edge destination node. Otherwise, we assume that this model is
            being used for an edge source node.
            If given, then this should be function that will return the
            selected edge source node.
    """

    def __init__(self, skeleton: Skeleton, src_node: Callable = None):
        super(SkeletonNodeModel, self).__init__()
        self._src_node = src_node
        self.skeleton = skeleton

    @property
    def skeleton(self):
        """Gets or sets current skeleton."""
        return self._skeleton

    @skeleton.setter
    def skeleton(self, val):
        self.beginResetModel()

        self._skeleton = val
        # if this is a dst node, then determine list based on source node
        if self._src_node is not None:
            self._node_list = self._valid_dst()
        # otherwise, show all nodes for skeleton
        else:
            self._node_list = self.skeleton.node_names

        self.endResetModel()

    def _valid_dst(self):
        # get source node using callback
        src_node = self._src_node()

        def is_valid_dst(node):
            # node cannot be dst of itself
            if node == src_node:
                return False
            # node cannot be dst if it's already dst of this src
            if (src_node, node) in self.skeleton.edge_names:
                return False
            return True

        # Filter down to valid destination nodes
        valid_dst_nodes = list(filter(is_valid_dst, self.skeleton.node_names))

        return valid_dst_nodes

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method, returns data for given row."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            return self._node_list[idx]

        return None

    def rowCount(self, parent):
        """Overrides Qt method, returns number of rows."""
        return len(self._node_list)

    def columnCount(self, parent):
        """Overrides Qt method, returns number of columns (1)."""
        return 1

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method, returns flags (editable etc)."""
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

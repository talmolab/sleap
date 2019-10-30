"""
Data table widgets and view models used in GUI app.
"""

from PySide2 import QtCore, QtWidgets, QtGui

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
    """Generic table model to show a list of properties for some items.

    Args:
        properties: The list of property names (table columns).
        items: The list of items with said properties (rows).
        context: A command context (required for editable items).
    """

    properties = None

    def __init__(
        self,
        items: Optional[list] = None,
        properties: Optional[List[str]] = None,
        context: Optional[CommandContext] = None,
    ):
        super(GenericTableModel, self).__init__()
        self.properties = properties or self.properties or []
        self.context = context

        self.uncached_items = []

        if items is not None:
            self.items = items
        else:
            self._data = []

    def object_to_items(self, item_list):
        """Virtual method, convert object to list of items to show in rows."""
        return item_list

    @property
    def items(self):
        """Gets or sets list of items to show in table."""
        return self._data

    @items.setter
    def items(self, obj):
        self.obj = obj
        item_list = self.object_to_items(obj)
        self.uncached_items = item_list

        self.beginResetModel()
        if hasattr(self, "item_to_data"):
            self._data = []
            for item in item_list:
                item_data = self.item_to_data(obj, item)
                self._data.append(item_data)
        else:
            self._data = item_list
        self.endResetModel()

    @property
    def uncached_items(self):
        """Gets or sets the uncached items."""
        return self._uncached_items

    @uncached_items.setter
    def uncached_items(self, val):
        self._uncached_items = val

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
            return self.get_item_color(self.uncached_items[idx], key)

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=QtCore.Qt.EditRole):
        """Overrides Qt method, dispatch for settable properties."""
        if role == QtCore.Qt.EditRole:
            item, key = self.get_from_idx(index)

            if self.can_set(item, key):
                self.set_item(item, key, value)
                # self.dataChanged.emit(index, index)
                return True

        return False

    def rowCount(self, parent=None):
        """Overrides Qt method, returns number of rows (items)."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Overrides Qt method, returns number of columns (attributes)."""
        return len(self.properties)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Overrides Qt method, returns column (attribute) names."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.properties[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def sort(self, column_idx: int, order: QtCore.Qt.SortOrder):
        """Sorts table by given column and order."""
        prop = self.properties[column_idx]

        sort_function = itemgetter(prop)
        if prop in ("video", "frame"):
            if "video" in self.properties and "frame" in self.properties:
                sort_function = itemgetter("video", "frame")

        reverse = order == QtCore.Qt.SortOrder.DescendingOrder

        self.beginResetModel()
        self._data.sort(key=sort_function, reverse=reverse)
        self.endResetModel()

    def get_from_idx(self, index: QtCore.QModelIndex):
        """Gets item from QModelIndex."""
        if not index.isValid():
            return None, None
        item = self.uncached_items[index.row()]
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
        """Activates item selected in table."""
        if self.is_activatable:
            self.state[self.row_name] = self.getSelectedRowItem()

    def selectRowItem(self, item):
        if not item:
            return

        idx = self.model().uncached_items.index(item)
        table_row_idx = self.model().createIndex(idx, 0)
        self.setCurrentIndex(table_row_idx)

        if self.row_name:
            self.state[self.name_prefix + self.row_name] = item

    def getSelectedRowItem(self):
        idx = self.currentIndex()
        if not idx.isValid():
            return None
        return self.model().uncached_items[idx.row()]


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

    def object_to_items(self, labels: Labels):
        """Converts given skeleton to list of nodes to show in table."""
        return labels.get_suggestions()
        # item_list = []

    def item_to_data(self, obj, item):
        labels = obj
        item_dict = dict()

        video_string = f"{labels.videos.index(item.video)}: {os.path.basename(item.video.filename)}"

        item_dict["group"] = str(item.group)
        item_dict["video"] = video_string
        item_dict["frame"] = int(item.frame_idx) + 1  # start at frame 1 rather than 0

        # show how many labeled instances are in this frame
        val = labels.instance_count(item.video, item.frame_idx)
        val = str(val) if val > 0 else ""
        item_dict["labeled"] = val

        # calculate score for frame
        scores = [
            inst.score
            for lf in labels.find(item.video, item.frame_idx)
            for inst in lf
            if hasattr(inst, "score")
        ]
        val = sum(scores) / len(scores) if scores else None
        item_dict["mean score"] = val

        return item_dict


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

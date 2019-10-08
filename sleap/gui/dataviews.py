"""
Data table widgets and view models used in GUI app.
"""

from PySide2 import QtCore, QtWidgets, QtGui

import os

from operator import itemgetter

from typing import Callable, List, Optional

from sleap.gui.overlays.tracks import TrackColorManager
from sleap.io.dataset import Labels
from sleap.instance import LabeledFrame, Instance
from sleap.skeleton import Skeleton


class VideosTable(QtWidgets.QTableView):
    """Table view widget for listing videos in dataset."""

    def __init__(self, videos: list = []):
        super(VideosTable, self).__init__()

        props = ("filename", "frames", "height", "width", "channels")
        model = GenericTableModel(props, videos, useCache=True)

        self.setModel(model)

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)


class GenericTableModel(QtCore.QAbstractTableModel):
    """Generic table model to show a list of properties for some items.

    Args:
        propList: The list of property names (table columns).
        itemList: The list of items with said properties (rows).
        useCache: Whether to build cache of property values for all items.
    """

    def __init__(
        self,
        propList: List[str],
        itemList: Optional[list] = None,
        useCache: bool = False,
    ):
        super(GenericTableModel, self).__init__()
        self._use_cache = useCache
        self._props = propList

        if itemList is not None:
            self.items = itemList
        else:
            self._data = []

    @property
    def items(self):
        """Gets or sets list of items to show in table."""
        return self._data

    @items.setter
    def items(self, val):
        self.beginResetModel()
        if self._use_cache:
            self._data = []
            for item in val:
                item_data = {key: getattr(item, key) for key in self._props}
                self._data.append(item_data)
        else:
            self._data = val
        self.endResetModel()

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method, returns data to show in table."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            key = self._props[index.column()]

            if idx < self.rowCount():
                item = self.items[idx]

                if isinstance(item, dict) and key in item:
                    return item[key]

                if hasattr(item, key):
                    return getattr(item, key)

        return None

    def rowCount(self, parent=None):
        """Overrides Qt method, returns number of rows (items)."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Overrides Qt method, returns number of columns (attributes)."""
        return len(self._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Overrides Qt method, returns column (attribute) names."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def sort(self, column_idx: int, order: QtCore.Qt.SortOrder):
        """Sorts table by given column and order."""
        prop = self._props[column_idx]

        sort_function = itemgetter(prop)
        if prop in ("video", "frame"):
            if "video" in self._props and "frame" in self._props:
                sort_function = itemgetter("video", "frame")

        reverse = order == QtCore.Qt.SortOrder.DescendingOrder

        self.beginResetModel()
        self._data.sort(key=sort_function, reverse=reverse)
        self.endResetModel()

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method, returns whether item is selectable etc."""
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class SkeletonNodesTable(QtWidgets.QTableView):
    """Table view widget for displaying and editing Skeleton nodes. """

    def __init__(self, skeleton: Skeleton):
        super(SkeletonNodesTable, self).__init__()
        self.setModel(SkeletonNodesTableModel(skeleton))
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)


class SkeletonNodesTableModel(QtCore.QAbstractTableModel):
    """Table model for skeleton nodes."""

    _props = ["name", "symmetry"]

    def __init__(self, skeleton: Skeleton):
        super(SkeletonNodesTableModel, self).__init__()
        self._skeleton = skeleton

    @property
    def skeleton(self):
        """Gets or sets current skeleton."""
        return self._skeleton

    @skeleton.setter
    def skeleton(self, val):
        self.beginResetModel()
        self._skeleton = val
        self.endResetModel()

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method, returns data to show in table."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            node_idx = index.row()
            prop = self._props[index.column()]
            node = self.skeleton.nodes[node_idx]
            node_name = node.name

            if prop == "name":
                return node_name
            elif prop == "symmetry":
                return self.skeleton.get_symmetry_name(node_name)

        return None

    def rowCount(self, parent):
        """Overrides Qt method, returns number of rows."""
        return len(self.skeleton.nodes)

    def columnCount(self, parent):
        """Overrides Qt method, returns number of columns."""
        return len(SkeletonNodesTableModel._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Overrides Qt method, returns column names."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=QtCore.Qt.EditRole):
        """Overrides Qt method, updates skeleton with new data from user."""
        if role == QtCore.Qt.EditRole:
            node_idx = index.row()
            prop = self._props[index.column()]
            node_name = self.skeleton.nodes[node_idx].name
            try:
                if prop == "name":
                    # Change node name (unless empty string)
                    if value:
                        self._skeleton.relabel_node(node_name, value)
                elif prop == "symmetry":
                    if value:
                        self._skeleton.add_symmetry(node_name, value)
                    else:
                        # Value was cleared by user, so delete symmetry
                        symmetric_to = self._skeleton.get_symmetry(node_name)
                        self._skeleton.delete_symmetry(node_name, symmetric_to)

                # send signal that data has changed
                self.dataChanged.emit(index, index)

                return True
            except:
                # TODO: display feedback on error?
                pass

        return False

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method, returns flags (editable etc)."""
        return (
            QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEditable
        )


class SkeletonEdgesTable(QtWidgets.QTableView):
    """Table view widget for skeleton edges."""

    def __init__(self, skeleton: Skeleton):
        super(SkeletonEdgesTable, self).__init__()
        self.setModel(SkeletonEdgesTableModel(skeleton))

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)


class SkeletonEdgesTableModel(GenericTableModel):
    """Table model for skeleton edges.

    Args:
        skeleton: The skeleton to show in table.
    """

    def __init__(self, skeleton: Skeleton):
        props = ("source", "destination")
        super(SkeletonEdgesTableModel, self).__init__(props)
        self.skeleton = skeleton

    @property
    def skeleton(self):
        """Gets or sets current skeleton."""
        return self._skeleton

    @skeleton.setter
    def skeleton(self, val):
        self._skeleton = val
        items = [
            dict(source=edge[0].name, destination=edge[1].name)
            for edge in self._skeleton.edges
        ]
        self.items = items


class LabeledFrameTable(QtWidgets.QTableView):
    """Table view widget for listing instances in labeled frame."""

    selectionChangedSignal = QtCore.Signal(Instance)

    def __init__(self, labeled_frame: LabeledFrame = None, labels: Labels = None):
        super(LabeledFrameTable, self).__init__()
        self.setModel(LabeledFrameTableModel(labeled_frame, labels))
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

    def selectionChanged(self, new, old):
        """Custom event handler, emits selectionChangedSignal signal."""
        super(LabeledFrameTable, self).selectionChanged(new, old)

        instance = None
        if len(new.indexes()):
            row_idx = new.indexes()[0].row()
            try:
                instance = self.model().labeled_frame.instances_to_show[row_idx]
            except:
                # Usually means that there's no labeled_frame
                pass

        self.selectionChangedSignal.emit(instance)


class LabeledFrameTableModel(QtCore.QAbstractTableModel):
    """Table model for listing instances in labeled frame.

    Allows editing track names.

    Args:
        labeled_frame: `LabeledFrame` to show
        labels: `Labels` datasource
    """

    _props = ("points", "track", "score", "skeleton")

    def __init__(self, labeled_frame: LabeledFrame, labels: Labels):
        super(LabeledFrameTableModel, self).__init__()
        self.labels = labels
        self._labeled_frame = labeled_frame

    @property
    def labeled_frame(self):
        """Gets or sets current labeled frame."""
        return self._labeled_frame

    @labeled_frame.setter
    def labeled_frame(self, val):
        self.beginResetModel()
        self._labeled_frame = val
        self.endResetModel()

    @property
    def labels(self):
        """Gets or sets current labels dataset object."""
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val
        self.color_manager = TrackColorManager(val)

    @property
    def color_manager(self):
        """Gets or sets object for determining track colors."""
        return self._color_manager

    @color_manager.setter
    def color_manager(self, val):
        self._color_manager = val

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method, returns data to show in table."""
        if index.isValid():
            idx = index.row()
            prop = self._props[index.column()]

            if len(self.labeled_frame.instances_to_show) > (idx - 1):
                instance = self.labeled_frame.instances_to_show[idx]

                # Cell value
                if role == QtCore.Qt.DisplayRole:
                    if prop == "points":
                        return f"{len(instance.nodes)}/{len(instance.skeleton.nodes)}"
                    elif prop == "track" and instance.track is not None:
                        return instance.track.name
                    elif prop == "skeleton":
                        return instance.skeleton.name
                    elif prop == "score":
                        if hasattr(instance, "score"):
                            return f"{round(instance.score, 2)}"
                        else:
                            return ""

                # Cell color
                elif role == QtCore.Qt.ForegroundRole:
                    if prop == "track" and instance.track is not None:
                        return QtGui.QColor(
                            *self.color_manager.get_color(instance.track)
                        )

        return None

    def rowCount(self, parent):
        """Overrides Qt method, returns number of rows."""
        return (
            len(self.labeled_frame.instances_to_show)
            if self.labeled_frame is not None
            else 0
        )

    def columnCount(self, parent):
        """Overrides Qt method, returns number of columns."""
        return len(LabeledFrameTableModel._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Overrides Qt method, returns column names."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=QtCore.Qt.EditRole):
        """
        Overrides Qt method, sets data in labeled frame from user changes.
        """
        if role == QtCore.Qt.EditRole:
            idx = index.row()
            prop = self._props[index.column()]
            instance = self.labeled_frame.instances_to_show[idx]
            if prop == "track":
                if len(value) > 0:
                    instance.track.name = value

            # send signal that data has changed
            self.dataChanged.emit(index, index)

            return True
        return False

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method, returns flags (editable etc)."""
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.isValid():
            idx = index.row()
            if idx < len(self.labeled_frame.instances_to_show):
                instance = self.labeled_frame.instances_to_show[idx]
                prop = self._props[index.column()]
                if prop == "track" and instance.track is not None:
                    f |= QtCore.Qt.ItemIsEditable
        return f


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


class SuggestionsTable(QtWidgets.QTableView):
    """Table view widget for showing frame suggestions."""

    def __init__(self, labels):
        super(SuggestionsTable, self).__init__()
        self.setModel(SuggestionsTableModel(labels))
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSortingEnabled(True)


class SuggestionsTableModel(GenericTableModel):
    """Table model for showing frame suggestions."""

    def __init__(self, labels):
        props = ("video", "frame", "labeled", "mean score")

        super(SuggestionsTableModel, self).__init__(propList=props)
        self.labels = labels

    @property
    def labels(self):
        """Gets or sets current labels dataset."""
        return self._labels

    @labels.setter
    def labels(self, val):
        self.beginResetModel()

        self._labels = val

        self._data = []
        for video, frame_idx in self.labels.get_suggestions():
            item = dict()

            item[
                "video"
            ] = f"{self.labels.videos.index(video)}: {os.path.basename(video.filename)}"
            item["frame"] = int(frame_idx) + 1  # start at frame 1 rather than 0

            # show how many labeled instances are in this frame
            val = self._labels.instance_count(video, frame_idx)
            val = str(val) if val > 0 else ""
            item["labeled"] = val

            # calculate score for frame
            scores = [
                inst.score
                for lf in self.labels.find(video, frame_idx)
                for inst in lf
                if hasattr(inst, "score")
            ]
            val = sum(scores) / len(scores) if scores else None
            item["mean score"] = val

            self._data.append(item)
        self.endResetModel()


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    labels = Labels.load_json(
        "tests/data/json_format_v2/centered_pair_predictions.json"
    )
    skeleton = labels.labels[0].instances[0].skeleton

    Labels.save_json(labels, "test.json")
    del labels
    labels = Labels.load_json("test.json")

    app = QApplication([])
    # table = SkeletonNodesTable(skeleton)
    # table = SkeletonEdgesTable(skeleton)
    # table = VideosTable(labels.videos)
    table = LabeledFrameTable(labels.labels[0], labels)
    table.show()

    app.exec_()

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

import logging
import os
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from sleap.gui.commands import CommandContext
from sleap.gui.state import (
    GuiState,
    INSTANCE_HIDDEN_KEY,
    VIEW_ONLY_INSTANCE_KEY,
    SHOW_NONVISIBLE_OVERRIDE_KEY,
    QC_DISPLAY_MODE_KEY,
    QC_MODE_MANUAL,
    instance_visible,
    instance_shows_non_visible,
)
from sleap_io.model.skeleton import Skeleton
from sleap_io import Video
from sleap_io import LabeledFrame
from sleap_io.io.video_reading import VideoBackend
from sleap.sleap_io_adaptors.skeleton_utils import get_symmetry_node
from sleap.sleap_io_adaptors.instance_utils import get_nodes_from_instance
from sleap.sleap_io_adaptors.lf_labels_utils import get_instances_to_show

logger = logging.getLogger(__name__)


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
            self.beginResetModel()
            self._data = []
            self.endResetModel()
            return

        self.obj = obj
        item_list = self.object_to_items(obj)

        self.beginResetModel()
        try:
            if hasattr(self, "item_to_data"):
                self._data = []
                for item in item_list:
                    try:
                        item_data = self.item_to_data(obj, item)
                    except Exception as e:
                        logger.warning("Skipping unreadable item in table: %s", e)
                        continue
                    item_data["_original_item"] = item
                    self._data.append(item_data)
            else:
                self._data = item_list
        finally:
            # Always end the reset, even if building a row raised, so the Qt
            # model is never left in a half-reset state (which blanks the table).
            self.endResetModel()

    @property
    def original_items(self):
        """
        Gets the original items (rather than the dictionary we build from it).
        """
        try:
            return [datum["_original_item"] for datum in self._data]
        except Exception:
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
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if isinstance(item, dict) and key in item:
                return item[key]

            if hasattr(item, key):
                return getattr(item, key)

        elif role == QtCore.Qt.ForegroundRole:
            return self.get_item_color(self.original_items[idx], key)

        elif role == QtCore.Qt.ToolTipRole:
            if isinstance(item, dict) and key in item:
                return item[key]

            if hasattr(item, key):
                return getattr(item, key)

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=QtCore.Qt.EditRole):
        """Overrides Qt method, dispatch for settable properties."""
        if role == QtCore.Qt.EditRole:
            item, key = self.get_from_idx(index)

            # If nothing changed of the item, return true. (Issue #1013)
            if isinstance(item, dict):
                item_value = item.get(key, None)
            elif hasattr(item, key):
                item_value = getattr(item, key)
            else:
                item_value = None

            if (item_value is not None) and (item_value == value):
                return True

            # Otherwise set the item
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

    "ellipsis_left" can be used to make the TableView truncate cell content on
    the left instead of the right side. By default, the argument is set to
    False, i.e. truncation on the right side, which is also the default for
    QTableView.
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
        ellipsis_left: bool = False,
        multiple_selection: bool = False,
    ):
        super(GenericTableView, self).__init__()

        self.state = state or GuiState()
        self.row_name = row_name or self.row_name
        self.name_prefix = name_prefix if name_prefix is not None else self.name_prefix
        self.is_sortable = is_sortable or self.is_sortable
        self.is_activatable = is_activatable or self.is_activatable
        self.multiple_selection = multiple_selection

        self.setModel(model)

        if ellipsis_left:
            self.setTextElideMode(QtCore.Qt.ElideLeft)
            self.setWordWrap(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        if self.multiple_selection:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
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

        if self.multiple_selection:
            idx_temp = set([x.row() for x in self.selectedIndexes()])
            self.state[f"selected_batch_{self.row_name}"] = idx_temp

        if not idx.isValid():
            return None
        return self.model().original_items[idx.row()]


class InstancesTableView(GenericTableView):
    """Instances table with shift/ctrl multi-select for Merge Instance.

    Selecting a second instance (shift- or ctrl-click) marks it as the merge
    *donor*: the first-selected instance is the survivor (``state["instance"]``,
    kept) and the second is the donor (``state["merge_partner"]``, merged in and
    removed). Single selection behaves exactly like the base view. Selection
    order is tracked explicitly because Qt's "current" index follows the last
    click, which would otherwise make the donor (2nd) the survivor.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("multiple_selection", True)
        super().__init__(*args, **kwargs)
        self._merge_order: List[Any] = []
        self._syncing = False

    def selectionChanged(self, new, old):
        """Set survivor/donor GuiState from the click-ordered selection."""
        # Do the visual update via QTableView; we set GuiState ourselves (the
        # base GenericTableView would set state["instance"] to the last-clicked).
        QtWidgets.QTableView.selectionChanged(self, new, old)

        # selectionChanged can fire during super().__init__() (setModel), before
        # our attributes exist; nothing is selected yet, so just bail.
        if not hasattr(self, "_merge_order"):
            return

        original_items = self.model().original_items
        selected, rows = [], set()
        for qidx in self.selectedIndexes():
            row = qidx.row()
            if row in rows or row >= len(original_items):
                continue
            rows.add(row)
            selected.append(original_items[row])

        # Preserve prior click order, drop deselected, append newly selected.
        selected_ids = {id(i) for i in selected}
        self._merge_order = [i for i in self._merge_order if id(i) in selected_ids]
        known = {id(i) for i in self._merge_order}
        for inst in selected:
            if id(inst) not in known:
                self._merge_order.append(inst)
                known.add(id(inst))

        survivor = self._merge_order[0] if self._merge_order else None
        donor = self._merge_order[1] if len(self._merge_order) >= 2 else None

        # Setting state["instance"] re-enters selectRowItem (connected); guard so
        # it can't collapse the multi-selection back to a single row.
        self._syncing = True
        try:
            self.state["instance"] = survivor
        finally:
            self._syncing = False
        self.state["merge_partner"] = donor

        if self.row_name:
            self.state[f"selected_batch_{self.row_name}"] = set(rows)

    def selectRowItem(self, item: Any):
        """Skip while syncing so our own state set doesn't clear the selection."""
        if getattr(self, "_syncing", False):
            return
        super().selectRowItem(item)


class VideosTableModel(GenericTableModel):
    properties = (
        "name",
        "filepath",
        "frames",
        "height",
        "width",
        "channels",
    )

    def item_to_data(self, obj, item: "VideoBackend"):
        data = {}
        video = item if isinstance(item, Video) else None
        if video is not None:
            item = video.backend

        # sleap-io leaves `Video.backend` as None when the file is inaccessible at
        # open time (a lock, an AV scan, a flaky/slow drive, or a still-flushing
        # recording). The `Video` still carries its filename and cached shape, so
        # render a best-effort row from that instead of dereferencing a None
        # backend and crashing the whole table (#2794). This extends the #2742
        # hardening below, which only guarded the per-frame `img_shape` read.
        if item is None:
            return self._row_from_unopened_video(video)

        # `img_shape` reads a frame from disk, which can fail intermittently for a
        # video on a flaky network drive or a truncated file. Read it once (instead
        # of three times) and fall back to a placeholder so a single unreadable
        # video doesn't blank the entire table (see discussion #2742).
        try:
            img_shape = item.img_shape
        except Exception:
            img_shape = None

        for property in self.properties:
            if property == "name":
                data[property] = (
                    Path(item.filename).name
                    if isinstance(item.filename, str)
                    else item.filename[0]
                )
            elif property == "filepath":
                data[property] = (
                    str(Path(item.filename).parent)
                    if isinstance(item.filename, str)
                    else item.filename[0]
                )
            elif property == "height":
                data[property] = img_shape[0] if img_shape is not None else "?"
            elif property == "width":
                data[property] = img_shape[1] if img_shape is not None else "?"
            elif property == "channels":
                data[property] = img_shape[2] if img_shape is not None else "?"
            else:
                data[property] = getattr(item, property)
        return data

    def _row_from_unopened_video(self, video: "Video | None") -> dict:
        """Build a videos-table row for a `Video` whose backend failed to open.

        Falls back to the `Video`'s filename and cached
        ``backend_metadata["shape"]`` (``[frames, height, width, channels]``); any
        field that can't be resolved is shown as ``"?"``. See #2794.
        """
        filename = video.filename if video is not None else None
        shape = (
            (video.backend_metadata or {}).get("shape") if video is not None else None
        )
        if not (isinstance(shape, (list, tuple)) and len(shape) >= 4):
            shape = None

        if isinstance(filename, str):
            name, filepath = Path(filename).name, str(Path(filename).parent)
        elif filename:  # list of filenames (e.g. an image sequence)
            name = filepath = filename[0]
        else:
            name = filepath = "?"

        values = {
            "name": name,
            "filepath": filepath,
            "frames": shape[0] if shape is not None else "?",
            "height": shape[1] if shape is not None else "?",
            "width": shape[2] if shape is not None else "?",
            "channels": shape[3] if shape is not None else "?",
        }
        return {prop: values.get(prop, "?") for prop in self.properties}


class SkeletonNodesTableModel(GenericTableModel):
    properties = ("name", "symmetry")

    def object_to_items(self, skeleton: Skeleton):
        """Converts given skeleton to list of nodes to show in table."""
        items = skeleton.nodes
        self.skeleton = skeleton
        return items

    def item_to_data(self, obj, item):
        return dict(name=item.name, symmetry=get_symmetry_node(obj, item.name))

    def can_set(self, item, key):
        return True

    def set_item(self, item, key, value):
        if key == "name" and value:
            self.context.setNodeName(skeleton=self.obj, node=item, name=value)
        elif key == "symmetry":
            self.context.setNodeSymmetry(skeleton=self.obj, node=item, symmetry=value)


class NodeSymmetryComboDelegate(QtWidgets.QStyledItemDelegate):
    """Combo-box editor for the skeleton Nodes table "symmetry" column.

    Constrains the symmetric partner to an existing node name (or blank to
    clear the symmetry) instead of a free-text field. Besides being easier to
    use, this prevents a typo from silently creating a phantom node: setting a
    symmetry routes through ``Skeleton.add_symmetry``, which adds any unknown
    partner name as a brand new node.
    """

    def createEditor(self, parent, option, index):
        """Return a combo box listing the other node names (plus a blank)."""
        combo = QtWidgets.QComboBox(parent)
        combo.addItem("")  # blank clears the symmetry
        model = index.model()
        skeleton = getattr(model, "skeleton", None)
        names = list(skeleton.node_names) if skeleton is not None else []
        # A node can't be symmetric with itself, so drop this row's own node.
        own_name = model.data(model.index(index.row(), 0), QtCore.Qt.EditRole)
        combo.addItems([name for name in names if name != own_name])
        return combo

    def setEditorData(self, editor, index):
        """Select the current symmetric partner in the combo box."""
        value = index.model().data(index, QtCore.Qt.EditRole) or ""
        pos = editor.findText(value)
        editor.setCurrentIndex(pos if pos >= 0 else 0)

    def setModelData(self, editor, model, index):
        """Write the chosen node name (or blank) back through the model."""
        model.setData(index, editor.currentText(), QtCore.Qt.EditRole)


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


class LabeledFrameTableModel(GenericTableModel):
    """Table model for listing instances in labeled frame.

    Allows editing track names.

    In addition to the informational columns, three checkbox columns control
    per-instance rendering on the video canvas (transient, session-only state;
    see `sleap.gui.state.instance_visible` /
    `sleap.gui.state.instance_shows_non_visible`):

    - "visibility": checked by default; unchecking hides that instance on the
      canvas while keeping its row in the table.
    - "view only": unchecked by default; checking exactly one row makes only
      that instance visible and disables the whole visibility column. Checking
      another row's "view only" auto-unchecks the previous (radio-like).
      Toggling any "visibility" box exits view-only mode.
    - "invisible nodes": per-instance override of the global "Show Non-Visible
      Nodes" flag; defaults to the effective value (override if present, else
      the global flag). Because non-visible nodes are baked into the canvas
      instance at creation time, toggling this forces a full replot.

    Args:
        labeled_frame: `LabeledFrame` to show
        labels: `Labels` datasource
    """

    # The checkbox columns are appended LAST so existing name-indexed
    # lookups (e.g. ``properties.index("mean node score")``) stay valid.
    VISIBILITY_KEY = "visibility"
    VIEW_ONLY_KEY = "view only"
    SHOW_NONVISIBLE_KEY = "invisible nodes"

    properties = (
        "points",
        "track",
        "score",
        "mean node score",
        "skeleton",
        VISIBILITY_KEY,
        VIEW_ONLY_KEY,
        SHOW_NONVISIBLE_KEY,
    )

    def object_to_items(self, labeled_frame: LabeledFrame):
        if not labeled_frame:
            return []
        return get_instances_to_show(labeled_frame)

    def item_to_data(self, obj, item):
        instance = item

        points = (
            f"{len(get_nodes_from_instance(instance))}/{len(instance.skeleton.nodes)}"
        )
        track_name = instance.track.name if instance.track else ""
        score = ""
        if hasattr(instance, "score"):
            score = str(round(instance.score, 2))

        mean_node_score = ""
        pts = getattr(instance, "points", None)
        if pts is not None and getattr(pts, "dtype", None) is not None:
            names = pts.dtype.names or ()
            if "score" in names and "xy" in names:
                # Visibility = non-NaN xy (matches sleap-nn's filter definition
                # and the "Points" column above).
                visible = ~np.isnan(pts["xy"]).any(axis=1)
                visible_scores = pts["score"][visible]
                visible_scores = visible_scores[~np.isnan(visible_scores)]
                if visible_scores.size > 0:
                    mean_node_score = f"{float(np.mean(visible_scores)):.2f}"

        return dict(
            points=points,
            track=track_name,
            score=score,
            **{"mean node score": mean_node_score},
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

    # -- Per-instance visibility checkbox columns -------------------------------

    @property
    def _checkbox_keys(self):
        return (self.VISIBILITY_KEY, self.VIEW_ONLY_KEY, self.SHOW_NONVISIBLE_KEY)

    @property
    def _vis_state(self) -> GuiState:
        """The `GuiState` holding the transient visibility keys.

        Uses the command context's `GuiState` (shared with the app and the
        instance overlay) when available. Falls back to a private `GuiState`
        when there is no context (e.g. unit tests that build the model without
        a `MainWindow`), so the checkbox columns still work in isolation.
        """
        if self.context is not None and self.context.state is not None:
            return self.context.state
        if getattr(self, "_local_vis_state", None) is None:
            self._local_vis_state = GuiState()
        return self._local_vis_state

    def _apply_canvas_visibility(self):
        """Re-apply effective per-instance visibility to the canvas, if any.

        Iterates the live `QtInstance` objects on the player and shows/hides
        each according to `instance_visible`. No-op when there is no player
        (headless/unit-test paths), since the overlay re-applies on the next
        replot regardless.
        """
        app = getattr(self.context, "app", None) if self.context else None
        player = getattr(app, "player", None)
        if player is None:
            return
        state = self._vis_state
        for qt_inst in player.view.all_instances:
            qt_inst.setVisible(instance_visible(state, qt_inst.instance))

    def is_visibility_checked(self, instance) -> bool:
        """Whether the "visibility" box is checked for the given instance."""
        hidden = self._vis_state.get(INSTANCE_HIDDEN_KEY, default=None)
        return not hidden or id(instance) not in hidden

    def is_view_only_checked(self, instance) -> bool:
        """Whether the "view only" box is checked for the given instance."""
        view_only = self._vis_state.get(VIEW_ONLY_INSTANCE_KEY, default=None)
        return view_only is not None and id(instance) == view_only

    def is_show_nonvisible_checked(self, instance) -> bool:
        """Whether the "invisible nodes" box is checked (effective value)."""
        global_default = self._vis_state.get("show non-visible nodes", default=True)
        return instance_shows_non_visible(self._vis_state, instance, global_default)

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Overrides Qt method to add checkbox state for the new columns."""
        if not index.isValid():
            return None

        key = self.properties[index.column()]
        if key in self._checkbox_keys:
            if role == QtCore.Qt.CheckStateRole:
                instance = self.original_items[index.row()]
                if key == self.VISIBILITY_KEY:
                    checked = self.is_visibility_checked(instance)
                elif key == self.VIEW_ONLY_KEY:
                    checked = self.is_view_only_checked(instance)
                else:
                    checked = self.is_show_nonvisible_checked(instance)
                return QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
            if (
                role == QtCore.Qt.BackgroundRole
                and key == self.VISIBILITY_KEY
                and self._vis_state.get(VIEW_ONLY_INSTANCE_KEY, default=None)
                is not None
            ):
                # Greyed-but-clickable: tint the visibility column while a
                # view-only instance is active to show it is overridden. The
                # cell stays checkable (see `flags`) so clicking it exits
                # view-only mode.
                return QtGui.QColor(128, 128, 128, 64)
            # No text/color/tooltip for the checkbox columns otherwise.
            return None

        return super().data(index, role)

    def flags(self, index: QtCore.QModelIndex):
        """Overrides Qt method to make the new columns user-checkable.

        Both checkbox columns stay enabled and user-checkable at all times.
        During view-only mode the "visibility" column is rendered greyed (see
        `data`) to signal it is overridden, but it remains clickable so that
        toggling any visibility box exits view-only mode (per the feature spec).
        """
        key = self.properties[index.column()]
        if key not in self._checkbox_keys:
            return super().flags(index)

        return (
            QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsUserCheckable
        )

    def setData(self, index: QtCore.QModelIndex, value, role=QtCore.Qt.EditRole):
        """Overrides Qt method to toggle the per-instance visibility state."""
        if index.isValid() and role == QtCore.Qt.CheckStateRole:
            key = self.properties[index.column()]
            if key in self._checkbox_keys:
                instance = self.original_items[index.row()]
                # ``value`` may be a plain int (PySide6 passes 0/2 from the view)
                # or a ``Qt.CheckState`` enum (e.g. our tests, or other
                # bindings). ``Qt.CheckState`` is a non-int Enum in PySide6, so
                # normalize via its ``.value`` before comparing (mirrors the
                # convention in sleap/gui/dialogs/qc.py:230).
                checked = getattr(value, "value", value) == QtCore.Qt.Checked.value
                # A manual per-instance visibility edit means the user is taking
                # over from any active Label QC display mode (#2783) -- which would
                # otherwise overwrite this edit on its next recompute. Drop back to
                # "manual" first so the edit sticks (no-op if already manual; in
                # manual mode this clears the mode-driven hides, giving a clean
                # slate the edit below then applies on top of).
                self._vis_state[QC_DISPLAY_MODE_KEY] = QC_MODE_MANUAL
                if key == self.VISIBILITY_KEY:
                    self._set_visibility(instance, checked)
                elif key == self.VIEW_ONLY_KEY:
                    self._set_view_only(instance, checked)
                else:
                    self._set_show_nonvisible(instance, checked)
                return True

        return super().setData(index, value, role)

    def _set_visibility(self, instance, checked: bool):
        """Toggle an instance's visibility box (also exits view-only mode)."""
        state = self._vis_state
        # Spec: clicking any visibility box exits view-only mode.
        state[VIEW_ONLY_INSTANCE_KEY] = None

        hidden = set(state.get(INSTANCE_HIDDEN_KEY, default=None) or set())
        if checked:
            hidden.discard(id(instance))
        else:
            hidden.add(id(instance))
        state[INSTANCE_HIDDEN_KEY] = hidden

        self._refresh_after_toggle()

    def _set_view_only(self, instance, checked: bool):
        """Toggle an instance's view-only box (radio-like exclusivity)."""
        state = self._vis_state
        if checked:
            # Overwriting auto-unchecks any previously selected view-only row.
            state[VIEW_ONLY_INSTANCE_KEY] = id(instance)
        elif state.get(VIEW_ONLY_INSTANCE_KEY, default=None) == id(instance):
            state[VIEW_ONLY_INSTANCE_KEY] = None

        self._refresh_after_toggle()

    def _set_show_nonvisible(self, instance, checked: bool):
        """Toggle an instance's "invisible nodes" override (forces a replot)."""
        state = self._vis_state
        override = dict(state.get(SHOW_NONVISIBLE_OVERRIDE_KEY, default=None) or {})
        override[id(instance)] = checked  # explicit True/False, never popped
        state[SHOW_NONVISIBLE_OVERRIDE_KEY] = override
        self._replot_after_show_nonvisible_toggle()

    def _refresh_after_toggle(self):
        """Apply the new state to the canvas and repaint the whole table.

        The full-table ``dataChanged`` re-queries both `data` (checkbox states)
        and `flags` (so the visibility column greys out / re-enables when
        view-only mode changes).
        """
        self._apply_canvas_visibility()
        rows = self.rowCount()
        cols = self.columnCount()
        if rows and cols:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(rows - 1, cols - 1),
            )

    def _replot_after_show_nonvisible_toggle(self):
        """Full replot after an "invisible nodes" toggle, then repaint the table.

        ``show_non_visible`` is baked into the canvas `QtInstance` at creation
        time, so ``setVisible`` cannot resurrect node/edge children that were
        never built -> a full replot is required (unlike the visibility/view-only
        columns, which only ``setVisible``). The ``getattr`` guard makes this a
        no-op headless / in unit tests (``context is None``), like
        `_apply_canvas_visibility`.
        """
        app = getattr(self.context, "app", None) if self.context else None
        if app is not None:
            app.plotFrame()
        rows = self.rowCount()
        cols = self.columnCount()
        if rows and cols:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(rows - 1, cols - 1),
            )


class SuggestionsTableModel(GenericTableModel):
    properties = ("video", "frame", "group", "labeled", "mean score")

    def item_to_data(self, obj, item):
        labels = self.context.labels
        item_dict = dict()

        item_dict["SuggestionFrame"] = item

        video_idx = labels.videos.index(item.video) + 1
        video_name = os.path.basename(item.video.filename)
        video_string = f"{video_idx}: {video_name}"

        item_dict["group"] = "0"
        item_dict["group_int"] = 0
        item_dict["video"] = video_string
        item_dict["frame"] = int(item.frame_idx) + 1  # start at frame 1 rather than 0

        # show how many labeled instances are in this frame
        lf = labels.find(item.video, item.frame_idx)
        lf = lf[0] if lf else None
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
        val = float(sum(scores) / len(scores)) if scores else ""
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
        self.context.labels.suggestions = resorted_suggestions


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

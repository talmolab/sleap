"""
Gui for merging two labels files with options to resolve conflicts.
"""


import logging
from typing import Dict, List, Optional

import attr
from qtpy import QtWidgets, QtCore

from sleap.instance import LabeledFrame
from sleap.io.dataset import Labels

USE_BASE_STRING = "Use base, discard conflicting new instances"
USE_NEW_STRING = "Use new, discard conflicting base instances"
USE_NEITHER_STRING = "Discard all conflicting instances"
CLEAN_STRING = "Accept clean merge"

log = logging.getLogger(__name__)


class MergeDialog(QtWidgets.QDialog):
    """
    Dialog window for complex merging of two SLEAP datasets.

    This will immediately merge any labeled frames that can be cleanly merged,
    show summary of merge and prompt user about how to handle merge conflict,
    and then finish merge (resolving conflicts as the user requested).
    """

    def __init__(self, base_labels: Labels, new_labels: Labels, *args, **kwargs):
        """
        Creates merge dialog and begins merging.

        Args:
            base_labels: The base dataset into which we're inserting data.
            new_labels: New dataset from which we're getting data to insert.

        Returns:
            None.
        """

        super(MergeDialog, self).__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()

        self.base_labels = base_labels
        self.new_labels = new_labels

        if self.base_labels.skeleton.node_names != self.new_labels.skeleton.node_names:
            # Warn about mismatching skeletons
            base_nodes = self.base_labels.skeleton.node_names
            merge_nodes = self.new_labels.skeleton.node_names
            missing_nodes = [node for node in base_nodes if node not in merge_nodes]
            new_nodes = [node for node in merge_nodes if node not in base_nodes]
            layout.addWidget(
                QtWidgets.QLabel(
                    "<p><strong>Warning:</strong> Skeletons do not match. "
                    "The following nodes will be added to all instances:<p>"
                    f"<p><em>From base labels</em>: {','.join(missing_nodes)}<br>"
                    f"<em>From new labels</em>: {','.join(new_nodes)}</p>"
                    "<p>Nodes can be deleted or merged from the skeleton editor after "
                    "merging labels.</p><br>"
                )
            )

        merged, self.extra_base, self.extra_new = Labels.complex_merge_between(
            self.base_labels, self.new_labels
        )

        merge_total = 0
        merge_frames = 0
        for vid_frame_list in merged.values():
            # number of frames for this video
            merge_frames += len(vid_frame_list.keys())
            # number of instances across frames for this video
            merge_total += sum((map(len, vid_frame_list.values())))

        merged_text = f"Cleanly merged {merge_total} instances"
        if merge_total:
            merged_text += f" across {merge_frames} frames"
        merged_text += "."
        merged_label = QtWidgets.QLabel(merged_text)
        layout.addWidget(merged_label)

        if merge_total:
            merge_table = MergeTable(merged)
            layout.addWidget(merge_table)

        if not self.extra_base:
            conflict_text = "There are no conflicts."
        else:
            conflict_text = "Merge conflicts:"

        conflict_label = QtWidgets.QLabel(conflict_text)
        layout.addWidget(conflict_label)

        if self.extra_base:
            conflict_table = ConflictTable(
                self.base_labels, self.extra_base, self.extra_new
            )
            layout.addWidget(conflict_table)

        self.merge_method = QtWidgets.QComboBox()
        if self.extra_base:
            self.merge_method.addItem(USE_NEW_STRING)
            self.merge_method.addItem(USE_BASE_STRING)
            self.merge_method.addItem(USE_NEITHER_STRING)
        else:
            self.merge_method.addItem(CLEAN_STRING)
        layout.addWidget(self.merge_method)

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Finish Merge", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.finishMerge)

        layout.addWidget(buttons)

        self.setLayout(layout)

    def finishMerge(self):
        """
        Finishes merge process, possibly resolving conflicts.

        This is connected to `accepted` signal.

        Args:
            None.

        Raises:
            ValueError: If no valid merge method was selected in dialog.

        Returns:
            None.
        """
        merge_method = self.merge_method.currentText()
        if merge_method == USE_BASE_STRING:
            Labels.finish_complex_merge(self.base_labels, self.extra_base)
        elif merge_method == USE_NEW_STRING:
            Labels.finish_complex_merge(self.base_labels, self.extra_new)
        elif merge_method in (USE_NEITHER_STRING, CLEAN_STRING):
            Labels.finish_complex_merge(self.base_labels, [])
        else:
            raise ValueError("No valid merge method selected.")

        self.accept()


class ConflictTable(QtWidgets.QTableView):
    """
    Qt table view for summarizing merge conflicts.

    Arguments are passed through to the table view object.

    The two lists of `LabeledFrame` objects should be correlated (idx in one will
    match idx of the conflicting frame in other).

    Args:
        base_labels: The base dataset.
        extra_base: `LabeledFrame` objects from base that conflicted.
        extra_new: `LabeledFrame` objects from new dataset that conflicts.
    """

    def __init__(
        self,
        base_labels: Labels,
        extra_base: List[LabeledFrame],
        extra_new: List[LabeledFrame],
    ):
        super(ConflictTable, self).__init__()
        self.setModel(ConflictTableModel(base_labels, extra_base, extra_new))


class ConflictTableModel(QtCore.QAbstractTableModel):
    """Qt table model for summarizing merge conflicts.

    See :class:`ConflictTable`.
    """

    _props = ["video", "frame", "base", "new"]

    def __init__(
        self,
        base_labels: Labels,
        extra_base: List[LabeledFrame],
        extra_new: List[LabeledFrame],
    ):
        super(ConflictTableModel, self).__init__()
        self.base_labels = base_labels
        self.extra_base = extra_base
        self.extra_new = extra_new

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            prop = self._props[index.column()]

            if idx < self.rowCount():
                if prop == "video":
                    return self.extra_base[idx].video.filename
                if prop == "frame":
                    return self.extra_base[idx].frame_idx
                if prop == "base":
                    return show_instance_type_counts(self.extra_base[idx])
                if prop == "new":
                    return show_instance_type_counts(self.extra_new[idx])

        return None

    def rowCount(self, *args):
        """Required by Qt."""
        return len(self.extra_base)

    def columnCount(self, *args):
        """Required by Qt."""
        return len(self._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None


class MergeTable(QtWidgets.QTableView):
    """
    Qt table view for summarizing cleanly merged frames.

    Arguments are passed through to the table view object.

    Args:
        merged: The frames that were cleanly merged.
            See :meth:`Labels.complex_merge_between` for details.
    """

    def __init__(self, merged, *args, **kwargs):
        super(MergeTable, self).__init__()
        self.setModel(MergeTableModel(merged))


class MergeTableModel(QtCore.QAbstractTableModel):
    """Qt table model for summarizing merge conflicts.

    See :class:`MergeTable`.
    """

    _props = ["video", "frame", "merged instances"]

    def __init__(self, merged: Dict["Video", Dict[int, List["Instance"]]]):
        super(MergeTableModel, self).__init__()
        self.merged = merged

        self.data_table = []
        for video in self.merged.keys():
            for frame_idx, frame_instance_list in self.merged[video].items():
                self.data_table.append(
                    dict(
                        filename=video.filename,
                        frame_idx=frame_idx,
                        instances=frame_instance_list,
                    )
                )

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            prop = self._props[index.column()]

            if idx < self.rowCount():
                if prop == "video":
                    return self.data_table[idx]["filename"]
                if prop == "frame":
                    return self.data_table[idx]["frame_idx"]
                if prop == "merged instances":
                    return show_instance_type_counts(self.data_table[idx]["instances"])

        return None

    def rowCount(self, *args):
        """Required by Qt."""
        return len(self.data_table)

    def columnCount(self, *args):
        """Required by Qt."""
        return len(self._props)

    def headerData(
        self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole
    ):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None


class ReplaceSkeletonTableDialog(QtWidgets.QDialog):
    """Qt dialog for handling skeleton replacement.

    Args:
        rename_nodes: The nodes that will be renamed.
        delete_nodes: The nodes that will be deleted.
        add_nodes: The nodes that will be added.
        skeleton_nodes: The nodes in the current skeleton.
        new_skeleton_nodes: The nodes in the new skeleton.

    Attributes:
        results_data: The results of the dialog. This is a dictionary with the keys
            being the new node names and the values being the old node names.
        delete_nodes: The nodes that will be deleted.
        add_nodes: The nodes that will be added.
        table: The table widget that displays the nodes.

    Methods:
        add_combo_boxes_to_table: Add combo boxes to the table.
        find_unused_nodes: Find unused nodes.
        create_combo_box: Create a combo box.
        get_table_data: Get the data from the table.
        accept: Accept the dialog.
        result: Get the result of the dialog.

    Returns:
        If accepted, returns a dictionary with the keys being the new node names and the values being the
        old node names. If rejected, returns None.
    """

    def __init__(
        self,
        rename_nodes: List[str],
        delete_nodes: List[str],
        add_nodes: List[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # The only data we need
        self.rename_nodes = rename_nodes
        self.delete_nodes = delete_nodes
        self.add_nodes = add_nodes

        # We want the skeleton nodes to be ordered with rename nodes first
        self.skeleton_nodes = list(self.rename_nodes)
        self.skeleton_nodes.extend(self.delete_nodes)
        self.new_skeleton_nodes = list(self.rename_nodes)
        self.new_skeleton_nodes.extend(self.add_nodes)

        self.results_data: Optional[Dict[str, str]] = None

        # Set table name
        self.setWindowTitle("Replace Nodes")

        # Add table to dialog (if any nodes exist to link)
        if (len(self.add_nodes) > 0) or (len(self.delete_nodes) > 0):
            self.create_table()
        else:
            self.table = None

        # Add table and message to application
        layout = QtWidgets.QVBoxLayout(self)

        # Dynamically create message
        message = "<p><b>Warning:</b> Pre-existing skeleton found."
        if len(self.delete_nodes) > 0:
            message += (
                "<p>The following nodes will be <b>deleted</b> from all instances:"
                f"<br><em>From base labels</em>: {', '.join(self.delete_nodes)}<br></p>"
            )
        else:
            message += "<p>No nodes will be deleted.</p>"
        if len(self.add_nodes) > 0:
            message += (
                "<p>The following nodes will be <b>added</b> to all instances:<br>"
                f"<em>From new labels</em>: {', '.join(self.add_nodes)}</p>"
            )
        else:
            message += "<p>No nodes will be added.</p>"
        if self.table is not None:
            message += (
                "<p>Old nodes to can be linked to new nodes via the table below.</p>"
            )

        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        if self.table is not None:
            layout.addWidget(self.table)

        # Add button to application
        button = QtWidgets.QPushButton("Replace")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

        # Set layout (otherwise nothing will be shown)
        self.setLayout(layout)

    def create_table(self: "ReplaceSkeletonTableDialog") -> QtWidgets.QTableWidget:
        """Create the table widget."""

        self.table = QtWidgets.QTableWidget(self)

        if self.table is None:
            return

        # Create QTable Widget to display skeleton differences
        self.table.setColumnCount(2)
        self.table.setRowCount(len(self.new_skeleton_nodes))
        self.table.setHorizontalHeaderLabels(["New", "Old"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)

        # Add data to table
        column = 0
        for i, new_node in enumerate(self.new_skeleton_nodes):
            row = i
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(new_node))
        self.add_combo_boxes_to_table(init=True)

    def add_combo_boxes_to_table(
        self: "ReplaceSkeletonTableDialog",
        init: bool = False,
    ):
        """Adds combo boxes to table.

        Args:
            init: If True, the combo boxes will be initialized with all
                `self.delete_nodes`. If False, the combo boxes will be initialized with
                all `self.delete_nodes` excluding nodes that have already been used by
                other combo boxes.
        """
        if self.table is None:
            return

        for i in range(self.table.rowCount()):
            # Get text from table item in column 1
            new_node_name = self.table.item(i, 0).text()
            if init and (new_node_name in self.rename_nodes):
                current_combo_text = new_node_name
            else:
                current_combo = self.table.cellWidget(i, 1)
                current_combo_text = (
                    current_combo.currentText() if current_combo else ""
                )
            self.table.setCellWidget(
                i,
                1,
                self.create_combo_box(set_text=current_combo_text, init=init),
            )

    def find_unused_nodes(self: "ReplaceSkeletonTableDialog"):
        """Finds set of nodes from `delete_nodes` that are not used by combo boxes.

        Returns:
            List of unused nodes.
        """
        if self.table is None:
            return

        unused_nodes = set(self.skeleton_nodes)
        for i in range(self.table.rowCount()):
            combo = self.table.cellWidget(i, 1)
            if combo is None:
                break
            elif combo.currentText() in unused_nodes:
                unused_nodes.remove(combo.currentText())
        return list(unused_nodes)

    def create_combo_box(
        self: "ReplaceSkeletonTableDialog",
        set_text: str = "",
        init: bool = False,
    ):
        """Creates combo box with unused nodes from `delete_nodes`.

        Args:
            set_text: Text to set combo box to.
            init: If True, the combo boxes will be initialized with all
                `self.delete_nodes`. If False, the combo boxes will be initialized with
                all `self.delete_nodes` excluding nodes that have already been used by
                other combo boxes.

        Returns:
            Combo box with unused nodes from `delete_nodes` plus an empty string and the
            `set_text`.
        """
        unused_nodes = self.delete_nodes if init else self.find_unused_nodes()
        combo = QtWidgets.QComboBox()
        combo.addItem("")
        if set_text != "":
            combo.addItem(set_text)
        combo.addItems(sorted(unused_nodes))
        combo.setCurrentText(set_text)  # Set text to current text
        combo.currentTextChanged.connect(
            lambda: self.add_combo_boxes_to_table(init=False)
        )
        return combo

    def get_table_data(self: "ReplaceSkeletonTableDialog"):
        """Gets data from table."""
        if self.table is None:
            return {}

        data = {}
        for i in range(self.table.rowCount()):
            new_node = self.table.item(i, 0).text()
            old_node = self.table.cellWidget(i, 1).currentText()
            if (old_node != "") and (new_node != old_node):
                data[new_node] = old_node

        # Sort the data s.t. skeleton nodes are renamed to new nodes first
        data = dict(
            sorted(data.items(), key=lambda item: item[0] in self.skeleton_nodes)
        )

        # This case happens if exclusively bipartite match (new) `self.rename_nodes`
        # with set including (old) `self.delete_nodes` and `self.rename_nodes`
        if len(data) > 0:
            first_new_node, first_old_node = list(data.items())[0]
            if first_new_node in self.skeleton_nodes:
                # Reordering has failed!
                log.debug(f"Linked nodes (new: old): {data}")
                raise ValueError(
                    f"Cannot rename skeleton node '{first_old_node}' to already existing "
                    f"node '{first_new_node}'. Please rename existing skeleton node "
                    f"'{first_new_node}' manually before linking."
                )
        return data

    def accept(self):
        """Overrides accept method to return table data."""
        try:
            self.results_data = self.get_table_data()
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return  # Allow user to fix error if possible instead of closing dialog
        super().accept()

    def result(self):
        """Overrides result method to return table data."""
        return self.get_table_data() if self.results_data is None else self.results_data


def show_instance_type_counts(instance_list: List["Instance"]) -> str:
    """
    Returns string of instance counts to show in table.

    Args:
        instance_list: The list of instances to count.

    Returns:
        String with numbers of user/predicted instances.
    """
    prediction_count = len(
        list(filter(lambda inst: hasattr(inst, "score"), instance_list))
    )
    user_count = len(instance_list) - prediction_count
    return f"{user_count} (user) / {prediction_count} (pred)"

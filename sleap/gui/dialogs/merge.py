"""
Gui for merging two labels files with options to resolve conflicts using sleap-io.
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional

from qtpy import QtWidgets, QtCore

from sleap_io import Instance, Labels

USE_BASE_STRING = "Use base, discard conflicting new instances"
USE_NEW_STRING = "Use new, discard conflicting base instances"
USE_NEITHER_STRING = "Discard all conflicting instances"
CLEAN_STRING = "Accept clean merge"

log = logging.getLogger(__name__)


class MergeDialog(QtWidgets.QDialog):
    """
    Dialog window for merging two SLEAP datasets using sleap-io merge functionality.

    This will attempt to merge datasets and show conflicts if any arise,
    then allow the user to choose how to resolve conflicts.
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
        self.merge_result = None
        self.conflicts = []

        # Check skeleton compatibility
        if self.base_labels.skeleton.node_names != self.new_labels.skeleton.node_names:
            self._add_skeleton_warning(layout)

        # Attempt merge and analyze results
        self._perform_merge_analysis()

        # Build UI based on merge results
        self._build_merge_ui(layout)

        self.setLayout(layout)

    def _add_skeleton_warning(self, layout):
        """Add warning about skeleton mismatches."""
        base_nodes = self.base_labels.skeleton.node_names
        merge_nodes = self.new_labels.skeleton.node_names
        missing_nodes = [node for node in base_nodes if node not in merge_nodes]
        new_nodes = [node for node in merge_nodes if node not in base_nodes]

        warning_text = (
            "<p><strong>Warning:</strong> Skeletons do not match. "
            "The following nodes will be added to all instances:<p>"
            f"<p><em>From base labels</em>: {','.join(missing_nodes)}<br>"
            f"<em>From new labels</em>: {','.join(new_nodes)}</p>"
            "<p>Nodes can be deleted or merged from the skeleton editor after "
            "merging labels.</p><br>"
        )

        layout.addWidget(QtWidgets.QLabel(warning_text))

    def _perform_merge_analysis(self):
        """Perform merge analysis using sleap-io functionality."""
        try:
            # Create a copy for analysis
            base_copy = deepcopy(self.base_labels)

            # Attempt merge with frame strategy
            merge_result = base_copy.merge(
                self.new_labels,
                frame_strategy="keep_both",  # Use sleap-io frame strategy
            )

            # Analyze what was merged vs conflicts
            self._analyze_merge_result(merge_result)

        except Exception as e:
            log.error(f"Error during merge analysis: {e}")
            # Fallback to manual conflict detection
            self._fallback_conflict_detection()

    def _analyze_merge_result(self, merge_result):
        """Analyze the result of the merge operation."""
        # Extract information about what was merged
        self.merge_result = merge_result

        # Count merged frames
        self.frames_merged = (
            merge_result.frames_merged if hasattr(merge_result, "frames_merged") else 0
        )

        # Check for conflicts (frames that couldn't be merged)
        self.conflicts = self._detect_conflicts()

    def _detect_conflicts(self):
        """Detect conflicts between base and new labels."""
        conflicts = []

        for new_frame in self.new_labels.labeled_frames:
            # Check if frame exists in base
            existing_frames = self.base_labels.find(
                new_frame.video,
                new_frame.frame_idx,
            )

            if existing_frames:
                existing_frame = existing_frames[0]
                # Check for instance conflicts
                if self._has_instance_conflicts(existing_frame, new_frame):
                    conflicts.append(
                        {
                            "video": new_frame.video,
                            "frame_idx": new_frame.frame_idx,
                            "base_instances": existing_frame.instances,
                            "new_instances": new_frame.instances,
                        }
                    )

        return conflicts

    def _has_instance_conflicts(self, base_frame, new_frame):
        """Check if there are conflicts between instances in two frames."""
        # Simple conflict detection: if both frames have instances,
        # there might be conflicts
        if len(base_frame.instances) > 0 and len(new_frame.instances) > 0:
            # Check if instances are compatible (same skeleton, etc.)
            return not self._are_instances_compatible(
                base_frame.instances,
                new_frame.instances,
            )
        return False

    def _are_instances_compatible(self, base_instances, new_instances):
        """Check if instances from two frames are compatible for merging."""
        # Basic compatibility check - can be enhanced
        if not base_instances or not new_instances:
            return True

        # Check if skeletons are compatible
        base_skeleton = base_instances[0].skeleton if base_instances else None
        new_skeleton = new_instances[0].skeleton if new_instances else None

        if base_skeleton and new_skeleton:
            return base_skeleton.node_names == new_skeleton.node_names

        return True

    def _fallback_conflict_detection(self):
        """Fallback conflict detection when merge analysis fails."""
        self.frames_merged = 0
        self.conflicts = self._detect_conflicts()

    def _build_merge_ui(self, layout):
        """Build the merge UI based on analysis results."""
        # Show merge summary
        if self.frames_merged > 0:
            merged_text = f"Successfully merged {self.frames_merged} frames."
            merged_label = QtWidgets.QLabel(merged_text)
            layout.addWidget(merged_label)

            # Show merge details table
            merge_table = MergeTable(self.merge_result)
            layout.addWidget(merge_table)
        else:
            merged_label = QtWidgets.QLabel("No frames were automatically merged.")
            layout.addWidget(merged_label)

        # Show conflicts if any
        if self.conflicts:
            conflict_text = f"Found {len(self.conflicts)} merge conflicts:"
            conflict_label = QtWidgets.QLabel(conflict_text)
            layout.addWidget(conflict_label)

            conflict_table = ConflictTable(self.conflicts)
            layout.addWidget(conflict_table)
        else:
            conflict_text = "No merge conflicts detected."
            conflict_label = QtWidgets.QLabel(conflict_text)
            layout.addWidget(conflict_label)

        # Add merge strategy selection
        self.merge_method = QtWidgets.QComboBox()
        if self.conflicts:
            self.merge_method.addItem(USE_NEW_STRING)
            self.merge_method.addItem(USE_BASE_STRING)
            self.merge_method.addItem(USE_NEITHER_STRING)
        else:
            self.merge_method.addItem(CLEAN_STRING)
        layout.addWidget(self.merge_method)

        # Add buttons
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Finish Merge", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.finishMerge)
        layout.addWidget(buttons)

    def finishMerge(self):
        """
        Finishes merge process using sleap-io merge functionality.
        """
        merge_method = self.merge_method.currentText()

        try:
            if merge_method == USE_NEW_STRING:
                # Use new labels, discard conflicting base instances
                self._merge_with_strategy("new")
            elif merge_method == USE_BASE_STRING:
                # Use base labels, discard conflicting new instances
                self._merge_with_strategy("base")
            elif merge_method == USE_NEITHER_STRING:
                # Discard all conflicting instances
                self._merge_with_strategy("neither")
            elif merge_method == CLEAN_STRING:
                # Clean merge - no conflicts
                self._perform_final_merge()
            else:
                raise ValueError("No valid merge method selected.")

            self.accept()

        except Exception as e:
            log.error(f"Error during final merge: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Merge Error", f"An error occurred during the merge: {str(e)}"
            )

    def _merge_with_strategy(self, strategy):
        """Merge using the selected conflict resolution strategy."""
        if strategy == "new":
            # Remove conflicting base instances, then merge
            self._remove_conflicting_instances(self.base_labels)
            self._perform_final_merge()
        elif strategy == "base":
            # Remove conflicting new instances, then merge
            self._remove_conflicting_instances(self.new_labels)
            self._perform_final_merge()
        elif strategy == "neither":
            # Remove all conflicting instances from both
            self._remove_conflicting_instances(self.base_labels)
            self._remove_conflicting_instances(self.new_labels)
            self._perform_final_merge()

    def _remove_conflicting_instances(self, labels):
        """Remove instances that would cause conflicts."""
        from sleap.sleap_io_adaptors.lf_labels_utils import remove_frames

        for conflict in self.conflicts:
            video = conflict["video"]
            frame_idx = conflict["frame_idx"]

            # Find and remove conflicting frames
            frames_to_remove = []
            for frame in labels.labeled_frames:
                if frame.video == video and frame.frame_idx == frame_idx:
                    frames_to_remove.append(frame)

            for frame in frames_to_remove:
                remove_frames(labels, [frame])

    def _perform_final_merge(self):
        """Perform the final merge operation."""
        # Use sleap-io merge with appropriate frame strategy
        self.base_labels.merge(
            self.new_labels,
            frame_strategy="keep_both",  # Adjust based on user preference
        )


class ConflictTable(QtWidgets.QTableView):
    """
    Qt table view for summarizing merge conflicts.
    """

    def __init__(self, conflicts: List[Dict]):
        super(ConflictTable, self).__init__()
        self.setModel(ConflictTableModel(conflicts))


class ConflictTableModel(QtCore.QAbstractTableModel):
    """Qt table model for summarizing merge conflicts."""

    _props = ["video", "frame", "base instances", "new instances"]

    def __init__(self, conflicts: List[Dict]):
        super(ConflictTableModel, self).__init__()
        self.conflicts = conflicts

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            prop = self._props[index.column()]

            if idx < self.rowCount():
                conflict = self.conflicts[idx]
                if prop == "video":
                    return conflict["video"].filename
                elif prop == "frame":
                    return conflict["frame_idx"]
                elif prop == "base instances":
                    return show_instance_type_counts(conflict["base_instances"])
                elif prop == "new instances":
                    return show_instance_type_counts(conflict["new_instances"])

        return None

    def rowCount(self, *args):
        """Required by Qt."""
        return len(self.conflicts)

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
    Qt table view for summarizing merged frames.
    """

    def __init__(self, merge_result):
        super(MergeTable, self).__init__()
        self.setModel(MergeTableModel(merge_result))


class MergeTableModel(QtCore.QAbstractTableModel):
    """Qt table model for summarizing merged frames."""

    _props = [
        "frames merged",
        "instances added",
        "instances updated",
        "instances skipped",
    ]

    def __init__(self, merge_result):
        super(MergeTableModel, self).__init__()
        self.merge_result = merge_result
        self.data_table = self._extract_merge_data()

    def _extract_merge_data(self):
        """Extract merge data from merge result."""
        metadata = {
            "frames_merged": self.merge_result.frames_merged
            if hasattr(self.merge_result, "frames_merged")
            else 0,
            "instances_added": self.merge_result.instances_added
            if hasattr(self.merge_result, "instances_added")
            else 0,
            "instances_updated": self.merge_result.instances_updated
            if hasattr(self.merge_result, "instances_updated")
            else 0,
            "instances_skipped": self.merge_result.instances_skipped
            if hasattr(self.merge_result, "instances_skipped")
            else 0,
        }

        data_table = [metadata]

        return data_table

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
        """Required by Qt."""
        if role == QtCore.Qt.DisplayRole and index.isValid():
            idx = index.row()
            prop = self._props[index.column()]

            if idx < self.rowCount():
                if prop == "frames merged":
                    return self.data_table[idx]["frames_merged"]
                elif prop == "instances added":
                    return self.data_table[idx]["instances_added"]
                elif prop == "instances updated":
                    return self.data_table[idx]["instances_updated"]
                elif prop == "instances skipped":
                    return self.data_table[idx]["instances_skipped"]

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
        If accepted, returns a dictionary with the keys being the new node names and
        the values being the old node names. If rejected, returns None.
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
                    f"Cannot rename skeleton node '{first_old_node}' to already "
                    f"existing node '{first_new_node}'. Please rename existing "
                    f"skeleton node "
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

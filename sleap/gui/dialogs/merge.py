"""
Gui for merging two labels files with options to resolve conflicts.
"""

import attr

from typing import Dict, List

from sleap.instance import LabeledFrame
from sleap.io.dataset import Labels

from PySide2 import QtWidgets, QtCore

USE_BASE_STRING = "Use base, discard conflicting new instances"
USE_NEW_STRING = "Use new, discard conflicting base instances"
USE_NEITHER_STRING = "Discard all conflicting instances"
CLEAN_STRING = "Accept clean merge"


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


if __name__ == "__main__":

    #     file_a = "tests/data/json_format_v1/centered_pair.json"
    #     file_b = "tests/data/json_format_v2/centered_pair_predictions.json"
    # file_a = "files/merge/a.h5"
    # file_b = "files/merge/b.h5"
    file_a = r"sleap_sandbox/skeleton_merge_conflicts/base_labels.slp"
    # file_b = r"sleap_sandbox/skeleton_merge_conflicts/labels.renamed_node.slp")
    # file_b = r"sleap_sandbox/skeleton_merge_conflicts/labels.new_node.slp"
    file_b = r"sleap_sandbox/skeleton_merge_conflicts/labels.deleted_node.slp"

    base_labels = Labels.load_file(file_a)
    new_labels = Labels.load_file(file_b)

    app = QtWidgets.QApplication()
    win = MergeDialog(base_labels, new_labels)
    win.show()
    app.exec_()

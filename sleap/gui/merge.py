"""
Gui for merging two labels files with options to resolve conflicts.
"""

import attr

from typing import List

from sleap.instance import LabeledFrame
from sleap.io.dataset import Labels

from PySide2 import QtWidgets, QtCore

USE_BASE_STRING = "Use base, discard conflicting new instances"
USE_NEW_STRING = "Use new, discard conflicting base instances"
USE_NEITHER_STRING = "Discard all conflicting instances"
CLEAN_STRING = "Accept clean merge"

class MergeDialog(QtWidgets.QDialog):

    def __init__(self,
                 base_labels: Labels,
                 new_labels: Labels,
                 *args, **kwargs):

        super(MergeDialog, self).__init__(*args, **kwargs)

        self.base_labels = base_labels
        self.new_labels = new_labels

        merged, self.extra_base, self.extra_new = \
            Labels.complex_merge_between(self.base_labels, self.new_labels)

        merge_total = 0
        merge_frames = 0
        for vid_frame_list in merged.values():
            # number of frames for this video
            merge_frames += len(vid_frame_list.keys())
            # number of instances across frames for this video
            merge_total += sum((map(len, vid_frame_list.values())))

        layout = QtWidgets.QVBoxLayout()

        merged_text = f"Cleanly merged {merge_total} instances"
        if merge_total:
            merged_text += f" across {merge_frames} frames"
        merged_text += "."
        merged_label = QtWidgets.QLabel(merged_text)
        layout.addWidget(merged_label)

        if merge_total:
            merge_table = MergeTable(merged)
            layout.addWidget(merge_table)

        conflict_text = "There are no conflicts." if not self.extra_base else "Merge conflicts:"
        conflict_label = QtWidgets.QLabel(conflict_text)
        layout.addWidget(conflict_label)

        if self.extra_base:
            conflict_table = ConflictTable(self.base_labels, self.extra_base, self.extra_new)
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
    def __init__(self, *args, **kwargs):
        super(ConflictTable, self).__init__()
        self.setModel(ConflictTableModel(*args, **kwargs))

class ConflictTableModel(QtCore.QAbstractTableModel):
    _props = ["video", "frame", "base", "new"]

    def __init__(self,
                 base_labels: Labels,
                 extra_base: List[LabeledFrame],
                 extra_new: List[LabeledFrame]):
        super(ConflictTableModel, self).__init__()
        self.base_labels = base_labels
        self.extra_base = extra_base
        self.extra_new = extra_new

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
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
        return len(self.extra_base)

    def columnCount(self, *args):
        return len(self._props)

    def headerData(self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None

class MergeTable(QtWidgets.QTableView):
    def __init__(self, *args, **kwargs):
        super(MergeTable, self).__init__()
        self.setModel(MergeTableModel(*args, **kwargs))

class MergeTableModel(QtCore.QAbstractTableModel):
    _props = ["video", "frame", "merged instances"]

    def __init__(self, merged: List[List['Instance']]):
        super(MergeTableModel, self).__init__()
        self.merged = merged

        self.data_table = []
        for video in self.merged.keys():
            for frame_idx, frame_instance_list in self.merged[video].items():
                self.data_table.append(dict(
                    filename=video.filename,
                    frame_idx=frame_idx,
                    instances=frame_instance_list))

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.DisplayRole):
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
        return len(self.data_table)

    def columnCount(self, *args):
        return len(self._props)

    def headerData(self, section, orientation: QtCore.Qt.Orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None

def show_instance_type_counts(instance_list):
    prediction_count = len(list(filter(lambda inst: hasattr(inst, "score"), instance_list)))
    user_count = len(instance_list) - prediction_count
    return f"{prediction_count}/{user_count}"

if __name__ == "__main__":

#     file_a = "tests/data/json_format_v1/centered_pair.json"
#     file_b = "tests/data/json_format_v2/centered_pair_predictions.json"
    file_a = "files/merge/a.h5"
    file_b = "files/merge/b.h5"

    base_labels = Labels.load_file(file_a)
    new_labels = Labels.load_file(file_b)

    app = QtWidgets.QApplication()
    win = MergeDialog(base_labels, new_labels)
    win.show()
    app.exec_()
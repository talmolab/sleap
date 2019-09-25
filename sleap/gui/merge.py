"""
Gui for merging two labels files with options to resolve conflicts.
"""

import attr

from typing import List

from sleap.instance import LabeledFrame
from sleap.io.dataset import Labels

from PySide2 import QtWidgets, QtCore

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
            merge_frames += len(vid_frame_list)
            # number of instances across frames for this video
            merge_total += sum((map(len, vid_frame_list)))

        buttons = self._make_buttons(conflict=self.extra_base)

        merged_label = QtWidgets.QLabel(f"Cleanly merged {merge_total} instances across {merge_frames} frames.")

        conflict_text = "There are no conflicts." if not self.extra_base else "Merge conflicts:"
        conflict_label = QtWidgets.QLabel(conflict_text)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(merged_label)

        layout.addWidget(conflict_label)
        if self.extra_base:
            conflict_table = ConflictTable(self.base_labels, self.extra_base, self.extra_new)
            layout.addWidget(conflict_table)        

        layout.addWidget(buttons)

        self.setLayout(layout)

    def _make_buttons(self, conflict: bool):
        self.use_base_button = None
        self.use_new_button = None
        self.okay_button = None

        buttons = QtWidgets.QDialogButtonBox()
        if conflict:
            self.use_base_button = buttons.addButton("Use Base", QtWidgets.QDialogButtonBox.YesRole)
            self.use_new_button = buttons.addButton("Use New", QtWidgets.QDialogButtonBox.NoRole)
        else:
            self.okay_button = buttons.addButton(QtWidgets.QDialogButtonBox.Ok)

        buttons.clicked.connect(self.finishMerge)

        return buttons

    def finishMerge(self, button):
        if button == self.use_base_button:
            Labels.finish_complex_merge(self.base_labels, self.extra_base)
        elif button == self.use_new_button:
            Labels.finish_complex_merge(self.base_labels, self.extra_new)
        elif button == self.okay_button:
            Labels.finish_complex_merge(self.base_labels, [])

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
                    return self._showInstanceCount(self.extra_base[idx])
                if prop == "new":
                    return self._showInstanceCount(self.extra_new[idx])

        return None

    @staticmethod
    def _showInstanceCount(instance_list):
        prediction_count = len(list(filter(lambda inst: hasattr(inst, "score"), instance_list)))
        user_count = len(instance_list) - prediction_count
        return f"{prediction_count}/{user_count}"

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

if __name__ == "__main__":

    file_a = "tests/data/json_format_v1/centered_pair.json"
    file_b = "tests/data/json_format_v2/centered_pair_predictions.json"

    base_labels = Labels.load_file(file_a)
    new_labels = Labels.load_file(file_b)

    app = QtWidgets.QApplication()
    win = MergeDialog(base_labels, new_labels)
    win.show()
    app.exec_()
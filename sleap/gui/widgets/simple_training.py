"""Simple training configuration widget."""

from typing import Optional, List, Dict

import sleap
from sleap.gui.dataviews import (
    GenericTableModel,
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import FieldComboWidget
from sleap.gui.learning.configs import TrainingConfigsGetter, ConfigFileInfo
from sleap.gui.widgets.models_table import ModelsTableWidget
from sleap.gui.widgets.videos_table import VideosTableWidget

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QWidget,
    QLabel,
    QDialog,
    QGridLayout,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QButtonGroup,
    QListWidget,
    QSpinBox,
    QRadioButton,
    QCheckBox,
    QListWidget,
)

from pathlib import Path


class ProfileSelectorWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        heads = {
            "single_instance": "Single-instance",
            "centroid": "Top-down: Centroid",
            "centered_instance": "Top-down: Centered-instance",
            "multi_instance": "Multi-animal bottom-up",
        }
        def make_head_group(head_filter, label):
            box = QGroupBox(label)
            box.setCheckable(True)
            vb = QVBoxLayout(box)
            grp = ProfileSelectorGroup(head_filter=head_filter)
            vb.addWidget(grp)
            layout.addWidget(box)
            return grp, box

        self.head_groups = {}
        self.head_boxes = {}
        for head_filter, label in heads.items():
            self.head_groups[head_filter], self.head_boxes[head_filter] = make_head_group(head_filter, label)

    def set_visible(self, head_types):
        for head_filter, head_box in self.head_boxes.items():
            head_box.setVisible(head_filter in head_types)


class ProfileSelectorGroup(QWidget):
    def __init__(self, head_filter: Optional[str] = None):
        super().__init__()
        self.cfg_getter = TrainingConfigsGetter.make_from_labels_filename(
            labels_filename=None, head_filter=head_filter
        )

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.list = QListWidget()
        self.desc = QLabel("Desc")

        layout.addWidget(self.list)
        layout.addWidget(self.desc)

        self.update_profiles(force_update=True)

    @property
    def config_infos(self):
        return self.cfg_getter._configs

    @property
    def config_names(self):
        names = []
        for cfg_info in self.config_infos:
            if cfg_info.config.name:
                names.append(cfg_info.config.name)
            elif cfg_info.config.outputs.run_name:
                names.append(cfg_info.config.outputs.run_name)
            elif (not cfg_info.cfg_filename.endswith("initial_config.json")) and (
                not cfg_info.cfg_filename.endswith("training_config.json")
            ):
                names.append(cfg_info.cfg_filename.replace(".json", ""))
            else:
                names.append(Path(cfg_info.folder_path).name)

        return names

    def update_profiles(self, force_update: bool = False):
        n_old = len(self.config_infos)
        self.cfg_getter.update()
        if force_update or len(self.config_infos) != n_old:
            self.list.clear()
            self.list.addItems(self.config_names)


class SimpleTrainingWidget(QWidget):
    def __init__(self, labels: sleap.Labels):
        super().__init__()
        self.labels = labels

        layout = QVBoxLayout(self)

        gb = QGroupBox("1. Select model type")
        hb = QHBoxLayout(gb)
        vb = QVBoxLayout()
        hb.addLayout(vb)

        model_types_description = QLabel("")
        model_types_description.setWordWrap(True)

        def _make_rb(label, desc):
            rb = QRadioButton(label)
            rb.toggled.connect(lambda: model_types_description.setText(desc))
            rb.toggled.connect(lambda: self.update_profiles())
            return rb

        model_types = {
            "Single-instance": (
                "<p>"
                "These models expect there to be a single animal in the frame."
                "</p>"
                "<p>"
                "Multiple animals can be tracked with this model if they are different "
                "in appearance and specified as distinct nodes in the skeleton."
                "</p>"
            ),
            "Top-down multi-instance": (
                "<p>"
                "These models first detect the animals by locating an anchor point, "
                "then detect all body parts within a centered crop around each animal."
                "</p>"
                "<p>"
                "This model type works best when animals are <em>small</em> relative "
                "to the size of the frame."
                "</p>"
            ),
            "Bottom-up multi-instance": (
                "<p>"
                "These models first detect all parts of all animals, then groups them "
                "based on their estimated connectivity."
                "</p>"
                "<p>"
                "This model type works best when animals are <em>large</em> relative "
                "to the size of the frame."
                "</p>"
            ),
        }

        self.model_types = []
        for label, desc in model_types.items():
            self.model_types.append(_make_rb(label, desc))
            vb.addWidget(self.model_types[-1])

        hb.addSpacing(5)
        hb.addWidget(model_types_description)
        layout.addWidget(gb)

        ############

        # 2. Select preset
        gb = QGroupBox("2. Select a preset")
        vb = QVBoxLayout(gb)

        self.profile_selector = ProfileSelectorWidget()
        vb.addWidget(self.profile_selector)

        # Name:
        # name field > filename > run_name if filename is {initial, training}_config.json
        # description
        # customizable fields

        layout.addWidget(gb)

        # model-specific:
        # self.scale = FieldComboWidget()
        # self.scale.set_options(["0.125", "0.25", "0.5", "0.75", "1.0"], select_item="1.0")
        # vb.addWidget(self.scale)
        ############

        # 3. Configure training
        gb = QGroupBox("3. Configure training")
        vb = QVBoxLayout(gb)

        vb.addWidget(QCheckBox("Visualize"))
        vb.addWidget(QCheckBox("Predict on unlabeled"))
        self.augmentation = FieldComboWidget()
        self.augmentation.set_options(["None", "Small", "Full"], select_item="Small")
        # maybe just make this binary? small/large?
        # TODO: scale?
        # TODO: flip
        vb.addWidget(self.augmentation)

        ############

        layout.addWidget(gb)

        self.update_model_types()

    def get_model_type(self) -> Optional[str]:
        for rb in self.model_types:
            if rb.isChecked():
                return rb.text()

    def set_model_type(self, model_type: str):
        for rb in self.model_types:
            rb.setChecked(rb.text() == model_type)

    def update_model_types(self):
        checked = self.get_model_type()
        labels = [rb.text() for rb in self.model_types]
        if self.labels.is_multi_instance:
            for rb in self.model_types:
                if rb.text() == "Single-instance":
                    rb.setEnabled(False)
                else:
                    rb.setEnabled(True)
            if checked is None or checked == "Single-instance":
                self.set_model_type("Top-down multi-instance")
        else:
            for rb in self.model_types:
                if rb.text() == "Single-instance":
                    rb.setEnabled(True)
                    rb.setChecked(True)
                else:
                    rb.setEnabled(False)

    def update_profiles(self):
        model_type = self.get_model_type()

        if model_type == "Single-instance":
            self.profile_selector.set_visible(["single_instance"])
        elif model_type == "Top-down multi-instance":
            self.profile_selector.set_visible(["centroid", "centered_instance"])
        elif model_type == "Bottom-up multi-instance":
            self.profile_selector.set_visible(["multi_instance"])


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    labels = sleap.load_file(
        r"D:\sleap-data\datasets\wt_gold.13pt\sample\labels.v000.slp"
    )

    app = QApplication([])
    window = SimpleTrainingWidget(labels=labels)
    window.show()
    app.exec_()

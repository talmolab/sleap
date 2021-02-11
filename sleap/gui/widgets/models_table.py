"""General purpose SLEAP model selector widget. Useful for prompting user for models."""

from typing import Optional, List, Dict

import sleap
from sleap.gui.dataviews import (
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.configs import TrainingConfigsGetter, ConfigFileInfo

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
)


class ModelsTableModel(GenericCheckableTableModel):
    properties = (
        "Run Name",
        "Model Type",
        "Timestamp",
        "Num Labels",
        "Architecture",
        "Skeleton Nodes",
        "OKS mAP",
        "Dist: 95%",
        "Path",
    )
    sort_as_string = (
        "Run Name",
        "Model Type",
        "Timestamp",
        "Architecture",
        "Path",
    )
    show_row_numbers = False
    row_name = "models"
    sort_by_check_state = False

    def item_to_data(self, obj, cfg: ConfigFileInfo):
        item_data = {
            "Run Name": cfg.config.outputs.run_name,
            "Model Type": cfg.head_name,
            "Timestamp": str(cfg.timestamp),
            "Num Labels": cfg.training_frame_count + cfg.validation_frame_count,
            "Architecture": cfg.config.model.backbone.which_oneof_attrib_name(),
            "Skeleton Nodes": len(cfg.skeleton) if cfg.skeleton is not None else "N/A",
            "OKS mAP": "N/A",
            "Dist: 95%": "N/A",
            "Path": cfg.folder_path,
        }

        metrics = cfg.metrics
        if metrics is not None:
            item_data["OKS mAP"] = f"{metrics['oks_voc.mAP']:.3f}"
            item_data["Dist: 95%"] = f"{metrics['dist.p95']:.1f}"

        return item_data

    def is_enabled(self, item, key):
        """Return whether an item should be enabled based on which other are checked.

        This prevents more than one model of the same type to be checked, and allows
        top-down model pairs to be selected.
        """
        checked_items = self.checked_items
        if len(checked_items) == 0 or self.is_checked(item):
            return True

        checked_types = [item.head_name for item in checked_items]

        if len(checked_types) == 1:
            if item.head_name == "centroid" and "centered_instance" in checked_types:
                return True
            if item.head_name == "centered_instance" and "centroid" in checked_types:
                return True

        return False


class ModelsTableView(GenericTableView):
    row_name = "model"
    is_activatable = True
    is_sortable = True
    resize_mode = "contents"


class ModelsTableWidget(QWidget):
    def __init__(self, models_folder: Optional[str] = None):
        super().__init__()

        if models_folder is None:
            models_folder = ""
        if not isinstance(models_folder, list):
            models_folder = [models_folder]
        self.cfg_getter = TrainingConfigsGetter(models_folder)
        self.cfg_getter.search_depth = 2

        self.table_model = ModelsTableModel()
        self.table_view = ModelsTableView(model=self.table_model)

        layout = QVBoxLayout()
        self.setLayout(layout)

        add_models_button = QPushButton("Add models...")
        add_models_button.clicked.connect(lambda: self.add_models())
        remove_models_button = QPushButton("Remove")
        remove_models_button.clicked.connect(self.table_model.remove_checked)
        remove_models_button.setEnabled(False)
        hl = QHBoxLayout()
        hl.addWidget(add_models_button)
        hl.addWidget(remove_models_button)
        layout.addLayout(hl)

        self.table_model.checked.connect(self.checked_model)
        self.table_model.checked.connect(
            lambda cfgs: remove_models_button.setEnabled(len(cfgs) > 0)
        )
        layout.addWidget(self.table_view)

    @property
    def config_infos(self) -> List[ConfigFileInfo]:
        return self.table_model.original_items

    @property
    def model_paths(self) -> List[str]:
        return [cfg.folder_path for cfg in self.config_infos]

    def add_models(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = FileDialog.openDir(
                None, dir=None, caption="Select model folder..."
            )

        if not model_path:
            return

        if isinstance(model_path, str):
            model_path = [model_path]

        # Search for new configs.
        self.cfg_getter.dir_paths = model_path
        self.cfg_getter.update()
        new_cfgs = self.cfg_getter.get_filtered_configs(only_trained=True)
        model_paths = self.model_paths
        keep_cfgs = [cfg for cfg in new_cfgs if cfg.folder_path not in model_paths]

        # Add ones that we don't already have.
        if len(keep_cfgs) > 0:
            self.table_model.items = self.config_infos + keep_cfgs
            self.table_view.resizeColumnsToContents()

    def checked_model(self, cfgs):
        print("checked:", len(cfgs))

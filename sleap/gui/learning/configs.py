import attr
import cattr
import os

from sleap import Labels, Video
from sleap import util as sleap_utils
from sleap.gui.filedialog import FileDialog
from sleap.nn.config import TrainingJobConfig
from sleap.gui.formbuilder import YamlFormWidget, FieldComboWidget
from sleap.gui.learning import runners, utils

from typing import Any, Callable, Dict, List, Optional, Union, Text

from PySide2 import QtWidgets, QtCore


class TrainingConfigFilesWidget(FieldComboWidget):
    setConfig = QtCore.Signal(TrainingJobConfig)
    # setDataDict = QtCore.Signal(dict)

    SELECT_FILE_OPTION = "Select training config file..."

    def __init__(
        self, cfg_getter: "TrainingConfigsGetter", head_name: Text, *args, **kwargs
    ):
        super(TrainingConfigFilesWidget, self).__init__(*args, **kwargs)
        self._cfg_getter = cfg_getter
        self._cfg_list = []
        self._head_name = head_name
        self._user_config_data_dict = None

        self.update()

        self.currentIndexChanged.connect(self.onSelectionIdxChange)

    def update(self, select: Optional[Dict] = None):
        cfg_list = self._cfg_getter.get_filtered_configs(filter=self._head_name)
        self._cfg_list = cfg_list

        select_key = None

        option_list = []
        option_list.append("")

        # add options for config files
        for cfg_info in cfg_list:
            cfg = cfg_info["config"]
            filename = cfg_info["filename"]

            display_name = f"{cfg.outputs.run_name_prefix or ''}{cfg.outputs.run_name}{cfg.outputs.run_name_suffix or ''} ({filename})"

            if select is not None:
                if select["config"] == cfg_info["config"]:
                    select_key = display_name

            option_list.append(display_name)

        option_list.append("---")
        option_list.append(self.SELECT_FILE_OPTION)

        self.set_options(option_list, select_item=select_key)

    def lastOptionIdx(self):
        return self.count()

    def getConfigByMenuIdx(self, menu_idx):
        cfg_idx = menu_idx - 1
        return (
            self._cfg_list[cfg_idx]["config"] if cfg_idx < len(self._cfg_list) else None
        )

    def onSelectionIdxChange(self, menu_idx):
        if self.value() == self.SELECT_FILE_OPTION:
            cfg_info = self.doFileSelection()

            if cfg_info:
                # We were able to load config from selected file,
                # so add to options and select it.
                self._cfg_getter.insert_first(cfg_info)
                self.update(select=cfg_info)
            else:
                # We couldn't load a valid config, so change menu to initial
                # item since this is "user" config.
                self.setCurrentIndex(0)

        elif menu_idx > 0:
            cfg = self.getConfigByMenuIdx(menu_idx)
            if cfg:
                self.setConfig.emit(cfg)
        elif menu_idx == 0:
            pass
            # if self._user_config_data_dict:
            #     self.setDataDict.emit(self._user_config_data_dict)

    def setUserConfigData(self, cfg_data_dict: Dict[Text, Any]):
        """Sets the user config option from settings made by user."""
        self._user_config_data_dict = cfg_data_dict

        # Select the "user config" option in the combobox menu
        if self.currentIndex() != 0:
            self.onSelectionIdxChange(menu_idx=0)

    def doFileSelection(self):
        """Allow user to add training profile for given model type."""
        filename, _ = FileDialog.open(
            None,
            dir=None,
            caption="Select training configuration file...",
            filter="JSON (*.json)",
        )

        if not filename:
            return None

        return self._cfg_getter.try_loading_path(filename)


@attr.s(auto_attribs=True)
class TrainingConfigsGetter:
    dir_paths: List[Text]
    head_filter: Optional[Text] = None

    def __attrs_post_init__(self):
        self._configs = self.find_configs()

    def find_configs(self):
        configs = []
        for dir in self.dir_paths:
            if os.path.exists(dir):
                configs_in_dir = self.find_configs_in_dir(dir)
                configs.extend(configs_in_dir)

        return configs

    def get_filtered_configs(self, filter: Text):
        return [
            cfg_info for cfg_info in self._configs if cfg_info["head_name"] == filter
        ]

    def insert_first(self, cfg_info: Dict[Text, Any]):
        if (
            "path" not in cfg_info
            or "filename" not in cfg_info
            or "config" not in cfg_info
            or "head_name" not in cfg_info
        ):
            raise ValueError("insert_first needs a cfg_info dict")

        self._configs.insert(0, cfg_info)

    def try_loading_path(self, path: Text):
        try:
            cfg = TrainingJobConfig.load_json(path)
        except Exception as e:
            # Couldn't load so just ignore
            print(e)
            pass
        else:
            # Get the head from the model (i.e., what the model will predict)
            key = cfg.model.heads.which_oneof_attrib_name()

            filename = os.path.basename(path)

            # If filter isn't set or matches head name, add config to list
            if self.head_filter in (None, key):
                cfg_info = {
                    "path": path,
                    "filename": filename,
                    "config": cfg,
                    "head_name": key,
                }
                return cfg_info

        return None

    def find_configs_in_dir(self, dir: Text, depth: int = 1):
        # Find all json files in dir and subdirs to specified depth
        json_files = sleap_utils.find_files_by_suffix(dir, ".json", depth=depth)

        # Sort files, starting with most recently modified
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Get just the paths for the files we found
        json_paths = [file.path for file in json_files]

        configs = []
        for json_path in json_paths:
            cfg_info = self.try_loading_path(json_path)
            if cfg_info:
                configs.append(cfg_info)

        return configs

    @classmethod
    def make_from_labels_filename(
        cls, labels_filename: Text, head_filter: Optional[Text] = None
    ):
        dir_paths = []
        if labels_filename:
            labels_model_dir = os.path.join(os.path.dirname(labels_filename), "models")
            dir_paths.append(labels_model_dir)

        base_config_dir = sleap_utils.get_package_file("sleap/training_profiles")
        dir_paths.append(base_config_dir)

        return cls(dir_paths=dir_paths, head_filter=head_filter)

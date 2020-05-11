import attr
import os

from sleap import util as sleap_utils
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.nn.config import TrainingJobConfig
from sleap.gui.dialogs.formbuilder import FieldComboWidget

from typing import Any, Dict, List, Optional, Text

from PySide2 import QtCore, QtWidgets


@attr.s(auto_attribs=True, slots=True)
class ConfigFileInfo:
    config: TrainingJobConfig
    path: Optional[Text] = None
    filename: Optional[Text] = None
    head_name: Optional[Text] = None
    dont_retrain: bool = False

    @property
    def has_trained_model(self):
        if self.config.outputs.run_name:
            # Check the run path saved in the file
            if self._check_path_for_model(self.config.outputs.run_path):
                return True

            # Check the directory where the file is currently
            if self._check_path_for_model(os.path.dirname(self.path)):
                return True

        return False

    def _check_path_for_model(self, dir):
        # Check if the model file exists.

        # TODO: inference only checks for the best model, so that's also
        #  what we'll do here, but both should check for other models
        #  depending on the training config settings.

        model_shortname = "best_model.h5"
        model_path = os.path.join(dir, model_shortname)

        if os.path.exists(model_path):
            return True

        return False

    @classmethod
    def from_config_file(cls, path):
        cfg = TrainingJobConfig.load_json(path)
        head_name = cfg.model.heads.which_oneof_attrib_name()
        filename = os.path.basename(path)
        return cls(config=cfg, path=path, filename=filename, head_name=head_name)


class TrainingConfigFilesWidget(FieldComboWidget):
    onConfigSelection = QtCore.Signal(ConfigFileInfo)

    SELECT_FILE_OPTION = "Select training config file..."
    SHOW_INITIAL_BLANK = 0

    def __init__(
        self,
        cfg_getter: "TrainingConfigsGetter",
        head_name: Text,
        require_trained: bool = False,
        *args,
        **kwargs,
    ):
        super(TrainingConfigFilesWidget, self).__init__(*args, **kwargs)
        self._cfg_getter = cfg_getter
        self._cfg_list = []
        self._head_name = head_name
        self._require_trained = require_trained
        self._user_config_data_dict = None

        self.currentIndexChanged.connect(self.onSelectionIdxChange)

    def update(self, select: Optional[ConfigFileInfo] = None):
        cfg_list = self._cfg_getter.get_filtered_configs(
            head_filter=self._head_name, only_trained=self._require_trained
        )
        self._cfg_list = cfg_list

        select_key = None

        option_list = []
        if self.SHOW_INITIAL_BLANK or len(cfg_list) == 0:
            option_list.append("")

        # add options for config files
        for cfg_info in cfg_list:
            cfg = cfg_info.config
            filename = cfg_info.filename

            display_name = f"{cfg.outputs.run_name_prefix or ''}{cfg.outputs.run_name}{cfg.outputs.run_name_suffix or ''} ({filename})"

            if cfg_info.has_trained_model:
                display_name += " *"

            if select is not None:
                if select.config == cfg_info.config:
                    select_key = display_name

            option_list.append(display_name)

        option_list.append("---")
        option_list.append(self.SELECT_FILE_OPTION)

        self.set_options(option_list, select_item=select_key)

    def selectByIdx(self, option_idx: int):
        self.setCurrentIndex(option_idx)
        self.onSelectionIdxChange(option_idx)

    def lastOptionIdx(self):
        return self.count()

    @property
    def _menu_cfg_idx_offset(self):
        if not hasattr(self, "options_list"):
            return 0
        if not self.options_list:
            return 0
        if self.options_list[0] == "":
            return 1
        return 0

    def getConfigInfoByMenuIdx(self, menu_idx):
        cfg_idx = menu_idx - self._menu_cfg_idx_offset
        return self._cfg_list[cfg_idx] if 0 <= cfg_idx < len(self._cfg_list) else None

    def getSelectedConfigInfo(self) -> Optional[ConfigFileInfo]:
        current_idx = self.currentIndex()
        return self.getConfigInfoByMenuIdx(current_idx)

    def onSelectionIdxChange(self, menu_idx):
        if self.value() == self.SELECT_FILE_OPTION:
            cfg_info = self.doFileSelection()
            self.addFileSelectionToMenu(cfg_info)

        elif menu_idx >= self._menu_cfg_idx_offset:
            cfg_info = self.getConfigInfoByMenuIdx(menu_idx)
            if cfg_info:
                self.onConfigSelection.emit(cfg_info)

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

    def addFileSelectionToMenu(self, cfg_info: Optional[ConfigFileInfo] = None):
        if cfg_info:
            # We were able to load config from selected file,
            # so add to options and select it.
            self._cfg_getter.insert_first(cfg_info)
            self.update(select=cfg_info)

            if cfg_info.head_name != self._head_name:
                QtWidgets.QMessageBox(
                    text=f"The file you selected was a training config for "
                    f"{cfg_info.head_name} and cannot be used for "
                    f"{self._head_name}."
                ).exec_()
        else:
            # We couldn't load a valid config, so change menu to initial
            # item since this is "user" config.
            self.setCurrentIndex(0)

            QtWidgets.QMessageBox(
                text="The file you selected was not a valid training config."
            ).exec_()


@attr.s(auto_attribs=True)
class TrainingConfigsGetter:
    dir_paths: List[Text]
    head_filter: Optional[Text] = None
    _configs: List[ConfigFileInfo] = attr.ib(default=attr.Factory(list))

    def __attrs_post_init__(self):
        self._configs = self.find_configs()

    def update(self):
        if len(self._configs) == 0:
            self._configs = self.find_configs()
        else:
            current_cfg_paths = {cfg.path for cfg in self._configs}
            new_cfgs = [
                cfg for cfg in self.find_configs() if cfg.path not in current_cfg_paths
            ]
            self._configs = new_cfgs + self._configs

    def find_configs(self):
        configs = []
        for dir in self.dir_paths:
            if os.path.exists(dir):
                configs_in_dir = self.find_configs_in_dir(dir)
                configs.extend(configs_in_dir)

        return configs

    def get_filtered_configs(
        self, head_filter: Text, only_trained: bool = False
    ) -> List[ConfigFileInfo]:

        cfgs_to_return = []
        paths_included = []

        for cfg_info in self._configs:
            if cfg_info.head_name == head_filter:
                if not only_trained or cfg_info.has_trained_model:
                    # At this point we know that config is appropriate
                    # for this head type and is trained if that is required.

                    # We just want a single config from each model directory.
                    # Taking the first config we see in the directory means
                    # we'll get the *trained* config if there is one, since
                    # it will be newer and we've sorted by desc date modified.

                    cfg_dir = os.path.dirname(cfg_info.path)
                    if cfg_dir not in paths_included:
                        paths_included.append(cfg_dir)
                        cfgs_to_return.append(cfg_info)

        return cfgs_to_return

    def get_first(self) -> Optional[ConfigFileInfo]:
        if self._configs:
            return self._configs[0]
        return None

    def insert_first(self, cfg_info: ConfigFileInfo):
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
                return ConfigFileInfo(
                    path=path, filename=filename, config=cfg, head_name=key
                )

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

"""
Find, load, and show lists of saved `TrainingJobConfig`.
"""
import attr
import datetime
import h5py
import os
import re
import numpy as np
from pathlib import Path

from sleap import Labels, Skeleton
from sleap import util as sleap_utils
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.nn.config import TrainingJobConfig
from sleap.gui.dialogs.formbuilder import FieldComboWidget

from typing import Any, Dict, List, Optional, Text

from PySide2 import QtCore, QtWidgets


@attr.s(auto_attribs=True, slots=True)
class ConfigFileInfo:
    """
    Object to represent a saved :py:class:`TrainingJobConfig`

    The :py:class:`TrainingJobConfig` class holds information about the model
    and can be saved as a file. This class holds information about that file,
    e.g., the path, and also provides some properties/methods that make it
    easier to access certain data in or about the file.

    Attributes:
        config: the :py:class:`TrainingJobConfig`
        path: path to the :py:class:`TrainingJobConfig`
        filename: just the filename, not the full path
        head_name: string which should match name of model.heads key
        dont_retrain: allows us to keep track of whether we should retrain
            this config
    """

    config: TrainingJobConfig
    path: Optional[Text] = None
    filename: Optional[Text] = None
    head_name: Optional[Text] = None
    dont_retrain: bool = False
    _skeleton: Optional[Skeleton] = None
    _tried_finding_skeleton: bool = False
    _dset_len_cache: dict = attr.ib(factory=dict)

    @property
    def has_trained_model(self) -> bool:
        # TODO: inference only checks for the best model, so that's also
        #  what we'll do here, but both should check for other models
        #  depending on the training config settings.

        return self._get_file_path("best_model.h5") is not None

    @property
    def path_dir(self):
        return os.path.dirname(self.path) if self.path.endswith("json") else self.path

    def _get_file_path(self, shortname) -> Optional[Text]:
        """
        Check for specified file in various directories related config.

        Args:
            shortname: Filename without path.
        Returns:
            Full path + filename if found, otherwise None.
        """
        if not self.config.outputs.run_name:
            return None

        for dir in [self.config.outputs.run_path, self.path_dir]:
            full_path = os.path.join(dir, shortname)
            if os.path.exists(full_path):
                return full_path

        return None

    @property
    def metrics(self):
        return self._get_metrics("val")

    @property
    def skeleton(self):
        # cache skeleton so we only search once
        if self._skeleton is None and not self._tried_finding_skeleton:

            # if skeleton was saved in config, great!
            if self.config.data.labels.skeletons:
                self._skeleton = self.config.data.labels.skeletons[0]

            # otherwise try loading it from validation labels (much slower!)
            else:
                filename = self._get_file_path(f"labels_gt.val.slp")
                if filename is not None:
                    val_labels = Labels.load_file(filename)
                    if val_labels.skeletons:
                        self._skeleton = val_labels.skeletons[0]

            # don't try loading again (needed in case it's still None)
            self._tried_finding_skeleton = True

        return self._skeleton

    @property
    def training_instance_count(self):
        """Number of instances in the training dataset"""
        return self._get_dataset_len("instances", "train")

    @property
    def validation_instance_count(self):
        """Number of instances in the validation dataset"""
        return self._get_dataset_len("instances", "val")

    @property
    def training_frame_count(self):
        """Number of labeled frames in the training dataset"""
        return self._get_dataset_len("frames", "train")

    @property
    def validation_frame_count(self):
        """Number of labeled frames in the validation dataset"""
        return self._get_dataset_len("frames", "val")

    @property
    def timestamp(self):
        """Timestamp on file; parsed from filename (not OS timestamp)."""
        match = re.match(
            r"(\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)\b", self.config.outputs.run_name
        )
        if match:
            year, month, day = int(match[1]), int(match[2]), int(match[3])
            hour, minute, sec = int(match[4]), int(match[5]), int(match[6])
            return datetime.datetime(2000 + year, month, day, hour, minute, sec)

        return None

    def _get_dataset_len(self, dset_name: Text, split_name: Text):
        cache_key = (dset_name, split_name)
        if cache_key not in self._dset_len_cache:
            n = None
            filename = self._get_file_path(f"labels_gt.{split_name}.slp")
            if filename is not None:
                with h5py.File(filename, "r") as f:
                    n = f[dset_name].shape[0]

            self._dset_len_cache[cache_key] = n

        return self._dset_len_cache[cache_key]

    def _get_metrics(self, split_name: Text):
        metrics_path = self._get_file_path(f"metrics.{split_name}.npz")

        if metrics_path is None:
            return None

        with np.load(metrics_path, allow_pickle=True) as data:
            return data["metrics"].item()

    @classmethod
    def from_config_file(cls, path: Text) -> "ConfigFileInfo":
        cfg = TrainingJobConfig.load_json(path)
        head_name = cfg.model.heads.which_oneof_attrib_name()
        filename = os.path.basename(path)
        return cls(config=cfg, path=path, filename=filename, head_name=head_name)


class TrainingConfigFilesWidget(FieldComboWidget):
    """
    Widget to show list of saved :py:class:`TrainingJobConfig` files.

    This is used inside :py:class:`TrainingEditorWidget`.

    Arguments:
        cfg_getter: the :py:class:`TrainingConfigsGetter` from which menu
            is populated.
        head_name: used to filter configs from `cfg_getter`.
        require_trained: used to filter configs from `cfg_getter`.

    Signals:
        onConfigSelection: triggered when user selects a config file

    """

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
        """Updates menu options, optionally selecting a specific config."""
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

            display_name = ""

            if cfg_info.has_trained_model:
                display_name += "[Trained] "

            display_name += (
                f"{cfg.outputs.run_name_prefix or ''}"
                f"{cfg.outputs.run_name}"
                f"{cfg.outputs.run_name_suffix or ''}"
                f"({filename})"
            )

            if select is not None:
                if select.config == cfg_info.config:
                    select_key = display_name

            option_list.append(display_name)

        option_list.append("---")
        option_list.append(self.SELECT_FILE_OPTION)

        self.set_options(option_list, select_item=select_key)

    @property
    def _menu_cfg_idx_offset(self):
        if (
            hasattr(self, "options_list")
            and self.options_list
            and self.options_list[0] == ""
        ):
            return 1
        return 0

    def getConfigInfoByMenuIdx(self, menu_idx: int) -> Optional[ConfigFileInfo]:
        """Return `ConfigFileInfo` for menu item index."""
        cfg_idx = menu_idx - self._menu_cfg_idx_offset
        return self._cfg_list[cfg_idx] if 0 <= cfg_idx < len(self._cfg_list) else None

    def getSelectedConfigInfo(self) -> Optional[ConfigFileInfo]:
        """
        Return currently selected `ConfigFileInfo` (if any, None otherwise).
        """
        return self.getConfigInfoByMenuIdx(self.currentIndex())

    def onSelectionIdxChange(self, menu_idx: int):
        """
        Handler for when user selects a menu item.

        Either allows selection of config using file browser, or emits
        `onConfigSelection` signal for selected config.
        """
        if self.value() == self.SELECT_FILE_OPTION:
            cfg_info = self.doFileSelection()
            self._add_file_selection_to_menu(cfg_info)

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
        """Shows file browser to add training profile for given model type."""
        filename, _ = FileDialog.open(
            None,
            dir=None,
            caption="Select training configuration file...",
            filter="JSON (*.json)",
        )
        return self._cfg_getter.try_loading_path(filename) if filename else None

    def _add_file_selection_to_menu(self, cfg_info: Optional[ConfigFileInfo] = None):
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
    """
    Searches for and loads :py:class:`TrainingJobConfig` files.

    Attributes:
        dir_paths: List of paths in which to search for
            :py:class:`TrainingJobConfig` files.
        head_filter: Name of head type to use when filtering,
            e.g., "centered_instance".
        search_depth: How many subdirectories deep to search for config files.
    """

    dir_paths: List[Text]
    head_filter: Optional[Text] = None
    search_depth: int = 1
    _configs: List[ConfigFileInfo] = attr.ib(default=attr.Factory(list))

    def __attrs_post_init__(self):
        self._configs = self.find_configs()

    def update(self):
        """Re-searches paths and loads any previously unloaded config files."""
        if len(self._configs) == 0:
            self._configs = self.find_configs()
        else:
            current_cfg_paths = {cfg.path for cfg in self._configs}
            new_cfgs = [
                cfg for cfg in self.find_configs() if cfg.path not in current_cfg_paths
            ]
            self._configs = new_cfgs + self._configs

    def find_configs(self) -> List[ConfigFileInfo]:
        """Load configs from all saved paths."""
        configs = []

        # Collect all configs from specified directories, sorted from most recently modified to least
        for config_dir in filter(lambda d: os.path.exists(d), self.dir_paths):
            # Find all json files in dir and subdirs to specified depth
            json_files = sleap_utils.find_files_by_suffix(
                config_dir, ".json", depth=self.search_depth
            )

            if Path(config_dir).as_posix().endswith("sleap/training_profiles"):
                # Use hardcoded sort.
                BUILTIN_ORDER = [
                    "baseline.centroid.json",
                    "baseline_medium_rf.bottomup.json",
                    "baseline_medium_rf.single.json",
                    "baseline_medium_rf.topdown.json",
                    "baseline_large_rf.bottomup.json",
                    "baseline_large_rf.single.json",
                    "baseline_large_rf.topdown.json",
                    "pretrained.bottomup.json",
                    "pretrained.centroid.json",
                    "pretrained.single.json",
                    "pretrained.topdown.json",
                ]
                json_files.sort(key=lambda f: BUILTIN_ORDER.index(f.name))

            else:
                # Sort files, starting with most recently modified
                json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Load the configs from files
            for json_path in [file.path for file in json_files]:
                cfg_info = self.try_loading_path(json_path)
                if cfg_info:
                    configs.append(cfg_info)

        return configs

    def get_filtered_configs(
        self, head_filter: Text = "", only_trained: bool = False
    ) -> List[ConfigFileInfo]:
        """Returns filtered subset of loaded configs."""

        base_config_dir = os.path.realpath(
            sleap_utils.get_package_file("sleap/training_profiles")
        )

        cfgs_to_return = []
        paths_included = []

        for cfg_info in self._configs:
            if cfg_info.head_name == head_filter or not head_filter:
                if not only_trained or cfg_info.has_trained_model:
                    # At this point we know that config is appropriate
                    # for this head type and is trained if that is required.

                    # We just want a single config from each model directory.
                    # Taking the first config we see in the directory means
                    # we'll get the *trained* config if there is one, since
                    # it will be newer and we've sorted by desc date modified.

                    # TODO: check filenames since timestamp sort could be off
                    #  if files were copied

                    cfg_dir = os.path.realpath(os.path.dirname(cfg_info.path))

                    if cfg_dir == base_config_dir or cfg_dir not in paths_included:
                        paths_included.append(cfg_dir)
                        cfgs_to_return.append(cfg_info)

        return cfgs_to_return

    def get_first(self) -> Optional[ConfigFileInfo]:
        """Get first loaded config."""
        return self._configs[0] if self._configs else None

    def insert_first(self, cfg_info: ConfigFileInfo):
        """Insert config at beginning of list."""
        self._configs.insert(0, cfg_info)

    def try_loading_path(self, path: Text) -> Optional[ConfigFileInfo]:
        """Attempts to load config file and wrap in `ConfigFileInfo` object."""
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

    @classmethod
    def make_from_labels_filename(
        cls, labels_filename: Text, head_filter: Optional[Text] = None
    ) -> "TrainingConfigsGetter":
        """
        Makes object which checks for models in default subdir for dataset.
        """
        dir_paths = []
        if labels_filename:
            labels_model_dir = os.path.join(os.path.dirname(labels_filename), "models")
            dir_paths.append(labels_model_dir)

        base_config_dir = sleap_utils.get_package_file("sleap/training_profiles")
        dir_paths.append(base_config_dir)

        return cls(dir_paths=dir_paths, head_filter=head_filter)

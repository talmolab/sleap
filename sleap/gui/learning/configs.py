"""
Find, load, and show lists of saved `TrainingJobConfig`.
"""

import datetime
import os
import re
import attr
import h5py
import numpy as np
import logging

from qtpy import QtCore, QtWidgets
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple

from sleap_io import Skeleton, load_file
from sleap import util as sleap_utils
from sleap.gui.config_utils import get_head_from_omegaconf, get_skeleton_from_config
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import FieldComboWidget
from omegaconf import OmegaConf
from sleap.util import show_sleap_nn_installation_message
from sleap.gui.learning.load_legacy_metrics import load_npz_extract_arrays

# Try to import rapidyaml for fast config scanning (~500x faster than OmegaConf)
try:
    import ryml

    _HAS_RAPIDYAML = True
except ImportError:
    _HAS_RAPIDYAML = False

logging.basicConfig(level=logging.DEBUG)


def _quick_scan_yaml_metadata(path: Text) -> Tuple[Optional[Text], Optional[Text]]:
    """Quickly extract head_type and run_name from a YAML config file.

    Uses rapidyaml for ~500x faster parsing compared to OmegaConf. Only extracts
    the minimal metadata needed for config list display, deferring full parsing
    until the config is actually selected.

    Args:
        path: Path to the YAML config file.

    Returns:
        Tuple of (head_type, run_name). Either may be None if not found.
    """
    if not _HAS_RAPIDYAML:
        # Fallback to OmegaConf if rapidyaml not available
        try:
            cfg = OmegaConf.load(path)
            head_type = get_head_from_omegaconf(cfg)
            run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
            return head_type, run_name
        except Exception:
            return None, None

    try:
        content = Path(path).read_text()
        tree = ryml.parse_in_arena(content.encode())
        root_id = tree.root_id()

        # Extract head_type from model_config.head_configs
        head_type = None
        model_config_id = tree.find_child(root_id, b"model_config")
        if model_config_id != ryml.NONE:
            head_configs_id = tree.find_child(model_config_id, b"head_configs")
            if head_configs_id != ryml.NONE:
                # Find first non-null head (has children = has config)
                child_id = tree.first_child(head_configs_id)
                while child_id != ryml.NONE:
                    if tree.has_children(child_id):
                        key = tree.key(child_id)
                        if key is not None:
                            head_type = bytes(key).decode()
                            break
                    child_id = tree.next_sibling(child_id)

        # Extract run_name from trainer_config.run_name
        run_name = None
        trainer_config_id = tree.find_child(root_id, b"trainer_config")
        if trainer_config_id != ryml.NONE:
            run_name_id = tree.find_child(trainer_config_id, b"run_name")
            if run_name_id != ryml.NONE and tree.has_val(run_name_id):
                val = tree.val(run_name_id)
                if val is not None:
                    run_name = bytes(val).decode()
                    if run_name in ("null", "~", "None", ""):
                        run_name = None

        return head_type, run_name
    except Exception:
        return None, None


@attr.s(auto_attribs=True, slots=True)
class ConfigFileInfo:
    """
    Object to represent a saved :py:class:`TrainingJobConfig`

    The :py:class:`TrainingJobConfig` class holds information about the model
    and can be saved as a file. This class holds information about that file,
    e.g., the path, and also provides some properties/methods that make it
    easier to access certain data in or about the file.

    Supports lazy loading: config can be None initially (quick metadata scan),
    and will be loaded on first access via the `config` property. This enables
    ~500x faster initial config list population using rapidyaml.

    Attributes:
        _config: the :py:class:`TrainingJobConfig` (lazy-loaded)
        path: path to the :py:class:`TrainingJobConfig`
        filename: just the filename, not the full path
        head_name: string which should match name of model_config.head_configs key
        dont_retrain: allows us to keep track of whether we should retrain
            this config
        _run_name_cache: cached run_name from quick scan (avoids full load)
    """

    _config: Optional[OmegaConf] = None
    path: Optional[Text] = None
    filename: Optional[Text] = None
    head_name: Optional[Text] = None
    dont_retrain: bool = False
    _skeleton: Optional[Skeleton] = None
    _tried_finding_skeleton: bool = False
    _dset_len_cache: dict = attr.ib(factory=dict)
    _run_name_cache: Optional[Text] = None
    _has_trained_model_cache: Optional[bool] = None

    @property
    def config(self) -> OmegaConf:
        """Lazy-load the full config on first access."""
        if self._config is None and self.path is not None:
            self._load_full_config()
        return self._config

    @config.setter
    def config(self, value: OmegaConf):
        """Allow setting config directly (for backward compatibility)."""
        self._config = value

    def _load_full_config(self):
        """Load the full OmegaConf config from the file path."""
        if self.path is None:
            return

        try:
            if self.path.endswith(("yaml", "yml")):
                self._config = OmegaConf.load(self.path)
            else:
                # JSON config - use sleap_nn loader
                from sleap_nn.config.training_job_config import (
                    TrainingJobConfig as snn_TrainingJobConfig,
                )

                self._config = snn_TrainingJobConfig.load_sleap_config(self.path)
        except Exception as e:
            logging.warning(f"Failed to load config from {self.path}: {e}")
            self._config = None

    @property
    def is_loaded(self) -> bool:
        """Check if the full config has been loaded."""
        return self._config is not None

    @property
    def has_trained_model(self) -> bool:
        """Check if this config has a trained model (best.ckpt or best_model.h5).

        This method is optimized to avoid loading the full config. It only checks
        path_dir (the directory containing the config file), which is where
        checkpoints are saved by sleap-nn training. The result is cached.

        Note: This does not check ckpt_dir from the config because:
        1. sleap-nn saves best.ckpt to the model directory (path_dir)
        2. Baseline configs don't have ckpt_dir set
        3. Loading config just for this check is too slow (~30-40ms per file)
        """
        if self._has_trained_model_cache is not None:
            return self._has_trained_model_cache

        # Check path_dir for checkpoint files (no config load needed)
        path_dir = self.path_dir
        for filename in ("best.ckpt", "best_model.h5"):
            if os.path.exists(os.path.join(path_dir, filename)):
                self._has_trained_model_cache = True
                return True

        self._has_trained_model_cache = False
        return False

    @property
    def path_dir(self):
        return (
            os.path.dirname(self.path)
            if (
                self.path.endswith("yaml")
                or self.path.endswith("json")
                or self.path.endswith("yml")
            )
            else self.path
        )

    def _get_file_path(self, shortname) -> Optional[Text]:
        """
        Check for specified file in various directories related config.

        Args:
            shortname: Filename without path.
        Returns:
            Full path + filename if found, otherwise None.
        """
        for dir in [
            OmegaConf.select(self.config, "trainer_config.ckpt_dir", default="."),
            self.path_dir,
        ]:
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
            if self.config.data_config.skeletons:
                skeletons = get_skeleton_from_config(self.config.data_config.skeletons)
                self._skeleton = skeletons[0] if skeletons else None

            # otherwise try loading it from validation labels (much slower!)
            else:
                # Try new naming first, then legacy, then old sleap-nn naming
                filename = (
                    self._get_file_path("labels_gt.val.0.slp")  # sleap-nn v0.1.0+
                    or self._get_file_path("labels_gt.val.slp")  # legacy SLEAP
                    or self._get_file_path("labels_val_gt_0.slp")  # sleap-nn < v0.1.0
                )
                if filename is not None:
                    val_labels = load_file(filename)
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
        timestamp_pattern = r".*?(?<!\d)(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\b"

        # Try cached run_name first (avoids full config load)
        run_name = self._run_name_cache

        # If no cache, try to get run_name from config
        if run_name is None and self._config is not None:
            try:
                # sleap-nn format
                run_name = self._config.trainer_config.run_name
            except (AttributeError, TypeError):
                try:
                    # Legacy SLEAP format (OmegaConf or dict)
                    if hasattr(self._config, "outputs"):
                        run_name = self._config.outputs.run_name
                    elif isinstance(self._config, dict):
                        run_name = self._config.get("outputs", {}).get("run_name")
                except (AttributeError, TypeError):
                    pass

        # Try matching run_name first
        if run_name:
            match = re.match(timestamp_pattern, run_name)
            if match:
                year, month, day = int(match[1]), int(match[2]), int(match[3])
                hour, minute, sec = int(match[4]), int(match[5]), int(match[6])
                return datetime.datetime(2000 + year, month, day, hour, minute, sec)

        # Fallback to parsing from path if run_name doesn't have timestamp
        if self.path:
            match = re.match(timestamp_pattern, self.path)
            if match:
                year, month, day = int(match[1]), int(match[2]), int(match[3])
                hour, minute, sec = int(match[4]), int(match[5]), int(match[6])
                return datetime.datetime(2000 + year, month, day, hour, minute, sec)

        return None

    def _get_dataset_len(self, dset_name: Text, split_name: Text):
        cache_key = (dset_name, split_name)
        if cache_key not in self._dset_len_cache:
            n = None
            # Try new naming first, then legacy, then old sleap-nn naming
            filename = (
                self._get_file_path(f"labels_gt.{split_name}.0.slp")  # v0.1.0+
                or self._get_file_path(f"labels_gt.{split_name}.slp")  # legacy
                or self._get_file_path(f"labels_{split_name}_gt_0.slp")  # < v0.1.0
            )
            if filename is not None:
                with h5py.File(filename, "r") as f:
                    n = f[dset_name].shape[0]

            self._dset_len_cache[cache_key] = n

        return self._dset_len_cache[cache_key]

    def _get_metrics(self, split_name: Text):
        # Try new naming first, then old sleap-nn naming
        metrics_path_nn = (
            self._get_file_path(f"metrics.{split_name}.0.npz")  # v0.1.0+
            or self._get_file_path(f"{split_name}_0_pred_metrics.npz")  # < v0.1.0
        )

        if metrics_path_nn is None:
            # Loading legacy metrics from SLEAP <= v1.4.1
            metrics_path = self._get_file_path(f"metrics.{split_name}.npz")
            if metrics_path is not None:
                metric_data = load_npz_extract_arrays(metrics_path)
                return_dict = {
                    "vis.tp": metric_data.get("metrics[0].vis.tp").item(),
                    "vis.fp": metric_data.get("metrics[0].vis.fp").item(),
                    "vis.tn": metric_data.get("metrics[0].vis.tn").item(),
                    "vis.fn": metric_data.get("metrics[0].vis.fn").item(),
                    "vis.precision": metric_data.get("metrics[0].vis.precision").item(),
                    "vis.recall": metric_data.get("metrics[0].vis.recall").item(),
                    "dist.dists": metric_data.get("metrics[0].dist.dists"),
                    "dist.avg": metric_data.get("metrics[0].dist.avg").item(),
                    "dist.p50": metric_data.get("metrics[0].dist.p50").item(),
                    "dist.p75": metric_data.get("metrics[0].dist.p75").item(),
                    "dist.p90": metric_data.get("metrics[0].dist.p90").item(),
                    "dist.p95": metric_data.get("metrics[0].dist.p95").item(),
                    "dist.p99": metric_data.get("metrics[0].dist.p99").item(),
                    "pck.mPCK": metric_data.get("metrics[0].pck.mPCK").item(),
                    "oks.mOKS": metric_data.get("metrics[0].oks.mOKS").item(),
                    "oks_voc.mAP": metric_data.get("metrics[0].oks_voc.mAP").item(),
                    "oks_voc.mAR": metric_data.get("metrics[0].oks_voc.mAR").item(),
                    "pck_voc.mAP": metric_data.get("metrics[0].pck_voc.mAP").item(),
                    "pck_voc.mAR": metric_data.get("metrics[0].pck_voc.mAR").item(),
                }
                return return_dict

        else:
            metrics_path = metrics_path_nn

            with np.load(metrics_path, allow_pickle=True) as data:
                metric_data = data["metrics"].item()

                return_dict = {
                    "vis.tp": metric_data["visibility_metrics"].get("tp"),
                    "vis.fp": metric_data["visibility_metrics"].get("fp"),
                    "vis.tn": metric_data["visibility_metrics"].get("tn"),
                    "vis.fn": metric_data["visibility_metrics"].get("fn"),
                    "vis.precision": metric_data["visibility_metrics"].get("precision"),
                    "vis.recall": metric_data["visibility_metrics"].get("recall"),
                    "dist.dists": metric_data["distance_metrics"].get("dists"),
                    "dist.avg": metric_data["distance_metrics"].get("avg"),
                    "dist.p50": metric_data["distance_metrics"].get("p50"),
                    "dist.p75": metric_data["distance_metrics"].get("p75"),
                    "dist.p90": metric_data["distance_metrics"].get("p90"),
                    "dist.p95": metric_data["distance_metrics"].get("p95"),
                    "dist.p99": metric_data["distance_metrics"].get("p99"),
                    "pck.mPCK": metric_data["pck_metrics"].get("mPCK"),
                    "oks.mOKS": metric_data["mOKS"].get("mOKS"),
                    "oks_voc.mAP": metric_data["voc_metrics"].get("oks_voc.mAP"),
                    "oks_voc.mAR": metric_data["voc_metrics"].get("oks_voc.mAR"),
                    "pck_voc.mAP": metric_data["voc_metrics"].get("pck_voc.mAP"),
                    "pck_voc.mAR": metric_data["voc_metrics"].get("pck_voc.mAR"),
                }
                return return_dict

    @classmethod
    def from_config_file(cls, path: Text) -> "ConfigFileInfo":
        """Load a config file with full parsing (for user-selected files)."""
        if path.endswith("yaml") or path.endswith("yml"):
            cfg = OmegaConf.load(path)
            run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
            cfg_info = cls(
                path=path,
                filename=os.path.basename(path),
                head_name=get_head_from_omegaconf(cfg),
            )
            cfg_info._config = cfg
            cfg_info._run_name_cache = run_name
            return cfg_info

        else:
            try:
                from sleap_nn.config.training_job_config import (
                    TrainingJobConfig as snn_TrainingJobConfig,
                )

                cfg = snn_TrainingJobConfig.load_sleap_config(path)
                head_name = get_head_from_omegaconf(cfg)
                filename = os.path.basename(path)
                run_name = OmegaConf.select(
                    cfg, "trainer_config.run_name", default=None
                )
                cfg_info = cls(
                    path=path,
                    filename=filename,
                    head_name=head_name,
                )
                cfg_info._config = cfg
                cfg_info._run_name_cache = run_name
                return cfg_info
            except ImportError:
                show_sleap_nn_installation_message()
                print(
                    "sleap-nn is not installed. This appears to be GUI-only install."
                    "To enable training, please install SLEAP with the 'nn' dependency."
                    "See the installation guide: https://docs.sleap.ai/latest/installation/"
                )
                return None


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
        """Updates menu options, optionally selecting a specific config.

        This method blocks signals during the update to prevent the expensive
        cascade of config selection events that would otherwise occur when
        set_options() changes the combo box selection.
        """
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
            filename = cfg_info.filename

            display_name = ""

            if cfg_info.has_trained_model:
                display_name += "[Trained] "
                # Use cached run_name to avoid triggering full config load
                run_name = cfg_info._run_name_cache
                if run_name is None and cfg_info._config is not None:
                    run_name = OmegaConf.select(
                        cfg_info._config, "trainer_config.run_name", default=""
                    )
            else:
                display_name += f"[{filename.split('.yaml')[0]}] "
                run_name = ""

            # Normalize run_name: convert None or "None" to empty string
            run_name = "" if run_name is None or run_name == "None" else run_name

            display_name += f"{run_name}({filename})"

            if select is not None:
                # Compare by path to avoid triggering config load
                if select.path == cfg_info.path:
                    select_key = display_name

            option_list.append(display_name)

        option_list.append("---")
        option_list.append(self.SELECT_FILE_OPTION)

        # Block signals to prevent cascade of config selection events.
        # Without this, set_options() triggers currentIndexChanged which causes
        # onSelectionIdxChange() -> onConfigSelection -> _load_config() +
        # update_receptive_field() for each tab during update_file_lists().
        self.blockSignals(True)
        try:
            self.set_options(option_list, select_item=select_key)
        finally:
            self.blockSignals(False)

        # After update completes, trigger config load for the selected item.
        # This ensures the form is populated with the selected config's values.
        selected = self.getSelectedConfigInfo()
        if selected is not None:
            self.onConfigSelection.emit(selected)

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
        filters = ["JSON (*.json)", "YAML (*.yaml;*.yml)"]
        filename, _ = FileDialog.open(
            None,
            dir=None,
            caption="Select training configuration file...",
            filter=";;".join(filters),
        )
        logging.debug(f"Selected training config file: {filename}")
        if not filename:
            logging.debug("No file selected for training config.")
            return None
        return self._cfg_getter.try_loading_path(filename)

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
        """Re-searches paths and loads any previously unloaded config files.

        This method is optimized to avoid re-loading already-loaded configs.
        On first call (when _configs is empty), it loads all configs.
        On subsequent calls, it only scans for new file paths and loads those.
        """
        if len(self._configs) == 0:
            self._configs = self.find_configs()
        else:
            # Only load configs for NEW file paths (avoids re-loading all)
            current_cfg_paths = {cfg.path for cfg in self._configs}
            new_file_paths = self._find_config_file_paths() - current_cfg_paths
            if new_file_paths:
                new_cfgs = [self.try_loading_path(p) for p in new_file_paths]
                self._configs = [c for c in new_cfgs if c] + self._configs

    def _find_config_file_paths(self) -> set:
        """Scan directories for config file paths without loading them.

        This is much faster than find_configs() because it only does filesystem
        operations, not YAML/JSON parsing.

        Returns:
            Set of file paths to config files.
        """
        paths = set()
        for config_dir in filter(lambda d: os.path.exists(d), self.dir_paths):
            for suffix in (".json", ".yaml", ".yml"):
                files = sleap_utils.find_files_by_suffix(
                    config_dir, suffix, depth=self.search_depth
                )
                paths.update(f.path for f in files)
        return paths

    def find_configs(self) -> List[ConfigFileInfo]:
        """Load configs from all saved paths."""
        configs = []

        # Collect all configs from specified directories, sorted from most recently
        # modified to least
        for config_dir in filter(lambda d: os.path.exists(d), self.dir_paths):
            # Find all json files in dir and subdirs to specified depth
            json_files = sleap_utils.find_files_by_suffix(
                config_dir, ".json", depth=self.search_depth
            )
            json_files.extend(
                sleap_utils.find_files_by_suffix(
                    config_dir, ".yaml", depth=self.search_depth
                )
            )
            json_files.extend(
                sleap_utils.find_files_by_suffix(
                    config_dir, ".yml", depth=self.search_depth
                )
            )

            if Path(config_dir).as_posix().endswith("sleap/training_profiles"):
                # Use hardcoded sort.
                BUILTIN_ORDER = [
                    "baseline.centroid.yaml",
                    "baseline_medium_rf.bottomup.yaml",
                    "baseline_medium_rf.single.yaml",
                    "baseline_medium_rf.topdown.yaml",
                    "baseline_large_rf.bottomup.yaml",
                    "baseline_large_rf.single.yaml",
                    "baseline_large_rf.topdown.yaml",
                    "baseline.multi_class_bottomup.yaml",
                    "baseline.multi_class_topdown.yaml",
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
            sleap_utils.get_package_file("training_profiles")
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

    def try_loading_path(
        self, path: Text, full_load: bool = False
    ) -> Optional[ConfigFileInfo]:
        """Attempts to load config file and wrap in `ConfigFileInfo` object.

        Args:
            path: Path to the config file.
            full_load: If True, load the full OmegaConf config immediately.
                If False (default), use quick metadata scan for YAML files,
                deferring full load until config is accessed. This provides
                ~500x faster initial loading.

        Returns:
            ConfigFileInfo or None if loading failed.
        """
        if path.endswith("yaml") or path.endswith("yml"):
            # Use quick scan for metadata extraction (~500x faster)
            try:
                head_type, run_name = _quick_scan_yaml_metadata(path)
                filename = os.path.basename(path)

                if head_type is None:
                    # Quick scan failed, skip this file
                    return None

                logging.debug(f"Quick-scanned YAML config file: {filename}")

                # If filter isn't set or matches head name, add config to list
                if self.head_filter in (None, head_type):
                    logging.debug(f"Config matches head filter: {self.head_filter}")

                    # Create ConfigFileInfo with lazy loading (config=None)
                    # Note: Private attrs must be set after construction
                    cfg_info = ConfigFileInfo(
                        path=path,
                        filename=filename,
                        head_name=head_type,
                    )
                    cfg_info._run_name_cache = run_name

                    # Optionally load full config immediately
                    if full_load:
                        cfg_info._load_full_config()

                    return cfg_info
            except Exception:
                # Couldn't load so just ignore
                return None
        else:
            # JSON config - use sleap_nn loader (no quick scan available)
            try:
                from sleap_nn.config.training_job_config import (
                    TrainingJobConfig as snn_TrainingJobConfig,
                )

                cfg = snn_TrainingJobConfig.load_sleap_config(path)
            except ImportError:
                show_sleap_nn_installation_message()
                print(
                    "sleap-nn is not installed. This appears to be a GUI-only install."
                    "To enable training, please install SLEAP with the 'nn' dependency."
                    "See the installation guide: https://docs.sleap.ai/latest/installation/"
                )
                return None
            except Exception as e:
                # Couldn't load so just ignore
                print(f"Couldn't load config from `{path}`: {e}")
                return None

            # Get the head from the model (i.e., what the model will predict)
            key = get_head_from_omegaconf(cfg)
            filename = os.path.basename(path)

            # If filter isn't set or matches head name, add config to list
            if self.head_filter in (None, key):
                cfg_info = ConfigFileInfo(path=path, filename=filename, head_name=key)
                cfg_info._config = cfg
                return cfg_info

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

        base_config_dir = sleap_utils.get_package_file("training_profiles")
        dir_paths.append(base_config_dir)

        return cls(dir_paths=dir_paths, head_filter=head_filter)

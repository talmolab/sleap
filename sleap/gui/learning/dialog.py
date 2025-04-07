"""
Dialogs for running training and/or inference in GUI.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Text, cast

import cattr
from qtpy import QtCore, QtGui, QtWidgets

import sleap
from sleap import Labels, Video
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import YamlFormWidget
from sleap.gui.learning import configs, datagen, receptivefield, runners, scopedkeydict

# List of fields which should show list of skeleton nodes
NODE_LIST_FIELDS = [
    "data.instance_cropping.center_on_part",
    "model.heads.centered_instance.anchor_part",
    "model.heads.centroid.anchor_part",
    "model.heads.multi_class_topdown.confmaps.anchor_part",
]


class LearningDialog(QtWidgets.QDialog):
    """
    Dialog for running training and/or inference.

    The dialog shows tabs for configuring the pipeline (
    :py:class:`TrainingPipelineWidget`) and, depending on the pipeline, for
    each specific model (:py:class:`TrainingEditorWidget`).

    In training mode, the model hyperpameters are editable unless you're using
    a trained model; they are read-only in inference mode.

    Arguments:
        mode: either "training" or "inference".
        labels_filename: path to labels file, used for default location to
            save models.
        labels: the `Labels` object (can also be loaded from given filename)
        skeleton: the `Skeleton` object (can also be taken from `Labels`), used
            for list of nodes for (e.g.) selecting anchor node
    """

    _handle_learning_finished = QtCore.Signal(int)

    def __init__(
        self,
        mode: Text,
        labels_filename: Text,
        labels: Optional[Labels] = None,
        skeleton: Optional["Skeleton"] = None,
        *args,
        **kwargs,
    ):
        super(LearningDialog, self).__init__()

        if labels is None:
            labels = Labels.load_file(labels_filename)

        if skeleton is None and labels.skeletons:
            skeleton = labels.skeletons[0]

        self.mode = mode
        self.labels_filename = labels_filename
        self.labels = labels
        self.skeleton = skeleton

        self._frame_selection = None

        self.current_pipeline = ""

        self.tabs: Dict[str, TrainingEditorWidget] = dict()
        self.shown_tab_names = []

        self._cfg_getter = configs.TrainingConfigsGetter.make_from_labels_filename(
            labels_filename=self.labels_filename
        )

        # Layout for buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.copy_button = buttons.addButton(
            "Copy to clipboard", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_button = buttons.addButton(
            "Save configuration files...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.export_button = buttons.addButton(
            "Export training job package...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.addButton("Run", QtWidgets.QDialogButtonBox.ApplyRole)

        self.copy_button.setToolTip("Copy configuration to the clipboard")
        self.save_button.setToolTip("Save scripts and configuration to run pipeline.")
        self.export_button.setToolTip(
            "Export data, configuration, and scripts for remote training and inference."
        )
        self.run_button.setToolTip("Run pipeline locally (GPU recommended).")

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        self.pipeline_form_widget = TrainingPipelineWidget(mode=mode, skeleton=skeleton)
        if mode == "training":
            tab_label = "Training Pipeline"
        elif mode == "inference":
            # self.pipeline_form_widget = InferencePipelineWidget()
            tab_label = "Inference Pipeline"
        else:
            raise ValueError(f"Invalid LearningDialog mode: {mode}")

        self.tab_widget = QtWidgets.QTabWidget()

        self.tab_widget.addTab(self.pipeline_form_widget, tab_label)
        self.make_tabs()

        self.message_widget = QtWidgets.QLabel("")

        # Layout for entire dialog
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)

        content_layout.addWidget(self.tab_widget)
        content_layout.addWidget(self.message_widget)
        content_layout.addWidget(buttons_layout_widget)

        # Create the QScrollArea.
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)

        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll_area)

        self.adjust_initial_size()

        # Default to most recently trained pipeline (if there is one)
        self.set_default_pipeline_tab()

        # Connect functions to update pipeline tabs when pipeline changes
        self.pipeline_form_widget.updatePipeline.connect(self.set_pipeline)
        self.pipeline_form_widget.emitPipeline()

        self.connect_signals()

        # Connect actions for buttons
        self.copy_button.clicked.connect(self.copy)
        self.save_button.clicked.connect(self.save)
        self.export_button.clicked.connect(self.export_package)
        self.cancel_button.clicked.connect(self.reject)
        self.run_button.clicked.connect(self.run)

        # Connect button for previewing the training data
        if "_view_datagen" in self.pipeline_form_widget.buttons:
            self.pipeline_form_widget.buttons["_view_datagen"].clicked.connect(
                self.view_datagen
            )

    def adjust_initial_size(self):
        # Get screen size
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()

        max_width = 1860
        max_height = 1150
        margin = 0.10

        # Calculate target width and height
        target_width = min(screen.width() - screen.width() * margin, max_width)
        target_height = min(screen.height() - screen.height() * margin, max_height)
        # Set the dialog's dimensions
        self.resize(target_width, target_height)

    def update_file_lists(self):
        self._cfg_getter.update()
        for tab in self.tabs.values():
            tab.update_file_list()

    @staticmethod
    def count_total_frames_for_selection_option(
        videos_frames: Dict[Video, List[int]]
    ) -> int:
        if not videos_frames:
            return 0

        count = 0
        for frame_list in videos_frames.values():
            # Check for [X, Y) range given as (X, -Y) tuple
            if len(frame_list) == 2 and frame_list[1] < 0:
                count += -frame_list[1] - frame_list[0]
            elif frame_list != (0, 0):
                count += len(frame_list)

        return count

    @property
    def frame_selection(self) -> Dict[str, Dict[Video, List[int]]]:
        """
        Returns dictionary with frames that user has selected for learning.
        """
        return self._frame_selection

    @frame_selection.setter
    def frame_selection(self, frame_selection: Dict[str, Dict[Video, List[int]]]):
        """Sets options of frames on which to run learning."""
        self._frame_selection = frame_selection

        if "_predict_frames" in self.pipeline_form_widget.fields.keys():
            prediction_options = []

            total_random = 0
            total_suggestions = 0
            total_user = 0
            random_video = 0
            clip_length = 0
            video_length = 0
            all_videos_length = 0

            # Determine which options are available given _frame_selection
            if "random" in self._frame_selection:
                total_random = self.count_total_frames_for_selection_option(
                    self._frame_selection["random"]
                )
            if "random_video" in self._frame_selection:
                random_video = self.count_total_frames_for_selection_option(
                    self._frame_selection["random_video"]
                )
            if "suggestions" in self._frame_selection:
                total_suggestions = self.count_total_frames_for_selection_option(
                    self._frame_selection["suggestions"]
                )
            if "user" in self._frame_selection:
                total_user = self.count_total_frames_for_selection_option(
                    self._frame_selection["user"]
                )
            if "clip" in self._frame_selection:
                clip_length = self.count_total_frames_for_selection_option(
                    self._frame_selection["clip"]
                )
            if "video" in self._frame_selection:
                video_length = self.count_total_frames_for_selection_option(
                    self._frame_selection["video"]
                )
            if "all_videos" in self._frame_selection:
                all_videos_length = self.count_total_frames_for_selection_option(
                    self._frame_selection["all_videos"]
                )

            # Build list of options
            # Priority for default (lowest to highest):
            #   "nothing" (if training)
            #   "current frame" (if inference)
            #   "suggested frames" (if available)
            #   "selected clip" (if available)
            if self.mode != "inference":
                prediction_options.append("nothing")
            prediction_options.append("current frame")
            default_option = "nothing" if self.mode != "inference" else "current frame"

            option = f"random frames ({total_random} total frames)"
            prediction_options.append(option)

            if random_video > 0:
                option = f"random frames in current video ({random_video} frames)"
                prediction_options.append(option)

            if total_suggestions > 0:
                option = f"suggested frames ({total_suggestions} total frames)"
                prediction_options.append(option)
                default_option = option

            if total_user > 0:
                option = f"user labeled frames ({total_user} total frames)"
                prediction_options.append(option)

            if clip_length > 0:
                option = f"selected clip ({clip_length} frames)"
                prediction_options.append(option)
                default_option = option

            prediction_options.append(f"entire current video ({video_length} frames)")

            if len(self.labels.videos) > 1:
                prediction_options.append(f"all videos ({all_videos_length} frames)")

            self.pipeline_form_widget.fields["_predict_frames"].set_options(
                prediction_options, default_option
            )

    def connect_signals(self):
        self.pipeline_form_widget.valueChanged.connect(self.on_tab_data_change)

        for head_name, tab in self.tabs.items():
            tab.valueChanged.connect(lambda n=head_name: self.on_tab_data_change(n))

    def disconnect_signals(self):
        self.pipeline_form_widget.valueChanged.disconnect()

        for head_name, tab in self.tabs.items():
            tab.valueChanged.disconnect()

    def make_tabs(self):
        heads = (
            "single_instance",
            "centroid",
            "centered_instance",
            "multi_instance",
            "multi_class_topdown",
            "multi_class_bottomup",
        )

        video = self.labels.videos[0] if self.labels else None

        for head_name in heads:
            self.tabs[head_name] = TrainingEditorWidget(
                video=video,
                skeleton=self.skeleton,
                head=head_name,
                cfg_getter=self._cfg_getter,
                require_trained=(self.mode == "inference"),
            )

    def adjust_data_to_update_other_tabs(self, source_data, updated_data=None):
        if updated_data is None:
            updated_data = source_data

        anchor_part = None
        set_anchor = False

        if "model.heads.centroid.anchor_part" in source_data:
            anchor_part = source_data["model.heads.centroid.anchor_part"]
            set_anchor = True
        elif "model.heads.centered_instance.anchor_part" in source_data:
            anchor_part = source_data["model.heads.centered_instance.anchor_part"]
            set_anchor = True
        elif "model.heads.multi_class_topdown.confmaps.anchor_part" in source_data:
            anchor_part = source_data[
                "model.heads.multi_class_topdown.confmaps.anchor_part"
            ]
            set_anchor = True

        # Use None instead of empty string/list
        anchor_part = anchor_part or None

        if set_anchor:
            updated_data["model.heads.centroid.anchor_part"] = anchor_part
            updated_data["model.heads.centered_instance.anchor_part"] = anchor_part
            updated_data[
                "model.heads.multi_class_topdown.confmaps.anchor_part"
            ] = anchor_part
            updated_data["data.instance_cropping.center_on_part"] = anchor_part

    def update_tabs_from_pipeline(self, source_data):
        self.adjust_data_to_update_other_tabs(source_data)

        for tab in self.tabs.values():
            tab.set_fields_from_key_val_dict(source_data)

    def update_tabs_from_tab(self, source_data):
        data_to_transfer = dict()
        self.adjust_data_to_update_other_tabs(source_data, data_to_transfer)

        if data_to_transfer:
            for tab in self.tabs.values():
                tab.set_fields_from_key_val_dict(data_to_transfer)

    def on_tab_data_change(self, tab_name=None):
        self.disconnect_signals()

        if tab_name is None:
            # Move data from pipeline tab to other tabs
            source_data = self.pipeline_form_widget.get_form_data()
            self.update_tabs_from_pipeline(source_data)
        else:
            # Get data from head-specific tab
            source_data = self.tabs[tab_name].get_all_form_data()

            self.update_tabs_from_tab(source_data)

            # Update pipeline tab
            self.pipeline_form_widget.set_form_data(source_data)

        self._validate_pipeline()

        self.connect_signals()

    def get_most_recent_pipeline_trained(self) -> Text:
        recent_cfg_info = self._cfg_getter.get_first()

        if recent_cfg_info and recent_cfg_info.head_name:
            if recent_cfg_info.head_name in ("multi_class_topdown",):
                return "top-down-id"
            if recent_cfg_info.head_name in ("centroid", "centered_instance"):
                return "top-down"
            if recent_cfg_info.head_name in ("multi_instance",):
                return "bottom-up"
            if recent_cfg_info.head_name in ("single_instance",):
                return "single"
            if recent_cfg_info.head_name in ("multi_class_bottomup",):
                return "bottom-up-id"
        return ""

    def set_default_pipeline_tab(self):
        recent_pipeline_name = self.get_most_recent_pipeline_trained()
        if recent_pipeline_name:
            self.pipeline_form_widget.current_pipeline = recent_pipeline_name
        else:
            # Set default based on detection of single- vs multi-animal project.
            if self.labels.max_user_instances == 1:
                self.pipeline_form_widget.current_pipeline = "single"
            else:
                self.pipeline_form_widget.current_pipeline = "top-down"

    def add_tab(self, tab_name):
        tab_labels = {
            "single_instance": "Single Instance Model Configuration",
            "centroid": "Centroid Model Configuration",
            "centered_instance": "Centered Instance Model Configuration",
            "multi_instance": "Bottom-Up Model Configuration",
            "multi_class_topdown": "Top-Down-Id Model Configuration",
            "multi_class_bottomup": "Bottom-Up-Id Model Configuration",
        }
        self.tab_widget.addTab(self.tabs[tab_name], tab_labels[tab_name])
        self.shown_tab_names.append(tab_name)

    def remove_tabs(self):
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        self.shown_tab_names = []

    def set_pipeline(self, pipeline: str):
        if pipeline != self.current_pipeline:
            self.remove_tabs()
            if pipeline == "top-down":
                self.add_tab("centroid")
                self.add_tab("centered_instance")
            elif pipeline == "bottom-up":
                self.add_tab("multi_instance")
            elif pipeline == "top-down-id":
                self.add_tab("centroid")
                self.add_tab("multi_class_topdown")
            elif pipeline == "bottom-up-id":
                self.add_tab("multi_class_bottomup")
            elif pipeline == "single":
                self.add_tab("single_instance")
        self.current_pipeline = pipeline

        self._validate_pipeline()

    def change_tab(self, tab_idx: int):
        print(tab_idx)

    def merge_pipeline_and_head_config_data(self, head_name, head_data, pipeline_data):
        for key, val in pipeline_data.items():
            # if key.starts_with("_"):
            #     continue
            if key.startswith("model.heads."):
                key_scope = key.split(".")
                if key_scope[2] != head_name:
                    continue
            head_data[key] = val

    @staticmethod
    def update_loaded_config(
        loaded_cfg: configs.TrainingJobConfig, tab_cfg_key_val_dict: dict
    ) -> scopedkeydict.ScopedKeyDict:
        """Update a loaded preset config with values from the training editor.

        Args:
            loaded_cfg: A `TrainingJobConfig` that was loaded from a preset or previous
                training run.
            tab_cfg_key_val_dict: A dictionary with the values extracted from the training
                editor GUI tab.

        Returns:
            A `ScopedKeyDict` with the loaded config values overriden by the corresponding
            ones from the `tab_cfg_key_val_dict`.
        """
        # Serialize training config
        loaded_cfg_hierarchical: dict = cattr.unstructure(loaded_cfg)

        # Clear backbone subfields since these will be set by the GUI
        if (
            "model" in loaded_cfg_hierarchical
            and "backbone" in loaded_cfg_hierarchical["model"]
        ):
            for k in loaded_cfg_hierarchical["model"]["backbone"]:
                loaded_cfg_hierarchical["model"]["backbone"][k] = None

        loaded_cfg_scoped: scopedkeydict.ScopedKeyDict = (
            scopedkeydict.ScopedKeyDict.from_hierarchical_dict(loaded_cfg_hierarchical)
        )

        # Replace params exposed in GUI with values from GUI
        for param, value in tab_cfg_key_val_dict.items():
            loaded_cfg_scoped.key_val_dict[param] = value

        return loaded_cfg_scoped

    def get_every_head_config_data(
        self, pipeline_form_data
    ) -> List[configs.ConfigFileInfo]:
        cfg_info_list = []

        # Copy relevant data into linked fields (i.e., anchor part).
        self.adjust_data_to_update_other_tabs(pipeline_form_data)

        for tab_name in self.shown_tab_names:
            trained_cfg_info = self.tabs[tab_name].trained_config_info_to_use
            if self.tabs[tab_name].use_trained and (trained_cfg_info is not None):
                cfg_info_list.append(trained_cfg_info)

            else:
                # Get config data from GUI
                tab_cfg_key_val_dict = self.tabs[tab_name].get_all_form_data()
                self.merge_pipeline_and_head_config_data(
                    head_name=tab_name,
                    head_data=tab_cfg_key_val_dict,
                    pipeline_data=pipeline_form_data,
                )
                scopedkeydict.apply_cfg_transforms_to_key_val_dict(tab_cfg_key_val_dict)

                if trained_cfg_info is None:
                    # Config could not be loaded, just use the values from the GUI
                    loaded_cfg_scoped: dict = tab_cfg_key_val_dict
                else:
                    # Config was loaded, override with the values from the GUI
                    loaded_cfg_scoped = LearningDialog.update_loaded_config(
                        trained_cfg_info.config, tab_cfg_key_val_dict
                    )

                # Deserialize merged dict to object
                cfg = scopedkeydict.make_training_config_from_key_val_dict(
                    loaded_cfg_scoped
                )

                if len(self.labels.tracks) > 0:

                    # For multiclass topdown, the class vectors output stride
                    # should be the max stride.
                    backbone_name = scopedkeydict.find_backbone_name_from_key_val_dict(
                        tab_cfg_key_val_dict
                    )
                    max_stride = tab_cfg_key_val_dict[
                        f"model.backbone.{backbone_name}.max_stride"
                    ]

                    # Classes should be added here to prevent value error in
                    # model since we don't add them in the training config yaml.
                    if cfg.model.heads.multi_class_bottomup is not None:
                        cfg.model.heads.multi_class_bottomup.class_maps.classes = [
                            t.name for t in self.labels.tracks
                        ]
                    elif cfg.model.heads.multi_class_topdown is not None:
                        cfg.model.heads.multi_class_topdown.class_vectors.classes = [
                            t.name for t in self.labels.tracks
                        ]
                        cfg.model.heads.multi_class_topdown.class_vectors.output_stride = (
                            max_stride
                        )

                cfg_info = configs.ConfigFileInfo(config=cfg, head_name=tab_name)

                cfg_info_list.append(cfg_info)

        return cfg_info_list

    def get_selected_frames_to_predict(
        self, pipeline_form_data
    ) -> Dict[Video, List[int]]:
        frames_to_predict = dict()

        if self._frame_selection is not None:
            predict_frames_choice = pipeline_form_data.get("_predict_frames", "")
            if predict_frames_choice.startswith("current frame"):
                frames_to_predict = self._frame_selection["frame"]
            elif predict_frames_choice.startswith("random frames in current video"):
                frames_to_predict = self._frame_selection["random_video"]
            elif predict_frames_choice.startswith("random"):
                frames_to_predict = self._frame_selection["random"]
            elif predict_frames_choice.startswith("selected clip"):
                frames_to_predict = self._frame_selection["clip"]
            elif predict_frames_choice.startswith("suggested"):
                frames_to_predict = self._frame_selection["suggestions"]
            elif predict_frames_choice.startswith("entire current video"):
                frames_to_predict = self._frame_selection["video"]
            elif predict_frames_choice.startswith("all videos"):
                frames_to_predict = self._frame_selection["all_videos"]
            elif predict_frames_choice.startswith("user"):
                frames_to_predict = self._frame_selection["user"]

        return frames_to_predict

    def get_items_for_inference(self, pipeline_form_data) -> runners.ItemsForInference:
        predict_frames_choice = pipeline_form_data.get("_predict_frames", "")
        batch_size = pipeline_form_data.get("batch_size")

        frame_selection = self.get_selected_frames_to_predict(pipeline_form_data)
        frame_count = self.count_total_frames_for_selection_option(frame_selection)

        if predict_frames_choice.startswith("user"):
            items_for_inference = runners.ItemsForInference(
                items=[
                    runners.DatasetItemForInference(
                        labels_path=self.labels_filename, frame_filter="user"
                    )
                ],
                total_frame_count=frame_count,
                batch_size=batch_size,
            )
        elif predict_frames_choice.startswith("suggested"):
            items_for_inference = runners.ItemsForInference(
                items=[
                    runners.DatasetItemForInference(
                        labels_path=self.labels_filename, frame_filter="suggested"
                    )
                ],
                total_frame_count=frame_count,
                batch_size=batch_size,
            )
        else:
            items_for_inference = runners.ItemsForInference.from_video_frames_dict(
                video_frames_dict=frame_selection,
                total_frame_count=frame_count,
                labels_path=self.labels_filename,
                labels=self.labels,
                batch_size=batch_size,
            )
        return items_for_inference

    def _validate_id_model(self) -> bool:
        """Make sure we have instances with tracks set for ID models."""
        if not self.labels.tracks:
            message = "Cannot run ID model training without tracks."
            return False

        found_tracks = False
        for inst in self.labels.instances():
            if type(inst) == sleap.Instance and inst.track is not None:
                found_tracks = True
                break

        return found_tracks

    def _validate_pipeline(self):
        can_run = True
        message = ""

        if self.mode == "inference":
            # Make sure we have trained models for each required head.
            untrained = [
                tab_name
                for tab_name in self.shown_tab_names
                if not self.tabs[tab_name].has_trained_config_selected
            ]
            if untrained:
                can_run = False
                message = (
                    "Cannot run inference with untrained models "
                    f"({', '.join(untrained)})."
                )

        # Make sure we have instances with tracks set for ID models.
        if self.mode == "training" and self.current_pipeline in (
            "top-down-id",
            "bottom-up-id",
        ):
            can_run = self.validate_id_model()
            if not can_run:
                message = "Cannot run ID model training without tracks."

        # Make sure skeleton will be valid for bottom-up inference.
        if self.mode == "training" and self.current_pipeline == "bottom-up":
            skeleton = self.labels.skeletons[0]

            if not skeleton.is_arborescence:
                message += (
                    "Cannot run bottom-up pipeline when skeleton is not an "
                    "arborescence."
                )

                root_names = [n.name for n in skeleton.root_nodes]
                over_max_in_degree = [n.name for n in skeleton.in_degree_over_one]
                cycles = skeleton.cycles

                if len(root_names) > 1:
                    message += (
                        f" There are multiple root nodes: {', '.join(root_names)} "
                        "(there should be exactly one node which is not a target)."
                    )

                if over_max_in_degree:
                    message += (
                        " There are nodes which are target in multiple edges: "
                        f"{', '.join(over_max_in_degree)} (maximum in-degree should be "
                        "1).</li>"
                    )

                if cycles:
                    cycle_strings = []
                    for cycle in cycles:
                        cycle_strings.append(
                            " &ndash;&gt; ".join((node.name for node in cycle))
                        )

                    message += (
                        f" There are cycles in graph: {'; '.join(cycle_strings)}."
                    )

                can_run = False

        if not can_run and message:
            message = f"<b>Unable to run:</b><br />{message}"

        self.message_widget.setText(message)
        self.run_button.setEnabled(can_run)

    def view_datagen(self):
        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        config_info_list = self.get_every_head_config_data(pipeline_form_data)
        datagen.show_datagen_preview(self.labels, config_info_list)
        self.hide()

    def run(self):
        """Run with current dialog settings."""

        pipeline_form_data = self.pipeline_form_widget.get_form_data()

        items_for_inference = self.get_items_for_inference(pipeline_form_data)

        config_info_list = self.get_every_head_config_data(pipeline_form_data)

        # Close the dialog now that we have the data from it
        self.accept()

        # Run training/learning pipeline using the TrainingJobs
        new_counts = runners.run_learning_pipeline(
            labels_filename=self.labels_filename,
            labels=self.labels,
            config_info_list=config_info_list,
            inference_params=pipeline_form_data,
            items_for_inference=items_for_inference,
        )

        self._handle_learning_finished.emit(new_counts)

        # count < 0 means there was an error and we didn't get any results.
        if new_counts is not None and new_counts >= 0:
            total_count = items_for_inference.total_frame_count
            no_result_count = total_count - new_counts

            message = (
                f"Inference ran on {total_count} frames."
                f"\n\nInstances were predicted on {new_counts} frames "
                f"({no_result_count} frame{'s' if no_result_count != 1 else ''} with "
                "no instances found)."
            )

            win = QtWidgets.QMessageBox(text=message)
            win.setWindowTitle("Inference Results")
            win.exec_()

    def copy(self):
        """Copy scripts and configs to clipboard"""

        # Get all info from dialog
        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        config_info_list = self.get_every_head_config_data(pipeline_form_data)
        pipeline_form_data = json.dumps(pipeline_form_data, indent=2)

        # Format information for each tab in dialog
        output = [pipeline_form_data]
        for config_info in config_info_list:
            config_info = config_info.config.to_json()
            config_info = json.loads(config_info)
            config_info = json.dumps(config_info, indent=2)
            output.append(config_info)
        output = "\n".join(output)

        # Set the clipboard text
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(output)

    def save(
        self, output_dir: Optional[str] = None, labels_filename: Optional[str] = None
    ):
        """Save scripts and configs to run pipeline."""
        if output_dir is None:
            labels_fn = Path(self.labels_filename)
            models_dir = Path(labels_fn.parent, "models")
            output_dir = FileDialog.openDir(
                None,
                dir=models_dir.as_posix(),
                caption="Select directory to save scripts",
            )

            if not output_dir:
                return

        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        items_for_inference = self.get_items_for_inference(pipeline_form_data)
        config_info_list = self.get_every_head_config_data(pipeline_form_data)

        if labels_filename is None:
            labels_filename = self.labels_filename

        runners.write_pipeline_files(
            output_dir=output_dir,
            labels_filename=labels_filename,
            config_info_list=config_info_list,
            inference_params=pipeline_form_data,
            items_for_inference=items_for_inference,
        )

    def export_package(self, output_path: Optional[str] = None, gui: bool = True):
        """Export training job package."""
        # TODO: Warn if self.mode != "training"?
        if output_path is None:
            # Prompt for output path.
            output_path, _ = FileDialog.save(
                caption="Export Training Job Package...",
                dir=f"{self.labels_filename}.training_job.zip",
                filter="Training Job Package (*.zip)",
            )
            if len(output_path) == 0:
                return

        # Create temp dir before packaging.
        tmp_dir = tempfile.TemporaryDirectory()

        # Remove the temp dir when program exits in case something goes wrong.
        # atexit.register(shutil.rmtree, tmp_dir.name, ignore_errors=True)

        # Check if we need to include suggestions.
        include_suggestions = False
        items_for_inference = self.get_items_for_inference(
            self.pipeline_form_widget.get_form_data()
        )
        for item in items_for_inference.items:
            if (
                isinstance(item, runners.DatasetItemForInference)
                and item.frame_filter == "suggested"
            ):
                include_suggestions = True

        # Save dataset with images.
        labels_pkg_filename = str(
            Path(self.labels_filename).with_suffix(".pkg.slp").name
        )
        if gui:
            ret = sleap.gui.commands.export_dataset_gui(
                self.labels,
                tmp_dir.name + "/" + labels_pkg_filename,
                all_labeled=False,
                suggested=include_suggestions,
            )
            if ret == "canceled":
                # Quit if user canceled during export.
                tmp_dir.cleanup()
                return
        else:
            self.labels.save(
                tmp_dir.name + "/" + labels_pkg_filename,
                with_images=True,
                embed_all_labeled=False,
                embed_suggested=include_suggestions,
            )

        # Save config and scripts.
        self.save(tmp_dir.name, labels_filename=labels_pkg_filename)

        # Package everything.
        shutil.make_archive(
            base_name=str(Path(output_path).with_suffix("")),
            format="zip",
            root_dir=tmp_dir.name,
        )

        msg = f"Saved training job package to: {output_path}"
        print(msg)

        # Close training editor.
        self.accept()

        if gui:
            msgBox = QtWidgets.QMessageBox(text=f"Created training job package.")
            msgBox.setDetailedText(output_path)
            msgBox.setWindowTitle("Training Job Package")
            okButton = msgBox.addButton(QtWidgets.QMessageBox.Ok)
            openFolderButton = msgBox.addButton(
                "Open containing folder", QtWidgets.QMessageBox.ActionRole
            )
            colabButton = msgBox.addButton(
                "Go to Colab", QtWidgets.QMessageBox.ActionRole
            )
            msgBox.exec_()

            if msgBox.clickedButton() == openFolderButton:
                sleap.gui.commands.open_file(str(Path(output_path).resolve().parent))
            elif msgBox.clickedButton() == colabButton:
                # TODO: Update this to more workflow-tailored notebook.
                sleap.gui.commands.copy_to_clipboard(output_path)
                sleap.gui.commands.open_website(
                    "https://colab.research.google.com/github/talmolab/sleap/blob/main/docs/notebooks/Training_and_inference_using_Google_Drive.ipynb"
                )

        tmp_dir.cleanup()


class TrainingPipelineWidget(QtWidgets.QWidget):
    """
    Widget used in :py:class:`LearningDialog` for configuring pipeline.
    """

    updatePipeline = QtCore.Signal(str)
    valueChanged = QtCore.Signal()

    def __init__(
        self, mode: Text, skeleton: Optional["Skeleton"] = None, *args, **kwargs
    ):
        super(TrainingPipelineWidget, self).__init__(*args, **kwargs)

        self.form_widget = YamlFormWidget.from_name(
            "pipeline_form", which_form=mode, title="Training Pipeline"
        )

        if hasattr(skeleton, "node_names"):
            for field_name in NODE_LIST_FIELDS:
                self.form_widget.set_field_options(
                    field_name,
                    skeleton.node_names,
                )

        # Connect actions for change to pipeline
        self.pipeline_field = self.form_widget.form_layout.find_field("_pipeline")[0]
        self.pipeline_field.valueChanged.connect(self.emitPipeline)

        self.form_widget.form_layout.valueChanged.connect(self.valueChanged)

        self.setLayout(self.form_widget.form_layout)

    @property
    def fields(self):
        return self.form_widget.fields

    @property
    def buttons(self):
        return self.form_widget.buttons

    def set_message(self, message: Text):
        self.form_widget.set_message()

    def get_form_data(self):
        return self.form_widget.get_form_data()

    def set_form_data(self, data):
        self.form_widget.set_form_data(data)

    def emitPipeline(self):
        val = self.current_pipeline
        self.updatePipeline.emit(val)

    @property
    def current_pipeline(self):
        pipeline_selected_label = self.pipeline_field.value()
        if "top-down" in pipeline_selected_label:
            if "id" not in pipeline_selected_label:
                return "top-down"
            else:
                return "top-down-id"
        if "bottom-up" in pipeline_selected_label:
            if "id" not in pipeline_selected_label:
                return "bottom-up"
            else:
                return "bottom-up-id"
        if "single" in pipeline_selected_label:
            return "single"
        return ""

    @current_pipeline.setter
    def current_pipeline(self, val):
        if val not in (
            "top-down",
            "bottom-up",
            "single",
            "top-down-id",
            "bottom-up-id",
        ):
            raise ValueError(f"Cannot set pipeline to {val}")

        # Match short name to full pipeline name shown in menu
        for full_option_name in self.pipeline_field.option_list:
            if val in full_option_name:
                val = full_option_name
                break

        self.pipeline_field.setValue(val)
        self.emitPipeline()


class TrainingEditorWidget(QtWidgets.QWidget):
    """
    Dialog for viewing and modifying training profiles (model hyperparameters).

    Args:
        video: `Video` to use for receptive field preview
        skeleton: `Skeleton` to use for node option list
        head: If given, then only show configs with specified head name
        cfg_getter: Object to use for getting list of config files.
            If given, then menu of config files will be shown so user can
            either copy hyperameters from another profile/model, or use a model
            that was already trained.
        require_trained: If True, then only show configs that are trained,
            and don't allow user to uncheck "use trained" setting. This is set
            when :py:class:`LearningDialog` is in "inference" mode.
    """

    valueChanged = QtCore.Signal()

    def __init__(
        self,
        video: Optional[Video] = None,
        skeleton: Optional["Skeleton"] = None,
        head: Optional[Text] = None,
        cfg_getter: Optional["TrainingConfigsGetter"] = None,
        require_trained: bool = False,
        *args,
        **kwargs,
    ):
        super(TrainingEditorWidget, self).__init__()

        self._video = video
        self._cfg_getter = cfg_getter
        self._cfg_list_widget = None
        self._receptive_field_widget = None
        self._use_trained_model = None
        self._resume_training = None
        self._require_trained = require_trained
        self.head = head

        yaml_name = "training_editor_form"

        self.form_widgets: Dict[str, YamlFormWidget] = dict()

        for key in ("model", "data", "augmentation", "optimization", "outputs"):
            self.form_widgets[key] = YamlFormWidget.from_name(
                yaml_name, which_form=key, title=key.title()
            )
            self.form_widgets[key].valueChanged.connect(self.emitValueChanged)

        self.form_widgets["model"].valueChanged.connect(self.update_receptive_field)
        self.form_widgets["data"].valueChanged.connect(self.update_receptive_field)

        if hasattr(skeleton, "node_names"):
            for field_name in NODE_LIST_FIELDS:
                form_name = field_name.split(".")[0]
                self.form_widgets[form_name].set_field_options(
                    field_name,
                    skeleton.node_names,
                )

        if self._video:
            self._receptive_field_widget = receptivefield.ReceptiveFieldWidget(
                self.head
            )
            self._receptive_field_widget.setImage(self._video.test_frame)

        self._set_head()

        # Layout for header and columns
        layout = QtWidgets.QVBoxLayout()

        # Two column layout for config parameters
        col1_layout = QtWidgets.QVBoxLayout()
        col2_layout = QtWidgets.QVBoxLayout()
        col3_layout = QtWidgets.QVBoxLayout()

        col1_layout.addWidget(self.form_widgets["data"])
        col1_layout.addWidget(self.form_widgets["optimization"])
        col2_layout.addWidget(self.form_widgets["augmentation"])
        col3_layout.addWidget(self.form_widgets["model"])

        if self._receptive_field_widget:
            col0_layout = QtWidgets.QVBoxLayout()
            col0_layout.addWidget(self._receptive_field_widget)
        else:
            col0_layout = None

        col_layout = QtWidgets.QHBoxLayout()
        if col0_layout:
            col_layout.addWidget(self._layout_widget(col0_layout))
        col_layout.addWidget(self._layout_widget(col1_layout))
        col_layout.addWidget(self._layout_widget(col2_layout))
        col_layout.addWidget(self._layout_widget(col3_layout))

        # If we have an object which gets a list of config files,
        # then we'll show a menu to allow selection from the list.
        if self._cfg_getter is not None:
            self._cfg_list_widget = configs.TrainingConfigFilesWidget(
                cfg_getter=self._cfg_getter,
                head_name=cast(str, head),  # Expect head to be a string
                require_trained=require_trained,
            )
            self._cfg_list_widget.onConfigSelection.connect(
                self.acceptSelectedConfigInfo
            )
            # self._cfg_list_widget.setDataDict.connect(self.set_fields_from_key_val_dict)

            layout.addWidget(self._cfg_list_widget)

        if self._require_trained:
            self._update_use_trained()
        elif self._cfg_list_widget is not None:
            # Add option for using trained model from selected config file
            self._use_trained_model = QtWidgets.QCheckBox("Use Trained Model")
            self._use_trained_model.setEnabled(False)
            self._use_trained_model.setVisible(False)
            self._resume_training = QtWidgets.QCheckBox("Resume Training")
            self._resume_training.setEnabled(False)
            self._resume_training.setVisible(False)

            self._use_trained_model.stateChanged.connect(self._update_use_trained)
            self._resume_training.stateChanged.connect(self._update_use_trained)

            layout.addWidget(self._use_trained_model)
            layout.addWidget(self._resume_training)

        layout.addWidget(self._layout_widget(col_layout))
        self.setLayout(layout)

    @classmethod
    def from_trained_config(
        cls, cfg_info: configs.ConfigFileInfo, cfg_getter: configs.TrainingConfigsGetter
    ):
        widget = cls(
            require_trained=True, head=cfg_info.head_name, cfg_getter=cfg_getter
        )
        widget.acceptSelectedConfigInfo(cfg_info)
        widget.setWindowTitle(cfg_info.path_dir)
        return widget

    @staticmethod
    def _layout_widget(layout):
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def emitValueChanged(self):
        self.valueChanged.emit()

        # When there's a config getter, we want to inform it that the data
        # has changed so that it can activate/update the "user" config
        # if self._cfg_list_widget:
        #     self._set_user_config()

    def acceptSelectedConfigInfo(self, cfg_info: configs.ConfigFileInfo):
        self._load_config(cfg_info)

        has_trained_model = cfg_info.has_trained_model
        if self._use_trained_model is not None:
            self._use_trained_model.setChecked(self._require_trained)
            self._use_trained_model.setVisible(has_trained_model)
            self._use_trained_model.setEnabled(has_trained_model)
        # Redundant check (for readability) since this checkbox exists if the above does
        if self._resume_training is not None:
            self._use_trained_model.setChecked(False)
            self._resume_training.setVisible(has_trained_model)
            self._resume_training.setEnabled(has_trained_model)

        self.update_receptive_field()

    def update_receptive_field(self):
        data_form_data = self.form_widgets["data"].get_form_data()

        model_cfg = scopedkeydict.make_model_config_from_key_val_dict(
            key_val_dict=self.form_widgets["model"].get_form_data()
        )

        rf_image_scale = data_form_data.get("data.preprocessing.input_scaling", 1.0)

        if self._receptive_field_widget:
            self._receptive_field_widget.setModelConfig(model_cfg, scale=rf_image_scale)
            self._receptive_field_widget.repaint()

    def update_file_list(self):
        self._cfg_list_widget.update()

    def _load_config_or_key_val_dict(self, cfg_data):
        if type(cfg_data) != dict:
            self._load_config(cfg_data)
        else:
            self.set_fields_from_key_val_dict(cfg_data)

    def _load_config(self, cfg_info: configs.ConfigFileInfo):
        if cfg_info is None:
            return

        cfg = cfg_info.config
        cfg_dict = cattr.unstructure(cfg)
        key_val_dict = scopedkeydict.ScopedKeyDict.from_hierarchical_dict(
            cfg_dict
        ).key_val_dict
        self.set_fields_from_key_val_dict(key_val_dict)

    # def _set_user_config(self):
    #     cfg_form_data_dict = self.get_all_form_data()
    #     self._cfg_list_widget.setUserConfigData(cfg_form_data_dict)

    def _update_use_trained(self, check_state=0):
        """Update config GUI based on _use_trained_model and _resume_training checkboxes.

        This function is called when either _use_trained_model or _resume_training checkbox
        is checked/unchecked or when _require_trained is changed.

        If _require_trained is True, then we'll disable all fields.
        If _use_trained_model is checked, then we'll disable all fields.
        If _resume_training is checked, then we'll disable only the model field.

        Args:
            check_state (int, optional): Check state of checkbox. Defaults to 0. Unused.

        Returns:
            None

        Side Effects:
            Disables/Enables fields based on checkbox values (and _required_training).
        """

        # Check which checkbox changed its value (if any)
        sender = self.sender()

        if sender is None:  # If sender is None, then _required_training is True
            pass
        # Uncheck _resume_training checkbox if _use_trained_model is unchecked
        elif (sender == self._use_trained_model) and (
            not self._use_trained_model.isChecked()
        ):
            self._resume_training.setChecked(False)

        # Check _use_trained_model checkbox if _resume_training is checked
        elif (sender == self._resume_training) and self._resume_training.isChecked():
            self._use_trained_model.setChecked(True)

        # Update form widgets
        use_trained_params = self.use_trained
        use_model_params = self.resume_training
        for form in self.form_widgets.values():
            form.set_enabled(not use_trained_params)

        if use_trained_params or use_model_params:
            cfg_info = self._cfg_list_widget.getSelectedConfigInfo()

        # If user wants to resume training, then reset only model form to match config
        if use_model_params:
            self.form_widgets["model"].set_enabled(False)

            # Set model form to match config
            cfg = cfg_info.config
            cfg_dict = cattr.unstructure(cfg)
            model_dict = {"model": cfg_dict["model"]}
            key_val_dict = scopedkeydict.ScopedKeyDict.from_hierarchical_dict(
                model_dict
            ).key_val_dict
            self.set_fields_from_key_val_dict(key_val_dict)

        # If user wants to use trained model, then reset entire form to match config
        if use_trained_params:
            self._load_config(cfg_info)

        self._set_head()

    def _set_head(self):
        if self.head:
            self.set_fields_from_key_val_dict(
                {
                    "_heads_name": self.head,
                }
            )

            self.form_widgets["model"].set_field_enabled("_heads_name", False)

    def set_fields_from_key_val_dict(self, cfg_key_val_dict):
        for form in self.form_widgets.values():
            form.set_form_data(cfg_key_val_dict)

        self._set_backbone_from_key_val_dict(cfg_key_val_dict)

    def _set_backbone_from_key_val_dict(self, cfg_key_val_dict):
        for key, val in cfg_key_val_dict.items():
            if key.startswith("model.backbone.") and val is not None:
                backbone_name = key.split(".")[2]
                self.set_fields_from_key_val_dict(dict(_backbone_name=backbone_name))
                break

    @property
    def use_trained(self) -> bool:
        if self._require_trained or (
            (self._use_trained_model is not None)
            and self._use_trained_model.isChecked()
            and (not self.resume_training)
        ):
            return True

        return False

    @property
    def resume_training(self) -> bool:
        if (self._resume_training is not None) and self._resume_training.isChecked():
            return True
        return False

    @property
    def trained_config_info_to_use(self) -> Optional[configs.ConfigFileInfo]:
        # If `TrainingEditorWidget` was initialized with a config getter, then
        # we expect to have a list of config files
        if self._cfg_list_widget is None:
            return None

        selected_config_info: Optional[
            configs.ConfigFileInfo
        ] = self._cfg_list_widget.getSelectedConfigInfo()
        if (selected_config_info is None) or (
            not selected_config_info.has_trained_model
        ):
            return None

        trained_config_info = configs.ConfigFileInfo.from_config_file(
            selected_config_info.path
        )
        if self.use_trained:
            trained_config_info.dont_retrain = True
        else:
            # Set certain parameters to defaults
            trained_config = trained_config_info.config
            trained_config.data.labels.validation_labels = None
            trained_config.data.labels.test_labels = None
            trained_config.data.labels.split_by_inds = False
            trained_config.data.labels.skeletons = []
            trained_config.outputs.run_name = None
            trained_config.outputs.run_name_prefix = ""
            trained_config.outputs.run_name_suffix = None

        if self.resume_training:
            # Get the folder path of trained config and set it as the output folder
            trained_config_info.config.model.base_checkpoint = str(
                Path(cast(str, trained_config_info.path)).parent
            )
        else:
            trained_config_info.config.model.base_checkpoint = None

        return trained_config_info

    @property
    def has_trained_config_selected(self) -> bool:
        if self._cfg_list_widget is None:
            return False

        cfg_info = self._cfg_list_widget.getSelectedConfigInfo()
        if cfg_info and cfg_info.has_trained_model:
            return True

        return False

    def get_all_form_data(self) -> dict:
        form_data = dict()
        for form in self.form_widgets.values():
            form_data.update(form.get_form_data())
        return form_data


def demo_training_dialog():
    app = QtWidgets.QApplication([])

    filename = "tests/data/json_format_v1/centered_pair.json"
    labels = Labels.load_file(filename)
    win = LearningDialog("inference", labels_filename=filename, labels=labels)

    win.frame_selection = {"clip": {labels.videos[0]: (1, 2, 3, 4)}}
    # win.training_editor_widget.set_fields_from_key_val_dict({
    #     "_backbone_name": "unet",
    #     "_heads_name": "centered_instance",
    # })
    #
    # win.training_editor_widget.form_widgets["model"].set_field_enabled("_heads_name", False)

    win.show()
    app.exec_()


if __name__ == "__main__":
    demo_training_dialog()

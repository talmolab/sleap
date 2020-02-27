import attr
import cattr
import os
import subprocess as sub
import tempfile
import time

from datetime import datetime

from sleap import Labels, Video
from sleap.nn import training
from sleap.nn.config import TrainingJobConfig
from sleap.gui.formbuilder import YamlFormWidget

from typing import Any, Callable, Dict, List, Optional, Text

from PySide2 import QtWidgets, QtCore

SKIP_TRAINING = False

NODE_LIST_FIELDS = [
    "data.instance_cropping.center_on_part",
    "model.heads.centered_instance.anchor_part",
    "model.heads.centroid.anchor_part",
]


@attr.s(auto_attribs=True)
class ScopedKeyDict:

    key_val_dict: Dict[Text, Any]

    @classmethod
    def set_hierarchical_key_val(cls, current_dict, key, val):
        # Ignore "private" keys starting with "_"
        if key[0] == "_":
            return

        if "." not in key:
            current_dict[key] = val
        else:
            top_key, *subkey_list = key.split(".")
            if top_key not in current_dict:
                current_dict[top_key] = dict()
            subkey = ".".join(subkey_list)
            cls.set_hierarchical_key_val(current_dict[top_key], subkey, val)

    def to_hierarchical_dict(self):
        hierarch_dict = dict()
        for key, val in self.key_val_dict.items():
            self.set_hierarchical_key_val(hierarch_dict, key, val)
        return hierarch_dict

    @classmethod
    def from_hierarchical_dict(cls, hierarch_dict):
        return cls(key_val_dict=cls._make_flattened_dict(hierarch_dict))

    @classmethod
    def _make_flattened_dict(cls, hierarch_dicts, scope_string=""):
        flattened_dict = dict()
        for key, val in hierarch_dicts:
            if isinstance(val, Dict):
                flattened_dict.update(
                    cls._make_flattened_dict(val, f"{scope_string}{key}.")
                )
            else:
                flattened_dict[f"{scope_string}.{key}"] = val
        return flattened_dict


def apply_cfg_transforms_to_key_val_dict(key_val_dict):
    if "outputs.tags" in key_val_dict and isinstance(key_val_dict["outputs.tags"], str):
        key_val_dict["outputs.tags"] = [
            tag.strip() for tag in key_val_dict["outputs.tags"].split(",")
        ]

    if "_ensure_channels" in key_val_dict:
        ensure_channels = key_val_dict["_ensure_channels"].lower()
        ensure_rgb = False
        ensure_grayscale = False
        if ensure_channels == "rgb":
            ensure_rgb = True
        elif ensure_channels == "grayscale":
            ensure_grayscale = True

        key_val_dict["data.preprocessing.ensure_rgb"] = ensure_rgb
        key_val_dict["data.preprocessing.ensure_grayscale"] = ensure_grayscale


def make_training_config_from_key_val_dict(key_val_dict):
    apply_cfg_transforms_to_key_val_dict(key_val_dict)
    cfg_dict = ScopedKeyDict(key_val_dict).to_hierarchical_dict()

    cfg = cattr.structure(cfg_dict, TrainingJobConfig)

    return cfg


class TrainingDialog(QtWidgets.QDialog):
    def __init__(
        self,
        labels_filename: Text,
        labels: Optional[Labels] = None,
        skeleton: Optional["Skeleton"] = None,
        *args,
        **kwargs,
    ):
        super(TrainingDialog, self).__init__()

        if labels is None:
            labels = Labels.load_file(labels_filename)

        if skeleton is None:
            skeleton = labels.skeletons[0]

        self.labels_filename = labels_filename
        self.labels = labels
        self.skeleton = skeleton

        self._frame_selection = None

        self.current_pipeline = ""

        self.tabs = dict()
        self.shown_tab_names = []

        # Layout for buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.addButton(
            "Run", QtWidgets.QDialogButtonBox.AcceptRole
        )

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        self.pipeline_form_widget = TrainingPipelineWidget(skeleton=skeleton)

        self.tab_widget = QtWidgets.QTabWidget()

        # self.training_editor_widget = TrainingEditorWidget(skeleton=skeleton)
        self.tab_widget.addTab(self.pipeline_form_widget, "Training Pipeline")
        self.make_tabs()

        # Layout for entire dialog
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tab_widget)
        layout.addWidget(buttons_layout_widget)

        self.setLayout(layout)

        # Connect functions to update pipeline tabs when pipeline changes
        self.pipeline_form_widget.updatePipeline.connect(self.set_pipeline)
        self.pipeline_form_widget.emitPipeline()

        self.connect_signals()

        # Connect actions for buttons
        buttons.accepted.connect(self.run)
        buttons.rejected.connect(self.reject)

    @property
    def frame_selection(self) -> Dict[str, Dict[Video, List[int]]]:
        """
        Returns dictionary with frames that user has selected for inference.
        """
        return self._frame_selection

    @frame_selection.setter
    def frame_selection(self, frame_selection: Dict[str, Dict[Video, List[int]]]):
        """Sets options of frames on which to run inference."""
        self._frame_selection = frame_selection

        if "_predict_frames" in self.pipeline_form_widget.fields.keys():
            prediction_options = []

            def count_total_frames(videos_frames):
                if not videos_frames:
                    return 0
                count = 0
                for frame_list in videos_frames.values():
                    # Check for [x, Y] range given as X, -Y
                    # (we're not using typical [X, Y)-style range here)
                    if len(frame_list) == 2 and frame_list[1] < 0:
                        count += -frame_list[1] - frame_list[0]
                    elif frame_list != (0, 0):
                        count += len(frame_list)
                return count

            total_random = 0
            total_suggestions = 0
            clip_length = 0
            video_length = 0

            # Determine which options are available given _frame_selection
            if "random" in self._frame_selection:
                total_random = count_total_frames(self._frame_selection["random"])
            if "suggestions" in self._frame_selection:
                total_suggestions = count_total_frames(
                    self._frame_selection["suggestions"]
                )
            if "clip" in self._frame_selection:
                clip_length = count_total_frames(self._frame_selection["clip"])
            if "video" in self._frame_selection:
                video_length = count_total_frames(self._frame_selection["video"])

            # Build list of options

            # if self.mode != "inference":
            prediction_options.append("nothing")
            prediction_options.append("current frame")

            option = f"random frames ({total_random} total frames)"
            prediction_options.append(option)
            default_option = option

            if total_suggestions > 0:
                option = f"suggested frames ({total_suggestions} total frames)"
                prediction_options.append(option)
                default_option = option

            if clip_length > 0:
                option = f"selected clip ({clip_length} frames)"
                prediction_options.append(option)
                default_option = option

            prediction_options.append(f"entire video ({video_length} frames)")

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
        heads = ("single_instance", "centroid", "centered_instance", "multi_instance")

        for head_name in heads:
            self.tabs[head_name] = TrainingEditorWidget(
                skeleton=self.skeleton, head=head_name
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

        if set_anchor:
            updated_data["model.heads.centroid.anchor_part"] = anchor_part
            updated_data["model.heads.centered_instance.anchor_part"] = anchor_part

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

        self.connect_signals()

    def add_tab(self, tab_name):
        tab_labels = {
            "single_instance": "Single Instance Model Configuration",
            "centroid": "Centroid Model Configuration",
            "centered_instance": "Centered Instance Model Configuration",
            "multi_instance": "Bottom-Up Model Configuration",
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
            elif pipeline == "single":
                self.add_tab("single_instance")
        self.current_pipeline = pipeline

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

    def get_every_head_config_data(self, pipeline_form_data):
        cfgs = dict()

        for tab_name in self.shown_tab_names:
            tab_cfg_key_val_dict = self.tabs[tab_name].get_all_form_data()

            self.merge_pipeline_and_head_config_data(
                head_name=tab_name,
                head_data=tab_cfg_key_val_dict,
                pipeline_data=pipeline_form_data,
            )

            print(tab_cfg_key_val_dict)
            cfgs[tab_name] = make_training_config_from_key_val_dict(
                tab_cfg_key_val_dict
            )

        return cfgs

    def get_selected_frames_to_predict(self, pipeline_form_data):
        frames_to_predict = dict()

        if self._frame_selection is not None:
            predict_frames_choice = pipeline_form_data.get("_predict_frames", "")
            if predict_frames_choice.startswith("current frame"):
                frames_to_predict = self._frame_selection["frame"]
            elif predict_frames_choice.startswith("random"):
                frames_to_predict = self._frame_selection["random"]
            elif predict_frames_choice.startswith("selected clip"):
                frames_to_predict = self._frame_selection["clip"]
            elif predict_frames_choice.startswith("suggested"):
                frames_to_predict = self._frame_selection["suggestions"]
            elif predict_frames_choice.startswith("entire video"):
                frames_to_predict = self._frame_selection["video"]

            # Convert [X, Y+1) ranges to [X, Y] ranges for inference cli
            for video, frame_list in frames_to_predict.items():
                # Check for [A, -B] list representing [A, B) range
                if len(frame_list) == 2 and frame_list[1] < 0:
                    frame_list = (frame_list[0], frame_list[1] + 1)
                    frames_to_predict[video] = frame_list

        return frames_to_predict

    def run(self):
        """Run with current dialog settings."""

        # import pprint
        # pp = pprint.PrettyPrinter(indent=1)

        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        frames_to_predict = self.get_selected_frames_to_predict(pipeline_form_data)

        configs = self.get_every_head_config_data(pipeline_form_data)
        run_learning_pipeline(
            labels_filename=self.labels_filename,
            labels=self.labels,
            training_jobs=configs,
            inference_params=pipeline_form_data,
            frames_to_predict=frames_to_predict,
        )
        # for head_name, cfg in self.get_every_head_config_data().items():
        # print()
        # print(f"============{head_name}============")
        # print()
        # pp.pprint(cattr.unstructure(cfg))

        # train_subprocess(
        #     cfg,
        #     labels_filename="tests/data/json_format_v1/centered_pair.json",
        #     # skip_training=True
        # )

        # Close the dialog now that we have the data from it
        self.accept()


class TrainingPipelineWidget(QtWidgets.QWidget):
    updatePipeline = QtCore.Signal(str)
    valueChanged = QtCore.Signal()

    def __init__(self, skeleton: Optional["Skeleton"] = None, *args, **kwargs):
        super(TrainingPipelineWidget, self).__init__(*args, **kwargs)

        self.form_widget = YamlFormWidget.from_name(
            "pipeline_form", which_form="pipeline", title="Inference Pipeline"
        )

        if hasattr(skeleton, "node_names"):
            for field_name in NODE_LIST_FIELDS:
                self.form_widget.set_field_options(
                    field_name, skeleton.node_names,
                )

        # Connect actions for change to pipeline
        self.pipeline_field = self.form_widget.form_layout.find_field("_pipeline")[0]
        self.pipeline_field.valueChanged.connect(self.emitPipeline)

        self.form_widget.form_layout.valueChanged.connect(self.valueChanged)

        self.setLayout(self.form_widget.form_layout)

    @property
    def fields(self):
        return self.form_widget.fields

    def get_form_data(self):
        return self.form_widget.get_form_data()

    def set_form_data(self, data):
        self.form_widget.set_form_data(data)

    def emitPipeline(self):
        self.updatePipeline.emit(self.current_pipeline)

    @property
    def current_pipeline(self):
        pipeline_selected_label = self.pipeline_field.value()
        if "top-down" in pipeline_selected_label:
            return "top-down"
        if "bottom-up" in pipeline_selected_label:
            return "bottom-up"
        if "single" in pipeline_selected_label:
            return "single"
        return ""


class TrainingEditorWidget(QtWidgets.QWidget):
    """
    Dialog for viewing and modifying training profiles.

    Args:
        profile_filename: Path to saved training profile to view.
        saved_files: When user saved profile, it's path is added to this
            list (which will be updated in code that created TrainingEditor).
    """

    valueChanged = QtCore.Signal()

    def __init__(
        self,
        skeleton: Optional["Skeleton"] = None,
        head: Optional[Text] = None,
        *args,
        **kwargs,
    ):
        super(TrainingEditorWidget, self).__init__()

        yaml_name = "training_editor_form"

        self.form_widgets = dict()

        for key in ("model", "data", "optimization", "outputs"):
            self.form_widgets[key] = YamlFormWidget.from_name(
                yaml_name, which_form=key, title=key.title()
            )
            self.form_widgets[key].valueChanged.connect(self.valueChanged)

        if hasattr(skeleton, "node_names"):
            for field_name in NODE_LIST_FIELDS:
                form_name = field_name.split(".")[0]
                self.form_widgets[form_name].set_field_options(
                    field_name, skeleton.node_names,
                )

        if head:
            self.set_fields_from_key_val_dict(
                {"_heads_name": head,}
            )

            self.form_widgets["model"].set_field_enabled("_heads_name", False)

        # Two column layout for config parameters
        col1_layout = QtWidgets.QVBoxLayout()
        col2_layout = QtWidgets.QVBoxLayout()

        # col1_layout.addWidget(self.form_widgets["data"])
        # col1_layout.addWidget(self.form_widgets["model"])
        #
        # col2_layout.addWidget(self.form_widgets["optimization"])
        # col2_layout.addWidget(self.form_widgets["outputs"])

        # col1_layout.addWidget(self.form_widgets["data"])
        col1_layout.addWidget(self.form_widgets["optimization"])

        col2_layout.addWidget(self.form_widgets["model"])

        col_layout = QtWidgets.QHBoxLayout()
        col_layout.addWidget(self._layout_widget(col1_layout))
        col_layout.addWidget(self._layout_widget(col2_layout))

        self.setLayout(col_layout)

    @staticmethod
    def _layout_widget(layout):
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _load_config(self, cfg):
        cfg_dict = cattr.unstructure(cfg)
        key_val_dict = ScopedKeyDict.from_hierarchical_dict(cfg_dict).key_val_dict
        self.set_fields_from_key_val_dict(key_val_dict)

    def set_fields_from_key_val_dict(self, cfg_key_val_dict):
        for form in self.form_widgets.values():
            form.set_form_data(cfg_key_val_dict)

    def get_all_form_data(self):
        form_data = dict()
        for form in self.form_widgets.values():
            form_data.update(form.get_form_data())
        return form_data


def run_learning_pipeline(
    labels_filename: str,
    labels: Labels,
    training_jobs: Dict[Text, TrainingJobConfig],
    inference_params: Dict[str, Any],
    frames_to_predict: Dict[Video, List[int]] = None,
) -> int:
    """Run training (as needed) and inference.

    Args:
        labels_filename: Path to already saved current labels object.
        labels: The current labels object; results will be added to this.
        training_jobs: The TrainingJobs with params/hyperparams for training.
        inference_params: Parameters to pass to inference.
        frames_to_predict: Dict that gives list of frame indices for each video.

    Returns:
        Number of new frames added to labels.

    """

    save_viz = inference_params.get("_save_viz", False)

    # Train the TrainingJobs
    trained_jobs = run_gui_training(
        labels_filename, training_jobs, gui=True, save_viz=save_viz
    )

    # Check that all the models were trained
    if None in trained_jobs.values():
        return -1

    trained_job_paths = list(trained_jobs.values())

    # Run the Predictor for suggested frames
    new_labeled_frame_count = run_gui_inference(
        labels=labels,
        trained_job_paths=trained_job_paths,
        inference_params=inference_params,
        frames_to_predict=frames_to_predict,
        labels_filename=labels_filename,
    )

    return new_labeled_frame_count


def has_jobs_to_train(training_jobs: Dict["ModelOutputType", "TrainingJob"]):
    """Returns whether any of the jobs need to be trained."""
    return any(not getattr(job, "use_trained_model", False) for job in training_jobs)


def run_gui_training(
    labels_filename: str,
    training_jobs: Dict[Text, TrainingJobConfig],
    gui: bool = True,
    save_viz: bool = False,
) -> Dict[Text, Text]:
    """
    Run training for each training job.

    Args:
        labels: Labels object from which we'll get training data.
        training_jobs: Dict of the jobs to train.
        gui: Whether to show gui windows and process gui events.
        save_viz: Whether to save visualizations from training.

    Returns:
        Dictionary, keys are head name, values are path to trained config.
    """

    trained_jobs = dict()

    if gui:
        from sleap.nn.monitor import LossViewer
        from sleap.gui.imagedir import QtImageDirectoryWidget

        # open training monitor window
        win = LossViewer()
        win.resize(600, 400)
        win.show()

    for model_type, job in training_jobs.items():
        if getattr(job, "use_trained_model", False):
            # set path to TrainingJob already trained from previous run
            # json_name = f"{job.run_name}.json"
            trained_jobs[model_type] = job.outputs.run_path
            print(f"Using already trained model: {trained_jobs[model_type]}")

        else:
            # Update save dir and run name for job we're about to train
            # so we have accessgi to them here (rather than letting
            # train_subprocess update them).
            # training.Trainer.set_run_name(job, labels_filename)
            job.outputs.runs_folder = os.path.join(
                os.path.dirname(labels_filename), "models"
            )
            training.setup_new_run_folder(job.outputs)

            if gui:
                print("Resetting monitor window.")
                win.reset(what=str(model_type))
                win.setWindowTitle(f"Training Model - {str(model_type)}")
                if save_viz:
                    viz_window = QtImageDirectoryWidget.make_training_vizualizer(
                        job.outputs.run_path
                    )
                    viz_window.move(win.x() + win.width() + 20, win.y())
                    win.on_epoch.connect(viz_window.poll)

            print(f"Start training {str(model_type)}...")

            def waiting():
                if gui:
                    QtWidgets.QApplication.instance().processEvents()

            # Run training
            trained_job_path, success = train_subprocess(
                job,
                labels_filename,
                waiting_callback=waiting,
                update_run_name=False,
                save_viz=save_viz,
            )

            if success:
                # get the path to the resulting TrainingJob file
                trained_jobs[model_type] = trained_job_path
                print(f"Finished training {str(model_type)}.")
            else:
                if gui:
                    win.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occurred while training {str(model_type)}. Your command line terminal may have more information about the error."
                    ).exec_()
                trained_jobs[model_type] = None

    if gui:
        # close training monitor window
        win.close()

    return trained_jobs


def run_gui_inference(
    labels: Labels,
    trained_job_paths: List[str],
    frames_to_predict: Dict[Video, List[int]],
    inference_params: Dict[str, str],
    labels_filename: str,
    gui: bool = True,
) -> int:
    """Run inference on specified frames using models from training_jobs.

    Args:
        labels: The current labels object; results will be added to this.
        trained_job_paths: List of paths to TrainingJobs with trained models.
        frames_to_predict: Dict that gives list of frame indices for each video.
        inference_params: Parameters to pass to inference.
        gui: Whether to show gui windows and process gui events.

    Returns:
        Number of new frames added to labels.
    """
    from sleap.nn import inference

    if gui:
        # show message while running inference
        progress = QtWidgets.QProgressDialog(
            f"Running inference on {len(frames_to_predict)} videos...",
            "Cancel",
            0,
            len(frames_to_predict),
        )
        progress.show()
        QtWidgets.QApplication.instance().processEvents()

    new_lfs = []
    for i, (video, frames) in enumerate(frames_to_predict.items()):

        if len(frames):

            def waiting():
                if gui:
                    QtWidgets.QApplication.instance().processEvents()
                    progress.setValue(i)
                    if progress.wasCanceled():
                        return -1

            # Run inference for desired frames in this video
            predictions_path, success = predict_subprocess(
                video=video,
                frames=frames,
                trained_job_paths=trained_job_paths,
                kwargs=inference_params,
                waiting_callback=waiting,
                labels_filename=labels_filename,
            )

            if success:
                predictions_labels = Labels.load_file(predictions_path, match_to=labels)
                new_lfs.extend(predictions_labels.labeled_frames)
            else:
                if gui:
                    progress.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occcured during inference. Your command line terminal may have more information about the error."
                    ).exec_()
                return -1

    # Remove any frames without instances
    new_lfs = list(filter(lambda lf: len(lf.instances), new_lfs))

    # Merge predictions into current labels dataset
    _, _, new_conflicts = Labels.complex_merge_between(
        labels,
        new_labels=Labels(new_lfs),
        unify=False,  # since we used match_to when loading predictions file
    )

    # new predictions should replace old ones
    Labels.finish_complex_merge(labels, new_conflicts)

    # close message window
    if gui:
        progress.close()

    # return total_new_lf_count
    return len(new_lfs)


def train_subprocess(
    job_config: TrainingJobConfig,
    labels_filename: str,
    waiting_callback: Optional[Callable] = None,
    update_run_name: bool = True,
    save_viz: bool = False,
):
    """Runs training inside subprocess."""

    # run_name = job_config.outputs.run_name
    run_path = job_config.outputs.run_path

    success = False

    with tempfile.TemporaryDirectory() as temp_dir:

        # Write a temporary file of the TrainingJob so that we can respect
        # any changed made to the job attributes after it was loaded.
        temp_filename = datetime.now().strftime("%y%m%d_%H%M%S") + "_training_job.json"
        training_job_path = os.path.join(temp_dir, temp_filename)
        job_config.save_json(training_job_path)

        # Build CLI arguments for training
        cli_args = [
            "python",
            "-m",
            "sleap.nn.training",
            training_job_path,
            labels_filename,
            "--zmq",
            # "--run_name",
            # run_name,
        ]

        if save_viz:
            cli_args.append("--save_viz")

        print(cli_args)

        if not SKIP_TRAINING:
            # Run training in a subprocess
            with sub.Popen(cli_args) as proc:

                # Wait till training is done, calling a callback if given.
                while proc.poll() is None:
                    if waiting_callback is not None:
                        if waiting_callback() == -1:
                            # -1 signals user cancellation
                            return "", False
                    time.sleep(0.1)

                success = proc.returncode == 0

    print("Run Path:", run_path)

    return run_path, success


def predict_subprocess(
    video: "Video",
    trained_job_paths: List[str],
    kwargs: Dict[str, str],
    frames: Optional[List[int]] = None,
    waiting_callback: Optional[Callable] = None,
    labels_filename: Optional[str] = None,
):
    cli_args = ["python", "-m", "sleap.nn.inference"]

    if not trained_job_paths and "tracking.tracker" in kwargs and labels_filename:
        # No models so we must want to re-track previous predictions
        cli_args.append(labels_filename)
    else:
        cli_args.append(video.filename)

    # TODO: better support for video params
    if hasattr(video.backend, "dataset"):
        cli_args.extend(("--video.dataset", video.backend.dataset))

    if hasattr(video.backend, "input_format"):
        cli_args.extend(("--video.input_format", video.backend.input_format))

    # Make path where we'll save predictions
    output_path = ".".join(
        (video.filename, datetime.now().strftime("%y%m%d_%H%M%S"), "predictions.h5",)
    )

    for job_path in trained_job_paths:
        cli_args.extend(("-m", job_path))

    for key, val in kwargs.items():
        if not key.startswith("_"):
            cli_args.extend((f"--{key}", str(val)))

    cli_args.extend(("--frames", ",".join(map(str, frames))))

    cli_args.extend(("-o", output_path))

    print("Command line call:")
    print(" \\\n".join(cli_args))
    print()

    with sub.Popen(cli_args) as proc:
        while proc.poll() is None:
            if waiting_callback is not None:

                if waiting_callback() == -1:
                    # -1 signals user cancellation
                    return "", False

            time.sleep(0.1)

        print(f"Process return code: {proc.returncode}")
        success = proc.returncode == 0

    return output_path, success


if __name__ == "__main__":

    app = QtWidgets.QApplication([])

    from sleap import Skeleton

    # skeleton = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    filename = "tests/data/json_format_v1/centered_pair.json"
    labels = Labels.load_file(filename)
    win = TrainingDialog(labels_filename=filename, labels=labels)

    win.frame_selection = {"clip": {labels.videos[0]: (1, 2, 3, 4)}}
    # win.training_editor_widget.set_fields_from_key_val_dict({
    #     "_backbone_name": "unet",
    #     "_heads_name": "centered_instance",
    # })
    #
    # win.training_editor_widget.form_widgets["model"].set_field_enabled("_heads_name", False)

    win.show()
    app.exec_()

"""
Run training/inference in background process via CLI.
"""
import abc
import attr
import os
import psutil
import json
import subprocess
import tempfile
import time
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Text, Tuple

from PySide2 import QtWidgets

from sleap import Labels, Video, LabeledFrame
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.nn import training
from sleap.nn.config import TrainingJobConfig


def kill_process(pid: int):
    """Force kill a running process and any child processes.

    Args:
        proc: A process ID.
    """
    proc_ = psutil.Process(pid)
    for subproc_ in proc_.children(recursive=True):
        subproc_.kill()
    proc_.kill()


@attr.s(auto_attribs=True)
class ItemForInference(abc.ABC):
    """
    Abstract base class for item on which we can run inference via CLI.

    Must have `path` and `cli_args` properties, used to build CLI call.
    """

    @property
    @abc.abstractmethod
    def path(self) -> Text:
        pass

    @property
    @abc.abstractmethod
    def cli_args(self) -> List[Text]:
        pass


@attr.s(auto_attribs=True)
class VideoItemForInference(ItemForInference):
    """
    Encapsulate data about video on which inference should run.

    This allows for inference on an arbitrary list of frames from video.

    Attributes:
        video: the :py:class:`Video` object (which already stores its own path)
        frames: list of frames for inference; if None, then all frames are used
        use_absolute_path: whether to use absolute path for inference cli call
    """

    video: Video
    frames: Optional[List[int]] = None
    use_absolute_path: bool = False

    @property
    def path(self):
        if self.use_absolute_path:
            return os.path.abspath(self.video.filename)
        return self.video.filename

    @property
    def cli_args(self):
        arg_list = list()
        arg_list.append(self.path)

        # TODO: better support for video params
        if hasattr(self.video.backend, "dataset") and self.video.backend.dataset:
            arg_list.extend(("--video.dataset", self.video.backend.dataset))

        if (
            hasattr(self.video.backend, "input_format")
            and self.video.backend.input_format
        ):
            arg_list.extend(("--video.input_format", self.video.backend.input_format))

        # -Y represents endpoint of [X, Y) range but inference cli expects
        # [X, Y-1] range (so add 1 since negative).
        frame_int_list = [i + 1 if i < 0 else i for i in self.frames]

        arg_list.extend(("--frames", ",".join(map(str, frame_int_list))))

        return arg_list


@attr.s(auto_attribs=True)
class DatasetItemForInference(ItemForInference):
    """
    Encapsulate data about frame selection based on dataset data.

    Attributes:
        labels_path: path to the saved :py:class:`Labels` dataset.
        frame_filter: which subset of frames to get from dataset, supports
            * "user"
            * "suggested"
        use_absolute_path: whether to use absolute path for inference cli call.
    """

    labels_path: str
    frame_filter: str = "user"
    use_absolute_path: bool = False

    @property
    def path(self):
        if self.use_absolute_path:
            return os.path.abspath(self.labels_path)
        return self.labels_path

    @property
    def cli_args(self):
        args_list = ["--labels", self.path]
        if self.frame_filter == "user":
            args_list.append("--only-labeled-frames")
        elif self.frame_filter == "suggested":
            args_list.append("--only-suggested-frames")
        return args_list


@attr.s(auto_attribs=True)
class ItemsForInference:
    """Encapsulates list of items for inference."""

    items: List[ItemForInference]
    total_frame_count: int

    def __len__(self):
        return len(self.items)

    @classmethod
    def from_video_frames_dict(
        cls, video_frames_dict: Dict[Video, List[int]], total_frame_count: int
    ):
        items = []
        for video, frames in video_frames_dict.items():
            if frames:
                items.append(VideoItemForInference(video=video, frames=frames))
        return cls(items=items, total_frame_count=total_frame_count)


@attr.s(auto_attribs=True)
class InferenceTask:
    """Encapsulates all data needed for running inference via CLI."""

    trained_job_paths: List[str]
    inference_params: Dict[str, Any] = attr.ib(default=attr.Factory(dict))
    labels: Optional[Labels] = None
    labels_filename: Optional[str] = None
    results: List[LabeledFrame] = attr.ib(default=attr.Factory(list))

    def make_predict_cli_call(
        self,
        item_for_inference: ItemForInference,
        output_path: Optional[str] = None,
        gui: bool = True,
    ) -> List[Text]:
        """Makes list of CLI arguments needed for running inference."""
        cli_args = ["sleap-track"]

        cli_args.extend(item_for_inference.cli_args)

        # TODO: encapsulate in inference item class
        if (
            not self.trained_job_paths
            and "tracking.tracker" in self.inference_params
            and self.labels_filename
        ):
            # No models so we must want to re-track previous predictions
            cli_args.extend(("--labels", self.labels_filename))

        # Make path where we'll save predictions (if not specified)
        if output_path is None:

            if self.labels_filename:
                # Make a predictions directory next to the labels dataset file
                predictions_dir = os.path.join(
                    os.path.dirname(self.labels_filename), "predictions"
                )
                os.makedirs(predictions_dir, exist_ok=True)
            else:
                # Dataset filename wasn't given, so save predictions in same dir
                # as the video
                predictions_dir = os.path.dirname(item_for_inference.video.filename)

            # Build filename with video name and timestamp
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            output_path = os.path.join(
                predictions_dir,
                f"{os.path.basename(item_for_inference.path)}.{timestamp}."
                "predictions.slp",
            )

        for job_path in self.trained_job_paths:
            cli_args.extend(("-m", job_path))

        optional_items_as_nones = (
            "tracking.target_instance_count",
            "tracking.kf_init_frame_count",
        )

        for key in optional_items_as_nones:
            if key in self.inference_params and self.inference_params[key] is None:
                del self.inference_params[key]

        # --tracking.kf_init_frame_count enables the kalman filter tracking
        # so if not set, then remove other (unused) args
        if "tracking.kf_init_frame_count" not in self.inference_params:
            if "tracking.kf_node_indices" in self.inference_params:
                del self.inference_params["tracking.kf_node_indices"]

        bool_items_as_ints = (
            "tracking.pre_cull_to_target",
            "tracking.post_connect_single_breaks",
        )

        for key in bool_items_as_ints:
            if key in self.inference_params:
                self.inference_params[key] = int(self.inference_params[key])

        for key, val in self.inference_params.items():
            if not key.startswith(("_", "outputs.", "model.", "data.")):
                cli_args.extend((f"--{key}", str(val)))

        cli_args.extend(("-o", output_path))

        if gui:
            cli_args.extend(("--verbosity", "json"))

        cli_args.extend(("--no-empty-frames",))

        return cli_args, output_path

    def predict_subprocess(
        self,
        item_for_inference: ItemForInference,
        append_results: bool = False,
        waiting_callback: Optional[Callable] = None,
        gui: bool = True,
    ) -> Tuple[Text, bool]:
        """Runs inference in a subprocess."""
        cli_args, output_path = self.make_predict_cli_call(item_for_inference, gui=gui)

        print("Command line call:")
        print(" ".join(cli_args))
        print()

        # Run inference CLI capturing output.
        with subprocess.Popen(cli_args, stdout=subprocess.PIPE) as proc:

            # Poll until finished.
            while proc.poll() is None:

                # Read line.
                line = proc.stdout.readline()
                line = line.decode().rstrip()

                is_json = False
                if line.startswith("{"):
                    try:
                        # Parse line.
                        line_data = json.loads(line)
                        is_json = True
                    except:
                        is_json = False

                if not is_json:
                    # Pass through non-json output.
                    print(line)
                    line_data = {}

                if waiting_callback is not None:
                    # Pass line data to callback.
                    ret = waiting_callback(**line_data)

                    if ret == "cancel":
                        # Stop if callback returned cancel signal.
                        kill_process(proc.pid)
                        print(f"Killed PID: {proc.pid}")
                        return "", "canceled"
                time.sleep(0.05)

            print(f"Process return code: {proc.returncode}")
            success = proc.returncode == 0

        if success and append_results:
            # Load frames from inference into results list
            new_inference_labels = Labels.load_file(output_path, match_to=self.labels)
            self.results.extend(new_inference_labels.labeled_frames)

        # Return "success" or return code if failed.
        ret = "success" if success else proc.returncode
        return output_path, ret

    def merge_results(self):
        """Merges result frames into labels dataset."""
        # Remove any frames without instances.
        new_lfs = list(filter(lambda lf: len(lf.instances), self.results))
        new_labels = Labels(new_lfs)

        # Remove potentially conflicting predictions from the base dataset.
        self.labels.remove_predictions(new_labels=new_labels)

        # Merge predictions into current labels dataset.
        _, _, new_conflicts = Labels.complex_merge_between(
            self.labels,
            new_labels=new_labels,
            unify=False,  # since we used match_to when loading predictions file
        )

        # new predictions should replace old ones
        Labels.finish_complex_merge(self.labels, new_conflicts)


def write_pipeline_files(
    output_dir: str,
    labels_filename: str,
    config_info_list: List[ConfigFileInfo],
    inference_params: Dict[str, Any],
    items_for_inference: ItemsForInference,
):
    """Writes the config files and scripts for manually running pipeline."""

    # Use absolute path for all files that aren't contained in the output dir.
    labels_filename = os.path.abspath(labels_filename)

    # Preserve current working directory and change working directory to the
    # output directory, so we can set local paths relative to that.
    old_cwd = os.getcwd()
    os.chdir(output_dir)

    new_cfg_filenames = []
    train_script = "#!/bin/bash\n"

    # Add head type to save path suffix to prevent overwriting.
    for cfg_info in config_info_list:
        if not cfg_info.dont_retrain:
            if (
                cfg_info.config.outputs.run_name_suffix is not None
                and len(cfg_info.config.outputs.run_name_suffix) > 0
            ):
                # Keep existing suffix if defined.
                suffix = "." + cfg_info.config.outputs.run_name_suffix
            else:
                suffix = ""

            # Add head name.
            suffix = "." + cfg_info.head_name + suffix

            # Update config.
            cfg_info.config.outputs.run_name_suffix = suffix

    training_jobs = []
    for cfg_info in config_info_list:
        if cfg_info.dont_retrain:
            # Use full absolute path to already trained model
            trained_path = os.path.normpath(os.path.join(old_cwd, cfg_info.path))
            new_cfg_filenames.append(trained_path)

        else:
            # We're training this model, so save config file...

            # First we want to set the run folder so that we know where to find
            # the model after it's trained.
            # We'll use local path to the output directory (cwd).
            # Note that setup_new_run_folder does things relative to cwd which
            # is the main reason we're setting it to the output directory rather
            # than just using normpath.
            # cfg_info.config.outputs.runs_folder = ""
            training.setup_new_run_folder(cfg_info.config.outputs)
            # training.setup_new_run_folder(
            #     cfg_info.config.outputs,
            #     # base_run_name=f"{model_type}.n={len(labels.user_labeled_frames)}",
            #     base_run_name=cfg_info.head_name,
            # )

            # Now we set the filename for the training config file
            new_cfg_filename = f"{cfg_info.head_name}.json"

            # Save the config file
            cfg_info.config.save_json(new_cfg_filename)

            # Keep track of the path where we'll find the trained model
            new_cfg_filenames.append(cfg_info.config.outputs.run_path)

            # Add a line to the script for training this model
            train_script += (
                f"sleap-train {new_cfg_filename} {os.path.basename(labels_filename)}\n"
            )

            # Setup job params
            training_jobs.append(
                {
                    "cfg": new_cfg_filename,
                    "run_path": Path(cfg_info.config.outputs.run_path).as_posix(),
                    "train_labels": os.path.basename(labels_filename),
                }
            )

    # Write the script to train the models which need to be trained
    with open(os.path.join(output_dir, "train-script.sh"), "w") as f:
        f.write(train_script)

    # Build the script for running inference
    inference_script = "#!/bin/bash\n"

    # Object with settings for inference
    inference_task = InferenceTask(
        labels_filename=labels_filename,
        trained_job_paths=new_cfg_filenames,
        inference_params=inference_params,
    )

    inference_jobs = []
    for item_for_inference in items_for_inference.items:
        if type(item_for_inference) == DatasetItemForInference:
            data_path = labels_filename
        else:
            data_path = item_for_inference.path

        # We want to save predictions in output dir so use local path
        prediction_output_path = f"{os.path.basename(data_path)}.predictions.slp"

        # Use absolute path to video
        item_for_inference.use_absolute_path = True

        # Get list of cli args
        cli_args, _ = inference_task.make_predict_cli_call(
            item_for_inference=item_for_inference,
            output_path=prediction_output_path,
        )
        # And join them into a single call to inference
        inference_script += " ".join(cli_args) + "\n"

        # Setup job params
        only_suggested_frames = False
        if type(item_for_inference) == DatasetItemForInference:
            only_suggested_frames = item_for_inference.frame_filter == "suggested"

        # TODO: support frame ranges, user-labeled frames
        tracking_args = {
            k: v for k, v in inference_params.items() if k.startswith("tracking.")
        }
        inference_jobs.append(
            {
                "data_path": os.path.basename(data_path),
                "models": [Path(p).as_posix() for p in new_cfg_filenames],
                "output_path": prediction_output_path,
                "type": "labels"
                if type(item_for_inference) == DatasetItemForInference
                else "video",
                "only_suggested_frames": only_suggested_frames,
                "tracking": tracking_args,
            }
        )

    # And write it
    with open(os.path.join(output_dir, "inference-script.sh"), "w") as f:
        f.write(inference_script)

    # Save jobs.yaml
    jobs = {"training": training_jobs, "inference": inference_jobs}
    with open(os.path.join(output_dir, "jobs.yaml"), "w") as f:
        yaml.dump(jobs, f)

    # Restore the working directory
    os.chdir(old_cwd)


def run_learning_pipeline(
    labels_filename: str,
    labels: Labels,
    config_info_list: List[ConfigFileInfo],
    inference_params: Dict[str, Any],
    items_for_inference: ItemsForInference,
) -> int:
    """Runs training (as needed) and inference.

    Args:
        labels_filename: Path to already saved current labels object.
        labels: The current labels object; results will be added to this.
        config_info_list: List of ConfigFileInfo with configs for training
            and inference.
        inference_params: Parameters to pass to inference.
        frames_to_predict: Dict that gives list of frame indices for each video.

    Returns:
        Number of new frames added to labels.

    """

    save_viz = inference_params.get("_save_viz", False)

    # Train the TrainingJobs
    trained_job_paths = run_gui_training(
        labels_filename=labels_filename,
        labels=labels,
        config_info_list=config_info_list,
        gui=True,
        save_viz=save_viz,
    )

    # Check that all the models were trained
    if None in trained_job_paths.values():
        return -1

    inference_task = InferenceTask(
        labels=labels,
        labels_filename=labels_filename,
        trained_job_paths=list(trained_job_paths.values()),
        inference_params=inference_params,
    )

    # Run the Predictor for suggested frames
    new_labeled_frame_count = run_gui_inference(inference_task, items_for_inference)

    return new_labeled_frame_count


def run_gui_training(
    labels_filename: str,
    labels: Labels,
    config_info_list: List[ConfigFileInfo],
    gui: bool = True,
    save_viz: bool = False,
) -> Dict[Text, Text]:
    """
    Runs training for each training job.

    Args:
        labels: Labels object from which we'll get training data.
        config_info_list: List of ConfigFileInfo with configs for training.
        gui: Whether to show gui windows and process gui events.
        save_viz: Whether to save visualizations from training.

    Returns:
        Dictionary, keys are head name, values are path to trained config.
    """

    trained_job_paths = dict()

    if gui:
        from sleap.gui.widgets.monitor import LossViewer
        from sleap.gui.widgets.imagedir import QtImageDirectoryWidget

        # open training monitor window
        win = LossViewer()
        win.resize(600, 400)
        win.show()

    for config_info in config_info_list:
        if config_info.dont_retrain:

            if not config_info.has_trained_model:
                raise ValueError(
                    "Config is set to not retrain but no trained model found: "
                    f"{config_info.path}"
                )

            print(
                f"Using already trained model for {config_info.head_name}: "
                f"{config_info.path}"
            )

            trained_job_paths[config_info.head_name] = config_info.path

        else:
            job = config_info.config
            model_type = config_info.head_name

            # We'll pass along the list of paths we actually used for loading
            # the videos so that we don't have to rely on the paths currently
            # saved in the labels file for finding videos.
            video_path_list = [video.filename for video in labels.videos]

            # Update save dir and run name for job we're about to train
            # so we have access to them here (rather than letting
            # train_subprocess update them).
            # training.Trainer.set_run_name(job, labels_filename)
            job.outputs.runs_folder = os.path.join(
                os.path.dirname(labels_filename), "models"
            )
            training.setup_new_run_folder(
                job.outputs,
                base_run_name=f"{model_type}.n={len(labels.user_labeled_frames)}",
            )

            if gui:
                print("Resetting monitor window.")
                win.reset(what=str(model_type), config=job)
                win.setWindowTitle(f"Training Model - {str(model_type)}")
                win.set_message(f"Preparing to run training...")
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
                    if win.canceled:
                        return "cancel"

            # Run training
            trained_job_path, ret = train_subprocess(
                job_config=job,
                labels_filename=labels_filename,
                video_paths=video_path_list,
                waiting_callback=waiting,
                save_viz=save_viz,
            )

            if ret == "success":
                # get the path to the resulting TrainingJob file
                trained_job_paths[model_type] = trained_job_path
                print(f"Finished training {str(model_type)}.")
            elif ret == "canceled":
                if gui:
                    win.close()
                print("Deleting canceled run data:", trained_job_path)
                shutil.rmtree(trained_job_path, ignore_errors=True)
                trained_job_paths[model_type] = None
                break
            else:
                if gui:
                    win.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occurred while training {str(model_type)}. "
                        "Your command line terminal may have more information about "
                        "the error."
                    ).exec_()
                trained_job_paths[model_type] = None

    if gui:
        # close training monitor window
        win.close()

    return trained_job_paths


def run_gui_inference(
    inference_task: InferenceTask,
    items_for_inference: ItemsForInference,
    gui: bool = True,
) -> int:
    """Run inference on specified frames using models from training_jobs.

    Args:
        inference_task: Encapsulates information needed for running inference,
            such as labels dataset and models.
        items_for_inference: Encapsulates information about the videos (etc.)
            on which we're running inference.
        gui: Whether to show gui windows and process gui events.

    Returns:
        Number of new frames added to labels.
    """

    if gui:
        progress = QtWidgets.QProgressDialog(
            "Initializing...",
            "Cancel",
            0,
            1,
        )
        progress.show()
        QtWidgets.QApplication.instance().processEvents()

    # Make callback to process events while running inference
    def waiting(
        n_processed: Optional[int] = None,
        n_total: Optional[int] = None,
        elapsed: Optional[float] = None,
        rate: Optional[float] = None,
        eta: Optional[float] = None,
        current_item: Optional[int] = None,
        total_items: Optional[int] = None,
        **kwargs,
    ) -> str:
        if gui:
            QtWidgets.QApplication.instance().processEvents()
            if n_total is not None:
                progress.setMaximum(n_total)
            if n_processed is not None:
                progress.setValue(n_processed)

            msg = "Predicting..."

            if n_processed is not None and n_total is not None:
                msg = f"<b>Predicted:</b> {n_processed:,}/{n_total:,}"

            # Show time elapsed?
            if rate is not None and eta is not None:
                eta_mins, eta_secs = divmod(eta, 60)
                if eta_mins > 60:
                    eta_hours, eta_mins = divmod(eta_mins, 60)
                    eta_str = f"{int(eta_hours)} hours, {int(eta_mins):02} mins"
                elif eta_mins > 0:
                    eta_str = f"{int(eta_mins)} mins, {int(eta_secs):02} secs"
                else:
                    eta_str = f"{int(eta_secs):02} secs"
                msg += f"<br><b>ETA:</b> {eta_str}"
                msg += f"<br><b>FPS:</b> {rate:.1f}"

            msg = msg.replace(" ", "&nbsp;")

            progress.setLabelText(msg)
            QtWidgets.QApplication.instance().processEvents()

            if progress.wasCanceled():
                return "cancel"

    for i, item_for_inference in enumerate(items_for_inference.items):

        def waiting_item(**kwargs):
            kwargs["current_item"] = i
            kwargs["total_items"] = len(items_for_inference.items)
            return waiting(**kwargs)

        # Run inference for desired frames in this video.
        predictions_path, ret = inference_task.predict_subprocess(
            item_for_inference,
            append_results=True,
            waiting_callback=waiting_item,
            gui=gui,
        )

        if ret == "canceled":
            return -1
        elif ret != "success":
            if gui:
                QtWidgets.QMessageBox(
                    text=(
                        "An error occcured during inference. Your command line "
                        "terminal may have more information about the error."
                    )
                ).exec_()
            return -1

    inference_task.merge_results()
    if gui:
        progress.close()
    return len(inference_task.results)


def train_subprocess(
    job_config: TrainingJobConfig,
    labels_filename: str,
    video_paths: Optional[List[Text]] = None,
    waiting_callback: Optional[Callable] = None,
    save_viz: bool = False,
):
    """Runs training inside subprocess."""
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
            "sleap-train",
            training_job_path,
            labels_filename,
            "--zmq",
        ]

        if save_viz:
            cli_args.append("--save_viz")

        # Use cli arg since cli ignores setting in config
        if job_config.outputs.tensorboard.write_logs:
            cli_args.append("--tensorboard")

        # Run training in a subprocess.
        print(cli_args)
        proc = subprocess.Popen(cli_args)

        # Wait till training is done, calling a callback if given.
        while proc.poll() is None:
            if waiting_callback is not None:
                ret = waiting_callback()
                if ret == "cancel":
                    print("Canceling training...")
                    kill_process(proc.pid)
                    print(f"Killed PID: {proc.pid}")
                    return run_path, "canceled"
            time.sleep(0.1)

        # Check return code.
        if proc.returncode == 0:
            ret = "success"
        else:
            ret = proc.returncode

    print("Run Path:", run_path)

    return run_path, ret

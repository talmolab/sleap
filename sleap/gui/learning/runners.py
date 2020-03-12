import copy
from collections import defaultdict

import numpy as np
import os
import subprocess as sub
import tempfile
import time

from datetime import datetime

from sleap import Labels, Video
from sleap.gui.overlays.confmaps import demo_confmaps
from sleap.gui.overlays.pafs import demo_pafs
from sleap.nn import training
from sleap.nn.config import (
    TrainingJobConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
)
from sleap.nn.data import pipelines
from sleap.nn.data.instance_cropping import find_instance_crop_size
from sleap.nn.data.providers import LabelsReader
from sleap.gui.learning.configs import ConfigFileInfo

from typing import Any, Callable, Dict, List, Optional, Text

from PySide2 import QtWidgets

from sleap.nn.data.resizing import Resizer

SKIP_TRAINING = False


def run_datagen_preview(labels: Labels, config_info_list: List[ConfigFileInfo]):

    labels_reader = LabelsReader(labels)

    last_win = None

    def show_win(results: dict, key: Text, video: Video):
        nonlocal last_win

        if key == "confmap":
            win = demo_confmaps(results[key], video)
        elif key == "paf":
            win = demo_pafs(results[key], video)
        else:
            raise ValueError(f"Cannot show preview window for {key}")

        win.activateWindow()
        win.resize(300, 300)
        if last_win:
            win.move(last_win.rect().right() + 20, 200)
        else:
            win.move(300, 300)

        last_win = win

    for cfg_info in config_info_list:
        results = make_datagen_results(labels_reader, cfg_info.config)

        if "image" in results:
            vid = Video.from_numpy(results["image"])
            if "confmap" in results:
                show_win(results, "confmap", vid)

            if "paf" in results:
                show_win(results, "paf", vid)


def make_datagen_results(reader: LabelsReader, cfg: TrainingJobConfig):
    cfg = copy.deepcopy(cfg)
    output_keys = dict()

    if cfg.data.preprocessing.pad_to_stride is None:
        cfg.data.preprocessing.pad_to_stride = (
            cfg.model.backbone.which_oneof().max_stride
        )

    pipeline = pipelines.Pipeline(reader)
    pipeline += Resizer.from_config(cfg.data.preprocessing)

    head_config = cfg.model.heads.which_oneof()
    if isinstance(head_config, CentroidsHeadConfig):
        pipeline += pipelines.InstanceCentroidFinder.from_config(
            cfg.data.instance_cropping, skeletons=reader.labels.skeletons
        )
        pipeline += pipelines.MultiConfidenceMapGenerator(
            sigma=cfg.model.heads.centroid.sigma,
            output_stride=cfg.model.heads.centroid.output_stride,
            centroids=True,
        )

        output_keys["image"] = "image"
        output_keys["confmap"] = "centroid_confidence_maps"

    elif isinstance(head_config, CenteredInstanceConfmapsHeadConfig):
        if cfg.data.instance_cropping.crop_size is None:
            cfg.data.instance_cropping.crop_size = find_instance_crop_size(
                labels=reader.labels,
                padding=cfg.data.instance_cropping.crop_size_detection_padding,
                maximum_stride=cfg.model.backbone.which_oneof().max_stride,
            )

        pipeline += pipelines.InstanceCentroidFinder.from_config(
            cfg.data.instance_cropping, skeletons=reader.labels.skeletons
        )
        pipeline += pipelines.InstanceCropper.from_config(cfg.data.instance_cropping)
        pipeline += pipelines.InstanceConfidenceMapGenerator(
            sigma=cfg.model.heads.centered_instance.sigma,
            output_stride=cfg.model.heads.centered_instance.output_stride,
        )

        output_keys["image"] = "instance_image"
        output_keys["confmap"] = "instance_confidence_maps"

    elif isinstance(head_config, MultiInstanceConfig):
        output_keys["image"] = "image"
        output_keys["confmap"] = "confidence_maps"
        output_keys["paf"] = "confidence_maps"

        pipeline += pipelines.MultiConfidenceMapGenerator(
            sigma=cfg.model.heads.multi_instance.confmaps.sigma,
            output_stride=cfg.model.heads.multi_instance.confmaps.output_stride,
        )
        pipeline += pipelines.PartAffinityFieldsGenerator(
            sigma=cfg.model.heads.multi_instance.pafs.sigma,
            output_stride=cfg.model.heads.multi_instance.pafs.output_stride,
            skeletons=reader.labels.skeletons,
        )

    ds = pipeline.make_dataset()

    output_lists = defaultdict(list)
    i = 0
    for example in ds:
        for key, from_key in output_keys.items():
            output_lists[key].append(example[from_key])
        i += 1
        if i == 10:
            break

    outputs = dict()
    for key in output_lists.keys():
        outputs[key] = np.stack(output_lists[key])

    return outputs


def run_learning_pipeline(
    labels_filename: str,
    labels: Labels,
    config_info_list: List[ConfigFileInfo],
    inference_params: Dict[str, Any],
    frames_to_predict: Dict[Video, List[int]] = None,
) -> int:
    """Run training (as needed) and inference.

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
        labels_filename, config_info_list, gui=True, save_viz=save_viz
    )

    # Check that all the models were trained
    if None in trained_job_paths.values():
        return -1

    trained_job_paths = list(trained_job_paths.values())

    # Run the Predictor for suggested frames
    new_labeled_frame_count = run_gui_inference(
        labels=labels,
        trained_job_paths=trained_job_paths,
        inference_params=inference_params,
        frames_to_predict=frames_to_predict,
        labels_filename=labels_filename,
    )

    return new_labeled_frame_count


def run_gui_training(
    labels_filename: str,
    config_info_list: List[ConfigFileInfo],
    gui: bool = True,
    save_viz: bool = False,
) -> Dict[Text, Text]:
    """
    Run training for each training job.

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
        from sleap.nn.monitor import LossViewer
        from sleap.gui.imagedir import QtImageDirectoryWidget

        # open training monitor window
        win = LossViewer()
        win.resize(600, 400)
        win.show()

    for config_info in config_info_list:
        if config_info.dont_retrain:

            if not config_info.has_trained_model:
                raise ValueError(
                    f"Config is set to not retrain but no trained model found: {config_info.path}"
                )

            print(
                f"Using already trained model for {config_info.head_name}: {config_info.path}"
            )

            trained_job_paths[config_info.head_name] = config_info.path

        else:
            job = config_info.config
            model_type = config_info.head_name

            # Update save dir and run name for job we're about to train
            # so we have access to them here (rather than letting
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
                trained_job_paths[model_type] = trained_job_path
                print(f"Finished training {str(model_type)}.")
            else:
                if gui:
                    win.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occurred while training {str(model_type)}. Your command line terminal may have more information about the error."
                    ).exec_()
                trained_job_paths[model_type] = None

    if gui:
        # close training monitor window
        win.close()

    return trained_job_paths


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

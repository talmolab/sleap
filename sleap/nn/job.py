"""Serializable training job to specify training parameters."""

import os
import json
from jsmin import jsmin
import attr
from datetime import datetime
from typing import Union, List
from pathlib import Path, PureWindowsPath

from sleap import Skeleton
from sleap.nn import model


@attr.s(auto_attribs=True)
class TrainerConfig:
    # Data generation:
    val_size: float = 0.1
    shuffle: bool = True
    shuffle_buffer_size: int = -1
    scale: float = 1.0
    sigma: float = 5.0
    instance_crop: bool = False
    bounding_box_size: int = 0
    instance_crop_use_ctr_node: bool = False
    instance_crop_ctr_node_ind: int = 0
    instance_crop_padding: int = 0

    # Training loop:
    batch_size: int = 8
    num_epochs: int = 100
    steps_per_epoch: int = -1
    val_steps_per_epoch: int = -1

    # Augmentation:
    augment_rotate: bool = True
    augment_rotation: float = 180.0
    augment_scale: bool = False
    augment_scale_min: float = 0.9
    augment_scale_max: float = 1.1
    augment_uniform_noise: bool = False
    augment_uniform_noise_min_val: float = 0.0
    augment_uniform_noise_max_val: float = 0.1
    augment_gaussian_noise: bool = False
    augment_gaussian_noise_mean: float = 0.05
    augment_gaussian_noise_stddev: float = 0.1

    # Optimization:
    optimizer: str = "adam"
    amsgrad: bool = True
    learning_rate: float = 1e-3

    # Training callbacks:
    reduce_lr_on_plateau: bool = True
    reduce_lr_min_delta: float = 1e-6
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: float = 5
    reduce_lr_cooldown: float = 3
    reduce_lr_min_lr: float = 1e-8
    early_stopping: bool = True
    early_stopping_min_delta: float = 1e-8
    early_stopping_patience: float = 15
    monitor_metric_name: str = "val_loss"

    # Checkpointing callbacks:
    csv_logging: bool = False
    csv_log_filename: str = "training_log.csv"
    save_every_epoch: bool = False
    save_best_val: bool = True
    save_final_model: bool = True

    # Deprecated:
    shuffle_initially: bool = True
    shuffle_every_epoch: bool = True
    min_crop_size: int = 32
    negative_samples: int = 10

    @property
    def input_scale(self):
        return self.scale


@attr.s(auto_attribs=True)
class TrainingJob:
    """
    A simple class that groups a model with a trainer to represent a record of a
    call to Trainer.train().

    Args:
        model: The SLEAP Model that was trained.
        trainer: The Trainer that was used to train the model.
        labels_filename: The name of the labels file using to run this training job.
        run_name: The run_name value passed to Trainer.train for this training run.
        save_dir: The save_dir value passed to Trainer.train for this training run.
        best_model_filename: The relative path (from save_dir) to the Keras model file
        that had best validation loss. Set to None when Trainer.save_best_val is False
        or if save_dir is None.
        newest_model_filename: The relative path (from save_dir) to the Keras model file
        from the state of the model after the last epoch run. Set to None when
        Trainer.save_every_epoch is False or save_dir is None.
        final_model_filename: The relative path (from save_dir) to the Keras model file
        from the final state of training. Set to None if save_dir is None. This model
        file is not created until training is finished.
    """

    model: model.Model
    trainer: TrainerConfig
    labels_filename: Union[str, None] = None
    val_set_filename: Union[str, None] = None
    test_set_filename: Union[str, None] = None
    save_dir: Union[str, None] = None
    run_name: Union[str, None] = None
    best_model_filename: Union[str, None] = None
    newest_model_filename: Union[str, None] = None
    final_model_filename: Union[str, None] = None

    @property
    def train_set_filename(self):
        return self.labels_filename

    @property
    def input_scale(self):
        return self.trainer.input_scale

    @property
    def run_path(self):
        if self.save_dir is not None and self.run_name is not None:
            return os.path.join(self.save_dir, self.run_name)
        else:
            return None

    def new_run_name(
        self,
        prefix=None,
        timestamp=True,
        backbone=True,
        output_type=True,
        suffix=None,
        check_existing=False,
    ) -> str:
        """Generates a new run name."""

        name_tokens = []
        if prefix is not None:
            if isinstance(prefix, str):
                prefix = [prefix]
            name_tokens.extend(prefix)

        if timestamp:
            name_tokens.append(datetime.now().strftime("%y%m%d_%H%M%S"))

        if backbone:
            name_tokens.append(self.model.backbone_name)

        if output_type:
            name_tokens.append(str(self.model.output_type))

        if suffix is not None:
            if isinstance(suffix, str):
                suffix = [suffix]
            name_tokens.extend(suffix)

        run_name = ".".join(name_tokens)

        if check_existing:
            if self.save_dir is None:
                raise ValueError(
                    "Cannot check for run path existance if save_dir is not set."
                )

            i = 0
            while os.path.exists(os.path.join(self.save_dir, run_name)):
                i += 1
                run_name = ".".join(name_tokens + [f"{i}"])

        return run_name

    @property
    def model_path(self):
        """Returns a path to an existing model, with preference for best model if it exists.

        Raises:
            ValueError: if neither the best model or final model could be found.
        """

        if self.run_path is None:
            raise ValueError("Training job has no run path specified.")

        # Try the best model first.
        if self.best_model_filename is not None:
            model_path = os.path.join(self.run_path, self.best_model_filename)
            if os.path.exists(model_path):
                return model_path

            model_path = os.path.join(self.save_dir, self.best_model_filename)
            if os.path.exists(model_path):
                return model_path

        # Then the newest model.
        if self.newest_model_filename is not None:
            model_path = os.path.join(self.run_path, self.newest_model_filename)
            if os.path.exists(model_path):
                return model_path

            model_path = os.path.join(self.save_dir, self.newest_model_filename)
            if os.path.exists(model_path):
                return model_path

        # Then the final model.
        if self.final_model_filename is not None:
            model_path = os.path.join(self.run_path, self.final_model_filename)
            if os.path.exists(model_path):
                return model_path

            model_path = os.path.join(self.save_dir, self.final_model_filename)
            if os.path.exists(model_path):
                return model_path

        # Raise error if all fail.
        raise ValueError(f"Could not find a saved model in run path: {self.run_path}")

    @property
    def is_trained(self):
        if self.run_path is None:
            return False
        if os.path.exists(self.model_path):
            return True
        return False

    @staticmethod
    def _to_dicts(training_job: "TrainingJob"):

        # We have some skeletons to deal with, make sure to setup a Skeleton cattr.
        my_cattr = Skeleton.make_cattr()
        dicts = my_cattr.unstructure(training_job)

        return dicts

    @staticmethod
    def save_json(training_job: "TrainingJob", filename: str):
        """
        Save a training run to a JSON file.

        Args:
            training_job: The TrainingJob instance to save.
            filename: The filename to save the JSON to.
        """

        with open(filename, "w") as file:
            dicts = TrainingJob._to_dicts(training_job)
            json_str = json.dumps(dicts, indent=4)
            file.write(json_str)

    def save(self, filename: str):
        """Save this training job to a JSON file."""
        TrainingJob.save_json(self, filename=filename)

    @classmethod
    def load_json(cls, filename: str = None, json_string: str = None):
        """
        Load a training run from a JSON file.

        Args:
            filename: The file to load the JSON from.

        Returns:
            A TrainingJob instance constructed from JSON in filename.
        """

        if filename is not None:

            # Check for old directory structure ({run_name}.json)
            if os.path.exists(f"{filename}.json"):
                filename = f"{filename}.json"

            # Check for training job file if save directory specified.
            if os.path.isdir(filename):
                filename = os.path.join(filename, "training_job.json")

            # Open and parse the JSON in filename
            with open(filename, "r") as f:
                dicts = json.loads(jsmin(f.read()))

        elif json_string is not None:
            dicts = json.loads(jsmin(json_string))

        else:
            raise ValueError("Filename to JSON file or a JSON string must be provided.")

        # We have some skeletons to deal with, make sure to setup a Skeleton cattr.
        converter = Skeleton.make_cattr()

        # Structure the nested skeletons if we have any.
        if "model" in dicts:
            if (
                "skeletons" in dicts["model"]
                and dicts["model"]["skeletons"] is not None
            ):
                dicts["model"]["skeletons"] = converter.structure(
                    dicts["model"]["skeletons"], List[Skeleton]
                )

            else:
                dicts["model"]["skeletons"] = []

        # Setup structuring hook for unambiguous backbone class resolution.
        converter.register_structure_hook(model.Model, model.Model._structure_model)

        # Build classes.
        run = converter.structure(dicts, cls)

        # If we can't find save_dir for job, set it to path of json we're loading.
        if run.run_path is not None and filename is not None:
            if not os.path.exists(run.run_path):

                # Check for old pattern where {run_name}.json is not inside
                # the {run_name} directory but next to it. We have to check
                # because the code for the standard (new) pattern doesn't
                # work in this case (it uses the parent directory of the json
                # as the run_name and the resulting path exists but is wrong).
                skip_new_pattern = False
                if filename.endswith(".json"):
                    if not filename.endswith("training_job.json"):
                        skip_new_pattern = True

                # Try the standard pattern:
                # {save_dir}/{run_name}/{job_json} -> run_path = {save_dir}/{run_name}/
                save_dir, run_name = os.path.split(os.path.dirname(filename))
                new_run_path = os.path.join(save_dir, run_name)
                if not skip_new_pattern and os.path.exists(new_run_path):
                    run.save_dir = save_dir
                    run.run_name = run_name

                else:
                    # Next, try the old pattern:
                    # {save_dir}/{run_name}.json -> run_path = {save_dir}/{run_name}/
                    save_dir, run_name = os.path.split(os.path.splitext(filename)[0])
                    new_run_path = os.path.join(save_dir, run_name)
                    if os.path.exists(new_run_path):
                        run.save_dir = save_dir
                        run.run_name = run_name

        if run.best_model_filename is not None:
            run.best_model_filename = cls._fix_path(run.best_model_filename)
        if run.newest_model_filename is not None:
            run.newest_model_filename = cls._fix_path(run.newest_model_filename)
        if run.final_model_filename is not None:
            run.final_model_filename = cls._fix_path(run.final_model_filename)

        return run

    @classmethod
    def _fix_path(cls, path):
        # convert from Windows path if necessary
        if path is not None:
            if path.find("\\"):
                path = str(Path(PureWindowsPath(path)))
        return path

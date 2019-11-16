"""Serializable training job to specify training parameters."""

import os
import json
import attr
from typing import Union, List
from pathlib import Path, PureWindowsPath

from sleap import Skeleton
from sleap.nn import model


@attr.s(auto_attribs=True)
class TrainerConfig:
    # Data generation:
    val_size: float = 0.1
    shuffle: bool = True
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
    steps_per_epoch: int = 200
    val_steps_per_epoch: int = 20

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
    early_stopping_patience: float = 3
    monitor_metric_name: str = "val_loss"

    # Checkpointing callbacks:
    save_every_epoch: bool = False
    save_best_val: bool = True

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
    run_name: Union[str, None] = None
    save_dir: Union[str, None] = None
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
    def model_path(self):
        """Returns a path to an existing model, with preference for best model if it exists.

        Raises:
            ValueError: if neither the best model or final model could be found.
        """

        # Try the best model first.
        model_path = os.path.join(self.save_dir, self.best_model_filename)

        # Try the final model if that didn't exist.
        if not os.path.exists(model_path):
            model_path = os.path.join(self.save_dir, self.final_model_filename)

        # Raise error if both fail.
        if not os.path.exists(model_path):
            raise ValueError(
                f"Could not find a saved model in job directory: {self.save_dir}"
            )

        return model_path

    @property
    def run_path(self):
        if self.save_dir is not None and self.run_name is not None:
            return os.path.join(self.save_dir, self.run_name)
        else:
            return None

    @property
    def is_trained(self):
        if self.final_model_filename is not None:
            path = os.path.join(self.save_dir, self.final_model_filename)
            if os.path.exists(path):
                return True
        return False

    @staticmethod
    def save_json(training_job: "TrainingJob", filename: str):
        """
        Save a training run to a JSON file.

        Args:
            training_job: The TrainingJob instance to save.
            filename: The filename to save the JSON to.

        Returns:
            None
        """

        with open(filename, "w") as file:

            # We have some skeletons to deal with, make sure to setup a Skeleton cattr.
            my_cattr = Skeleton.make_cattr()
            dicts = my_cattr.unstructure(training_job)
            json_str = json.dumps(dicts)
            file.write(json_str)

    @classmethod
    def load_json(cls, filename: str):
        """
        Load a training run from a JSON file.

        Args:
            filename: The file to load the JSON from.

        Returns:
            A TrainingJob instance constructed from JSON in filename.
        """

        # Check for training job file if save directory specified.
        if os.path.isdir(filename):
            filename = os.path.join(filename, "training_job.json")

        # Open and parse the JSON in filename
        with open(filename, "r") as f:
            dicts = json.load(f)

        # We have some skeletons to deal with, make sure to setup a Skeleton cattr.
        converter = Skeleton.make_cattr()

        # Structure the nested skeletons if we have any.
        if ("model" in dicts) and ("skeletons" in dicts["model"]):
            if dicts["model"]["skeletons"]:
                dicts["model"]["skeletons"] = converter.structure(
                    dicts["model"]["skeletons"], List[Skeleton]
                )

            else:
                dicts["model"]["skeletons"] = []

        # Setup structuring hook for unambiguous backbone class resolution.
        converter.register_structure_hook(model.Model, model.Model._structure_model)

        # Build classes.
        run = converter.structure(dicts, cls)

        # if we can't find save_dir for job, set it to path of json we're loading
        if run.save_dir is not None:
            if not os.path.exists(run.save_dir):
                run.save_dir = os.path.dirname(filename)

                run.final_model_filename = cls._fix_path(run.final_model_filename)
                run.best_model_filename = cls._fix_path(run.best_model_filename)
                run.newest_model_filename = cls._fix_path(run.newest_model_filename)

        return run

    @classmethod
    def _fix_path(cls, path):
        # convert from Windows path if necessary
        if path is not None:
            if path.find("\\"):
                path = Path(PureWindowsPath(path))
        return path

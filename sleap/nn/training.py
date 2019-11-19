"""SLEAP model training."""

import os
import attr
import argparse
import json
from pkg_resources import Requirement, resource_filename
from typing import Union, Dict, List, Text, Tuple
from time import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from sleap import Labels, Skeleton
from sleap.nn import callbacks
from sleap.nn import data
from sleap.nn import job
from sleap.nn import model


@attr.s(auto_attribs=True)
class Trainer:
    training_job: job.TrainingJob

    tensorboard: bool = False
    tensorboard_freq: Union[Text, int] = "epoch"
    tensorboard_dir: Union[Text, None] = None
    zmq: bool = False
    control_zmq_port: int = 9000
    progress_report_zmq_port: int = 9001
    verbosity: int = 2

    _img_shape: Tuple[int, int, int] = None
    _n_output_channels: int = None
    _train: data.TrainingData = None
    _val: data.TrainingData = None
    _test: data.TrainingData = None
    _ds_train: tf.data.Dataset = None
    _ds_val: tf.data.Dataset = None
    _ds_test: tf.data.Dataset = None
    _simple_skeleton: data.SimpleSkeleton = None
    _model: tf.keras.Model = None
    _optimizer: tf.keras.optimizers.Optimizer = None
    _loss_fn: tf.keras.losses.Loss = None
    _training_callbacks: List[tf.keras.callbacks.Callback] = None
    _history: dict = None

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def n_output_channels(self):
        return self._n_output_channels

    @property
    def data_train(self):
        return self._train

    @property
    def data_val(self):
        return self._val

    @property
    def data_test(self):
        return self._test

    @property
    def ds_train(self):
        return self._ds_train

    @property
    def ds_val(self):
        return self._ds_val

    @property
    def ds_test(self):
        return self._ds_test

    @property
    def simple_skeleton(self):
        return self._simple_skeleton

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def training_callbacks(self):
        return self._training_callbacks

    @property
    def history(self):
        return self._history

    def setup_data(
        self,
        labels_train: Union[Labels, Text] = None,
        labels_val: Union[Labels, Text] = None,
        labels_test: Union[Labels, Text] = None,
        data_train: Union[data.TrainingData, Text] = None,
        data_val: Union[data.TrainingData, Text] = None,
        data_test: Union[data.TrainingData, Text] = None,
    ):

        train = labels_train
        if train is None:
            train = self.training_job.train_set_filename
        if train is not None and isinstance(train, str):
            if self.verbosity > 0:
                print(f"Loading labels: {train}")
            train = Labels.load_file(train)
        if train is not None:
            train = data.TrainingData.from_labels(train)
        if train is None:
            train = data_train
        if train is not None and isinstance(train, str):
            if self.verbosity > 0:
                print(f"Loading data: {train}")
            train = data.TrainingData.load_file(train)
        if train is None:
            raise ValueError("Training data was not specified.")

        val = labels_val
        if val is None:
            val = self.training_job.val_set_filename
        if val is not None and isinstance(val, str):
            if self.verbosity > 0:
                print(f"Loading labels: {val}")
            val = Labels.load_file(val)
        if val is not None:
            val = data.TrainingData.from_labels(val)
        if val is None:
            val = data_val
        if val is not None and isinstance(val, str):
            if self.verbosity > 0:
                print(f"Loading data: {val}")
            val = data.TrainingData.load_file(val)
        if val is None and self.training_job.trainer.val_size is not None:
            train, val = data.split_training_data(
                train, first_split_fraction=self.training_job.trainer.val_size
            )
        if val is None:
            raise ValueError("Validation set or fraction must be specified.")

        test = labels_test
        if test is None:
            test = self.training_job.test_set_filename
        if test is not None and isinstance(test, str):
            if self.verbosity > 0:
                print(f"Loading labels: {test}")
            test = Labels.load_file(test)
        if test is not None:
            test = data.TrainingData.from_labels(test)
        if test is None:
            test = data_test
        if test is not None and isinstance(test, str):
            if self.verbosity > 0:
                print(f"Loading data: {test}")
            test = data.TrainingData.load_file(test)

        # Setup initial zipped datasets.
        ds_train = train.to_ds()
        ds_val = val.to_ds()
        ds_test = None
        if test is not None:
            ds_test = test.to_ds()

        # Adjust for input scaling and add padding to the model's minimum multiple.
        ds_train = data.adjust_dataset_input_scale(
            ds_train,
            input_scale=self.training_job.input_scale,
            min_multiple=self.training_job.model.input_min_multiple,
            normalize_image=False,
        )
        ds_val = data.adjust_dataset_input_scale(
            ds_val,
            input_scale=self.training_job.input_scale,
            min_multiple=self.training_job.model.input_min_multiple,
            normalize_image=False,
        )

        if ds_test is not None:
            ds_test = data.adjust_dataset_input_scale(
                ds_test,
                input_scale=self.training_job.input_scale,
                min_multiple=self.training_job.model.input_min_multiple,
                normalize_image=False,
            )

        # Cache the data with the current transformations.
        # ds_train = ds_train.cache()
        # ds_val = ds_val.cache()
        # if ds_test is not None:
        #     ds_test = ds_test.cache()

        # Apply augmentations.
        aug_params = dict(
            rotate=self.training_job.trainer.augment_rotate,
            rotation_min_angle=-self.training_job.trainer.augment_rotation,
            rotation_max_angle=self.training_job.trainer.augment_rotation,
            scale=self.training_job.trainer.augment_scale,
            scale_min=self.training_job.trainer.augment_scale_min,
            scale_max=self.training_job.trainer.augment_scale_max,
            uniform_noise=self.training_job.trainer.augment_uniform_noise,
            min_noise_val=self.training_job.trainer.augment_uniform_noise_min_val,
            max_noise_val=self.training_job.trainer.augment_uniform_noise_max_val,
            gaussian_noise=self.training_job.trainer.augment_gaussian_noise,
            gaussian_noise_mean=self.training_job.trainer.augment_gaussian_noise_mean,
            gaussian_noise_stddev=self.training_job.trainer.augment_gaussian_noise_stddev,
        )
        ds_train = data.augment_dataset(ds_train, **aug_params)
        ds_val = data.augment_dataset(ds_val, **aug_params)

        if self.training_job.trainer.instance_crop:
            # Crop around instances.

            if (
                self.training_job.trainer.bounding_box_size is None
                or self.training_job.trainer.bounding_box_size <= 0
            ):
                # Estimate bounding box size from the data if not specified.
                # TODO: Do this earlier with more points if available.
                box_size = data.estimate_instance_crop_size(
                    train.points,
                    min_multiple=self.training_job.model.input_min_multiple,
                    padding=self.training_job.trainer.instance_crop_padding,
                )
                self.training_job.trainer.bounding_box_size = box_size

            crop_params = dict(
                box_height=self.training_job.trainer.bounding_box_size,
                box_width=self.training_job.trainer.bounding_box_size,
                use_ctr_node=self.training_job.trainer.instance_crop_use_ctr_node,
                ctr_node_ind=self.training_job.trainer.instance_crop_ctr_node_ind,
                normalize_image=True,
            )
            ds_train = data.instance_crop_dataset(ds_train, **crop_params)
            ds_val = data.instance_crop_dataset(ds_val, **crop_params)
            if ds_test is not None:
                ds_test = data.instance_crop_dataset(ds_test, **crop_params)

        else:
            # We're not instance cropping, so at this point the images are still not
            # normalized. Let's account for that before moving on.
            ds_train = data.normalize_dataset(ds_train)
            ds_val = data.normalize_dataset(ds_val)
            if ds_test is not None:
                ds_test = data.normalize_dataset(ds_test)

        # Setup remaining pipeline by output type.
        # rel_output_scale = (
        # self.training_job.model.output_scale / self.training_job.input_scale
        # )
        # TODO: Update this to the commented calculation above when model config
        # includes metadata about absolute input scale.
        rel_output_scale = self.training_job.model.output_scale
        output_type = self.training_job.model.output_type
        if output_type == model.ModelOutputType.CONFIDENCE_MAP:
            ds_train = data.make_confmap_dataset(
                ds_train,
                output_scale=rel_output_scale,
                sigma=self.training_job.trainer.sigma,
            )
            ds_val = data.make_confmap_dataset(
                ds_val,
                output_scale=rel_output_scale,
                sigma=self.training_job.trainer.sigma,
            )
            if ds_test is not None:
                ds_test = data.make_confmap_dataset(
                    ds_test,
                    output_scale=rel_output_scale,
                    sigma=self.training_job.trainer.sigma,
                )
            n_output_channels = train.skeleton.n_nodes

        elif output_type == model.ModelOutputType.TOPDOWN_CONFIDENCE_MAP:
            if not self.training_job.trainer.instance_crop:
                raise ValueError(
                    "Cannot train a topddown model without instance cropping enabled."
                )

            # TODO: Parametrize multiple heads in the training configuration.
            cm_params = dict(
                sigma=self.training_job.trainer.sigma,
                output_scale=rel_output_scale,
                with_instance_cms=False,
                with_all_peaks=False,
                with_ctr_peaks=True,
            )
            ds_train = data.make_instance_confmap_dataset(ds_train, **cm_params)
            ds_val = data.make_instance_confmap_dataset(ds_val, **cm_params)
            if ds_test is not None:
                ds_test = data.make_instance_confmap_dataset(ds_test, **cm_params)
            n_output_channels = train.skeleton.n_nodes

        elif output_type == model.ModelOutputType.PART_AFFINITY_FIELD:
            ds_train = data.make_paf_dataset(
                ds_train,
                train.skeleton.edges,
                output_scale=rel_output_scale,
                distance_threshold=self.training_job.trainer.sigma,
            )
            ds_val = data.make_paf_dataset(
                ds_val,
                train.skeleton.edges,
                output_scale=rel_output_scale,
                distance_threshold=self.training_job.trainer.sigma,
            )
            if ds_test is not None:
                ds_test = data.make_paf_dataset(
                    ds_test,
                    train.skeleton.edges,
                    output_scale=rel_output_scale,
                    distance_threshold=self.training_job.trainer.sigma,
                )
            n_output_channels = train.skeleton.n_edges * 2

        elif output_type == model.ModelOutputType.CENTROIDS:
            cm_params = dict(
                sigma=self.training_job.trainer.sigma,
                output_scale=rel_output_scale,
                use_ctr_node=self.training_job.trainer.instance_crop_use_ctr_node,
                ctr_node_ind=self.training_job.trainer.instance_crop_ctr_node_ind,
            )

            ds_train = data.make_centroid_confmap_dataset(ds_train, **cm_params)
            ds_val = data.make_centroid_confmap_dataset(ds_val, **cm_params)
            if ds_test is not None:
                ds_test = data.make_centroid_confmap_dataset(ds_test, **cm_params)

            n_output_channels = 1

        else:
            raise ValueError(
                f"Invalid model output type specified ({self.training_job.model.output_type})."
            )

        if self.training_job.trainer.steps_per_epoch <= 0:
            self.training_job.trainer.steps_per_epoch = int(
                len(train.images) // self.training_job.trainer.batch_size
            )
        if self.training_job.trainer.val_steps_per_epoch <= 0:
            self.training_job.trainer.val_steps_per_epoch = int(
                np.ceil(len(val.images) / self.training_job.trainer.batch_size)
            )

        # Set up shuffling, batching, repeating and prefetching.
        shuffle_buffer_size = self.training_job.trainer.shuffle_buffer_size
        if shuffle_buffer_size is None or shuffle_buffer_size <= 0:
            shuffle_buffer_size = len(train.images)
        ds_train = (
            ds_train.shuffle(shuffle_buffer_size)
            .repeat(-1)
            .batch(self.training_job.trainer.batch_size, drop_remainder=True)
            .prefetch(buffer_size=self.training_job.trainer.steps_per_epoch)
            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        ds_val = (
            ds_val.repeat(-1)
            .batch(self.training_job.trainer.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        if ds_test is not None:
            ds_test = ds_val.batch(self.training_job.trainer.batch_size).prefetch(
                buffer_size=self.training_job.trainer.val_steps_per_epoch
                # buffer_size=tf.data.experimental.AUTOTUNE
            )

        # Get image shape after all the dataset transformations are applied.
        img_shape = list(ds_val.take(1))[0][0][0].shape

        # Update internal attributes.
        self._img_shape = img_shape
        self._n_output_channels = n_output_channels
        self._train = train
        self._val = val
        self._test = test
        self._ds_train = ds_train
        self._ds_val = ds_val
        self._ds_test = ds_test
        self._simple_skeleton = train.skeleton

        if (
            self.training_job.model.skeletons is None
            or len(self.training_job.model.skeletons) == 0
        ):
            # Save skeleton to training job/model config if none were already stored.
            skeleton = Skeleton.from_names_and_edge_inds(
                node_names=self.simple_skeleton.node_names,
                edge_inds=self.simple_skeleton.edge_inds,
            )
            self.training_job.model.skeletons = [skeleton]

        if self.verbosity > 0:
            print("Data:")
            print("  Input scale:", self.training_job.input_scale)
            print("  Relative output scale:", self.training_job.model.output_scale)
            print(
                "  Output scale:",
                self.training_job.input_scale * self.training_job.model.output_scale,
            )
            print("  Training data:", self.data_train.images.shape)
            print("  Validation data:", self.data_val.images.shape)
            if self.data_test is not None:
                print("  Test data:", self.data_test.images.shape)
            else:
                print("  Test data: N/A")
            print("  Image shape:", self.img_shape)
            print("  Output channels:", self.n_output_channels)
            print("  Skeleton:", self.simple_skeleton)
            print()

        return ds_train, ds_val, ds_test

    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:

        callback_list = []

        if self.training_job.trainer.reduce_lr_on_plateau:
            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    min_delta=self.training_job.trainer.reduce_lr_min_delta,
                    factor=self.training_job.trainer.reduce_lr_factor,
                    patience=self.training_job.trainer.reduce_lr_patience,
                    cooldown=self.training_job.trainer.reduce_lr_cooldown,
                    min_lr=self.training_job.trainer.reduce_lr_min_lr,
                    monitor=self.training_job.trainer.monitor_metric_name,
                    mode="auto",
                    verbose=self.verbosity,
                )
            )

        if self.training_job.trainer.early_stopping:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor=self.training_job.trainer.monitor_metric_name,
                    min_delta=self.training_job.trainer.early_stopping_min_delta,
                    patience=self.training_job.trainer.early_stopping_patience,
                    verbose=self.verbosity,
                )
            )

        if self.training_job.run_path is not None:

            if self.training_job.trainer.csv_logging:
                callback_list.append(
                    callbacks.CSVLogger(
                        filename=os.path.join(
                            self.training_job.run_path,
                            self.training_job.trainer.csv_log_filename,
                        )
                    )
                )

            if self.tensorboard:
                if self.tensorboard_dir is None:
                    self.tensorboard_dir = self.training_job.run_path
                callback_list.append(
                    callbacks.TensorBoard(
                        log_dir=self.tensorboard_dir,
                        update_freq=self.tensorboard_freq,
                        )
                    )

            if self.training_job.trainer.save_every_epoch:
                if self.training_job.newest_model_filename is None:
                    self.training_job.newest_model_filename = "newest_model.h5"

                callback_list.append(
                    callbacks.ModelCheckpoint(
                        filepath=os.path.join(
                            self.training_job.run_path,
                            self.training_job.newest_model_filename,
                        ),
                        monitor=self.training_job.trainer.monitor_metric_name,
                        save_best_only=False,
                        save_weights_only=False,
                        save_freq="epoch",
                        verbose=self.verbosity,
                    )
                )

            if self.training_job.trainer.save_best_val:
                if self.training_job.best_model_filename is None:
                    self.training_job.best_model_filename = "best_model.h5"

                callback_list.append(
                    callbacks.ModelCheckpoint(
                        filepath=os.path.join(
                            self.training_job.run_path,
                            self.training_job.best_model_filename,
                        ),
                        monitor=self.training_job.trainer.monitor_metric_name,
                        save_best_only=True,
                        save_weights_only=False,
                        save_freq="epoch",
                        verbose=self.verbosity,
                    )
                )

            if self.training_job.trainer.save_final_model:
                if self.training_job.final_model_filename is None:
                    self.training_job.final_model_filename = "final_model.h5"

                callback_list.append(
                    callbacks.ModelCheckpointOnEvent(
                        filepath=os.path.join(
                            self.training_job.run_path,
                            self.training_job.final_model_filename,
                        ),
                        event="train_end",
                    )
                )

            self.training_job.save(
                os.path.join(self.training_job.run_path, "training_job.json")
            )

        self._training_callbacks = callback_list
        return callback_list

    def setup_optimization(self):

        if self.training_job.trainer.optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.training_job.trainer.learning_rate,
                amsgrad=self.training_job.trainer.amsgrad,
            )

        elif self.training_job.trainer.optimizer.lower() == "rmsprop":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.training_job.trainer.learning_rate
            )

        else:
            raise ValueError(
                "Unrecognized optimizer specified: %s",
                self.training_job.trainer.optimizer,
            )

        loss_fn = tf.keras.losses.MeanSquaredError()
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        return optimizer, loss_fn

    def setup_model(self, img_shape=None, n_output_channels=None):

        if img_shape is None:
            img_shape = self.img_shape

        if n_output_channels is None:
            n_output_channels = self.n_output_channels

        input_layer = tf.keras.layers.Input(img_shape, name="input")

        outputs = self.training_job.model.output(input_layer, n_output_channels)
        if isinstance(outputs, tf.keras.Model):
            outputs = outputs.outputs

        keras_model = tf.keras.Model(
            input_layer, outputs, name=self.training_job.model.backbone_name
        )

        if self.verbosity > 0:
            print(f"Model: {keras_model.name}")
            print(f"  Input: {keras_model.input_shape}")
            print(f"  Output: {keras_model.output_shape}")
            print(f"  Layers: {len(keras_model.layers)}")
            print(f"  Params: {keras_model.count_params():3,}")
            print()

        self._model = keras_model

        return keras_model

    def train(
        self,
        labels_train: Union[Labels, Text] = None,
        labels_val: Union[Labels, Text] = None,
        labels_test: Union[Labels, Text] = None,
        data_train: Union[data.TrainingData, Text] = None,
        data_val: Union[data.TrainingData, Text] = None,
        data_test: Union[data.TrainingData, Text] = None,
    ) -> tf.keras.Model:

        self.setup_data(
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
        )
        self.setup_model()
        self.setup_optimization()

        if (
            self.training_job.save_dir is not None
            and self.training_job.run_name is None
        ):
            # Generate new run name if save_dir specified but not the run name.
            self.training_job.run_name = self.training_job.new_run_name(
                suffix=f"n={len(self.data_train.images)}"
            )

        if self.training_job.run_path is not None:
            if not os.path.exists(self.training_job.run_path):
                os.makedirs(self.training_job.run_path, exist_ok=True)
            if self.verbosity > 0:
                print(f"Run path: {self.training_job.run_path}")
        else:
            if self.verbosity > 0:
                print(f"Run path: Not provided, nothing will be saved to disk.")

        self.setup_callbacks()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        t0 = datetime.now()
        if self.verbosity > 0:
            print(f"Training started: {str(t0)}")

        self._history = self.model.fit(
            self.ds_train,
            epochs=self.training_job.trainer.num_epochs,
            callbacks=self.training_callbacks,
            validation_data=self.ds_val,
            steps_per_epoch=self.training_job.trainer.steps_per_epoch,
            validation_steps=self.training_job.trainer.val_steps_per_epoch,
            verbose=self.verbosity,
        )
        t1 = datetime.now()
        elapsed = t1 - t0
        if self.verbosity > 0:
            print(f"Training finished: {str(t1)}")
            print(f"Total runtime: {str(elapsed)}")

        # TODO: Evaluate final test set performance if available

        return self.model


def main():
    """CLI for training."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_job_path", help="Path to training job profile JSON file."
    )
    parser.add_argument("labels_path", help="Path to labels file to use for training.")
    parser.add_argument(
        "--val_labels",
        "--val",
        help="Path to labels file to use for validation (overrides training job path if set).",
    )
    parser.add_argument(
        "--test_labels",
        "--test",
        help="Path to labels file to use for test (overrides training job path if set).",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enables TensorBoard logging to the run path.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        help="Prefix to append to run name. Can be specified multiple times.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        help="Suffix to append to run name. Can be specified multiple times.",
    )

    args = parser.parse_args()

    job_filename = args.training_job_path
    if not os.path.exists(job_filename):
        profile_dir = resource_filename(
            Requirement.parse("sleap"), "sleap/training_profiles"
        )
        if os.path.exists(os.path.join(profile_dir, job_filename)):
            job_filename = os.path.join(profile_dir, job_filename)
        else:
            raise FileNotFoundError(f"Could not find training profile: {job_filename}")

    labels_train_path = args.labels_path

    print(f"Training labels file: {labels_train_path}")
    print(f"Training profile: {job_filename}")

    training_job = job.TrainingJob.load_json(job_filename)

    # Set data paths in job.
    training_job.labels_filename = labels_train_path
    if args.val_labels is not None:
        training_job.val_set_filename = args.val_labels
    if args.test_labels is not None:
        training_job.test_set_filename = args.test_labels

    if training_job.save_dir is None:
        # Default save dir to models subdir of training labels.
        training_job.save_dir = os.path.join(
            os.path.dirname(labels_train_path), "models"
        )

    prefixes = args.prefix
    if training_job.run_name is not None:
        # Add run name specified in file to prefixes.
        prefixes.append(training_job.run_name)

    # Create new run name.
    training_job.run_name = training_job.new_run_name(
        prefix=prefixes, suffix=args.suffix, check_existing=True
    )

    # Log configuration to console.
    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    print()
    print("Training job:")
    print(json.dumps(job.TrainingJob._to_dicts(training_job), indent=4))
    print()

    print("Initializing training...")
    # Create a trainer and run!
    trainer = Trainer(
        training_job, tensorboard=args.tensorboard, zmq=False, verbosity=2
    )
    trained_model = trainer.train()


if __name__ == "__main__":
    main()

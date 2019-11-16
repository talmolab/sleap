"""SLEAP model training."""

import os
import attr
from typing import Union, Dict, List, Text

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
    tensorboard_dir: Union[str, None] = None
    zmq: bool = False
    control_zmq_port: int = 9000
    progress_report_zmq_port: int = 9001
    verbosity: int = 1

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
            train = Labels.load_file(train)
        if train is not None:
            train = data.TrainingData.from_labels(train)
        if train is None:
            train = data_train
        if train is not None and isinstance(train, str):
            train = data.TrainingData.load_file(train)
        if train is None:
            raise ValueError("Training data was not specified.")

        val = labels_val
        if val is None:
            val = self.training_job.val_set_filename
        if val is not None and isinstance(val, str):
            val = Labels.load_file(val)
        if val is not None:
            val = data.TrainingData.from_labels(val)
        if val is None:
            val = data_val
        if val is not None and isinstance(val, str):
            val = data.TrainingData.load_file(val)
        if val is None and self.val_size is not None:
            train, val = data.split_training_data(
                train, first_split_fraction=self.training_job.trainer.val_size
            )
        if val is None:
            raise ValueError("Validation set or fraction must be specified.")

        test = labels_test
        if test is None:
            test = self.training_job.test_set_filename
        if test is not None and isinstance(test, str):
            test = Labels.load_file(test)
        if test is not None:
            test = data.TrainingData.from_labels(test)
        if test is None:
            test = data_test
        if test is not None and isinstance(test, str):
            test = data.TrainingData.load_file(test)

        img_shape = (
            int(train.images.shape[1] * self.training_job.input_scale),
            int(train.images.shape[2] * self.training_job.input_scale),
            int(train.images.shape[3]),
        )

        if self.training_job.model.output_type == model.ModelOutputType.CONFIDENCE_MAP:
            n_output_channels = train.skeleton.n_nodes

        elif (
            self.training_job.model.output_type ==
            model.ModelOutputType.TOPDOWN_CONFIDENCE_MAP
        ):
            n_output_channels = train.skeleton.n_nodes

        elif (
            self.training_job.model.output_type ==
            model.ModelOutputType.PART_AFFINITY_FIELD
        ):
            n_output_channels = train.skeleton.n_edges * 2

        elif self.training_job.model.output_type == model.ModelOutputType.CENTROIDS:
            n_output_channels = 1

        else:
            raise ValueError(
                f"Invalid model output type specified ({self.training_job.model.output_type})."
            )

        ds_train = None
        ds_val = None
        ds_test = None

        return ds_train, ds_val, ds_test, img_shape, n_output_channels

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
                    verbose=1,
                )
            )

        if self.training_job.trainer.early_stopping:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor=self.training_job.trainer.monitor_metric_name,
                    min_delta=self.training_job.trainer.early_stopping_min_delta,
                    patience=self.training_job.trainer.early_stopping_patience,
                    verbose=1,
                )
            )

        if self.training_job.run_path is not None:

            if self.training_job.trainer.save_every_epoch:
                if self.training_job.newest_model_filename is None:
                    full_path = os.path.join(
                        self.training_job.run_path, "newest_model.h5"
                    )
                    self.training_job.newest_model_filename = os.path.relpath(
                        full_path, self.training_job.save_dir
                    )

                callback_list.append(
                    callbacks.ModelCheckpoint(
                        filepath=self.training_job.newest_model_filename,
                        monitor=self.training_job.trainer.monitor_metric_name,
                        save_best_only=False,
                        save_weights_only=False,
                        save_freq="epoch",
                    )
                )

            if self.save_best_val:
                if self.training_job.best_model_filename is None:
                    full_path = os.path.join(
                        self.training_job.run_path, "best_model.h5"
                    )
                    self.training_job.best_model_filename = os.path.relpath(
                        full_path, self.training_job.save_dir
                    )

                callback_list.append(
                    callbacks.ModelCheckpoint(
                        filepath=self.training_job.best_model_filename,
                        monitor=self.training_job.trainer.monitor_metric_name,
                        save_best_only=True,
                        save_weights_only=False,
                        save_freq="epoch",
                    )
                )

            job.TrainingJob.save_json(
                self.training_job,
                os.path.join(self.training_job.run_path, "training_job.json"),
            )

        return callback_list

    def setup_optimization(self):

        if self.training_job.trainer.optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.training_job.trainer.learning_rate,
                amsgrad=self.training_job.trainer.amsgrad,
            )

        elif self.training_job.trainer.optimizer.lower() == "rmsprop":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.training_job.trainer.learning_rate,
            )

        else:
            raise ValueError(
                "Unrecognized optimizer specified: %s",
                self.training_job.trainer.optimizer,
            )

        loss_fn = tf.keras.losses.MeanSquaredError()

        return optimizer, loss_fn

    def setup_model(self, img_shape, n_output_channels):

        input_layer = tf.keras.layers.Input(img_shape, name="input")

        outputs = self.training_job.model.output(input_layer, n_output_channels)
        if isinstance(outputs, tf.keras.Model):
            outputs = outputs.outputs

        keras_model = tf.keras.Model(
            input_layer, outputs, name=self.training_job.model.backbone_name
        )

        return keras_model

    def train(
        self,
        *args,
        labels_train: Union[Labels, Text] = None,
        labels_val: Union[Labels, Text] = None,
        labels_test: Union[Labels, Text] = None,
        data_train: Union[data.TrainingData, Text] = None,
        data_val: Union[data.TrainingData, Text] = None,
        data_test: Union[data.TrainingData, Text] = None,
        **kwargs,
    ) -> tf.keras.Model:

        ds_train, ds_val, ds_test, img_shape, n_output_channels = self.setup_data(
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
        )
        optimizer, loss_fn = self.setup_optimization()
        training_callbacks = self.setup_callbacks()
        keras_model = self.setup_model(img_shape, n_output_channels)
        keras_model.compile(optimizer=optimizer, loss=loss_fn)

        history = keras_model.fit(
            ds_train,
            epochs=self.training_job.trainer.epochs,
            callbacks=training_callbacks,
            validation_data=ds_val,
            steps_per_epoch=self.training_job.trainer.steps_per_epoch,
            validation_steps=self.training_job.trainer.val_steps_per_epoch,
            verbose=self.verbosity,
        )

        # TODO: Save training history
        # TODO: Evaluate final test set performance if available

        return keras_model

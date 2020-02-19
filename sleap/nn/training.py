"""Training functionality and high level APIs."""

import os
from datetime import datetime

import tensorflow as tf
import numpy as np

import attr
from typing import Optional, Callable, List, Union, Text

import cattr
import json
import copy

import sleap
# from sleap.nn.config import TrainingJobConfig
# from sleap.nn.model import Model
# from sleap.nn.data.pipelines import Pipeline

# Data
from sleap.nn.data.pipelines import LabelsReader
from sleap.nn.config import LabelsConfig

# Optimization
from sleap.nn.config import OptimizationConfig
from sleap.nn.losses import OHKMLoss, PartLoss
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Outputs
from sleap.nn.config import OutputsConfig, ZMQConfig, TensorBoardConfig, CheckpointingConfig
from sleap.nn.callbacks import TrainingControllerZMQ, ProgressReporterZMQ, ModelCheckpointOnEvent
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sleap.nn.callbacks import TensorBoardMatplotlibWriter, MatplotlibSaver


@attr.s(auto_attribs=True)
class DataReaders:
    training_labels_reader: LabelsReader
    validation_labels_reader: LabelsReader
    test_labels_reader: Optional[LabelsReader] = None
    
    @classmethod
    def from_config(cls, labels_config: LabelsConfig) -> "DataReaders":
        validation_labels = labels_config.validation_labels
        if validation_labels is None:
            validation_labels = labels_config.validation_fraction

        # TODO: use labels_config.search_path_hints for loading
        return cls.from_labels(training=labels_config.training_labels, validation=labels_config.validation_labels, test=labels_config.test_labels)
    
    @classmethod
    def from_labels(cls, training: Union[Text, sleap.Labels], validation: Union[Text, sleap.Labels, float], test: Optional[Union[Text, sleap.Labels]] = None) -> "DataReaders":
        if isinstance(training, str):
            training = sleap.Labels.load_file(training)

        if isinstance(validation, str):
            validation = sleap.Labels.load_file(validation)
        elif isinstance(validation, float):
            # TODO: split
            pass

        if isinstance(test, str):
            test = sleap.Labels.load_file(test)

        test_reader = None
        if test is not None:
            test_reader = LabelsReader.from_user_instances(test)

        return cls(
            training_labels_reader=LabelsReader.from_user_instances(training),
            validation_labels_reader=LabelsReader.from_user_instances(validation),
            test_labels_reader=test_reader
        )
    
    @property
    def training_labels(self) -> sleap.Labels:
        return self.training_labels_reader.labels

    @property
    def validation_labels(self) -> sleap.Labels:
        return self.validation_labels_reader.labels
    
    @property
    def test_labels(self) -> sleap.Labels:
        if self.test_labels_reader is None:
            raise None
        return self.test_labels_reader.labels



def setup_optimizer(config: OptimizationConfig) -> tf.keras.optimizers.Optimizer:
    if config.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_learning_rate, amsgrad=True)
    else:
        # TODO: explicit lookup
        optimizer = config.optimizer
    return optimizer


def setup_losses(config: OptimizationConfig) -> Callable[[tf.Tensor], tf.Tensor]:

    losses = [tf.keras.losses.MeanSquaredError()]

    if config.hard_keypoint_mining.online_mining:
        losses.append(OHKMLoss.from_config(config.hard_keypoint_mining))

    def loss_fn(y_gt, y_pr):
        loss = 0
        for loss_fn in losses:
            loss += loss_fn(y_gt, y_pr)
        return loss
    
    return loss_fn


def setup_metrics(config: OptimizationConfig, part_names: Optional[List[Text]] = None) -> List[Union[tf.keras.losses.Loss, tf.keras.metrics.Metric]]:
    metrics = []

    if config.hard_keypoint_mining.online_mining:
        metrics.append(OHKMLoss.from_config(config.hard_keypoint_mining))

    if part_names is not None:
        for channel_ind, part_name in enumerate(part_names):
            metrics.append(PartLoss(channel_ind=channel_ind, name=part_name))

    return metrics


def setup_optimization_callbacks(config: OptimizationConfig) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    if config.learning_rate_schedule.reduce_on_plateau:
        callbacks.append(ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.learning_rate_schedule.reduction_factor,
            patience=config.learning_rate_schedule.plateau_patience,
            min_delta=config.learning_rate_schedule.plateau_min_delta,
            cooldown=config.learning_rate_schedule.plateau_cooldown,
            min_lr=config.learning_rate_schedule.min_learning_rate,
            verbose=1,
        ))
    
    if config.early_stopping.stop_training_on_plateau:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping.plateau_patience,
            min_delta=config.early_stopping.plateau_min_delta,
            verbose=1,
        ))

    return callbacks


def get_timestamp() -> Text:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def setup_new_run_folder(config: OutputsConfig, run_name_base: Optional[Text] = None) -> Text:

    run_path = None
    if config.save_outputs:
        # Auto-generate run name.
        if config.run_name is None:
            config.run_name = get_timestamp()
            if isinstance(run_name_base, str):
                config.run_name = config.run_name + "." + run_name_base

        # Find new run name suffix if needed.
        if config.run_name_suffix is None:
            config.run_name_suffix = ""
            run_path = os.path.join(config.runs_folder, f"{config.run_name_prefix}{config.run_name}")
            i = 0
            while os.path.exists(run_path):
                i += 1
                config.run_name_suffix = f"_{i}"
                run_path = os.path.join(config.runs_folder, f"{config.run_name_prefix}{config.run_name}{config.run_name_suffix}")

        # Build run path.
        run_path = os.path.join(config.runs_folder, f"{config.run_name_prefix}{config.run_name}{config.run_name_suffix}")

    return run_path


def setup_zmq_callbacks(zmq_config: ZMQConfig) -> List[tf.keras.callbacks.Callback]:
    callbacks = []

    if zmq_config.subscribe_to_controller:
        callbacks.append(TrainingControllerZMQ(address=zmq_config.controller_address, poll_timeout=zmq_config.controller_polling_timeout))

    if zmq_config.publish_updates:
        callbacks.append(ProgressReporterZMQ(address=zmq_config.zmq.publish_address))

    return callbacks


def setup_checkpointing(config: CheckpointingConfig, run_path: Text) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    if config.initial_model:
        callbacks.append(ModelCheckpointOnEvent(
            filepath=os.path.join(run_path, "initial_model.h5"),
            event="train_begin",
        ))
    
    if config.best_model:
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(run_path, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            verbose=0,
        ))

    if config.every_epoch:
        callbacks.append(ModelCheckpointOnEvent(
            filepath=os.path.join(run_path, "model.epoch%04d.h5"),
            event="epoch_end",
        ))

    if config.latest_model:
        callbacks.append(ModelCheckpointOnEvent(
            filepath=os.path.join(run_path, "latest_model.h5"),
            event="epoch_end",
        ))
    
    if config.final_model:
        callbacks.append(ModelCheckpointOnEvent(
            filepath=os.path.join(run_path, "final_model.h5"),
            event="train_end",
        ))
        
    return callbacks


def setup_tensorboard(config: TensorBoardConfig, run_path: Text) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    if config.write_logs:
        callbacks.append(
            TensorBoard(
                log_dir=run_path,
                histogram_freq=0,
                write_graph=config.architecture_graph,
                update_freq=config.loss_frequency,
                profile_batch=2 if config.profile_graph else 0,
                embeddings_freq=0,
                embeddings_metadata=None,
            ))

    return callbacks


def setup_output_callbacks(config: OutputsConfig, run_path: Optional[Text] = None) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    if config.save_outputs and run_path is not None:
        callbacks.extend(setup_checkpointing(config.checkpointing, run_path))
        callbacks.extend(setup_tensorboard(config.tensorboard, run_path))

        if config.log_to_csv:
            callbacks.append(CSVLogger(filename=os.path.join(run_path, "training_log.csv")))
    callbacks.extend(setup_zmq_callbacks(config.zmq))
    return callbacks


def setup_visualization(config: OutputsConfig, run_path: Text, viz_fn: Callable[[], matplotlib.figure.Figure], name: Text) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    if config.save_visualizations:
        callbacks.append(MatplotlibSaver(save_folder=os.path.join(run_path, "viz"), plot_fn=viz_fn, prefix=name))
    
    if config.tensorboard.write_logs and config.tensorboard.visualizations:
        callbacks.append(TensorBoardMatplotlibWriter(log_dir=os.path.join(run_path, name), plot_fn=viz_fn, tag=name))

    return callbacks


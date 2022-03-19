"""Training functionality and high level APIs."""

import os
import re
from datetime import datetime
from time import time
import logging
import shutil

import tensorflow as tf
import numpy as np

import attr
from typing import Optional, Callable, List, Union, Text, TypeVar
from abc import ABC, abstractmethod

import cattr
import json
import copy

import sleap
from sleap.util import get_package_file

# Config
from sleap.nn.config import (
    TrainingJobConfig,
    SingleInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
    MultiClassBottomUpConfig,
    MultiClassTopDownConfig,
)

# Model
from sleap.nn.model import Model

# Data
from sleap.nn.config import LabelsConfig
from sleap.nn.data.pipelines import LabelsReader
from sleap.nn.data.pipelines import (
    Pipeline,
    SingleInstanceConfmapsPipeline,
    CentroidConfmapsPipeline,
    TopdownConfmapsPipeline,
    BottomUpPipeline,
    BottomUpMultiClassPipeline,
    TopDownMultiClassPipeline,
    KeyMapper,
)
from sleap.nn.data.training import split_labels_train_val

# Optimization
from sleap.nn.config import OptimizationConfig
from sleap.nn.losses import OHKMLoss, PartLoss
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Outputs
from sleap.nn.config import (
    OutputsConfig,
    ZMQConfig,
    TensorBoardConfig,
    CheckpointingConfig,
)
from sleap.nn.callbacks import (
    TrainingControllerZMQ,
    ProgressReporterZMQ,
    ModelCheckpointOnEvent,
)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

# Inference
from sleap.nn.inference import FindInstancePeaks, SingleInstanceInferenceLayer

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from sleap.nn.callbacks import TensorBoardMatplotlibWriter, MatplotlibSaver
from sleap.nn.viz import plot_img, plot_confmaps, plot_peaks, plot_pafs


logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class DataReaders:
    """Container class for SLEAP labels that serve as training data sources.

    Attributes:
        training_labels_reader: LabelsReader pipeline provider for a training data from
            a sleap.Labels instance.
        validation_labels_reader: LabelsReader pipeline provider for a validation data
            from a sleap.Labels instance.
        test_labels_reader: LabelsReader pipeline provider for a test set data from a
            sleap.Labels instance. This is not necessary for training.
    """

    training_labels_reader: LabelsReader
    validation_labels_reader: LabelsReader
    test_labels_reader: Optional[LabelsReader] = None

    @classmethod
    def from_config(
        cls,
        labels_config: LabelsConfig,
        training: Union[Text, sleap.Labels],
        validation: Union[Text, sleap.Labels, float],
        test: Optional[Union[Text, sleap.Labels]] = None,
        video_search_paths: Optional[List[Text]] = None,
        update_config: bool = False,
        with_track_only: bool = False,
    ) -> "DataReaders":
        """Create data readers from a (possibly incomplete) configuration."""
        # Use config values if not provided in the arguments.
        if training is None:
            training = labels_config.training_labels
        if validation is None:
            if labels_config.validation_labels is not None:
                validation = labels_config.validation_labels
            else:
                validation = labels_config.validation_fraction
        if test is None:
            test = labels_config.test_labels

        if video_search_paths is None:
            video_search_paths = []
        if labels_config.search_path_hints is not None:
            video_search_paths.extend(labels_config.search_path_hints)

        # Update the config fields with arguments (if not a full sleap.Labels instance).
        if update_config:
            if isinstance(training, Text):
                labels_config.training_labels = training
            if isinstance(validation, Text):
                labels_config.validation_labels = validation
            elif isinstance(validation, float):
                labels_config.validation_fraction = validation
            if isinstance(test, Text):
                labels_config.test_labels = test
            labels_config.search_path_hints = video_search_paths

        # Build class.
        return cls.from_labels(
            training=training,
            validation=validation,
            test=test,
            video_search_paths=video_search_paths,
            labels_config=labels_config,
            update_config=update_config,
            with_track_only=with_track_only,
        )

    @classmethod
    def from_labels(
        cls,
        training: Union[Text, sleap.Labels],
        validation: Union[Text, sleap.Labels, float],
        test: Optional[Union[Text, sleap.Labels]] = None,
        video_search_paths: Optional[List[Text]] = None,
        labels_config: Optional[LabelsConfig] = None,
        update_config: bool = False,
        with_track_only: bool = False,
    ) -> "DataReaders":
        """Create data readers from sleap.Labels datasets as data providers."""
        if isinstance(training, str):
            logger.info(f"Loading training labels from: {training}")
            training = sleap.load_file(training, search_paths=video_search_paths)

        if labels_config is not None and labels_config.split_by_inds:
            # First try to split by indices if specified in config.
            if (
                labels_config.validation_inds is not None
                and len(labels_config.validation_inds) > 0
            ):
                logger.info(
                    "Creating validation split from explicit indices "
                    f"(n = {len(labels_config.validation_inds)})."
                )
                validation = training[labels_config.validation_inds]

            if labels_config.test_inds is not None and len(labels_config.test_inds) > 0:
                logger.info(
                    "Creating test split from explicit indices "
                    f"(n = {len(labels_config.test_inds)})."
                )
                test = training[labels_config.test_inds]

            if (
                labels_config.training_inds is not None
                and len(labels_config.training_inds) > 0
            ):
                logger.info(
                    "Creating training split from explicit indices "
                    f"(n = {len(labels_config.training_inds)})."
                )
                training = training[labels_config.training_inds]

        if isinstance(validation, str):
            # If validation is still a path, load it.
            logger.info(f"Loading validation labels from: {validation}")
            validation = sleap.Labels.load_file(
                validation, search_paths=video_search_paths
            )
        elif isinstance(validation, float):
            logger.info(
                "Creating training and validation splits from "
                f"validation fraction: {validation}"
            )
            # If validation is still a float, create the split from training.
            (
                training,
                training_inds,
                validation,
                validation_inds,
            ) = split_labels_train_val(
                training.with_user_labels_only(
                    with_track_only=with_track_only, copy=True
                ),
                validation,
            )
            logger.info(
                f"  Splits: Training = {len(training_inds)} /"
                f" Validation = {len(validation_inds)}."
            )
            if update_config and labels_config is not None:
                labels_config.training_inds = training_inds
                labels_config.validation_inds = validation_inds

        if isinstance(test, str):
            # If test is still a path, load it.
            logger.info(f"Loading test labels from: {test}")
            test = sleap.load_file(test, search_paths=video_search_paths)

        test_reader = None
        if test is not None:
            test_reader = LabelsReader.from_user_instances(
                test, with_track_only=with_track_only
            )

        return cls(
            training_labels_reader=LabelsReader.from_user_instances(
                training, with_track_only=with_track_only
            ),
            validation_labels_reader=LabelsReader.from_user_instances(
                validation, with_track_only=with_track_only
            ),
            test_labels_reader=test_reader,
        )

    @property
    def training_labels(self) -> sleap.Labels:
        """Return the sleap.Labels underlying the training data reader."""
        return self.training_labels_reader.labels

    @property
    def validation_labels(self) -> sleap.Labels:
        """Return the sleap.Labels underlying the validation data reader."""
        return self.validation_labels_reader.labels

    @property
    def test_labels(self) -> sleap.Labels:
        """Return the sleap.Labels underlying the test data reader."""
        if self.test_labels_reader is None:
            raise ValueError("No test labels provided to data reader.")
        return self.test_labels_reader.labels


def setup_optimizer(config: OptimizationConfig) -> tf.keras.optimizers.Optimizer:
    """Set up model optimizer from config."""
    if config.optimizer.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.initial_learning_rate, amsgrad=True
        )
    elif config.optimizer.lower() == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config.initial_learning_rate
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.initial_learning_rate)
    else:
        # TODO: explicit lookup
        optimizer = config.optimizer
    return optimizer


def setup_losses(
    config: OptimizationConfig,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Set up model loss function from config."""
    losses = [tf.keras.losses.MeanSquaredError()]

    if config.hard_keypoint_mining.online_mining:
        losses.append(OHKMLoss.from_config(config.hard_keypoint_mining))
        logger.info(f"  OHKM enabled: {config.hard_keypoint_mining}")

    def loss_fn(y_gt, y_pr):
        loss = 0
        for loss_fn in losses:
            loss += loss_fn(y_gt, y_pr)
        return loss

    return loss_fn


def setup_metrics(
    config: OptimizationConfig, part_names: Optional[List[Text]] = None
) -> List[Union[tf.keras.losses.Loss, tf.keras.metrics.Metric]]:
    """Set up training metrics from config."""
    metrics = []

    if config.hard_keypoint_mining.online_mining:
        metrics.append(OHKMLoss.from_config(config.hard_keypoint_mining))

    if part_names is not None:
        for channel_ind, part_name in enumerate(part_names):
            metrics.append(PartLoss(channel_ind=channel_ind, name=part_name))

    return metrics


def setup_optimization_callbacks(
    config: OptimizationConfig,
) -> List[tf.keras.callbacks.Callback]:
    """Set up optimization callbacks from config."""
    callbacks = []
    if config.learning_rate_schedule.reduce_on_plateau:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=config.learning_rate_schedule.reduction_factor,
                patience=config.learning_rate_schedule.plateau_patience,
                min_delta=config.learning_rate_schedule.plateau_min_delta,
                cooldown=config.learning_rate_schedule.plateau_cooldown,
                min_lr=config.learning_rate_schedule.min_learning_rate,
                verbose=1,
            )
        )
    logger.info(f"  Learning rate schedule: {config.learning_rate_schedule}")

    if config.early_stopping.stop_training_on_plateau:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.early_stopping.plateau_patience,
                min_delta=config.early_stopping.plateau_min_delta,
                verbose=1,
            )
        )
    logger.info(f"  Early stopping: {config.early_stopping}")

    return callbacks


def get_timestamp() -> Text:
    """Return the date and time as a string."""
    return datetime.now().strftime("%y%m%d_%H%M%S")


def setup_new_run_folder(
    config: OutputsConfig, base_run_name: Optional[Text] = None
) -> Text:
    """Create a new run folder from config."""
    run_path = None
    if config.save_outputs:
        # Auto-generate run name.
        if config.run_name is None:
            config.run_name = get_timestamp()
            if isinstance(base_run_name, str):
                config.run_name = config.run_name + "." + base_run_name

        # Find new run name suffix if needed.
        if config.run_name_suffix is None:
            config.run_name_suffix = ""
            run_path = os.path.join(
                config.runs_folder, f"{config.run_name_prefix}{config.run_name}"
            )
            i = 0
            while os.path.exists(run_path):
                i += 1
                config.run_name_suffix = f"_{i}"
                run_path = os.path.join(
                    config.runs_folder,
                    f"{config.run_name_prefix}{config.run_name}{config.run_name_suffix}",
                )

        # Build run path.
        run_path = config.run_path

    return run_path


def setup_zmq_callbacks(zmq_config: ZMQConfig) -> List[tf.keras.callbacks.Callback]:
    """Set up ZeroMQ callbacks from config."""
    callbacks = []

    if zmq_config.subscribe_to_controller:
        callbacks.append(
            TrainingControllerZMQ(
                address=zmq_config.controller_address,
                poll_timeout=zmq_config.controller_polling_timeout,
            )
        )
        logger.info(f"  ZMQ controller subcribed to: {zmq_config.controller_address}")
    if zmq_config.publish_updates:
        callbacks.append(ProgressReporterZMQ(address=zmq_config.publish_address))
        logger.info(f"  ZMQ progress reporter publish on: {zmq_config.publish_address}")

    return callbacks


def setup_checkpointing(
    config: CheckpointingConfig, run_path: Text
) -> List[tf.keras.callbacks.Callback]:
    """Set up model checkpointing callbacks from config."""
    callbacks = []
    if config.initial_model:
        callbacks.append(
            ModelCheckpointOnEvent(
                filepath=os.path.join(run_path, "initial_model.h5"), event="train_begin"
            )
        )

    if config.best_model:
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(run_path, "best_model.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                save_freq="epoch",
                verbose=0,
            )
        )

    if config.every_epoch:
        callbacks.append(
            ModelCheckpointOnEvent(
                filepath=os.path.join(run_path, "model.epoch%04d.h5"), event="epoch_end"
            )
        )

    if config.latest_model:
        callbacks.append(
            ModelCheckpointOnEvent(
                filepath=os.path.join(run_path, "latest_model.h5"), event="epoch_end"
            )
        )

    if config.final_model:
        callbacks.append(
            ModelCheckpointOnEvent(
                filepath=os.path.join(run_path, "final_model.h5"), event="train_end"
            )
        )

    return callbacks


def setup_tensorboard(
    config: TensorBoardConfig, run_path: Text
) -> List[tf.keras.callbacks.Callback]:
    """Set up TensorBoard callbacks from config."""
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
            )
        )

    return callbacks


def setup_output_callbacks(
    config: OutputsConfig, run_path: Optional[Text] = None
) -> List[tf.keras.callbacks.Callback]:
    """Set up training outputs callbacks from config."""
    callbacks = []
    if config.save_outputs and run_path is not None:
        callbacks.extend(setup_checkpointing(config.checkpointing, run_path))
        callbacks.extend(setup_tensorboard(config.tensorboard, run_path))

        if config.log_to_csv:
            callbacks.append(
                CSVLogger(filename=os.path.join(run_path, "training_log.csv"))
            )
    callbacks.extend(setup_zmq_callbacks(config.zmq))
    return callbacks


def setup_visualization(
    config: OutputsConfig,
    run_path: Text,
    viz_fn: Callable[[], matplotlib.figure.Figure],
    name: Text,
) -> List[tf.keras.callbacks.Callback]:
    """Set up visualization callbacks from config."""
    callbacks = []

    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        print(
            "Unable to use Qt backend for matplotlib. "
            "This probably means Qt is running headless."
        )

    if config.save_visualizations and config.save_outputs:
        callbacks.append(
            MatplotlibSaver(
                save_folder=os.path.join(run_path, "viz"), plot_fn=viz_fn, prefix=name
            )
        )

    if (
        config.tensorboard.write_logs
        and config.tensorboard.visualizations
        and config.save_outputs
    ):
        callbacks.append(
            TensorBoardMatplotlibWriter(
                log_dir=os.path.join(run_path, name), plot_fn=viz_fn, tag=name
            )
        )

    return callbacks


def sanitize_scope_name(name: Text) -> Text:
    """Sanitizes string which will be used as TensorFlow scope name."""
    # Add "." to beginning if first character isn't acceptable
    name = re.sub("^([^A-Za-z0-9.])", ".\\1", name)
    # Replace invalid characters with "_"
    name = re.sub("([^A-Za-z0-9._])", "_", name)
    return name


PipelineBuilder = TypeVar(
    "PipelineBuilder",
    CentroidConfmapsPipeline,
    TopdownConfmapsPipeline,
    BottomUpPipeline,
    TopDownMultiClassPipeline,
    BottomUpMultiClassPipeline,
    SingleInstanceConfmapsPipeline,
)


@attr.s(auto_attribs=True)
class Trainer(ABC):
    """Base trainer class that provides general model training functionality.

    This class is intended to be instantiated using the `from_config()` class method,
    which will return the appropriate subclass based on the input configuration.

    This class should not be used directly. It is intended to be subclassed by a model
    output type-specific trainer that provides more specific functionality.

    Attributes:
        data_readers: A `DataReaders` instance that contains training data providers.
        model: A `Model` instance describing the SLEAP model to train.
        config: A `TrainingJobConfig` that describes the training parameters.
        initial_config: This attribute will contain a copy of the input configuration
            before any attributes are updated in `config`.
        pipeline_builder: A model output type-specific data pipeline builder to create
            pipelines that generate data used for training. This must be specified in
            subclasses.
        training_pipeline: The data pipeline that generates examples from the training
            set for optimization.
        validation_pipeline: The data pipeline that generates examples from the
            validation set for optimization.
        training_viz_pipeline: The data pipeline that generates examples from the
            training set for visualization.
        validation_viz_pipeline: The data pipeline that generates examples from the
            validation set for visualization.
        optimization_callbacks: Keras callbacks related to optimization.
        output_callbacks: Keras callbacks related to outputs.
        visualization_callbacks: Keras callbacks related to visualization.
        run_path: The path to the run folder that will contain training results, if any.
    """

    data_readers: DataReaders
    model: Model
    config: TrainingJobConfig
    initial_config: Optional[TrainingJobConfig] = None

    pipeline_builder: PipelineBuilder = attr.ib(init=False)
    training_pipeline: Pipeline = attr.ib(init=False)
    validation_pipeline: Pipeline = attr.ib(init=False)
    training_viz_pipeline: Pipeline = attr.ib(init=False)
    validation_viz_pipeline: Pipeline = attr.ib(init=False)

    optimization_callbacks: List[tf.keras.callbacks.Callback] = attr.ib(
        factory=list, init=False
    )
    output_callbacks: List[tf.keras.callbacks.Callback] = attr.ib(
        factory=list, init=False
    )
    visualization_callbacks: List[tf.keras.callbacks.Callback] = attr.ib(
        factory=list, init=False
    )

    run_path: Optional[Text] = attr.ib(default=None, init=False)

    @classmethod
    def from_config(
        cls,
        config: TrainingJobConfig,
        training_labels: Optional[Union[Text, sleap.Labels]] = None,
        validation_labels: Optional[Union[Text, sleap.Labels, float]] = None,
        test_labels: Optional[Union[Text, sleap.Labels]] = None,
        video_search_paths: Optional[List[Text]] = None,
    ) -> "Trainer":
        """Initialize the trainer from a training job configuration.

        Args:
            config: A `TrainingJobConfig` instance.
            training_labels: Training labels to use instead of the ones in the config,
                if any. If a path is specified, it will overwrite the one in the config.
            validation_labels: Validation labels to use instead of the ones in the
                 config, if any. If a path is specified, it will overwrite the one in
                 the config.
            test_labels: Teset labels to use instead of the ones in the config, if any.
                If a path is specified, it will overwrite the one in the config.
        """
        # Copy input config before we make any changes.
        initial_config = copy.deepcopy(config)

        # Store SLEAP version on the training process.
        config.sleap_version = sleap.__version__

        # Determine output type to create type-specific model trainer.
        head_config = config.model.heads.which_oneof()
        is_id_model = False
        if isinstance(head_config, SingleInstanceConfmapsHeadConfig):
            trainer_cls = SingleInstanceModelTrainer
        elif isinstance(head_config, CentroidsHeadConfig):
            trainer_cls = CentroidConfmapsModelTrainer
        elif isinstance(head_config, CenteredInstanceConfmapsHeadConfig):
            trainer_cls = TopdownConfmapsModelTrainer
        elif isinstance(head_config, MultiInstanceConfig):
            trainer_cls = BottomUpModelTrainer
        elif isinstance(head_config, MultiClassBottomUpConfig):
            trainer_cls = BottomUpMultiClassModelTrainer
            is_id_model = True
        elif isinstance(head_config, MultiClassTopDownConfig):
            trainer_cls = TopDownMultiClassModelTrainer
            is_id_model = True
        else:
            raise ValueError(
                "Model head not specified or configured. Check the config.model.heads"
                " setting."
            )

        # Create data readers and store loaded skeleton.
        data_readers = DataReaders.from_config(
            config.data.labels,
            training=training_labels,
            validation=validation_labels,
            test=test_labels,
            video_search_paths=video_search_paths,
            update_config=True,
            with_track_only=is_id_model,
        )
        config.data.labels.skeletons = data_readers.training_labels.skeletons

        # Create model.
        model = Model.from_config(
            config.model,
            skeleton=config.data.labels.skeletons[0],
            tracks=data_readers.training_labels_reader.tracks,
            update_config=True,
        )

        return trainer_cls(
            config=config,
            initial_config=initial_config,
            data_readers=data_readers,
            model=model,
        )

    @abstractmethod
    def _update_config(self):
        """Implement in subclasses."""
        pass

    @abstractmethod
    def _setup_pipeline_builder(self):
        """Implement in subclasses."""
        pass

    @property
    @abstractmethod
    def input_keys(self):
        """Implement in subclasses."""
        pass

    @property
    @abstractmethod
    def output_keys(self):
        """Implement in subclasses."""
        pass

    @abstractmethod
    def _setup_visualization(self):
        """Implement in subclasses."""
        pass

    def _setup_model(self):
        """Set up the keras model."""
        # Infer the input shape by evaluating the data pipeline.
        logger.info("Building test pipeline...")
        t0 = time()
        base_pipeline = self.pipeline_builder.make_base_pipeline(
            self.data_readers.training_labels_reader
        )
        base_example = next(iter(base_pipeline.make_dataset()))
        input_shape = base_example[self.input_keys[0]].shape
        # TODO: extend input shape determination for multi-input
        logger.info(f"Loaded test example. [{time() - t0:.3f}s]")
        logger.info(f"  Input shape: {input_shape}")

        # Create the tf.keras.Model instance.
        self.model.make_model(input_shape)
        logger.info("Created Keras model.")
        logger.info(f"  Backbone: {self.model.backbone}")
        logger.info(f"  Max stride: {self.model.maximum_stride}")
        logger.info(f"  Parameters: {self.model.keras_model.count_params():3,d}")
        logger.info("  Heads: ")
        for i, head in enumerate(self.model.heads):
            logger.info(f"    [{i}] = {head}")
        logger.info("  Outputs: ")
        for i, output in enumerate(self.model.keras_model.outputs):
            logger.info(f"    [{i}] = {output}")

    @property
    def keras_model(self) -> tf.keras.Model:
        """Alias for `self.model.keras_model`."""
        return self.model.keras_model

    def _setup_pipelines(self):
        """Set up training data pipelines for consumption by the keras model."""
        # Create the training and validation pipelines with appropriate tensor names.
        key_mapper = KeyMapper(
            [
                {
                    input_key: input_name
                    for input_key, input_name in zip(
                        self.input_keys, self.keras_model.input_names
                    )
                },
                {
                    output_key: output_name
                    for output_key, output_name in zip(
                        self.output_keys, self.keras_model.output_names
                    )
                },
            ]
        )
        self.training_pipeline = (
            self.pipeline_builder.make_training_pipeline(
                self.data_readers.training_labels_reader
            )
            + key_mapper
        )
        logger.info(
            f"Training set: n = {len(self.data_readers.training_labels_reader)}"
        )
        self.validation_pipeline = (
            self.pipeline_builder.make_training_pipeline(
                self.data_readers.validation_labels_reader
            )
            + key_mapper
        )
        logger.info(
            f"Validation set: n = {len(self.data_readers.validation_labels_reader)}"
        )

    def _setup_optimization(self):
        """Set up optimizer, loss functions and compile the model."""
        optimizer = setup_optimizer(self.config.optimization)
        loss_fn = setup_losses(self.config.optimization)

        # TODO: Implement general part loss reporting.
        part_names = None
        if (
            isinstance(self.pipeline_builder, TopdownConfmapsPipeline)
            and self.pipeline_builder.offsets_head is None
        ):
            part_names = [
                sanitize_scope_name(name) for name in self.model.heads[0].part_names
            ]
        metrics = setup_metrics(self.config.optimization, part_names=part_names)

        self.optimization_callbacks = setup_optimization_callbacks(
            self.config.optimization
        )

        self.keras_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
            loss_weights={
                output_name: head.loss_weight
                for output_name, head in zip(
                    self.keras_model.output_names, self.model.heads
                )
            },
        )

    def _setup_outputs(self):
        """Set up output-related functionality."""
        if self.config.outputs.save_outputs:
            # Build path to run folder. Timestamp will be added automatically.
            # Example: 210204_041707.centroid.n=300
            model_type = self.config.model.heads.which_oneof_attrib_name()
            n = len(self.data_readers.training_labels_reader) + len(
                self.data_readers.validation_labels_reader
            )
            self.run_path = setup_new_run_folder(
                self.config.outputs, base_run_name=f"{model_type}.n={n}"
            )

        # Setup output callbacks.
        self.output_callbacks = setup_output_callbacks(
            self.config.outputs, run_path=self.run_path
        )

        if self.run_path is not None and self.config.outputs.save_outputs:
            # Create run directory.
            os.makedirs(self.run_path, exist_ok=True)
            logger.info(f"Created run path: {self.run_path}")

            # Save configs.
            if self.initial_config is not None:
                self.initial_config.save_json(
                    os.path.join(self.run_path, "initial_config.json")
                )

            self.config.save_json(os.path.join(self.run_path, "training_config.json"))

            # Save input (ground truth) labels.
            sleap.Labels.save_file(
                self.data_readers.training_labels_reader.labels,
                os.path.join(self.run_path, "labels_gt.train.slp"),
            )
            sleap.Labels.save_file(
                self.data_readers.validation_labels_reader.labels,
                os.path.join(self.run_path, "labels_gt.val.slp"),
            )
            if self.data_readers.test_labels_reader is not None:
                sleap.Labels.save_file(
                    self.data_readers.test_labels_reader.labels,
                    os.path.join(self.run_path, "labels_gt.test.slp"),
                )

    @property
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Return all callbacks currently configured."""
        callbacks = (
            self.optimization_callbacks
            + self.visualization_callbacks
            + self.output_callbacks
        )

        # Some callbacks should be called after all previous ones since they depend on
        # the state of some shared objects (e.g., tf.keras.Model).
        final_callbacks = []
        for callback in callbacks[::-1]:
            if isinstance(callback, tf.keras.callbacks.EarlyStopping):
                final_callbacks.append(callback)
                callbacks.remove(callback)

        return callbacks + final_callbacks

    def setup(self):
        """Set up data pipeline and model for training."""
        logger.info(f"Setting up for training...")
        t0 = time()
        self._update_config()
        logger.info(f"Setting up pipeline builders...")
        self._setup_pipeline_builder()
        logger.info(f"Setting up model...")
        self._setup_model()
        logger.info(f"Setting up data pipelines...")
        self._setup_pipelines()
        logger.info(f"Setting up optimization...")
        self._setup_optimization()
        logger.info(f"Setting up outputs...")
        self._setup_outputs()
        logger.info(f"Setting up visualization...")
        self._setup_visualization()
        logger.info(f"Finished trainer set up. [{time() - t0:.1f}s]")

    def train(self):
        """Execute the optimization loop to train the model."""
        if self.keras_model is None:
            self.setup()

        logger.info(f"Creating tf.data.Datasets for training data generation...")
        t0 = time()
        training_ds = self.training_pipeline.make_dataset()
        validation_ds = self.validation_pipeline.make_dataset()
        logger.info(f"Finished creating training datasets. [{time() - t0:.1f}s]")

        logger.info(f"Starting training loop...")
        t0 = time()
        self.keras_model.fit(
            training_ds,
            epochs=self.config.optimization.epochs,
            validation_data=validation_ds,
            steps_per_epoch=self.config.optimization.batches_per_epoch,
            validation_steps=self.config.optimization.val_batches_per_epoch,
            callbacks=self.callbacks,
            verbose=2,
        )
        logger.info(f"Finished training loop. [{(time() - t0) / 60:.1f} min]")

        # Run post-training actions.
        if self.config.outputs.save_outputs:
            if (
                self.config.outputs.save_visualizations
                and self.config.outputs.delete_viz_images
            ):
                self.cleanup()

            self.evaluate()

            if self.config.outputs.zip_outputs:
                self.package()

    def evaluate(self):
        """Compute evaluation metrics on data splits and save them."""
        logger.info("Saving evaluation metrics to model folder...")
        sleap.nn.evals.evaluate_model(
            cfg=self.config,
            labels_reader=self.data_readers.training_labels_reader,
            model=self.model,
            save=True,
            split_name="train",
        )
        sleap.nn.evals.evaluate_model(
            cfg=self.config,
            labels_reader=self.data_readers.validation_labels_reader,
            model=self.model,
            save=True,
            split_name="val",
        )
        if self.data_readers.test_labels_reader is not None:
            sleap.nn.evals.evaluate_model(
                cfg=self.config,
                labels_reader=self.data_readers.test_labels_reader,
                model=self.model,
                save=True,
                split_name="test",
            )

    def cleanup(self):
        """Delete visualization images subdirectory."""
        viz_path = os.path.join(self.run_path, "viz")
        if os.path.exists(viz_path):
            try:
                logger.info(f"Deleting visualization directory: {viz_path}")
                shutil.rmtree(viz_path)
            except PermissionError:
                logger.info(
                    "Failed to delete visualization directory. If you are training "
                    "through the GUI, this is likely because the visualizer was "
                    "checking for new images. Delete the directory manually if needed."
                )

    def package(self):
        """Package model folder into a zip file for portability."""
        if self.config.outputs.delete_viz_images:
            self.cleanup()
        logger.info(f"Packaging results to: {self.run_path}.zip")
        shutil.make_archive(
            base_name=self.run_path, root_dir=self.run_path, format="zip"
        )


@attr.s(auto_attribs=True)
class SingleInstanceModelTrainer(Trainer):
    """Trainer for models that output single-instance confidence maps."""

    pipeline_builder: SingleInstanceConfmapsPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.single_instance.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = self.model.maximum_stride

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(
                self.data_readers.training_labels_reader.labels.user_instances
            )
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(
                self.data_readers.validation_labels_reader.labels.user_instances
            )
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        """Initialize pipeline builder."""
        self.pipeline_builder = SingleInstanceConfmapsPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            single_instance_confmap_head=self.model.heads[0],
            offsets_head=self.model.heads[1] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        outputs = ["confidence_maps"]
        if self.has_offsets:
            outputs.append("offsets")
        return outputs

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader, self.keras_model
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader, self.keras_model
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        inference_layer = SingleInstanceInferenceLayer(
            keras_model=self.keras_model,
            input_scale=self.config.data.preprocessing.input_scaling,
            pad_to_stride=self.config.data.preprocessing.pad_to_stride,
            peak_threshold=0.2,
            return_confmaps=True,
        )

        def visualize_example(example):
            img = example["image"].numpy()
            preds = inference_layer(tf.expand_dims(img, axis=0))
            cms = preds["confmaps"].numpy()[0]
            pts_gt = example["instances"].numpy()[0]
            pts_pr = preds["instance_peaks"].numpy()[0][0]

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plot_peaks(pts_gt, pts_pr, paired=True)
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(training_viz_ds_iter)),
                name=f"train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(validation_viz_ds_iter)),
                name=f"validation",
            )
        )


@attr.s(auto_attribs=True)
class CentroidConfmapsModelTrainer(Trainer):
    """Trainer for models that output centroid confidence maps."""

    pipeline_builder: CentroidConfmapsPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.centroid.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = self.model.maximum_stride

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(self.data_readers.training_labels)
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(self.data_readers.validation_labels)
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        """Initialize pipeline builder."""
        self.pipeline_builder = CentroidConfmapsPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            centroid_confmap_head=self.model.heads[0],
            offsets_head=self.model.heads[1] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        outputs = ["centroid_confidence_maps"]
        if self.has_offsets:
            outputs.append("offsets")
        return outputs

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader, self.keras_model
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader, self.keras_model
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        def visualize_example(example):
            img = example["image"].numpy()
            cms = example["predicted_centroid_confidence_maps"].numpy()
            pts_gt = example["centroids"].numpy()
            pts_pr = example["predicted_centroids"].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plot_peaks(pts_gt, pts_pr, paired=False)
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(training_viz_ds_iter)),
                name=f"train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(validation_viz_ds_iter)),
                name=f"validation",
            )
        )


@attr.s(auto_attribs=True)
class TopdownConfmapsModelTrainer(Trainer):
    """Trainer for models that output instance centered confidence maps."""

    pipeline_builder: TopdownConfmapsPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.centered_instance.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = 1

        if self.config.data.instance_cropping.crop_size is None:
            self.config.data.instance_cropping.crop_size = sleap.nn.data.instance_cropping.find_instance_crop_size(
                self.data_readers.training_labels,
                padding=self.config.data.instance_cropping.crop_size_detection_padding,
                maximum_stride=self.model.maximum_stride,
                input_scaling=self.config.data.preprocessing.input_scaling,
            )

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(
                self.data_readers.training_labels_reader.labels.user_instances
            )
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(
                self.data_readers.validation_labels_reader.labels.user_instances
            )
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        # Initialize pipeline builder.
        self.pipeline_builder = TopdownConfmapsPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            instance_confmap_head=self.model.heads[0],
            offsets_head=self.model.heads[1] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["instance_image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        outputs = ["instance_confidence_maps"]
        if self.has_offsets:
            outputs.append("offsets")
        return outputs

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        # Create an instance peak finding layer.
        find_peaks = FindInstancePeaks(
            keras_model=self.keras_model,
            input_scale=self.config.data.preprocessing.input_scaling,
            peak_threshold=0.2,
            refinement="local",
            return_confmaps=True,
        )

        def visualize_example(example):
            # Find peaks by evaluating model.
            preds = find_peaks(tf.expand_dims(example["instance_image"], axis=0))
            img = example["instance_image"].numpy()
            cms = preds["instance_confmaps"][0][0].numpy()
            pts_gt = example["center_instance"].numpy()
            pts_pr = preds["instance_peaks"][0][0].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plot_peaks(pts_gt, pts_pr, paired=True)
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(training_viz_ds_iter)),
                name="train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(validation_viz_ds_iter)),
                name="validation",
            )
        )


@attr.s(auto_attribs=True)
class BottomUpModelTrainer(Trainer):
    """Trainer for models that output multi-instance confidence maps and PAFs."""

    pipeline_builder: BottomUpPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.multi_instance.confmaps.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = self.model.maximum_stride

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(self.data_readers.training_labels)
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(self.data_readers.validation_labels)
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        """Initialize pipeline builder."""
        self.pipeline_builder = BottomUpPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            confmaps_head=self.model.heads[0],
            pafs_head=self.model.heads[1],
            offsets_head=self.model.heads[2] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        output_keys = ["confidence_maps", "part_affinity_fields"]
        if self.has_offsets:
            output_keys.append("offsets")
        return output_keys

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader, self.keras_model
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader, self.keras_model
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        def visualize_confmaps_example(example):
            img = example["image"].numpy()
            cms = example["predicted_confidence_maps"].numpy()
            pts_gt = example["instances"].numpy()
            pts_pr = example["predicted_peaks"].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plot_peaks(pts_gt, pts_pr, paired=False)
            return fig

        def visualize_pafs_example(example):
            img = example["image"].numpy()
            pafs = example["predicted_part_affinity_fields"].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)

            pafs = pafs.reshape((pafs.shape[0], pafs.shape[1], -1, 2))
            pafs_mag = np.sqrt(pafs[..., 0] ** 2 + pafs[..., 1] ** 2)
            plot_confmaps(pafs_mag, output_scale=pafs_mag.shape[0] / img.shape[0])
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_confmaps_example(next(training_viz_ds_iter)),
                name=f"train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_confmaps_example(next(validation_viz_ds_iter)),
                name=f"validation",
            )
        )

        # Memory leak:
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_pafs_example(next(training_viz_ds_iter)),
                name=f"train_pafs_magnitude",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_pafs_example(next(validation_viz_ds_iter)),
                name=f"validation_pafs_magnitude",
            )
        )


@attr.s(auto_attribs=True)
class BottomUpMultiClassModelTrainer(Trainer):
    """Trainer for models that output multi-instance confidence maps and class maps."""

    pipeline_builder: BottomUpMultiClassPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.multi_class_bottomup.confmaps.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = self.model.maximum_stride

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(self.data_readers.training_labels)
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(self.data_readers.validation_labels)
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        """Initialize pipeline builder."""
        self.pipeline_builder = BottomUpMultiClassPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            confmaps_head=self.model.heads[0],
            class_maps_head=self.model.heads[1],
            offsets_head=self.model.heads[2] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        output_keys = ["confidence_maps", "class_maps"]
        if self.has_offsets:
            output_keys.append("offsets")
        return output_keys

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader, self.keras_model
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader, self.keras_model
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        def visualize_confmaps_example(example):
            img = example["image"].numpy()
            cms = example["predicted_confidence_maps"].numpy()
            pts_gt = example["instances"].numpy()
            pts_pr = example["predicted_peaks"].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plot_peaks(pts_gt, pts_pr, paired=False)
            return fig

        def visualize_class_maps_example(example):
            img = example["image"].numpy()
            class_maps = example["predicted_class_maps"].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(class_maps, output_scale=class_maps.shape[0] / img.shape[0])
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_confmaps_example(next(training_viz_ds_iter)),
                name=f"train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_confmaps_example(next(validation_viz_ds_iter)),
                name=f"validation",
            )
        )

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_class_maps_example(next(training_viz_ds_iter)),
                name=f"train_class_maps",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_class_maps_example(
                    next(validation_viz_ds_iter)
                ),
                name=f"validation_class_maps",
            )
        )


@attr.s(auto_attribs=True)
class TopDownMultiClassModelTrainer(Trainer):
    """Trainer for models that output multi-instance confidence maps and class maps."""

    pipeline_builder: TopDownMultiClassPipeline = attr.ib(init=False)

    @property
    def has_offsets(self) -> bool:
        """Whether model is configured to output refinement offsets."""
        return self.config.model.heads.multi_class_topdown.confmaps.offset_refinement

    def _update_config(self):
        """Update the configuration with inferred values."""
        if self.config.data.preprocessing.pad_to_stride is None:
            self.config.data.preprocessing.pad_to_stride = self.model.maximum_stride

        if self.config.data.instance_cropping.crop_size is None:
            self.config.data.instance_cropping.crop_size = sleap.nn.data.instance_cropping.find_instance_crop_size(
                self.data_readers.training_labels,
                padding=self.config.data.instance_cropping.crop_size_detection_padding,
                maximum_stride=self.model.maximum_stride,
                input_scaling=self.config.data.preprocessing.input_scaling,
            )

        if self.config.optimization.batches_per_epoch is None:
            n_training_examples = len(self.data_readers.training_labels.user_instances)
            n_training_batches = (
                n_training_examples // self.config.optimization.batch_size
            )
            self.config.optimization.batches_per_epoch = max(
                self.config.optimization.min_batches_per_epoch, n_training_batches
            )

        if self.config.optimization.val_batches_per_epoch is None:
            n_validation_examples = len(
                self.data_readers.validation_labels.user_instances
            )
            n_validation_batches = (
                n_validation_examples // self.config.optimization.batch_size
            )
            self.config.optimization.val_batches_per_epoch = max(
                self.config.optimization.min_val_batches_per_epoch, n_validation_batches
            )

    def _setup_pipeline_builder(self):
        """Initialize pipeline builder."""
        self.pipeline_builder = TopDownMultiClassPipeline(
            data_config=self.config.data,
            optimization_config=self.config.optimization,
            instance_confmap_head=self.model.heads[0],
            class_vectors_head=self.model.heads[1],
            offsets_head=self.model.heads[2] if self.has_offsets else None,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return example keys to be mapped to model inputs."""
        return ["instance_image"]

    @property
    def output_keys(self) -> List[Text]:
        """Return example keys to be mapped to model outputs."""
        output_keys = ["instance_confidence_maps", "class_vectors"]
        if self.has_offsets:
            output_keys.append("offsets")
        return output_keys

    def _setup_optimization(self):
        """Set up optimizer, loss functions and compile the model."""
        optimizer = setup_optimizer(self.config.optimization)
        # loss_fn = setup_losses(self.config.optimization)

        # part_names = None
        # metrics = setup_metrics(self.config.optimization, part_names=None)
        metrics = {"ClassVectorsHead": "accuracy"}

        self.optimization_callbacks = setup_optimization_callbacks(
            self.config.optimization
        )

        self.keras_model.compile(
            optimizer=optimizer,
            loss={
                output_name: head.loss_function
                for output_name, head in zip(
                    self.keras_model.output_names, self.model.heads
                )
            },
            metrics=metrics,
            loss_weights={
                output_name: head.loss_weight
                for output_name, head in zip(
                    self.keras_model.output_names, self.model.heads
                )
            },
        )

    def _setup_visualization(self):
        """Set up visualization pipelines and callbacks."""
        # Create visualization/inference pipelines.
        self.training_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.training_labels_reader
        )
        self.validation_viz_pipeline = self.pipeline_builder.make_viz_pipeline(
            self.data_readers.validation_labels_reader
        )

        # Create static iterators.
        training_viz_ds_iter = iter(self.training_viz_pipeline.make_dataset())
        validation_viz_ds_iter = iter(self.validation_viz_pipeline.make_dataset())

        # Create an instance peak finding layer.
        find_peaks = FindInstancePeaks(
            keras_model=self.keras_model,
            input_scale=self.config.data.preprocessing.input_scaling,
            peak_threshold=0.2,
            refinement="local",
            return_confmaps=True,
        )

        def visualize_example(example):
            # Find peaks by evaluating model.
            preds = find_peaks(tf.expand_dims(example["instance_image"], axis=0))
            img = example["instance_image"].numpy()
            cms = preds["instance_confmaps"][0][0].numpy()
            pts_gt = example["center_instance"].numpy()
            pts_pr = preds["instance_peaks"][0][0].numpy()

            scale = 1.0
            if img.shape[0] < 512:
                scale = 2.0
            if img.shape[0] < 256:
                scale = 4.0
            fig = plot_img(img, dpi=72 * scale, scale=scale)
            plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plot_peaks(pts_gt, pts_pr, paired=True)
            return fig

        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(training_viz_ds_iter)),
                name="train",
            )
        )
        self.visualization_callbacks.extend(
            setup_visualization(
                self.config.outputs,
                run_path=self.run_path,
                viz_fn=lambda: visualize_example(next(validation_viz_ds_iter)),
                name="validation",
            )
        )


def main():
    """Create CLI for training and run."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_job_path", help="Path to training job profile JSON file."
    )
    parser.add_argument(
        "labels_path",
        nargs="?",
        default="",
        help=(
            "Path to labels file to use for training. If specified, overrides the path "
            "specified in the training job config."
        ),
    )
    parser.add_argument(
        "--video-paths",
        type=str,
        default="",
        help=(
            "List of paths for finding videos in case paths inside labels file are "
            "not accessible."
        ),
    )
    parser.add_argument(
        "--val_labels",
        "--val",
        help=(
            "Path to labels file to use for validation. If specified, overrides the "
            "path specified in the training job config."
        ),
    )
    parser.add_argument(
        "--test_labels",
        "--test",
        help=(
            "Path to labels file to use for test. If specified, overrides the path "
            "specified in the training job config."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help=(
            "Enable TensorBoard logging to the run path if not already specified in "
            "the training job config."
        ),
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help=(
            "Enable saving of prediction visualizations to the run folder if not "
            "already specified in the training job config."
        ),
    )
    parser.add_argument(
        "--zmq",
        action="store_true",
        help=(
            "Enable ZMQ logging (for GUI) if not already specified in the training "
            "job config."
        ),
    )
    parser.add_argument(
        "--run_name",
        default="",
        help="Run name to use when saving file, overrides other run name settings.",
    )
    parser.add_argument("--prefix", default="", help="Prefix to prepend to run name.")
    parser.add_argument("--suffix", default="", help="Suffix to append to run name.")

    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Run training only on CPU. If not specified, will use available GPU.",
    )
    device_group.add_argument(
        "--first-gpu",
        action="store_true",
        help="Run training on the first GPU, if available.",
    )
    device_group.add_argument(
        "--last-gpu",
        action="store_true",
        help="Run training on the last GPU, if available.",
    )
    device_group.add_argument(
        "--gpu", type=int, default=0, help="Run training on the i-th GPU on the system."
    )

    args, _ = parser.parse_known_args()

    # Find job configuration file.
    job_filename = args.training_job_path
    if not os.path.exists(job_filename):
        profile_dir = get_package_file("sleap/training_profiles")

        if os.path.exists(os.path.join(profile_dir, job_filename)):
            job_filename = os.path.join(profile_dir, job_filename)
        else:
            raise FileNotFoundError(f"Could not find training profile: {job_filename}")

    # Load job configuration.
    job_config = TrainingJobConfig.load_json(job_filename)

    # Override config settings for CLI-based training.
    job_config.outputs.save_outputs = True
    job_config.outputs.tensorboard.write_logs |= args.tensorboard
    job_config.outputs.zmq.publish_updates |= args.zmq
    job_config.outputs.zmq.subscribe_to_controller |= args.zmq
    if args.run_name != "":
        job_config.outputs.run_name = args.run_name
    if args.prefix != "":
        job_config.outputs.run_name_prefix = args.prefix
    if args.suffix != "":
        job_config.outputs.run_name_suffix = args.suffix
    job_config.outputs.save_visualizations |= args.save_viz
    if args.labels_path == "":
        args.labels_path = None
    args.video_paths = args.video_paths.split(",")
    if len(args.video_paths) == 0:
        args.video_paths = None

    logger.info("Versions:")
    sleap.versions()

    logger.info(f"Training labels file: {args.labels_path}")
    logger.info(f"Training profile: {job_filename}")
    logger.info("")

    # Log configuration to console.
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    logger.info("")
    logger.info("Training job:")
    logger.info(job_config.to_json())
    logger.info("")

    # Setup devices.
    if args.cpu or not sleap.nn.system.is_gpu_system():
        sleap.nn.system.use_cpu_only()
        logger.info("Running in CPU-only mode.")
    else:
        if args.first_gpu:
            sleap.nn.system.use_first_gpu()
            logger.info("Using the first GPU for acceleration.")
        elif args.last_gpu:
            sleap.nn.system.use_last_gpu()
            logger.info("Using the last GPU for acceleration.")
        else:
            sleap.nn.system.use_gpu(args.gpu)
            logger.info(f"Using GPU {args.gpu} for acceleration.")

        # Disable preallocation to handle Linux/low GPU memory issue.
        sleap.disable_preallocation()
        logger.info("Disabled GPU memory pre-allocation.")
    logger.info("System:")
    sleap.nn.system.summary()
    logger.info("")

    logger.info("Initializing trainer...")
    # Create a trainer and run!
    trainer = Trainer.from_config(
        job_config,
        training_labels=args.labels_path,
        validation_labels=args.val_labels,
        test_labels=args.test_labels,
        video_search_paths=args.video_paths,
    )
    trainer.train()


if __name__ == "__main__":
    main()

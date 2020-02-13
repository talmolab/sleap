"""Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
       human, as well as easily be serialized/deserialized to/from simple dictionaries
       and JSON.

    2. Decoupling from the implementation. This makes it easier to design functional
       modules with attributes/parameters that contain objects that may not be easily
       serializable or may implement additional logic that relies on runtime information
       or other parameters.

In general, classes that implement the actual functionality related to these
configuration classes should provide a classmethod for instantiation from the
configuration class instances. This makes it easier to implement other logic not related
to the high level parameters at creation time.

Conveniently, this format also provides a single location where all user-facing
parameters are aggregated and documented for end users (as opposed to developers).
"""

import attr
from typing import Optional, Union, Text, List

import sleap
from sleap.nn.data.augmentation import AugmentationConfig
from sleap.nn.model_config import ModelConfig


@attr.s(auto_attribs=True)
class LabelsConfig:
    """Labels configuration.

    Attributes:
        training_labels: A `sleap.Labels` instance or filepath to a saved labels file
            containing user labeled frames to use for generating the training set.
        validation_labels: A `sleap.Labels` instance or filepath to a saved labels file
            containing user labeled frames to use for generating validation data. These
            will not be trained on directly, but will be used to tune hyperparameters
            such as learning rate or early stopping. If not specified, the validation
            set will be sampled from the training labels.
        validation_fraction: Float between 0 and 1 specifying the fraction of the
            training set to sample for generating the validation set. The remaining
            labeled frames will be left in the training set. If the `validation_labels`
            are already specified, this has no effect.
        test_labels: A `sleap.Labels` instance or filepath to a saved labels file
            containing user labeled frames to use for generating the test set. This is
            typically a held out set of examples that are never used for training or
            hyperparameter tuning (like the validation set). This is optional, but
            useful for benchmarking as metrics can be computed from these data during
            model optimization. This is also useful to explicitly keep track of the test
            set that should be used when multiple splits are created for training.
        search_path_hints: List of paths to use for searching for missing data. This is
            useful when labels and data are moved across computers, network storage, or
            operating systems that may have different absolute paths than those stored
            in the labels. This has no effect if the labels were exported as a package
            with the user labeled data.
        skeletons: List of `sleap.Skeleton` instances that can be used by the model. If
            not specified, these will be pulled out of the labels during training, but
            must be specified for inference in order to generate predicted instances.
    """

    training_labels: Optional[Union[Text, sleap.Labels]] = None
    validation_labels: Optional[Union[Text, sleap.Labels]] = None
    validation_fraction: float = 0.1
    test_labels: Optional[Union[Text, sleap.Labels]] = None
    search_path_hints: List[Text] = attr.ib(factory=list)
    skeletons: List[sleap.Skeleton] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class PreprocessingConfig:
    """Preprocessing configuration.

    Attributes:
        ensure_rgb: If True, converts the image to RGB if not already.
        ensure_grayscale: If True, converts the image to grayscale if not already.
        imagenet_mode: Specifies an ImageNet-based normalization mode commonly used in
            `tf.keras.applications`-based pretrained models. This has no effect if None
            or not specified.
            Valid values are:
            "tf": Values will be scaled to [-1, 1], expanded to RGB if grayscale. This
                preprocessing mode is required when using pretrained ResNetV2,
                MobileNetV1, MobileNetV2 and NASNet models.
            "caffe": Values will be scaled to [0, 255], expanded to RGB if grayscale,
                RGB channels flipped to BGR, and subtracted by a fixed mean. This
                preprocessing mode is required when using pretrained ResNetV1 models.
            "torch": Values will be scaled to [0, 1], expanded to RGB if grayscale,
                subtracted by a fixed mean, and scaled by fixed standard deviation. This
                preprocessing mode is required when using pretrained DenseNet models.
        input_scale: Scalar float specifying scaling factor to resize raw images by.
            This can considerably increase performance and memory requirements at the
            cost of accuracy. Generally, it should only be used when the raw images are
            at a much higher resolution than the smallest features in the data.
        pad_to_stride: Number of pixels that the image size must be divisible by. This
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `input_scale` attribute. If set to "auto", this will be automatically
            detected from the model architecture. This must be divisible by the model's
            max stride (typically 32). This padding will be ignored when instance
            cropping inputs since the crop size should already be divisible by the
            model's max stride.
    """

    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    imagenet_mode: Optional[Text] = attr.ib(
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(["tf", "caffe", "torch"])
        ),
    )
    input_scaling: float = 1.0
    pad_to_stride: Union[Text, int] = "auto"


@attr.s(auto_attribs=True)
class InstanceCroppingConfig:
    """Instance cropping configuration.

    These are only used in topdown or centroid models.

    Attributes:
        center_on_part: String name of the part to center the instance to. If None or
            not specified, instances will be centered to the centroid of their bounding
            box. This value will be used for both topdown and centroid models. It must
            match the name of a node on the skeleton.
        crop_size: Integer size of bounding box height and width to crop out of the full
            image. This should be greater than the largest size of the instances in
            pixels. The crop is applied after any input scaling, so be sure to adjust
            this to changes in the input image scale. If set to "auto", this will be
            automatically detected from the data during training or from the model input
            layer during inference. This must be divisible by the model's max stride
            (typically 32).
        crop_size_detection_padding: Integer specifying how much extra padding should be
            applied around the instance bounding boxes when automatically detecting the
            appropriate crop size from the data. No effect if the `crop_size` is
            already specified.
    """

    center_on_part: Optional[Text] = None
    crop_size: Union[int, Text] = "auto"
    crop_size_detection_padding: int = 16


@attr.s(auto_attribs=True)
class DataConfig:
    """Data configuration.

    labels: Configuration options related to user labels for training or testing.
    preprocessing: Configuration options related to data preprocessing.
    instance_cropping: Configuration options related to instance cropping for centroid
        and topdown models.
    """

    labels: LabelsConfig = attr.ib(factory=LabelsConfig)
    preprocessing: PreprocessingConfig = attr.ib(factory=PreprocessingConfig)
    instance_cropping: InstanceCroppingConfig = attr.ib(factory=InstanceCroppingConfig)


@attr.s(auto_attribs=True)
class HardKeypointMiningConfig:
    """Configuration for online hard keypoint mining.

    Attributes:
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When
            this is enabled, the loss is computed per keypoint (or edge for PAFs) and
            sorted from lowest (easy) to highest (hard). The hard keypoint loss will be
            scaled to have a higher weight in the total loss, encouraging the training
            to focus on tricky body parts that are more difficult to learn.
            If False, no mining will be performed and all keypoints will be weighted
            equally in the loss.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with
            respect to the lowest keypoint loss in order to be considered as "hard".
            This helps to switch focus on across groups of keypoints during training.
        min_hard_keypoints: The minimum number of keypoints that will be considered as
            "hard", even if they are not below the `hard_to_easy_ratio`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to.
            This can help when there are few very easy keypoints which may skew the
            ratio and result in loss scaling being applied to most keypoints, which can
            reduce the impact of hard mining altogether.
        loss_scale: Factor to scale the hard keypoint losses by.
    """

    online_mining: bool = True
    hard_to_easy_ratio: float = 0.5
    min_hard_keypoints: int = 2
    max_hard_keypoints: Optional[int] = None
    loss_scale: float = 5.0


@attr.s(auto_attribs=True)
class LearningRateScheduleConfig:
    """Configuration for learning rate scheduling.

    Attributes:
        reduce_on_plateau: If True, learning rate will be reduced when the validation
            set loss plateaus. This improves training at later epochs when finer weight
            updates are required for fine-tuning the optimization, balancing out an
            initially high learning rate required for practical initial optimization.
        reduction_factor: Factor by which the learning rate will be scaled when a
            plateau is detected.
        plateau_min_delta: Minimum absolute decrease in the loss in order to consider an
            epoch as not in a plateau.
        plateau_patience: Number of epochs without an improvement of at least
            `plateau_min_delta` in order for a plateau to be detected.
        plateau_cooldown: Number of epochs after a reduction step before epochs without
            improvement will begin to be counted again.
    """

    reduce_on_plateau: bool = True
    reduction_factor: float = 0.5
    plateau_min_delta: float = 1e-6
    plateau_patience: int = 5
    plateau_cooldown: int = 3


@attr.s(auto_attribs=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        stop_training_on_plateau: If True, the training will terminate automatically
            when the validation set loss plateaus. This can save time and compute
            resources when there are minimal improvements to be gained from further
            training, as well as to prevent training into the overfitting regime.
        plateau_min_delta: Minimum absolute decrease in the loss in order to consider an
            epoch as not in a plateau.
        plateau_patience: Number of epochs without an improvement of at least
            `plateau_min_delta` in order for a plateau to be detected.
        plateau_cooldown: Number of epochs after a reduction step before epochs without
            improvement will begin to be counted again.
    """

    stop_training_on_plateau: bool = True
    plateau_min_delta: float = 1e-6
    plateau_patience: int = 10
    plateau_cooldown: int = 3


@attr.s(auto_attribs=True)
class OptimizationConfig:
    """Optimization configuration.

    Attributes:
        preload_data: If True, the data from the training/validation/test labels will be
            loaded into memory at the beginning of training. If False, the data will be
            loaded every time it is accessed. Preloading can considerably speed up the
            performance of data generation at the cost of memory as all the images will
            be loaded into memory. This is especially beneficial for datasets with few
            examples or when the raw data is on (slow) network storage.
        augmentation_config: Configuration options related to data augmentation.
        online_shuffling: If True, data will be shuffled online by maintaining a buffer
            of examples that are sampled from at each step. This allows for
            randomization of data ordering, resulting in more varied mini-batch
            composition, which in turn can promote generalization. Note that the data
            are shuffled at the start of training regardless.
        shuffle_buffer_size: Number of examples to keep in a buffer to sample uniformly
            from. This should be set to relatively low number as it requires an
            additional copy of each data example to be stored in memory. If set to -1,
            the entire dataset will be used, which results in perfect randomization at
            the cost of increased memory usage.
        prefetch: If True, data will generated in parallel to training to minimize the
            bottleneck of the preprocessing pipeline.
        batch_size: Number of examples per minibatch, i.e., a single step of training.
            Higher numbers can increase generalization performance by averaging model
            gradient updates over a larger number of examples at the cost of
            considerably more GPU memory, especially for larger sized images. Lower
            numbers may lead to overfitting, but may be beneficial to the optimization
            process when few but varied examples are available.
        batches_per_epoch: Number of minibatches (steps) to train for in an epoch. If
            set to "auto", this is set to the number of batches in the training data or
            `min_batches_per_epoch`, whichever is largest. At the end of each epoch, the
            validation and test sets are evaluated, the model is saved if its
            performance improved, visualizations are generated, learning rate may be
            tuned, and several other non-optimization procedures executed.
            If this is set too low, training may be slowed down as these end-of-epoch
            procedures can take longer than the optimization itself, especially if
            model saving is enabled, which can take a while for larger models. If set
            too high, the training procedure may take longer, especially since several
            hyperparameter tuning heuristics only consider epoch-to-epoch performance.
        min_batches_per_epoch: The minimum number of batches per epoch if
            `batches_per_epoch` is set to "auto". No effect if the batches per epoch is
            explicitly specified. This should be set to 200-400 to compensate for short
            loops through the data when there are few examples.
        epochs: Maximum number of epochs to train for. Training can be stopped manually
            or automatically if early stopping is enabled and a plateau is detected.
        optimizer: Name of the optimizer to use for training. This is typically "adam"
            but the name of any class from `tf.keras.optimizers` may be used. If "adam"
            is specified, the Adam optimizer will have AMSGrad enabled.
        initial_learning_rate: The initial learning rate to use for the optimizer. This
            is typically set to 1e-3 or 1e-4, and can be decreased automatically if
            learning rate reduction on plateau is enabled. If this is too high or too
            low, the training may fail to find good initial local minima to descend.
        learning_rate_schedule: Configuration options related to learning rate
            scheduling.
        hard_keypoint_mining: Configuration options related to online hard keypoint
            mining.
        early_stopping: Configuration options related to early stopping of training on
            plateau/convergence is detected.
    """

    preload_data: bool = True
    augmentation_config: AugmentationConfig = attr.ib(factory=AugmentationConfig)
    online_shuffling: bool = True
    shuffle_buffer_size: int = 128
    prefetch: bool = True
    batch_size: int = 8
    batches_per_epoch: Union[int, Text] = "auto"
    min_batches_per_epoch: int = 200
    epochs: int = 100
    optimizer: Text = "adam"
    initial_learning_rate: float = 1e-4
    learning_rate_schedule: LearningRateScheduleConfig = attr.ib(
        factory=LearningRateScheduleConfig
    )
    hard_keypoint_mining: HardKeypointMiningConfig = attr.ib(
        factory=HardKeypointMiningConfig
    )
    early_stopping: EarlyStoppingConfig = attr.ib(factory=EarlyStoppingConfig)


@attr.s(auto_attribs=True)
class CheckpointingConfig:
    """Configuration of model checkpointing.

    Attributes:
        initial_model: If True, the initial model is saved before any training occurs.
            If the model was not pretrained, these will just be the model with random
            weights. This is mostly useful for comparisons to a random baseline. If
            enabled, the model will be serialized to:
                "{run_folder}/initial_model.h5"
        best_model: If True, the model will be saved at the end of an epoch if the
            validation loss has improved. If enabled, the model will be serialized to:
                "{run_folder}/best_model.h5"
        every_epoch: If True, the model will be saved at the end of every epoch,
            regardless of whether there was an improvement detected. If enabled, the
            models will be serialized to:
                "{run_folder}/model.epoch{04d}.h5"
        latest_model: If True, the model will be saved at the end of every epoch,
            regardless of whether there was an improvement detected, but will overwrite
            the previous latest model. If enabled, the model will be serialized to:
                "{run_folder}/latest_model.h5"
        final_model: If True, the model will be saved at the end of training, whether it
            was stopped early or finished all epochs. If enabled, the model will be
            serialized to:
                "{run_folder}/final_model.h5"
    """

    initial_model: bool = False
    best_model: bool = True
    every_epoch: bool = False
    latest_model: bool = False
    final_model: bool = False


@attr.s(auto_attribs=True)
class TensorBoardConfig:
    """Configuration of TensorBoard-based monitoring of the training.

    Attributes:
        write_logs: If True, logging data will be written to disk within the run folder.
            TensorBoard can monitor either the specific run folder, or the parent runs
            folder that may contain multiple models/runs. Both will be displayed
            correctly in the dashboard.
        loss_frequency: How often loss and metrics will be written out to disk. This can
            be "epoch" to only write summaries at the end of every epoch, "batch" to
            write summaries after every batch, or an integer to specify the frequency as
            a number of batches. High frequency writing can considerably slow down
            training, so this is only recommended to be anything other than "epoch" if
            training interactively. This value only affects the monitored losses and
            metrics, not other summaries like visualizations.
        part_metrics: If True, metrics will be reported for each keypoint type for
            models with part confidence maps and edge type for PAF models.
        architecture_graph: If True, the architecture of the model will be saved
            and can be viewed graphically in TensorBoard. This is only saved at the
            beginning of training, but can consume a lot of disk space for large models,
            as well as potentially freezing the browser tab when rendered.
        visualizations: If True, visualizations of the model predictions are rendered
            and logged for display in TensorBoard -> Images.
    """

    write_logs: bool = False
    loss_frequency: Union[Text, int] = "epoch"
    part_metrics: bool = True
    architecture_graph: bool = False
    visualizations: bool = True


@attr.s(auto_attribs=True)
class ZMQConfig:
    """Configuration of ZeroMQ-based monitoring of the training.

    Attributes:
        subscribe_to_controller: If True, will listen for commands broadcast over a
            socket or another messaging endpoint using the ZeroMQ SUB protocol. This
            allows for external/asynchronous control of the training loop from other
            programs, e.g., GUIs or job schedulers. Commands are expected to be
            JSON-serialized strings of dictionaries with a key named "command". The
            endpoint is polled for messages at the end of each batch.
            Current commands are:
                "stop": Stops the training after the current batch.
                "set_lr": Sets the optimizer's learning rate after the current batch.
                    The new learning rate should be a float specified in the "lr" key.
        controller_address: IP address/hostname and port number of the endpoint to
            listen for command messages from. For TCP-based endpoints, this must be in
            the form of "tcp://{ip_address}:{port_number}". Defaults to
            "tcp://127.0.0.1:9000".
        controller_polling_timeout: Polling timeout in microseconds specified as an
            integer. This controls how long the poller should wait to receive a response
            and should be set to a small value to minimize the impact on training speed.
        publish_updates: If True, training summaries will be broadcast over a socket or
            another messaging endpoint using the ZeroMQ PUB protocol. This is useful for
            asynchronously monitoring training with external programs without writing to
            the file system and without requiring special dependencies like TensorBoard.
            All data will be broadcast as JSON serialized strings.
            TODO: Describe published message keys.
        publish_address: IP address/hostname and port number of the endpoint to publish
            updates to. For TCP-based endpoints, this must be in the form of
            "tcp://{ip_address}:{port_number}". Defaults to "tcp://127.0.0.1:9001".
    """

    subscribe_to_controller: bool = False
    controller_address: Text = "tcp://127.0.0.1:9000"
    controller_polling_timeout: int = 10
    publish_updates: bool = False
    publish_address: Text = "tcp://127.0.0.1:9001"


@attr.s(auto_attribs=True)
class OutputsConfig:
    """Configuration of training outputs.
    
    Attributes:
        save_outputs: If True, file system-based outputs will be saved. If False,
            nothing will be written to disk, which may be useful for interactive
            training where no outputs are desired.
        run_name: Name of the training run. This is the name of the folder that all
            outputs related to the training job are stored. If not specified explicitly,
            this will be automatically generated from the configuration options and the
            timestamp of the start of the training job.
            Note that if this is specified rather than automatically generated, multiple
            runs can end up overwriting each other if `run_name_prefix` or
            `run_name_suffix` are not specified.
        run_name_prefix: String to prepend to the run name. This is useful to prevent
            multiple runs started at the same exact time to be mapped to the same
            folder, or when a fixed run name is specified.
        run_name_suffix: String to append to the run name. This is useful to prevent
            multiple runs started at the same exact time to be mapped to the same
            folder, or when a fixed run name is specified. If set to None, this will be
            automatically set to a number (e.g., "_1") that does not conflict with an
            existing folder, so sequential jobs with a fixed run name will have an
            increasing counter as the suffix.
            Warning: This can fail to prevent overwriting if multiple jobs are run in
                parallel and attempt to detect the run name at the same time, especially
                over network storage which can have a short delay in updating the
                directory listing across clients.
        runs_folder: Path to the folder that run data should be stored in. All the data
            for a single run are stored in the path:
                "{runs_folder}/{run_name_prefix}{run_name}{run_name_suffix}"
            These are specified separately to allow the `run_name` to be auto-generated.
            This can be specified as an absolute or relative path. Relative paths
            specify a path with respect to the current working directory. Non-existing
            folders will be created if they do not already exist. Defaults to the
            "models" subdirectory of the current working directory.
        tags: A list of strings to use as "tags" that can be used to organize multiple
            runs. These are not used for anything during training or inference, so they
            can be used to store arbitrary user-specified metadata.
        save_visualizations: If True, will render and save visualizations of the model
            predictions as PNGs to "{run_folder}/viz/{split}/{04d}.png", where the
            split is one of "train", "val", "test", and the filenames are the epoch.
        log_to_csv: If True, loss and metrics will be saved to a simple CSV after each
            epoch to "{run_folder}/training_log.csv"
        checkpointing: Configuration options related to model checkpointing.
        tensorboard: Configuration options related to TensorBoard logging.
        zmq: Configuration options related to ZeroMQ-based control and monitoring.
    """

    save_outputs: bool = True
    run_name: Optional[Text] = None
    run_name_prefix: Text = ""
    run_name_suffix: Optional[Text] = None
    runs_folder: Text = "models"
    tags: List[Text] = attr.ib(factory=list)
    save_visualizations: bool = True
    log_to_csv: bool = True
    checkpointing: CheckpointingConfig = attr.ib(factory=CheckpointingConfig)
    tensorboard: TensorBoardConfig = attr.ib(factory=TensorBoardConfig)
    zmq: ZMQConfig = attr.ib(factory=ZMQConfig)


@attr.s(auto_attribs=True)
class TrainingJobConfig:
    """Configuration of a training job.

    Attributes:
        data: Configuration options related to the training data.
        model: Configuration options related to the model architecture.
        optimization: Configuration options related to the training.
        outputs: Configuration options related to outputs during training.
    """

    data: DataConfig = attr.ib(factory=DataConfig)
    # model: ModelConfig = attr.ib(factory=ModelConfig)
    optimization: OptimizationConfig = attr.ib(factory=OptimizationConfig)
    outputs: OutputsConfig = attr.ib(factory=OutputsConfig)
    # TODO: store fixed config format version + SLEAP version?

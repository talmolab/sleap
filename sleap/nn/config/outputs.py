import attr
from typing import Optional, Text, List
import os


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
            write summaries after every batch. High frequency writing can considerably
            slow down training, so this is not recommended to be anything other than
            "epoch" if training interactively. This value only affects the monitored
            losses and metrics, not other summaries like visualizations.
        architecture_graph: If True, the architecture of the model will be saved
            and can be viewed graphically in TensorBoard. This is only saved at the
            beginning of training, but can consume a lot of disk space for large models,
            as well as potentially freezing the browser tab when rendered.
        profile_graph: If True, profiles the second batch of examples to collect compute
            statistics.
        visualizations: If True, visualizations of the model predictions are rendered
            and logged for display in TensorBoard -> Images.
    """

    write_logs: bool = False
    loss_frequency: Text = "epoch"
    architecture_graph: bool = False
    profile_graph: bool = False
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
            predictions as PNGs to "{run_folder}/viz/{split}.{epoch:04d}.png", where the
            split is one of "train", "validation", "test".
        delete_viz_images: If True, delete the saved visualizations after training
            completes. This is useful to reduce the model folder size if you do not need
            to keep the visualization images.
        zip_outputs: If True, compress the run folder to a zip file. This will be named
            "{run_folder}.zip".
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
    delete_viz_images: bool = True
    zip_outputs: bool = False
    log_to_csv: bool = True
    checkpointing: CheckpointingConfig = attr.ib(factory=CheckpointingConfig)
    tensorboard: TensorBoardConfig = attr.ib(factory=TensorBoardConfig)
    zmq: ZMQConfig = attr.ib(factory=ZMQConfig)

    @property
    def run_path(self) -> Text:
        """Return the complete run path where all training outputs are stored.

        This path is determined by other attributes using the pattern:
            `{runs_folder}/{run_name_prefix}{run_name}{run_name_suffix}`

        If `run_name_suffix` is set to None, it will be ignored.

        Raises:
            ValueError: If `run_name` is not set.

        Notes:
            This does not perform any checks for existence or validity and should only
            be used when the above fields are complete.

            This path will not be updated if the files are moved. To ensure this path is
            valid, use a relative path for the `runs_folder` or manually update it.
        """
        if self.run_name is None:
            raise ValueError(
                "Run path cannot be determined when the run name is not set."
            )
        folder_name = self.run_name_prefix + self.run_name
        if self.run_name_suffix is not None:
            folder_name += self.run_name_suffix
        return os.path.join(self.runs_folder, folder_name)

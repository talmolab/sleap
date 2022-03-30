"""Training-related tf.keras callbacks."""

import jsonpickle
import logging
import numpy as np
import tensorflow as tf
import zmq
import io
import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Text, Callable, Optional


logger = logging.getLogger(__name__)


class TrainingControllerZMQ(tf.keras.callbacks.Callback):
    def __init__(self, address="tcp://127.0.0.1:9000", topic="", poll_timeout=10):
        self.address = address
        self.topic = topic
        self.timeout = poll_timeout

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.subscribe(self.topic)
        self.socket.connect(self.address)
        logger.info(
            f"Training controller subscribed to: {self.address} (topic: {self.topic})"
        )

        # TODO: catch/throw exception about failure to connect

        # Callback initialization
        super().__init__()

    def __del__(self):
        logger.info(f"Closing the training controller socket/context.")
        self.socket.close()
        self.context.term()

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        if self.socket.poll(self.timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.socket.recv_string())
            logger.info(f"Received control message: {msg}")

            # Stop training
            if msg["command"] == "stop":
                # self.model is set when training begins in Model.fit_generator
                self.model.stop_training = True

            # Adjust learning rate
            elif msg["command"] == "set_lr":
                self.set_lr(msg["lr"])

    def set_lr(self, lr):
        """Adjust the model learning rate.

        This is the based off of the implementation used in the native learning rate
        scheduling callbacks.
        """
        if not isinstance(lr, (float, np.float32, np.float64)):
            lr = np.array(lr).astype(np.float64)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class ProgressReporterZMQ(tf.keras.callbacks.Callback):
    def __init__(self, address="tcp://127.0.0.1:9001", what="not_set"):
        self.address = address
        self.what = what
        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(self.address)
        logger.info(f"Progress reporter publishing on: {self.address} for: {self.what}")

        # TODO: catch/throw exception about failure to connect

        # Callback initialization
        super().__init__()

    def __del__(self):
        logger.info(f"Closing the reporter controller/context.")
        self.socket.setsockopt(zmq.LINGER, 0)
        # url = self.socket.LAST_ENDPOINT
        # self.socket.unbind(url)
        self.socket.close()
        self.context.term()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.socket.send_string(
            jsonpickle.encode(dict(what=self.what, event="train_begin", logs=logs))
        )

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""
        # self.logger.info("batch_begin")
        self.socket.send_string(
            jsonpickle.encode(
                dict(what=self.what, event="batch_begin", batch=batch, logs=logs)
            )
        )

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
        self.socket.send_string(
            jsonpickle.encode(
                dict(what=self.what, event="batch_end", batch=batch, logs=logs)
            )
        )

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.socket.send_string(
            jsonpickle.encode(
                dict(what=self.what, event="epoch_begin", epoch=epoch, logs=logs)
            )
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        self.socket.send_string(
            jsonpickle.encode(
                dict(what=self.what, event="epoch_end", epoch=epoch, logs=logs)
            )
        )

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.socket.send_string(
            jsonpickle.encode(dict(what=self.what, event="train_end", logs=logs))
        )


class ModelCheckpointOnEvent(tf.keras.callbacks.Callback):
    """Callback for model checkpointing on a fixed event.

    Attributes:
        filepath: Path to save model to.
        event: Event to trigger model saving ("train_start" or "train_end").
    """

    def __init__(self, filepath: str, event: str = "train_end"):
        self.filepath = filepath
        self.event = event

        # Callback initialization
        super().__init__()

    def on_train_begin(self, logs=None):
        """Called at the start of training."""
        if self.event == "train_begin":
            self.model.save(self.filepath)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if self.event == "epoch_end":
            if "%" in self.filepath:
                self.model.save(self.filepath % epoch)
            else:
                self.model.save(self.filepath)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.event == "train_end":
            self.model.save(self.filepath)


class TensorBoardMatplotlibWriter(tf.keras.callbacks.Callback):
    """Callback for writing image summaries with visualizations during training.

    Attributes:
        logdir: Path to log directory.
        plot_fn: Function with no arguments that returns a matplotlib figure handle.
        tag: Text to append to the summary label in TensorBoard.
    """

    def __init__(
        self,
        log_dir: Text,
        plot_fn: Callable[[], matplotlib.figure.Figure],
        tag: Text = "viz",
    ):
        self.log_dir = log_dir
        self.plot_fn = plot_fn
        self.tag = tag

        self.file_writer = tf.summary.create_file_writer(self.log_dir)

        # Callback initialization
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""

        # Call plotting function.
        figure = self.plot_fn()

        # Render to in-memory PNG.
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", pad_inches=0)
        plt.close(figure)

        # Convert PNG to tensor.
        image_buffer.seek(0)
        image_tensor = tf.expand_dims(
            tf.image.decode_png(image_buffer.getvalue(), channels=4), axis=0
        )

        # Log to TensorBoard.
        with self.file_writer.as_default():
            tf.summary.image(name=self.tag, data=image_tensor, step=epoch)


class MatplotlibSaver(tf.keras.callbacks.Callback):
    """Callback for saving images rendered with matplotlib during training.

    This is useful for saving visualizations of the training to disk. It will be called
    at the end of each epoch.

    Attributes:
        plot_fn: Function with no arguments that returns a matplotlib figure handle.
            See `sleap.nn.training.Trainer.visualize_predictions` for example.
        save_folder: Path to a directory to save images to. This folder will be created
            if it does not exist.
        prefix: String that will be prepended to the filenames. This is useful for
            indicating which dataset the visualization was sampled from, for example.

    Notes:
        This will save images with the naming pattern:
            "{save_folder}/{prefix}.{epoch}.png"
        or:
            "{save_folder}/{epoch}.png"
        if a prefix is not specified.
    """

    def __init__(
        self,
        save_folder: Text,
        plot_fn: Callable[[], matplotlib.figure.Figure],
        prefix: Optional[Text] = None,
    ):
        """Initialize callback."""
        self.save_folder = save_folder
        self.plot_fn = plot_fn
        self.prefix = prefix
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """Save figure at the end of each epoch."""
        # Call plotting function.
        figure = self.plot_fn()

        # Check if output folder exists.
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Build filename.
        prefix = ""
        if self.prefix is not None:
            prefix = self.prefix + "."
        figure_path = os.path.join(self.save_folder, f"{prefix}{epoch:04d}.png")

        # Save rendered figure.
        figure.savefig(figure_path, format="png", pad_inches=0)
        plt.close(figure)

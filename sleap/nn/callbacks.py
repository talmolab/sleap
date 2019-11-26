"""Training-related tf.Keras callbacks."""

import jsonpickle
import logging
import numpy as np
import tensorflow as tf
import zmq

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    LambdaCallback,
    ModelCheckpoint,
    CSVLogger
)

logger = logging.getLogger(__name__)


class TrainingControllerZMQ(tf.keras.callbacks.Callback):
    def __init__(self, address="tcp://127.0.0.1", port=9000, topic="", poll_timeout=10):
        self.address = "%s:%d" % (address, port)
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
        """ Called at the end of a training batch. """
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
        """ Adjust the model learning rate.

        This is the based off of the implementation used in the native learning rate
        scheduling callbacks.
        """
        if not isinstance(lr, (float, np.float32, np.float64)):
            lr = np.array(lr).astype(np.float64)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class ProgressReporterZMQ(tf.keras.callbacks.Callback):
    def __init__(self, address="tcp://127.0.0.1", port=9001, what="not_set"):
        self.address = "%s:%d" % (address, port)
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

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.event == "train_end":
            self.model.save(self.filepath)

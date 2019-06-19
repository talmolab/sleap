import numpy as np
from time import time, sleep
from datetime import datetime
import os

import tensorflow as tf
import keras
from keras import backend as K

from sklearn.model_selection import train_test_split
from sleap.nn.augmentation import Augmenter

from keras.layers import Input, Conv2D, BatchNormalization, Add, MaxPool2D, UpSampling2D, Concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, LambdaCallback, ModelCheckpoint

from sleap.nn.architectures.common import conv
from sleap.nn.architectures.hourglass import stacked_hourglass
from sleap.nn.architectures.unet import unet, stacked_unet
from sleap.nn.architectures.leap import leap_cnn
from sleap.nn.monitor import LossViewer

from sleap.nn.datagen import generate_confmaps_from_points, generate_pafs_from_points

from multiprocessing import Process
import zmq
import jsonpickle


class TrainingControllerZMQ(keras.callbacks.Callback):
    def __init__(self, address="tcp://127.0.0.1", port=9000, topic="", poll_timeout=10):
        self.address = "%s:%d" % (address, port)
        self.topic = topic
        self.timeout = poll_timeout

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.subscribe(self.topic)
        self.socket.connect(self.address)
        print(f"Training controller subscribed to: {self.address} (topic: {self.topic})")

        # TODO: catch/throw exception about failure to connect

        # Callback initialization
        super().__init__()

    def __del__(self):
        print(f"Closing the training controller socket/context.")
        self.socket.close()
        self.context.term()

    def on_batch_end(self, batch, logs=None):
        """ Called at the end of a training batch. """
        if self.socket.poll(self.timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.socket.recv_string())
            print(f"Received control message: {msg}")

            # Stop training
            if msg["command"] == "stop":
                # self.model is set when training begins in Model.fit_generator
                self.model.stop_training = True

            # Adjust learning rate
            elif msg["command"] == "set_lr":
                self.set_lr(msg["lr"])

    def set_lr(self, lr):
        """ Adjust the model learning rate. 

        This is the based off of the implementation used in the native learning rate scheduling callbacks.
        """
        if not isinstance(lr, (float, np.float32, np.float64)):
            lr = np.array(lr).astype(np.float64)
        K.set_value(self.model.optimizer.lr, lr)


class ProgressReporterZMQ(keras.callbacks.Callback):
    def __init__(self, address="tcp://*", port=9001):
        self.address = "%s:%d" % (address, port)

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.address)
        print(f"Progress reporter publishing on: {self.address}")

        # TODO: catch/throw exception about failure to connect

        # Callback initialization
        super().__init__()

    def __del__(self):
        print(f"Closing the reporter controller/context.")
        self.socket.close()
        self.context.term()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        # self.logger.info("train_begin")
        self.socket.send_string(jsonpickle.encode(dict(event="train_begin", logs=logs)))


    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""
        # self.logger.info("batch_begin")
        self.socket.send_string(jsonpickle.encode(dict(event="batch_begin", batch=batch, logs=logs)))

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
        self.socket.send_string(jsonpickle.encode(dict(event="batch_end", batch=batch, logs=logs)))

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.socket.send_string(jsonpickle.encode(dict(event="epoch_begin", epoch=epoch, logs=logs)))

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
        self.socket.send_string(jsonpickle.encode(dict(event="epoch_end", epoch=epoch, logs=logs)))

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.socket.send_string(jsonpickle.encode(dict(event="train_end", logs=logs)))


def train(
    # Data
    imgs, outputs, skeleton=None, output_type="confmaps", val_size=0.1,
    # Architecture
    arch="unet", num_stacks=1, depth=3, convs_per_depth=2, num_filters=64, batch_norm=True, upsampling="bilinear", intermediate_inputs=True, intermediate_outputs=True,
    # Optimizer
    optimizer="adam", learning_rate=1e-3, amsgrad=True,
    # Augmentation
    augment_rotation=180, augment_scale_min=1.0, augment_scale_max=1.0,
    # Training loop
    batch_size=4, num_epochs=100, steps_per_epoch=200, shuffle_initially=True, multiprocessing_workers=0,
    # Saving
    save_dir=None, run_name=None, save_viz=False,
    # Callbacks: Intermediate saving
    save_every_epoch=False, save_best_val=True,
    # Callbacks: Shuffle after every epoch
    shuffle_every_epoch=True,
    # Callbacks: LR reduction
    reduce_lr_min_delta=1e-6, reduce_lr_factor=0.5, reduce_lr_patience=5, reduce_lr_cooldown=3, reduce_lr_min_lr=1e-8,
    # Callbacks: Early stopping
    early_stopping_min_delta=1e-8, early_stopping_patience=3,
    # Callbacks: Tensorboard
    tensorboard_dir=None,
    # Callbacks: ZMQ control
    control_zmq_port=9000,
    # Callbacks: ZMQ progress reporter
    progress_report_zmq_port=9001,
    # Callbacks: ZMQ visualization preview
    viz_preview_zmq_port=9002,
    ):

    # Split data into train/validation
    imgs_train, imgs_val, outputs_train, outputs_val = train_test_split(imgs, outputs, test_size=val_size)
    del imgs, outputs

    # Infer shapes
    num_train, img_height, img_width, img_channels = imgs_train.shape
    num_val = len(imgs_val)
    num_total = num_train + num_val

    if type(outputs_train) == np.ndarray:
        outputs_channels = outputs_train.shape[-1]
    elif output_type.lower() == "confmaps":
        outputs_channels = len(skeleton.nodes)
    elif output_type.lower() == "pafs":
        outputs_channels = len(skeleton.edges) * 2

    print(f"Training set: {imgs_train.shape} -> {outputs_channels} channels")
    print(f"Validation set: {imgs_val.shape} -> {outputs_channels} channels")

    # Build model architecture
    # TODO: separate architecture builder

    # Input layer
    img_input = Input((img_height, img_width, img_channels))

    # Rectify image sizes not divisible by pooling factor
    pool_factor = 2 ** depth
    if img_height % pool_factor != 0 or img_width % pool_factor != 0:
        print(f"Image dimensions ({img_height}, {img_width}) are not divisible by the pooling factor ({pool_factor}).")
        # gap_height = (np.ceil(img_height / pool_factor) * pool_factor) - img_height
        # gap_width = (np.ceil(img_width / pool_factor) * pool_factor) - img_width

        # TODO: combination of ZeroPadding2D and Cropping2D
        #   - can't do non-symmetric padding, so pad first and then crop left/top so coordinates aren't messed up
        #   - Solution: https://www.tensorflow.org/api_docs/python/tf/pad + Lambda layer + corresponding crop at the end?

    # Backbone
    if arch == "unet":
        x_outs = unet(img_input, outputs_channels, depth=depth, convs_per_depth=convs_per_depth, num_filters=num_filters, interp=upsampling)
    elif arch == "stacked_unet":
        x_outs = stacked_unet(img_input, outputs_channels, depth=depth, convs_per_depth=convs_per_depth, num_filters=num_filters, num_stacks=num_stacks, interp=upsampling)
    elif arch == "hourglass":
        # Initial downsampling
        x = conv(num_filters, kernel_size=(7, 7))(img_input)
        if batch_norm: x = BatchNormalization()(x)

        # Stacked hourglass modules
        x_outs = stacked_hourglass(x, outputs_channels, num_hourglass_blocks=num_stacks, num_filters=num_filters, depth=depth, batch_norm=batch_norm, interp=upsampling)
    elif arch == "leap" or arch == "leap_cnn":
        x_outs = leap_cnn(img_input, outputs_channels, down_blocks=depth, up_blocks=depth, upsampling_layers=True, num_filters=num_filters, interp=upsampling)

    # Create training model
    model = keras.Model(inputs=img_input, outputs=x_outs)

    # Create optimizer
    if optimizer.lower() == "adam":
        _optimizer = keras.optimizers.Adam(lr=learning_rate, amsgrad=amsgrad)
    elif optimizer.lower() == "rmsprop":
        _optimizer = keras.optimizers.RMSprop(lr=learning_rate)

    # Compile
    model.compile(
        optimizer=_optimizer,
        loss="mean_squared_error",
    )
    print("Params: {:,}".format(model.count_params()))

    # Default to one loop through dataset per epoch
    if steps_per_epoch is None:
        steps_per_epoch = np.ceil(len(imgs_train) / batch_size).astype(int)

    # Setup saving
    save_path = None
    if save_dir is not None:
        # Generate run name
        if run_name is None:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            run_name = f"{timestamp}.{output_type}.{arch}.n={num_total}"

        # Build save path
        save_path = os.path.join(save_dir, run_name)
        print(f"Save path: {save_path}")

        # Check if it already exists
        if os.path.exists(save_path):
            print(f"Warning: Save path already exists. Previous run data may be overwritten.")

        # Create run folder
        os.makedirs(save_path, exist_ok=True)

        # TODO: save serialized training info

    img_shape = (imgs_train.shape[1], imgs_train.shape[2])
    if output_type.lower() == "confmaps":
        def datagen_function(points):
            return generate_confmaps_from_points(points, skeleton, img_shape)
    elif output_type.lower() == "pafs":
        def datagen_function(points):
            return generate_pafs_from_points(points, skeleton, img_shape)
    else:
        datagen_function = None

    if datagen_function is not None:
        outputs_val = datagen_function(outputs_val)

    # Initialize data generator with augmentation
    train_datagen = Augmenter(imgs_train, Points=outputs_train,
        datagen=datagen_function, output_names=model.output_names,
        batch_size=batch_size, shuffle=shuffle_initially,
        rotation=augment_rotation, scale=(augment_scale_min, augment_scale_max))

    # Setup callbacks
    callbacks = []

    # Callbacks: Intermediate saving
    if save_path is not None:
        if save_every_epoch:
            callbacks.append(
                ModelCheckpoint(filepath=os.path.join(save_path, "newest_model.h5"), monitor="val_loss", save_best_only=False, save_weights_only=False, period=1)
                )
        if save_best_val:
            callbacks.append(
                ModelCheckpoint(filepath=os.path.join(save_path, "best_model.h5"), monitor="val_loss", save_best_only=True, save_weights_only=False, period=1)
                )

    # Callbacks: Shuffle after every epoch
    if shuffle_every_epoch:
        callbacks.append(
            LambdaCallback(on_epoch_end=lambda epoch, logs: train_datagen.shuffle())
            )

    # Callbacks: LR reduction
    callbacks.append(
        ReduceLROnPlateau(min_delta=reduce_lr_min_delta, factor=reduce_lr_factor, patience=reduce_lr_patience,
            cooldown=reduce_lr_cooldown, min_lr=reduce_lr_min_lr, monitor="val_loss", mode="auto", verbose=1,)
        )

    # Callbacks: Early stopping
    callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=early_stopping_min_delta, patience=early_stopping_patience, verbose=1)
        )

    # Callbacks: Tensorboard
    if tensorboard_dir is not None:
        callbacks.append(
            TensorBoard(log_dir=f"{tensorboard_dir}/{arch}{time()}",
                batch_size=32, update_freq=150, histogram_freq=0,
                write_graph=False, write_grads=False, write_images=False,
                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
            )

    # Callbacks: ZMQ control
    if control_zmq_port is not None:
        callbacks.append(
            TrainingControllerZMQ(address="tcp://127.0.0.1", port=control_zmq_port, topic="", poll_timeout=10)
            )

    # Callbacks: ZMQ progress reporter
    if progress_report_zmq_port is not None:
        callbacks.append(
            ProgressReporterZMQ(port=progress_report_zmq_port)
            )

    # Callbacks: ZMQ visualization preview
    # viz_preview_zmq_port=9002,

    # Train!
    history = model.fit_generator(
        train_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=(imgs_val, {output_name: outputs_val for output_name in model.output_names}),
        callbacks=callbacks,
        verbose=2,
        use_multiprocessing=multiprocessing_workers > 0,
        workers=multiprocessing_workers,
    )

    # Save once done training
    if save_path is not None:
        final_model_path = os.path.join(save_path, "final_model.h5")
        model.save(filepath=final_model_path, overwrite=True, include_optimizer=True)
        print(f"Saved final model: {final_model_path}")

        # TODO: save metadata (e.g., skeleton)
        # TODO: save training history

    return model


if __name__ == "__main__":
    
    from PySide2 import QtWidgets

    from sleap.io.dataset import Labels
    labels = Labels.load_json("tests/data/json_format_v1/centered_pair.json")

    from sleap.nn.datagen import generate_images, generate_points
    t0 = time()
    imgs = generate_images(labels)
    points = generate_points(labels)
    skeleton = labels.skeletons[0]

    print(f"Generated training data. [{time() - t0:.2f}s]")

    proc = Process(target=train, args=(imgs, points), kwargs=dict(output_type="confmaps", skeleton=skeleton, val_size=0.1, batch_norm=False, num_filters=16, batch_size=4, num_epochs=100, steps_per_epoch=100, arch="unet"))
    proc.start()

    ctx = zmq.Context()

    app = QtWidgets.QApplication()
    loss_viewer = LossViewer(zmq_context = ctx)
    loss_viewer.resize(600, 400)
    loss_viewer.show()
    app.setQuitOnLastWindowClosed(True)
    app.processEvents()

    # Controller
    ctrl = ctx.socket(zmq.PUB)
    ctrl.bind("tcp://*:9000")

    # Progress monitoring
    sub = ctx.socket(zmq.SUB)
    sub.subscribe("")
    sub.connect("tcp://127.0.0.1:9001")

    def poll(timeout=10):
        if sub.poll(timeout, zmq.POLLIN):
            return jsonpickle.decode(sub.recv_string())
        return None

    t0 = time()

    epoch = 0
    while (time() - t0) < 30:
    # while True:
        msg = poll()
        if msg is not None:
            print(msg)
            if msg["event"] == "train_begin":
                loss_viewer.set_start_time(time())
            elif msg["event"] == "epoch_begin":
                epoch = msg["epoch"]
            elif msg["event"] == "batch_end":
                loss_viewer.add_datapoint((epoch * 100) + msg["logs"]["batch"], msg["logs"]["loss"])

        loss_viewer.update_runtime()
        app.processEvents()

    # Stop training
    ctrl.send_string(jsonpickle.encode(dict(command="stop")))

    while True:
        app.processEvents()

import os
import json

import logging
logger = logging.getLogger(__name__)

import numpy as np
import attr
import cattr
import zmq
import jsonpickle
import keras

from multiprocessing import Process, Pool
from multiprocessing.pool import AsyncResult

from typing import Union, List, Dict, Tuple
from time import time, sleep
from datetime import datetime

from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, Add, MaxPool2D, UpSampling2D, Concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, LambdaCallback, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sleap.skeleton import Skeleton
from sleap.io.dataset import Labels
from sleap.nn.augmentation import Augmenter
from sleap.nn.model import Model, ModelOutputType
from sleap.nn.monitor import LossViewer
from sleap.nn.datagen import generate_confmaps_from_points, generate_pafs_from_points, generate_images, generate_points, instance_crops


@attr.s(auto_attribs=True)
class Trainer:
    """
    The Trainer class implements a training program on a SLEAP model.
    Its main purpose is to capture training program hyperparameters
    separate from model specific hyper parameters. The train function is
    used to invoke training on a specific model and dataset.

    Note: some of these values are passed directly to lower level Keras
    APIs and thus their documentation has been lifted from https://keras.io.
    This should be noted for these paramters, please consulte https://keras.io
    for the most up to date documentation for these parameters.

    Args:
        val_size: If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples. If None,
        the value is set to it will be set to 0.25.
        optimizer: Either "adam" or "rmsprop". This selects whether to use
        keras.optimizers.Adam or keras.optimizers.RMSProp for the underlying
        keras model optimizer.
        learning_rate: The learning rate to pass to Adam or RMSProp.
        amsgrad: If optimizer is "adam", whether to apply
        the AMSGrad variant of this algorithm from the paper
        "On the Convergence of Adam and Beyond". If optimizer is "RMSProp",
        unused.

        batch_size: The batch size to use for training.
        num_epochs: Number of epochs to train the model. An epoch is an
        iteration over the entire data provided, as defined by steps_per_epoch.
        steps_per_epoch: Total number of steps (batches of samples) to yield
        from generator before declaring one epoch finished and starting the
        next epoch. If set to None, do one loop through dataset per epoch.
        shuffle_initially: Whether to shuffle training data before training begins.
        shuffle_every_epoch: Whether to continually shuffle the data after each epoch.

        augment_rotation: A float or two dimensional tuple expressing the range
        of rotataion (in degrees) augmentations to perform. If rotation is a scalar,
        it is transformed to (-rotation, +rotation)
        augment_scale_min: The lower bound of random scaling factors to apply to the
        images when augmenting. If augment_scale_min == augment_scale_max then scaling
        augmentation will be skipped.
        augment_scale_max: The upper bound of random scaling factors to apply to the
        images when augmenting. If augment_scale_min == augment_scale_max then scaling
        augmentation will be skipped.

        save_dir: The directory to save trained Keras models.
        save_every_epoch: Should a model be saved every epoch.
        save_best_val: Should we save the model with the lowest validation loss.

        reduce_lr_min_delta: Threshold for measuring the new optimum, to only focus on significant
        changes. Parameter passed to keras callback ReduceLROnPlateau(min_delta, monitor='val_loss').
        reduce_lr_factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
        Parameter passed to keras callback ReduceLROnPlateau(factor, monitor='val_loss').
        reduce_lr_patience: Number of epochs with no improvement after which learning rate will be reduced.
        Parameter passed to keras callback ReduceLROnPlateau(patience, monitor='val_loss').
        reduce_lr_cooldown: Number of epochs to wait before resuming normal operation after lr has
        been reduced.Parameter passed to keras callback ReduceLROnPlateau(cooldown, monitor='val_loss').
        reduce_lr_min_lr: Lower bound on the learning rate. Parameter passed to keras callback
        ReduceLROnPlateau(min_lr, monitor='val_loss').

        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement,
        i.e. an absolute change of less than min_delta, will count as no improvement. Parameter passed to
        keras callback EarlyStopping(min_delta, monitor='val_loss')
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
        Parameter passed to keras callback EarlyStopping(patience, monitor='val_loss')

        scale: Scale factor for downsampling (2 means we downsample by factor of 2)
        instance_crop: Whether to crop images around each instance (and adjust points)
    """

    val_size: float = 0.1
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    amsgrad: bool = True
    batch_size: int = 4
    num_epochs: int = 100
    steps_per_epoch: int = 200
    shuffle_initially: bool = True
    shuffle_every_epoch: bool = True
    augment_rotation: float = 180.0
    augment_scale_min: float = 1.0
    augment_scale_max: float = 1.0
    save_every_epoch: bool = False
    save_best_val: bool = True
    reduce_lr_min_delta: float = 1e-6
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: float = 5
    reduce_lr_cooldown: float = 3
    reduce_lr_min_lr: float = 1e-8
    early_stopping_min_delta: float = 1e-8
    early_stopping_patience: float = 3
    scale: int = 1
    instance_crop: bool = False

    def train(self,
              model: Model,
              labels: Union[str, Labels],
              run_name: str = None,
              save_dir: Union[str, None] = None,
              tensorboard_dir: Union[str, None] = None,
              control_zmq_port: int = 9000,
              progress_report_zmq_port: int = 9001,
              multiprocessing_workers: int = 0) -> str:
        """
        Train a given model using labels and the Trainer's current hyper-parameter settings.
        This method executes synchronously, thus it blocks until training is finished.

        Args:
            model: The model to run training on.
            labels: The SLEAP Labels dataset of labeled frames to run training on.
            tensorboard_dir: An optional tensorboard directory.
            run_name: A string name to use to prefix each model file name. If set to None,
            the default value is:
            f"{timestamp}.{str(model.output_type).lower()}.{model.name}.n={num_total}".
            Where num_total is the number of total training and validation images.
            control_zmq_port: This training process can be controlled by a control server.
            This is the port to connect to on the server at localhost. If None, no connection
            is made.
            progress_report_zmq_port: Progress indications can be generated to a server
            listening on localhost using this port. If None, no connection is made.
            multiprocessing_workers: Whether to user multiple worker processes to train the model.
            If set < 1, use_multiprocessing will be set to false on call to
            keras.Model.fit_generator(). Otherwise, multiprocessing_workers will
            be passed to fit_generator().

        Returns:
            If save_dir is not None, the file path of the JSON TrainingJob object. If save_dir
            is None, then the returns None.
        """

        labels_file_name = None
        if type(labels) is str:
            labels_file_name = labels
            labels = Labels.load_json(labels)


        # FIXME: We need to handle multiple skeletons.
        skeleton = labels.skeletons[0]

        # Modify the model to add the skeletons, not sure if skeletons should be
        # on the Model class or on TrainingJob instead. Oh well.
        model.skeletons = labels.skeletons

        # Generate images and points (outputs) datasets from the labels
        imgs = generate_images(labels, scale=self.scale)
        points = generate_points(labels, scale=self.scale)

        # Crop images to instances (if desired)
        if self.instance_crop:
            imgs, points = instance_crops(imgs, points)

        # Split data into train/validation
        imgs_train, imgs_val, outputs_train, outputs_val = \
            train_test_split(imgs, points, test_size=self.val_size)

        # Free up the original datasets after test and train split.
        del imgs, points

        # Infer shapes
        num_train, img_height, img_width, img_channels = imgs_train.shape
        num_val = len(imgs_val)
        num_total = num_train + num_val

        # Figure out how many output channels we will have.
        if type(outputs_train) == np.ndarray:
            num_outputs_channels = outputs_train.shape[-1]
        elif model.output_type == ModelOutputType.CONFIDENCE_MAP:
            num_outputs_channels = len(skeleton.nodes)
        elif model.output_type == ModelOutputType.PART_AFFINITY_FIELD:
            num_outputs_channels = len(skeleton.edges) * 2

        logger.info(f"Training set: {imgs_train.shape} -> {num_outputs_channels} channels")
        logger.info(f"Validation set: {imgs_val.shape} -> {num_outputs_channels} channels")

        # Input layer
        img_input = Input((img_height, img_width, img_channels))

        # Rectify image sizes not divisible by pooling factor
        if hasattr(model.backbone, 'depth'):
            depth = model.backbone.depth

            pool_factor = 2 ** depth
            if img_height % pool_factor != 0 or img_width % pool_factor != 0:
                logger.warning(
                    f"Image dimensions ({img_height}, {img_width}) are "
                    f"not divisible by the pooling factor ({pool_factor}).")
                # gap_height = (np.ceil(img_height / pool_factor) * pool_factor) - img_height
                # gap_width = (np.ceil(img_width / pool_factor) * pool_factor) - img_width

                # TODO: combination of ZeroPadding2D and Cropping2D
                # can't do non-symmetric padding, so pad first and then crop
                # left/top so coordinates aren't messed up
                # Solution: https://www.tensorflow.org/api_docs/python/tf/pad + Lambda layer + corresponding crop at the end?

        # Instantiate the backbone, this builds the Tensorflow graph
        x_outs = model.output(input_tesnor=img_input, num_output_channels=num_outputs_channels)

        # Create training model by combining the input layer and backbone graph.
        keras_model = keras.Model(inputs=img_input, outputs=x_outs)

        # Specify the optimizer.
        if self.optimizer.lower() == "adam":
            _optimizer = keras.optimizers.Adam(lr=self.learning_rate, amsgrad=self.amsgrad)
        elif self.optimizer.lower() == "rmsprop":
            _optimizer = keras.optimizers.RMSprop(lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer, value = {optimizer}!")

        # Compile the Keras model
        keras_model.compile(
            optimizer=_optimizer,
            loss="mean_squared_error",
        )
        logger.info("Params: {:,}".format(keras_model.count_params()))

        # Default to one loop through dataset per epoch
        if self.steps_per_epoch is None:
            steps_per_epoch = np.ceil(len(imgs_train) / self.batch_size).astype(int)
        else:
            steps_per_epoch = self.steps_per_epoch

        # TODO: Add support for multiple skeletons
        # Setup data generation
        img_shape = (imgs_train.shape[1], imgs_train.shape[2])
        if model.output_type == ModelOutputType.CONFIDENCE_MAP:
            def datagen_function(points):
                return generate_confmaps_from_points(points, skeleton, img_shape)
        elif model.output_type == ModelOutputType.PART_AFFINITY_FIELD:
            def datagen_function(points):
                return generate_pafs_from_points(points, skeleton, img_shape)
        else:
            datagen_function = None

        if datagen_function is not None:
            outputs_val = datagen_function(outputs_val)

        # Initialize data generator with augmentation
        train_datagen = Augmenter(
            imgs_train, points=outputs_train,
            datagen=datagen_function, output_names=keras_model.output_names,
            batch_size=self.batch_size, shuffle_initially=self.shuffle_initially,
            rotation=self.augment_rotation,
            scale=(self.augment_scale_min, self.augment_scale_max))

        train_run = TrainingJob(model=model, trainer=self,
                                save_dir=save_dir, run_name=run_name,
                                labels_filename=labels_file_name)

        # Setup saving
        save_path = None
        if save_dir is not None:
            # Generate run name
            if run_name is None:
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                train_run.run_name = f"{timestamp}.{str(model.output_type)}." \
                           f"{model.name}.n={num_total}"

            # Build save path
            save_path = os.path.join(save_dir, run_name)
            logger.info(f"Save path: {save_path}")

            # Check if it already exists
            if os.path.exists(save_path):
                logger.warning(f"Save path already exists. "
                               f"Previous run data may be overwritten!")

            # Create run folder
            os.makedirs(save_path, exist_ok=True)

        # Setup a list of necessary callbacks to invoke while training.
        callbacks = self._setup_callbacks(
            train_run, save_path, train_datagen,
            tensorboard_dir, control_zmq_port,
            progress_report_zmq_port)

        # Train!
        history = keras_model.fit_generator(
            train_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.num_epochs,
            validation_data=(imgs_val, {output_name: outputs_val for output_name in keras_model.output_names}),
            callbacks=callbacks,
            verbose=2,
            use_multiprocessing=multiprocessing_workers > 0,
            workers=multiprocessing_workers,
        )

        # Save once done training
        if save_path is not None:
            final_model_path = os.path.join(save_path, "final_model.h5")
            keras_model.save(filepath=final_model_path, overwrite=True, include_optimizer=True)
            logger.info(f"Saved final model: {final_model_path}")

            # TODO: save training history

            train_run.final_model_filename = os.path.relpath(final_model_path, save_dir)
            TrainingJob.save_json(train_run, f"{save_path}.json")

            return f"{save_path}.json"
        else:
            return train_run

    def train_async(self, *args, **kwargs) -> Tuple[Pool, AsyncResult]:
        """
        Train a given model using labels and the Trainer's current hyper-parameter settings.
        This method executes asynchronously, that is, it launches training in a another
        process and returns immediately.

        Args:
            See Trainer.train().

        Returns:
            A tuple containing the multiprocessing.Process that is running training, start() has been called.
            And the AysncResult object that will contain the result when the job finishes.
        """

        # Use an pool because we want to use apply_async so we can get the return value of
        # train when things are done.
        pool = Pool(processes=1)
        result = pool.apply_async(self.train, args=args, kwds=kwargs)

        # Tell the pool to accept no new tasks
        pool.close()

        return pool, result

    def _setup_callbacks(self, train_run: 'TrainingJob',
                         save_path, train_datagen,
                         tensorboard_dir, control_zmq_port,
                         progress_report_zmq_port):
        """
        Setup callbacks for the call to Keras fit_generator.

        Returns:
            The list of callbacks
        """

        # Setup callbacks
        callbacks = []

        # Callbacks: Intermediate saving
        if save_path is not None:
            if self.save_every_epoch:
                full_path = os.path.join(save_path, "newest_model.h5")
                train_run.newest_model_filename = os.path.relpath(full_path, train_run.save_dir)
                callbacks.append(
                    ModelCheckpoint(filepath=full_path,
                                    monitor="val_loss", save_best_only=False,
                                    save_weights_only=False, period=1))
            if self.save_best_val:
                full_path = os.path.join(save_path, "best_model.h5")
                train_run.best_model_filename = os.path.relpath(full_path, train_run.save_dir)
                callbacks.append(
                    ModelCheckpoint(filepath=full_path,
                                    monitor="val_loss", save_best_only=True,
                                    save_weights_only=False, period=1))

        # Callbacks: Shuffle after every epoch
        if self.shuffle_every_epoch:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: train_datagen.shuffle()))

        # Callbacks: LR reduction
        callbacks.append(
            ReduceLROnPlateau(min_delta=self.reduce_lr_min_delta,
                              factor=self.reduce_lr_factor,
                              patience=self.reduce_lr_patience,
                              cooldown=self.reduce_lr_cooldown,
                              min_lr=self.reduce_lr_min_lr,
                              monitor="val_loss", mode="auto", verbose=1, )
        )

        # Callbacks: Early stopping
        callbacks.append(
            EarlyStopping(monitor="val_loss",
                          min_delta=self.early_stopping_min_delta,
                          patience=self.early_stopping_patience, verbose=1))

        # Callbacks: Tensorboard
        if tensorboard_dir is not None:
            callbacks.append(
                TensorBoard(log_dir=f"{tensorboard_dir}/{model.name}{time()}",
                            batch_size=32, update_freq=150, histogram_freq=0,
                            write_graph=False, write_grads=False, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None))

        # Callbacks: ZMQ control
        if control_zmq_port is not None:
            callbacks.append(
                TrainingControllerZMQ(address="tcp://127.0.0.1",
                                      port=control_zmq_port,
                                      topic="", poll_timeout=10))

        # Callbacks: ZMQ progress reporter
        if progress_report_zmq_port is not None:
            callbacks.append(
                ProgressReporterZMQ(port=progress_report_zmq_port))

        return callbacks


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
    model: Model
    trainer: Trainer
    labels_filename: Union[str, None] = None
    run_name: Union[str, None] = None
    save_dir: Union[str, None] = None
    best_model_filename: Union[str, None] = None
    newest_model_filename: Union[str, None] = None
    final_model_filename: Union[str, None] = None

    @staticmethod
    def save_json(training_job: 'TrainingJob', filename: str):
        """
        Save a training run to a JSON file.

        Args:
            training_job: The TrainingJob instance to save.
            filename: The filename to save the JSON to.

        Returns:
            None
        """

        with open(filename, 'w') as file:

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

        # Open and parse the JSON in filename
        with open(filename, 'r') as file:
            json_str = file.read()
            dicts = json.loads(json_str)

            # We have some skeletons to deal with, make sure to setup a Skeleton cattr.
            my_cattr = Skeleton.make_cattr()

            run = my_cattr.structure(dicts, cls)

        return run


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
        logger.info(f"Training controller subscribed to: {self.address} (topic: {self.topic})")

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
        logger.info(f"Progress reporter publishing on: {self.address}")

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


def main():
    from PySide2 import QtWidgets

    from sleap.nn.architectures.unet import UNet
    model = Model(output_type=ModelOutputType.CONFIDENCE_MAP,
                  backbone=UNet(num_filters=16))

    # Setup a Trainer object to train the model above
    trainer = Trainer(val_size=0.1, batch_size=4,
                      num_epochs=1, steps_per_epoch=5,
                      save_best_val=True,
                      save_every_epoch=True)

    # Run training asynchronously
    pool, result = trainer.train_async(model=model,
                                  labels="tests/data/json_format_v1/centered_pair.json",
                                  save_dir='test_train/',
                                  run_name="training_run_1")

    ctx = zmq.Context()

    app = QtWidgets.QApplication()
    loss_viewer = LossViewer(zmq_context=ctx)
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
    while True:
        msg = poll()
        if msg is not None:
            logger.info(msg)
            if msg["event"] == "train_begin":
                loss_viewer.set_start_time(time())
            elif msg["event"] == "epoch_begin":
                epoch = msg["epoch"]
            elif msg["event"] == "batch_end":
                loss_viewer.add_datapoint((epoch * 100) + msg["logs"]["batch"], msg["logs"]["loss"])
            elif msg["event"] == "train_end":
                break

        loss_viewer.update_runtime()
        app.processEvents()

    print("Get")
    train_job_path = result.get()

    # Stop training
    ctrl.send_string(jsonpickle.encode(dict(command="stop")))

    app.closeAllWindows()

    # Now lets load the training job we just ran
    train_job = TrainingJob.load_json(train_job_path)

    assert os.path.exists(os.path.join(train_job.save_dir, train_job.newest_model_filename))
    assert os.path.exists(os.path.join(train_job.save_dir, train_job.best_model_filename))
    assert os.path.exists(os.path.join(train_job.save_dir, train_job.final_model_filename))

    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()


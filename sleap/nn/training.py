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
    verbosity: int = 2

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
            test = Labels.load_file(test)
        if test is not None:
            test = data.TrainingData.from_labels(test)
        if test is None:
            test = data_test
        if test is not None and isinstance(test, str):
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
                or self.training_job.trainer.bounding_box_size == 0
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
        print("model.output_scale:", self.training_job.model.output_scale)
        print("training_job.input_scale:", self.training_job.input_scale)
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

        # Set up shuffling, batching, repeating and prefetching.
        ds_train = (
            ds_train.shuffle(len(train.images))
            .repeat(-1)
            .batch(self.training_job.trainer.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        ds_val = (
            ds_val.shuffle(len(val.images))
            .repeat(-1)
            .batch(self.training_job.trainer.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        if ds_test is not None:
            ds_test = ds_val.batch(self.training_job.trainer.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            )

        # Get image shape after all the dataset transformations are applied.
        img_shape = list(ds_val.take(1))[0][0][0].shape
        # cm_shape = list(ds_val.take(1))[0][1][0].shape
        print("img_shape:", img_shape)
        # print("cm_shape:", cm_shape)

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

        print(f"Model: {keras_model.name}")
        print(f"  Input: {keras_model.input_shape}")
        print(f"  Output: {keras_model.output_shape}")
        print(f"  Layers: {len(keras_model.layers)}")
        print(f"  Params: {keras_model.count_params():3,}")

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
            epochs=self.training_job.trainer.num_epochs,
            callbacks=training_callbacks,
            validation_data=ds_val,
            steps_per_epoch=self.training_job.trainer.steps_per_epoch,
            validation_steps=self.training_job.trainer.val_steps_per_epoch,
            verbose=self.verbosity,
        )

        # TODO: Save training history
        # TODO: Evaluate final test set performance if available

        return keras_model

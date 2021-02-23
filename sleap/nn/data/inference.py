"""Transformers for performing inference."""

import tensorflow as tf
import numpy as np
import attr
from typing import List, Text, Optional, Tuple
from sleap.nn.data.utils import expand_to_rank, ensure_list
from sleap.nn.system import best_logical_device_name
from sleap.nn.peak_finding import (
    find_local_peaks,
    find_global_peaks,
    find_local_peaks_integral,
    find_global_peaks_integral,
)


@attr.s(auto_attribs=True)
class KerasModelPredictor:
    """Transformer for performing tf.keras model inference."""

    keras_model: tf.keras.Model
    model_input_keys: Text = attr.ib(default="instance_image", converter=ensure_list)
    model_output_keys: Text = attr.ib(
        default="predicted_instance_confidence_maps", converter=ensure_list
    )
    device_name: Optional[Text] = None

    @property
    def input_keys(self) -> List[Text]:
        return self.model_input_keys

    @property
    def output_keys(self) -> List[Text]:
        return self.input_keys + self.model_output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        test_ex = next(iter(input_ds))
        input_shapes = [test_ex[k].shape for k in self.model_input_keys]
        input_layers = [tf.keras.layers.Input(shape) for shape in input_shapes]
        keras_model = tf.keras.Model(input_layers, self.keras_model(input_layers))

        device_name = self.device_name
        if device_name is None:
            device_name = best_logical_device_name()

        def predict(example):
            with tf.device(device_name):
                X = []
                for input_key in self.model_input_keys:
                    input_rank = tf.rank(example[input_key])
                    X.append(
                        expand_to_rank(example[input_key], target_rank=4, prepend=True)
                    )

                Y = keras_model(X)
                if not isinstance(Y, list):
                    Y = [Y]

                for output_key, y in zip(self.model_output_keys, Y):
                    if isinstance(y, list):
                        y = y[0]
                    if input_rank < tf.rank(y):
                        y = tf.squeeze(y, axis=0)
                    example[output_key] = y

                return example

        output_ds = input_ds.map(
            predict, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return output_ds


@attr.s(auto_attribs=True)
class GlobalPeakFinder:
    """Global peak finding transformer."""

    confmaps_key: Text = "predicted_instance_confidence_maps"
    confmaps_stride: int = 1
    peak_threshold: float = 0.2
    peaks_key: Text = "predicted_center_instance_points"
    peak_vals_key: Text = "predicted_center_instance_confidences"
    keep_confmaps: bool = True
    device_name: Optional[Text] = None
    integral: bool = True
    integral_patch_size: int = 5

    @property
    def input_keys(self) -> List[Text]:
        return [self.confmaps_key]

    @property
    def output_keys(self) -> List[Text]:
        output_keys = [self.peaks_key, self.peak_vals_key]
        if self.keep_confmaps:
            output_keys.append(self.confmaps_key)
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        device_name = self.device_name
        if device_name is None:
            device_name = best_logical_device_name()

        def find_peaks(example):
            with tf.device(device_name):
                confmaps = example[self.confmaps_key]
                confmaps = expand_to_rank(confmaps, target_rank=4, prepend=True)

                if self.integral:
                    # Find peaks via integral regression.
                    peaks, peak_vals = find_global_peaks_integral(
                        confmaps,
                        threshold=self.peak_threshold,
                        crop_size=self.integral_patch_size,
                    )
                    peaks *= tf.cast(self.confmaps_stride, tf.float32)

                else:
                    # Find peaks via standard grid aligned global argmax.
                    peaks, peak_vals = find_global_peaks(
                        confmaps, threshold=self.peak_threshold
                    )

                    peaks *= tf.cast(self.confmaps_stride, tf.float32)

                if tf.rank(example[self.confmaps_key]) == 3:
                    peaks = tf.squeeze(peaks, axis=0)
                    peak_vals = tf.squeeze(peak_vals, axis=0)

                example[self.peaks_key] = peaks
                example[self.peak_vals_key] = peak_vals

                if not self.keep_confmaps:
                    example.pop(self.confmaps_key)

                return example

        output_ds = input_ds.map(
            find_peaks, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return output_ds


@attr.s(auto_attribs=True)
class MockGlobalPeakFinder:
    """Transformer that mimics `GlobalPeakFinder` but passes ground truth data."""

    all_peaks_in_key: Text = "instances"
    peaks_out_key: Text = "predicted_center_instance_points"
    peak_vals_key: Text = "predicted_center_instance_confidences"
    keep_confmaps: bool = True
    confmaps_in_key: Text = "instance_confidence_maps"
    confmaps_out_key: Text = "predicted_instance_confidence_maps"

    @property
    def input_keys(self) -> List[Text]:
        input_keys = [self.all_peaks_in_key, "centroid", "bbox", "scale"]
        if self.keep_confmaps:
            input_keys.append(self.confmaps_in_key)
        return input_keys

    @property
    def output_keys(self) -> List[Text]:
        output_keys = [self.peaks_out_key, self.peak_vals_key]
        if self.keep_confmaps:
            output_keys.append(self.confmaps_out_key)
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        def find_peaks(example):
            # Match example centroid to the instance with the closest node.
            centroid = example["centroid"] / example["scale"]
            all_peaks = example[self.all_peaks_in_key]  # (n_instances, n_nodes, 2)
            dists = tf.reduce_min(
                tf.norm(all_peaks - tf.reshape(centroid, [1, 1, 2]), axis=-1),
                axis=1,
            )  # (n_instances,)
            instance_ind = tf.argmin(dists)
            center_instance = tf.gather(all_peaks, instance_ind)

            # Adjust to coordinates relative to bounding box.
            center_instance -= tf.reshape(tf.gather(example["bbox"], [1, 0]), [1, 2])

            # Fill in mock data.
            example[self.peaks_out_key] = center_instance
            example[self.peak_vals_key] = tf.ones(
                [tf.shape(center_instance)[0]], dtype=tf.float32
            )
            example.pop(self.all_peaks_in_key)

            if self.keep_confmaps:
                example[self.confmaps_out_key] = example[self.confmaps_in_key]
                example.pop(self.confmaps_in_key)

            return example

        output_ds = input_ds.map(
            find_peaks, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return output_ds


@attr.s(auto_attribs=True)
class LocalPeakFinder:
    """Local peak finding transformer."""

    confmaps_key: Text = "centroid_confidence_maps"
    confmaps_stride: int = 1
    peak_threshold: float = 0.2
    peaks_key: Text = "predicted_centroids"
    peak_vals_key: Text = "predicted_centroid_confidences"
    peak_sample_inds_key: Text = "predicted_centroid_sample_inds"
    peak_channel_inds_key: Text = "predicted_centroid_channel_inds"
    keep_confmaps: bool = True
    device_name: Optional[Text] = None
    integral: bool = True

    @property
    def input_keys(self) -> List[Text]:
        return [self.confmaps_key]

    @property
    def output_keys(self) -> List[Text]:
        output_keys = [
            self.peaks_key,
            self.peak_vals_key,
            self.peak_sample_inds_key,
            self.peak_channel_inds_key,
        ]
        if self.keep_confmaps:
            output_keys.append(self.confmaps_key)
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        device_name = self.device_name
        if device_name is None:
            device_name = best_logical_device_name()

        def find_peaks(example):
            with tf.device(device_name):
                confmaps = example[self.confmaps_key]
                confmaps = expand_to_rank(confmaps, target_rank=4, prepend=True)

                if self.integral:
                    # Find local peaks with local NMS + integral refinement.
                    (
                        peaks,
                        peak_vals,
                        peak_sample_inds,
                        peak_channel_inds,
                    ) = find_local_peaks_integral(
                        confmaps, threshold=self.peak_threshold
                    )

                else:
                    # Find local peaks with grid-aligned NMS.
                    (
                        peaks,
                        peak_vals,
                        peak_sample_inds,
                        peak_channel_inds,
                    ) = find_local_peaks(confmaps, threshold=self.peak_threshold)

                # Adjust for confidence map stride.
                peaks *= tf.cast(self.confmaps_stride, tf.float32)

                example[self.peaks_key] = peaks
                example[self.peak_vals_key] = peak_vals
                example[self.peak_sample_inds_key] = peak_sample_inds
                example[self.peak_channel_inds_key] = peak_channel_inds

                if not self.keep_confmaps:
                    example.pop(self.confmaps_key)

                return example

        output_ds = input_ds.map(
            find_peaks, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return output_ds


@attr.s(auto_attribs=True)
class PredictedCenterInstanceNormalizer:
    """Transformer for adjusting centered instance coordinates."""

    centroid_key: Text = "centroid"
    centroid_confidence_key: Text = "centroid_confidence"
    peaks_key: Text = "predicted_center_instance_points"
    peak_confidences_key: Text = "predicted_center_instance_confidences"

    new_centroid_key: Text = "predicted_centroid"
    new_centroid_confidence_key: Text = "predicted_centroid_confidence"
    new_peaks_key: Text = "predicted_instance"
    new_peak_confidences_key: Text = "predicted_instance_confidences"

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [
            self.centroid_key,
            self.centroid_confidence_key,
            self.peaks_key,
            self.peak_confidences_key,
            "scale",
            "bbox",
        ]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        output_keys = [
            self.new_centroid_key,
            self.new_centroid_confidence_key,
            self.new_peaks_key,
            self.new_peak_confidences_key,
        ]
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains instance cropped data."""

        def norm_instance(example):
            """Local processing function for dataset mapping."""
            centroids = example[self.centroid_key] / example["scale"]

            bboxes = example["bbox"]
            bboxes = expand_to_rank(bboxes, 2)
            bboxes_x1y1 = tf.gather(bboxes, [1, 0], axis=1)

            pts = example[self.peaks_key]
            pts += bboxes_x1y1
            pts /= example["scale"]

            example[self.new_centroid_key] = centroids
            example[self.new_centroid_confidence_key] = example[
                self.centroid_confidence_key
            ]
            example[self.new_peaks_key] = pts
            example[self.new_peak_confidences_key] = example[self.peak_confidences_key]
            return example

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            norm_instance, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds

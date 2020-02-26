"""Transformers for performing inference."""

import tensorflow as tf
import numpy as np
import attr
from typing import List, Text, Optional, Tuple
from sleap.nn.data.utils import expand_to_rank, ensure_list
from sleap.nn.system import best_logical_device_name


@attr.s(auto_attribs=True)
class KerasModelPredictor:
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

                Y = self.keras_model(X)
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


def find_global_peaks(img: tf.Tensor, threshold: float = 0.1) -> tf.Tensor:
    """Find the global maximum for each sample and channel.

    Args:
        img: Tensor of shape (samples, height, width, channels).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will be replaced with NaNs.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    # Find row maxima.
    max_img_rows = tf.reduce_max(img, axis=2)
    argmax_rows = tf.reshape(tf.argmax(max_img_rows, axis=1), [-1])

    # Find col maxima.
    max_img_cols = tf.reduce_max(img, axis=1)
    argmax_cols = tf.reshape(tf.argmax(max_img_cols, axis=1), [-1])

    # Construct sample and channel subscripts.
    channels = tf.cast(tf.shape(img)[-1], tf.int64)
    total_peaks = tf.cast(tf.shape(argmax_cols)[0], tf.int64)
    sample_subs = tf.range(total_peaks, dtype=tf.int64) // channels
    channel_subs = tf.range(total_peaks, dtype=tf.int64) % channels

    # Gather subscripts.
    peak_subs = tf.stack([sample_subs, argmax_rows, argmax_cols, channel_subs], axis=1)

    # Gather values at global maxima.
    peak_vals = tf.gather_nd(img, peak_subs)

    # Convert to points form (samples, channels, 2).
    peak_points = tf.transpose(
        tf.reshape(
            tf.cast(tf.stack([argmax_cols, argmax_rows], axis=-1), tf.float32),
            [channels, -1, 2],
        ),
        [1, 0, 2],
    )
    peak_vals = tf.transpose(tf.reshape(peak_vals, [channels, -1]), [1, 0])

    # Mask out low confidence points.
    peak_points = tf.where(
        tf.expand_dims(peak_vals, axis=-1) < threshold,
        x=tf.constant(np.nan, dtype=tf.float32),
        y=peak_points,
    )

    return peak_points, peak_vals


@attr.s(auto_attribs=True)
class GlobalPeakFinder:
    confmaps_key: Text = "predicted_instance_confidence_maps"
    confmaps_stride: int = 1
    peak_threshold: float = 0.2
    peaks_key: Text = "predicted_center_instance_points"
    peak_vals_key: Text = "predicted_center_instance_confidences"
    keep_confmaps: bool = True
    device_name: Optional[Text] = None

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


def find_local_peaks(
    img: tf.Tensor, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Find local maxima via non-maximum suppresion.

    Args:
        img: Tensor of shape (samples, height, width, channels).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will not be returned.

    Returns:
        A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).

        peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
        points.

        peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
        sample each peak belongs to.

        peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
        the channel each peak belongs to.
    """
    # Build custom local NMS kernel.
    kernel = tf.reshape(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32), (3, 3, 1)
    )

    # Reshape to have singleton channels.
    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    channels = tf.shape(img)[3]
    flat_img = tf.reshape(tf.transpose(img, [0, 3, 1, 2]), [-1, height, width, 1])

    # Perform dilation filtering to find local maxima per channel and reshape back.
    max_img = tf.nn.dilation2d(
        flat_img, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
    )
    max_img = tf.transpose(
        tf.reshape(max_img, [-1, channels, height, width]), [0, 2, 3, 1]
    )

    # Filter for maxima and threshold.
    argmax_and_thresh_img = (img > max_img) & (img > threshold)

    # Convert to subscripts.
    peak_subs = tf.where(argmax_and_thresh_img)

    # Get peak values.
    peak_vals = tf.gather_nd(img, peak_subs)

    # Convert to points format.
    peak_points = tf.cast(tf.gather(peak_subs, [2, 1], axis=1), tf.float32)

    # Pull out indexing vectors.
    peak_sample_inds = tf.gather(peak_subs, 0, axis=1)
    peak_channel_inds = tf.gather(peak_subs, 3, axis=1)

    return peak_points, peak_vals, peak_sample_inds, peak_channel_inds


@attr.s(auto_attribs=True)
class LocalPeakFinder:
    confmaps_key: Text = "centroid_confidence_maps"
    confmaps_stride: int = 1
    peak_threshold: float = 0.2
    peaks_key: Text = "predicted_centroids"
    peak_vals_key: Text = "predicted_centroid_confidences"
    peak_sample_inds_key: Text = "predicted_centroid_sample_inds"
    peak_channel_inds_key: Text = "predicted_centroid_channel_inds"
    keep_confmaps: bool = True
    device_name: Optional[Text] = None

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
                (
                    peaks,
                    peak_vals,
                    peak_sample_inds,
                    peak_channel_inds,
                ) = find_local_peaks(confmaps, threshold=self.peak_threshold)

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

    centroids_key: Text = "centroid"
    peaks_key: Text = "predicted_center_instance_points"

    new_centroid_key: Text = "predicted_centroid"
    new_peaks_key: Text = "predicted_instance"

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [self.peaks_key, self.centroids_key, "scale", "bbox"]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        output_keys = [self.new_centroid_key, self.new_peaks_key]
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains instance cropped data."""

        def norm_instance(example):
            """Local processing function for dataset mapping."""
            centroids = example[self.centroids_key] / example["scale"]

            bboxes = example["bbox"]
            bboxes = expand_to_rank(bboxes, 2)
            bboxes_x1y1 = tf.gather(bboxes, [1, 0], axis=1)

            pts = example[self.peaks_key]
            pts += bboxes_x1y1

            example[self.new_centroid_key] = centroids
            example[self.new_peaks_key] = pts
            return example

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            norm_instance, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds

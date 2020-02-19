"""Transformers for performing inference."""

import tensorflow as tf
import numpy as np
import attr
from typing import List, Text, Optional
from sleap.nn.data.utils import expand_to_rank, ensure_list


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
            gpus = tf.config.list_logical_devices("GPU")
            if len(gpus) > 0:
                device_name = gpus[0].name
            else:
                cpus = tf.config.list_logical_devices("CPU")
                device_name = cpus[0].name

        def predict(example):
            with tf.device(device_name):
                X = []
                input_ranks = []
                for input_key in self.model_input_keys:
                    input_ranks.append(tf.rank(example[input_key]))
                    X.append(
                        expand_to_rank(example[input_key], target_rank=4, prepend=True)
                    )

                    Y = self.keras_model(X)

                for output_key, y, input_rank in zip(
                    self.model_output_keys, Y, input_ranks
                ):
                    if input_rank < tf.rank(y):
                        y = tf.squeeze(y, axis=0)
                    example[output_key] = y

                return example

        output_ds = input_ds.map(predict)
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
        def find_peaks(example):
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

        output_ds = input_ds.map(find_peaks)
        return output_ds

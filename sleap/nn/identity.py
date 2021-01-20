"""Utilities for models that learn identity."""

import tensorflow as tf
import numpy as np
from typing import Tuple
from sleap.nn.utils import tf_linear_sum_assignment


def group_class_peaks(peak_class_probs: tf.Tensor, peak_sample_inds: tf.Tensor, peak_channel_inds: tf.Tensor, n_samples: int, n_channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    peak_inds = tf.TensorArray(tf.int32, size=n_samples * n_channels, infer_shape=False, element_shape=[None])
    class_inds = tf.TensorArray(tf.int32, size=n_samples * n_channels, infer_shape=False, element_shape=[None])

    for sample in range(n_samples):
        for channel in range(n_channels):
            is_sample_channel = (peak_sample_inds == sample) & (peak_channel_inds == channel)

            probs = peak_class_probs[is_sample_channel]

            peak_inds_sc, class_inds_sc = tf_linear_sum_assignment(probs)

            peak_inds_sc = tf.gather(tf.where(is_sample_channel), peak_inds_sc)
            

            i = sample * n_channels + channel
            peak_inds = peak_inds.write(i, tf.cast(tf.reshape(peak_inds_sc, [-1]), tf.int32))
            class_inds = class_inds.write(i, tf.cast(tf.reshape(class_inds_sc, [-1]), tf.int32))

    peak_inds = peak_inds.concat()
    class_inds = class_inds.concat()
    
    return peak_inds, class_inds


def classify_peaks(class_maps: tf.Tensor, peak_points: tf.Tensor, peak_vals: tf.Tensor, peak_sample_inds: tf.Tensor, peak_channel_inds: tf.Tensor, n_channels: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    n_samples = tf.shape(class_maps)[0]
    n_instances = tf.shape(class_maps)[3]

    subs = tf.concat([
        tf.reshape(peak_sample_inds, [-1, 1]),
        tf.cast(tf.round(tf.reverse(peak_points, [1])), tf.int32),
    ], axis=1)
    peak_class_probs = tf.gather_nd(class_maps, subs)

    peak_inds, class_inds = group_class_peaks(peak_class_probs, peak_sample_inds, peak_channel_inds, n_samples, n_channels)

    subs = tf.stack([tf.gather(peak_sample_inds, peak_inds), class_inds, tf.gather(peak_channel_inds, peak_inds),], axis=1)
    points = tf.tensor_scatter_nd_update(tf.fill([n_samples, n_instances, n_channels, 2], np.nan), subs, tf.gather(peak_points, peak_inds))
    point_vals = tf.tensor_scatter_nd_update(tf.fill([n_samples, n_instances, n_channels], np.nan), subs, tf.gather(peak_vals, peak_inds))
    class_probs = tf.tensor_scatter_nd_update(tf.fill([n_samples, n_instances, n_channels], np.nan), subs, tf.gather_nd(peak_class_probs, tf.stack([peak_inds, class_inds], axis=1)))

    return points, point_vals, class_probs

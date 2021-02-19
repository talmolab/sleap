"""Utilities for models that learn identity.

These functions implement the inference logic for classifying peaks using class maps or
classification vectors.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple
from sleap.nn.utils import tf_linear_sum_assignment


def group_class_peaks(
    peak_class_probs: tf.Tensor,
    peak_sample_inds: tf.Tensor,
    peak_channel_inds: tf.Tensor,
    n_samples: int,
    n_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Group local peaks using class probabilities.

    This is useful for matching peaks that span multiple samples and channels into
    classes (e.g., instance identities) by their class probability.

    Args:
        peak_class_probs: Class probabilities for each peak as `tf.Tensor` of dtype
            `tf.float32` and shape `(n_peaks, n_classes)`.
        peak_sample_inds: Sample index for each peak as `tf.Tensor` of dtype `tf.int32`
            and shape `(n_peaks,)`.
        peak_channel_inds: Channel index for each peak as `tf.Tensor` of dtype
            `tf.int32` and shape `(n_peaks,)`.
        n_samples: Integer number of samples in the batch.
        n_channels: Integer number of channels (nodes) the instances should have.

    Returns:
        A tuple of `(peak_inds, class_inds)`.

        `peak_inds`: Indices of class-grouped peaks within `[0, n_peaks)`. Will be at
            most `n_classes` long.

        `class_inds`: Indices of the corresponding class for each peak within
            `[0, n_peaks)`. Will be at most `n_classes` long.

    Notes:
        Peaks will be assigned to classes by their probability using the Hungarian
        algorithm. Peaks that are assigned to classes that are not the highest
        probability for each class are removed from the matches.

    See also: classify_peaks_from_maps, classify_peaks_from_vectors
    """
    peak_sample_inds = tf.cast(peak_sample_inds, tf.int32)
    peak_channel_inds = tf.cast(peak_channel_inds, tf.int32)
    peak_inds = tf.TensorArray(
        tf.int32, size=n_samples * n_channels, infer_shape=False, element_shape=[None]
    )
    class_inds = tf.TensorArray(
        tf.int32, size=n_samples * n_channels, infer_shape=False, element_shape=[None]
    )

    # Match peaks samples and channels-wise.
    for sample in range(n_samples):
        for channel in range(n_channels):
            # Get relevant peaks.
            is_sample_channel = (peak_sample_inds == sample) & (
                peak_channel_inds == channel
            )
            probs = tf.boolean_mask(peak_class_probs, is_sample_channel)

            # Assign group peaks by probability.
            peak_inds_sc, class_inds_sc = tf_linear_sum_assignment(-probs)

            # Adjust indices and save.
            peak_inds_sc = tf.gather(tf.where(is_sample_channel), peak_inds_sc)
            i = sample * n_channels + channel
            peak_inds = peak_inds.write(
                i, tf.cast(tf.reshape(peak_inds_sc, [-1]), tf.int32)
            )
            class_inds = class_inds.write(
                i, tf.cast(tf.reshape(class_inds_sc, [-1]), tf.int32)
            )

    peak_inds = peak_inds.concat()
    class_inds = class_inds.concat()

    # Keep only the matches that are the most likely class for each peak.
    matched_probs = tf.gather_nd(
        peak_class_probs, tf.stack([peak_inds, class_inds], axis=1)
    )
    best_probs = tf.reduce_max(tf.gather(peak_class_probs, peak_inds), axis=1)
    is_best_match = matched_probs == best_probs
    peak_inds = peak_inds[is_best_match]
    class_inds = class_inds[is_best_match]

    return peak_inds, class_inds


def classify_peaks_from_maps(
    class_maps: tf.Tensor,
    peak_points: tf.Tensor,
    peak_vals: tf.Tensor,
    peak_sample_inds: tf.Tensor,
    peak_channel_inds: tf.Tensor,
    n_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Classify and group local peaks by their class map probability.

    Args:
        class_maps: Class maps for a batch as a `tf.Tensor` of dtype `tf.float32` and
            shape `(n_samples, height, width, n_classes)`.
        peak_points: Local peak coordinates as a `tf.Tensor` of dtype `tf.float32` and
            shape `(n_peaks,)`. These should be in the same scale as the class maps.
        peak_vals: Confidence map value each peak as a `tf.Tensor` of dtype `tf.float32`
            and shape `(n_peaks,)`.
        peak_sample_inds: Sample index for each peak as a `tf.Tensor` of dtype `tf.int32`
            and shape `(n_peaks,)`.
        peak_channel_inds: Channel index for each peak as a `tf.Tensor` of dtype
            `tf.int32` and shape `(n_peaks,)`.
        n_channels: Integer number of channels (nodes) the instances should have.

    Returns:
        A tuple of `(points, point_vals, class_probs)` containing the grouped peaks.

        `points`: Class-grouped peaks as a `tf.Tensor` of dtype `tf.float32` and shape
            `(n_samples, n_classes, n_channels, 2)`. Missing points will be denoted by
            NaNs.

        `point_vals`: The confidence map values for each point as a `tf.Tensor` of dtype
            `tf.float32` and shape `(n_samples, n_classes, n_channels)`.

        `class_probs`: Classification probabilities for matched points as a `tf.Tensor`
            of dtype `tf.float32` and shape `(n_samples, n_classes, n_channels)`.

    See also: group_class_peaks
    """
    # Build subscripts and pull out class probabilities for each peak from class maps.
    n_samples = tf.shape(class_maps)[0]
    n_instances = tf.shape(class_maps)[3]
    peak_sample_inds = tf.cast(peak_sample_inds, tf.int32)
    peak_channel_inds = tf.cast(peak_channel_inds, tf.int32)
    subs = tf.concat(
        [
            tf.reshape(peak_sample_inds, [-1, 1]),
            tf.cast(tf.round(tf.reverse(peak_points, [1])), tf.int32),
        ],
        axis=1,
    )
    peak_class_probs = tf.gather_nd(class_maps, subs)

    # Classify the peaks.
    peak_inds, class_inds = group_class_peaks(
        peak_class_probs, peak_sample_inds, peak_channel_inds, n_samples, n_channels
    )

    # Assign the results to fixed size tensors.
    subs = tf.stack(
        [
            tf.gather(peak_sample_inds, peak_inds),
            class_inds,
            tf.gather(peak_channel_inds, peak_inds),
        ],
        axis=1,
    )
    points = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances, n_channels, 2], np.nan),
        subs,
        tf.gather(peak_points, peak_inds),
    )
    point_vals = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances, n_channels], np.nan),
        subs,
        tf.gather(peak_vals, peak_inds),
    )
    class_probs = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances, n_channels], np.nan),
        subs,
        tf.gather_nd(peak_class_probs, tf.stack([peak_inds, class_inds], axis=1)),
    )

    return points, point_vals, class_probs


def classify_peaks_from_vectors(
    peak_points: tf.Tensor,
    peak_vals: tf.Tensor,
    peak_class_probs: tf.Tensor,
    crop_sample_inds: tf.Tensor,
    n_samples: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Group peaks by classification probabilities.

    This is used in top-down classification models.

    Args:
        peak_points:
        peak_vals:
        peak_class_probs:
        crop_sample_inds:
        n_samples: Number of samples in the batch.

    Returns:
        A tuple of `(points, point_vals, class_probs)`.

        `points`: Class-grouped peaks as a `tf.Tensor` of dtype `tf.float32` and shape
            `(n_samples, n_classes, n_channels, 2)`. Missing points will be denoted by
            NaNs.

        `point_vals`: The confidence map values for each point as a `tf.Tensor` of dtype
            `tf.float32` and shape `(n_samples, n_classes, n_channels)`.

        `class_probs`: Classification probabilities for matched points as a `tf.Tensor`
            of dtype `tf.float32` and shape `(n_samples, n_classes, n_channels)`.
    """
    crop_sample_inds = tf.cast(crop_sample_inds, tf.int32)
    n_samples = tf.cast(n_samples, tf.int32)
    n_channels = tf.shape(peak_points)[1]
    n_instances = tf.shape(peak_class_probs)[1]

    peak_inds, class_inds = group_class_peaks(
        peak_class_probs,
        crop_sample_inds,
        tf.zeros_like(crop_sample_inds),
        n_samples,
        1,
    )

    # Assign the results to fixed size tensors.
    subs = tf.stack(
        [
            tf.gather(crop_sample_inds, peak_inds),
            class_inds,
        ],
        axis=1,
    )
    points = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances, n_channels, 2], np.nan),
        subs,
        tf.gather(peak_points, peak_inds),
    )
    point_vals = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances, n_channels], np.nan),
        subs,
        tf.gather(peak_vals, peak_inds),
    )
    class_probs = tf.tensor_scatter_nd_update(
        tf.fill([n_samples, n_instances], np.nan),
        subs,
        tf.gather_nd(peak_class_probs, tf.stack([peak_inds, class_inds], axis=1)),
    )

    return points, point_vals, class_probs

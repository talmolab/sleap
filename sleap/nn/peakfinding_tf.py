import os
import time

import h5py
import keras
import tensorflow as tf

import numpy as np

from typing import Generator, Tuple

from sleap.nn.util import batch


def find_maxima_tf(x):

    col_max = tf.reduce_max(x, axis=1)
    row_max = tf.reduce_max(x, axis=2)

    cols = tf.cast(tf.argmax(col_max, 1), tf.float32)
    rows = tf.cast(tf.argmax(row_max, 1), tf.float32)
    cols = tf.reshape(cols, (-1, 1))
    rows = tf.reshape(rows, (-1, 1))

    maxima = tf.concat([rows, cols], -1)
    # max_val = tf.reduce_max(col_max, axis=1) # should match tf.reduce_max(x, axis=[1,2])

    return maxima  # , max_val


def impeaksnms_tf(I, min_thresh=0.3):

    # Apply the minimum threshold, that is, all values
    # less than min_thresh are set to 0.
    It = tf.cast(I > min_thresh, I.dtype) * I

    kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])[..., None]
    # kernel = np.array([[1, 1, 1],
    #                    [1, 0, 1],
    #                    [1, 1, 1]])[..., None]
    #     m = tf.nn.dilation2d(I, kernel, [1,1,1,1], "SAME", "NCHW", [1,1,1,1]) # TF 2.0
    m = tf.nn.dilation2d(It, kernel, [1, 1, 1, 1], [1, 1, 1, 1], "SAME")  # TF 1.13
    inds = tf.where(tf.greater(It, m))
    peak_vals = tf.gather_nd(I, inds)

    return inds, peak_vals


def find_peaks_tf(
    confmaps,
    confmaps_shape,
    min_thresh=0.3,
    upsample_factor: int = 1,
    win_size: int = 5,
):
    # n, h, w, c = confmaps.get_shape().as_list()

    h, w, c = confmaps_shape

    unrolled_confmaps = tf.reshape(
        tf.transpose(confmaps, perm=[0, 3, 1, 2]), [-1, h, w, 1]
    )  # (nc, h, w, 1)
    peak_inds, peak_vals = impeaksnms_tf(unrolled_confmaps, min_thresh=min_thresh)

    channel_sample_ind, y, x, _ = tf.split(peak_inds, 4, axis=1)

    channel_ind = tf.floormod(channel_sample_ind, c)
    sample_ind = tf.floordiv(channel_sample_ind, c)

    peaks = tf.concat([sample_ind, y, x, channel_ind], axis=1)  # (nc, 4)

    # If we have run prediction on low res and need to upsample the peaks
    # to a higher resolution. Compute sub-pixel accurate peaks
    # from these approximate peaks and return the upsampled sub-pixel peaks.
    if upsample_factor > 1:

        offset = (win_size - 1) / 2

        # Get the boxes coordinates centered on the peaks, normalized to image
        # coordinates
        box_ind = tf.squeeze(tf.cast(channel_sample_ind, tf.int32))
        top_left = (
            tf.cast(peaks[:, 1:3], tf.float32)
            + tf.constant([-offset, -offset], dtype="float32")
        ) / (h - 1.0)
        bottom_right = (
            tf.cast(peaks[:, 1:3], tf.float32)
            + tf.constant([offset, offset], dtype="float32")
        ) / (w - 1.0)
        boxes = tf.concat([top_left, bottom_right], axis=1)

        small_windows = tf.image.crop_and_resize(
            unrolled_confmaps, boxes, box_ind, crop_size=[win_size, win_size]
        )

        # Upsample cropped windows
        windows = tf.image.resize_bicubic(
            small_windows, [upsample_factor * win_size, upsample_factor * win_size]
        )

        windows = tf.squeeze(windows)

        # Find global maximum of each window
        windows_peaks = find_maxima_tf(windows)  # [row_ind, col_ind] ==> (nc, 2)

        # Adjust back to resolution before upsampling
        windows_peaks = tf.cast(windows_peaks, tf.float32) / tf.cast(
            upsample_factor, tf.float32
        )

        # Convert to offsets relative to the original peaks (center of cropped windows)
        windows_offsets = windows_peaks - tf.cast(offset, tf.float32)  # (nc, 2)
        windows_offsets = tf.pad(
            windows_offsets, [[0, 0], [1, 1]], mode="CONSTANT", constant_values=0
        )  # (nc, 4)

        # Apply offsets
        peaks = tf.cast(peaks, tf.float32) + windows_offsets

    return peaks, peak_vals


# Blurring:
# Ref: https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum("i,j->ij", vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def peak_tf_inference(
    model,
    data,
    confmaps_shape: Tuple[int],
    min_thresh: float = 0.3,
    gaussian_size: int = 9,
    gaussian_sigma: float = 3.0,
    upsample_factor: int = 1,
    return_confmaps: bool = False,
    batch_size: int = 4,
    win_size: int = 7,
):

    sess = keras.backend.get_session()

    # TODO: Unfuck this.
    confmaps = model.outputs[-1]
    h, w, c = confmaps_shape

    if gaussian_size > 0 and gaussian_sigma > 0:

        # Make Gaussian Kernel with desired specs.
        gauss_kernel = gaussian_kernel(size=gaussian_size, mean=0.0, std=gaussian_sigma)

        # Expand dimensions of `gauss_kernel` for `tf.nn.separable_conv2d` signature.
        gauss_kernel = tf.tile(gauss_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, c, 1])

        # Create a pointwise filter that does nothing, we are using separable convultions to blur
        # each channel separately
        pointwise_filter = tf.eye(c, batch_shape=[1, 1])

        # Convolve.
        confmaps = tf.nn.separable_conv2d(
            confmaps,
            gauss_kernel,
            pointwise_filter,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

    # Setup peak finding computations.
    peaks, peak_vals = find_peaks_tf(
        confmaps,
        confmaps_shape=confmaps_shape,
        min_thresh=min_thresh,
        upsample_factor=upsample_factor,
        win_size=win_size,
    )

    # We definitely want to capture the peaks in the output
    # We will map the tensorflow outputs onto a dict to return
    outputs_dict = dict(peaks=peaks, peak_vals=peak_vals)

    if return_confmaps:
        outputs_dict["confmaps"] = confmaps

    # Convert dict to list of keys and list of tensors (to evaluate)
    outputs_keys, output_tensors = (
        list(outputs_dict.keys()),
        list(outputs_dict.values()),
    )

    # Run the graph and retrieve output arrays.
    peaks_arr = []
    peak_vals_arr = []
    confmaps_arr = []
    for batch_number, row_offset, data_batch in batch(data, batch_size=batch_size):

        # This does the actual evaluation
        outputs_arr = sess.run(output_tensors, feed_dict={model.input: data_batch})

        # Convert list of results to dict using saved list of keys
        outputs_arr_dict = dict(zip(outputs_keys, outputs_arr))

        batch_peaks = outputs_arr_dict["peaks"]

        # First column should match row number in full data matrix,
        # so we add row offset of batch to row number in batch matrix.
        batch_peaks[:, 0] += row_offset

        peaks_arr.append(batch_peaks)
        peak_vals_arr.append(outputs_arr_dict["peak_vals"])

        if "confmaps" in outputs_dict:
            confmaps.append(outputs_arr_dict["confmaps"])

    peaks_arr = np.concatenate(peaks_arr, axis=0)
    peak_vals_arr = np.concatenate(peak_vals_arr, axis=0)
    confmaps_arr = np.concatenate(confmaps_arr, axis=0) if len(confmaps_arr) else None

    # Extract frame and node index columns
    sample_channel_ind = peaks_arr[:, [0, 3]]  # (nc, 2)

    # Extract X and Y columns
    peak_points = peaks_arr[:, [2, 1]].astype("float")  # [x, y]  ==> (nc, 2)

    # Use indices to convert matrices to lists of lists
    # (this matches the format of cpu-based peak-finding)
    peak_list, peak_val_list = split_matrices_by_double_index(
        sample_channel_ind,
        peak_points,
        peak_vals_arr,
        n_samples=len(data),
        n_channels=c,
    )

    return peak_list, peak_val_list, confmaps


def split_matrices_by_double_index(idxs, *data_list, n_samples=None, n_channels=None):
    """Convert data matrices to lists of lists expected by other functions."""

    # Return empty array if there are no idxs
    if len(idxs) == 0:
        return [], []

    # Determine the list length for major and minor indices
    if n_samples is None:
        n_samples = np.max(idxs[:, 0]) + 1

    if n_channels is None:
        n_channels = np.max(idxs[:, 1]) + 1

    # We can accept a variable number of data matrices
    data_matrix_count = len(data_list)

    # Empty list for results from each data matrix
    r = [[] for _ in range(data_matrix_count)]

    # Loop over major index (frame)
    for t in range(n_samples):

        # Empty list for this value of major index
        # for results from each data matrix
        major = [[] for _ in range(data_matrix_count)]

        # Loop over minor index (node)
        for c in range(n_channels):

            # Use idxs matrix to determine which rows
            # to retrieve from each data matrix
            mask = np.all((idxs == [t, c]), axis=1)

            # Get rows from each data matrix
            for data_matrix_idx, matrix in enumerate(data_list):
                major[data_matrix_idx].append(matrix[mask])

        # For each data matrix, append its data from this major index
        # to the appropriate list of results
        for data_matrix_idx in range(data_matrix_count):
            r[data_matrix_idx].append(major[data_matrix_idx])

    return r

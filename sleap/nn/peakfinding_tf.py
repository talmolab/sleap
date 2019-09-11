import os
import time

import h5py
import tensorflow as tf

keras = tf.keras
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

    return maxima #, max_val

def impeaksnms_tf(I, min_thresh=0.3):

    # Apply the minimum threshold, that is, all values
    # less than min_thresh are set to 0.
    It = tf.cast(I > min_thresh, I.dtype) * I

    kernel = np.array([[0, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]])[..., None]
    # kernel = np.array([[1, 1, 1],
    #                    [1, 0, 1],
    #                    [1, 1, 1]])[..., None]
    #     m = tf.nn.dilation2d(I, kernel, [1,1,1,1], "SAME", "NCHW", [1,1,1,1]) # TF 2.0
    m = tf.nn.dilation2d(It, kernel, [1, 1, 1, 1], [1, 1, 1, 1], "SAME")  # TF 1.13
    inds = tf.where(tf.greater(It, m))
    peak_vals = tf.gather_nd(I, inds)

    return inds, peak_vals


def find_peaks_tf(confmaps, min_thresh=0.3, upsample_factor: int = 1):
    n, h, w, c = confmaps.get_shape().as_list()

    unrolled_confmaps = tf.reshape(tf.transpose(confmaps, perm=[0, 3, 1, 2]), [-1, h, w, 1])  # nc, h, w, 1
    peak_inds, peak_vals = impeaksnms_tf(unrolled_confmaps, min_thresh=min_thresh)

    channel_sample, y, x, _ = tf.split(peak_inds, 4, axis=1)

    channel = tf.floormod(channel_sample, c)
    sample = tf.floordiv(channel_sample, c)

    peaks = tf.concat([sample, y, x, channel], axis=1)

    # If we have run prediction on low res and need to upsample the peaks
    # to a higher resolution. Compute sub-pixel accurate peaks
    # from these approximate peaks and return the upsampled sub-pixel peaks.
    if upsample_factor > 1:

        win_size = 5 # Must be odd
        offset = (win_size - 1) / 2

        # Get the boxes coordinates centered on the peaks, normalized to image
        # coordinates
        box_ind = tf.squeeze(tf.cast(channel_sample, tf.int32))
        top_left = (tf.to_float(peaks[:, 1:3]) +
                    tf.constant([-offset, -offset], dtype='float32')) / (h - 1.0)
        bottom_right = (tf.to_float(peaks[:, 1:3]) + tf.constant([offset, offset], dtype='float32')) / (w - 1.0)
        boxes = tf.concat([top_left, bottom_right], axis=1)

        small_windows = tf.image.crop_and_resize(
            unrolled_confmaps,
            boxes,
            box_ind,
            crop_size=[win_size, win_size])

        windows = tf.image.resize_bicubic(
            small_windows,
            [upsample_factor*win_size, upsample_factor*win_size])

        windows = tf.squeeze(windows)
        windows_peaks = find_maxima_tf(windows)
        windows_peaks = windows_peaks / win_size
    else:
        windows_peaks = None

    return peaks, peak_vals, windows_peaks

# Blurring:
# Ref: https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

# Now we can do peak finding on the GPU like this:
def peak_tf_inference(model, data,
                  min_thresh: float = 0.3,
                  gaussian_size: int = 9,
                  gaussian_sigma: float = 3.0,
                  upsample_factor: int = 1,
                  downsample_factor: int = 1,
                  return_confmaps: bool = False):

    sess = keras.backend.get_session()

    confmaps = model.outputs[-1]

    n, h, w, c = confmaps.get_shape().as_list()

    if gaussian_size and upsample_factor == 1:

        # Make Gaussian Kernel with desired specs.
        gauss_kernel = gaussian_kernel(size=gaussian_size, mean=0.0, std=gaussian_sigma)

        # Expand dimensions of `gauss_kernel` for `tf.nn.seprable_conv2d` signature.
        gauss_kernel = tf.tile(gauss_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, c, 1])

        # Create a pointwise filter that does nothing, we are using separable convultions to blur
        # each channel separately
        pointwise_filter = tf.eye(c, batch_shape=[1, 1])

        # Convolve.
        blurred_confmaps = tf.nn.separable_conv2d(confmaps, gauss_kernel, pointwise_filter,
                                       strides=[1, 1, 1, 1], padding='SAME')

        inds, peak_vals, windows = find_peaks_tf(blurred_confmaps, min_thresh=min_thresh,
                                   upsample_factor=upsample_factor)
    else:
        inds, peak_vals, windows = find_peaks_tf(confmaps, min_thresh=min_thresh,
                                   upsample_factor=upsample_factor)

    # We definitely want to capture the peaks in the output
    # We will map the tensorflow outputs onto a dict to return
    outputs_dict = dict(peaks=inds, peak_vals=peak_vals)

    if upsample_factor > 1:
        outputs_dict["windows"] = windows

    if return_confmaps:
        outputs_dict["confmaps"] = confmaps

    # Convert dict to list of keys and list of tensors (to evaluate)
    outputs_keys, outputs_vals = list(outputs_dict.keys()), list(outputs_dict.values())

    peaks = []
    peak_vals = []
    windows = []
    confmaps = []

    for batch_number, row_offset, data_batch in batch(data, batch_size=2):

        # This does the actual evaluation
        outputs = sess.run(outputs_vals, feed_dict={ model.input: data_batch })

        # Convert list of results to dict using saved list of keys
        outputs_dict = dict(zip(outputs_keys, outputs))

        batch_peaks = outputs_dict["peaks"]

        # First column should match row number in full data matrix,
        # so we add row offset of batch to row number in batch matrix.
        batch_peaks[:,0] += row_offset

        peaks.append(batch_peaks)
        peak_vals.append(outputs_dict["peak_vals"])

        if "windows" in outputs_dict:
            windows.append(outputs_dict["windows"])

        if "confmaps" in outputs_dict:
            confmaps.append(outputs_dict["confmaps"])

    peaks = np.concatenate(peaks)
    peak_vals = np.concatenate(peak_vals)
    confmaps = np.concatenate(confmaps) if len(confmaps) else None

    # Extract frame and node index columns
    frame_node_idx = peaks[:, [0, 3]]

    # Extract X and Y columns
    peak_points = peaks[:,[1,2]].astype("float")

    # Add offset from upsampling window peak if upsampling
    if upsample_factor > 1 and len(windows):
        windows = np.concatenate(windows)
        peak_points += windows/upsample_factor

    if downsample_factor > 1:
        peak_points /= downsample_factor

    # Swap the X and Y columns (order was [row idx, col idx])
    peak_points = peak_points[:,[1,0]]

    # Use indices to convert matrices to lists of lists
    # (this matches the format of cpu-based peak-finding)
    peak_list, peak_val_list = split_matrices_by_double_index(frame_node_idx, peak_points, peak_vals) 

    return peak_list, peak_val_list, confmaps

def split_matrices_by_double_index(idxs, *data_list):
    """Convert data matrices to lists of lists expected by other functions."""

    # Return empty array if there are no idxs
    if len(idxs) == 0: return [], []

    # Determine the list length for major and minor indices
    max_idx_vals = np.max(idxs, axis=0).astype("int") + 1

    # We can accept a variable number of data matrices
    data_matrix_count = len(data_list)

    # Empty list for results from each data matrix
    r = [[] for _ in range(data_matrix_count)]

    # Loop over major index (frame)
    for i in range(max_idx_vals[0]):

        # Empty list for this value of major index
        # for results from each data matrix
        major = [[] for _ in range(data_matrix_count)]

        # Loop over minor index (node)
        for j in range(max_idx_vals[1]):

            # Use idxs matrix to determine which rows
            # to retrieve from each data matrix
            mask = np.all((idxs == [i,j]), axis = 1)

            # Get rows from each data matrix
            for data_matrix_idx, matrix in enumerate(data_list):
                major[data_matrix_idx].append(matrix[mask])

        # For each data matrix, append its data from this major index
        # to the appropriate list of results
        for data_matrix_idx in range(data_matrix_count):
            r[data_matrix_idx].append(major[data_matrix_idx])

    return r
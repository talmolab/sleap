"""This module contains routines for peak finding."""

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List


def expand_to_4d(img: tf.Tensor) -> tf.Tensor:
    """Expands an image to rank 4 by adding singleton dimensions.
    
    Args:
        img: Image tensor with rank of 2, 3 or 4.
            If img is rank 2, it is assumed to have shape (height, width).
            if img is rank 3, it is assumed to have shape (height, width, channels).
        
    Returns:
        Rank 4 tensor of shape (samples, height, width, channels).
    """

    # Add singleton channel dimension.
    if tf.rank(img) == 2:
        img = tf.expand_dims(img, axis=-1)

    # Add singleton samples dimension.
    if tf.rank(img) == 3:
        img = tf.expand_dims(img, axis=0)

    return img


def ensure_odd(x: tf.Tensor) -> tf.Tensor:
    """Rounds numbers up to the nearest odd value."""

    return tf.floor(x * 0.5) * 2 + 1


def find_local_peaks(
    img: tf.Tensor, min_val: float = 0.3, return_vals: bool = False
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Finds local maxima via non-maximum suppresion.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
            Also supports rank 2 or 3 inputs, but will be expanded.
        min_val: Minimum threshold to consider a pixel a local maximum.
        return_vals: If True, returns the values at the maxima.
        
    Returns:
        A int64 tensor of shape (n_peaks, 4), where n_peaks is the number of local
        maxima detected. The location of the i-th peak is specified by its subscripts
        in img, e.g.:
            sample, row, col, channel = find_local_peaks(img)[i]
            
        If return_vals is True, also returns a tensor of shape (n_peaks,) with the
        values of img at the peaks.
    """

    # Ensure rank 4.
    img = expand_to_4d(img)

    # Call autographed function.
    return _find_local_peaks(img, min_val=min_val, return_vals=return_vals)


# @tf.function
def _find_local_peaks(
    img: tf.Tensor, min_val: float = 0.3, return_vals: bool = False
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Finds local maxima via non-maximum suppresion.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
        min_val: Minimum threshold to consider a pixel a local maximum.
        return_vals: If True, returns the values at the maxima.
        
    Returns:
        A int64 tensor of shape (n_peaks, 4), where n_peaks is the number of local
        maxima detected. The location of the i-th peak is specified by its subscripts
        in img, e.g.:
            sample, row, col, channel = _find_local_peaks(img)[i]
            
        If return_vals is True, also returns a tensor of shape (n_peaks,) with the
        values of img at the peaks.
    """

    # Store initial shape.
    samples, height, width, channels = tf.unstack(
        tf.cast(tf.shape(img), tf.int64), num=4
    )

    # Pack channels along samples axis to enforce a singleton channel.
    packed_img = tf.reshape(
        tf.transpose(img, (0, 3, 1, 2)), (samples * channels, height, width, 1)
    )

    # Create local maximum kernel.
    kernel = tf.reshape(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32), (3, 3, 1)
    )

    # Apply a dilation (max filter) with custom kernel.
    max_img = tf.nn.dilation2d(
        packed_img, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
    )

    # Find peaks by comparing to dilation and threshold.
    peaks_mask = tf.greater(packed_img, max_img) & tf.greater(packed_img, min_val)

    # Convert to subscripts where rows are [sample_channel, row, col, 0].
    packed_peak_subs = tf.where(peaks_mask)

    # Adjust coordinates to account for channel packing.
    sample_channel, row, col, _ = tf.unstack(packed_peak_subs, num=4, axis=1)
    sample = sample_channel // channels
    channel = sample_channel % channels
    peak_subs = tf.stack([sample, row, col, channel], axis=1)

    if return_vals:
        return peak_subs, tf.boolean_mask(packed_img, peaks_mask)
    else:
        return peak_subs


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None))])
def _find_global_peaks(img: tf.Tensor) -> tf.Tensor:
    """Finds the global maximum for each sample and channel.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
        
    Returns:
        A int64 tensor of shape (n_peaks, 2), where n_peaks is the number of global
        peaks, i.e.:
            n_peaks = samples * channels.
        
        The location of the i-th peak is specified by its row and column in img, e.g.:
            row, col = _find_global_peaks(img)[i]
    """

    # Store initial shape.
    samples, height, width, channels = tf.unstack(tf.shape(img), num=4)

    # Collapse height/width into a single dimension.
    flat_img = tf.reshape(img, (samples, -1, channels))

    # Find maximum indices within collapsed height/width axis (samples, 1, channels).
    inds_max = tf.argmax(flat_img, axis=1)
    inds_max = tf.reshape(inds_max, (-1,))  # (samples * channels)

    # Convert to subscripts (samples * channels, [row, col]).
    subs_max = tf.transpose(tf.unravel_index(inds_max, (height, width)))

    return subs_max


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None))])
def _find_global_peaks_with_vals(img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Finds the global maximum for each sample and channel.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
        
    Returns:
        A int64 tensor of shape (n_peaks, 2), where n_peaks is the number of global
        peaks, i.e.:
            n_peaks = samples * channels.
        
        The location of the i-th peak is specified by its row and column in img, e.g.:
            row, col = _find_global_peaks_with_vals(img)[i]
            
        Also returns a tensor of shape (n_peaks,) with the values of img at the peaks.
    """

    # Store initial shape.
    samples, height, width, channels = tf.unstack(tf.shape(img), num=4)

    # Collapse height/width into a single dimension.
    flat_img = tf.reshape(img, (samples, -1, channels))

    # Find maximum indices within collapsed height/width axis (samples, 1, channels).
    inds_max = tf.argmax(flat_img, axis=1)
    inds_max = tf.reshape(inds_max, (-1,))  # (samples * channels)

    # Convert to subscripts (samples * channels, [row, col]).
    subs_max = tf.transpose(tf.unravel_index(inds_max, (height, width)))

    # Compute values at maxima.
    vals_max = tf.reshape(tf.reduce_max(flat_img, axis=1), (-1,))

    return subs_max, vals_max


@tf.function
def find_global_peaks(
    img: tf.Tensor, return_vals: bool = False, return_all_subs: bool = True
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Finds the global maximum for each sample and channel.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
            Also supports rank 2 or 3 inputs, but will be expanded.
        return_vals: If True, returns the values at the maxima.
        return_all_subs: If True, returns full subscripts of each peak, similar
            to tf.where.
        
    Returns:
        A int64 tensor of shape (n_peaks, 2) or (n_peaks, 4), where n_peaks is the
        number of global peaks, i.e., n_peaks = samples * channels.
        If return_all_subs is True, the location of the i-th peak is specified by its
        full subscripts in img, e.g.:
            sample, row, col, channel = find_global_peaks(img, return_all_subs=True)[i]
        
        If return_all_subs is False, the location only specifies the row and column:
            row, col = find_global_peaks(img, return_all_subs=False)[i]
            
        If return_vals is True, also returns a tensor of shape (n_peaks,) with the
        values of img at the peaks.
    """

    # Ensure rank 4.
    img = expand_to_4d(img)

    # Find peaks with reduced subscripts (row, col).
    if return_vals:
        subs_max, vals_max = _find_global_peaks_with_vals(img)
    else:
        subs_max = _find_global_peaks(img)

    # Append the sample and channel subscripts to match tf.where notation.
    if return_all_subs:
        samples, height, width, channels = tf.unstack(
            tf.cast(tf.shape(img), tf.int64), num=4
        )

        inds = tf.range(0, samples * channels, dtype=tf.int64)
        sample_subs = inds // channels
        channel_subs = inds % channels

        # Each row specifies peak subscripts as [sample, row, col, channel].
        subs_max = tf.concat(
            [
                tf.expand_dims(sample_subs, axis=1),
                subs_max,
                tf.expand_dims(channel_subs, axis=1),
            ],
            axis=1,
        )

    if return_vals:
        return subs_max, vals_max
    else:
        return subs_max


@tf.function
def crop_centered_boxes(
    img: tf.Tensor, peaks: tf.Tensor, window_length: int = 5
) -> tf.Tensor:
    """Crops boxes centered around peaks.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
        peaks: Tensor of shape (n_peaks, 4) where subscripts of peak locations are
            specified in each row as [sample, row, col, channel].
        window_length: Size (width and height) of windows to be cropped. This parameter
            will be rounded up to nearest odd number.
        
    Returns:
        A tensor of shape (n_peaks, window_length, window_length, 1) corresponding to
        the box cropped around each peak.
    """

    # Compute window offset from odd window length.
    window_length = ensure_odd(window_length)
    crop_size = tf.cast((window_length, window_length), tf.int32)
    half_window = tf.cast(window_length // 2, tf.float32)

    # Store initial shape.
    samples, height, width, channels = tf.unstack(
        tf.cast(tf.shape(img), tf.float32), num=4
    )

    # Pack channels along samples axis to enforce a singleton channel.
    packed_img = tf.reshape(
        tf.transpose(img, (0, 3, 1, 2)), (samples * channels, height, width, 1)
    )

    # Pull out peak subscripts as vectors.
    sample, y, x, channel = tf.unstack(peaks, num=4, axis=1)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Compute packed sample_channel indices for each peak.
    box_indices = (tf.cast(sample, tf.int32) * tf.cast(channels, tf.int32)) + tf.cast(
        channel, tf.int32
    )

    # Define centered boxes with the form [y_min, x_min, y_max, x_max].
    boxes = tf.stack(
        [
            (y - half_window) / (height - 1),
            (x - half_window) / (width - 1),
            (y + half_window) / (height - 1),
            (x + half_window) / (width - 1),
        ],
        axis=1,
    )

    # Crop with padding.
    cropped_boxes = tf.image.crop_and_resize(
        packed_img, boxes, box_indices, crop_size, method="nearest"
    )

    return cropped_boxes


@tf.function
def _find_center_offsets(
    centered_boxes: tf.Tensor, upsampling_factor: float
) -> tf.Tensor:
    """Computes offsets of peaks in centered boxes after bicubic upsampling.
    
    Args:
        centered_boxes: Tensor of shape (n_peaks, height, width, 1). This is expected
            to be a set of cropped heatmaps with a single peak near the center.
        upsampling_factor: Scalar float specifying the relative scale of upsampling.
        
    Returns:
        Tensor of (n_peaks, 2) denoting the offsets computed for each centered box
        associated with a peak.
        
        For the i-th peak, its offsets are defined as:
            dy, dx = _find_center_offsets(centered_boxes, 10)[i]
            
        These offsets can be added to the original peak subscripts to obtain a more
        accurate estimate of their location.
    """

    # Store input shape.
    n_peaks, height, width, channels = tf.unstack(tf.shape(centered_boxes), num=4)
    #     assert(height == width)
    #     assert(channels == 1)

    # Compute shape after upsampling.
    upsampling_factor = tf.cast(upsampling_factor, tf.float32)
    input_length = tf.cast(height, tf.float32)
    new_length = ensure_odd(input_length * upsampling_factor)

    # Upsample boxes via bicubic interpolation.
    upsampled_boxes = tf.image.resize(
        centered_boxes, tf.cast((new_length, new_length), tf.int32), method="bicubic"
    )

    # Find global peaks within upsampled boxes.
    peaks = _find_global_peaks(upsampled_boxes)

    # Compute offsets at the input scale.
    offsets = ((tf.cast(peaks, tf.float32) / (new_length - 1)) - 0.5) * (
        input_length - 1
    )

    return offsets


# @tf.function
def refine_peaks(
    img: tf.Tensor,
    peaks: tf.Tensor,
    window_length: int = 5,
    upsampling_factor: float = 20,
) -> tf.Tensor:
    """Refines a set of detected peaks.
    
    This function is primarily a convenience wrapper around substeps of the peak
    refinement procedure.
    
    Args:
        img: Tensor of shape (samples, height, width, channels).
        peaks: Tensor of shape (n_peaks, 4) where subscripts of peak locations are
            specified in each row as [sample, row, col, channel].
        window_length: Size (width and height) of windows to be cropped. This parameter
            will be rounded up to nearest odd number.
        upsampling_factor: Scalar float specifying the relative scale of upsampling.
        
    Returns:
        A float32 tensor of shape (n_peaks, 4) with refined estimates for the peak
        locations. These will still be in the same format as tf.where, i.e., each row
        specifies peak locations as [sample, row, col, channel].
    """

    # Crop boxes around peaks.
    centered_boxes = crop_centered_boxes(img, peaks, window_length=window_length)

    # Compute offsets after upsampling and peak detection.
    offsets = _find_center_offsets(centered_boxes, upsampling_factor)

    # Apply offsets to peak detections.
    refined_peaks = tf.cast(peaks, tf.float32) + tf.pad(offsets, [[0, 0], [1, 1]])

    return refined_peaks


def peak_subs_to_list(
    peak_subs: tf.Tensor,
    peak_vals: tf.Tensor = None,
    samples: int = None,
    channels: int = None,
) -> Union[
    List[List[np.ndarray]], Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]
]:
    """Converts peak subscripts to lists of arrays.
    
    Args:
        peak_subs: Tensor of shape (n_peaks, 4) where subscripts of peak locations
            are specified in each row as [sample, row, col, channel].
        peak_vals: Tensor of shape (n_peaks,) where elements correspond to the value
            of the peaks.
        samples: Scalar specifying the maximum number of samples. Inferred from
            peak_subs if not specified.
        channels: Scalar specifying the maximum number of channels. Inferred from
            peak_subs if not specified.
        
    Returns:
        A list of lists of np.ndarrays or a tuple if peak_vals is also specified.
        
        The first-level list indicates samples, the second-level nested list indicates
        the channel each peak came from, and elements are float32 arrays of shape
        (n_peaks_i_k, 2) with rows indicating the [x, y] coordinates of the j peaks
        found in the i-th sample of the k-th channel, e.g.:
            x_ikj, y_ikj = peak_subs_to_list(peak_subs)[i][k][j, :]
            
        When the peak_vals are also specified, the second output follows the same
        structure, but with scalar elements indicating the value at the peak.
        
        If a specific sample/channel does not have any peaks, the list element will be
        an empty array (len == 0).
    """

    if samples is None:
        samples = int(tf.reduce_max(peak_subs[:, 0]).numpy()) + 1

    if channels is None:
        channels = int(tf.reduce_max(peak_subs[:, -1]).numpy()) + 1

    peaks_list = []
    if peak_vals is not None:
        peak_vals_list = []

    for i in range(samples):
        # Create empty sample-level list.
        peaks_list_i = []
        if peak_vals is not None:
            peak_vals_list_i = []

        for k in range(channels):
            # Create empty channel-level array.
            peaks_list_ik = np.zeros((0, 2), dtype="float32")
            if peak_vals is not None:
                peak_vals_list_ik = np.zeros((0,), dtype="float32")

            # Find peaks at this level.
            is_ik_peak = (peak_subs[:, 0] == i) & (peak_subs[:, -1] == k)

            # Convert and add to list.
            if tf.reduce_any(is_ik_peak):
                peaks_list_ik = (
                    peak_subs[is_ik_peak][:, 2:0:-1].numpy().astype("float32")
                )

                if peak_vals is not None:
                    peak_vals_list_ik = peak_vals[is_ik_peak]

            # Append to sample-level list.
            peaks_list_i.append(peaks_list_ik)
            if peak_vals is not None:
                peak_vals_list_i.append(peak_vals_list_ik)

        # Append to top-level list.
        peaks_list.append(peaks_list_i)
        if peak_vals is not None:
            peak_vals_list.append(peak_vals_list_i)

    if peak_vals is None:
        return peaks_list
    else:
        return peaks_list, peak_vals_list

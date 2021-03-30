"""This module contains TensorFlow-based peak finding methods.

In general, the inputs to the functions provided here operate on confidence maps
(sometimes referred to as heatmaps), which are image-based representations of the
locations of landmark coordinates.

In these representations, landmark locations are encoded as probability that it is
present each pixel. This is often represented by an unnormalized 2D Gaussian PDF
centered at the true location and evaluated over the entire image grid.

Peak finding entails finding either the global or local maxima of these confidence maps.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
from sleap.nn.data.instance_cropping import make_centered_bboxes, normalize_bboxes


def make_gaussian_kernel(size: int, sigma: float) -> tf.Tensor:
    """Generates a square unnormalized 2D symmetric Gaussian kernel.

    Args:
        size: Length of kernel. This should be an odd integer.
        sigma: Standard deviation of the Gaussian specified as a scalar float.

    Returns:
        kernel, a float32 tensor of shape (size, size) with values corresponding to the
        unnormalized probability density of a 2D Gaussian distribution with symmetric
        covariance along the x and y directions.

    Note:
        The maximum value of this kernel will be 1.0. To normalize it, divide each
        element by (2 * np.pi * sigma ** 2).
    """

    # Create 1D grid vector.
    gv = tf.range(-(size // 2), (size // 2) + 1, dtype=tf.float32)

    # Generate kernel and broadcast to 2D.
    kernel = tf.exp(
        -(tf.reshape(gv, (1, -1)) ** 2 + tf.reshape(gv, (-1, 1)) ** 2)
        / (2 * sigma ** 2)
    )

    return kernel


def smooth_imgs(imgs: tf.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> tf.Tensor:
    """Smooths the input image by convolving it with a Gaussian kernel.

    Args:
        imgs: A rank-4 tensor of shape (samples, height, width, channels).
        kernel_size: An odd-valued scalar integer specifying the width and height of
            the Gaussian kernel.
        sigma: Standard deviation of the Gaussian specified as a scalar float.

    Returns:
        A tensor of the same shape as imgs after convolving with a Gaussian sample and
        channelwise.
    """

    # Create kernel and broadcast to rank-4.
    kernel = tf.broadcast_to(
        tf.reshape(
            make_gaussian_kernel(kernel_size, sigma), (kernel_size, kernel_size, 1, 1)
        ),
        (kernel_size, kernel_size, tf.shape(imgs)[-1], 1),
    )

    # Normalize kernel weights to keep output in the same range as the input.
    kernel /= 2 * np.pi * sigma ** 2

    # Convolve with padding to keep the shape fixed.
    return tf.nn.depthwise_conv2d(imgs, kernel, strides=[1, 1, 1, 1], padding="SAME")


def find_offsets_local_direction(
    centered_patches: tf.Tensor, delta: float = 0.25
) -> tf.Tensor:
    """Computes subpixel offsets from the direction of the pixels around the peak.

    This function finds the delta-offset from the center pixel of peak-centered patches
    by finding the direction of the gradient around each center.

    Args:
        centered_patches: A rank-4 tensor of shape (samples, 3, 3, 1) corresponding
            to the centered crops around the grid-anchored peaks. For multi-channel
            images, stack the channels along the samples axis before calling this
            function.
        delta: Scalar float that will scaled by the gradient direction.

    Returns:
        offsets, a float32 tensor of shape (samples, 2) where the columns correspond to
        the offsets relative to the center pixel for the x and y directions
        respectively, i.e., for the i-th sample:

            dx_i, dy_i = offsets[i]

    Notes:
        For symmetric patches, the offset will be 0.

        This function can be used to refine peak coordinates by:
            1. Cropping 3 x 3 patches around each peak.
            2. Stacking patches along the samples axis.
            3. Computing the local gradient around each centered patch.
            4. Applying subpixel offsets to each peak.

        This is a commonly used algorithm for subpixel peak refinement, described for
        pose estimation applications in [1].

    Example: ::

        >>> find_offsets_local_direction(np.array(
        ...     [[0., 1., 0.],
        ...      [1., 3., 2.],
        ...      [0., 1., 0.]]).reshape(1, 3, 3, 1), 0.25)
        <tf.Tensor: shape=(1, 2), dtype=float64, numpy=array([[0.25, 0.  ]])>

    References:
        .. [1] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked Hourglass Networks
           for Human Pose Estimation. In _European conference on computer vision_, 2016.
    """

    # Compute directional gradients.
    dx = centered_patches[:, 1, 2, :] - centered_patches[:, 1, 0, :]  # right - left
    dy = centered_patches[:, 2, 1, :] - centered_patches[:, 0, 1, :]  # bottom - top

    # Concatenate and scale signed direction by delta.
    offsets = tf.sign(tf.squeeze(tf.stack([dx, dy], axis=1), axis=-1)) * delta

    return offsets


def crop_bboxes(
    images: tf.Tensor, bboxes: tf.Tensor, sample_inds: tf.Tensor
) -> tf.Tensor:
    """Crop bounding boxes from a batch of images.

    This method serves as a convenience method for specifying the arguments of
    `tf.image.crop_and_resize`.

    Args:
        images: Tensor of shape (samples, height, width, channels) of a batch of images.
        bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the last axis
            corresponds to unnormalized (y1, x1, y2, x2) coordinates of the bounding
            boxes. This can be generated from centroids using `make_centered_bboxes`.
        sample_inds: Tensor of shape (n_bboxes,) specifying which samples each bounding
            box should be cropped from.

    Returns:
        A tensor of shape (n_bboxes, crop_height, crop_width, channels) of the same
        dtype as the input image. The crop size is inferred from the bounding box
        coordinates.

    Notes:
        This function expects bounding boxes with coordinates at the centers of the
        pixels in the box limits. Technically, the box will span (x1 - 0.5, x2 + 0.5)
        and (y1 - 0.5, y2 + 0.5).

        For example, a 3x3 patch centered at (1, 1) would be specified by
        (y1, x1, y2, x2) = (0, 0, 2, 2). This would be exactly equivalent to indexing
        the image with `image[0:3, 0:3]`.

    See also: `make_centered_bboxes`
    """
    # Compute bounding box size to use for crops.
    y1x1 = tf.gather_nd(bboxes, [[0, 0], [0, 1]])
    y2x2 = tf.gather_nd(bboxes, [[0, 2], [0, 3]])
    box_size = tf.cast(tf.math.round((y2x2 - y1x1) + 1), tf.int32)  # (height, width)

    # Normalize bounding boxes.
    image_height = tf.shape(images)[1]
    image_width = tf.shape(images)[2]
    normalized_bboxes = normalize_bboxes(
        bboxes, image_height=image_height, image_width=image_width
    )

    # Crop.
    crops = tf.image.crop_and_resize(
        images,
        boxes=normalized_bboxes,
        box_indices=tf.cast(sample_inds, tf.int32),
        crop_size=box_size,
        method="bilinear",
    )

    # Cast back to original dtype and return.
    crops = tf.cast(crops, images.dtype)
    return crops


def find_global_peaks_rough(
    cms: tf.Tensor, threshold: float = 0.1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find the global maximum for each sample and channel.

    Args:
        cms: Tensor of shape (samples, height, width, channels).
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
    max_img_rows = tf.reduce_max(cms, axis=2)
    argmax_rows = tf.reshape(tf.argmax(max_img_rows, axis=1), [-1])

    # Find col maxima.
    max_img_cols = tf.reduce_max(cms, axis=1)
    argmax_cols = tf.reshape(tf.argmax(max_img_cols, axis=1), [-1])

    # Construct sample and channel subscripts.
    channels = tf.cast(tf.shape(cms)[-1], tf.int64)
    total_peaks = tf.cast(tf.shape(argmax_cols)[0], tf.int64)
    sample_subs = tf.range(total_peaks, dtype=tf.int64) // channels
    channel_subs = tf.range(total_peaks, dtype=tf.int64) % channels

    # Gather subscripts.
    peak_subs = tf.stack([sample_subs, argmax_rows, argmax_cols, channel_subs], axis=1)

    # Gather values at global maxima.
    peak_vals = tf.gather_nd(cms, peak_subs)

    # Convert to points form (samples, channels, 2).
    peak_points = tf.reshape(
        tf.cast(tf.stack([argmax_cols, argmax_rows], axis=-1), tf.float32),
        [-1, channels, 2],
    )
    peak_vals = tf.reshape(peak_vals, [-1, channels])

    # Mask out low confidence points.
    peak_points = tf.where(
        tf.expand_dims(peak_vals, axis=-1) < threshold,
        x=tf.constant(np.nan, dtype=tf.float32),
        y=peak_points,
    )

    return peak_points, peak_vals


def find_local_peaks_rough(
    cms: tf.Tensor, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Find local maxima via non-maximum suppresion.

    Args:
        cms: Tensor of shape (samples, height, width, channels).
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
    height = tf.shape(cms)[1]
    width = tf.shape(cms)[2]
    channels = tf.shape(cms)[3]
    flat_img = tf.reshape(tf.transpose(cms, [0, 3, 1, 2]), [-1, height, width, 1])

    # Perform dilation filtering to find local maxima per channel and reshape back.
    max_img = tf.nn.dilation2d(
        flat_img, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
    )
    max_img = tf.transpose(
        tf.reshape(max_img, [-1, channels, height, width]), [0, 2, 3, 1]
    )

    # Filter for maxima and threshold.
    argmax_and_thresh_img = (cms > max_img) & (cms > threshold)

    # Convert to subscripts.
    peak_subs = tf.where(argmax_and_thresh_img)

    # Get peak values.
    peak_vals = tf.gather_nd(cms, peak_subs)

    # Convert to points format.
    peak_points = tf.cast(tf.gather(peak_subs, [2, 1], axis=1), tf.float32)

    # Pull out indexing vectors.
    peak_sample_inds = tf.cast(tf.gather(peak_subs, 0, axis=1), tf.int32)
    peak_channel_inds = tf.cast(tf.gather(peak_subs, 3, axis=1), tf.int32)

    return peak_points, peak_vals, peak_sample_inds, peak_channel_inds


def integral_regression(
    cms: tf.Tensor, xv: tf.Tensor, yv: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute regression by integrating over the confidence maps on a grid.

    Args:
        cms: Confidence maps with shape (samples, height, width, channels).
        xv: X grid vector tf.float32 of grid coordinates to sample.
        yv: Y grid vector tf.float32 of grid coordinates to sample.

    Returns:
        A tuple of (x_hat, y_hat) with the regressed x- and y-coordinates for each
        channel of the confidence maps.

        x_hat and y_hat are of shape (samples, channels)
    """
    # Compute normalizing factor.
    z = tf.reduce_sum(cms, axis=[1, 2])

    # Regress to expectation.
    x_hat = tf.reduce_sum(tf.reshape(xv, [1, 1, -1, 1]) * cms, axis=[1, 2]) / z
    y_hat = tf.reduce_sum(tf.reshape(yv, [1, -1, 1, 1]) * cms, axis=[1, 2]) / z

    return x_hat, y_hat


def find_global_peaks(
    cms: tf.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find global peaks with optional refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, height, width, channels).
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset.
        integral_patch_size: Size of patches to crop around each rough peak as an
            integer scalar.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    # Find grid aligned peaks.
    rough_peaks, peak_vals = find_global_peaks_rough(
        cms, threshold=threshold
    )  # (samples, channels, 2)

    # Return early if not refining or no rough peaks found.
    if refinement is None or tf.reduce_all(tf.math.is_nan(rough_peaks)):
        return rough_peaks, peak_vals

    if refinement == "integral":
        crop_size = integral_patch_size
    elif refinement == "local":
        crop_size = 3
    else:
        return rough_peaks, peak_vals

    # Flatten samples and channels to (n_peaks, 2).
    samples = tf.shape(cms)[0]
    channels = tf.shape(cms)[3]
    rough_peaks = tf.reshape(rough_peaks, [samples * channels, 2])

    # Keep only peaks that are not NaNs.
    valid_idx = tf.squeeze(
        tf.where(~tf.math.is_nan(tf.gather(rough_peaks, 0, axis=1))), axis=1
    )
    valid_peaks = tf.gather(rough_peaks, valid_idx, axis=0)

    # Make bounding boxes for cropping around peaks.
    bboxes = make_centered_bboxes(
        valid_peaks, box_height=crop_size, box_width=crop_size
    )

    # Crop patch around each grid-aligned peak.
    cms = tf.reshape(
        tf.transpose(cms, [0, 3, 1, 2]),
        [samples * channels, tf.shape(cms)[1], tf.shape(cms)[2], 1],
    )
    cm_crops = crop_bboxes(cms, bboxes, sample_inds=valid_idx)

    # Compute offsets via integral regression on a local patch.
    if refinement == "integral":
        gv = tf.cast(tf.range(crop_size), tf.float32) - ((crop_size - 1) / 2)
        dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
        offsets = tf.concat([dx_hat, dy_hat], axis=1)
    else:
        offsets = find_offsets_local_direction(cm_crops, 0.25)

    # Apply offsets.
    refined_peaks = tf.tensor_scatter_nd_add(
        rough_peaks, tf.expand_dims(valid_idx, axis=1), offsets
    )

    # Reshape to (samples, channels, 2).
    refined_peaks = tf.reshape(refined_peaks, [samples, channels, 2])

    return refined_peaks, peak_vals


def find_global_peaks_integral(
    cms: tf.Tensor, crop_size: int = 5, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find local peaks with integral refinement.

    Integral regression refinement will be computed by taking the weighted average of
    the local neighborhood around each rough peak.

    Args:
        cms: Confidence maps. Tensor of shape (samples, height, width, channels).
        crop_size: Size of patches to crop around each rough peak as an integer scalar.
        threshold: Minimum confidence threshold. Peaks with values below this will be
            set to NaNs.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    return find_global_peaks(
        cms, threshold=threshold, refinement="integral", integral_patch_size=crop_size
    )


def find_local_peaks(
    cms: tf.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find local peaks with optional refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, height, width, channels).
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression. If `"local"`,
            peaks will be refined with quarter pixel local gradient offset.
        integral_patch_size: Size of patches to crop around each rough peak as an
            integer scalar.

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
    # Find grid aligned peaks.
    (
        rough_peaks,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_rough(cms, threshold=threshold)

    # Return early if no rough peaks found.
    if tf.shape(rough_peaks)[0] == 0 or refinement is None:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    if refinement == "integral":
        crop_size = integral_patch_size
    elif refinement == "local":
        crop_size = 3
    else:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    # Make bounding boxes for cropping around peaks.
    bboxes = make_centered_bboxes(
        rough_peaks, box_height=crop_size, box_width=crop_size
    )

    # Reshape to (samples * channels, height, width, 1).
    n_samples = tf.shape(cms)[0]
    n_channels = tf.shape(cms)[3]
    cms = tf.reshape(
        tf.transpose(cms, [0, 3, 1, 2]),
        [n_samples * n_channels, tf.shape(cms)[1], tf.shape(cms)[2], 1],
    )
    box_sample_inds = (peak_sample_inds * n_channels) + peak_channel_inds

    # Crop patch around each grid-aligned peak.
    cm_crops = crop_bboxes(cms, bboxes, sample_inds=box_sample_inds)

    # Compute offsets via integral regression on a local patch.
    if refinement == "integral":
        gv = tf.cast(tf.range(crop_size), tf.float32) - ((crop_size - 1) / 2)
        dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
        offsets = tf.concat([dx_hat, dy_hat], axis=1)
    else:
        offsets = find_offsets_local_direction(cm_crops, 0.25)

    # Apply offsets.
    refined_peaks = rough_peaks + offsets

    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds


def find_local_peaks_integral(
    cms: tf.Tensor, crop_size: int = 5, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Find local peaks with integral refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, height, width, channels).
        crop_size:
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.

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
    return find_local_peaks(
        cms, threshold=threshold, refinement="integral", integral_patch_size=crop_size
    )


def find_global_peaks_with_offsets(
    cms: tf.Tensor, offsets: tf.Tensor, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Find global peaks and refine with learned offset maps.

    Args:
        cms: Confidence maps tensor of shape `(samples, height, width, channels)`.
        offsets: Offset maps tensor of shape `(samples, height, width, 2 * channels)`.
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will be replaced with NaNs.

    Returns:
        A tuple of `(peak_points, peak_vals)`.

        `peak_points`: `float32` tensor of shape `(samples, channels, 2)`, where the
        last axis indicates peak locations in xy order.

        `peak_vals`: `float32` tensor of shape `(samples, channels)` containing the
        values at the peak points.
    """
    # Find grid aligned peaks.
    rough_peaks, peak_vals = find_global_peaks_rough(
        cms, threshold=threshold
    )  # (samples, channels, 2)

    # Return early if not refining or no rough peaks found.
    if tf.reduce_all(tf.math.is_nan(rough_peaks)):
        return rough_peaks, peak_vals

    samples = tf.shape(cms)[0]
    channels = tf.shape(cms)[3]

    # Setup subscript indexing to pull out offsets at the peaks.
    subs_samples = tf.broadcast_to(
        tf.reshape(tf.range(samples, dtype=tf.int32), [-1, 1, 1]),
        [samples, channels, 1],
    )
    subs_channels = tf.broadcast_to(
        tf.reshape(tf.range(channels, dtype=tf.int32), [1, -1, 1]),
        [samples, channels, 1],
    )
    subs = tf.concat(
        [
            subs_samples,
            tf.cast(tf.reverse(rough_peaks, axis=[2]), tf.int32),
            subs_channels,
        ],
        axis=2,
    )

    # Flatten samples and channels to (n_peaks, 2).
    rough_peaks = tf.reshape(rough_peaks, [samples * channels, 2])
    subs = tf.reshape(subs, [samples * channels, 4])

    # Keep only peaks that are not NaNs.
    valid_idx = tf.squeeze(
        tf.where(~tf.math.is_nan(tf.gather(rough_peaks, 0, axis=1))), axis=1
    )
    valid_peaks = tf.gather(rough_peaks, valid_idx, axis=0)
    valid_subs = tf.gather(subs, valid_idx, axis=0)

    # Expand last axes of offsets.
    shape = tf.shape(offsets)
    offsets = tf.reshape(offsets, [shape[0], shape[1], shape[2], -1, 2])

    # Extract offsets at the peak locations.
    peak_offsets = tf.gather_nd(offsets, valid_subs)

    # Apply offsets.
    refined_peaks = tf.tensor_scatter_nd_add(
        rough_peaks, tf.expand_dims(valid_idx, axis=1), peak_offsets
    )

    # Reshape to (samples, channels, 2).
    refined_peaks = tf.reshape(refined_peaks, [samples, channels, 2])

    return refined_peaks, peak_vals


def find_local_peaks_with_offsets(
    cms: tf.Tensor,
    offsets: tf.Tensor,
    threshold: float = 0.2,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Find local peaks and refine with learned offset maps.

    Args:
        cms: Confidence maps. Tensor of shape `(samples, height, width, channels)`.
        offsets: Offset maps tensor of shape `(samples, height, width, 2 * channels)`.
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.

    Returns:
        A tuple of `(peak_points, peak_vals, peak_sample_inds, peak_channel_inds)`.

        `peak_points`: float32 tensor of shape `(n_peaks, 2)`, where the last axis
        indicates peak locations in xy order.

        `peak_vals`: float32 tensor of shape `(n_peaks,)` containing the values at the
        peak points.

        `peak_sample_inds`: int32 tensor of shape `(n_peaks,)` containing the indices of
        the sample each peak belongs to.

        `peak_channel_inds`: int32 tensor of shape `(n_peaks,)` containing the indices
        of the channel each peak belongs to.
    """
    # Find grid aligned peaks.
    (
        rough_peaks,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_rough(cms, threshold=threshold)

    # Return early if no rough peaks found.
    if tf.shape(rough_peaks)[0] == 0:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    # Setup subscript indexing to pull out offsets at the peaks.
    subs = tf.concat(
        [
            tf.reshape(peak_sample_inds, [-1, 1]),
            tf.cast(tf.reverse(rough_peaks, axis=[1]), tf.int32),
            tf.reshape(peak_channel_inds, [-1, 1]),
        ],
        axis=1,
    )

    # Expand last axes of offsets.
    shape = tf.shape(offsets)
    offsets = tf.reshape(offsets, [shape[0], shape[1], shape[2], -1, 2])

    # Extract offsets at the peak locations.
    peak_offsets = tf.gather_nd(offsets, subs)

    # Apply offsets.
    refined_peaks = rough_peaks + peak_offsets

    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds

"""This module contains TensorFlow-based peak finding methods.

In general, the inputs to the functions provided here operate on confidence maps
(sometimes referred to as heatmaps), which are image-based representations of the
locations of landmark coordinates.

In these representations, landmark locations are encoded as probability that it is
present each pixel. This is often represented by an unnormalized 2D Gaussian PDF
centered at the true location and evaluated over the entire image grid.

Peak finding entails finding either the global or local maxima of these confidence maps.
"""

import attr
import tensorflow as tf
import numpy as np
from typing import Union, Tuple

from sleap.nn import model
from sleap.nn import utils


@tf.function
def find_global_peaks(img: tf.Tensor) -> tf.Tensor:
    """Finds the global maximum for each sample and channel.

    Args:
        img: Tensor of shape (samples, height, width, channels).

    Returns:
        A tuple of (peak_subs, peak_vals).

        peak_subs: float32 tensor of shape (n_peaks, 4), where n_peaks is the number
        of global peaks (samples * channels). The location of the i-th peak is
        specified by its subscripts in img, e.g.:
            sample, row, col, channel = find_global_peaks(img)[i]

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the
        subscripts indicated by peak_subs within img.
    """

    # Find row maxima.
    max_img_rows = tf.reduce_max(img, axis=2)
    argmax_rows = tf.reshape(tf.argmax(max_img_rows, axis=1), [-1])

    # Find col maxima.
    max_img_cols = tf.reduce_max(img, axis=1)
    argmax_cols = tf.reshape(tf.argmax(max_img_cols, axis=1), [-1])

    # Construct sample and channel subscripts.
    samples = tf.range(argmax_cols.shape[0], dtype=tf.int64) // img.shape[-1]
    channels = tf.range(argmax_cols.shape[0], dtype=tf.int64) % img.shape[-1]

    # Gather subscripts.
    peak_subs = tf.stack([samples, argmax_rows, argmax_cols, channels], axis=1)

    # Gather values at global maxima.
    peak_vals = tf.gather_nd(img, peak_subs)

    return tf.cast(peak_subs, tf.float32), peak_vals


@tf.function
def find_local_peaks(
    img: tf.Tensor, min_val: float = 0.3
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Finds local maxima via non-maximum suppresion.

    Args:
        img: Tensor of shape (samples, height, width, channels).
        min_val: Minimum threshold to consider a pixel a local maximum.

    Returns:
        A tuple of (peak_subs, peak_vals).

        peak_subs: float32 tensor of shape (n_peaks, 4), where n_peaks is the number of
        local maxima detected. The location of the i-th peak is specified by its
        subscripts in img, e.g.:
            sample, row, col, channel = find_local_peaks(img)[i]

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the
        subscripts indicated by peak_subs within img.
    """

    # Build custom local NMS kernel.
    kernel = tf.reshape(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32), (3, 3, 1)
    )

    # Perform dilation filtering to find local maxima per channel.
    img_channels = tf.split(img, tf.ones([tf.shape(img)[-1]], dtype=tf.int32), axis=-1)
    max_img = []
    for img_channel in img_channels:
        max_img.append(
            tf.nn.dilation2d(
                img_channel, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
            )
        )
    max_img = tf.concat(max_img, axis=-1)

    # Filter for maxima and threshold.
    argmax_and_thresh_img = tf.greater(img, max_img) & tf.greater(img, min_val)

    # Convert to subscripts.
    peak_subs = tf.where(argmax_and_thresh_img)

    # Get peak values.
    peak_vals = tf.gather_nd(img, peak_subs)

    return tf.cast(peak_subs, tf.float32), peak_vals


def crop_centered_boxes(
    img: tf.Tensor, peaks: tf.Tensor, window_length: int
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
    window_length = utils.ensure_odd(window_length)
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


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int64),
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ]
)
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


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 3, 3, 1), dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ]
)
def find_offsets_local_direction(
    centered_patches: tf.Tensor, delta: float
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
        the offsets relative to the center pixel for the y and x directions
        respectively, i.e., for the i-th sample:

            dy_i, dx_i = offsets[i]

    Note:
        For symmetric patches, the offset will be 0.

    Example:
        >>> find_offsets_local_direction(np.array(
        ...     [[0., 1., 0.],
        ...      [1., 3., 2.],
        ...      [0., 1., 0.]]).reshape(1, 3, 3, 1), 0.25)
        <tf.Tensor: id=21250, shape=(1, 2), dtype=float64, numpy=array([[0.  , 0.25]])>
    """

    # Compute directional gradients.
    dx = centered_patches[:, 1, 2, :] - centered_patches[:, 1, 0, :]  # right - left
    dy = centered_patches[:, 2, 1, :] - centered_patches[:, 0, 1, :]  # bottom - top

    # Concatenate and scale signed direction by delta.
    offsets = tf.sign(tf.squeeze(tf.stack([dy, dx], axis=1), axis=-1)) * delta

    return offsets


def refine_peaks_local_direction(
    imgs: tf.Tensor, peaks: tf.Tensor, delta: float = 0.25
) -> tf.Tensor:
    """Refines peaks by applying a fixed offset along the gradients around the peaks.

    This function wraps other methods to refine peak coordinates by:
        1. Cropping 3 x 3 patches around each peak.
        2. Stacking patches along the samples axis.
        3. Computing the local gradient around each centered patch.
        4. Applying subpixel offsets to each peak.

    This is a commonly used algorithm for subpixel peak refinement, described for pose
    estimation applications in [1].

    Args:
        imgs: A float32 tensor of shape (samples, height, width, channels) in which the
            peaks were detected.
        peaks: Tensor of shape (n_peaks, 4) where subscripts of peak locations are
            specified in each row as [sample, row, col, channel].
        delta: Scalar float specifying the step to take along the local peak gradients.

    Returns:
        refined_peaks, a float32 tensor of shape (n_peaks, 4) in the same format as the
        input peaks, but with offsets applied.

    References:
        .. [1] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked Hourglass Networks
           for Human Pose Estimation. In _European conference on computer vision_, 2016.
    """

    # Extract peak-centered patches.
    all_peak_patches = crop_centered_boxes(imgs, peaks, window_length=3)

    # Compute local offsets.
    offsets = find_offsets_local_direction(all_peak_patches, delta=delta)

    # Apply offsets to refine peaks.
    refined_peaks = tf.cast(peaks, tf.float32) + tf.pad(offsets, [[0, 0], [1, 1]])

    return refined_peaks


@attr.s(auto_attribs=True, slots=True)
class RegionPeakSet:
    peaks: np.ndarray
    peak_vals: np.ndarray
    patch_inds: np.ndarray

    @property
    def sample_inds(self):
        return self.peaks[:, 0]

    @property
    def peaks_with_patch_inds(self):
        return np.concatenate((self.patch_inds[:, None], self.peaks[:, 1:]), axis=1)


@attr.s(auto_attribs=True, eq=False)
class ConfmapPeakFinder:

    inference_model: model.InferenceModel
    batch_size: int = 16
    rps_batch_size: int = 64
    smoothing_kernel_size: int = 5
    smoothing_sigma: float = 1.0
    min_peak_threshold: float = 0.3

    def preproc(self, imgs):
        # Scale to model input size.
        imgs = utils.resize_imgs(
            imgs,
            self.inference_model.input_scale,
            common_divisor=2 ** self.inference_model.down_blocks,
        )

        # Convert to float32 and scale values to [0., 1.].
        imgs = utils.normalize_imgs(imgs)

        return imgs

    @tf.function
    def inference(self, imgs):

        # Model inference
        confmaps = self.inference_model.keras_model(imgs)

        if self.smoothing_sigma > 0:
            # Smooth
            confmaps = smooth_imgs(
                confmaps,
                kernel_size=self.smoothing_kernel_size,
                sigma=self.smoothing_sigma,
            )

        return confmaps

    @tf.function
    def postproc(self, confmaps):

        # Peak finding
        peak_subs, peak_vals = find_local_peaks(
            confmaps, min_val=self.min_peak_threshold
        )
        peak_subs = refine_peaks_local_direction(confmaps, peak_subs)
        peak_subs /= tf.constant(
            [
                [
                    1,
                    self.inference_model.output_scale,
                    self.inference_model.output_scale,
                    1,
                ]
            ]
        )

        return tf.concat([peak_subs, tf.expand_dims(peak_vals, axis=1)], axis=1)

    def predict_rps(self, rps: "RegionProposalSet") -> RegionPeakSet:

        peak_subs_and_vals, batch_inds = utils.batched_call(
            lambda imgs: self.postproc(self.inference(self.preproc(imgs))),
            rps.patches,
            batch_size=self.batch_size,
            return_batch_inds=True,
        )

        # Split.
        peaks, peak_vals = tf.split(peak_subs_and_vals, [4, 1], axis=1)

        # Pull out patch indices and adjust for batching.
        patch_inds = tf.cast(peaks[:, 0] + (batch_inds * self.batch_size), tf.int32)

        # Copy everything to CPU.
        peaks = peaks.numpy()
        peak_vals = peak_vals.numpy().squeeze()
        patch_inds = patch_inds.numpy()

        # Update subscripts with the sample indices and adjust to image coords.
        peaks[:, 0] = rps.sample_inds[patch_inds]
        peaks[:, 1] += rps.bboxes[patch_inds, 0]
        peaks[:, 2] += rps.bboxes[patch_inds, 1]

        region_peaks = RegionPeakSet(
            peaks=peaks, peak_vals=peak_vals, patch_inds=patch_inds
        )

        return region_peaks

"""This module contains generic utilities used for training and inference."""

import tensorflow as tf
import numpy as np
from collections import defaultdict
from typing import Callable, Dict


def ensure_odd(x: tf.Tensor) -> tf.Tensor:
    """Rounds numbers up to the nearest odd value."""

    return (x // 2) * 2 + 1


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


def batched_call(
    fn: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, batch_size: int = 8
) -> tf.Tensor:
    """Calls a TensorFlow-based function in batches and returns concatenated outputs.

    This function concatenates the outputs from calling fn(x_batch) for each batch
    in x. This is useful when applying functions to large tensors that may not fit in
    GPU memory.

    Args:
        fn: Any callable, such as a function or a tf.keras.Model.__call__ method. This
            function should accept inputs of the same rank as x.
        x: Input data tensor to be batched along the first axis. Rank will not change
            with batching when provided to fn.
        batch_size: Number of elements along the first axis of x to evaluate at a time.

    Returns:
        The output of fn(x) as if called without batching.

    Notes:
        The input will be batched into as many batches as possible with a length of
        batch_size. If len(x) is not divisible by batch_size, the remainder will always
        be in the last batch.

        In contrast to tf.keras.Model.predict, this function will not transfer the
        output of fn(x) to the CPU. This is useful when performing subsequent operations
        on the output tensor on the same device (e.g., GPU).

        Be aware that this method of batching incurs some performance overhead at small
        batch sizes.
    """

    # Split indices into batches.
    all_indices = np.arange(len(x))
    batched_indices = np.split(all_indices, all_indices[batch_size::batch_size])

    # Evaluate each batch.
    outputs = []
    for indices in batched_indices:
        x_batch = x[indices[0] : (indices[-1] + 1)]
        outputs.append(fn(x_batch))

    # Return concatenated outputs.
    return tf.concat(outputs, axis=0)


def group_array(
    X: np.ndarray, groups: np.ndarray, axis=0
) -> Dict[np.ndarray, np.ndarray]:
    """Groups an array into a dictionary keyed by a grouping vector.
    
    Args:
        X: Numpy array with length n along the specified axis.
        groups: Vector of n values denoting the group that each slice of X should be
            assigned to. This is also referred to as an indicator, indexing, class,
            or labels vector.
        axis: Dimension of X to group on. The length of this axis in X must correspond
            to the length of groups.
    
    Returns:
        A dictionary with keys mapping each unique value in groups to a subset of X.
        
    References:
        See this `blog post<https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/>`
        for performance comparisons of different approaches.
        
    Example:
        >>> group_array(np.arange(5), np.array([1, 5, 2, 1, 5]))
        {1: array([0, 3]), 5: array([1, 4]), 2: array([2])}
    """

    group_inds = defaultdict(list)
    for ind, key in enumerate(groups):
        group_inds[key].append(ind)

    return {key: np.take(X, inds, axis=axis) for key, inds in group_inds.items()}


def resize_imgs(
    imgs: tf.Tensor, scale: float, method="bilinear", common_divisor: int = 1
) -> tf.Tensor:
    """Resizes a stack of images by a scaling factor with optional padding.

    This method is primarily a convenience wrapper that calculates the target shape
    while maintaining the input aspect ratio and optionally padding to the next
    largest shape that is a common divisor.

    Args:
        imgs: A tensor of shape (samples, height, width, channels).
        scale: Scalar float factor to rescale the images by.
        method: Resizing method to use. Valid values include: "nearest", "bilinear",
            "cubic", "area" and any other supported by tf.image.resize.
        common_divisor: Scalar integer. Target shape computation will be adjusted to
            ensure that the result is divisible by this number. If needed, the inputs
            will be padded (bottom and right) to satisfy this constraint.

    Returns:
        The resized imgs tensor.
    """

    # Get input image dimensions.
    img_shape = tf.shape(imgs)
    img_height = tf.cast(img_shape[1], tf.float32)
    img_width = tf.cast(img_shape[2], tf.float32)

    # Compute initial target dimensions.
    target_height = tf.cast(img_height * scale, tf.int32)
    target_width = tf.cast(img_width * scale, tf.int32)

    # Apply resizing.
    resized_imgs = tf.image.resize(imgs, [target_height, target_width], method=method)

    if common_divisor > 1:

        # Calculate next largest size that is a common divisor.
        divisible_height = tf.cast(
            tf.math.ceil(
                tf.cast(target_height, tf.float32) / tf.cast(common_divisor, tf.float32)
            )
            * common_divisor, tf.int32
        )
        divisible_width = tf.cast(
            tf.math.ceil(
                tf.cast(target_width, tf.float32) / tf.cast(common_divisor, tf.float32)
            )
            * common_divisor, tf.int32
        )

        # Pad bottom/right as needed.
        resized_imgs = tf.image.pad_to_bounding_box(
            resized_imgs, 0, 0, divisible_height, divisible_width
        )

    return resized_imgs

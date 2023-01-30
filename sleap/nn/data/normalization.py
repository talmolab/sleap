"""Transformers for normalizing data formats."""

import tensorflow as tf
import numpy as np
from sleap.nn.data.utils import expand_to_rank
import attr
from typing import List, Text, Optional
from sleap.nn.config import PreprocessingConfig


def ensure_min_image_rank(image: tf.Tensor) -> tf.Tensor:
    """Expand the image to a minimum rank of 3 by adding single dimensions.

    Args:
        image: Tensor of any rank and dtype.

    Returns:
        The expanded image to a minimum rank of 3.

        If the input was rank-2, it is assumed be of shape (height, width), so a
        singleton channels axis is appended to produce a tensor of shape
        (height, width, 1).

        If the image was already of rank >= 3, it will be returned without changes.

    See also: sleap.nn.data.utils.expand_to_rank
    """
    if tf.rank(image) < 3:
        return expand_to_rank(image, 3, prepend=False)
    else:
        return image


def ensure_float(image: tf.Tensor) -> tf.Tensor:
    """Convert the image to a tf.float32.

    Args:
        image: Tensor of any dtype.

    Returns:
        A tensor of the same shape as `image` but with dtype tf.float32. If the image
        was already of tf.float32 dtype, it will not be changed.

        If the input was of an integer type, it will be scaled to the range [0, 1]
        according to the dtype's maximum value.

    See also: tf.image.convert_image_dtype
    """
    return tf.image.convert_image_dtype(image, tf.float32)


def ensure_int(image: tf.Tensor) -> tf.Tensor:
    """
    Convert the image to a tf.uint8.

    If the image is a floating dtype, then converts and scales data from [0, 1]
    to [0, 255] as needed. Otherwise, returns image as is.

    Args:
        image: Tensor of any dtype.

    Returns:
        A tensor of the same shape as `image` but with dtype tf.uint8.
        If the image was not a floating dtype, then it will not be changed.

        If the input was float with range [0, 1], it will be scaled to [0, 255].
    """
    # Tensors have is_floating attribute, ndarrays don't.
    is_float = getattr(
        image.dtype, "is_floating", image.dtype in (np.float32, np.float64)
    )

    if is_float:
        if tf.reduce_max(image) <= 1.0:
            return tf.image.convert_image_dtype(image, tf.uint8)
        return tf.cast(image, tf.uint8)

    return image


def ensure_grayscale(image: tf.Tensor) -> tf.Tensor:
    """Convert image to grayscale if in RGB format.

    Args:
        image: Tensor of any dtype of shape (height, width, channels). Channels are
            expected to be 1 or 3.

    Returns:
        A grayscale image of shape (height, width, 1) of the same dtype as the input.

    See also: tf.image.rgb_to_grayscale
    """
    if image.shape[-1] == 3:
        return tf.image.rgb_to_grayscale(image)
    else:
        return image


def ensure_rgb(image: tf.Tensor) -> tf.Tensor:
    """Convert image to RGB if in grayscale format.

    Args:
        image: Tensor of any dtype of shape (height, width, channels). Channels are
            expected to be 1 or 3.

    Returns:
        A grayscale image of shape (height, width, 1) of the same dtype as the input.

    See also: tf.image.grayscale_to_rgb
    """
    if image.shape[-1] == 1:
        return tf.image.grayscale_to_rgb(image)
    else:
        return image


def convert_rgb_to_bgr(image: tf.Tensor) -> tf.Tensor:
    """Convert an RGB image to BGR format by reversing the channel order.

    Args:
        image: Tensor of any dtype with shape (..., 3) in RGB format. If grayscale, the
            image will be converted to RGB first.

    Returns:
        The input image with the channels axis reversed.
    """
    return tf.reverse(ensure_rgb(image), axis=[-1])


def scale_image_range(image: tf.Tensor, min_val: float, max_val: float) -> tf.Tensor:
    """Scale the range of image values.

    Args:
        image: Tensor of any shape of dtype tf.float32 with values in the range [0, 1].
        min_val: The minimum number that values will be scaled to.
        max_val: The maximum number that values will be scaled to.

    Returns:
        The scaled image of the same shape and dtype tf.float32. Values in the input
        that were 0 will now be scaled to `min_val`, and values that were 1.0 will be
        scaled to `max_val`.
    """
    return (image * (max_val - min_val)) + min_val


def scale_to_imagenet_tf_mode(image: tf.Tensor) -> tf.Tensor:
    """Scale images according to the "tf" preprocessing mode.

    This applies the preprocessing operations implemented in `tf.keras.applications` for
    models pretrained on ImageNet.

    Args:
        image: Any image tensor of rank >= 2.

    Returns:
        The preprocessed image of dtype tf.float32 and shape (..., height, width, 3)
        with RGB channel ordering.

        Values will be in the range [-1.0, 1.0].

    Notes:
        The preprocessing steps applied are:
            1. If needed, expand to rank-3 by adding singleton dimensions to the end.
               This assumes rank-2 images are grayscale of shape (height, width) and
               will be expanded to (height, width, 1).
            2. Convert to RGB if not already in 3 channel format.
            3. Convert to tf.float32 in the range [0.0, 1.0].
            4. Scale the values to the range [-1.0, 1.0].

        This preprocessing mode is required when using pretrained ResNetV2, MobileNetV1,
        MobileNetV2 and NASNet models.
    """
    image = ensure_min_image_rank(image)  # at least [height, width, 1]
    image = ensure_rgb(image)  # 3 channels
    image = ensure_float(image)  # float32 in [0., 1.]
    image = scale_image_range(image, min_val=-1.0, max_val=1.0)  # float32 in [-1, 1]
    return image


def scale_to_imagenet_caffe_mode(image: tf.Tensor) -> tf.Tensor:
    """Scale images according to the "caffe" preprocessing mode.

    This applies the preprocessing operations implemented in `tf.keras.applications` for
    models pretrained on ImageNet.

    Args:
        image: Any image tensor of rank >= 2. If rank >=3, the last axis is assumed to
            be of size 3 corresponding to RGB-ordered channels.

    Returns:
        The preprocessed image of dtype tf.float32 and shape (..., height, width, 3)
        with BGR channel ordering.

        Values will be in the approximate range of [-127.5, 127.5].

    Notes:
        The preprocessing steps applied are:
            1. If needed, expand to rank-3 by adding singleton dimensions to the end.
               This assumes rank-2 images are grayscale of shape (height, width) and
               will be expanded to (height, width, 1).
            2. Convert to RGB if not already in 3 channel format.
            3. Reverse the channel ordering to convert RGB to BGR format.
            4. Convert to tf.float32 in the range [0.0, 1.0].
            5. Scale the values to the range [0.0, 255.0].
            6. Subtract the ImageNet mean values (103.939, 116.779, 123.68) for channels
               in BGR format.

        This preprocessing mode is required when using pretrained ResNetV1 models.
    """
    image = ensure_min_image_rank(image)  # at least [height, width, 1]
    image = ensure_rgb(image)  # 3 channels
    image = convert_rgb_to_bgr(image)  # reverse channel order
    image = ensure_float(image)  # float32 in range [0., 1.]
    image = scale_image_range(
        image, min_val=0.0, max_val=255.0
    )  # float32 in range [0, 255]
    imagenet_mean = tf.convert_to_tensor(
        [103.939, 116.779, 123.68], tf.float32
    )  # [B, G, R]
    image = image - expand_to_rank(
        imagenet_mean, tf.rank(image)
    )  # subtract from channels
    return image


def scale_to_imagenet_torch_mode(image: tf.Tensor) -> tf.Tensor:
    """Scale images according to the "torch" preprocessing mode.

    This applies the preprocessing operations implemented in `tf.keras.applications` for
    models pretrained on ImageNet.

    Args:
        image: Any image tensor of rank >= 2. If rank >=3, the last axis is assumed to
            be of size 3 corresponding to RGB-ordered channels.

    Returns:
        The preprocessed image of dtype tf.float32 and shape (..., height, width, 3)
        with RGB channel ordering.

        Values will be in the approximate range of [-0.5, 0.5].

    Notes:
        The preprocessing steps applied are:
            1. If needed, expand to rank-3 by adding singleton dimensions to the end.
               This assumes rank-2 images are grayscale of shape (height, width) and
               will be expanded to (height, width, 1).
            2. Convert to RGB if not already in 3 channel format.
            3. Convert to tf.float32 in the range [0.0, 1.0].
            4. Subtract the ImageNet mean values (0.485, 0.456, 0.406) for channels in
               RGB format.
            5. Divide by the ImageNet standard deviation values (0.229, 0.224, 0.225)
               for channels in RGB format.

        This preprocessing mode is required when using pretrained DenseNet models.
    """
    image = ensure_min_image_rank(image)  # at least [height, width, 1]
    image = ensure_rgb(image)  # 3 channels
    image = ensure_float(image)  # float32 in range [0., 1.]
    imagenet_mean = tf.convert_to_tensor([0.485, 0.456, 0.406], tf.float32)  # [R, G, B]
    image = image - expand_to_rank(
        imagenet_mean, tf.rank(image)
    )  # subtract from channels
    imagenet_std = tf.convert_to_tensor([0.229, 0.224, 0.225], tf.float32)  # [R, G, B]
    image = image / expand_to_rank(imagenet_std, tf.rank(image))
    return image


@attr.s(auto_attribs=True)
class Normalizer:
    """Data transformer to normalize images.

    This is useful as a transformation to data streams that require specific data ranges
    such as for pretrained models with specific preprocessing constraints.

    Attributes:
        image_key: String name of the key containing the images to normalize.
        ensure_float: If True, converts the image to a `tf.float32` if not already.
        ensure_rgb: If True, converts the image to RGB if not already.
        ensure_grayscale: If True, converts the image to grayscale if not already.
        imagenet_mode: Specifies an ImageNet-based normalization mode commonly used in
            `tf.keras.applications`-based pretrained models. No effect if not set.
            Valid values are:
            "tf": Values will be scaled to [-1, 1], expanded to RGB if grayscale.
            "caffe": Values will be scaled to [0, 255], expanded to RGB if grayscale,
                RGB channels flipped to BGR, and subtracted by a fixed mean.
            "torch": Values will be scaled to [0, 1], expanded to RGB if grayscale,
                subtracted by a fixed mean, and scaled by fixed standard deviation.
    """

    image_key: Text = "image"
    ensure_float: bool = True
    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    imagenet_mode: Optional[Text] = attr.ib(
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(["tf", "caffe", "torch"])
        ),
    )

    @classmethod
    def from_config(
        cls, config: PreprocessingConfig, image_key: Text = "image"
    ) -> "Normalizer":
        """Build an instance of this class from its configuration options.

        Args:
            config: An `PreprocessingConfig` instance with the desired parameters.
            image_key: String name of the key containing the images to normalize.

        Returns:
            An instance of this class.
        """
        return cls(
            image_key=image_key,
            ensure_float=True,
            ensure_rgb=config.ensure_rgb,
            ensure_grayscale=config.ensure_grayscale,
            imagenet_mode=config.imagenet_mode,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [self.image_key]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.input_keys

    def transform_dataset(self, ds_input: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains centroids computed from the inputs.

        Args:
            ds_input: A dataset with image key specified in the `image_key` attribute.

        Returns:
            A `tf.data.Dataset` with elements containing the same images with
            normalization applied.
        """
        test_ex = next(iter(ds_input))
        img_shape = test_ex[self.image_key].shape
        output_shape = img_shape[-3:]
        if self.ensure_rgb or self.imagenet_mode is not None:
            output_shape = img_shape[-3:-1] + (3,)
        if self.ensure_grayscale:
            output_shape = img_shape[-3:-1] + (1,)
        if len(img_shape) == 4:
            output_shape = (None,) + output_shape

        def normalize(example):
            """Local processing function for dataset mapping."""
            if self.ensure_float:
                example[self.image_key] = ensure_float(example[self.image_key])
            if self.ensure_rgb:
                example[self.image_key] = ensure_rgb(example[self.image_key])
            if self.ensure_grayscale:
                example[self.image_key] = ensure_grayscale(example[self.image_key])
            if self.imagenet_mode == "tf":
                example[self.image_key] = scale_to_imagenet_tf_mode(
                    example[self.image_key]
                )
            if self.imagenet_mode == "caffe":
                example[self.image_key] = scale_to_imagenet_caffe_mode(
                    example[self.image_key]
                )
            if self.imagenet_mode == "torch":
                example[self.image_key] = scale_to_imagenet_torch_mode(
                    example[self.image_key]
                )
            example[self.image_key] = tf.ensure_shape(
                example[self.image_key], output_shape
            )

            return example

        # Map transformation.
        ds_output = ds_input.map(
            normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return ds_output

"""Transformers for image resizing and padding."""

import tensorflow as tf
import attr
from typing import List, Text, Optional, Tuple
from sleap.nn.config import PreprocessingConfig
from sleap.nn.data.utils import expand_to_rank


def find_padding_for_stride(
    image_height: int, image_width: int, max_stride: int
) -> Tuple[int, int]:
    """Compute padding required to ensure image is divisible by a stride.

    This function is useful for determining how to pad images such that they will not
    have issues with divisibility after repeated pooling steps.

    Args:
        image_height: Scalar integer specifying the image height (rows).
        image_width: Scalar integer specifying the image height (columns).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.

    Returns:
        A tuple of (pad_bottom, pad_right), integers with the number of pixels that the
        image would need to be padded by to meet the divisibility requirement.
    """
    pad_bottom = (max_stride - (image_height % max_stride)) % max_stride
    pad_right = (max_stride - (image_width % max_stride)) % max_stride
    return pad_bottom, pad_right


def pad_to_stride(image: tf.Tensor, max_stride: int) -> tf.Tensor:
    """Pad an image to meet a max stride constraint.

    This is useful for ensuring there is no size mismatch between an image and the
    output tensors after multiple downsampling and upsampling steps.

    Args:
        image: Single image tensor of shape (height, width, channels).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by. This is the ratio between the length of the image and the
            length of the smallest tensor it is converted to. This is typically
            `2 ** n_down_blocks`, where `n_down_blocks` is the number of 2-strided
            reduction layers in the model.

    Returns:
        The input image with 0-padding applied to the bottom and/or right such that the
        new shape's height and width are both divisible by `max_stride`.
    """
    pad_bottom, pad_right = find_padding_for_stride(
        image_height=tf.shape(image)[0],
        image_width=tf.shape(image)[1],
        max_stride=max_stride,
    )
    if pad_bottom > 0 or pad_right > 0:
        paddings = tf.cast([[0, pad_bottom], [0, pad_right], [0, 0]], tf.int32)
        image = tf.pad(image, paddings, mode="CONSTANT", constant_values=0)
    return image


def resize_image(image: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    """Rescale an image by a scale factor.

    This function is primarily a convenience wrapper for `tf.image.resize` that
    calculates the new shape from the scale factor.

    Args:
        image: Single image tensor of shape (height, width, channels).
        scale: Factor to resize the image dimensions by, specified as either a float
            scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both
            dimensions are resized by the same factor.

    Returns:
        The resized image tensor of the same dtype but scaled height and width.

    See also: tf.image.resize
    """
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    new_size = tf.reverse(
        tf.cast(
            tf.cast([width, height], tf.float32) * tf.cast(scale, tf.float32), tf.int32
        ),
        [0],
    )
    return tf.cast(
        tf.image.resize(
            image,
            size=new_size,
            method="bilinear",
            preserve_aspect_ratio=False,
            antialias=False,
        ),
        image.dtype,
    )


@attr.s(auto_attribs=True)
class Resizer:
    """Data transformer to resize or pad images.

    This is useful as a transformation to data streams that require resizing or padding
    in order to be downsampled or meet divisibility criteria.

    Attributes:
        image_key: String name of the key containing the images to resize.
        scale_key: String name of the key containing the scale of the images.
        points_key: String name of the key containing points to adjust for the resizing
            operation.
        scale: Scalar float specifying scaling factor to resize images by.
        pad_to_stride: Maximum stride in a model that the images must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `scale` attribute.
        keep_full_image: If True, keeps the (original size) full image in the examples.
            This is useful for multi-scale inference.
        full_image_key: String name of the key containing the full images.
    """

    image_key: Text = "image"
    scale_key: Text = "scale"
    points_key: Optional[Text] = "instances"
    scale: float = 1.0
    pad_to_stride: int = 1
    keep_full_image: bool = False
    full_image_key: Text = "full_image"

    @classmethod
    def from_config(
        cls,
        config: PreprocessingConfig,
        image_key: Text = "image",
        scale_key: Text = "scale",
        pad_to_stride: Optional[int] = None,
        keep_full_image: bool = False,
        full_image_key: Text = "full_image",
        points_key: Optional[Text] = "instances"
    ) -> "Resizer":
        """Build an instance of this class from its configuration options.

        Args:
            config: An `PreprocessingConfig` instance with the desired parameters. If
                `config.pad_to_stride` is not an explicit integer, the `pad_to_stride`
                parameter must be provided.
            image_key: String name of the key containing the images to resize.
            scale_key: String name of the key containing the scale of the images.
            pad_to_stride: An integer specifying the `pad_to_stride` if
                `config.pad_to_stride` is not an explicit integer (e.g., set to None).
            keep_full_image: If True, keeps the (original size) full image in the
                examples. This is useful for multi-scale inference.
            full_image_key: String name of the key containing the full images.
            points_key: String name of the key containing points to adjust for the
                resizing operation.

        Returns:
            An instance of this class.

        Raises:
            ValueError: If `config.pad_to_stride` is not set to an integer and the
                `pad_to_stride` argument is not provided.
        """
        if isinstance(config.pad_to_stride, int):
            pad_to_stride = config.pad_to_stride
        if not isinstance(pad_to_stride, int):
            raise ValueError(
                "Pad to stride must be specified in the arguments if not explicitly "
                f"set to an integer (config.pad_to_stride = {config.pad_to_stride})."
            )

        return cls(
            image_key=image_key,
            points_key=points_key,
            scale_key=scale_key,
            scale=config.input_scaling,
            pad_to_stride=pad_to_stride,
            keep_full_image=keep_full_image,
            full_image_key=full_image_key,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        input_keys = [self.image_key, self.scale_key]
        if self.points_key is not None:
            input_keys.append(self.points_key)
        return input_keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        output_keys = self.input_keys
        if self.keep_full_image:
            output_keys.append(self.full_image_key)
        return output_keys

    def transform_dataset(self, ds_input: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains centroids computed from the inputs.

        Args:
            ds_input: A dataset with the image specified in the `image_key` attribute,
                points specified in the `points_key` attribute, and the "scale" key for
                tracking scaling transformations.

        Returns:
            A `tf.data.Dataset` with elements containing the same images and points with
            resizing applied.

            The "scale" key of the example will be multipled by the `scale` attribute of
            this transformer.

            If the `keep_full_image` attribute is True, a key specified by
            `full_image_key` will be added with the to the example containing the image
            before any processing.
        """

        def resize(example):
            """Local processing function for dataset mapping."""
            if self.keep_full_image:
                example[self.full_image_key] = example[self.image_key]

            if self.scale != 1.0:
                # Ensure image is rank-3 for resizing ops.
                example[self.image_key] = tf.ensure_shape(
                    example[self.image_key], (None, None, None)
                )
                example[self.image_key] = resize_image(
                    example[self.image_key], self.scale
                )
                if self.points_key:
                    example[self.points_key] = example[self.points_key] * self.scale
                example[self.scale_key] = example[self.scale_key] * self.scale

            if self.pad_to_stride > 1:
                example[self.image_key] = pad_to_stride(
                    example[self.image_key], max_stride=self.pad_to_stride
                )
            return example

        # Map transformation.
        ds_output = ds_input.map(
            resize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return ds_output


@attr.s(auto_attribs=True)
class PointsRescaler:
    """Transformer to apply or invert scaling operations on points."""

    points_key: Text = "predicted_instances"
    scale_key: Text = "scale"
    invert: bool = True

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [self.points_key, self.scale_key]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.input_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains instance cropped data."""

        def rescale_points(example):
            """Local processing function for dataset mapping."""
            # Pull out data.
            points = example[self.points_key]
            scale = example[self.scale_key]

            # Make sure the scale lines up with the last dimension of the points.
            scale = expand_to_rank(scale, tf.rank(points))

            # Scale.
            if self.invert:
                points /= scale
            else:
                points *= scale

            # Update example.
            example[self.points_key] = points
            return example

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            rescale_points, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return output_ds

"""Transformers for cropping instances for topdown processing."""

import tensorflow as tf
import numpy as np
import attr
from typing import Optional, List, Text
import sleap
from sleap.nn.config import InstanceCroppingConfig


def find_instance_crop_size(
    labels: sleap.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    input_scaling: float = 1.0,
) -> int:
    """Compute the size of the largest instance bounding box from labels.

    Args:
        labels: A `sleap.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin padding.
        maximum_stride: Ensure that the returned crop size is divisible by this value.
            Useful for ensuring that the crop size will not be truncated in a given
            architecture.
        input_scaling: Float factor indicating the scale of the input images if any
            scaling will be done before cropping.

    Returns:
        An integer crop size denoting the length of the side of the bounding boxes that
        will contain the instances when cropped.

        This accounts for stride, padding and scaling when ensuring divisibility.
    """
    max_length = 0.0
    for inst in labels.user_instances:
        pts = inst.points_array
        pts *= input_scaling
        max_length = np.maximum(max_length, np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0]))
        max_length = np.maximum(max_length, np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1]))

    max_length += float(padding)
    crop_size = np.math.ceil(max_length / float(maximum_stride)) * maximum_stride

    return int(crop_size)


def normalize_bboxes(
    bboxes: tf.Tensor, image_height: int, image_width: int
) -> tf.Tensor:
    """Normalize bounding box coordinates to the range [0, 1].

    This is useful for transforming points for TensorFlow operations that require
    normalized image coordinates.

    Args:
        bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the last axis
            corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes.
        image_height: Scalar integer indicating the height of the image.
        image_width: Scalar integer indicating the width of the image.

    Returns:
        Tensor of the normalized points of the same shape as `bboxes`.

        The normalization applied to each point is `x / (image_width - 1)` and
        `y / (image_width - 1)`.

    See also: unnormalize_bboxes
    """
    # Compute normalizing factor of shape (1, 4).
    factor = (
        tf.convert_to_tensor(
            [[image_height, image_width, image_height, image_width]], tf.float32
        )
        - 1
    )

    # Normalize and return.
    normalized_bboxes = bboxes / factor
    return normalized_bboxes


def unnormalize_bboxes(
    normalized_bboxes: tf.Tensor, image_height: int, image_width: int
) -> tf.Tensor:
    """Convert bounding boxes coordinates in the range [0, 1] to absolute coordinates.

    Args:
        normalized_bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the
            last axis corresponds to (y1, x1, y2, x2) normalized coordinates of the
            bounding boxes in the range [0, 1].
        image_height: Scalar integer indicating the height of the image.
        image_width: Scalar integer indicating the width of the image.

    Returns:
        Tensor of the same shape as `bboxes` mapped back to absolute image coordinates
        by multiplying (x, y) coordinates by `(image_width - 1, image_height - 1)`.

    See also: normalize_bboxes
    """
    # Compute normalizing factor.
    factor = (
        tf.convert_to_tensor(
            [[image_height, image_width, image_height, image_width]], tf.float32
        )
        - 1
    )

    # Unnormalize and return.
    bboxes = normalized_bboxes * factor
    return bboxes


def make_centered_bboxes(
    centroids: tf.Tensor, box_height: int, box_width: int
) -> tf.Tensor:
    """Generate bounding boxes centered on a set of centroid coordinates.

    Args:
        centroids: A tensor of shape (n_centroids, 2) and dtype tf.float32, where the
            last axis corresponds to the (x, y) coordinates of each centroid.
        box_height: Scalar integer indicating the height of the bounding boxes.
        box_width: Scalar integer indicating the width of the bounding boxes.

    Returns:
        Tensor of shape (n_centroids, 4) and dtype tf.float32, where the last axis
        corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes in absolute
        image coordinates.

    Notes:
        The bounding box coordinates are calculated such that the centroid coordinates
        map onto the center of the pixel. For example:

        For a single row image of shape (1, 4) with values: `[[a, b, c, d]]`, the x
        coordinates can be visualized in the diagram below:
                 _______________________
                |  a  |  b  |  c  |  d  |
                |  |  |  |  |  |  |  |  |
              -0.5 | 0.5 | 1.5 | 2.5 | 3.5
                   0     1     2     3

        To get a (1, 3) patch centered at c, the centroid would be at (x, y) = (2, 0)
        with box height of 1 and box width of 3, to yield `[[b, c, d]]`.

        For even sized bounding boxes, e.g., to get the center 2 elements, the centroid
        would be at (x, y) = (1.5, 0) with box width of 2, to yield `[[b, c]]`.
    """
    delta = (
        tf.convert_to_tensor(
            [[-box_height + 1, -box_width + 1, box_height - 1, box_width - 1]],
            tf.float32,
        )
        * 0.5
    )
    bboxes = tf.gather(centroids, [1, 0, 1, 0], axis=-1) + delta
    return bboxes


def crop_bboxes(image: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
    """Crop bounding boxes from an image.

    This method serves as a convenience method for specifying the arguments of
    `tf.image.crop_and_resize`, becoming especially useful in the case of multiple
    bounding boxes with a single image and no resizing.

    Args:
        image: Tensor of shape (height, width, channels) of a single image.
        bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the last axis
            corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes. This can
            be generated from centroids using `make_centered_bboxes`.

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
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    normalized_bboxes = normalize_bboxes(
        bboxes, image_height=image_height, image_width=image_width
    )

    # Crop.
    crops = tf.image.crop_and_resize(
        image=tf.expand_dims(image, axis=0),
        boxes=normalized_bboxes,
        box_indices=tf.zeros([tf.shape(bboxes)[0]], dtype=tf.int32),
        crop_size=box_size,
        method="bilinear",
    )

    # Cast back to original dtype and return.
    crops = tf.cast(crops, image.dtype)
    return crops


@attr.s(auto_attribs=True)
class InstanceCropper:
    """Data transformer to crop and generate individual examples for instances.

    This generates datasets that are instance cropped for topdown processing.

    Attributes:
        crop_width: Width of the crops in pixels.
        crop_height: Height of the crops in pixels.
        keep_full_image: If True, the output examples will contain the full images
            provided as input to the instance cropped. This can be useful for pipelines
            that use both full and cropped images, at the cost of increased memory
            requirements usage. Setting this to False can substantially improve
            performance of large pipelines if the full images are no longer required.
        mock_centroid_confidence: If True, add confidence keys for compatibility with
            predicted instance cropping.
        unbatch: If True (the default), split frame-level examples into multiple
            instance-level examples. If False, all instance crops will be kept within
            the same example. Use this when building pipelines that require knowledge
            about all instances within a single example.
        image_key: Name of the example key where the image is stored. Defaults to
            "image".
        instances_key: Name of the example key where the instance points are stored.
            Defaults to "instances".
        centroids_key: Name of the example key where the instance centroids are stored.
            Defaults to "centroids".
    """

    crop_width: int
    crop_height: int
    keep_full_image: bool = False
    mock_centroid_confidence: bool = False
    unbatch: bool = True
    image_key: Text = "image"
    instances_key: Text = "instances"
    centroids_key: Text = "centroids"

    @classmethod
    def from_config(
        cls, config: InstanceCroppingConfig, crop_size: Optional[int] = None
    ) -> "InstanceCropper":
        """Build an instance of this class from its configuration options.

        Args:
            config: An `InstanceCroppingConfig` instance with the desired parameters.
            crop_size: Integer specifying the crop height and width. This is only
                required and will only be used if the `config.crop_size` attribute does
                not specify an explicit integer crop size (e.g., it is set to None).

        Returns:
            An instance of this class.

        Raises:
            ValueError: If the `crop_size` is not specified in either the config
                attribute or function arguments.
        """
        if isinstance(config.crop_size, int):
            crop_size = config.crop_size

        if not isinstance(crop_size, int):
            raise ValueError(
                "Crop size not specified in config and not provided in the arguments."
            )

        return cls(crop_width=crop_size, crop_height=crop_size, keep_full_image=False)

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [self.image_key, self.instances_key, self.centroids_key]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        output_keys = [
            "instance_image",
            "bbox",
            "center_instance",
            "center_instance_ind",
            "track_ind",
            "all_instances",
            "centroid",
            "full_image_height",
            "full_image_width",
        ]
        if self.keep_full_image:
            output_keys.append(self.image_key)
        if self.mock_centroid_confidence:
            output_keys.append("centroid_confidence")
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains instance cropped data.

        Args:
            ds_input: A dataset with examples containing the following keys:
                "image": The full image in a tensor of shape (height, width, channels).
                "instances": Instance points in a tf.float32 tensor of shape
                    (n_instances, n_nodes, 2).
                "centroids": The computed centroid for each instance in a tf.float32
                    tensor of shape (n_instances, 2).
                "track_inds": The track indices of the indices if available.
                Any additional keys present will be replicated in each output.

        Returns:
            A `tf.data.Dataset` with elements containing instance cropped data. Each
            instance will generate an example, so the total number of elements may
            change relative to the input dataset.

            Each element in the output dataset will have the following keys:
                "instance_image": A cropped image of the same dtype as the input image
                    but with shape (crop_width, crop_height, channels) and will be
                    centered on an instance.
                "bbox": The bounding box in absolute image coordinates in the format
                    (y1, x1, y2, x2) that resulted in the cropped image in
                    "instance_image". This will be a tf.float32 tensor of shape (4,).
                "center_instance": The points of the centered instance in image
                    coordinates in the "instance_image". This will be a tf.float32
                    tensor of shape (n_nodes, 2). The absolute image coordinates can be
                    recovered by adding (x1, y1) from the "bbox" key.
                "center_instance_ind": Scalar tf.int32 index of the centered instance
                    relative to all the instances in the frame. This can be used to
                    index into additional keys that may contain data from all instances.
                "track_ind": Index of the track the instance belongs to if available.
                "all_instances": The points of all instances in the frame in image
                    coordinates in the "instance_image". This will be a tf.float32
                    tensor of shape (n_instances, n_nodes, 2). This is useful for multi-
                    stage models that first predict all nodes and subsequently refine it
                    to just the centered instance. The "center_instance_ind"-th row of
                    this tensor is equal to "center_instance".
                "centroid": The centroid coordinate that was used to generate this crop,
                    specified as a tf.float32 tensor of shape (2,) in absolute image
                    coordinates.
                "full_image_height": The height of the full image from which the crop
                    was generated, specified as a scalar tf.int32 tensor.
                "full_image_width": The width of the full image from which the crop was
                    generated, specified as a scalar tf.int32 tensor.

            If `keep_full_image` is True, examples will also have an "image" key
            containing the same image as the input.

            if `mock_centroid_confidence` is True, examples will also have a
            "centroid_confidence" key with all ones. This is useful for evaluating
            models that use crops independently from centroid inference.

            Additional keys will be replicated in each example under the same name.
        """
        # Draw a test example from the input dataset to find extra keys to replicate.
        test_example = next(iter(input_ds))
        keys_to_expand = [
            key for key in test_example.keys() if key not in self.input_keys
        ]
        if "class_vectors" in keys_to_expand:
            keys_to_expand.remove("class_vectors")
        img_channels = test_example[self.image_key].shape[-1]
        if self.keep_full_image:
            keys_to_expand.append(self.image_key)

        def crop_instances(frame_data):
            """Local processing function for dataset mapping."""
            # Make bounding boxes from centroids.
            bboxes = make_centered_bboxes(
                frame_data[self.centroids_key],
                box_height=self.crop_height,
                box_width=self.crop_width,
            )

            # Crop images from bounding boxes.
            instance_images = crop_bboxes(frame_data[self.image_key], bboxes)

            # Ensure shape is statically specified.
            instance_images = tf.ensure_shape(
                instance_images, [None, self.crop_height, self.crop_width, img_channels]
            )

            # Pull out the bbox offsets as (n_instances, 2) in xy order.
            bboxes_x1y1 = tf.gather(bboxes, [1, 0], axis=1)

            # Expand the instance points to (n_instances, n_instances, n_nodes, 2).
            n_instances = tf.shape(bboxes)[0]
            all_instances = tf.repeat(
                tf.expand_dims(frame_data[self.instances_key], axis=0),
                n_instances,
                axis=0,
            )

            # Subtract offsets such that each row is relative to an instance.
            all_instances = all_instances - tf.reshape(
                bboxes_x1y1, [n_instances, 1, 1, 2]
            )

            # Pull out the centered instance from each row as (n_instances, n_nodes, 2).
            center_instances = tf.gather_nd(
                all_instances,
                tf.stack([tf.range(n_instances), tf.range(n_instances)], axis=1),
            )

            # Create multi-instance example.
            instances_data = {
                "instance_image": instance_images,
                "bbox": bboxes,
                "center_instance": center_instances,
                "center_instance_ind": tf.range(n_instances, dtype=tf.int32),
                "track_ind": frame_data["track_inds"],
                "all_instances": all_instances,
                "centroid": frame_data[self.centroids_key],
                "full_image_height": tf.repeat(
                    tf.shape(frame_data[self.image_key])[0], n_instances
                ),
                "full_image_width": tf.repeat(
                    tf.shape(frame_data[self.image_key])[1], n_instances
                ),
            }
            if "class_vectors" in frame_data:
                instances_data["class_vectors"] = frame_data["class_vectors"]
            if self.mock_centroid_confidence:
                instances_data["centroid_confidence"] = tf.ones(
                    [n_instances], dtype=tf.float32
                )  # (n_instances,)
            for key in keys_to_expand:
                instances_data[key] = tf.repeat(
                    tf.expand_dims(frame_data[key], axis=0), n_instances, axis=0
                )
            return instances_data

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            crop_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if self.unbatch:
            # Unbatch to split frame-level examples into individual instance-level
            # examples.
            output_ds = output_ds.unbatch()

        return output_ds


@attr.s(auto_attribs=True)
class PredictedInstanceCropper:

    crop_width: int
    crop_height: int
    centroids_key: Text = "predicted_centroids"
    centroid_confidences_key: Text = "predicted_centroid_confidences"
    full_image_key: Text = "full_image"
    full_image_scale_key: Text = "full_image_scale"
    other_keys_to_keep: List[Text] = attr.ib(factory=list)
    keep_instances_gt: bool = False

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        input_keys = [
            self.full_image_key,
            self.full_image_scale_key,
            self.centroids_key,
            self.centroid_confidences_key,
            "video_ind",
            "frame_ind",
        ]
        if self.keep_instances_gt:
            input_keys.append("instances")
        return input_keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        output_keys = [
            "instance_image",
            "bbox",
            "center_instance_ind",
            "centroid",
            "centroid_confidence",
            "full_image_height",
            "full_image_width",
            "scale",
            "video_ind",
            "frame_ind",
        ]
        if self.other_keys_to_keep:
            output_keys.extend(self.other_keys_to_keep)
        if self.keep_instances_gt:
            output_keys.append("instances")
        return output_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a dataset that contains instance cropped data."""
        keys_to_expand = ["scale", "video_ind", "frame_ind"]

        if self.other_keys_to_keep:
            keys_to_expand.extend(self.other_keys_to_keep)
        if self.keep_instances_gt:
            keys_to_expand.append("instances")

        def crop_instances(frame_data):
            """Local processing function for dataset mapping."""
            # Make bounding boxes from centroids.
            full_centroids = frame_data[self.centroids_key] / frame_data["scale"]
            full_centroids = full_centroids * frame_data[self.full_image_scale_key]
            bboxes = make_centered_bboxes(
                full_centroids, box_height=self.crop_height, box_width=self.crop_width
            )

            frame_data["scale"] = frame_data[self.full_image_scale_key]

            # Crop images from bounding boxes.
            instance_images = crop_bboxes(frame_data[self.full_image_key], bboxes)
            n_instances = tf.shape(bboxes)[0]

            # Create multi-instance example.
            instances_data = {
                "instance_image": instance_images,
                "bbox": bboxes,
                "center_instance_ind": tf.range(n_instances, dtype=tf.int32),
                "centroid": full_centroids,
                "centroid_confidence": frame_data[self.centroid_confidences_key],
                "full_image_height": tf.repeat(
                    tf.shape(frame_data[self.full_image_key])[0], n_instances
                ),
                "full_image_width": tf.repeat(
                    tf.shape(frame_data[self.full_image_key])[1], n_instances
                ),
            }
            for key in keys_to_expand:
                instances_data[key] = tf.repeat(
                    tf.expand_dims(frame_data[key], axis=0), n_instances, axis=0
                )
            return instances_data

        # Map the main processing function to each example.
        output_ds = input_ds.map(
            crop_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Unbatch to split frame-level examples into individual instance-level examples.
        output_ds = output_ds.unbatch()

        return output_ds

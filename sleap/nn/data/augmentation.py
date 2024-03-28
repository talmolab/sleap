"""Transformers for applying data augmentation."""

import sleap
import numpy as np
import tensorflow as tf
import attr
from typing import List, Text, Optional
import albumentations as A
from sleap.nn.config import AugmentationConfig
from sleap.nn.data.instance_cropping import crop_bboxes


def flip_instances_lr(
    instances: tf.Tensor, img_width: int, symmetric_inds: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Flip a set of instance points horizontally with symmetric node adjustment.

    Args:
        instances: Instance points as a `tf.Tensor` of shape `(n_instances, n_nodes, 2)`
            and dtype `tf.float32`.
        img_width: Width of image in the same units as `instances`.
        symmetric_inds: Indices of symmetric pairs of nodes as a `tf.Tensor` of shape
            `(n_symmetries, 2)` and dtype `tf.int32`. Each row contains the indices of
            nodes that are mirror symmetric, e.g., left/right body parts. The ordering
            of the list or which node comes first (e.g., left/right vs right/left) does
            not matter. Each pair of nodes will be swapped to account for the
            reflection if this is not `None` (the default).

    Returns:
        The instance points with x-coordinates flipped horizontally.
    """
    instances = (tf.cast([[[img_width - 1, 0]]], tf.float32) - instances) * tf.cast(
        [[[1, -1]]], tf.float32
    )

    if symmetric_inds is not None:
        n_instances = tf.shape(instances)[0]
        n_symmetries = tf.shape(symmetric_inds)[0]

        inst_inds = tf.reshape(tf.repeat(tf.range(n_instances), n_symmetries), [-1, 1])
        sym_inds1 = tf.reshape(tf.gather(symmetric_inds, 0, axis=1), [-1, 1])
        sym_inds2 = tf.reshape(tf.gather(symmetric_inds, 1, axis=1), [-1, 1])

        inst_inds = tf.cast(inst_inds, tf.int32)
        sym_inds1 = tf.cast(sym_inds1, tf.int32)
        sym_inds2 = tf.cast(sym_inds2, tf.int32)

        subs1 = tf.concat([inst_inds, tf.tile(sym_inds1, [n_instances, 1])], axis=1)
        subs2 = tf.concat([inst_inds, tf.tile(sym_inds2, [n_instances, 1])], axis=1)

        pts1 = tf.gather_nd(instances, subs1)
        pts2 = tf.gather_nd(instances, subs2)
        instances = tf.tensor_scatter_nd_update(instances, subs1, pts2)
        instances = tf.tensor_scatter_nd_update(instances, subs2, pts1)

    return instances


def flip_instances_ud(
    instances: tf.Tensor, img_height: int, symmetric_inds: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Flip a set of instance points vertically with symmetric node adjustment.

    Args:
        instances: Instance points as a `tf.Tensor` of shape `(n_instances, n_nodes, 2)`
            and dtype `tf.float32`.
        img_height: Height of image in the same units as `instances`.
        symmetric_inds: Indices of symmetric pairs of nodes as a `tf.Tensor` of shape
            `(n_symmetries, 2)` and dtype `tf.int32`. Each row contains the indices of
            nodes that are mirror symmetric, e.g., left/right body parts. The ordering
            of the list or which node comes first (e.g., left/right vs right/left) does
            not matter. Each pair of nodes will be swapped to account for the
            reflection if this is not `None` (the default).

    Returns:
        The instance points with y-coordinates flipped horizontally.
    """
    instances = (tf.cast([[[0, img_height - 1]]], tf.float32) - instances) * tf.cast(
        [[[-1, 1]]], tf.float32
    )

    if symmetric_inds is not None:
        n_instances = tf.shape(instances)[0]
        n_symmetries = tf.shape(symmetric_inds)[0]

        inst_inds = tf.reshape(tf.repeat(tf.range(n_instances), n_symmetries), [-1, 1])
        sym_inds1 = tf.reshape(tf.gather(symmetric_inds, 0, axis=1), [-1, 1])
        sym_inds2 = tf.reshape(tf.gather(symmetric_inds, 1, axis=1), [-1, 1])

        inst_inds = tf.cast(inst_inds, tf.int32)
        sym_inds1 = tf.cast(sym_inds1, tf.int32)
        sym_inds2 = tf.cast(sym_inds2, tf.int32)

        subs1 = tf.concat([inst_inds, tf.tile(sym_inds1, [n_instances, 1])], axis=1)
        subs2 = tf.concat([inst_inds, tf.tile(sym_inds2, [n_instances, 1])], axis=1)

        pts1 = tf.gather_nd(instances, subs1)
        pts2 = tf.gather_nd(instances, subs2)
        instances = tf.tensor_scatter_nd_update(instances, subs1, pts2)
        instances = tf.tensor_scatter_nd_update(instances, subs2, pts1)

    return instances


@attr.s(auto_attribs=True)
class AlbumentationsAugmenter:
    """Data transformer based on the `albumentations` library.

    This class can generate a `tf.data.Dataset` from an existing one that generates
    image and instance data. Element of the output dataset will have a set of
    augmentation transformations applied.

    Attributes:
        augmenter: An instance of `albumentations.Compose` that will be applied to
            each element of the input dataset.
        image_key: Name of the example key where the image is stored. Defaults to
            "image".
        instances_key: Name of the example key where the instance points are stored.
            Defaults to "instances".
    """

    augmenter: A.Compose
    image_key: str = "image"
    instances_key: str = "instances"

    @classmethod
    def from_config(
        cls,
        config: AugmentationConfig,
        image_key: Text = "image",
        instances_key: Text = "instances",
    ) -> "AlbumentationsAugmenter":
        """Create an augmenter from a set of configuration parameters.

        Args:
            config: An `AugmentationConfig` instance with the desired parameters.
            image_key: Name of the example key where the image is stored. Defaults to
                "image".
            instances_key: Name of the example key where the instance points are stored.
                Defaults to "instances".

        Returns:
            An instance of `AlbumentationsAugmenter` with the specified augmentation
            configuration.
        """
        aug_stack = []
        if config.rotate:
            aug_stack.append(
                A.Rotate(
                    limit=(config.rotation_min_angle, config.rotation_max_angle), p=1.0
                )
            )
        if config.translate:
            aug_stack.append(
                A.Affine(
                    translate_px={
                        "x": (config.translate_min, config.translate_max),
                        "y": (config.translate_min, config.translate_max),
                    },
                    p=1.0,
                )
            )
        if config.scale:
            aug_stack.append(
                A.Affine(scale=(config.scale_min, config.scale_max), p=1.0)
            )
        if config.uniform_noise:

            def uniform_noise(image, **kwargs):
                return image + np.random.uniform(
                    config.uniform_noise_min_val, config.uniform_noise_max_val
                )

            aug_stack.append(A.Lambda(image=uniform_noise))
        if config.gaussian_noise:
            aug_stack.append(
                A.GaussNoise(
                    mean=config.gaussian_noise_mean,
                    var_limit=config.gaussian_noise_stddev,
                )
            )
        if config.contrast:
            aug_stack.append(
                A.RandomGamma(
                    gamma_limit=(config.contrast_min_gamma, config.contrast_max_gamma),
                    p=1.0,
                )
            )
        if config.brightness:
            aug_stack.append(
                A.RandomBrightness(
                    limit=(config.brightness_min_val, config.brightness_max_val), p=1.0
                )
            )

        return cls(
            augmenter=A.Compose(
                aug_stack,
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            ),
            image_key=image_key,
            instances_key=instances_key,
        )

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return [self.image_key, self.instances_key]

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return self.input_keys

    def transform_dataset(self, input_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Create a `tf.data.Dataset` with elements containing augmented data.

        Args:
            input_ds: A dataset with elements that contain the keys "image" and
                "instances". This is typically raw data from a data provider.

        Returns:
            A `tf.data.Dataset` with the same keys as the input, but with images and
            instance points updated with the applied augmentations.

        Notes:
            The "scale" key in examples are not modified when scaling augmentation is
            applied.
        """
        # Define augmentation function to map over each sample.
        def py_augment(image, instances):
            """Local processing function that will not be autographed."""
            # Convert to numpy arrays.
            img = image.numpy()
            kps = instances.numpy()
            original_shape = kps.shape
            kps = kps.reshape(-1, 2)

            # Augment.
            augmented = self.augmenter(image=img, keypoints=kps)
            aug_img = augmented["image"]
            aug_instances = np.array(augmented["keypoints"]).reshape(original_shape)

            return aug_img, aug_instances

        def augment(frame_data):
            """Wrap local processing function for dataset mapping."""
            image, instances = tf.py_function(
                py_augment,
                [frame_data["image"], frame_data["instances"]],
                [frame_data["image"].dtype, frame_data["instances"].dtype],
            )
            image.set_shape(frame_data["image"].get_shape())
            instances.set_shape(frame_data["instances"].get_shape())
            frame_data.update({"image": image, "instances": instances})
            return frame_data

        # Apply the augmentation to each element.
        output_ds = input_ds.map(augment)

        return output_ds


@attr.s(auto_attribs=True)
class RandomCropper:
    """Data transformer for applying random crops to input images.

    This class can generate a `tf.data.Dataset` from an existing one that generates
    image and instance data. Element of the output dataset will have random crops
    applied.

    Attributes:
        crop_height: The height of the cropped region in pixels.
        crop_width: The width of the cropped region in pixels.
    """

    crop_height: int = 256
    crop_width: int = 256

    @property
    def input_keys(self):
        return ["image", "instances"]

    @property
    def output_keys(self):
        return ["image", "instances", "crop_bbox"]

    def transform_dataset(self, input_ds: tf.data.Dataset):
        """Create a `tf.data.Dataset` with elements containing augmented data.

        Args:
            input_ds: A dataset with elements that contain the keys "image" and
                "instances". This is typically raw data from a data provider.

        Returns:
            A `tf.data.Dataset` with the same keys as the input, but with images and
            instance points updated with the applied random crop.

            Additionally, the `"crop_bbox"` key will contain the bounding box of the
            crop in the form `[y1, x1, y2, x2]`.
        """

        def random_crop(ex):
            """Apply random crop to an example."""
            # Generate random value for the top-left of the crop.
            img_width = tf.shape(ex["image"])[1]
            img_height = tf.shape(ex["image"])[0]
            dx = tf.random.uniform(
                (), minval=0, maxval=tf.cast(img_width - self.crop_width, tf.float32)
            )
            dy = tf.random.uniform(
                (), minval=0, maxval=tf.cast(img_height - self.crop_height, tf.float32)
            )
            ex["instances"] = ex["instances"] - tf.reshape(
                tf.stack([dx, dy], axis=0), [1, 1, 2]
            )
            bbox = tf.expand_dims(
                tf.stack(
                    [
                        dy,
                        dx,
                        dy + tf.cast(self.crop_height, tf.float32) - 1,
                        dx + tf.cast(self.crop_width, tf.float32) - 1,
                    ],
                    axis=0,
                ),
                axis=0,
            )
            ex["crop_bbox"] = bbox
            ex["image"] = tf.squeeze(crop_bboxes(ex["image"], bbox), axis=0)
            return ex

        return input_ds.map(random_crop)


@attr.s(auto_attribs=True)
class RandomFlipper:
    """Data transformer for applying random flipping to input images.

    This class can generate a `tf.data.Dataset` from an existing one that generates
    image and instance data. Elements of the output dataset will have random horizontal
    flips applied.

    Attributes:
        symmetric_inds: Indices of symmetric pairs of nodes as a an array of shape
            `(n_symmetries, 2)`. Each row contains the indices of nodes that are mirror
            symmetric, e.g., left/right body parts. The ordering of the list or which
            node comes first (e.g., left/right vs right/left) does not matter. Each pair
            of nodes will be swapped to account for the reflection if this is not `None`
            (the default).
        horizontal: If `True` (the default), flips are applied horizontally instead of
            vertically.
        probability: The probability that the augmentation should be applied.
    """

    symmetric_inds: Optional[np.ndarray] = None
    horizontal: bool = True
    probability: float = 0.5

    @classmethod
    def from_skeleton(
        cls, skeleton: sleap.Skeleton, horizontal: bool = True, probability: float = 0.5
    ) -> "RandomFlipper":
        """Create an instance of `RandomFlipper` from a skeleton.

        Args:
            skeleton: A `sleap.Skeleton` that may define symmetric nodes.
            horizontal: If `True` (the default), flips are applied horizontally instead
                of vertically.
            probability: The probability that the augmentation should be applied.

        Returns:
            An instance of `RandomFlipper`.
        """
        return cls(
            symmetric_inds=skeleton.symmetric_inds,
            horizontal=horizontal,
            probability=probability,
        )

    @property
    def input_keys(self):
        return ["image", "instances"]

    @property
    def output_keys(self):
        return self.input_keys

    def transform_dataset(self, input_ds: tf.data.Dataset):
        """Create a `tf.data.Dataset` with elements containing augmented data.

        Args:
            input_ds: A dataset with elements that contain the keys `"image"` and
                `"instances"`. This is typically raw data from a data provider.

        Returns:
            A `tf.data.Dataset` with the same keys as the input, but with images and
            instance points updated with the applied random flip.
        """
        symmetric_inds = self.symmetric_inds
        if symmetric_inds is not None:
            symmetric_inds = np.array(symmetric_inds)
            if len(symmetric_inds) == 0:
                symmetric_inds = None

        def random_flip(ex):
            """Apply random flip to an example."""
            p = tf.random.uniform((), minval=0, maxval=1.0)
            if p <= self.probability:
                if self.horizontal:
                    img_width = tf.shape(ex["image"])[1]
                    ex["instances"] = flip_instances_lr(
                        ex["instances"], img_width, symmetric_inds=symmetric_inds
                    )
                    ex["image"] = tf.image.flip_left_right(ex["image"])
                else:
                    img_height = tf.shape(ex["image"])[0]
                    ex["instances"] = flip_instances_ud(
                        ex["instances"], img_height, symmetric_inds=symmetric_inds
                    )
                    ex["image"] = tf.image.flip_up_down(ex["image"])
            return ex

        return input_ds.map(random_flip)

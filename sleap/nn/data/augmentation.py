"""Transformers for applying data augmentation."""

# Monkey patch for: https://github.com/aleju/imgaug/issues/537
# TODO: Fix when PyPI/conda packages are available for version fencing.
import numpy

if hasattr(numpy.random, "_bit_generator"):
    numpy.random.bit_generator = numpy.random._bit_generator

import numpy as np
import tensorflow as tf
import attr
from typing import List, Text
import imgaug as ia
import imgaug.augmenters as iaa
from sleap.nn.config import AugmentationConfig


@attr.s(auto_attribs=True)
class ImgaugAugmenter:
    """Data transformer based on the `imgaug` library.

    This class can generate a `tf.data.Dataset` from an existing one that generates
    image and instance data. Element of the output dataset will have a set of
    augmentation transformations applied.

    Attributes:
        augmenter: An instance of `imgaug.augmenters.Sequential` that will be applied to
            each element of the input dataset.
    """

    augmenter: iaa.Sequential

    @classmethod
    def from_config(cls, config: AugmentationConfig) -> "ImgaugAugmenter":
        """Create an augmenter from a set of configuration parameters.

        Args:
            config: An `AugmentationConfig` instance with the desired parameters.

        Returns:
            An instance of this class with the specified augmentation configuration.
        """
        aug_stack = []
        if config.rotate:
            aug_stack.append(
                iaa.Affine(
                    rotate=(config.rotation_min_angle, config.rotation_max_angle)
                )
            )
        if config.translate:
            aug_stack.append(
                iaa.Affine(
                    translate_px={
                        "x": (config.translate_min, config.translate_max),
                        "y": (config.translate_min, config.translate_max),
                    }
                )
            )
        if config.scale:
            aug_stack.append(iaa.Affine(scale=(config.scale_min, config.scale_max)))
        if config.uniform_noise:
            aug_stack.append(
                iaa.AddElementwise(
                    value=(config.uniform_noise_min_val, config.uniform_noise_max_val)
                )
            )
        if config.gaussian_noise:
            aug_stack.append(
                iaa.AdditiveGaussianNoise(
                    loc=config.gaussian_noise_mean, scale=config.gaussian_noise_stddev
                )
            )
        if config.contrast:
            aug_stack.append(
                iaa.GammaContrast(
                    gamma=(config.contrast_min_gamma, config.contrast_max_gamma)
                )
            )
        if config.brightness:
            aug_stack.append(
                iaa.Add(value=(config.brightness_min_val, config.brightness_max_val))
            )

        return cls(augmenter=iaa.Sequential(aug_stack))

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return ["image", "instances"]

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
            # Ensure that the transformations applied to all data within this
            # example are kept consistent.
            aug_det = self.augmenter.to_deterministic()

            # Augment the image.
            aug_img = aug_det.augment_image(image.numpy())

            # Augment each set of points for each instance.
            aug_instances = []
            for instance in instances:
                kps = ia.KeypointsOnImage.from_xy_array(
                    instance.numpy(), tuple(image.shape)
                )
                aug_instance = aug_det.augment_keypoints(kps).to_xy_array()
                aug_instances.append(aug_instance)

            # Convert the results to tensors.
            # aug_img = tf.convert_to_tensor(aug_img, dtype=image.dtype)

            # This will get converted to a rank 3 tensor (n_instances, n_nodes, 2).
            aug_instances = np.stack(aug_instances, axis=0)
            # aug_instances = [
            #     tf.convert_to_tensor(x, dtype=instances.dtype) for x in aug_instances
            # ]

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
        # Note: We map sequentially since imgaug gets slower with tf.data parallelism.
        output_ds = input_ds.map(augment)

        return output_ds

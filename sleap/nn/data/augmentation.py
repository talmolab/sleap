"""Transformers for applying data augmentation."""

import tensorflow as tf
import attr
from typing import List, Text
import imgaug as ia
import imgaug.augmenters as iaa


@attr.s(auto_attribs=True)
class AugmentationConfig:
    """Parameters for configuring an augmentation stack.

    The augmentations will be applied in the the order of the attributes.

    Attributes:
        rotate: If True, rotational augmentation will be applied. Rotation is relative
            to the center of the image. See `imgaug.augmenters.geometric.Affine`.
        rotation_min_angle: Minimum rotation angle in degrees in [-180, 180].
        rotation_max_angle: Maximum rotation angle in degrees in [-180, 180].
        translate: If True, translational augmentation will be applied. The values are
            sampled independently for x and y coordinates. See
            `imgaug.augmenters.geometric.Affine`.
        translate_min: Minimum translation in integer pixel units.
        translate_max: Maximum translation in integer pixel units.
        scale: If True, scaling augmentation will be applied. See
            `imgaug.augmenters.geometric.Affine`.
        scale_min: Minimum scaling factor.
        scale_max: Maximum scaling factor.
        uniform_noise: If True, uniformly distributed noise will be added to the image.
            This is effectively adding a different random value to each pixel to
            simulate shot noise. See `imgaug.augmenters.arithmetic.AddElementwise`.
        uniform_noise_min_val: Minimum value to add.
        uniform_noise_max_val: Maximum value to add.
        gaussian_noise: If True, normally distributed noise will be added to the image.
            This is similar to uniform noise, but can provide a tigher bound around a
            mean noise magnitude. This is applied independently to each pixel.
            See `imgaug.augmenters.arithmetic.AdditiveGaussianNoise`.
        gaussian_noise_mean: Mean of the distribution to sample from.
        gaussian_noise_stddev: Standard deviation of the distribution to sample from.
        contrast: If True, gamma constrast adjustment will be applied to the image.
            This scales all pixel values by `x ** gamma` where `x` is the pixel value in
            the [0, 1] range. Values in [0, 255] are first scaled to [0, 1]. See
            `imgaug.augmenters.contrast.GammaContrast`.
        contrast_min_gamma: Minimum gamma to use for augmentation. Reasonable values are
            in [0.5, 2.0].
        contrast_max_gamma: Maximum gamma to use for augmentation. Reasonable values are
            in [0.5, 2.0].
        brightness: If True, the image brightness will be augmented. This adjustment
            simply adds the same value to all pixels in the image to simulate broadfield
            illumination change. See `imgaug.augmenters.arithmetic.Add`.
        brightness_min_val: Minimum value to add to all pixels.
        brightness_max_val: Maximum value to add to all pixels.
    """

    rotate: bool = False
    rotation_min_angle: float = -180
    rotation_max_angle: float = 180
    translate: bool = False
    translate_min: int = -5
    translate_max: int = 5
    scale: bool = False
    scale_min: float = 0.9
    scale_max: float = 1.1
    uniform_noise: bool = False
    uniform_noise_min_val: float = 0.0
    uniform_noise_max_val: float = 10.0
    gaussian_noise: bool = False
    gaussian_noise_mean: float = 5.0
    gaussian_noise_stddev: float = 1.0
    contrast: bool = False
    contrast_min_gamma: float = 0.5
    contrast_max_gamma: float = 2.0
    brightness: bool = False
    brightness_min_val: float = 0.0
    brightness_max_val: float = 10.0

    def make_iaa_augmenter(self) -> iaa.Sequential:
        """Create a sequential `imgaug` augmenter with the specified parameters.
        
        Returns:
            An instance of `imgaug.augmenters.Sequential` with the specified
            augmentation operations.
        """
        aug_stack = []
        if self.rotate:
            aug_stack.append(
                iaa.Affine(rotate=(self.rotation_min_angle, self.rotation_max_angle))
            )
        if self.translate:
            aug_stack.append(
                iaa.Affine(
                    translate_px={
                        "x": (self.translate_min, self.translate_max),
                        "y": (self.translate_min, self.translate_max),
                    }
                )
            )
        if self.scale:
            aug_stack.append(iaa.Affine(scale=(self.scale_min, self.scale_max)))
        if self.uniform_noise:
            aug_stack.append(
                iaa.AddElementwise(
                    value=(self.uniform_noise_min_noise, self.uniform_noise_max_val)
                )
            )
        if self.gaussian_noise:
            aug_stack.append(
                iaa.AdditiveGaussianNoise(
                    loc=self.gaussian_noise_mean, scale=self.gaussian_noise_stddev
                )
            )
        if self.contrast:
            aug_stack.append(
                iaa.GammaContrast(
                    gamma=(self.contrast_min_gamma, self.contrast_max_gamma)
                )
            )
        if self.brightness:
            aug_stack.append(
                iaa.Add(value=(self.brightness_min_val, self.brightness_max_val))
            )
        return iaa.Sequential(aug_stack)


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
    def from_config(cls, augmentation_config: AugmentationConfig) -> "ImgaugAugmenter":
        """Create an augmenter from a set of configuration parameters.
        
        Args:
            augmentation_config: Instance of `AugmentationConfig` that can instantiate
                an imgaug sequential augmenter.
        
        Returns:
            An instance of this class with the specified augmentation configuration.
        """
        return cls(augmenter=augmentation_config.make_iaa_augmenter())

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
            aug_img = aug_det.augment_image(image)

            # Augment each set of points for each instance.
            aug_instances = []
            for instance in instances:
                kps = ia.KeypointsOnImage.from_xy_array(instance, tuple(image.shape))
                aug_instance = aug_det.augment_keypoints(kps).to_xy_array()
                aug_instances.append(aug_instance)

            # Convert the results to tensors.
            aug_img = tf.convert_to_tensor(aug_img, dtype=image.dtype)

            # This will get converted to a rank 3 tensor (n_instances, n_nodes, 2).
            aug_instances = [
                tf.convert_to_tensor(x, dtype=instances.dtype) for x in aug_instances
            ]

            return aug_img, aug_instances

        def augment(frame_data):
            """Wrap local processing function for dataset mapping."""
            image, instances = tf.py_function(
                py_augment,
                [frame_data["image"], frame_data["instances"]],
                [frame_data["image"].dtype, frame_data["instances"].dtype],
            )
            frame_data.update({"image": image, "instances": instances})
            return frame_data

        # Apply the augmentation to each element.
        # Note: We map sequentially since imgaug gets slower with tf.data parallelism.
        output_ds = input_ds.map(augment)

        return output_ds

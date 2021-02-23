import attr
from typing import Optional, Text


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
        random_crop: If `True`, performs random crops on the image. This is useful for
            training efficiently on large resolution images, but may fail to learn
            global structure beyond the crop size. Random cropping will be applied after
            the augmentations above.
        random_crop_width: Width of random crops.
        random_crop_height: Height of random crops.
        random_flip: If `True`, images will be randomly reflected. The coordinates of
            the instances will be adjusted accordingly. Body parts that are left/right
            symmetric must be marked on the skeleton in order to be swapped correctly.
        flip_horizontal: If `True`, flip images left/right when randomly reflecting
            them. If `False`, flipping is down up/down instead.
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
    random_crop: bool = False
    random_crop_height: int = 256
    random_crop_width: int = 256
    random_flip: bool = False
    flip_horizontal: bool = True


@attr.s(auto_attribs=True)
class HardKeypointMiningConfig:
    """Configuration for online hard keypoint mining.

    Attributes:
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When
            this is enabled, the loss is computed per keypoint (or edge for PAFs) and
            sorted from lowest (easy) to highest (hard). The hard keypoint loss will be
            scaled to have a higher weight in the total loss, encouraging the training
            to focus on tricky body parts that are more difficult to learn.
            If False, no mining will be performed and all keypoints will be weighted
            equally in the loss.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with
            respect to the lowest keypoint loss in order to be considered as "hard".
            This helps to switch focus on across groups of keypoints during training.
        min_hard_keypoints: The minimum number of keypoints that will be considered as
            "hard", even if they are not below the `hard_to_easy_ratio`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to.
            This can help when there are few very easy keypoints which may skew the
            ratio and result in loss scaling being applied to most keypoints, which can
            reduce the impact of hard mining altogether.
        loss_scale: Factor to scale the hard keypoint losses by.
    """

    online_mining: bool = False
    hard_to_easy_ratio: float = 2.0
    min_hard_keypoints: int = 2
    max_hard_keypoints: Optional[int] = None
    loss_scale: float = 5.0


@attr.s(auto_attribs=True)
class LearningRateScheduleConfig:
    """Configuration for learning rate scheduling.

    Attributes:
        reduce_on_plateau: If True, learning rate will be reduced when the validation
            set loss plateaus. This improves training at later epochs when finer weight
            updates are required for fine-tuning the optimization, balancing out an
            initially high learning rate required for practical initial optimization.
        reduction_factor: Factor by which the learning rate will be scaled when a
            plateau is detected.
        plateau_min_delta: Minimum absolute decrease in the loss in order to consider an
            epoch as not in a plateau.
        plateau_patience: Number of epochs without an improvement of at least
            `plateau_min_delta` in order for a plateau to be detected.
        plateau_cooldown: Number of epochs after a reduction step before epochs without
            improvement will begin to be counted again.
        min_learning_rate: The minimum absolute value that the learning rate can be
            reduced to.
    """

    reduce_on_plateau: bool = True
    reduction_factor: float = 0.5
    plateau_min_delta: float = 1e-6
    plateau_patience: int = 5
    plateau_cooldown: int = 3
    min_learning_rate: float = 1e-8


@attr.s(auto_attribs=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        stop_training_on_plateau: If True, the training will terminate automatically
            when the validation set loss plateaus. This can save time and compute
            resources when there are minimal improvements to be gained from further
            training, as well as to prevent training into the overfitting regime.
        plateau_min_delta: Minimum absolute decrease in the loss in order to consider an
            epoch as not in a plateau.
        plateau_patience: Number of epochs without an improvement of at least
            `plateau_min_delta` in order for a plateau to be detected.
    """

    stop_training_on_plateau: bool = True
    plateau_min_delta: float = 1e-6
    plateau_patience: int = 10


@attr.s(auto_attribs=True)
class OptimizationConfig:
    """Optimization configuration.

    Attributes:
        preload_data: If True, the data from the training/validation/test labels will be
            loaded into memory at the beginning of training. If False, the data will be
            loaded every time it is accessed. Preloading can considerably speed up the
            performance of data generation at the cost of memory as all the images will
            be loaded into memory. This is especially beneficial for datasets with few
            examples or when the raw data is on (slow) network storage.
        augmentation_config: Configuration options related to data augmentation.
        online_shuffling: If True, data will be shuffled online by maintaining a buffer
            of examples that are sampled from at each step. This allows for
            randomization of data ordering, resulting in more varied mini-batch
            composition, which in turn can promote generalization. Note that the data
            are shuffled at the start of training regardless.
        shuffle_buffer_size: Number of examples to keep in a buffer to sample uniformly
            from. This should be set to relatively low number as it requires an
            additional copy of each data example to be stored in memory. If set to -1,
            the entire dataset will be used, which results in perfect randomization at
            the cost of increased memory usage.
        prefetch: If True, data will generated in parallel to training to minimize the
            bottleneck of the preprocessing pipeline.
        batch_size: Number of examples per minibatch, i.e., a single step of training.
            Higher numbers can increase generalization performance by averaging model
            gradient updates over a larger number of examples at the cost of
            considerably more GPU memory, especially for larger sized images. Lower
            numbers may lead to overfitting, but may be beneficial to the optimization
            process when few but varied examples are available.
        batches_per_epoch: Number of minibatches (steps) to train for in an epoch. If
            set to None, this is set to the number of batches in the training data or
            `min_batches_per_epoch`, whichever is largest. At the end of each epoch, the
            validation and test sets are evaluated, the model is saved if its
            performance improved, visualizations are generated, learning rate may be
            tuned, and several other non-optimization procedures executed.
            If this is set too low, training may be slowed down as these end-of-epoch
            procedures can take longer than the optimization itself, especially if
            model saving is enabled, which can take a while for larger models. If set
            too high, the training procedure may take longer, especially since several
            hyperparameter tuning heuristics only consider epoch-to-epoch performance.
        min_batches_per_epoch: The minimum number of batches per epoch if
            `batches_per_epoch` is set to None. No effect if the batches per epoch is
            explicitly specified. This should be set to 200-400 to compensate for short
            loops through the data when there are few examples.
        val_batches_per_epoch: Same as `batches_per_epoch`, but for the validation set.
        min_val_batches_per_epoch: Same as `min_batches_per_epoch`, but for the
            validation set.
        epochs: Maximum number of epochs to train for. Training can be stopped manually
            or automatically if early stopping is enabled and a plateau is detected.
        optimizer: Name of the optimizer to use for training. This is typically "adam"
            but the name of any class from `tf.keras.optimizers` may be used. If "adam"
            is specified, the Adam optimizer will have AMSGrad enabled.
        initial_learning_rate: The initial learning rate to use for the optimizer. This
            is typically set to 1e-3 or 1e-4, and can be decreased automatically if
            learning rate reduction on plateau is enabled. If this is too high or too
            low, the training may fail to find good initial local minima to descend.
        learning_rate_schedule: Configuration options related to learning rate
            scheduling.
        hard_keypoint_mining: Configuration options related to online hard keypoint
            mining.
        early_stopping: Configuration options related to early stopping of training on
            plateau/convergence is detected.
    """

    preload_data: bool = True
    augmentation_config: AugmentationConfig = attr.ib(factory=AugmentationConfig)
    online_shuffling: bool = True
    shuffle_buffer_size: int = 128
    prefetch: bool = True
    batch_size: int = 8
    batches_per_epoch: Optional[int] = None
    min_batches_per_epoch: int = 200
    val_batches_per_epoch: Optional[int] = None
    min_val_batches_per_epoch: int = 10
    epochs: int = 100
    optimizer: Text = "adam"
    initial_learning_rate: float = 1e-4
    learning_rate_schedule: LearningRateScheduleConfig = attr.ib(
        factory=LearningRateScheduleConfig
    )
    hard_keypoint_mining: HardKeypointMiningConfig = attr.ib(
        factory=HardKeypointMiningConfig
    )
    early_stopping: EarlyStoppingConfig = attr.ib(factory=EarlyStoppingConfig)

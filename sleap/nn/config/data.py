import attr
from typing import Optional, Text, List
import sleap


@attr.s(auto_attribs=True)
class LabelsConfig:
    """Labels configuration.

    Attributes:
        training_labels: A filepath to a saved labels file containing user labeled
            frames to use for generating the training set.
        validation_labels: A filepath to a saved labels file containing user labeled
            frames to use for generating validation data. These will not be trained on
            directly, but will be used to tune hyperparameters such as learning rate or
            early stopping. If not specified, the validation set will be sampled from
            the training labels.
        validation_fraction: Float between 0 and 1 specifying the fraction of the
            training set to sample for generating the validation set. The remaining
            labeled frames will be left in the training set. If the `validation_labels`
            are already specified, this has no effect.
        test_labels: A filepath to a saved labels file containing user labeled frames to
            use for generating the test set. This is typically a held out set of
            examples that are never used for training or hyperparameter tuning (like the
            validation set). This is optional, but useful for benchmarking as metrics
            can be computed from these data during model optimization. This is also
            useful to explicitly keep track of the test set that should be used when
            multiple splits are created for training.
        split_by_inds: If `True`, splits used for training will be determined by the
            lists below by indexing into the labels in `training_labels`. If this is
            `False`, the indices below will not be used even if specified. This is
            useful for specifying the fixed split sets from examples within a single
            labels file. If splits are generated automatically (using
            `validation_fraction`), the selected indices are stored below for reference.
        training_inds: List of indices of the training split labels.
        validation_inds: List of indices of the validation split labels.
        test_inds: List of indices of the test split labels.
        search_path_hints: List of paths to use for searching for missing data. This is
            useful when labels and data are moved across computers, network storage, or
            operating systems that may have different absolute paths than those stored
            in the labels. This has no effect if the labels were exported as a package
            with the user labeled data.
        skeletons: List of `sleap.Skeleton` instances that can be used by the model. If
            not specified, these will be pulled out of the labels during training, but
            must be specified for inference in order to generate predicted instances.
    """

    training_labels: Optional[Text] = None
    validation_labels: Optional[Text] = None
    validation_fraction: float = 0.1
    test_labels: Optional[Text] = None
    split_by_inds: bool = False
    training_inds: Optional[List[int]] = None
    validation_inds: Optional[List[int]] = None
    test_inds: Optional[List[int]] = None
    search_path_hints: List[Text] = attr.ib(factory=list)
    skeletons: List[sleap.Skeleton] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class PreprocessingConfig:
    """Preprocessing configuration.

    Attributes:
        ensure_rgb: If True, converts the image to RGB if not already.
        ensure_grayscale: If True, converts the image to grayscale if not already.
        imagenet_mode: Specifies an ImageNet-based normalization mode commonly used in
            `tf.keras.applications`-based pretrained models. This has no effect if None
            or not specified.
            Valid values are:
            "tf": Values will be scaled to [-1, 1], expanded to RGB if grayscale. This
                preprocessing mode is required when using pretrained ResNetV2,
                MobileNetV1, MobileNetV2 and NASNet models.
            "caffe": Values will be scaled to [0, 255], expanded to RGB if grayscale,
                RGB channels flipped to BGR, and subtracted by a fixed mean. This
                preprocessing mode is required when using pretrained ResNetV1 models.
            "torch": Values will be scaled to [0, 1], expanded to RGB if grayscale,
                subtracted by a fixed mean, and scaled by fixed standard deviation. This
                preprocessing mode is required when using pretrained DenseNet models.
        input_scale: Scalar float specifying scaling factor to resize raw images by.
            This can considerably increase performance and memory requirements at the
            cost of accuracy. Generally, it should only be used when the raw images are
            at a much higher resolution than the smallest features in the data.
        pad_to_stride: Number of pixels that the image size must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `input_scale` attribute. If set to None, this will be automatically
            detected from the model architecture. This must be divisible by the model's
            max stride (typically 32). This padding will be ignored when instance
            cropping inputs since the crop size should already be divisible by the
            model's max stride.
        resize_and_pad_to_target: If True, will resize and pad all images in the dataset
            to match target dimensions. This is useful when preprocessing datasets with
            mixed image dimensions (from different video resolutions). Aspect ratio is
            preserved, and padding applied (if needed) to bottom or right of image only.
        target_height: Target image height for 'resize_and_pad_to_target'. When not
            explicitly provided, inferred as the max image height from the dataset.
        target_width: Target image width for 'resize_and_pad_to_target'. When not
            explicitly provided, inferred as the max image width from the dataset.
    """

    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    imagenet_mode: Optional[Text] = attr.ib(
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(["tf", "caffe", "torch"])
        ),
    )
    input_scaling: float = 1.0
    pad_to_stride: Optional[int] = None
    resize_and_pad_to_target: bool = True
    target_height: Optional[int] = None
    target_width: Optional[int] = None


@attr.s(auto_attribs=True)
class InstanceCroppingConfig:
    """Instance cropping configuration.

    These are only used in topdown or centroid models.

    Attributes:
        center_on_part: String name of the part to center the instance to. If None or
            not specified, instances will be centered to the centroid of their bounding
            box. This value will be used for both topdown and centroid models. It must
            match the name of a node on the skeleton.
        crop_size: Integer size of bounding box height and width to crop out of the full
            image. This should be greater than the largest size of the instances in
            pixels. The crop is applied after any input scaling, so be sure to adjust
            this to changes in the input image scale. If set to None, this will be
            automatically detected from the data during training or from the model input
            layer during inference. This must be divisible by the model's max stride
            (typically 32).
        crop_size_detection_padding: Integer specifying how much extra padding should be
            applied around the instance bounding boxes when automatically detecting the
            appropriate crop size from the data. No effect if the `crop_size` is
            already specified.
    """

    center_on_part: Optional[Text] = None
    crop_size: Optional[int] = None
    crop_size_detection_padding: int = 16


@attr.s(auto_attribs=True)
class DataConfig:
    """Data configuration.

    labels: Configuration options related to user labels for training or testing.
    preprocessing: Configuration options related to data preprocessing.
    instance_cropping: Configuration options related to instance cropping for centroid
        and topdown models.
    """

    labels: LabelsConfig = attr.ib(factory=LabelsConfig)
    preprocessing: PreprocessingConfig = attr.ib(factory=PreprocessingConfig)
    instance_cropping: InstanceCroppingConfig = attr.ib(factory=InstanceCroppingConfig)

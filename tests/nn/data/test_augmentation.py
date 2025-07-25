import pytest
import albumentations as A
import numpy as np
import tensorflow as tf
import sleap
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import augmentation
from sleap.nn.data.pipelines import Pipeline


@pytest.fixture
def dummy_instances_data_nans():
    return np.full((2, 2), np.nan, dtype=np.float32)


@pytest.fixture
def dummy_instances_data_mixed():
    return np.array([[0.1, np.nan], [0.0, 0.8]], dtype=np.float32)


@pytest.fixture
def dummy_image_data():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_instances_data_zeros():
    return np.zeros((2, 2), dtype=np.float32)


@pytest.fixture
def rotation_min_angle():
    return 90


@pytest.fixture
def rotation_max_angle():
    return 90


@pytest.fixture
def augmentation_config(rotation_min_angle, rotation_max_angle):
    return augmentation.AugmentationConfig(
        rotate=True,
        rotation_min_angle=rotation_min_angle,
        rotation_max_angle=rotation_max_angle,
    )


@pytest.fixture
def dummy_dataset(dummy_image_data, dummy_instances_data_zeros):
    dataset = tf.data.Dataset.from_tensor_slices(
        {"image": [dummy_image_data], "instances": [dummy_instances_data_zeros]}
    )
    return dataset


@pytest.fixture
def augmenter(augmentation_config):
    return augmentation.AlbumentationsAugmenter.from_config(augmentation_config)


# Test class instantiation and augmentation
@pytest.mark.parametrize(
    "dummy_instances_data",
    [
        pytest.param("dummy_instances_data_zeros", id="zeros"),
        pytest.param("dummy_instances_data_nans", id="nans"),
        pytest.param("dummy_instances_data_mixed", id="mixed"),
    ],
)
def test_albumentations_augmenter(
    dummy_image_data, dummy_instances_data, augmenter, dummy_dataset
):
    # Apply augmentation
    augmented_dataset = augmenter.transform_dataset(dummy_dataset)

    # Check if augmentation is applied
    augmented_example = next(iter(augmented_dataset))
    assert augmented_example["image"].shape == (100, 100, 3)
    assert augmented_example["instances"].shape == (2, 2)


# Test class method from_config
def test_albumentations_augmenter_from_config(augmentation_config):
    augmenter = augmentation.AlbumentationsAugmenter.from_config(augmentation_config)
    assert isinstance(augmenter, augmentation.AlbumentationsAugmenter)
    assert augmenter.image_key == "image"
    assert augmenter.instances_key == "instances"


def test_augmentation(min_labels):
    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))

    augmenter = augmentation.AlbumentationsAugmenter.from_config(
        augmentation.AugmentationConfig(
            rotate=True, rotation_min_angle=90, rotation_max_angle=90
        )
    )
    ds = augmenter.transform_dataset(ds)

    example = next(iter(ds))

    assert example["image"].shape == (384, 384, 1)
    assert example["image"].dtype == tf.uint8

    np.testing.assert_allclose(
        tf.image.rot90(example_preaug["image"]), example["image"]
    )

    assert example["instances"].shape == (2, 2, 2)
    assert example["instances"].dtype == tf.float32
    # TODO: check for correctness
    assert tf.reduce_all(example["instances"] != example_preaug["instances"])


def test_augmentation_with_no_instances(min_labels):
    # reproduces #555
    labels = min_labels
    labels.append(
        sleap.LabeledFrame(
            video=labels.video,
            frame_idx=labels[-1].frame_idx + 1,
            instances=[
                sleap.Instance.from_numpy(
                    np.full([len(labels.skeleton.nodes), 2], np.nan),
                    skeleton=labels.skeleton,
                )
            ],
        )
    )

    reader = providers.LabelsReader(labels)
    p: Pipeline = Pipeline(reader)
    p += augmentation.AlbumentationsAugmenter.from_config(
        augmentation.AugmentationConfig(rotate=True)
    )
    exs = p.run()
    assert exs[-1]["instances"].shape[0] == 0


def test_augmentation_edges(min_labels):
    # Tests 1722
    height, width = min_labels[0].video.shape[1:3]
    min_labels[0].instances.append(
        sleap.Instance.from_numpy(
            [[0, 0], [width, height]],
            skeleton=min_labels.skeleton,
        )
    )

    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))

    augmenter = augmentation.AlbumentationsAugmenter.from_config(
        augmentation.AugmentationConfig(
            rotate=True, rotation_min_angle=90, rotation_max_angle=90
        )
    )
    ds = augmenter.transform_dataset(ds)

    example = next(iter(ds))
    # TODO: check for correctness
    assert example["instances"].shape == (3, 2, 2)


def test_random_cropper(min_labels):
    cropper = augmentation.RandomCropper(crop_height=64, crop_width=32)
    assert "image" in cropper.input_keys
    assert "instances" in cropper.input_keys
    assert "crop_bbox" in cropper.output_keys

    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))
    ds = cropper.transform_dataset(ds)
    example = next(iter(ds))

    assert example["image"].shape == (64, 32, 1)
    assert "crop_bbox" in example
    offset = tf.stack([example["crop_bbox"][0, 1], example["crop_bbox"][0, 0]], axis=-1)
    assert tf.reduce_all(
        example["instances"]
        == (example_preaug["instances"] - tf.expand_dims(offset, axis=0))
    )


def test_flip_instances_lr():
    insts = tf.cast(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ],
        tf.float32,
    )

    insts_flip = augmentation.flip_instances_lr(insts, 8)
    np.testing.assert_array_equal(insts_flip, [[[7, 1], [5, 3]], [[3, 5], [1, 7]]])

    insts_flip1 = augmentation.flip_instances_lr(insts, 8, [[0, 1]])
    insts_flip2 = augmentation.flip_instances_lr(insts, 8, [[1, 0]])
    np.testing.assert_array_equal(insts_flip1, [[[5, 3], [7, 1]], [[1, 7], [3, 5]]])
    np.testing.assert_array_equal(insts_flip1, insts_flip2)


def test_flip_instances_ud():
    insts = tf.cast(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ],
        tf.float32,
    )

    insts_flip = augmentation.flip_instances_ud(insts, 8)
    np.testing.assert_array_equal(insts_flip, [[[0, 6], [2, 4]], [[4, 2], [6, 0]]])

    insts_flip1 = augmentation.flip_instances_ud(insts, 8, [[0, 1]])
    insts_flip2 = augmentation.flip_instances_ud(insts, 8, [[1, 0]])
    np.testing.assert_array_equal(insts_flip1, [[[2, 4], [0, 6]], [[6, 0], [4, 2]]])
    np.testing.assert_array_equal(insts_flip1, insts_flip2)


def test_random_flipper():
    vid = sleap.Video.from_filename(
        "tests/data/json_format_v1/centered_pair_low_quality.mp4"
    )
    skel = sleap.Skeleton.from_names_and_edge_inds(["A", "BL", "BR"], [[0, 1], [0, 2]])
    labels = sleap.Labels(
        [
            sleap.LabeledFrame(
                video=vid,
                frame_idx=0,
                instances=[
                    sleap.Instance.from_pointsarray(
                        [[25, 50], [50, 25], [25, 25]], skeleton=skel
                    ),
                    sleap.Instance.from_pointsarray(
                        [[125, 150], [150, 125], [125, 125]], skeleton=skel
                    ),
                ],
            )
        ]
    )

    p: Pipeline = Pipeline.from_data(labels)
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=True, probability=1.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0][:, ::-1])
    np.testing.assert_array_equal(
        ex["instances"],
        [
            [[358.0, 50.0], [333.0, 25.0], [358.0, 25.0]],
            [[258.0, 150.0], [233.0, 125.0], [258.0, 125.0]],
        ],
    )

    skel.add_symmetry("BL", "BR")

    p: Pipeline = Pipeline.from_data(labels)
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=True, probability=1.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0][:, ::-1])
    np.testing.assert_array_equal(
        ex["instances"],
        [
            [[358.0, 50.0], [358.0, 25.0], [333.0, 25.0]],
            [[258.0, 150.0], [258.0, 125.0], [233.0, 125.0]],
        ],
    )

    p: Pipeline = Pipeline.from_data(labels)
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=True, probability=0.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0])
    np.testing.assert_array_equal(
        ex["instances"],
        [[[25, 50], [50, 25], [25, 25]], [[125, 150], [150, 125], [125, 125]]],
    )

    p: Pipeline = Pipeline.from_data(labels)
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=False, probability=1.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0][::-1, :])
    np.testing.assert_array_equal(
        ex["instances"],
        [[[25, 333], [25, 358], [50, 358]], [[125, 233], [125, 258], [150, 258]]],
    )


def test_custom_albumentations_augmenter():
    """Test custom albumentations functionality with configuration validation."""

    # Test 1: Valid configuration
    config = augmentation.AugmentationConfig(
        custom_albumentation_funcs=[
            {"function": "MotionBlur", "params": {"blur_limit": [3, 7], "p": 0.5}},
            {"function": "CLAHE", "params": {"clip_limit": 2.0, "p": 0.3}},
            {"function": "Blur", "params": {"blur_limit": 3, "p": 0.2}},
        ]
    )

    augmenter = augmentation.AlbumentationsAugmenter.from_config(config)
    assert len(augmenter.augmenter.transforms) == 3

    # Test 2: Invalid function name should raise ValueError
    with pytest.raises(ValueError, match="not a valid albumentation function"):
        config = augmentation.AugmentationConfig(
            custom_albumentation_funcs=[
                {"function": "NonExistentFunction", "params": {"p": 0.5}}
            ]
        )
        augmentation.AlbumentationsAugmenter.from_config(config)

    # Test 3: Invalid parameter should raise ValueError
    with pytest.raises(ValueError, match="not valid for albumentations function"):
        config = augmentation.AugmentationConfig(
            custom_albumentation_funcs=[
                {"function": "MotionBlur", "params": {"invalid_param": 5, "p": 0.5}}
            ]
        )
        augmentation.AlbumentationsAugmenter.from_config(config)

    # Test 4: Empty/None configuration should work
    config = augmentation.AugmentationConfig(custom_albumentation_funcs=None)
    augmenter = augmentation.AlbumentationsAugmenter.from_config(config)
    assert len(augmenter.augmenter.transforms) == 0

    config = augmentation.AugmentationConfig(custom_albumentation_funcs=[])
    augmenter = augmentation.AlbumentationsAugmenter.from_config(config)
    assert len(augmenter.augmenter.transforms) == 0


def test_custom_albumentations_with_dummy_data(
    dummy_image_data, dummy_instances_data_zeros
):
    """Test custom albumentations with dummy data using existing fixtures."""

    # Create dataset using existing fixtures
    dataset = tf.data.Dataset.from_tensor_slices(
        {"image": [dummy_image_data], "instances": [dummy_instances_data_zeros]}
    )

    # Test with noise augmentation (deterministic for testing)
    config = augmentation.AugmentationConfig(
        custom_albumentation_funcs=[
            {
                "function": "GaussNoise",
                "params": {"var_limit": [10, 10], "p": 1.0},
            },  # Fixed variance for consistency
        ]
    )

    augmenter = augmentation.AlbumentationsAugmenter.from_config(config)
    augmented_dataset = augmenter.transform_dataset(dataset)

    original_example = next(iter(dataset))
    augmented_example = next(iter(augmented_dataset))

    # Check shapes are preserved
    assert augmented_example["image"].shape == original_example["image"].shape
    assert augmented_example["instances"].shape == original_example["instances"].shape

    # Check that augmentation was applied (image should be different due to noise)
    assert not tf.reduce_all(augmented_example["image"] == original_example["image"])

    # Check that keypoints are preserved (GaussNoise doesn't affect keypoints)
    np.testing.assert_array_equal(
        augmented_example["instances"].numpy(), original_example["instances"].numpy()
    )


def test_custom_albumentations_with_pipeline(min_labels):
    """Test custom albumentations integration with SLEAP pipeline using min_labels fixture."""

    # Test with a transform that affects both image and keypoints
    config = augmentation.AugmentationConfig(
        custom_albumentation_funcs=[
            {
                "function": "Affine",
                "params": {"scale": 1.1, "p": 1.0},
            },  # Deterministic scaling
        ]
    )

    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))

    augmenter = augmentation.AlbumentationsAugmenter.from_config(config)
    ds = augmenter.transform_dataset(ds)
    example = next(iter(ds))

    # Check that shapes are preserved
    assert example["image"].shape == example_preaug["image"].shape
    assert example["instances"].shape == example_preaug["instances"].shape
    assert example["image"].dtype == example_preaug["image"].dtype
    assert example["instances"].dtype == example_preaug["instances"].dtype

    # Check that scaling was applied (keypoints should be different)
    assert not tf.reduce_all(example["instances"] == example_preaug["instances"])

    # Test combining with built-in augmentations
    config_combined = augmentation.AugmentationConfig(
        rotate=True,
        rotation_min_angle=0,
        rotation_max_angle=0,  # No rotation for predictable results
        custom_albumentation_funcs=[
            {"function": "CLAHE", "params": {"clip_limit": 2.0, "p": 1.0}},
        ],
    )

    augmenter_combined = augmentation.AlbumentationsAugmenter.from_config(
        config_combined
    )
    # Should have 2 transforms: rotation + CLAHE
    assert len(augmenter_combined.augmenter.transforms) == 2

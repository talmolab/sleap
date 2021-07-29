import numpy as np
import tensorflow as tf
import sleap
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.data import providers
from sleap.nn.data import augmentation


def test_augmentation(min_labels):
    labels_reader = providers.LabelsReader.from_user_instances(min_labels)
    ds = labels_reader.make_dataset()
    example_preaug = next(iter(ds))

    augmenter = augmentation.ImgaugAugmenter.from_config(
        augmentation.AugmentationConfig(
            rotate=True, rotation_min_angle=-90, rotation_max_angle=-90
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
    min_labels.append(
        sleap.LabeledFrame(
            video=min_labels.video,
            frame_idx=min_labels[-1].frame_idx + 1,
            instances=[
                sleap.Instance.from_numpy(
                    np.full([len(min_labels.skeleton.nodes), 2], np.nan),
                    skeleton=min_labels.skeleton,
                )
            ],
        )
    )

    p = min_labels.to_pipeline(user_labeled_only=False)
    p += augmentation.ImgaugAugmenter.from_config(
        augmentation.AugmentationConfig(rotate=True)
    )
    exs = p.run()
    assert exs[-1]["instances"].shape[0] == 0


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

    p = labels.to_pipeline()
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

    p = labels.to_pipeline()
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

    p = labels.to_pipeline()
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=True, probability=0.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0])
    np.testing.assert_array_equal(
        ex["instances"],
        [[[25, 50], [50, 25], [25, 25]], [[125, 150], [150, 125], [125, 125]]],
    )

    p = labels.to_pipeline()
    p += sleap.nn.data.augmentation.RandomFlipper.from_skeleton(
        skel, horizontal=False, probability=1.0
    )
    ex = p.peek()
    np.testing.assert_array_equal(ex["image"], vid[0][0][::-1, :])
    np.testing.assert_array_equal(
        ex["instances"],
        [[[25, 333], [25, 358], [50, 358]], [[125, 233], [125, 258], [150, 258]]],
    )

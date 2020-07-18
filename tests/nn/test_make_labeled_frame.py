from sleap.nn.inference import make_grouped_labeled_frame
import numpy as np
import tensorflow as tf


def test_make_labeled_frame_single(centered_pair_labels):
    frame_examples = [
        {"points": tf.constant(np.ones((3, 2))), "scores": tf.constant(np.ones(3))}
    ]

    lfs = make_grouped_labeled_frame(
        video_ind=0,
        frame_ind=1,
        frame_examples=frame_examples,
        videos=centered_pair_labels.videos,
        skeleton=centered_pair_labels.skeletons[0],
        points_key="points",
        point_confidences_key="scores",
    )

    assert len(lfs) == 1
    assert lfs[0].instances[0].score == 3


def test_make_labeled_frame_multiple(centered_pair_labels):
    # 3-dim data for multiple instances
    frame_examples = [
        {
            "points": tf.constant(np.ones((2, 3, 2))),
            "scores": tf.constant(np.ones((2, 3))),
        }
    ]

    lfs = make_grouped_labeled_frame(
        video_ind=0,
        frame_ind=1,
        frame_examples=frame_examples,
        videos=centered_pair_labels.videos,
        skeleton=centered_pair_labels.skeletons[0],
        points_key="points",
        point_confidences_key="scores",
    )

    assert len(lfs) == 1
    assert len(lfs[0].instances) == 2
    assert lfs[0].instances[0].score == 3


def test_make_labeled_frame_nans(centered_pair_labels):
    # Make example with nan's for all points
    frame_examples = [
        {
            "points": tf.constant(np.full((3, 2), np.nan)),
            "scores": tf.constant(np.ones(3)),
        }
    ]

    lfs = make_grouped_labeled_frame(
        video_ind=0,
        frame_ind=1,
        frame_examples=frame_examples,
        videos=centered_pair_labels.videos,
        skeleton=centered_pair_labels.skeletons[0],
        points_key="points",
        point_confidences_key="scores",
    )

    # There shouldn't be any frames since no *valid* instances
    assert len(lfs) == 0

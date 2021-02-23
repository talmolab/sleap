import numpy as np
import sleap
from sleap.nn.data.training import split_labels_train_val


sleap.use_cpu_only()  # hide GPUs for test


def test_split_labels_train_val():
    vid = sleap.Video(backend=sleap.io.video.MediaVideo)
    labels = sleap.Labels([sleap.LabeledFrame(video=vid, frame_idx=0)])

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0)
    assert len(train) == 1
    assert len(val) == 1

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.1)
    assert len(train) == 1
    assert len(val) == 1

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.5)
    assert len(train) == 1
    assert len(val) == 1

    train, train_inds, val, val_inds = split_labels_train_val(labels, 1.0)
    assert len(train) == 1
    assert len(val) == 1

    labels = sleap.Labels(
        [
            sleap.LabeledFrame(video=vid, frame_idx=0),
            sleap.LabeledFrame(video=vid, frame_idx=1),
        ]
    )
    train, train_inds, val, val_inds = split_labels_train_val(labels, 0)
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].frame_idx != val[0].frame_idx

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.1)
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].frame_idx != val[0].frame_idx

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.5)
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].frame_idx != val[0].frame_idx

    train, train_inds, val, val_inds = split_labels_train_val(labels, 1.0)
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].frame_idx != val[0].frame_idx

    labels = sleap.Labels(
        [
            sleap.LabeledFrame(video=vid, frame_idx=0),
            sleap.LabeledFrame(video=vid, frame_idx=1),
            sleap.LabeledFrame(video=vid, frame_idx=2),
        ]
    )
    train, train_inds, val, val_inds = split_labels_train_val(labels, 0)
    assert len(train) == 2
    assert len(val) == 1

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.1)
    assert len(train) == 2
    assert len(val) == 1

    train, train_inds, val, val_inds = split_labels_train_val(labels, 0.5)
    assert len(train) + len(val) == 3

    train, train_inds, val, val_inds = split_labels_train_val(labels, 1.0)
    assert len(train) == 1
    assert len(val) == 2

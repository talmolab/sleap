"""Tests for sleap.gui.learning.receptivefield crop size functions."""

import numpy as np
import sleap_io as sio

from sleap.gui.learning.receptivefield import (
    find_instance_crop_size,
    find_max_instance_bbox_size,
)


def _make_labels(user_points, predicted_points=None):
    """Build a Labels object with user and optionally predicted instances."""
    skeleton = sio.Skeleton(nodes=[sio.Node("a"), sio.Node("b")])
    video = sio.Video(filename="test.mp4")
    frames = []
    for frame_idx, pts in enumerate(user_points):
        instances = [sio.Instance.from_numpy(np.array(pts), skeleton=skeleton)]
        if predicted_points and frame_idx < len(predicted_points):
            instances.append(
                sio.PredictedInstance.from_numpy(
                    np.array(predicted_points[frame_idx]),
                    skeleton=skeleton,
                    point_scores=np.ones(len(predicted_points[frame_idx])),
                )
            )
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    return sio.Labels(labeled_frames=frames)


class TestFindMaxInstanceBboxSize:
    def test_basic(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_max_instance_bbox_size(labels) == 100.0

    def test_skips_predicted_instances(self):
        """Predicted instances with large bboxes should not inflate the result."""
        labels = _make_labels(
            user_points=[[[0, 0], [100, 50]]],
            predicted_points=[[[0, 0], [800, 800]]],
        )
        assert find_max_instance_bbox_size(labels) == 100.0

    def test_multiple_frames(self):
        labels = _make_labels([[[0, 0], [50, 50]], [[0, 0], [120, 30]]])
        assert find_max_instance_bbox_size(labels) == 120.0


class TestFindInstanceCropSize:
    def test_basic_stride_rounding(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_instance_crop_size(labels, maximum_stride=16) == 112

    def test_skips_predicted_instances(self):
        """Predicted instances should not affect crop size computation."""
        labels = _make_labels(
            user_points=[[[0, 0], [100, 50]]],
            predicted_points=[[[0, 0], [800, 800]]],
        )
        assert find_instance_crop_size(labels, maximum_stride=16) == 112

    def test_with_padding(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_instance_crop_size(labels, padding=20, maximum_stride=16) == 128

    def test_min_crop_size_divisible(self):
        """If min_crop_size is already divisible by stride, return it directly."""
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert (
            find_instance_crop_size(labels, maximum_stride=16, min_crop_size=256) == 256
        )

"""Tests for crop size computation and rotation math."""

import math

import numpy as np

from sleap.gui.learning.crop_size import InstanceCropInfo, compute_instance_crop_sizes


class TestInstanceCropInfo:
    """Tests for InstanceCropInfo dataclass and rotation math."""

    def test_no_rotation(self):
        """Test that zero rotation returns raw crop size."""
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=80.0,
            raw_crop_size=100.0,
        )
        assert info.get_rotated_crop_size(0) == 100.0

    def test_rotation_15_square(self):
        """Test +/-15 degree rotation on a square bounding box."""
        # For a square, rotation by 15 degrees expands by cos(15) + sin(15)
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=100.0,
            raw_crop_size=100.0,
        )
        rotated = info.get_rotated_crop_size(15)

        # Expected: 100 * (cos(15) + sin(15)) ~= 100 * 1.2247 = 122.47
        expected = 100 * (math.cos(math.radians(15)) + math.sin(math.radians(15)))
        assert abs(rotated - expected) < 0.01

    def test_rotation_45_square(self):
        """Test +/-45 degree rotation on a square bounding box."""
        # For a square at 45 degrees, expansion is sqrt(2)
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=100.0,
            raw_crop_size=100.0,
        )
        rotated = info.get_rotated_crop_size(45)

        # Expected: 100 * sqrt(2) ~= 141.42
        expected = 100 * math.sqrt(2)
        assert abs(rotated - expected) < 0.01

    def test_rotation_180_square(self):
        """Test +/-180 degree rotation uses 45 degrees as worst case."""
        # For +/-180, the worst case is still at 45 degrees
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=100.0,
            raw_crop_size=100.0,
        )
        rotated_180 = info.get_rotated_crop_size(180)
        rotated_45 = info.get_rotated_crop_size(45)

        # Should be the same since 45 is the worst case
        assert rotated_180 == rotated_45

    def test_rotation_90_square(self):
        """Test +/-90 degree rotation uses 45 degrees as worst case."""
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=100.0,
            raw_crop_size=100.0,
        )
        rotated_90 = info.get_rotated_crop_size(90)
        rotated_45 = info.get_rotated_crop_size(45)

        # Should be the same since 45 is the worst case for any range >= 45
        assert rotated_90 == rotated_45

    def test_rotation_rectangular(self):
        """Test rotation on a rectangular (non-square) bounding box."""
        # For a rectangle, expansion is less than the theoretical max for a square
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=50.0,
            raw_crop_size=100.0,
        )
        rotated = info.get_rotated_crop_size(45)

        # At 45 degrees: new_width = 100*cos(45) + 50*sin(45) = 106.07
        #                new_height = 100*sin(45) + 50*cos(45) = 106.07
        # (Both happen to be equal at 45 degrees for any rectangle)
        theta = math.radians(45)
        expected = 100 * math.cos(theta) + 50 * math.sin(theta)
        assert abs(rotated - expected) < 0.01

    def test_negative_angle(self):
        """Test that negative angles are handled (absolute value used)."""
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=100.0,
            raw_height=100.0,
            raw_crop_size=100.0,
        )
        rotated_pos = info.get_rotated_crop_size(15)
        rotated_neg = info.get_rotated_crop_size(-15)

        # Should be the same due to symmetry
        assert rotated_pos == rotated_neg

    def test_zero_size_instance(self):
        """Test handling of zero-size bounding box."""
        info = InstanceCropInfo(
            video_idx=0,
            frame_idx=0,
            instance_idx=0,
            raw_width=0.0,
            raw_height=0.0,
            raw_crop_size=0.0,
        )
        assert info.get_rotated_crop_size(45) == 0.0


class TestComputeInstanceCropSizes:
    """Tests for compute_instance_crop_sizes function."""

    def test_empty_labels(self, centered_pair_labels):
        """Test with labels that have no user instances."""
        # Remove all user instances
        for lf in centered_pair_labels:
            lf.instances = []

        result = compute_instance_crop_sizes(centered_pair_labels)
        assert result == []

    def test_basic_computation(self, centered_pair_labels):
        """Test basic crop size computation."""
        result = compute_instance_crop_sizes(centered_pair_labels)

        # Should have some instances
        assert len(result) > 0

        # All entries should have positive crop sizes
        for info in result:
            assert info.raw_crop_size >= 0
            assert info.raw_width >= 0
            assert info.raw_height >= 0

    def test_user_instances_only(self, centered_pair_labels):
        """Test that only user instances are included by default."""
        result = compute_instance_crop_sizes(
            centered_pair_labels, user_instances_only=True
        )

        # Count expected user instances
        expected_count = 0
        for lf in centered_pair_labels:
            for inst in lf.user_instances:
                if not inst.is_empty:
                    pts = inst.numpy()
                    valid_x = pts[:, 0][~np.isnan(pts[:, 0])]
                    valid_y = pts[:, 1][~np.isnan(pts[:, 1])]
                    if len(valid_x) >= 2 and len(valid_y) >= 2:
                        expected_count += 1

        assert len(result) == expected_count

    def test_frame_idx_preserved(self, centered_pair_labels):
        """Test that frame indices are correctly preserved."""
        result = compute_instance_crop_sizes(centered_pair_labels)

        # Check that frame indices match the labeled frames
        for info in result:
            found = False
            for lf in centered_pair_labels:
                if lf.frame_idx == info.frame_idx:
                    found = True
                    break
            assert found, f"Frame index {info.frame_idx} not found in labels"

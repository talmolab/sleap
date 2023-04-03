"""Module to test functions in `sleap.io.cameras`."""

import pytest
from sleap.io.cameras import CameraCluster

def test_cameras(min_session_calibration_toml_path):
    """Test cameras."""
    calibration = min_session_calibration_toml_path
    cameras = CameraCluster.load(calibration)
    
    # Test __len__
    assert len(cameras) == len(cameras.cameras)
    assert len(cameras) == 4

    # Test __getitem__, __iter__, and __contains
    for idx, cam in enumerate(cameras):
        assert cam == cameras[idx]
        assert cam in cameras

    # Test __repr__
    assert f"{cameras.__class__.__name__}(" in repr(cameras)

if __name__ == "__main__":
    pytest.main([__file__])
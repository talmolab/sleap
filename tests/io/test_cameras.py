"""Module to test functions in `sleap.io.cameras`."""

import numpy as np
import pytest
from sleap.io.cameras import Camcorder, CameraCluster

def test_camcorder(min_session_calibration_toml_path):
    """Test camcorder."""
    calibration = min_session_calibration_toml_path
    cameras = CameraCluster.load(calibration)
    cam: Camcorder = cameras[0]
    
    cam_dict = cam.get_dict()
    cam2 = Camcorder.from_dict(cam_dict)
    
    # Test __repr__
    assert f"{cam.__class__.__name__}(" in repr(cam)

    assert np.array_equal(cam.matrix, cam2.matrix)
    assert np.array_equal(cam.dist, cam2.dist)
    assert np.array_equal(cam.size, cam2.size)
    assert np.array_equal(cam.rvec, cam2.rvec)
    assert np.array_equal(cam.tvec, cam2.tvec)
    assert cam.name == cam2.name

    assert cam == cam2

def test_camera_cluster(min_session_calibration_toml_path):
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
    pytest.main([f"{__file__}::test_camcorder"])

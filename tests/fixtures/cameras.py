"""Camera fixtures for pytest."""

import pytest

from sleap.io.cameras import CameraCluster, RecordingSession


@pytest.fixture
def min_session_calibration_toml_path():
    return "tests/data/cameras/minimal_session/calibration.toml"


@pytest.fixture
def min_session_camera_cluster(min_session_calibration_toml_path):
    return CameraCluster.load(min_session_calibration_toml_path)


@pytest.fixture
def min_session_camcorder_0(min_session_camera_cluster):
    return min_session_camera_cluster[0]


@pytest.fixture
def min_session_session(min_session_calibration_toml_path):
    return RecordingSession.load(min_session_calibration_toml_path)

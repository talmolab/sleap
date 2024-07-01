"""Camera fixtures for pytest."""

import shutil
import toml
from pathlib import Path

import pytest

from sleap.io.cameras import CameraCluster, RecordingSession


@pytest.fixture
def min_session_calibration_toml_path():
    return "tests/data/cameras/minimal_session/calibration.toml"


@pytest.fixture
def min_session_camera_cluster(min_session_calibration_toml_path):
    return CameraCluster.load(min_session_calibration_toml_path)


@pytest.fixture
def min_session_session(min_session_calibration_toml_path):
    return RecordingSession.load(min_session_calibration_toml_path)


@pytest.fixture
def min_session_directory(tmpdir, min_session_calibration_toml_path):
    # Create a new RecordingSession object
    camera_calibration_path = min_session_calibration_toml_path

    # Create temporary directory with the structured video files
    temp_dir = tmpdir.mkdir("recording_session")

    # Copy and paste the calibration toml
    shutil.copy(camera_calibration_path, temp_dir)

    # Create directories for each camera
    calibration_data = toml.load(camera_calibration_path)
    camera_dnames = [
        value["name"] for value in calibration_data.values() if "name" in value
    ]

    for cam_name in camera_dnames:
        cam_dir = Path(temp_dir, cam_name)
        cam_dir.mkdir()

    # Copy and paste the videos in the directories (only min_session_[camera_name].mp4)
    videos_dir = Path("tests/data/videos")
    for file in videos_dir.iterdir():
        if file.suffix == ".mp4" and "min_session" in file.stem:
            camera_fname = file.stem.split("_")[2]
            if camera_fname in camera_dnames:
                shutil.copy(file, Path(temp_dir, camera_fname))

    return temp_dir

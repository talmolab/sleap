"""Module to test functions in `sleap.io.cameras`."""

import numpy as np
import pytest
import toml

from sleap.io.cameras import Camcorder, CameraCluster, RecordingSession
from sleap.io.video import Video


def test_camcorder(
    min_session_session: RecordingSession,
    centered_pair_vid: Video,
):
    """Test `Camcorder` data structure."""
    session: RecordingSession = min_session_session
    cam: Camcorder = session.cameras[0]
    video: Video = centered_pair_vid

    # Test from_dict
    cam_dict = cam.get_dict()
    cam2 = Camcorder.from_dict(cam_dict)

    # Test __repr__
    assert f"{cam.__class__.__name__}(" in repr(cam)

    # Check that attributes are the same
    assert np.array_equal(cam.matrix, cam2.matrix)
    assert np.array_equal(cam.dist, cam2.dist)
    assert np.array_equal(cam.size, cam2.size)
    assert np.array_equal(cam.rvec, cam2.rvec)
    assert np.array_equal(cam.tvec, cam2.tvec)
    assert cam.name == cam2.name
    assert cam.extra_dist == cam2.extra_dist

    # Test __eq__
    assert cam == cam2

    # Test videos property
    assert cam.videos == []
    session.add_video(video, cam)
    assert cam.videos == [video]

    # Test sessions property
    assert cam.sessions == [session]

    # Test __getitem__
    assert cam[session] == video
    assert cam[video] == session
    with pytest.raises(KeyError):
        cam["foo"]


def test_camera_cluster(
    min_session_calibration_toml_path: str,
    min_session_session: RecordingSession,
    centered_pair_vid: Video,
):
    """Test `CameraCluster` data structure."""
    # Test load
    calibration = min_session_calibration_toml_path
    camera_cluster = CameraCluster.load(calibration)
    assert camera_cluster.metadata["calibration_path"] == calibration
    assert camera_cluster.metadata["calibration"] == toml.load(calibration)

    # Test __len__
    assert len(camera_cluster) == len(camera_cluster.cameras)
    assert len(camera_cluster) == 4

    # Test __getitem__, __iter__, and __contains__
    for idx, cam in enumerate(camera_cluster):
        assert cam == camera_cluster[idx]
        assert cam in camera_cluster

    # Test __repr__
    assert f"{camera_cluster.__class__.__name__}(" in repr(camera_cluster)

    # Test validator
    with pytest.raises(TypeError):
        camera_cluster.cameras = [1, 2, 3]

    # Test converter
    assert isinstance(camera_cluster.cameras[0], Camcorder)

    # Test sessions property and add_session
    assert camera_cluster.sessions == []
    camera_cluster.add_session(min_session_session)
    assert camera_cluster.sessions == [min_session_session]

    # Test videos property
    camera = camera_cluster.cameras[0]
    min_session_session.add_video(centered_pair_vid, camera)
    assert camera_cluster.videos == [centered_pair_vid]

    # Test __getitem__
    assert camera_cluster[centered_pair_vid] == (camera, min_session_session)
    assert camera_cluster[camera] == [centered_pair_vid]
    assert camera_cluster[min_session_session] == [centered_pair_vid]
    min_session_session.remove_video(centered_pair_vid)
    assert camera_cluster[centered_pair_vid] is None
    assert camera_cluster[camera] == []
    assert camera_cluster[min_session_session] == []


def test_recording_session(
    min_session_calibration_toml_path: str,
    min_session_camera_cluster: CameraCluster,
    centered_pair_vid: Video,
):
    """Test `RecordingSession` data structure."""
    calibration: str = min_session_calibration_toml_path
    camera_cluster: CameraCluster = min_session_camera_cluster

    # Test load
    session = RecordingSession.load(calibration)
    session.metadata = {"test": "we can access this information!"}
    session.camera_cluster.metadata = {
        "another_test": "we can even access this information!"
    }

    # Test __attrs_post_init__
    assert session in session.camera_cluster.sessions

    # Test __iter__, __contains__, and __getitem__ (with int key)
    for idx, cam in enumerate(session):
        assert isinstance(cam, Camcorder)
        assert cam in camera_cluster
        assert cam == camera_cluster[idx]

    # Test __getattr__
    assert session.cameras == camera_cluster.cameras

    # Test __getitem__ with string key
    assert session["test"] == "we can access this information!"
    assert session["another_test"] == "we can even access this information!"

    # Test __len__
    assert len(session) == len(session.videos)

    # Test __repr__
    assert f"{session.__class__.__name__}(" in repr(session)

    # Test add_video
    camcorder = session.camera_cluster.cameras[0]
    session.add_video(centered_pair_vid, camcorder)
    assert centered_pair_vid is session.camera_cluster._videos_by_session[session][0]
    assert centered_pair_vid is camcorder._video_by_session[session]
    assert session is session.camera_cluster._session_by_video[centered_pair_vid]
    assert camcorder is session.camera_cluster._camcorder_by_video[centered_pair_vid]
    assert centered_pair_vid is session._video_by_camcorder[camcorder]

    # Test videos property
    assert centered_pair_vid in session.videos

    # Test linked_cameras property
    assert camcorder in session.linked_cameras
    assert camcorder not in session.unlinked_cameras

    # Test __getitem__ with `Video` key
    assert session[centered_pair_vid] is camcorder

    # Test __getitem__ with `Camcorder` key
    assert session[camcorder] is centered_pair_vid

    # Test remove_video
    session.remove_video(centered_pair_vid)
    assert centered_pair_vid not in session.videos
    assert camcorder not in session.linked_cameras
    assert camcorder in session.unlinked_cameras
    assert centered_pair_vid not in session.camera_cluster._videos_by_session[session]
    assert session not in camcorder._video_by_session
    assert centered_pair_vid not in session.camera_cluster._session_by_video
    assert centered_pair_vid not in session.camera_cluster._camcorder_by_video
    assert camcorder not in session._video_by_camcorder

    # Test __getitem__ with `Video` key
    assert session[centered_pair_vid] is None

    # Test __getitem__ with `Camcorder` key
    assert session[camcorder] is None

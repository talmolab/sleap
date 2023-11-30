"""Module to test functions in `sleap.io.cameras`."""

import numpy as np
import pytest

from sleap.io.cameras import Camcorder, CameraCluster, InstanceGroup, RecordingSession
from sleap.io.dataset import Instance, Labels
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

    # Test __len__
    assert len(camera_cluster) == len(camera_cluster.cameras)
    assert len(camera_cluster) == 8

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

    # Test to_calibration_dict
    calibration_dict = camera_cluster.to_calibration_dict()
    assert isinstance(calibration_dict, dict)
    for cam_idx, cam in enumerate(camera_cluster):
        cam_key = f"cam_{cam_idx}"
        cam_value = calibration_dict[cam_key]

        assert calibration_dict[cam_key]["name"] == cam.name
        assert np.array_equal(cam_value["matrix"], cam.matrix)
        assert np.array_equal(cam_value["distortions"], cam.dist)
        assert np.array_equal(cam_value["size"], cam.size)
        assert np.array_equal(cam_value["rotation"], cam.rvec)
        assert np.array_equal(cam_value["translation"], cam.tvec)

    # Test from_calibration_dict
    camera_cluster2 = CameraCluster.from_calibration_dict(calibration_dict)
    assert isinstance(camera_cluster2, CameraCluster)
    assert len(camera_cluster2) == len(camera_cluster)
    for cam_1, cam_2 in zip(camera_cluster, camera_cluster2):
        assert cam_1 == cam_2
    assert camera_cluster2.sessions == []


def test_recording_session(
    min_session_calibration_toml_path: str,
    min_session_camera_cluster: CameraCluster,
    centered_pair_vid: Video,
    hdf5_vid: Video,
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

    # Test from_calibration_dict
    def compare_cameras(session_1: RecordingSession, session_2: RecordingSession):
        assert len(session_2.camera_cluster) == len(session_1.camera_cluster)
        for cam_1, cam_2 in zip(session_1, session_2):
            assert cam_1 == cam_2

    calibration_dict = session.camera_cluster.to_calibration_dict()
    session_2 = RecordingSession.from_calibration_dict(calibration_dict)
    assert isinstance(session_2, RecordingSession)
    assert len(session_2.videos) == 0
    compare_cameras(session, session_2)

    # Test to_session_dict
    camcorder_2 = session.camera_cluster.cameras[2]
    session.add_video(hdf5_vid, camcorder_2)
    videos_list = [centered_pair_vid, hdf5_vid]
    video_to_idx = {video: idx for idx, video in enumerate(videos_list)}
    session_dict = session.to_session_dict(video_to_idx)
    assert isinstance(session_dict, dict)
    assert session_dict["calibration"] == calibration_dict
    assert session_dict["camcorder_to_video_idx_map"] == {
        "0": str(video_to_idx[centered_pair_vid]),
        "2": str(video_to_idx[hdf5_vid]),
    }

    # Test from_session_dict
    def compare_sessions(session_1: RecordingSession, session_2: RecordingSession):
        assert isinstance(session_2, RecordingSession)
        assert not (session_2 == session)  # Not the same object in memory
        assert len(session_2.camera_cluster) == len(session_1.camera_cluster)
        compare_cameras(session_1, session_2)
        assert len(session_2.videos) == len(session_1.videos)
        assert np.array_equal(session_2.videos, session_1.videos)

    session_2 = RecordingSession.from_session_dict(session_dict, videos_list)
    compare_sessions(session, session_2)

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

    # Test make_cattr
    sessions_cattr = RecordingSession.make_cattr(videos_list)
    session_dict_2 = sessions_cattr.unstructure(session_2)
    assert session_dict_2 == session_dict
    session_3 = sessions_cattr.structure(session_dict_2, RecordingSession)
    compare_sessions(session_2, session_3)


def test_recording_session_get_videos_from_selected_cameras(
    multiview_min_session_labels: Labels,
):
    session = multiview_min_session_labels.sessions[0]

    # Test get_videos_from_selected_cameras
    selected_cam = session.linked_cameras[0]
    selected_videos = session.get_videos_from_selected_cameras([selected_cam])
    assert len(selected_videos) == 1
    assert selected_videos[selected_cam] == session.get_video(selected_cam)
    # Now without any cameras selected: expect to return all videos
    selected_videos = session.get_videos_from_selected_cameras()
    assert len(selected_videos) == len(session.linked_cameras)
    for cam in session.linked_cameras:
        assert cam in selected_videos
        assert session.get_video(cam) == selected_videos[cam]


def test_recording_session_remove_video(multiview_min_session_labels: Labels):
    """Test `RecordingSession.remove_video`."""

    labels = multiview_min_session_labels
    labels_cache = labels._cache
    session = labels.sessions[0]

    video = session.videos[0]
    assert session.labels is not None
    assert video in session.videos

    session.remove_video(video)
    assert labels_cache._session_by_video.get(video, None) is None
    assert video not in session.videos


def test_instance_group(multiview_min_session_labels: Labels):
    """Test `InstanceGroup` data structure."""

    labels = multiview_min_session_labels
    session = labels.sessions[0]
    camera_cluster = session.camera_cluster

    lf = labels.labeled_frames[0]
    frame_idx = lf.frame_idx
    instance = lf.instances[0]

    dummy_instance = Instance.from_numpy(
        np.full(
            shape=(len(instance.skeleton.nodes), 2),
            fill_value=np.nan,
        ),
        skeleton=instance.skeleton,
    )
    instance_by_camera = {}
    for cam in session.linked_cameras:
        video = session.get_video(cam)
        lfs_in_view = labels.find(video=video, frame_idx=frame_idx)
        instance = (
            lfs_in_view[0].instances[0] if len(lfs_in_view) > 0 else dummy_instance
        )
        instance_by_camera[cam] = instance

    # Add a dummy instance to make sure it gets ignored
    instance_by_camera[cam] = dummy_instance

    # Test `from_dict`
    instance_group = InstanceGroup.from_dict(d=instance_by_camera)
    assert isinstance(instance_group, InstanceGroup)
    assert instance_group.frame_idx == frame_idx
    assert instance_group.camera_cluster == camera_cluster
    for camera in session.linked_cameras:
        if camera == cam:
            assert instance_by_camera[camera] == dummy_instance
            assert camera not in instance_group.cameras
        else:
            instance = instance_group[camera]
            assert isinstance(instance, Instance)
            assert instance_group[camera] == instance_by_camera[camera]
            assert instance_group[instance] == camera

    # Test `__repr__`
    print(instance_group)

    # Test `__len__`
    assert len(instance_group) == len(instance_by_camera) - 1

    # Test `get_cam`
    assert instance_group.get_cam(dummy_instance) is None

    # Test `get_instance`
    assert instance_group.get_instance(cam) is None

    # Test `instances` property
    assert len(instance_group.instances) == len(instance_by_camera) - 1

    # Test `cameras` property
    assert len(instance_group.cameras) == len(instance_by_camera) - 1

    # Test `__getitem__` with `int` key
    assert isinstance(instance_group[0], Instance)
    with pytest.raises(KeyError):
        instance_group[len(instance_group)]

    # Populate with only dummy instance and test `from_dict`
    instance_by_camera = {cam: dummy_instance}
    instance_group = InstanceGroup.from_dict(d=instance_by_camera)
    assert instance_group is None

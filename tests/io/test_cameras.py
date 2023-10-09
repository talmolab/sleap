"""Module to test functions in `sleap.io.cameras`."""

from typing import List

import numpy as np
import pytest

from sleap.io.cameras import Camcorder, CameraCluster, RecordingSession
from sleap.io.dataset import Instance, LabeledFrame, Labels
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
    multiview_min_session_labels: Labels,
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
        assert len(session_2.camera_cluster) == len(session.camera_cluster)
        for cam_1, cam_2 in zip(session, session_2):
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
    assert selected_videos[0] == session.get_video(selected_cam)
    # Now without any cameras selected: expect to return all videos
    selected_videos = session.get_videos_from_selected_cameras()
    assert len(selected_videos) == len(session.linked_cameras)
    for cam in session.linked_cameras:
        assert session.get_video(cam) in selected_videos


def test_recording_session_get_instances_across_views(
    multiview_min_session_labels: Labels,
):
    # Test get_instances_across_views
    labels = multiview_min_session_labels
    lf: LabeledFrame = labels[0]
    track = labels.tracks[0]
    session_from_labels = labels.sessions[0]
    instances: List[Instance] = session_from_labels.get_instances_across_views(
        frame_idx=lf.frame_idx, track=track
    )
    assert len(instances) == len(session_from_labels.videos)
    for inst, vid in zip(instances, session_from_labels.videos):
        assert inst.frame_idx == lf.frame_idx
        assert inst.track == track
        assert inst.video == vid
    # Try with excluding cam views
    lf: LabeledFrame = labels[2]
    track = labels.tracks[1]
    cams_to_include = session_from_labels.linked_cameras[:4]
    videos_to_include = session_from_labels.get_videos_from_selected_cameras(
        cams_to_include=cams_to_include
    )
    assert len(cams_to_include) == 4
    assert len(videos_to_include) == len(cams_to_include)
    instances: List[Instance] = session_from_labels.get_instances_across_views(
        frame_idx=lf.frame_idx, track=track, cams_to_include=cams_to_include
    )
    assert len(instances) == len(
        videos_to_include
    )  # May not be true if no instances at that frame
    for inst, vid in zip(instances, videos_to_include):
        assert inst.frame_idx == lf.frame_idx
        assert inst.track == track
        assert inst.video == vid
    # Try with only a single view
    cams_to_include = [session_from_labels.linked_cameras[0]]
    with pytest.raises(ValueError):
        instances = session_from_labels.get_instances_across_views(
            frame_idx=lf.frame_idx,
            cams_to_include=cams_to_include,
            track=track,
            require_multiple_views=True,
        )
    # Try with multiple views, but not enough instances
    track = labels.tracks[1]
    cams_to_include = session_from_labels.linked_cameras[4:6]
    with pytest.raises(ValueError):
        instances = session_from_labels.get_instances_across_views(
            frame_idx=lf.frame_idx,
            cams_to_include=cams_to_include,
            track=track,
            require_multiple_views=True,
        )


def test_recording_session_calculate_reprojected_points(
    multiview_min_session_labels: Labels,
):
    """Test `RecordingSession.calculate_reprojected_points`."""
    session = multiview_min_session_labels.sessions[0]
    lf: LabeledFrame = multiview_min_session_labels[0]
    track = multiview_min_session_labels.tracks[0]
    instances: List[Instance] = session.get_instances_across_views(
        frame_idx=lf.frame_idx, track=track
    )
    inst_coords_list = session.calculate_reprojected_points(instances)

    # Check that we get the same number of instances as input
    assert len(instances) == len(inst_coords_list)

    # Check that each instance has the same number of points
    for inst, inst_coords in zip(instances, inst_coords_list):
        assert inst_coords.shape[1] == len(inst.skeleton)  # (1, 15, 2)


def test_recording_session_update_instances(multiview_min_session_labels: Labels):
    """Test `RecordingSession.update_instances`."""

    # Test update_instances
    session = multiview_min_session_labels.sessions[0]
    lf: LabeledFrame = multiview_min_session_labels[0]
    track = multiview_min_session_labels.tracks[0]
    instances: List[Instance] = session.get_instances_across_views(
        frame_idx=lf.frame_idx, track=track
    )
    inst_coords_list = session.calculate_reprojected_points(instances)
    for inst, inst_coords in zip(instances, inst_coords_list):
        assert inst_coords.shape == (1, len(inst.skeleton), 2)  # Tracks, Nodes, 2
        # Assert coord are different from original
        assert not np.array_equal(inst_coords, inst.points_array)

    # Just run for code coverage testing, do not test output here (race condition)
    # (see "functional core, imperative shell" pattern)
    session.update_instances(instances)


# TODO(LM): Remove debugging code
if __name__ == "__main__":
    pytest.main([f"{__file__}::test_recording_session_update_instances"])

"""Module to test functions in `sleap.io.cameras`."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from sleap.io.cameras import (
    Camcorder,
    CameraCluster,
    InstanceGroup,
    FrameGroup,
    RecordingSession,
)
from sleap.io.dataset import Instance, Labels, PredictedInstance
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

    # Test new_frame_group
    frame_group = session.new_frame_group(frame_idx=0)
    assert isinstance(frame_group, FrameGroup)
    assert frame_group.session == session
    assert frame_group.frame_idx == 0
    assert frame_group == session.frame_groups[0]

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
    labels = multiview_min_session_labels
    camcorder_2 = session.camera_cluster.cameras[2]
    session.add_video(hdf5_vid, camcorder_2)
    videos_list = [centered_pair_vid, hdf5_vid]
    video_to_idx = {video: idx for idx, video in enumerate(videos_list)}
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labels.labeled_frames)}
    session_dict = session.to_session_dict(
        video_to_idx=video_to_idx, labeled_frame_to_idx=labeled_frame_to_idx
    )
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

    labeled_frames_list = labels.labeled_frames
    session_2 = RecordingSession.from_session_dict(
        session_dict=session_dict,
        videos_list=videos_list,
        labeled_frames_list=labeled_frames_list,
    )
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

    # Test projection_bounds
    projection_bounds = session.projection_bounds
    n_cameras, n_coords = projection_bounds.shape
    assert n_cameras == len(session.camera_cluster.cameras)
    assert n_coords == 2
    n_linked_cameras = len(session.linked_cameras)
    assert n_linked_cameras < n_cameras
    assert projection_bounds[np.isnan(projection_bounds)].shape == (
        n_coords * (n_cameras - n_linked_cameras),
    )
    for cam_idx, cam in enumerate(session.camera_cluster.cameras):
        if cam in session.linked_cameras:
            linked_video = session.get_video(cam)
            assert projection_bounds[cam_idx, 0] == linked_video.width
            assert projection_bounds[cam_idx, 1] == linked_video.height
        else:
            assert np.all(np.isnan(projection_bounds[cam_idx]) == True)

    # Test make_cattr
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labels.labeled_frames)}
    sessions_cattr = RecordingSession.make_cattr(
        videos_list=videos_list,
        labeled_frames_list=labels.labeled_frames,
        labeled_frame_to_idx=labeled_frame_to_idx,
    )
    session_dict_2 = sessions_cattr.unstructure(session_2)
    assert session_dict_2 == session_dict
    session_3 = sessions_cattr.structure(session_dict_2, RecordingSession)
    compare_sessions(session_2, session_3)

    # Test id
    assert session.id == hash(session)
    labels.add_session(session)
    assert session.id == labels.sessions.index(session)


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


def test_recording_session_frame_group_integration(
    multiview_min_session_user_labels: Labels,
    tmp_path,
):
    """Test `RecordingSession` integration with `FrameGroup` and `InstanceGroup`.

    Note: This is how the multiview_min_session_frame_groups fixture was created.
    """

    labels = multiview_min_session_user_labels
    session = labels.sessions[0]

    # Get all tracks
    tracks = labels.tracks
    frame_idxs = [0, 1, 2]

    # Create a new `FrameGroup` for each frame_idx
    for frame_idx in frame_idxs:

        frame_group = session.new_frame_group(frame_idx=frame_idx)

        # Create new instance group per track
        inst_group_by_track = {}
        for track in tracks:
            instance_group = frame_group.add_instance_group()
            inst_group_by_track[track] = instance_group

        # Add instances to frame and instance groups
        for camera in session.linked_cameras:

            # Get video and labeled frame
            video = session.get_video(camcorder=camera)
            labeled_frames = labels.find(video=video, frame_idx=frame_idx)
            if len(labeled_frames) < 1:
                continue
            else:
                labeled_frame = labeled_frames[0]

            # Add instances to frame and instance groups
            for instance in labeled_frame.user_instances:
                track = instance.track
                instance_group: InstanceGroup = inst_group_by_track[track]
                frame_group.add_instance(
                    instance=instance, camera=camera, instance_group=instance_group
                )

    assert len(session.frame_groups) == len(frame_idxs)
    for frame_idx in frame_idxs:
        assert frame_idx in session.frame_groups
        assert len(session.frame_groups[frame_idx].instance_groups) == len(tracks)
        for instance_group in session.frame_groups[frame_idx].instance_groups:
            assert (
                len(instance_group.instances) == 6 or len(instance_group.instances) == 8
            )

    # Save the labels to a temporary file
    ds_base = tmp_path
    ds_base.mkdir(exist_ok=True)
    ds_new = ds_base / "frame_groups.slp"
    labels.save(filename=ds_new.as_posix())

    # Load the labels from the temporary file
    labels_new = Labels.load_file(ds_new.as_posix())
    session_ln = labels_new.sessions[0]

    # Check that the loaded labels are the same as the original labels
    assert len(session_ln.frame_groups) == len(session.frame_groups)
    for frame_idx in frame_idxs:
        assert frame_idx in session_ln.frame_groups
        assert len(session_ln.frame_groups[frame_idx].instance_groups) == len(
            session.frame_groups[frame_idx].instance_groups
        )
        for instance_group_ln, instance_group in zip(
            session_ln.frame_groups[frame_idx].instance_groups,
            session.frame_groups[frame_idx].instance_groups,
        ):
            assert len(instance_group.instances) == len(instance_group.instances)
            for instance_ln, instance in zip(
                instance_group_ln.instances, instance_group.instances
            ):
                assert instance_ln.matches(instance)


# TODO(LM): Remove after adding method to (de)seralize `InstanceGroup`
def create_instance_group(
    labels: Labels,
    frame_idx: int,
    add_dummy: bool = False,
    name: Optional[str] = None,
    score: Optional[float] = None,
) -> Union[
    InstanceGroup, Tuple[InstanceGroup, Dict[Camcorder, Instance], Instance, Camcorder]
]:
    """Create an `InstanceGroup` from a `Labels` object.

    Args:
        labels: The `Labels` object to use.
        frame_idx: The frame index to use.
        add_dummy: Whether to add a dummy instance to the `InstanceGroup`.

    Returns:
        The `InstanceGroup` object.
    """

    if name is None:
        name = "test_instance_group"

    session = labels.sessions[0]

    lf = labels.labeled_frames[0]
    instance = lf.instances[0]

    instance_by_camera = {}
    for cam in session.linked_cameras:
        video = session.get_video(cam)
        lfs_in_view = labels.find(video=video, frame_idx=frame_idx)
        if len(lfs_in_view) > 0:
            instance = lfs_in_view[0].instances[0]
            instance_by_camera[cam] = instance

    # Add a dummy instance to make sure it gets ignored
    if add_dummy:
        dummy_instance = Instance.from_numpy(
            np.full(
                shape=(len(instance.skeleton.nodes), 2),
                fill_value=np.nan,
            ),
            skeleton=instance.skeleton,
        )
        instance_by_camera[cam] = dummy_instance

    instance_group = InstanceGroup.from_instance_by_camcorder_dict(
        instance_by_camcorder=instance_by_camera,
        name="test_instance_group",
        name_registry={},
        score=score,
    )
    return (
        (instance_group, instance_by_camera, dummy_instance, cam)
        if add_dummy
        else instance_group
    )


def test_instance_group(
    multiview_min_session_labels: Labels, multiview_min_session_frame_groups: Labels
):
    """Test `InstanceGroup` data structure."""

    labels = multiview_min_session_labels
    session = labels.sessions[0]
    camera_cluster = session.camera_cluster

    lf = labels.labeled_frames[0]
    frame_idx = lf.frame_idx

    # Test `_create_dummy_instance` (fail)
    instance_group = InstanceGroup(name="test_instance_group", frame_idx=frame_idx)
    with pytest.raises(ValueError):
        dummy_instance = instance_group._create_dummy_instance()

    # Test `from_instance_by_camcorder_dict`
    name = "test_instance_group"
    instance_group, instance_by_camera, dummy_instance, cam = create_instance_group(
        labels=labels, frame_idx=frame_idx, add_dummy=True, name=name
    )
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

    # Test `_create_dummy_instance` (pass)
    dummy_instance = instance_group.dummy_instance
    assert isinstance(dummy_instance, PredictedInstance)
    matched_instance = instance_group.instances[0]
    assert dummy_instance.skeleton == matched_instance.skeleton
    assert np.all(np.isnan(dummy_instance.points))
    assert np.isnan(
        dummy_instance.score
    )  # TODO(LM): Should be OKS (g.t. vs reprojection)
    assert dummy_instance.tracking_score == 0.0

    # Test `name` property
    assert instance_group.name == name

    # Test `name.setter`
    with pytest.raises(ValueError):
        instance_group.name = "test_instance_group_2"

    # Test `set_name`
    new_name = "test_instance_group_2"
    name_registry = {instance_group.name}
    instance_group.set_name(name=new_name, name_registry=name_registry)
    assert instance_group.name == new_name
    assert instance_group.name in name_registry
    assert name not in name_registry
    with pytest.raises(ValueError):  # Name already in registry
        instance_group.set_name(name=new_name, name_registry=name_registry)

    # Test `return_unique_name`
    name_registry = {"instance_group_1"}
    new_name = instance_group.return_unique_name(name_registry=name_registry)
    assert new_name not in name_registry
    assert new_name == "instance_group_2"

    # Test `to_dict`
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labels.labeled_frames)}
    instance_to_lf_and_inst_idx = {
        instance: (labeled_frame_to_idx[lf], inst_idx)
        for lf in labels.labeled_frames
        for inst_idx, instance in enumerate(lf.instances)
    }
    instance_group_dict = instance_group.to_dict(
        instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx
    )
    assert isinstance(instance_group_dict, dict)
    assert instance_group_dict["name"] == instance_group.name
    assert "camcorder_to_lf_and_inst_idx_map" in instance_group_dict
    assert "score" not in instance_group_dict

    # Test `score` property (and `to_dict`)
    assert instance_group.score is None
    instance_group.score = 0.5
    for instance in instance_group.instances:
        assert instance.score == 0.5
    instance_group.score = 0.75
    for instance in instance_group.instances:
        assert instance.score == 0.75
    instance_group_dict = instance_group.to_dict(
        instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx
    )
    assert instance_group_dict["score"] == str(0.75)

    # Test `from_dict`
    instance_group_2 = InstanceGroup.from_dict(
        instance_group_dict=instance_group_dict,
        name_registry={},
        labeled_frames_list=labels.labeled_frames,
        camera_cluster=camera_cluster,
    )
    assert isinstance(instance_group_2, InstanceGroup)
    assert instance_group_2.camera_cluster == camera_cluster
    assert instance_group_2.name == instance_group.name
    assert instance_group_2.frame_idx == instance_group.frame_idx
    assert (
        instance_group_2._instance_by_camcorder == instance_group._instance_by_camcorder
    )
    assert (
        instance_group_2._camcorder_by_instance == instance_group._camcorder_by_instance
    )
    assert instance_group_2.dummy_instance.matches(instance_group.dummy_instance)

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

    # Populate with only dummy instance and test `from_instance_by_camcorder_dict`
    instance_by_camera = {cam: dummy_instance}
    with pytest.raises(ValueError):
        instance_group = InstanceGroup.from_instance_by_camcorder_dict(
            instance_by_camcorder=instance_by_camera,
            name="test_instance_group",
            name_registry={},
        )
    instance_by_camera = {
        cam: instance_group.get_instance(cam) for cam in instance_group.cameras
    }
    instance_group_from_dict = InstanceGroup.from_instance_by_camcorder_dict(
        instance_by_camcorder=instance_by_camera,
        name="test_instance_group",
        name_registry={},
        score=0.5,
    )
    assert instance_group_from_dict.score == 0.5
    # The score of instances will NOT be updated on initialization.
    for instance in instance_group_from_dict.instances:
        if isinstance(instance, PredictedInstance):
            assert instance.score != instance_group_from_dict.score

    # Test `__repr__`
    print(instance_group)

    # Switch Labels files to one that already contains populated `InstanceGroup`s
    labels = multiview_min_session_frame_groups
    session: RecordingSession = labels.sessions[0]
    frame_idx = 0
    frame_group = session.frame_groups[frame_idx]
    instance_group = frame_group.instance_groups[0]

    # Test `numpy` method
    instance_group_numpy = instance_group.numpy()
    n_views, n_nodes, n_coords = instance_group_numpy.shape
    assert n_views == len(instance_group.camera_cluster.cameras)
    assert n_nodes == len(instance_group.dummy_instance.skeleton.nodes)
    assert n_coords == 2
    # Different instance groups should have different coordinates
    for inst_idx, _ in enumerate(instance_group.instances[:-1]):
        assert not np.allclose(
            instance_group_numpy[:, inst_idx],
            instance_group_numpy[:, inst_idx + 1],
            equal_nan=True,
        )
    # Different views should have different coordinates
    for view_idx, _ in enumerate(instance_group.camera_cluster.cameras[:-1]):
        assert not np.allclose(
            instance_group_numpy[view_idx],
            instance_group_numpy[view_idx + 1],
            equal_nan=True,
        )

    # Test `update_points` method
    assert not np.all(instance_group.numpy(invisible_as_nan=False) == 72317)
    # Remove some Instances to "expose" underlying PredictedInstances
    for inst in instance_group.instances[:2]:
        lf = inst.frame
        labels.remove_instance(lf, inst)
    instance_group.update_points(points=np.full((n_views, n_nodes, n_coords), 72317))
    for inst in instance_group.instances:
        if isinstance(inst, PredictedInstance):
            assert inst.score == instance_group.score
    prev_score = instance_group.score
    instance_group.update_points(points=np.full((n_views, n_nodes, n_coords), 72317))
    for inst in instance_group.instances:
        if isinstance(inst, PredictedInstance):
            assert inst.score == instance_group.score
    instance_group_numpy = instance_group.numpy(invisible_as_nan=False)
    assert np.all(instance_group_numpy == 72317)
    assert instance_group.score == 1.0  # Score should be 1.0 because same points

    # Test `add_instance`, `replace_instance`, and `remove_instance`
    cam = instance_group.cameras[0]
    instance = instance_group.instances[1]
    assert instance_group.get_instance(cam) != instance
    with pytest.raises(ValueError):  # `Instance` already in this `InstanceGroup`
        instance_group.add_instance(instance=instance, cam=cam)
    # Let's replace an instance
    instance_being_replaced = instance_group.get_instance(cam=cam)
    old_instance_cam = instance_group.get_cam(instance=instance)
    instance_group.replace_instance(instance=instance, cam=cam)
    assert instance_group.get_instance(cam=cam) == instance
    assert instance_being_replaced not in instance_group.instances
    assert old_instance_cam not in instance_group.cameras
    # Let's remove an instance (using `Camcorder` input)
    instance_group.remove_instance(instance_or_cam=cam)
    assert instance not in instance_group.instances
    assert cam not in instance_group.cameras


def test_frame_group(
    multiview_min_session_labels: Labels, multiview_min_session_frame_groups: Labels
):
    """Test `FrameGroup` data structure."""

    labels = multiview_min_session_labels
    session = labels.sessions[0]

    # Test `from_instance_groups` from list of instance groups
    frame_idx_1 = 0
    instance_group = create_instance_group(labels=labels, frame_idx=frame_idx_1)
    instance_groups: List[InstanceGroup] = [instance_group]
    frame_group_1 = FrameGroup.from_instance_groups(
        session=session, instance_groups=instance_groups
    )
    assert isinstance(frame_group_1, FrameGroup)
    assert frame_idx_1 in session.frame_groups
    assert len(session.frame_groups) == 1
    assert frame_group_1 == session.frame_groups[frame_idx_1]
    assert len(frame_group_1.instance_groups) == 1

    # Test `RecordingSession.frame_groups` property
    frame_idx_2 = 1
    instance_group = create_instance_group(labels=labels, frame_idx=frame_idx_2)
    instance_groups: List[InstanceGroup] = [instance_group]
    frame_group_2 = FrameGroup.from_instance_groups(
        session=session, instance_groups=instance_groups
    )
    assert isinstance(frame_group_2, FrameGroup)
    assert frame_idx_2 in session.frame_groups
    assert len(session.frame_groups) == 2
    assert frame_group_2 == session.frame_groups[frame_idx_2]
    assert len(frame_group_2.instance_groups) == 1

    frame_idx_3 = 2
    frame_group_3 = FrameGroup(frame_idx=frame_idx_3, session=session)
    assert isinstance(frame_group_3, FrameGroup)
    assert frame_idx_3 in session.frame_groups
    assert len(session.frame_groups) == 3
    assert frame_group_3 == session.frame_groups[frame_idx_3]
    assert len(frame_group_3.instance_groups) == 0

    # Test `to_dict`
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labels.labeled_frames)}
    frame_group_dict = frame_group_1.to_dict(labeled_frame_to_idx=labeled_frame_to_idx)
    assert isinstance(frame_group_dict, dict)
    assert "instance_groups" in frame_group_dict
    assert len(frame_group_dict["instance_groups"]) == 1
    instance_group_dict = frame_group_dict["instance_groups"][0]
    assert instance_group_dict["name"] == instance_group.name
    assert "camcorder_to_lf_and_inst_idx_map" in instance_group_dict

    # Test `from_dict`
    frame_group_4 = FrameGroup.from_dict(
        frame_group_dict=frame_group_dict,
        session=session,
        labeled_frames_list=labels.labeled_frames,
    )
    assert isinstance(frame_group_4, FrameGroup)
    assert frame_group_4.frame_idx == frame_idx_1
    assert frame_group_4.session == session
    assert len(frame_group_4.instance_groups) == 1
    assert (
        frame_group_4._instance_group_name_registry
        == frame_group_1._instance_group_name_registry
    )

    # TODO(LM): Test underlying dictionaries more thoroughly

    # Test `add_instance_group`
    instance_group: InstanceGroup = frame_group_4.add_instance_group()
    assert isinstance(instance_group, InstanceGroup)
    assert instance_group.frame_idx == frame_idx_1
    assert instance_group.name == "instance_group_1"
    assert len(frame_group_4.instance_groups) == 2
    assert len(instance_group.instances) == 0

    # Test `add_instance`
    video = session.videos[0]
    camera = session.get_camera(video=video)
    labeled_frame = labels.find(video=video, frame_idx=frame_group_4.frame_idx)[0]
    instance = labeled_frame.instances[0]
    frame_group_4.add_instance(
        instance=instance,
        camera=camera,
        instance_group=instance_group,
    )
    assert len(instance_group.instances) == 1

    # Test `__repr__`
    print(frame_group_4)

    # Switch Labels files to one that already contains populated `FrameGroup`s
    labels = multiview_min_session_frame_groups
    session: RecordingSession = labels.sessions[0]
    frame_idx = 0
    frame_group = session.frame_groups[frame_idx]

    # Test `cams_to_include`
    session.cams_to_include = session.cams_to_include[1:]
    assert frame_group.cams_to_include == session.cams_to_include
    assert len(frame_group.cams_to_include) == len(session.linked_cameras) - 1
    with pytest.raises(ValueError):
        frame_group.cams_to_include = session.linked_cameras

    # Test `numpy` method
    frame_group_np = frame_group.numpy()
    n_views, n_inst_groups, n_nodes, n_coords = frame_group_np.shape
    assert n_views == len(frame_group.cams_to_include)
    assert n_inst_groups == len(frame_group.instance_groups)
    assert n_nodes == len(labels.skeleton.nodes)
    assert n_coords == 2
    # Different instance groups should have different coordinates
    assert not np.allclose(frame_group_np[:, 0], frame_group_np[:, 1], equal_nan=True)
    # Different views should have different coordinates
    assert not np.allclose(frame_group_np[0], frame_group_np[1], equal_nan=True)

    # Test `get_instance_group`
    instance_group = frame_group.instance_groups[0]
    camera = session.cameras[0]
    instance = instance_group.get_instance(cam=camera)
    assert frame_group.get_instance_group(instance=instance) == instance_group

    # Test `instance_groups.setter`
    inst_group_to_remove = frame_group.instance_groups[0]
    len_instance_groups = len(frame_group.instance_groups)
    frame_group.instance_groups = frame_group.instance_groups[1:]
    assert inst_group_to_remove not in frame_group.instance_groups
    assert len(frame_group.instance_groups) == len_instance_groups - 1
    # # TODO(LM): Create custom class for `frame_group.instance_groups`
    # frame_group.instance_groups.append(inst_group_to_remove)
    # assert inst_group_to_remove not in frame_group.instance_groups
    frame_group.instance_groups = frame_group.instance_groups + [inst_group_to_remove]
    assert inst_group_to_remove in frame_group.instance_groups

    # Test `remove_instance_group`
    len_instance_groups = len(frame_group.instance_groups)
    frame_group.remove_instance_group(instance_group=inst_group_to_remove)
    assert inst_group_to_remove not in frame_group.instance_groups
    assert len(frame_group.instance_groups) == len_instance_groups - 1
    assert inst_group_to_remove.name not in frame_group._instance_group_name_registry
    for camera, instance in inst_group_to_remove.instance_by_camcorder.items():
        assert instance not in frame_group._instances_by_cam[camera]
    instance_removed = inst_group_to_remove.instances[0]

    # Test `remove_instance`
    frame_group.remove_instance(instance=instance_removed)  # Does nothing
    instance_to_remove = frame_group.instance_groups[0].instances[0]
    labeled_frame_to_remove = instance_to_remove.frame
    camera_to_remove = frame_group.get_camera(labeled_frame=labeled_frame_to_remove)
    for instance_to_remove in labeled_frame_to_remove.instances:
        instance_group = frame_group.get_instance_group(instance=instance_to_remove)
        frame_group.remove_instance(instance=instance_to_remove)
        if instance_group is not None:
            assert camera_to_remove not in instance_group.cameras
            assert instance_to_remove not in instance_group.instances
            assert (
                instance_to_remove
                not in frame_group._instances_by_cam[camera_to_remove]
            )
    assert camera_to_remove not in frame_group.cameras
    assert camera_to_remove not in frame_group._labeled_frame_by_cam
    assert camera_to_remove not in frame_group._cam_by_labeled_frame
    assert labeled_frame_to_remove not in frame_group.labeled_frames
    assert labeled_frame_to_remove not in frame_group._cam_by_labeled_frame

    # Test `set_instance_group_name`
    new_name = "instance_group_2"
    with pytest.raises(ValueError):  # `InstanceGroup` not in this `FrameGroup`
        frame_group.set_instance_group_name(
            instance_group=inst_group_to_remove, name=new_name
        )
    instance_group = frame_group.instance_groups[0]
    old_name = instance_group.name
    frame_group.set_instance_group_name(instance_group=instance_group, name=new_name)
    assert instance_group.name == new_name
    assert new_name in frame_group._instance_group_name_registry
    assert old_name not in frame_group._instance_group_name_registry

    # Test `get_labeled_frame` and `get_camera`
    camera = frame_group.cameras[0]
    labeled_frame = frame_group.get_labeled_frame(camera=camera)
    assert frame_group.get_camera(labeled_frame=labeled_frame) == camera

    # Test `remove_labeled_frame` method and `cameras` and `labeled_frames` properties
    assert camera in frame_group._labeled_frame_by_cam
    assert labeled_frame in frame_group._cam_by_labeled_frame
    assert camera in frame_group.cameras
    assert labeled_frame in frame_group.labeled_frames
    # Test with neither `LabeledFrame` nor `Camera` input
    frame_group.remove_labeled_frame(labeled_frame_or_camera="neither")  # Does nothing
    # Test with `LabeledFrame` input
    frame_group.remove_labeled_frame(labeled_frame_or_camera=labeled_frame)
    assert camera not in frame_group._labeled_frame_by_cam
    assert labeled_frame not in frame_group._cam_by_labeled_frame
    assert camera not in frame_group.cameras
    assert labeled_frame not in frame_group.labeled_frames
    # Test with `Camera` input
    camera = frame_group.cameras[0]
    labeled_frame = frame_group.get_labeled_frame(camera=camera)
    frame_group.remove_labeled_frame(labeled_frame_or_camera=camera)
    assert camera not in frame_group.cameras
    assert labeled_frame not in frame_group.labeled_frames

    # Test `_create_and_add_labeled_frame`
    labeled_frame_created = frame_group._create_and_add_labeled_frame(camera=camera)
    assert labeled_frame.video == session.get_video(camera)
    assert labeled_frame.frame_idx == frame_group.frame_idx
    assert camera in frame_group.cameras
    assert labeled_frame_created in frame_group.labeled_frames
    assert labeled_frame in frame_group.session.labels.labeled_frames

    # Test `upsert_points` (all in bounds, all updated)
    n_cameras = len(frame_group.cams_to_include)
    n_instance_groups = len(frame_group.instance_groups)
    n_nodes = len(frame_group.session.labels.skeleton.nodes)
    n_coords = 2
    value = 100
    points = np.full((n_cameras, n_instance_groups, n_nodes, n_coords), value)
    frame_group.upsert_points(
        points=points, instance_groups=frame_group.instance_groups
    )
    assert np.all(frame_group.numpy(invisible_as_nan=False) == value)

    # Test `upsert_points` (all out of bound, none updated)
    projection_bounds = frame_group.session.projection_bounds
    min_bound = projection_bounds.min()
    prev_value = value
    oob_value = 5000
    assert oob_value > min_bound
    points = np.full((n_cameras, n_instance_groups, n_nodes, n_coords), oob_value)
    frame_group.upsert_points(
        points=points, instance_groups=frame_group.instance_groups
    )
    assert np.any(frame_group.numpy(invisible_as_nan=False) == oob_value) == False
    assert np.all(frame_group.numpy(invisible_as_nan=False) == prev_value)

    # Test `upsert_points` (some out of bound, some updated)
    value = 200
    oob_value = 5000
    assert oob_value > min_bound
    oob_mask = np.random.choice(
        [True, False], size=(n_cameras, n_instance_groups, n_nodes, n_coords)
    )
    points = np.full((n_cameras, n_instance_groups, n_nodes, n_coords), value)
    points[oob_mask] = oob_value
    frame_group.upsert_points(
        points=points, instance_groups=frame_group.instance_groups
    )
    # Get the logical or for either x or y being out of bounds
    oob_mask_1d = np.any(oob_mask, axis=-1)  # Collapse last axis
    oob_mask_1d_expanded = np.expand_dims(oob_mask_1d, axis=-1)
    oob_mask_1d_expanded = np.broadcast_to(oob_mask_1d_expanded, oob_mask.shape)
    frame_group_numpy = frame_group.numpy(invisible_as_nan=False)
    assert np.any(frame_group_numpy > min_bound) == False
    assert np.all(frame_group_numpy[oob_mask_1d_expanded] == prev_value)  # Not updated
    assert np.all(frame_group_numpy[~oob_mask_1d_expanded] == value)  # Updated

    # Test `upsert_points` (between x,y bounds, some out of bound, some updated)
    value = 300
    points = np.full((n_cameras, n_instance_groups, n_nodes, n_coords), value)
    # Reset the points to all in bounds
    frame_group.upsert_points(
        points=points, instance_groups=frame_group.instance_groups
    )
    assert np.all(frame_group.numpy(invisible_as_nan=False) == value)
    # Add some out of bounds points
    prev_value = value
    value = 400
    points = np.full((n_cameras, n_instance_groups, n_nodes, n_coords), value)
    max_bound = projection_bounds.max()
    oob_value = max_bound - 1
    assert oob_value < max_bound and oob_value > min_bound
    oob_mask = np.random.choice(
        [True, False], size=(n_cameras, n_instance_groups, n_nodes, n_coords)
    )
    points[oob_mask] = oob_value
    frame_group.upsert_points(
        points=points, instance_groups=frame_group.instance_groups
    )
    # Get the logical or for either x or y being out of bounds
    bound_x, bound_y = projection_bounds[:, 0].min(), projection_bounds[:, 1].min()
    oob_mask_x = np.where(points[:, :, :, 0] > bound_x, True, False)
    oob_mask_y = np.where(points[:, :, :, 1] > bound_y, True, False)
    oob_mask_1d = np.logical_or(oob_mask_x, oob_mask_y)
    oob_mask_1d_expanded = np.expand_dims(oob_mask_1d, axis=-1)
    oob_mask_1d_expanded = np.broadcast_to(oob_mask_1d_expanded, oob_mask.shape)
    frame_group_numpy = frame_group.numpy(invisible_as_nan=False)
    assert np.any(frame_group_numpy[:, :, :, 0] > bound_x) == False
    assert np.any(frame_group_numpy[:, :, :, 1] > bound_y) == False
    assert np.all(frame_group_numpy[oob_mask_1d_expanded] == prev_value)  # Not updated
    oob_value_mask = np.logical_and(~oob_mask_1d_expanded, oob_mask)
    value_mask = np.logical_and(~oob_mask_1d_expanded, ~oob_mask)
    assert np.all(frame_group_numpy[oob_value_mask] == oob_value)  # Updated to oob
    assert np.all(frame_group_numpy[value_mask] == value)  # Updated to value

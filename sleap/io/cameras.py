"""Module for storing information for camera groups."""
import logging
import tempfile
import toml

import cattr
import numpy as np

from pathlib import Path
from typing import List, Optional, Union, Iterator, Any, Dict, Tuple

from aniposelib.cameras import Camera, FisheyeCamera, CameraGroup
from attrs import define, field
from attrs.validators import deep_iterable, instance_of
from sleap_anipose import triangulate, reproject

# from sleap.io.dataset import Labels  # TODO(LM): Circular import, implement Observer
from sleap.io.video import Video
from sleap.util import deep_iterable_converter


logger = logging.getLogger(__name__)


@define
class Camcorder:
    """Wrapper for `Camera` and `FishEyeCamera` classes.

    Attributes:
        camera: `Camera` or `FishEyeCamera` object.
        videos: List of `Video` objects.
    """

    camera: Union[Camera, FisheyeCamera]
    camera_cluster: "CameraCluster" = None
    _video_by_session: Dict["RecordingSession", Video] = field(factory=dict)

    @property
    def videos(self) -> List[Video]:
        return list(self.camera_cluster._session_by_video.keys())

    @property
    def sessions(self) -> List["RecordingSession"]:
        return list(self._video_by_session.keys())

    def get_video(self, session: "RecordingSession") -> Optional[Video]:
        if session not in self._video_by_session:
            logger.warning(f"{session} not found in {self}.")
            return None
        return self._video_by_session[session]

    def get_session(self, video: Video) -> Optional["RecordingSession"]:
        if video not in self.camera_cluster._session_by_video:
            logger.warning(f"{video} not found in {self}.")
            return None
        return self.camera_cluster._session_by_video[video]

    def __attrs_post_init__(self):
        # Avoid overwriting `CameraCluster` if already set.
        if not isinstance(self.camera_cluster, CameraCluster):
            self.camera_cluster = CameraCluster()

    def __eq__(self, other):
        if not isinstance(other, Camcorder):
            return NotImplemented

        for attr in vars(self):
            other_attr = getattr(other, attr)
            if isinstance(other_attr, np.ndarray):
                if not np.array_equal(getattr(self, attr), other_attr):
                    return False
            elif getattr(self, attr) != other_attr:
                return False

        return True

    def __getattr__(self, attr):
        """Used to grab methods from `Camera` or `FishEyeCamera` objects."""
        if self.camera is None:
            raise AttributeError(
                f"No camera has been specified. "
                f"This is likely because the `Camcorder.from_dict` method was not used to initialize this object. "
                f"Please use `Camcorder.from_dict` to recreate the object."
            )
        return getattr(self.camera, attr)

    def __getitem__(
        self, key: Union[str, "RecordingSession", Video]
    ) -> Union["RecordingSession", Video]:  # Raises KeyError if key not found
        """Return linked `Video` or `RecordingSession`.

        Args:
            key: Key to use for lookup. Can be a `RecordingSession` or `Video` object.

        Returns:
            `Video` or `RecordingSession` object.

        Raises:
            KeyError: If key is not found.
        """

        # If key is a RecordingSession, return the Video
        if isinstance(key, RecordingSession):
            return self._video_by_session[key]

        # If key is a Video, return the RecordingSession
        elif isinstance(key, Video):
            return self.camera_cluster._session_by_video[key]

        raise KeyError(f"Key {key} not found in {self}.")

    def __hash__(self) -> int:
        return hash(self.camera)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, size={self.size})"

    @classmethod
    def from_dict(cls, d) -> "Camcorder":
        """Creates a `Camcorder` object from a dictionary.

        Args:
            d: Dictionary with keys for matrix, dist, size, rvec, tvec, and name.

        Returns:
            `Camcorder` object.
        """
        if "fisheye" in d and d["fisheye"]:
            cam = FisheyeCamera.from_dict(d)
        else:
            cam = Camera.from_dict(d)
        return Camcorder(cam)

    @classmethod
    def from_camera(
        cls, cam: Union[Camera, FisheyeCamera], *args, **kwargs
    ) -> "Camcorder":
        """Creates a `Camcorder` object from a `Camera` or `FishEyeCamera` object.

        Args:
            cam: `Camera` or `FishEyeCamera` object.

        Returns:
            `Camcorder` object.
        """
        # Do not convert if already a Camcorder
        if isinstance(cam, Camcorder):
            return cam

        # Do not convert if not a `Camera` or `FishEyeCamera`
        if not isinstance(cam, Camera):
            raise TypeError(
                f"Expected `Camera` or `FishEyeCamera` object, got {type(cam)}"
            )

        # Convert!
        return Camcorder(cam)


@define
class CameraCluster(CameraGroup):
    """Class for storing information for camera groups.

    Attributes:
        cameras: List of `Camcorder`s.
        metadata: Dictionary of metadata.
        sessions: List of `RecordingSession`s.
        videos: List of `Video`s.
    """

    cameras: List[Camcorder] = field(
        factory=list,
        validator=deep_iterable(
            member_validator=instance_of(Camcorder),
            iterable_validator=instance_of(list),
        ),
        converter=deep_iterable_converter(
            member_converter=Camcorder.from_camera,
            iterable_converter=list,
        ),
    )
    metadata: dict = field(factory=dict)
    _videos_by_session: Dict["RecordingSession", List[Video]] = field(factory=dict)
    _session_by_video: Dict[Video, "RecordingSession"] = field(factory=dict)
    _camcorder_by_video: Dict[Video, Camcorder] = field(factory=dict)

    @property
    def sessions(self) -> List["RecordingSession"]:
        """List of `RecordingSession`s."""
        return list(self._videos_by_session.keys())

    @property
    def videos(self) -> List[Video]:
        """List of `Video`s."""
        return list(self._session_by_video.keys())

    def get_videos_from_session(
        self, session: "RecordingSession"
    ) -> Optional[List[Video]]:
        """Get `Video`s from `RecordingSession` object.

        Args:
            session: `RecordingSession` object.

        Returns:
            List of `Video` objects or `None` if not found.
        """
        if session not in self.sessions:
            logger.warning(
                f"RecordingSession not linked to {self}. "
                "Use `self.add_session(session)` to add it."
            )
            return None
        return self._videos_by_session[session]

    def get_session_from_video(self, video: Video) -> Optional["RecordingSession"]:
        """Get `RecordingSession` from `Video` object.

        Args:
            video: `Video` object.

        Returns:
            `RecordingSession` object or `None` if not found.
        """
        if video not in self.videos:
            logger.warning(f"Video not linked to any RecordingSession in {self}.")
            return None
        return self._session_by_video[video]

    def get_camcorder_from_video(self, video: Video) -> Optional[Camcorder]:
        """Get `Camcorder` from `Video` object.

        Args:
            video: `Video` object.

        Returns:
            `Camcorder` object or `None` if not found.
        """
        if video not in self.videos:
            logger.warning(f"Video not linked to any Camcorders in {self}.")
            return None
        return self._camcorder_by_video[video]

    def get_videos_from_camcorder(self, camcorder: Camcorder) -> List[Video]:
        """Get `Video`s from `Camcorder` object.

        Args:
            camcorder: `Camcorder` object.

        Returns:
            List of `Video` objects.

        Raises:
            ValueError: If `camcorder` is not in `self.cameras`.
        """
        if camcorder not in self.cameras:
            raise ValueError(f"Camcorder not in {self}.")
        return camcorder.videos

    def add_session(self, session: "RecordingSession"):
        """Adds a `RecordingSession` to the `CameraCluster`."""
        self._videos_by_session[session] = []
        session.camera_cluster = self

    def __attrs_post_init__(self):
        """Initialize `CameraCluster` object."""
        super().__init__(cameras=self.cameras, metadata=self.metadata)
        for cam in self.cameras:
            cam.camera_cluster = self

    def __contains__(self, item):
        return item in self.cameras

    def __iter__(self) -> Iterator[Camcorder]:
        return iter(self.cameras)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(
        self, idx_or_key: Union[int, Video, Camcorder, "RecordingSession", str]
    ) -> Optional[
        Union[Camcorder, Tuple[Camcorder, Video], List[Video], "RecordingSession", Any]
    ]:
        """Get item from `CameraCluster`.

        Args:
            idx_or_key: Index, `Video`, `Camcorder`, `RecordingSession`, or `str` name.

        Returns:
            `Camcorder`, (`Camcorder`, `Video`), `List[Video]`, `RecordingSession`,
            metadata value, or None if not found.

        Raises:
            ValueError: If `idx_or_key` used as a metadata key and not found or
                `idx_or_key` is a `Camcorder` which is not in `self.cameras`.
        """

        # If key is int, index into cameras -> Camcorder
        if isinstance(idx_or_key, int):
            return self.cameras[idx_or_key]

        # If key is Video, return linked
        # (Camcorder, RecordingSession) -> Optional[Tuple[Camcorder, Video]]
        elif isinstance(idx_or_key, Video):
            camcorder = self.get_camcorder_from_video(idx_or_key)
            session = self.get_session_from_video(idx_or_key)
            if camcorder is None or session is None:
                return None
            return (camcorder, session)

        # If key is Camcorder, return linked Videos -> Optional[List[Video]]
        elif isinstance(idx_or_key, Camcorder):
            return self.get_videos_from_camcorder(idx_or_key)

        # If key is RecordingSession, return linked Videos -> Optional[List[Video]]
        elif isinstance(idx_or_key, RecordingSession):
            return self.get_videos_from_session(idx_or_key)

        # Last resort: look in metadata for matching key -> Any
        elif idx_or_key in self.metadata:
            return self.metadata[idx_or_key]

        # Raise error if not found
        else:
            raise KeyError(
                f"Key {idx_or_key} not found in {self.__class__.__name__} or "
                "associated metadata."
            )

    def __repr__(self):
        message = (
            f"{self.__class__.__name__}(sessions={len(self.sessions)}, "
            f"cameras={len(self)}: "
        )
        for cam in self:
            message += f"{cam.name}, "
        return f"{message[:-2]})"

    @classmethod
    def load(cls, filename) -> "CameraCluster":
        """Loads cameras as `Camcorder`s from a calibration.toml file.

        Args:
            filename: Path to calibration.toml file.

        Returns:
            `CameraCluster` object.
        """
        cgroup: CameraGroup = super().load(filename)
        return cls(cameras=cgroup.cameras, metadata=cgroup.metadata)

    @classmethod
    def from_calibration_dict(cls, calibration_dict: Dict[str, str]) -> "CameraCluster":
        """Structure a cluster dictionary to a `CameraCluster`.

        This method is intended to be used for restructuring a `CameraCluster` object
        (that was previously unstructured to a serializable format). Note: this method
        does not handle any mapping between `Video`s, `RecordingSession`s, and
        `Camcorder`s.

        Args:
            calibration_dict: A dictionary containing just the calibration info needed
                to partially restructure a `CameraCluster` (no mapping between `Video`s,
                `RecordingSession`s, and `Camcorder`s).

        Returns:
            `CameraCluster` object.
        """

        # Save the calibration dictionary to a temp file and load as `CameraGroup`
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = str(Path(temp_dir, "calibration.toml"))
            with open(temp_file, "w") as f:
                toml.dump(calibration_dict, f)
            cgroup: CameraGroup = super().load(temp_file)

        return cls(cameras=cgroup.cameras, metadata=cgroup.metadata)

    def to_calibration_dict(self) -> Dict[str, str]:
        """Unstructure the `CameraCluster` object to a dictionary.

        This method is intended to be used for unstructuring a `CameraCluster` object
        to a serializable format. Note: this method does not save any mapping between
        `Video`s, `RecordingSession`s, and `Camcorders`.

        Returns:
            Dictionary of `CameraCluster` object.
        """

        # Use existing `CameraGroup.dump` method to get the calibration dictionary
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = str(Path(temp_dir, "calibration.toml"))
            self.dump(fname=temp_file)
            calibration_dict = toml.load(temp_file)

        return calibration_dict


@define(eq=False)
class RecordingSession:
    """Class for storing information for a recording session.

    Attributes:
        camera_cluster: `CameraCluster` object.
        metadata: Dictionary of metadata.
        videos: List of `Video`s that have been linked to a `Camcorder` in the
            `self.camera_cluster`.
        linked_cameras: List of `Camcorder`s in the `self.camera_cluster` that are
            linked to a `Video`.
        unlinked_cameras: List of `Camcorder`s in the `self.camera_cluster` that are
            not linked to a `Video`.
    """

    # TODO(LM): Consider implementing Observer pattern for `camera_cluster` and `labels`
    camera_cluster: CameraCluster = field(factory=CameraCluster)
    metadata: dict = field(factory=dict)
    _video_by_camcorder: Dict[Camcorder, Video] = field(factory=dict)
    labels: Optional["Labels"] = None

    @property
    def videos(self) -> List[Video]:
        """List of `Video`s."""

        return self.camera_cluster._videos_by_session[self]

    @property
    def linked_cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s in `self.camera_cluster` that are linked to a video."""

        return list(self._video_by_camcorder.keys())

    @property
    def unlinked_cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s in `self.camera_cluster` that are not linked to a video."""

        return list(set(self.camera_cluster.cameras) - set(self.linked_cameras))

    def get_video(self, camcorder: Camcorder) -> Optional[Video]:
        """Retrieve `Video` linked to `Camcorder`.

        Args:
            camcorder: `Camcorder` object.

        Returns:
            If `Camcorder` in `self.camera_cluster`, then `Video` object if found, else
            `None` if `Camcorder` has no linked `Video`.

        Raises:
            ValueError: If `Camcorder` is not in `self.camera_cluster`.
        """

        if camcorder not in self.camera_cluster:
            raise ValueError(
                f"Camcorder {camcorder.name} is not in this RecordingSession's "
                f"{self.camera_cluster}."
            )

        if camcorder not in self._video_by_camcorder:
            logger.warning(
                f"Camcorder {camcorder.name} is not linked to a video in this "
                f"RecordingSession."
            )
            return None

        return self._video_by_camcorder[camcorder]

    def get_camera(self, video: Video) -> Optional[Camcorder]:
        """Retrieve `Camcorder` linked to `Video`.

        Args:
            video: `Video` object.

        Returns:
            `Camcorder` object if found, else `None`.
        """

        if video not in self.camera_cluster._camcorder_by_video:
            logger.warning(
                f"{video} is not linked to a Camcorder in this "
                f"RecordingSession's {self.camera_cluster}."
            )
            return None

        return self.camera_cluster._camcorder_by_video[video]

    def add_video(self, video: Video, camcorder: Camcorder):
        """Adds a `Video` to the `RecordingSession`.

        Args:
            video: `Video` object.
            camcorder: `Camcorder` object.
        """

        # Ensure the `Camcorder` is in this `RecordingSession`'s `CameraCluster`
        try:
            assert camcorder in self.camera_cluster
        except AssertionError:
            raise ValueError(
                f"Camcorder {camcorder.name} is not in this RecordingSession's "
                f"{self.camera_cluster}."
            )

        # Add session-to-videos (1-to-many) map to `CameraCluster`
        if self not in self.camera_cluster._videos_by_session:
            self.camera_cluster.add_session(self)
        if video not in self.camera_cluster._videos_by_session[self]:
            self.camera_cluster._videos_by_session[self].append(video)

        # Add session-to-video (1-to-1) map to `Camcorder`
        if video not in camcorder._video_by_session:
            camcorder._video_by_session[self] = video

        # Add video-to-session (1-to-1) map to `CameraCluster`
        self.camera_cluster._session_by_video[video] = self

        # Add video-to-camcorder (1-to-1) map to `CameraCluster`
        if video not in self.camera_cluster._camcorder_by_video:
            self.camera_cluster._camcorder_by_video[video] = []
        self.camera_cluster._camcorder_by_video[video] = camcorder

        # Add camcorder-to-video (1-to-1) map to `RecordingSession`
        self._video_by_camcorder[camcorder] = video

        # Update labels cache
        if self.labels is not None:
            self.labels.update_session(self, video)

    def remove_video(self, video: Video):
        """Removes a `Video` from the `RecordingSession`.

        Args:
            video: `Video` object.
        """

        # Remove video-to-camcorder map from `CameraCluster`
        camcorder = self.camera_cluster._camcorder_by_video.pop(video)

        # Remove video-to-session map from `CameraCluster`
        self.camera_cluster._session_by_video.pop(video)

        # Remove session-to-video(s) maps from related `CameraCluster` and `Camcorder`
        self.camera_cluster._videos_by_session[self].remove(video)
        camcorder._video_by_session.pop(self)

        # Remove camcorder-to-video map from `RecordingSession`
        self._video_by_camcorder.pop(camcorder)

        # Update labels cache
        if self.labels is not None and self.labels.get_session(video) is not None:
            self.labels.remove_session_video(self, video)

    def get_instances_accross_views(
        self, frame_idx: int, track: Optional["Track"] = None
    ) -> List["LabeledFrame"]:
        """Get all `Instances` accross all views at a given frame index.

        Args:
            frame_idx: Frame index to get instances from (0-indexed).
            track: `Track` object used to find instances accross views.

        Returns:
            List of `Instances` objects.
        """

        views: List["LabeledFrame"] = []
        instances: List["Instances"] = []

        # Get all views at this frame index
        for video in self.videos:
            lfs: List["LabeledFrame"] = self.labels.get((video, [frame_idx]))
            if len(lfs) == 0:
                continue

            lf = lfs[0]
            if len(lf.instances) > 0:
                views.append(lf)

        # If no views, then return
        if len(views) <= 1:
            logger.warning(
                "One or less views found for frame "
                f"{frame_idx} in {self.camera_cluster}."
            )
            return instances

        # Find all instance accross all views
        instances: List["Instances"] = []
        for lf in views:
            insts = lf.find(track=track)
            if len(insts) > 0:
                instances.append(insts[0])

        return instances

    def calculate_reprojected_points(self, instances: List["Instances"]):
        """Triangulate and reproject instance coordinates.

        Args:
            instances: List of `Instances` objects.

        Returns:
            List of reprojected instance coordinates. Each element in the list is a
            numpy array of shape (1, N, 2) where N is the number of nodes.
        """

        # Gather instances into M x F x T x N x 2 arrays
        # (M = # views, F = # frames = 1, T = # tracks = 1, N = # nodes, 2 = x, y)
        inst_coords = np.stack([inst.numpy() for inst in instances], axis=0)
        inst_coords = np.expand_dims(inst_coords, axis=1)
        inst_coords = np.expand_dims(inst_coords, axis=1)
        points_3d = triangulate(p2d=inst_coords, calib=self.camera_cluster)

        # Update the views with the new 3D points
        inst_coords_reprojected = reproject(points_3d, calib=self.camera_cluster)
        insts_coords_list: List[np.ndarray] = np.split(
            inst_coords_reprojected.squeeze(), inst_coords_reprojected.shape[0], axis=0
        )

        return insts_coords_list

    def update_views(
        self,
        frame_idx: int,
        track: Optional["Track"] = None,
        cams_to_include: Optional[List[int]] = None,
    ):
        """Update the views of the `RecordingSession`.

        Args:
            frame_idx: Frame index to update (0-indexed).
            track: `Track` object used to find instances accross views for updating.
            cams_to_include: List of views by indices in `self.camera_cluster.cameras` (0-indexed).
        """

        # TODO(LM): Add support for taking in `cams_to_include` to use for triangulation

        # Get all views at this frame index
        instances = self.get_instances_accross_views(frame_idx, track=track)

        # If no instances, then return
        if len(instances) <= 1:
            logger.warning(
                "One or less instances found for frame "
                f"{frame_idx} in {self.camera_cluster}."
            )
            return

        # Triangulate, reproject, and update coordinates
        insts_coords_list: List[np.ndarray] = self.calculate_reprojected_points(
            instances
        )
        for inst, inst_coord in zip(instances, insts_coords_list):
            inst.update_points(
                inst_coord[0], exclude_complete=True
            )  # inst_coord is (1, N, 2)

    def __attrs_post_init__(self):
        self.camera_cluster.add_session(self)

    def __iter__(self) -> Iterator[List[Camcorder]]:
        return iter(self.camera_cluster)

    def __len__(self):
        return len(self.videos)

    def __getattr__(self, attr: str) -> Any:

        """Try to find the attribute in the camera_cluster next."""
        return getattr(self.camera_cluster, attr)

    def __getitem__(
        self, idx_or_key: Union[int, Video, Camcorder, str]
    ) -> Union[Camcorder, Video, Any]:
        """Grab a `Camcorder`, `Video`, or metadata from the `RecordingSession`.

        Try to index into `camera_cluster.cameras` first, then check
        video-to-camera map and camera-to-video map. Lastly check in the `metadata`s.
        """

        # Try to find in `self.camera_cluster.cameras`
        if isinstance(idx_or_key, int):
            try:
                return self.camera_cluster[idx_or_key]
            except IndexError:
                pass  # Try to find in metadata

        # Return a `Camcorder` if `idx_or_key` is a `Video
        if isinstance(idx_or_key, Video):
            return self.get_camera(idx_or_key)

        # Return a `Video` if `idx_or_key` is a `Camcorder`
        elif isinstance(idx_or_key, Camcorder):
            return self.get_video(idx_or_key)

        # Try to find in `self.metadata`
        elif idx_or_key in self.metadata:
            return self.metadata[idx_or_key]

        # Try to find in `self.camera_cluster.metadata`
        elif idx_or_key in self.camera_cluster.metadata:
            return self.camera_cluster.metadata[idx_or_key]

        # Raise error if not found
        else:
            raise KeyError(
                f"Key {idx_or_key} not found in {self.__class__.__name__} or "
                "associated metadata."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(camera_cluster={self.camera_cluster})"

    @classmethod
    def load(
        cls,
        filename,
        metadata: Optional[dict] = None,
    ) -> "RecordingSession":
        """Loads cameras as `Camcorder`s from a calibration.toml file.

        Args:
            filename: Path to calibration.toml file.
            metadata: Dictionary of metadata.

        Returns:
            `RecordingSession` object.
        """

        camera_cluster: CameraCluster = CameraCluster.load(filename)
        return cls(
            camera_cluster=camera_cluster,
            metadata=(metadata or {}),
        )

    @classmethod
    def from_calibration_dict(cls, calibration_dict: dict) -> "RecordingSession":
        """Loads cameras as `Camcorder`s from a calibration dictionary.

        Args:
            calibration_dict: Dictionary of calibration data.

        Returns:
            `RecordingSession` object.
        """

        camera_cluster: CameraCluster = CameraCluster.from_calibration_dict(
            calibration_dict
        )
        return cls(camera_cluster=camera_cluster)

    def to_session_dict(self, video_to_idx: Dict[Video, int]) -> dict:
        """Unstructure `RecordingSession` to an invertible dictionary.

        Returns:
            Dictionary of "calibration" and "camcorder_to_video_idx_map" needed to
            restructure a `RecordingSession`.
        """

        # Unstructure `CameraCluster` and `metadata`
        calibration_dict = self.camera_cluster.to_calibration_dict()

        # Store camcorder-to-video indices map where key is camcorder index
        # and value is video index from `Labels.videos`
        camcorder_to_video_idx_map = {}
        for cam_idx, camcorder in enumerate(self.camera_cluster):

            # Skip if Camcorder is not linked to any Video
            if camcorder not in self._video_by_camcorder:
                continue

            # Get video index from `Labels.videos`
            video = self._video_by_camcorder[camcorder]
            video_idx = video_to_idx.get(video, None)

            if video_idx is not None:
                camcorder_to_video_idx_map[cam_idx] = video_idx
            else:
                logger.warning(
                    f"Video {video} not found in `Labels.videos`. "
                    "Not saving to `RecordingSession` serialization."
                )

        return {
            "calibration": calibration_dict,
            "camcorder_to_video_idx_map": camcorder_to_video_idx_map,
        }

    @classmethod
    def from_session_dict(
        cls, session_dict, videos_list: List[Video]
    ) -> "RecordingSession":
        """Restructure `RecordingSession` from an invertible dictionary.

        Args:
            session_dict: Dictionary of "calibration" and "camcorder_to_video_idx_map"
                needed to fully restructure a `RecordingSession`.
            videos_list: List containing `Video` objects (expected `Labels.videos`).

        Returns:
            `RecordingSession` object.
        """

        # Restructure `RecordingSession` without `Video` to `Camcorder` mapping
        calibration_dict = session_dict["calibration"]
        session: RecordingSession = RecordingSession.from_calibration_dict(
            calibration_dict
        )

        # Retrieve all `Camcorder` and `Video` objects, then add to `RecordingSession`
        camcorder_to_video_idx_map = session_dict["camcorder_to_video_idx_map"]
        for cam_idx, video_idx in camcorder_to_video_idx_map.items():
            camcorder = session.camera_cluster.cameras[cam_idx]
            video = videos_list[video_idx]
            session.add_video(video, camcorder)

        return session

    @staticmethod
    def make_cattr(videos_list: List[Video]):
        """Make a `cattr.Converter` for `RecordingSession` serialization.

        Args:
            videos_list: List containing `Video` objects (expected `Labels.videos`).

        Returns:
            `cattr.Converter` object.
        """
        sessions_cattr = cattr.Converter()
        sessions_cattr.register_structure_hook(
            RecordingSession,
            lambda x, cls: RecordingSession.from_session_dict(x, videos_list),
        )

        video_to_idx = {video: i for i, video in enumerate(videos_list)}
        sessions_cattr.register_unstructure_hook(
            RecordingSession, lambda x: x.to_session_dict(video_to_idx)
        )
        return sessions_cattr

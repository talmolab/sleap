"""Module for storing information for camera groups."""

from typing import List, Optional, Union, Iterator, Any, Dict

from aniposelib.cameras import Camera, FisheyeCamera, CameraGroup
from attrs import define, field
from attrs.validators import deep_iterable, instance_of
import numpy as np

from sleap.util import deep_iterable_converter
from sleap.io.video import Video


@define
class Camcorder:
    """Wrapper for `Camera` and `FishEyeCamera` classes.

    Attributes:
        camera: `Camera` or `FishEyeCamera` object.
        videos: List of `Video` objects.
    """

    camera: Optional[Union[Camera, FisheyeCamera]] = field(factory=None)
    _video_by_session: Dict["RecordingSession", Video] = field(factory=dict)

    @property
    def videos(self) -> List[Video]:
        return list(set(self._video_by_session.values()))

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
        return getattr(self.camera, attr)

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

    def add_session(self, session: "RecordingSession"):
        """Adds a `RecordingSession` to the cluster."""
        self._videos_by_session[session] = []

    def __attrs_post_init__(self):
        super().__init__(cameras=self.cameras, metadata=self.metadata)

    def __contains__(self, item):
        return item in self.cameras

    def __iter__(self) -> Iterator[List[Camcorder]]:
        return iter(self.cameras)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

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
        cam_group: CameraGroup = super().load(filename)
        return cls(cameras=cam_group.cameras, metadata=cam_group.metadata)


@define(eq=False)
class RecordingSession:
    """Class for storing information for a recording session.

    Attributes:
        camera_cluster: `CameraCluster` object.
        metadata: Dictionary of metadata.
        videos: List of `Video`s.
    """

    camera_cluster: CameraCluster = field(factory=CameraCluster)
    metadata: dict = field(factory=dict)
    _video_by_camcorder: Dict[Camcorder, Video] = field(factory=dict)

    @property
    def videos(self) -> List[Video]:
        """List of `Video`s."""
        return self.camera_cluster._videos_by_session[self]

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

    def __attrs_post_init__(self):
        self.camera_cluster.add_session(self)

    def __iter__(self) -> Iterator[List[Camcorder]]:
        return iter(self.camera_cluster)

    def __len__(self):
        return len(self.videos)

    def __getattr__(self, attr: str) -> Any:
        """Try to find the attribute in the camera_cluster next."""
        return getattr(self.camera_cluster, attr)

    def __getitem__(self, idx_or_key: Union[int, str]):
        """Try to find item in `camera_cluster.cameras` first, then in `metadata`s."""
        # Try to find in `self.camera_cluster.cameras`
        if isinstance(idx_or_key, int):
            try:
                return self.camera_cluster[idx_or_key]
            except IndexError:
                pass  # Try to find in metadata

        # Try to find in `self.metadata`
        if idx_or_key in self.metadata:
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

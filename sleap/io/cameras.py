"""Module for storing information for camera groups."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import cattr
import numpy as np
import toml
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
from attrs import define, field
from attrs.validators import deep_iterable, instance_of

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


@define
class InstanceGroup:
    """Defines a group of instances across the same frame index.

    Args:
        camera_cluster: `CameraCluster` object.
        instances: List of `Instance` objects.

    """

    frame_idx: int = field(validator=instance_of(int))
    camera_cluster: Optional[CameraCluster] = None
    locked: bool = field(default=False)
    _instance_by_camcorder: Dict[Camcorder, "Instance"] = field(factory=dict)
    _camcorder_by_instance: Dict["Instance", Camcorder] = field(factory=dict)
    _dummy_instance: Optional["Instance"] = field(default=None)

    def __attrs_post_init__(self):
        """Initialize `InstanceGroup` object."""

        for cam, instance in self._instance_by_camcorder.items():
            self._camcorder_by_instance[instance] = cam

        # Create a dummy instance to fill in for missing instances
        if self._dummy_instance is None:

            # Get `Instance.from_numpy` method
            if hasattr(instance, "score"):
                # The example instance is a `PredictedInstance`
                from_numpy = instance.__class__.__bases__[0].from_numpy
            else:
                # The example instance is an `Instance`
                from_numpy = instance.__class__.from_numpy

            skeleton: "Skeleton" = instance.skeleton
            self._dummy_instance = from_numpy(
                points=np.full(
                    shape=(len(skeleton.nodes), 2),
                    fill_value=np.nan,
                ),
                skeleton=skeleton,
            )

    @property
    def instances(self) -> List["Instance"]:
        """List of `Instance` objects."""
        return list(self._instance_by_camcorder.values())

    @property
    def cameras(self) -> List[Camcorder]:
        """List of `Camcorder` objects."""
        return list(self._instance_by_camcorder.keys())

    def numpy(self) -> np.ndarray:
        """Return instances as a numpy array of shape (n_views, n_nodes, 2).
        The ordering of views is based on the ordering of `Camcorder`s in the
        `self.camera_cluster: CameraCluster`.
        If an instance is missing for a `Camcorder`, then the instance is filled in with
        the dummy instance (all NaNs).
        Returns:
            Numpy array of shape (n_views, n_nodes, 2).
        """

        instance_numpys: List[np.ndarray] = []  # len(M) x N x 2
        for cam in self.camera_cluster.cameras:
            instance = self.get_instance(cam) or self._dummy_instance
            instance_numpy: np.ndarray = instance.numpy()  # N x 2
            instance_numpys.append(instance_numpy)

        return np.stack(instance_numpys, axis=0)  # M x N x 2

    def create_and_add_instance(self, cam: Camcorder, labeled_frame: "LabeledFrame"):
        """Create an `Instance` at a labeled_frame and add it to the `InstanceGroup`.

        Args:
            cam: `Camcorder` object that the `Instance` is for.
            labeled_frame: `LabeledFrame` object that the `Instance` is contained in.

        Returns:
            `Instance` created and added to the `InstanceGroup`.
        """

        # Get the `Skeleton`
        skeleton: "Skeleton" = self._dummy_instance.skeleton

        # Create an all nan `Instance`
        instance: "Instance" = self._dummy_instance.__class__(
            skeleton=skeleton,
            frame=labeled_frame,
        )

        # Add the instance to the `InstanceGroup`
        self.add_instance(cam, instance)

        return instance

    def add_instance(self, cam: Camcorder, instance: "Instance"):
        """Add an `Instance` to the `InstanceGroup`.

        Args:
            cam: `Camcorder` object that the `Instance` is for.
            instance: `Instance` object to add.

        Raises:
            ValueError: If the `Camcorder` is not in the `CameraCluster`.
            ValueError: If the `Instance` is already in the `InstanceGroup` at another
                camera.
        """

        # Ensure the `Camcorder` is in the `CameraCluster`
        self._raise_if_cam_not_in_cluster(cam=cam)

        # Ensure the `Instance` is not already in the `InstanceGroup` at another camera
        if (
            instance in self._camcorder_by_instance
            and self._camcorder_by_instance[instance] != cam
        ):
            raise ValueError(
                f"Instance {instance} is already in this InstanceGroup at camera "
                f"{self.get_instance(instance)}."
            )

        # Add the instance to the `InstanceGroup`
        self.replace_instance(cam, instance)

    def replace_instance(self, cam: Camcorder, instance: "Instance"):
        """Replace an `Instance` in the `InstanceGroup`.

        If the `Instance` is already in the `InstanceGroup`, then it is removed and
        replaced. If the `Instance` is not already in the `InstanceGroup`, then it is
        added.

        Args:
            cam: `Camcorder` object that the `Instance` is for.
            instance: `Instance` object to replace.

        Raises:
            ValueError: If the `Camcorder` is not in the `CameraCluster`.
        """

        # Ensure the `Camcorder` is in the `CameraCluster`
        self._raise_if_cam_not_in_cluster(cam=cam)

        # Remove the instance if it already exists
        self.remove_instance(instance_or_cam=instance)

        # Replace the instance in the `InstanceGroup`
        self._instance_by_camcorder[cam] = instance
        self._camcorder_by_instance[instance] = cam

    def remove_instance(self, instance_or_cam: Union["Instance", Camcorder]):
        """Remove an `Instance` from the `InstanceGroup`.

        Args:
            instance_or_cam: `Instance` or `Camcorder` object to remove from
                `InstanceGroup`.

        Raises:
            ValueError: If the `Camcorder` is not in the `CameraCluster`.
        """

        if isinstance(instance_or_cam, Camcorder):
            cam = instance_or_cam

            # Ensure the `Camcorder` is in the `CameraCluster`
            self._raise_if_cam_not_in_cluster(cam=cam)

            # Remove the instance from the `InstanceGroup`
            if cam in self._instance_by_camcorder:
                instance = self._instance_by_camcorder.pop(cam)
                self._camcorder_by_instance.pop(instance)

        else:
            # The input is an `Instance`
            instance = instance_or_cam

            # Remove the instance from the `InstanceGroup`
            if instance in self._camcorder_by_instance:
                cam = self._camcorder_by_instance.pop(instance)
                self._instance_by_camcorder.pop(cam)
            else:
                logger.debug(
                    f"Instance {instance} not found in this InstanceGroup {self}."
                )

    def _raise_if_cam_not_in_cluster(self, cam: Camcorder):
        """Raise a ValueError if the `Camcorder` is not in the `CameraCluster`."""

        if cam not in self.camera_cluster:
            raise ValueError(
                f"Camcorder {cam} is not in this InstanceGroup's "
                f"{self.camera_cluster}."
            )

    def get_instance(self, cam: Camcorder) -> Optional["Instance"]:
        """Retrieve `Instance` linked to `Camcorder`.

        Args:
            camcorder: `Camcorder` object.

        Returns:
            If `Camcorder` in `self.camera_cluster`, then `Instance` object if found, else
            `None` if `Camcorder` has no linked `Instance`.
        """

        if cam not in self._instance_by_camcorder:
            logger.debug(
                f"Camcorder {cam} has no linked `Instance` in this `InstanceGroup` "
                f"{self}."
            )
            return None

        return self._instance_by_camcorder[cam]

    def get_instances(self, cams: List[Camcorder]) -> List["Instance"]:
        instances = []
        for cam in cams:
            instance = self.get_instance(cam)
            instances.append(instance)
        return instance

    def get_cam(self, instance: "Instance") -> Optional[Camcorder]:
        """Retrieve `Camcorder` linked to `Instance`.

        Args:
            instance: `Instance` object.

        Returns:
            `Camcorder` object if found, else `None`.
        """

        if instance not in self._camcorder_by_instance:
            logger.debug(
                f"{instance} is not in this InstanceGroup.instances: "
                f"\n\t{self.instances}."
            )
            return None

        return self._camcorder_by_instance[instance]

    def update_points(
        self,
        points: np.ndarray,
        cams_to_include: Optional[List[Camcorder]] = None,
        exclude_complete: bool = True,
    ):
        """Update the points in the `Instance` for the specified `Camcorder`s.

        Args:
            points: Numpy array of shape (M, N, 2) where M is the number of views, N is
                the number of Nodes, and 2 is for x, y.
            cams_to_include: List of `Camcorder`s to include in the update. The order of
                the `Camcorder`s in the list should match the order of the views in the
                `points` array. If None, then all `Camcorder`s in the `CameraCluster`
                are included. Default is None.
            exclude_complete: If True, then do not update points that are marked as
                complete. Default is True.
        """

        # If no `Camcorder`s specified, then update `Instance`s for all `CameraCluster`
        if cams_to_include is None:
            cams_to_include = self.camera_cluster.cameras

        # Check that correct shape was passed in
        n_views, n_nodes, _ = points.shape
        assert n_views == len(cams_to_include), (
            f"Number of views in `points` ({n_views}) does not match the number of "
            f"Camcorders in `cams_to_include` ({len(cams_to_include)})."
        )

        for cam_idx, cam in enumerate(cams_to_include):
            # Get the instance for the cam
            instance: Optional["Instance"] = self.get_instance(cam)
            if instance is None:
                logger.warning(
                    f"Camcorder {cam.name} not found in this InstanceGroup's instances."
                )
                continue

            # Update the points for the instance
            instance.update_points(
                points=points[cam_idx, :, :], exclude_complete=exclude_complete
            )

    def __getitem__(
        self, idx_or_key: Union[int, Camcorder, "Instance"]
    ) -> Union[Camcorder, "Instance"]:
        """Grab a `Camcorder` of `Instance` from the `InstanceGroup`."""

        def _raise_key_error():
            raise KeyError(f"Key {idx_or_key} not found in {self.__class__.__name__}.")

        # Try to find in `self.camera_cluster.cameras`
        if isinstance(idx_or_key, int):
            try:
                return self.instances[idx_or_key]
            except IndexError:
                _raise_key_error()

        # Return a `Instance` if `idx_or_key` is a `Camcorder``
        if isinstance(idx_or_key, Camcorder):
            return self.get_instance(idx_or_key)

        else:
            # isinstance(idx_or_key, "Instance"):
            try:
                return self.get_cam(idx_or_key)
            except:
                pass

        _raise_key_error()

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return f"{self.__class__.__name__}(frame_idx={self.frame_idx}, instances={len(self)}, camera_cluster={self.camera_cluster})"

    @classmethod
    def from_dict(cls, d: dict) -> Optional["InstanceGroup"]:
        """Creates an `InstanceGroup` object from a dictionary.

        Args:
            d: Dictionary with `Camcorder` keys and `Instance` values.

        Returns:
            `InstanceGroup` object or None if no "real" (determined by `frame_idx` other
            than None) instances found.
        """

        # Ensure not to mutate the original dictionary
        d_copy = d.copy()

        frame_idx = None
        for cam, instance in d_copy.copy().items():
            camera_cluster = cam.camera_cluster

            # Remove dummy instances (determined by not having a frame index)
            if instance.frame_idx is None:
                d_copy.pop(cam)
            # Grab the frame index from non-dummy instances
            elif frame_idx is None:
                frame_idx = instance.frame_idx
            # Ensure all instances have the same frame index
            else:
                try:
                    assert frame_idx == instance.frame_idx
                except AssertionError:
                    logger.warning(
                        f"Cannot create `InstanceGroup`: Frame index {frame_idx} "
                        f"does not match instance frame index {instance.frame_idx}."
                    )

        if len(d_copy) == 0:
            logger.warning("Cannot create `InstanceGroup`: No real instances found.")
            return None

        frame_idx = cast(
            int, frame_idx
        )  # Could be None if no real instances in dictionary

        return cls(
            frame_idx=frame_idx,
            camera_cluster=camera_cluster,
            instance_by_camcorder=d_copy,
        )


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
    _instance_groups_by_frame_idx: Dict[int, InstanceGroup] = field(factory=dict)

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

    @property
    def instance_groups(self) -> Dict[int, InstanceGroup]:
        """Dict of `InstanceGroup`s by frame index."""

        return self._instance_groups_by_frame_idx

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

    def get_videos_from_selected_cameras(
        self, cams_to_include: Optional[List[Camcorder]] = None
    ) -> Dict[Camcorder, Video]:
        """Get all `Video`s from selected `Camcorder`s.

        Args:
            cams_to_include: List of `Camcorder`s to include. Defualt is all.

        Returns:
            Dictionary with `Camcorder` key and `Video` value.
        """

        # If no `Camcorder`s specified, then return all videos in session
        if cams_to_include is None:
            return self._video_by_camcorder

        # Get all videos from selected `Camcorder`s
        videos: Dict[Camcorder, Video] = {}
        for cam in cams_to_include:
            video = self.get_video(cam)
            if video is not None:
                videos[cam] = video

        return videos

    def get_instance_group(self, frame_idx: int) -> Optional[InstanceGroup]:
        """Get `InstanceGroup` from frame index.

        Args:
            frame_idx: Frame index.

        Returns:
            `InstanceGroup` object or `None` if not found.
        """

        if frame_idx not in self.instance_groups:
            logger.warning(
                f"Frame index {frame_idx} not found in this RecordingSession's "
                f"InstanceGroup's keys: \n\t{self.instance_groups.keys()}."
            )
            return None

        return self.instance_groups[frame_idx]

    def update_instance_group(self, frame_idx: int, instance_group: InstanceGroup):
        """Update `InstanceGroup` from frame index.

        Args:
            frame_idx: Frame index.
            instance_groups: `InstanceGroup` object.
        """

        self._instance_groups_by_frame_idx[frame_idx] = instance_group

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
        return (
            f"{self.__class__.__name__}(videos:{len(self.videos)},"
            f"camera_cluster={self.camera_cluster})"
        )

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
                camcorder_to_video_idx_map[str(cam_idx)] = str(video_idx)
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
            camcorder = session.camera_cluster.cameras[int(cam_idx)]
            video = videos_list[int(video_idx)]
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

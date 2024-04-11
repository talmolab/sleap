"""Module for storing information for camera groups."""

from itertools import permutations, product
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast, Set

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

        instance = None
        for cam, instance in self._instance_by_camcorder.items():
            self._camcorder_by_instance[instance] = cam

        # Create a dummy instance to fill in for missing instances
        if self._dummy_instance is None:
            self._create_dummy_instance(instance=instance)

    def _create_dummy_instance(self, instance: Optional["Instance"] = None):
        """Create a dummy instance to fill in for missing instances.

        Args:
            instance: Optional `Instance` object to use as an example instance. If None,
                then the first instance in the `InstanceGroup` is used.

        Raises:
            ValueError: If no instances are available to create a dummy instance.
        """

        if self._dummy_instance is None:
            # Get an example instance
            if instance is None:
                if len(self.instances) < 1:
                    raise ValueError(
                        "Cannot create a dummy instance without any instances."
                    )
                instance = self.instances[0]

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
    def dummy_instance(self) -> "Instance":
        """Dummy `Instance` object to fill in for missing instances.

        Also used to create instances that are not found in the `InstanceGroup`.

        Returns:
            `Instance` object or None if unable to create the dummy instance.
        """

        if self._dummy_instance is None:
            self._create_dummy_instance()
        return self._dummy_instance

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
            instance = self.get_instance(cam) or self.dummy_instance
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
        skeleton: "Skeleton" = self.dummy_instance.skeleton

        # Create an all nan `Instance`
        instance: "Instance" = self.dummy_instance.__class__(
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

    # TODO(LM): Remove this, replace with `FrameGroup`s
    _instance_groups_by_frame_idx: Dict[int, InstanceGroup] = field(factory=dict)

    # TODO(LM): We should serialize all locked instances in a FrameGroup (or the entire FrameGroup)
    _frame_group_by_frame_idx: Dict[int, "FrameGroup"] = field(factory=dict)

    @property
    def videos(self) -> List[Video]:
        """List of `Video`s."""

        return self.camera_cluster._videos_by_session[self]

    @property
    def linked_cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s in `self.camera_cluster` that are linked to a video.

        The list is ordered based on the order of the `Camcorder`s in the `CameraCluster`.
        """

        return sorted(
            self._video_by_camcorder.keys(), key=self.camera_cluster.cameras.index
        )

    @property
    def unlinked_cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s in `self.camera_cluster` that are not linked to a video.

        The list is ordered based on the order of the `Camcorder`s in the `CameraCluster`.
        """

        return sorted(
            set(self.camera_cluster.cameras) - set(self.linked_cameras),
            key=self.camera_cluster.cameras.index,
        )

    # TODO(LM): Remove this
    @property
    def instance_groups(self) -> Dict[int, InstanceGroup]:
        """Dict of `InstanceGroup`s by frame index."""

        return self._instance_groups_by_frame_idx

    @property
    def frame_groups(self) -> Dict[int, "FrameGroup"]:
        """Dict of `FrameGroup`s by frame index."""

        return self._frame_group_by_frame_idx

    @property
    def frame_inds(self) -> List[int]:
        """List of frame indices."""

        return list(self.frame_groups.keys())

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

        # Sort `_videos_by_session` by order of linked `Camcorder` in `CameraCluster.cameras`
        self.camera_cluster._videos_by_session[self].sort(
            key=lambda video: self.camera_cluster.cameras.index(self.get_camera(video))
        )

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

    # TODO(LM): There can be multiple `InstanceGroup`s per frame index
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

    # TODO(LM): There can be multiple `InstanceGroup`s per frame index
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


@define
class FrameGroup:
    """Defines a group of `InstanceGroups` across views at the same frame index."""

    # Instance attributes
    frame_idx: int = field(validator=instance_of(int))
    instance_groups: List[InstanceGroup] = field(
        validator=deep_iterable(
            member_validator=instance_of(InstanceGroup),
            iterable_validator=instance_of(list),
        ),
    )  # Akin to `LabeledFrame.instances`
    session: RecordingSession = field(validator=instance_of(RecordingSession))

    # Class attribute to keep track of frame indices across all `RecordingSession`s
    _frame_idx_registry: Dict[RecordingSession, Set[int]] = {}

    # "Hidden" class attribute
    _cams_to_include: Optional[List[Camcorder]] = None
    _excluded_views: Optional[Tuple[str]] = ()
    _dummy_labeled_frame: Optional["LabeledFrame"] = None

    # "Hidden" instance attributes

    # TODO(LM): This dict should be updated each time a LabeledFrame is added/removed
    # from the Labels object. Or if a video is added/removed from the RecordingSession.
    _labeled_frames_by_cam: Dict[Camcorder, "LabeledFrame"] = field(factory=dict)
    _instances_by_cam: Dict[Camcorder, Set["Instance"]] = field(factory=dict)

    # TODO(LM): This dict should be updated each time an InstanceGroup is
    # added/removed/locked/unlocked
    _locked_instance_groups: List[InstanceGroup] = field(factory=list)
    _locked_instances_by_cam: Dict[Camcorder, Set["Instance"]] = field(
        factory=dict
    )  # Internally updated in `update_locked_instances_by_cam`

    def __attrs_post_init__(self):
        """Initialize `FrameGroup` object."""

        # Remove existing `FrameGroup` object from the `RecordingSession._frame_group_by_frame_idx`
        self.enforce_frame_idx_unique(self.session, self.frame_idx)

        # Reorder `cams_to_include` to match `CameraCluster` order (via setter method)
        if self._cams_to_include is not None:
            self.cams_to_include = self._cams_to_include

        # Add frame index to registry
        if self.session not in self._frame_idx_registry:
            self._frame_idx_registry[self.session] = set()

        self._frame_idx_registry[self.session].add(self.frame_idx)

        # Add `FrameGroup` to `RecordingSession`
        self.session._frame_group_by_frame_idx[self.frame_idx] = self

        # Initialize `_labeled_frames_by_cam` dictionary
        self.update_labeled_frames_and_instances_by_cam()

        # Initialize `_locked_instance_groups` dictionary
        self.update_locked_instance_groups()

        # The dummy labeled frame will only be set once for the first `FrameGroup` made
        if self._dummy_labeled_frame is None:
            self._dummy_labeled_frame = self.labeled_frames[0]

    @property
    def cams_to_include(self) -> Optional[List[Camcorder]]:
        """List of `Camcorder`s to include in this `FrameGroup`."""

        if self._cams_to_include is None:
            self._cams_to_include = self.session.camera_cluster.cameras.copy()

        # TODO(LM): Should we store this in another attribute?
        # Filter cams to include based on videos linked to the session
        cams_to_include = [
            cam for cam in self._cams_to_include if cam in self.session.linked_cameras
        ]

        return cams_to_include

    @property
    def excluded_views(self) -> Optional[Tuple[str]]:
        """List of excluded views (names of Camcorders)."""

        return self._excluded_views

    @cams_to_include.setter
    def cams_to_include(self, cams_to_include: List[Camcorder]):
        """Setter for `cams_to_include` that sorts by `CameraCluster` order."""

        # Sort the `Camcorder`s to include based on the order of `CameraCluster` cameras
        self._cams_to_include = cams_to_include.sort(
            key=self.session.camera_cluster.cameras.index
        )

        # Update the `excluded_views` attribute
        excluded_cams = list(
            set(self.session.camera_cluster.cameras) - set(cams_to_include)
        )
        excluded_cams.sort(key=self.session.camera_cluster.cameras.index)
        self._excluded_views = (cam.name for cam in excluded_cams)

    @property
    def labeled_frames(self) -> List["LabeledFrame"]:
        """List of `LabeledFrame`s."""

        return list(self._labeled_frames_by_cam.values())

    @property
    def cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s."""

        return list(self._labeled_frames_by_cam.keys())

    @property
    def instances_by_cam_to_include(self) -> Dict[Camcorder, Set["Instance"]]:
        """List of `Camcorder`s."""

        return {cam: self._instances_by_cam[cam] for cam in self.cams_to_include}

    @property
    def locked_instance_groups(self) -> List[InstanceGroup]:
        """List of locked `InstanceGroup`s."""

        return self._locked_instance_groups

    def numpy(
        self, instance_groups: Optional[List[InstanceGroup]] = None
    ) -> np.ndarray:
        """Numpy array of all `InstanceGroup`s in `FrameGroup.cams_to_include`.

        Args:
            instance_groups: `InstanceGroup`s to include. Default is None and uses all
                self.instance_groups.

        Returns:
            Numpy array of shape (M, T, N, 2) where M is the number of views (determined
            by self.cames_to_include), T is the number of `InstanceGroup`s, N is the
            number of Nodes, and 2 is for x, y.
        """

        # Use all `InstanceGroup`s if not specified
        if instance_groups is None:
            instance_groups = self.instance_groups
        else:
            # Ensure that `InstanceGroup`s is in this `FrameGroup`
            for instance_group in instance_groups:
                if instance_group not in self.instance_groups:
                    raise ValueError(
                        f"InstanceGroup {instance_group} is not in this FrameGroup: "
                        f"{self.instance_groups}"
                    )

        instance_group_numpys: List[np.ndarray] = []  # len(T) M=all x N x 2
        for instance_group in instance_groups:
            instance_group_numpy = instance_group.numpy()  # M=all x N x 2
            instance_group_numpys.append(instance_group_numpy)

        frame_group_numpy = np.stack(instance_group_numpys, axis=1)  # M=all x T x N x 2
        cams_to_include_mask = np.array(
            [1 if cam in self.cams_to_include else 0 for cam in self.cameras]
        )  # M=include x 1

        return frame_group_numpy[cams_to_include_mask]  # M=include x T x N x 2

    def add_instance(
        self,
        instance: "Instance",
        camera: Camcorder,
        instance_group: Optional[InstanceGroup] = None,
    ):
        """Add an (existing) `Instance` to the `FrameGroup`.

        If no `InstanceGroup` is provided, then check the `Instance` is already in an
        `InstanceGroup` contained in the `FrameGroup`.

        Args:
            instance: `Instance` to add to the `FrameGroup`.
            camera: `Camcorder` to link the `Instance` to.
            instance_group: `InstanceGroup` to add the `Instance` to. If None, then
                check the `Instance` is already in an `InstanceGroup`.

        Raises:
            ValueError: If the `InstanceGroup` is not in the `FrameGroup`.
            ValueError: If the `Instance` is not linked to a `LabeledFrame`.
            ValueError: If the frame index of the `Instance` does not match the frame index
                of the `FrameGroup`.
            ValueError: If the `LabeledFrame` of the `Instance` does not match the existing
                `LabeledFrame` for the `Camcorder` in the `FrameGroup`.
            ValueError: If the `Instance` is not in an `InstanceGroup` in the
                `FrameGroup`.
        """

        # Ensure the `InstanceGroup` is in this `FrameGroup`
        if instance_group is not None:
            self._raise_if_instance_group_not_in_frame_group(
                instance_group=instance_group
            )

        # Ensure `Instance` is compatible with `FrameGroup`
        self._raise_if_instance_incompatibile(instance=instance, camera=camera)

        # Add the `Instance` to the `InstanceGroup`
        if instance_group is not None:
            instance_group.add_instance(cam=camera, instance=instance)
        else:
            self._raise_if_instance_not_in_instance_group(instance=instance)

        # Add the `Instance` to the `FrameGroup`
        self._instances_by_cam[camera].add(instance)

        # Update the labeled frames if necessary
        labeled_frame = self.get_labeled_frame(camera=camera)
        if labeled_frame is None:
            labeled_frame = instance.frame
            self.add_labeled_frame(labeled_frame=labeled_frame, camera=camera)

    def add_instance_group(self, instance_group: Optional[InstanceGroup] = None):
        """Add an `InstanceGroup` to the `FrameGroup`.

        Args:
            instance_group: `InstanceGroup` to add to the `FrameGroup`. If None, then
                create a new `InstanceGroup` and add it to the `FrameGroup`.

        Raises:
            ValueError: If the `InstanceGroup` is already in the `FrameGroup`.
        """

        if instance_group is None:
            # Create an empty `InstanceGroup` with the frame index of the `FrameGroup`
            instance_group = InstanceGroup(
                frame_idx=self.frame_idx,
                camera_cluster=self.session.camera_cluster,
            )

        else:
            # Ensure the `InstanceGroup` is not already in this `FrameGroup`
            self._raise_if_instance_group_in_frame_group(instance_group=instance_group)

            # Ensure the `InstanceGroup` is compatible with the `FrameGroup`
            self._raise_if_instance_group_incompatible(instance_group=instance_group)

        # Add the `InstanceGroup` to the `FrameGroup`
        self.instance_groups.append(instance_group)

        # Add `Instance`s and `LabeledFrame`s to the `FrameGroup`
        for instance in instance_group.instances:
            camera = instance_group.get_cam(instance=instance)
            self.add_instance(instance=instance, camera=camera)

        # TODO(LM): Integrate with RecordingSession
        # Add the `InstanceGroup` to the `RecordingSession`
        ...

    def get_instance_group(self, instance: "Instance") -> Optional[InstanceGroup]:
        """Get `InstanceGroup` that contains `Instance` if exists. Otherwise, None.

        Args:
            instance: `Instance`

        Returns:
            `InstanceGroup`
        """

        instance_group: Optional[InstanceGroup] = next(
            (
                instance_group
                for instance_group in self.instance_groups
                if instance in instance_group.instances
            ),
            None,
        )

        return instance_group

    def add_labeled_frame(self, labeled_frame: "LabeledFrame", camera: Camcorder):
        """Add a `LabeledFrame` to the `FrameGroup`.

        Args:
            labeled_frame: `LabeledFrame` to add to the `FrameGroup`.
            camera: `Camcorder` to link the `LabeledFrame` to.
        """

        # Add the `LabeledFrame` to the `FrameGroup`
        self._labeled_frames_by_cam[camera] = labeled_frame

        # TODO(LM): Should this be an EditCommand instead?
        # Add the `LabeledFrame` to the `RecordingSession`'s `Labels` object
        if labeled_frame not in self.session.labels:
            self.session.labels.append(labeled_frame)

    def get_labeled_frame(self, camera: Camcorder) -> Optional["LabeledFrame"]:
        """Get `LabeledFrame` for `Camcorder` if exists. Otherwise, None.

        Args:
            camera: `Camcorder`

        Returns:
            `LabeledFrame`
        """

        return self._labeled_frames_by_cam.get(camera, None)

    def create_and_add_labeled_frame(self, camera: Camcorder) -> "LabeledFrame":
        """Create and add a `LabeledFrame` to the `FrameGroup`.

        This also adds the `LabeledFrame` to the `RecordingSession`'s `Labels` object.

        Args:
            camera: `Camcorder`

        Returns:
            `LabeledFrame` that was created and added to the `FrameGroup`.
        """

        video = self.session.get_video(camera)
        if video is None:
            # There should be a `Video` linked to all cams_to_include
            raise ValueError(
                f"Camcorder {camera} is not linked to a video in this "
                f"RecordingSession {self.session}."
            )

        # Use _dummy_labeled_frame to access the `LabeledFrame`` class here
        labeled_frame = self._dummy_labeled_frame.__class__(
            video=video, frame_idx=self.frame_idx
        )
        self.add_labeled_frame(labeled_frame=labeled_frame)

        return labeled_frame

    def create_and_add_instance(
        self,
        instance_group: InstanceGroup,
        camera: Camcorder,
        labeled_frame: "LabeledFrame",
    ):
        """Add an `Instance` to the `InstanceGroup` (and `FrameGroup`).

        Args:
            instance_group: `InstanceGroup` to add the `Instance` to.
            camera: `Camcorder` to link the `Instance` to.
            labeled_frame: `LabeledFrame` that the `Instance` is in.
        """

        # Add the `Instance` to the `InstanceGroup`
        instance = instance_group.create_and_add_instance(
            cam=camera, labeled_frame=labeled_frame
        )

        # Add the `Instance` to the `FrameGroup`
        self._instances_by_cam[camera].add(instance=instance)

    def create_and_add_missing_instances(self, instance_group: InstanceGroup):
        """Add missing instances to `FrameGroup` from `InstanceGroup`s.

        If an `InstanceGroup` does not have an `Instance` for a `Camcorder` in
        `FrameGroup.cams_to_include`, then create an `Instance` and add it to the
        `InstanceGroup`.

        Args:
            instance_group: `InstanceGroup` objects to add missing `Instance`s for.

        Raises:
            ValueError: If a `Camcorder` in `FrameGroup.cams_to_include` is not in the
                `InstanceGroup`.
        """

        # Check that the `InstanceGroup` has `LabeledFrame`s for all included views
        for cam in self.cams_to_include:

            # If the `Camcorder` is in the `InstanceGroup`, then `Instance` exists
            if cam in instance_group.cameras:
                continue  # Skip to next cam

            # Get the `LabeledFrame` for the view
            labeled_frame = self.get_labeled_frame(camera=cam)
            if labeled_frame is None:
                # There is no `LabeledFrame` for this view, so lets make one
                labeled_frame = self.create_and_add_labeled_frame(camera=cam)

            # Create an instance
            self.create_and_add_instance(
                instance_group=instance_group, cam=cam, labeled_frame=labeled_frame
            )

    def upsert_points(
        self,
        points: np.ndarray,
        instance_groups: List[InstanceGroup],
        exclude_complete: bool = True,
    ):
        """Upsert points for `Instance`s at included cams in specified `InstanceGroup`.

        This will update the points for existing `Instance`s in the `InstanceGroup`s and
        also add new `Instance`s if they do not exist.


        Included cams are specified by `FrameGroup.cams_to_include`.

        The ordering of the `InstanceGroup`s in `instance_groups` should match the
        ordering of the second dimension (T) in `points`.

        Args:
            points: Numpy array of shape (M, T, N, 2) where M is the number of views, T
                is the number of Tracks, N is the number of Nodes, and 2 is for x, y.
            instance_groups: List of `InstanceGroup` objects to update points for.
            exclude_complete: If True, then only update points that are not marked as
                complete. Default is True.
        """

        # Check that the correct shape was passed in
        n_views, n_instances, n_nodes, n_coords = points.shape
        assert n_views == len(
            self.cams_to_include
        ), f"Expected {len(self.cams_to_include)} views, got {n_views}."
        assert n_instances == len(
            instance_groups
        ), f"Expected {len(instance_groups)} instances, got {n_instances}."
        assert n_coords == 2, f"Expected 2 coordinates, got {n_coords}."

        # Update points for each `InstanceGroup`
        for ig_idx, instance_group in enumerate(instance_groups):
            # Ensure that `InstanceGroup`s is in this `FrameGroup`
            self._raise_if_instance_group_not_in_frame_group(
                instance_group=instance_group
            )

            # Check that the `InstanceGroup` has `Instance`s for all cams_to_include
            self.create_and_add_missing_instances(instance_group=instance_group)

            # Update points for each `Instance` in `InstanceGroup`
            instance_points = points[:, ig_idx, :, :]  # M x N x 2
            instance_group.update_points(
                points=instance_points,
                cams_to_include=self.cams_to_include,
                exclude_complete=exclude_complete,
            )

    def _raise_if_instance_not_in_instance_group(self, instance: "Instance"):
        """Raise a ValueError if the `Instance` is not in an `InstanceGroup`.

        Args:
            instance: `Instance` to check if in an `InstanceGroup`.

        Raises:
            ValueError: If the `Instance` is not in an `InstanceGroup`.
        """

        instance_group = self.get_instance_group(instance=instance)
        if instance_group is None:
            raise ValueError(
                f"Instance {instance} is not in an InstanceGroup within the FrameGroup."
            )

    def _raise_if_instance_incompatibile(self, instance: "Instance", camera: Camcorder):
        """Raise a ValueError if the `Instance` is incompatible with the `FrameGroup`.

        The `Instance` is incompatible if:
        1. the `Instance` is not linked to a `LabeledFrame`.
        2. the frame index of the `Instance` does not match the frame index of the
            `FrameGroup`.
        3. the `LabeledFrame` of the `Instance` does not match the existing
            `LabeledFrame` for the `Camcorder` in the `FrameGroup`.

        Args:
            instance: `Instance` to check compatibility of.
            camera: `Camcorder` to link the `Instance` to.
        """

        labeled_frame = instance.frame
        if labeled_frame is None:
            raise ValueError(
                f"Instance {instance} is not linked to a LabeledFrame. "
                "Cannot add to FrameGroup."
            )

        frame_idx = labeled_frame.frame_idx
        if frame_idx != self.frame_idx:
            raise ValueError(
                f"Instance {instance} frame index {frame_idx} does not match "
                f"FrameGroup frame index {self.frame_idx}."
            )

        labeled_frame_fg = self.get_labeled_frame(camera=camera)
        if labeled_frame_fg is None:
            pass
        elif labeled_frame != labeled_frame_fg:
            raise ValueError(
                f"Instance's LabeledFrame {labeled_frame} is not the same as "
                f"FrameGroup's LabeledFrame {labeled_frame_fg} for Camcorder {camera}."
            )

    def _raise_if_instance_group_in_frame_group(self, instance_group: InstanceGroup):
        """Raise a ValueError if the `InstanceGroup` is already in the `FrameGroup`.

        Args:
            instance_group: `InstanceGroup` to check if already in the `FrameGroup`.

        Raises:
            ValueError: If the `InstanceGroup` is already in the `FrameGroup`.
        """

        if instance_group in self.instance_groups:
            raise ValueError(
                f"InstanceGroup {instance_group} is already in this FrameGroup "
                f"{self.instance_groups}."
            )

    def _raise_if_instance_group_incompatible(self, instance_group: InstanceGroup):
        """Raise a ValueError if `InstanceGroup` is incompatible with `FrameGroup`.

        An `InstanceGroup` is incompatible if the `frame_idx` does not match the
        `FrameGroup`'s `frame_idx`.

        Args:
            instance_group: `InstanceGroup` to check compatibility of.

        Raises:
            ValueError: If the `InstanceGroup` is incompatible with the `FrameGroup`.
        """

        if instance_group.frame_idx != self.frame_idx:
            raise ValueError(
                f"InstanceGroup {instance_group} frame index {instance_group.frame_idx} "
                f"does not match FrameGroup frame index {self.frame_idx}."
            )

    def _raise_if_instance_group_not_in_frame_group(
        self, instance_group: InstanceGroup
    ):
        """Raise a ValueError if `InstanceGroup` is not in this `FrameGroup`."""

        if instance_group not in self.instance_groups:
            raise ValueError(
                f"InstanceGroup {instance_group} is not in this FrameGroup: "
                f"{self.instance_groups}."
            )

    def update_labeled_frames_and_instances_by_cam(
        self, return_instances_by_camera: bool = False
    ) -> Union[Dict[Camcorder, "LabeledFrame"], Dict[Camcorder, List["Instance"]]]:
        """Get all views and `Instance`s across all `RecordingSession`s.

        Updates the `_labeled_frames_by_cam` and `_instances_by_cam`
        dictionary attributes.

        Args:
            return_instances_by_camera: If true, then returns a dictionary with
                `Camcorder` key and `Set[Instance]` values instead. Default is False.

        Returns:
            Dictionary with `Camcorder` key and `LabeledFrame` value or `Set[Instance]`
                value if `return_instances_by_camera` is True.
        """

        logger.debug(
            "Updating LabeledFrames for FrameGroup."
            "\n\tPrevious LabeledFrames by Camcorder:"
            f"\n\t{self._labeled_frames_by_cam}"
        )

        views: Dict[Camcorder, "LabeledFrame"] = {}
        instances_by_cam: Dict[Camcorder, Set["Instance"]] = {}
        videos = self.session.get_videos_from_selected_cameras()
        for cam, video in videos.items():
            lfs: List["LabeledFrame"] = self.session.labels.get(
                (video, [self.frame_idx])
            )
            if len(lfs) == 0:
                logger.debug(
                    f"No LabeledFrames found for video {video} at {self.frame_idx}."
                )
                continue

            lf = lfs[0]
            if len(lf.instances) == 0:
                logger.warning(
                    f"No Instances found for {lf}."
                    " There should not be empty LabeledFrames."
                )
                continue

            views[cam] = lf

            # Find instances in frame
            insts = lf.find(track=-1, user=True)
            if len(insts) > 0:
                instances_by_cam[cam] = set(insts)

        # Update `_labeled_frames_by_cam` dictionary and return it
        self._labeled_frames_by_cam = views
        logger.debug(
            f"\tUpdated LabeledFrames by Camcorder:\n\t{self._labeled_frames_by_cam}"
        )
        # Update `_instances_by_camera` dictionary and return it
        self._instances_by_cam = instances_by_cam
        return (
            self._instances_by_cam
            if return_instances_by_camera
            else self._labeled_frames_by_cam
        )

    def update_locked_instance_groups(self) -> List[InstanceGroup]:
        """Updates locked `InstanceGroup`s in `FrameGroup`.

        Returns:
            List of locked `InstanceGroup`s.
        """

        self._locked_instance_groups: List[InstanceGroup] = [
            instance_group
            for instance_group in self.instance_groups
            if instance_group.locked
        ]

        # Also update locked instances by cam
        self.update_locked_instances_by_cam(self._locked_instance_groups)

        return self._locked_instance_groups

    def update_locked_instances_by_cam(
        self, locked_instance_groups: List[InstanceGroup] = None
    ) -> Dict[Camcorder, Set["Instance"]]:
        """Updates locked `Instance`s in `FrameGroup`.

        Args:
            locked_instance_groups: List of locked `InstanceGroup`s. Default is None.
                If None, then uses `self.locked_instance_groups`.

        Returns:
            Dictionary with `Camcorder` key and `Set[Instance]` value.
        """

        if locked_instance_groups is None:
            locked_instance_groups = self.locked_instance_groups

        locked_instances_by_cam: Dict[Camcorder, Set["Instance"]] = {}

        # Loop through each camera and append locked instances in specific order
        for cam in self.cams_to_include:
            locked_instances_by_cam[cam] = set()
            for instance_group in locked_instance_groups:
                instance = instance_group.get_instance(cam)  # Returns None if not found

                # TODO(LM): Should this be adding the dummy instance here?
                # LM: No, since just using the number of locked instance groups will
                # account for the dummy instances
                if instance is not None:
                    locked_instances_by_cam[cam].add(instance)

        # Only update if there were no errors
        self._locked_instances_by_cam = locked_instances_by_cam
        return self._locked_instances_by_cam

    # TODO(LM): Should we move this to TriangulateSession?
    def generate_hypotheses(
        self, as_matrix: bool = True
    ) -> Union[np.ndarray, Dict[int, List[InstanceGroup]]]:
        """Generates all possible hypotheses from the `FrameGroup`.

        Args:
            as_matrix: If True (defualt), then return as a matrix of
                `Instance.points_array`. Else return as `Dict[int, List[InstanceGroup]]`
                where `int` is the hypothesis identifier and `List[InstanceGroup]` is
                the list of `InstanceGroup`s.

        Returns:
            Either a `np.ndarray` of shape M x F x T x N x 2 an array if as_matrix where
            M: # views, F: # frames = 1, T: # tracks, N: # nodes, 2: x, y
            or a dictionary with hypothesis ID key and list of `InstanceGroup`s value.
        """

        # Get all `Instance`s for this frame index across all views to include
        instances_by_camera: Dict[Camcorder, Set["Instance"]] = (
            self.instances_by_cam_to_include
        )

        # Get max number of instances across all views
        all_instances_by_camera: List[Set["Instance"]] = instances_by_camera.values()
        max_num_instances = max(
            [len(instances) for instances in all_instances_by_camera], default=0
        )

        # Create a dummy instance of all nan values
        example_instance: "Instance" = next(iter(all_instances_by_camera[0]))
        skeleton: "Skeleton" = example_instance.skeleton
        dummy_instance: "Instance" = example_instance.from_numpy(
            np.full(
                shape=(len(skeleton.nodes), 2),
                fill_value=np.nan,
            ),
            skeleton=skeleton,
        )

        def _fill_in_missing_instances(
            unlocked_instances_in_view: List["Instance"],
        ):
            """Fill in missing instances with dummy instances up to max number.

            Note that this function will mutate the input list in addition to returning
            the mutated list.

            Args:
                unlocked_instances_in_view: List of instances in a view that are not in
                    a locked InstanceGroup.

            Returns:
                List of instances in a view that are not in a locked InstanceGroup with
                    dummy instances appended.
            """

            # Subtracting the number of locked instance groups accounts for there being
            # dummy instances in the locked instance groups.
            num_instances_missing = (
                max_num_instances
                - len(unlocked_instances_in_view)
                - len(
                    self.locked_instance_groups
                )  # TODO(LM): Make sure this property is getting updated properly
            )

            if num_instances_missing > 0:
                # Extend the list of instances with dummy instances
                unlocked_instances_in_view.extend(
                    [dummy_instance] * num_instances_missing
                )

            return unlocked_instances_in_view

        # For each view, get permutations of unlocked instances
        unlocked_instance_permutations: Dict[Camcorder, Iterator[Tuple["Instance"]]] = (
            {}
        )
        for cam, instances_in_view in instances_by_camera.items():
            # Gather all instances for this cam from locked `InstanceGroup`s
            locked_instances_in_view: Set["Instance"] = (
                self._locked_instances_by_cam.get(cam, set())
            )

            # Remove locked instances from instances in view
            unlocked_instances_in_view: List["Instance"] = list(
                instances_in_view - locked_instances_in_view
            )

            # Fill in missing instances with dummy instances up to max number
            unlocked_instances_in_view = _fill_in_missing_instances(
                unlocked_instances_in_view
            )

            # Permuate all `Instance`s in the unlocked `InstanceGroup`s
            unlocked_instance_permutations[cam] = permutations(
                unlocked_instances_in_view
            )

        # Get products of instances from other views into all possible groupings
        # Ordering of dict_values is preserved in Python 3.7+
        products_of_unlocked_instances: Iterator[Iterator[Tuple]] = product(
            *unlocked_instance_permutations.values()
        )

        # Reorganize products by cam and add selected instance to each permutation
        grouping_hypotheses: Dict[int, List[InstanceGroup]] = {}
        for frame_id, prod in enumerate(products_of_unlocked_instances):
            grouping_hypotheses[frame_id] = {
                # TODO(LM): This is where we would create the `InstanceGroup`s instead
                cam: list(inst)
                for cam, inst in zip(self.cams_to_include, prod)
            }

        # TODO(LM): Should we return this as instance matrices or `InstanceGroup`s?
        # Answer: Definitely not instance matrices since we need to keep track of the
        # `Instance`s, but I kind of wonder if we could just return a list of
        # `InstanceGroup`s instead of a dict then the `InstanceGroup`

        return grouping_hypotheses

    @classmethod
    def from_instance_groups(
        cls,
        session: RecordingSession,
        instance_groups: List["InstanceGroup"],
    ) -> Optional["FrameGroup"]:
        """Creates a `FrameGroup` object from an `InstanceGroup` object.

        Args:
            session: `RecordingSession` object.
            instance_groups: A list of `InstanceGroup` objects.

        Returns:
            `FrameGroup` object or None if no "real" (determined by `frame_idx` other
            than None) frames found.
        """

        if len(instance_groups) == 0:
            raise ValueError("instance_groups must contain at least one InstanceGroup")

        # Get frame index from first instance group
        frame_idx = instance_groups[0].frame_idx

        # Create and return `FrameGroup` object
        return cls(
            frame_idx=frame_idx, instance_groups=instance_groups, session=session
        )

    def enforce_frame_idx_unique(
        self, session: RecordingSession, frame_idx: int
    ) -> bool:
        """Enforces that all frame indices are unique in `RecordingSession`.

        Removes existing `FrameGroup` object from the
        `RecordingSession._frame_group_by_frame_idx`.

        Args:
            session: `RecordingSession` object.
            frame_idx: Frame index.
        """

        if frame_idx in self._frame_idx_registry.get(session, set()):
            # Remove existing `FrameGroup` object from the
            # `RecordingSession._frame_group_by_frame_idx`
            logger.warning(
                f"Frame index {frame_idx} for FrameGroup already exists in this "
                "RecordingSession. Overwriting."
            )
            session._frame_group_by_frame_idx.pop(frame_idx)

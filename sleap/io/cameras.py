"""Module for storing information for camera groups."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import cattr
import numpy as np
import toml
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
from attrs import define, field
from attrs.validators import deep_iterable, instance_of

# from sleap.io.dataset import Labels  # TODO(LM): Circular import, implement Observer
from sleap.instance import Instance, LabeledFrame, PredictedInstance
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
            logger.debug(f"{session} not found in {self}.")
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

    Attributes:
        name: Name of the `InstanceGroup`.
        frame_idx: Frame index for the `InstanceGroup`.
        dummy_instance: Optional `PredictedInstance` object to fill in for missing
            instances.
        camera_cluster: `CameraCluster` object that the `InstanceGroup` uses.
        cameras: List of `Camcorder` objects that have an `Instance` associated.
        instances: List of `Instance` objects.
        instance_by_camcorder: Dictionary of `Instance` objects by `Camcorder`.
    """

    _name: str = field()
    frame_idx: int = field(validator=instance_of(int))
    _instance_by_camcorder: Dict[Camcorder, Instance] = field(factory=dict)
    _camcorder_by_instance: Dict[Instance, Camcorder] = field(factory=dict)
    _dummy_instance: Optional[Instance] = field(default=None)
    camera_cluster: Optional[CameraCluster] = field(default=None)

    def __attrs_post_init__(self):
        """Initialize `InstanceGroup` object."""

        instance = None
        for cam, instance in self._instance_by_camcorder.items():
            self._camcorder_by_instance[instance] = cam

    def _create_dummy_instance(self, instance: Optional[Instance] = None):
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

            # Use the example instance to create a dummy instance
            skeleton: "Skeleton" = instance.skeleton
            self._dummy_instance = PredictedInstance.from_numpy(
                points=np.full(
                    shape=(len(skeleton.nodes), 2),
                    fill_value=np.nan,
                ),
                point_confidences=np.full(
                    shape=(len(skeleton.nodes),),
                    fill_value=np.nan,
                ),
                instance_score=np.nan,
                skeleton=skeleton,
            )

    @property
    def dummy_instance(self) -> PredictedInstance:
        """Dummy `PredictedInstance` object to fill in for missing instances.

        Also used to create instances that are not found in the `InstanceGroup`.

        Returns:
            `PredictedInstance` object or None if unable to create the dummy instance.
        """

        if self._dummy_instance is None:
            self._create_dummy_instance()
        return self._dummy_instance

    @property
    def name(self) -> str:
        """Name of the `InstanceGroup`."""

        return self._name

    @name.setter
    def name(self, name: str):
        """Set the name of the `InstanceGroup`."""

        raise ValueError(
            "Cannot set name directly. Use `set_name` method instead (preferably "
            "through FrameGroup.set_instance_group_name)."
        )

    def set_name(self, name: str, name_registry: Set[str]):
        """Set the name of the `InstanceGroup`.

        This function mutates the name_registry input (see side-effect).

        Args:
            name: Name to set for the `InstanceGroup`.
            name_registry: Set of names to check for uniqueness.

        Raises:
            ValueError: If the name is already in use (in the name_registry).
        """

        # Check if the name is already in use
        if name in name_registry:
            raise ValueError(
                f"Name {name} already in use. Please use a unique name not currently "
                f"in the registry: {name_registry}"
            )

        # Remove the old name from the registry
        if self._name in name_registry:
            name_registry.remove(self._name)

        self._name = name
        name_registry.add(name)

    @classmethod
    def return_unique_name(cls, name_registry: Set[str]) -> str:
        """Return a unique name for the `InstanceGroup`.

        Args:
            name_registry: Set of names to check for uniqueness.

        Returns:
            Unique name for the `InstanceGroup`.
        """

        base_name = "instance_group_"
        count = len(name_registry)
        new_name = f"{base_name}{count}"

        while new_name in name_registry:
            count += 1
            new_name = f"{base_name}{count}"

        return new_name

    @property
    def instances(self) -> List[Instance]:
        """List of `Instance` objects."""
        return list(self._instance_by_camcorder.values())

    @property
    def cameras(self) -> List[Camcorder]:
        """List of `Camcorder` objects."""
        return list(self._instance_by_camcorder.keys())

    @property
    def instance_by_camcorder(self) -> Dict[Camcorder, Instance]:
        """Dictionary of `Instance` objects by `Camcorder`."""
        return self._instance_by_camcorder

    def numpy(self, pred_as_nan: bool = False) -> np.ndarray:
        """Return instances as a numpy array of shape (n_views, n_nodes, 2).

        The ordering of views is based on the ordering of `Camcorder`s in the
        `self.camera_cluster: CameraCluster`.

        If an instance is missing for a `Camcorder`, then the instance is filled in with
        the dummy instance (all NaNs).

        Args:
            pred_as_nan: If True, then replaces `PredictedInstance`s with all nan
                self.dummy_instance. Default is False.

        Returns:
            Numpy array of shape (n_views, n_nodes, 2).
        """

        instance_numpys: List[np.ndarray] = []  # len(M) x N x 2
        for cam in self.camera_cluster.cameras:
            instance = self.get_instance(cam)

            # Determine whether to use a dummy (all nan) instance
            instance_is_missing = instance is None
            instance_as_nan = pred_as_nan and isinstance(instance, PredictedInstance)
            use_dummy_instance = instance_is_missing or instance_as_nan

            # Add the dummy instance if the instance is missing
            if use_dummy_instance:
                instance = self.dummy_instance  # This is an all nan PredictedInstance

            instance_numpy: np.ndarray = instance.numpy()  # N x 2
            instance_numpys.append(instance_numpy)

        return np.stack(instance_numpys, axis=0)  # M x N x 2

    def create_and_add_instance(self, cam: Camcorder, labeled_frame: LabeledFrame):
        """Create an `Instance` at a labeled_frame and add it to the `InstanceGroup`.

        Args:
            cam: `Camcorder` object that the `Instance` is for.
            labeled_frame: `LabeledFrame` object that the `Instance` is contained in.

        Returns:
            All nan `PredictedInstance` created and added to the `InstanceGroup`.
        """

        # Get the `Skeleton`
        skeleton: "Skeleton" = self.dummy_instance.skeleton

        # Create an all nan `Instance`
        instance: PredictedInstance = PredictedInstance.from_numpy(
            points=self.dummy_instance.points_array,
            point_confidences=self.dummy_instance.scores,
            instance_score=self.dummy_instance.score,
            skeleton=skeleton,
        )
        instance.frame = labeled_frame

        # Add the instance to the `InstanceGroup`
        self.add_instance(cam, instance)

        return instance

    def add_instance(self, cam: Camcorder, instance: Instance):
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

    def replace_instance(self, cam: Camcorder, instance: Instance):
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

    def remove_instance(self, instance_or_cam: Union[Instance, Camcorder]):
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

    def get_instance(self, cam: Camcorder) -> Optional[Instance]:
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

    def get_instances(self, cams: List[Camcorder]) -> List[Instance]:
        instances = []
        for cam in cams:
            instance = self.get_instance(cam)
            if instance is not None:
                instances.append(instance)
        return instances

    def get_cam(self, instance: Instance) -> Optional[Camcorder]:
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
            instance: Optional[Instance] = self.get_instance(cam)
            if instance is None:
                logger.warning(
                    f"Camcorder {cam.name} not found in this InstanceGroup's instances."
                )
                continue

            # Update the points (and scores) for the (predicted) instance
            instance.update_points(
                points=points[cam_idx, :, :], exclude_complete=exclude_complete
            )

    def __getitem__(
        self, idx_or_key: Union[int, Camcorder, Instance]
    ) -> Union[Camcorder, Instance]:
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
            # isinstance(idx_or_key, Instance):
            try:
                return self.get_cam(idx_or_key)
            except:
                pass

        _raise_key_error()

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, frame_idx={self.frame_idx}, "
            f"instances:{len(self)}, camera_cluster={self.camera_cluster})"
        )

    def __hash__(self) -> int:
        return hash(self._name)

    @classmethod
    def from_instance_by_camcorder_dict(
        cls,
        instance_by_camcorder: Dict[Camcorder, Instance],
        name: str,
        name_registry: Set[str],
    ) -> Optional["InstanceGroup"]:
        """Creates an `InstanceGroup` object from a dictionary.

        Args:
            instance_by_camcorder: Dictionary with `Camcorder` keys and `Instance` values.
            name: Name to use for the `InstanceGroup`.
            name_registry: Set of names to check for uniqueness.

        Raises:
            ValueError: If the `InstanceGroup` name is already in use.

        Returns:
            `InstanceGroup` object or None if no "real" (determined by `frame_idx` other
            than None) instances found.
        """

        if name in name_registry:
            raise ValueError(
                f"Cannot create `InstanceGroup`: Name {name} already in use. Please "
                f"use a unique name that is not in the registry: {name_registry}."
            )

        # Ensure not to mutate the original dictionary
        instance_by_camcorder_copy = instance_by_camcorder.copy()

        frame_idx = None
        for cam, instance in instance_by_camcorder_copy.copy().items():
            camera_cluster = cam.camera_cluster

            # Remove dummy instances (determined by not having a frame index)
            if instance.frame_idx is None:
                instance_by_camcorder_copy.pop(cam)
            # Grab the frame index from non-dummy instances
            elif frame_idx is None:
                frame_idx = instance.frame_idx
            # Ensure all instances have the same frame index
            elif frame_idx != instance.frame_idx:
                raise ValueError(
                    f"Cannot create `InstanceGroup`: Frame index {frame_idx} does "
                    f"not match instance frame index {instance.frame_idx}."
                )

        if len(instance_by_camcorder_copy) == 0:
            raise ValueError("Cannot create `InstanceGroup`: No frame idx found.")

        return cls(
            name=name,
            frame_idx=frame_idx,
            camera_cluster=camera_cluster,
            instance_by_camcorder=instance_by_camcorder_copy,
        )

    def to_dict(
        self, instance_to_lf_and_inst_idx: Dict[Instance, Tuple[str, str]]
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """Converts the `InstanceGroup` to a dictionary.

        Args:
            instance_to_lf_and_inst_idx: Dictionary mapping `Instance` objects to
                `LabeledFrame` indices (in `Labels.labeled_frames`) and `Instance`
                indices (in containing `LabeledFrame.instances`).

        Returns:
            Dictionary of the `InstanceGroup` with items:
                - name: Name of the `InstanceGroup`.
                - camcorder_to_lf_and_inst_idx_map: Dictionary mapping `Camcorder` indices
                    (in `InstanceGroup.camera_cluster.cameras`) to both `LabeledFrame`
                    and `Instance` indices (from `instance_to_lf_and_inst_idx`).
        """

        camcorder_to_lf_and_inst_idx_map: Dict[str, Tuple[str, str]] = {
            str(self.camera_cluster.cameras.index(cam)): instance_to_lf_and_inst_idx[
                instance
            ]
            for cam, instance in self._instance_by_camcorder.items()
        }

        return {
            "name": self.name,
            "camcorder_to_lf_and_inst_idx_map": camcorder_to_lf_and_inst_idx_map,
        }

    @classmethod
    def from_dict(
        cls,
        instance_group_dict: dict,
        name_registry: Set[str],
        labeled_frames_list: List[LabeledFrame],
        camera_cluster: CameraCluster,
    ):
        """Creates an `InstanceGroup` object from a dictionary.

        Args:
            instance_group_dict: Dictionary with keys for name and
                camcorder_to_lf_and_inst_idx_map.
            name_registry: Set of names to check for uniqueness.
            labeled_frames_list: List of `LabeledFrame` objects (expecting
                `Labels.labeled_frames`).
            camera_cluster: `CameraCluster` object.

        Returns:
            `InstanceGroup` object.
        """

        # Get the `Instance` objects
        camcorder_to_lf_and_inst_idx_map: Dict[
            str, Tuple[str, str]
        ] = instance_group_dict["camcorder_to_lf_and_inst_idx_map"]

        instance_by_camcorder: Dict[Camcorder, Instance] = {}
        for cam_idx, (lf_idx, inst_idx) in camcorder_to_lf_and_inst_idx_map.items():
            # Retrieve the `Camcorder`
            camera = camera_cluster.cameras[int(cam_idx)]

            # Retrieve the `Instance` from the `LabeledFrame
            labeled_frame = labeled_frames_list[int(lf_idx)]
            instance = labeled_frame.instances[int(inst_idx)]

            # Link the `Instance` to the `Camcorder`
            instance_by_camcorder[camera] = instance

        return cls.from_instance_by_camcorder_dict(
            instance_by_camcorder=instance_by_camcorder,
            name=instance_group_dict["name"],
            name_registry=name_registry,
        )


@define(eq=False)
class RecordingSession:
    """Class for storing information for a recording session.

    Attributes:
        camera_cluster: `CameraCluster` object.
        metadata: Dictionary of metadata.
        labels: `Labels` object.
        videos: List of `Video`s that have been linked to a `Camcorder` in the
            `self.camera_cluster`.
        linked_cameras: List of `Camcorder`s in the `self.camera_cluster` that are
            linked to a `Video`.
        unlinked_cameras: List of `Camcorder`s in the `self.camera_cluster` that are
            not linked to a `Video`.
        frame_groups: Dictionary of `FrameGroup`s by frame index.
        frame_inds: List of frame indices.
        cams_to_include: List of `Camcorder`s to include in this `FrameGroup`.
        excluded_views: List of excluded views (names of `Camcorder`s).
    """

    # TODO(LM): Consider implementing Observer pattern for `camera_cluster` and `labels`
    camera_cluster: CameraCluster = field(factory=CameraCluster)
    metadata: dict = field(factory=dict)
    labels: Optional["Labels"] = field(default=None)
    _video_by_camcorder: Dict[Camcorder, Video] = field(factory=dict)
    _frame_group_by_frame_idx: Dict[int, "FrameGroup"] = field(factory=dict)
    _cams_to_include: Optional[List[Camcorder]] = field(default=None)
    _excluded_views: Optional[Tuple[str]] = field(default=None)

    @property
    def videos(self) -> List[Video]:
        """List of `Video`s."""

        # TODO(LM): Should these be in the same order as `self.labels.videos`?
        # e.g. switching between views in GUI should keep the same order, but not enforced.
        # We COULD implicitly enforce this by adding videos in the same order as
        # `self.labels.videos`, but "explicit is better than implicit".
        # Instead, we could sort the videos by their index in labels.videos. This might
        # bottleneck switching between views for sessions with lots of cameras/videos.
        # Unless! We do this (each time) when adding the videos to the session instead
        # of when accessing the videos. This would be a good compromise.
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

    @property
    def frame_groups(self) -> Dict[int, "FrameGroup"]:
        """Dict of `FrameGroup`s by frame index."""

        return self._frame_group_by_frame_idx

    @property
    def frame_inds(self) -> List[int]:
        """List of frame indices."""

        return list(self.frame_groups.keys())

    @property
    def cams_to_include(self) -> Optional[List[Camcorder]]:
        """List of `Camcorder`s to include in this `FrameGroup`."""

        if self._cams_to_include is None:
            self._cams_to_include = self.camera_cluster.cameras

        # Filter cams to include based on videos linked to the session
        cams_to_include = [
            cam for cam in self._cams_to_include if cam in self.linked_cameras
        ]

        return cams_to_include

    @cams_to_include.setter
    def cams_to_include(self, cams_to_include: List[Camcorder]):
        """Setter for `cams_to_include` that sorts by `CameraCluster` order."""

        # Sort the `Camcorder`s to include based on the order of `CameraCluster` cameras
        self._cams_to_include = sorted(
            cams_to_include, key=self.camera_cluster.cameras.index
        )

        # Update the `excluded_views` attribute
        excluded_cams = list(set(self.camera_cluster.cameras) - set(cams_to_include))
        excluded_cams.sort(key=self.camera_cluster.cameras.index)
        self._excluded_views = tuple([cam.name for cam in excluded_cams])

    @property
    def excluded_views(self) -> Optional[Tuple[str]]:
        """List of excluded views (names of Camcorders)."""

        return self._excluded_views

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
            logger.debug(
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
        if camcorder not in self.camera_cluster:
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
            self.labels.remove_session_video(video=video)

    def new_frame_group(self, frame_idx: int):
        """Creates and adds an empty `FrameGroup` to the `RecordingSession`.

        Args:
            frame_idx: Frame index for the `FrameGroup`.

        Returns:
            `FrameGroup` object.
        """

        # `FrameGroup.__attrs_post_init` will manage `_frame_group_by_frame_idx`
        frame_group = FrameGroup(frame_idx=frame_idx, session=self)

        return frame_group

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

    def __bool__(self):
        return True

    def __attrs_post_init__(self):
        self.camera_cluster.add_session(self)

        # Reorder `cams_to_include` to match `CameraCluster` order (via setter method)
        if self._cams_to_include is not None:
            self.cams_to_include = self._cams_to_include

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
            f"camera_cluster={self.camera_cluster},frame_groups:{len(self.frame_groups)})"
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

    def to_session_dict(
        self,
        video_to_idx: Dict[Video, int],
        labeled_frame_to_idx: Dict[LabeledFrame, int],
    ) -> dict:
        """Unstructure `RecordingSession` to an invertible dictionary.

        Args:
            video_to_idx: Dictionary of `Video` to index in `Labels.videos`.
            labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
                `Labels.labeled_frames`.

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

        # Store frame groups by frame index
        frame_group_dicts = []
        if len(labeled_frame_to_idx) > 0:  # Don't save if skipping labeled frames
            for frame_group in self._frame_group_by_frame_idx.values():
                # Only save `FrameGroup` if it has `InstanceGroup`s
                if len(frame_group.instance_groups) > 0:
                    frame_group_dict = frame_group.to_dict(
                        labeled_frame_to_idx=labeled_frame_to_idx
                    )
                    frame_group_dicts.append(frame_group_dict)

        return {
            "calibration": calibration_dict,
            "camcorder_to_video_idx_map": camcorder_to_video_idx_map,
            "frame_group_dicts": frame_group_dicts,
        }

    @classmethod
    def from_session_dict(
        cls,
        session_dict: dict,
        videos_list: List[Video],
        labeled_frames_list: List[LabeledFrame],
    ) -> "RecordingSession":
        """Restructure `RecordingSession` from an invertible dictionary.

        Args:
            session_dict: Dictionary of "calibration" and "camcorder_to_video_idx_map"
                needed to fully restructure a `RecordingSession`.
            videos_list: List containing `Video` objects (expected `Labels.videos`).
            labeled_frames_list: List containing `LabeledFrame` objects (expected
                `Labels.labeled_frames`).

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

        # Reconstruct all `FrameGroup` objects and add to `RecordingSession`
        frame_group_dicts = session_dict.get("frame_group_dicts", [])
        for frame_group_dict in frame_group_dicts:

            try:
                # Add `FrameGroup` to `RecordingSession`
                FrameGroup.from_dict(
                    frame_group_dict=frame_group_dict,
                    session=session,
                    labeled_frames_list=labeled_frames_list,
                )
            except ValueError as e:
                logger.warning(
                    f"Error reconstructing FrameGroup: {frame_group_dict}. Skipping..."
                    f"\n{e}"
                )

        return session

    @staticmethod
    def make_cattr(
        videos_list: List[Video],
        labeled_frames_list: Optional[List[LabeledFrame]] = None,
        labeled_frame_to_idx: Optional[Dict[LabeledFrame, int]] = None,
    ):
        """Make a `cattr.Converter` for `RecordingSession` serialization.

        Note: `labeled_frames_list` is needed to structure and `labeled_frame_to_idx` is
            needed to unstructure.

        Args:
            videos_list: List containing `Video` objects (expected `Labels.videos`).
            labeled_frames_list: List containing `LabeledFrame` objects (expected
                `Labels.labeled_frames`). Default is None. Needed for structuring.
            labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
                `Labels.labeled_frames`. Default is None. Needed for unstructuring.

        Returns:
            `cattr.Converter` object.
        """

        if labeled_frames_list is None and labeled_frame_to_idx is None:
            raise ValueError(
                "labeled_frames_list and labeled_frame_to_idx cannot both be None."
            )

        sessions_cattr = cattr.Converter()

        # Create the structure hook for `RecordingSession`
        if labeled_frames_list is not None:
            sessions_cattr.register_structure_hook(
                RecordingSession,
                lambda x, cls: RecordingSession.from_session_dict(
                    session_dict=x,
                    videos_list=videos_list,
                    labeled_frames_list=labeled_frames_list,
                ),
            )

        # Create the unstructure hook for `RecordingSession`
        if labeled_frame_to_idx is not None:
            video_to_idx = {video: i for i, video in enumerate(videos_list)}
            sessions_cattr.register_unstructure_hook(
                RecordingSession,
                lambda x: x.to_session_dict(
                    video_to_idx=video_to_idx, labeled_frame_to_idx=labeled_frame_to_idx
                ),
            )

        return sessions_cattr


@define
class FrameGroup:
    """Defines a group of `InstanceGroups` across views at the same frame index.

    Attributes:
        frame_idx: Frame index for the `FrameGroup`.
        session: `RecordingSession` object that the `FrameGroup` is in.
        instance_groups: List of `InstanceGroup`s in the `FrameGroup`.
        labeled_frames: List of `LabeledFrame`s in the `FrameGroup`.
        cameras: List of `Camcorder`s that have `LabeledFrame`s.
    """

    # Instance attributes
    frame_idx: int = field(validator=instance_of(int))
    session: RecordingSession = field(validator=instance_of(RecordingSession))
    _instance_groups: List[InstanceGroup] = field(
        factory=list,
        validator=deep_iterable(
            member_validator=instance_of(InstanceGroup),
            iterable_validator=instance_of(list),
        ),
    )  # Akin to `LabeledFrame.instances`
    _instance_group_name_registry: Set[str] = field(factory=set)

    # "Hidden" instance attributes

    # TODO(LM): This dict should be updated each time a LabeledFrame is added/removed
    # from the Labels object. Or if a video is added/removed from the RecordingSession.
    _labeled_frame_by_cam: Dict[Camcorder, LabeledFrame] = field(factory=dict)
    _cam_by_labeled_frame: Dict[LabeledFrame, Camcorder] = field(factory=dict)
    _instances_by_cam: Dict[Camcorder, Set[Instance]] = field(factory=dict)

    def __attrs_post_init__(self):
        """Initialize `FrameGroup` object."""

        # Check that `InstanceGroup` names unique (later added via add_instance_group)
        instance_group_name_registry_copy = set(self._instance_group_name_registry)
        for instance_group in self.instance_groups:
            if instance_group.name in instance_group_name_registry_copy:
                raise ValueError(
                    f"InstanceGroup name {instance_group.name} already in use. "
                    f"Please use a unique name not currently in the registry: "
                    f"{self._instance_group_name_registry}"
                )
            instance_group_name_registry_copy.add(instance_group.name)

        # Remove existing `FrameGroup` object from the `RecordingSession._frame_group_by_frame_idx`
        self.enforce_frame_idx_unique(self.session, self.frame_idx)

        # Add `FrameGroup` to `RecordingSession`
        self.session._frame_group_by_frame_idx[self.frame_idx] = self

        # Build `_labeled_frame_by_cam` and `_instances_by_cam` dictionary
        for camera in self.session.camera_cluster.cameras:
            self._instances_by_cam[camera] = set()
        for instance_group in self.instance_groups:
            self.add_instance_group(instance_group)

    @property
    def instance_groups(self) -> List[InstanceGroup]:
        """List of `InstanceGroup`s."""

        return self._instance_groups

    @instance_groups.setter
    def instance_groups(self, instance_groups: List[InstanceGroup]):
        """Setter for `instance_groups` that updates `LabeledFrame`s and `Instance`s."""

        instance_groups_to_remove = set(self.instance_groups) - set(instance_groups)
        instance_groups_to_add = set(instance_groups) - set(self.instance_groups)

        # Update the `_labeled_frame_by_cam` and `_instances_by_cam` dictionary
        for instance_group in instance_groups_to_remove:
            self.remove_instance_group(instance_group=instance_group)

        for instance_group in instance_groups_to_add:
            self.add_instance_group(instance_group=instance_group)

    @property
    def cams_to_include(self) -> Optional[List[Camcorder]]:
        """List of `Camcorder`s to include in this `FrameGroup`."""

        return self.session.cams_to_include

    @property
    def excluded_views(self) -> Optional[Tuple[str]]:
        """List of excluded views (names of Camcorders)."""

        return self.session.excluded_views

    @cams_to_include.setter
    def cams_to_include(self, cams_to_include: List[Camcorder]):
        """Setter for `cams_to_include` that sorts by `CameraCluster` order."""

        raise ValueError(
            "Cannot set `cams_to_include` directly. Please set `RecordingSession` "
            "attribute to update `FrameGroup`."
        )

    @property
    def labeled_frames(self) -> List[LabeledFrame]:
        """List of `LabeledFrame`s."""

        # TODO(LM): Revisit whether we need to return a list instead of a view object
        return list(self._labeled_frame_by_cam.values())

    @property
    def cameras(self) -> List[Camcorder]:
        """List of `Camcorder`s."""

        # TODO(LM): Revisit whether we need to return a list instead of a view object
        return list(self._labeled_frame_by_cam.keys())

    def numpy(
        self,
        instance_groups: Optional[List[InstanceGroup]] = None,
        pred_as_nan: bool = False,
    ) -> np.ndarray:
        """Numpy array of all `InstanceGroup`s in `FrameGroup.cams_to_include`.

        Args:
            instance_groups: `InstanceGroup`s to include. Default is None and uses all
                self.instance_groups.
            pred_as_nan: If True, then replaces `PredictedInstance`s with all nan
                self.dummy_instance. Default is False.

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
            instance_group_numpy = instance_group.numpy(
                pred_as_nan=pred_as_nan
            )  # M=all x N x 2
            instance_group_numpys.append(instance_group_numpy)

        frame_group_numpy = np.stack(instance_group_numpys, axis=1)  # M=all x T x N x 2
        cams_to_include_mask = np.array(
            [cam in self.cams_to_include for cam in self.cameras]
        )  # M=all x 1

        return frame_group_numpy[cams_to_include_mask]  # M=include x T x N x 2

    def add_instance(
        self,
        instance: Instance,
        camera: Camcorder,
        instance_group: Optional[InstanceGroup] = None,
    ):
        """Add an (existing) `Instance` to the `FrameGroup`.

        If no `InstanceGroup` is provided, then check the `Instance` is already in an
        `InstanceGroup` contained in the `FrameGroup`. Otherwise, add the `Instance` to
        the `InstanceGroup` and `FrameGroup`.

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

    def remove_instance(self, instance: Instance):
        """Removes an `Instance` from the `FrameGroup`.

        Args:
            instance: `Instance` to remove from the `FrameGroup`.
        """

        instance_group = self.get_instance_group(instance=instance)

        if instance_group is None:
            logger.warning(
                f"Instance {instance} not found in this FrameGroup.instance_groups: "
                f"{self.instance_groups}."
            )
            return

        # Remove the `Instance` from the `InstanceGroup`
        camera = instance_group.get_cam(instance=instance)
        instance_group.remove_instance(instance_or_cam=instance)

        # Remove the `Instance` from the `FrameGroup`
        self._instances_by_cam[camera].remove(instance)

        # Remove "empty" `LabeledFrame`s from the `FrameGroup`
        if len(self._instances_by_cam[camera]) < 1:
            self.remove_labeled_frame(labeled_frame_or_camera=camera)

    def add_instance_group(
        self, instance_group: Optional[InstanceGroup] = None
    ) -> InstanceGroup:
        """Add an `InstanceGroup` to the `FrameGroup`.

        This method updates the underlying dictionaries in calling add_instance:
                - `_instances_by_cam`
                - `_labeled_frame_by_cam`
                - `_cam_by_labeled_frame`

        Args:
            instance_group: `InstanceGroup` to add to the `FrameGroup`. If None, then
                create a new `InstanceGroup` and add it to the `FrameGroup`.

        Raises:
            ValueError: If the `InstanceGroup` is already in the `FrameGroup`.
        """

        if instance_group is None:

            # Find a unique name for the `InstanceGroup`
            instance_group_name = InstanceGroup.return_unique_name(
                name_registry=self._instance_group_name_registry
            )

            # Create an empty `InstanceGroup` with the frame index of the `FrameGroup`
            instance_group = InstanceGroup(
                name=instance_group_name,
                frame_idx=self.frame_idx,
                camera_cluster=self.session.camera_cluster,
            )
        else:
            # Ensure the `InstanceGroup` is compatible with the `FrameGroup`
            self._raise_if_instance_group_incompatible(instance_group=instance_group)

        # Add the `InstanceGroup` to the `FrameGroup`
        # We only expect this to be false on initialization
        if instance_group not in self.instance_groups:
            self.instance_groups.append(instance_group)

        # Add instance group name to the registry
        self._instance_group_name_registry.add(instance_group.name)

        # Add `Instance`s and `LabeledFrame`s to the `FrameGroup`
        for camera, instance in instance_group.instance_by_camcorder.items():
            self.add_instance(instance=instance, camera=camera)

        return instance_group

    def remove_instance_group(self, instance_group: InstanceGroup):
        """Remove an `InstanceGroup` from the `FrameGroup`."""

        if instance_group not in self.instance_groups:
            logger.warning(
                f"InstanceGroup {instance_group} not found in this FrameGroup: "
                f"{self.instance_groups}."
            )
            return

        # Remove the `InstanceGroup` from the `FrameGroup`
        self.instance_groups.remove(instance_group)
        self._instance_group_name_registry.remove(instance_group.name)

        # Remove the `Instance`s from the `FrameGroup`
        for camera, instance in instance_group.instance_by_camcorder.items():
            self._instances_by_cam[camera].remove(instance)

        # Remove the `LabeledFrame` from the `FrameGroup`
        labeled_frame = self.get_labeled_frame(camera=camera)
        if labeled_frame is not None:
            self.remove_labeled_frame(camera=camera)

    # TODO(LM): maintain this as a dictionary for quick lookups
    def get_instance_group(self, instance: Instance) -> Optional[InstanceGroup]:
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

    def set_instance_group_name(self, instance_group: InstanceGroup, name: str):
        """Set the name of an `InstanceGroup` in the `FrameGroup`."""

        self._raise_if_instance_group_not_in_frame_group(instance_group=instance_group)

        instance_group.set_name(
            name=name, name_registry=self._instance_group_name_registry
        )

    def add_labeled_frame(self, labeled_frame: LabeledFrame, camera: Camcorder):
        """Add a `LabeledFrame` to the `FrameGroup`.

        Args:
            labeled_frame: `LabeledFrame` to add to the `FrameGroup`.
            camera: `Camcorder` to link the `LabeledFrame` to.

        Raises:
            ValueError: If the `LabeledFrame` is not compatible with the `FrameGroup`.
        """

        # Some checks to ensure the `LabeledFrame` is compatible with the `FrameGroup`
        if not isinstance(labeled_frame, LabeledFrame):
            raise ValueError(
                f"Cannot add LabeledFrame: {labeled_frame} is not a LabeledFrame."
            )
        elif labeled_frame.frame_idx != self.frame_idx:
            raise ValueError(
                f"Cannot add LabeledFrame: Frame index {labeled_frame.frame_idx} does "
                f"not match FrameGroup frame index {self.frame_idx}."
            )
        elif not isinstance(camera, Camcorder):
            raise ValueError(f"Cannot add LabeledFrame: {camera} is not a Camcorder.")

        # Add the `LabeledFrame` to the `FrameGroup`
        self._labeled_frame_by_cam[camera] = labeled_frame
        self._cam_by_labeled_frame[labeled_frame] = camera

        # Add the `LabeledFrame` to the `RecordingSession`'s `Labels` object
        if (self.session.labels is not None) and (
            labeled_frame not in self.session.labels
        ):
            self.session.labels.append(labeled_frame)

    def remove_labeled_frame(
        self, labeled_frame_or_camera: Union[LabeledFrame, Camcorder]
    ):
        """Remove a `LabeledFrame` from the `FrameGroup`.

        Args:
            labeled_frame_or_camera: `LabeledFrame` or `Camcorder` to remove the
                `LabeledFrame` for.
        """

        if isinstance(labeled_frame_or_camera, LabeledFrame):
            labeled_frame: LabeledFrame = labeled_frame_or_camera
            camera = self.get_camera(labeled_frame=labeled_frame)

        elif isinstance(labeled_frame_or_camera, Camcorder):
            camera: Camcorder = labeled_frame_or_camera
            labeled_frame = self.get_labeled_frame(camera=camera)

        else:
            logger.warning(
                f"Cannot remove LabeledFrame: {labeled_frame_or_camera} is not a "
                "LabeledFrame or Camcorder."
            )

        # Remove the `LabeledFrame` from the `FrameGroup`
        self._labeled_frame_by_cam.pop(camera, None)
        self._cam_by_labeled_frame.pop(labeled_frame, None)

    def get_labeled_frame(self, camera: Camcorder) -> Optional[LabeledFrame]:
        """Get `LabeledFrame` for `Camcorder` if exists. Otherwise, None.

        Args:
            camera: `Camcorder`

        Returns:
            `LabeledFrame`
        """

        return self._labeled_frame_by_cam.get(camera, None)

    def get_camera(self, labeled_frame: LabeledFrame) -> Optional[Camcorder]:
        """Get `Camcorder` for `LabeledFrame` if exists. Otherwise, None.

        Args:
            labeled_frame: `LabeledFrame`

        Returns:
            `Camcorder`
        """

        return self._cam_by_labeled_frame.get(labeled_frame, None)

    def _create_and_add_labeled_frame(self, camera: Camcorder) -> LabeledFrame:
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

        labeled_frame = LabeledFrame(video=video, frame_idx=self.frame_idx)
        self.add_labeled_frame(labeled_frame=labeled_frame)

        return labeled_frame

    def _create_and_add_instance(
        self,
        instance_group: InstanceGroup,
        camera: Camcorder,
        labeled_frame: LabeledFrame,
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
        self._instances_by_cam[camera].add(instance)

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
                labeled_frame = self._create_and_add_labeled_frame(camera=cam)

            # Create an instance
            self._create_and_add_instance(
                instance_group=instance_group, camera=cam, labeled_frame=labeled_frame
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

    def _raise_if_instance_not_in_instance_group(self, instance: Instance):
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

    def _raise_if_instance_incompatibile(self, instance: Instance, camera: Camcorder):
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

    def _raise_if_instance_group_incompatible(self, instance_group: InstanceGroup):
        """Raise a ValueError if `InstanceGroup` is incompatible with `FrameGroup`.

        An `InstanceGroup` is incompatible if
            - the `frame_idx` does not match the `FrameGroup`'s `frame_idx`.
            - the `InstanceGroup.name` is already used in the `FrameGroup`.

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

        if instance_group.name in self._instance_group_name_registry:
            raise ValueError(
                f"InstanceGroup name {instance_group.name} is already registered in "
                "this FrameGroup's list of names: "
                f"{self._instance_group_name_registry}\n"
                "Please use a unique name for the new InstanceGroup."
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(frame_idx={self.frame_idx}, instance_groups:"
            f"{len(self.instance_groups)}, labeled_frames:{len(self.labeled_frames)}, "
            f"cameras:{len(self.cameras)}, session={self.session})"
        )

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
            frame_idx=frame_idx, session=session, instance_groups=instance_groups
        )

    def to_dict(
        self,
        labeled_frame_to_idx: Dict[LabeledFrame, int],
    ) -> Dict[str, Union[int, List[Dict[str, Any]]]]:
        """Convert `FrameGroup` to a dictionary.

        Args:
            labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
                `Labels.labeled_frames`.
        """

        # Create dictionary of `Instance` to `LabeledFrame` index (in
        # `Labels.labeled_frames`) and `Instance` index in `LabeledFrame.instances``.
        instance_to_lf_and_inst_idx: Dict[Instance, Tuple[str, str]] = {
            inst: (str(labeled_frame_to_idx[labeled_frame]), str(inst_idx))
            for labeled_frame in self.labeled_frames
            for inst_idx, inst in enumerate(labeled_frame.instances)
        }

        frame_group_dict = {
            "instance_groups": [
                instance_group.to_dict(
                    instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
                )
                for instance_group in self.instance_groups
            ],
        }

        return frame_group_dict

    @classmethod
    def from_dict(
        cls,
        frame_group_dict: Dict[str, Any],
        session: RecordingSession,
        labeled_frames_list: List[LabeledFrame],
    ):
        """Convert dictionary to `FrameGroup` object.

        Args:
            frame_group_dict: Dictionary of `FrameGroup` object.
            session: `RecordingSession` object.
            labeled_frames_list: List of `LabeledFrame` objects (expecting
                `Labels.labeled_frames`).

        Returns:
            `FrameGroup` object.
        """

        # Get `InstanceGroup` objects
        name_registry = set()
        instance_groups = []
        for instance_group_dict in frame_group_dict["instance_groups"]:
            instance_group = InstanceGroup.from_dict(
                instance_group_dict=instance_group_dict,
                name_registry=name_registry,
                labeled_frames_list=labeled_frames_list,
                camera_cluster=session.camera_cluster,
            )
            name_registry.add(instance_group.name)
            instance_groups.append(instance_group)

        return cls.from_instance_groups(
            session=session, instance_groups=instance_groups
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

        if session.frame_groups.get(frame_idx, None) is not None:
            # Remove existing `FrameGroup` object from the
            # `RecordingSession._frame_group_by_frame_idx`
            logger.warning(
                f"Frame index {frame_idx} for FrameGroup already exists in this "
                "RecordingSession. Overwriting."
            )
            session.frame_groups.pop(frame_idx)

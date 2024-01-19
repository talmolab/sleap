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
    _instance_by_camcorder: Dict[Camcorder, "Instance"] = field(factory=dict)
    _camcorder_by_instance: Dict["Instance", Camcorder] = field(factory=dict)

    def __attrs_post_init__(self):
        """Initialize `InstanceGroup` object."""

        for cam, instance in self._instance_by_camcorder.items():
            self._camcorder_by_instance[instance] = cam

    @property
    def instances(self) -> List["Instance"]:
        """List of `Instance` objects."""
        return list(self._instance_by_camcorder.values())

    @property
    def cameras(self) -> List[Camcorder]:
        """List of `Camcorder` objects."""
        return list(self._instance_by_camcorder.keys())

    def get_instance(self, cam: Camcorder) -> Optional["Instance"]:
        """Retrieve `Instance` linked to `Camcorder`.

        Args:
            camcorder: `Camcorder` object.

        Returns:
            If `Camcorder` in `self.camera_cluster`, then `Instance` object if found, else
            `None` if `Camcorder` has no linked `Instance`.
        """

        if cam not in self._instance_by_camcorder:
            logger.warning(
                f"Camcorder {cam.name} is not linked to a video in this "
                f"RecordingSession."
            )
            return None

        return self._instance_by_camcorder[cam]

    def get_cam(self, instance: "Instance") -> Optional[Camcorder]:
        """Retrieve `Camcorder` linked to `Instance`.

        Args:
            instance: `Instance` object.

        Returns:
            `Camcorder` object if found, else `None`.
        """

        if instance not in self._camcorder_by_instance:
            logger.warning(
                f"{instance} is not in this InstanceGroup's Instances: \n\t{self.instances}."
            )
            return None

        return self._camcorder_by_instance[instance]

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


@define
class FrameGroup:
    """Defines a group of `InstanceGroups` across views at the same frame index."""

    # Class attribute to keep track of frame indices across all `RecordingSession`s
    _frame_idx_registry: Dict["RecordingSession", Set[int]] = {}

    # Instance attributes
    frame_idx: int = field(validator=instance_of(int))
    instance_groups: List[InstanceGroup] = field(
        validator=deep_iterable(
            member_validator=instance_of(InstanceGroup),
            iterable_validator=instance_of(list),
        ),
    )  # Akin to `LabeledFrame.instances`
    session: "RecordingSession" = field(validator=instance_of("RecordingSession"))

    # Hidden attributes
    _cams_to_include: Optional[List[Camcorder]] = None

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

    @property
    def cams_to_include(self) -> Optional[List[Camcorder]]:
        """List of `Camcorder`s to include in this `FrameGroup`."""

        if self._cams_to_include is None:
            self._cams_to_include = self.session.camera_cluster.cameras.copy()
        return self._cams_to_include

    @cams_to_include.setter
    def cams_to_include(self, cams_to_include: List[Camcorder]):
        """Setter for `cams_to_include` attribute that sorts by `CameraCluster` order."""

        self._cams_to_include = cams_to_include.sort(
            key=self.session.camera_cluster.cameras.index
        )

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
        self._labeled_frames_by_cam = views.copy()
        logger.debug(
            f"\tUpdated LabeledFrames by Camcorder:\n\t{self._labeled_frames_by_cam}"
        )
        # Update `_instances_by_camera` dictionary and return it
        self._instances_by_cam = instances_by_cam.copy()
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
                # LM: No, since just using the number of locked instance groups will account for the dummy instances
                if instance is not None:
                    locked_instances_by_cam[cam].add(instance)

        # Only update if there were no errors
        self._locked_instances_by_cam = locked_instances_by_cam.copy()
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
        instances_by_camera: Dict[
            Camcorder, Set["Instance"]
        ] = self.instances_by_cam_to_include

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
            """Fill in missing instances with dummy instances up to max number of instances.

            Note that this function will mutate the input list in addition to returning the mutated list.

            Args:
                unlocked_instances_in_view: List of instances in a view that are not in a locked InstanceGroup.

            Returns:
                List of instances in a view that are not in a locked InstanceGroup with dummy instances appended.
            """

            # Subtracting the number of locked instance groups accounts for there being dummy instances in the locked instance groups.
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
        unlocked_instance_permutations: Dict[
            Camcorder, Iterator[Tuple["Instance"]]
        ] = {}
        for cam, instances_in_view in instances_by_camera.items():
            # Gather all instances for this cam from locked `InstanceGroup`s
            locked_instances_in_view: Set[
                "Instance"
            ] = self._locked_instances_by_cam.get(cam, set())

            # Remove locked instances from instances in view
            unlocked_instances_in_view: List["Instance"] = list(
                instances_in_view - locked_instances_in_view
            )

            # Fill in missing instances with dummy instances up to max number of instances
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
                # TODO(LM): This is where we would create the `InstanceGroup` objects instead
                cam: list(inst)
                for cam, inst in zip(self.cams_to_include, prod)
            }

        # TODO(LM): Should we return this as instance matrices or `InstanceGroup`s?
        # Answer: Definitely not instance matrices since we need to keep track of the `Instance`s,
        # but I kind of wonder if we could just return a list of `InstanceGroup`s instead of a dict
        # then the `InstanceGroup`

        return grouping_hypotheses

    @classmethod
    def from_instances_by_view(
        cls,
        session: "RecordingSession",
        instances_by_camera: Dict[Camcorder, List["Instance"]],
    ) -> Optional["FrameGroup"]:
        """Creates a `FrameGroup` object from a dictionary.

        Args:
            session: `RecordingSession` object.
            instances_by_camera: Dictionary with `Camcorder` keys and `LabeledFrame` values.

        Returns:
            `FrameGroup` object or None if no "real" (determined by `frame_idx` other
            than None) frames found.
        """
        ...

    @classmethod
    def from_instance_groups(
        cls,
        session: "RecordingSession",
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

        ...

    @classmethod
    def from_tensor(
        cls, session: "RecordingSession", frame_idx: int, tensor: np.ndarray
    ) -> Union["FrameGroup", List["FrameGroup"]]:
        """Creates a `FrameGroup` object from a tensor.

        Args:
            session: `RecordingSession` object.
            tensor: A tensor of shape M x F x T x N x 2 where
                M: # views, F: # frames = 1, T: # tracks, N: # nodes, 2: x, y

        Returns:
            `FrameGroup` object.
        """

        # Check value of F and assert that it is 1
        ...

        # If F is 1, then return a single `FrameGroup`
        ...

    @classmethod
    def enforce_frame_idx_unique(
        cls, session: "RecordingSession", frame_idx: int
    ) -> bool:
        """Enforces that all frame indices are unique in `RecordingSession`.

        Removes existing `FrameGroup` object from the `RecordingSession._frame_group_by_frame_idx`.

        Args:
            session: `RecordingSession` object.
            frame_idx: Frame index.
        """

        if frame_idx in cls._frame_idx_registry.get(session, set()):
            # Remove existing `FrameGroup` object from the `RecordingSession._frame_group_by_frame_idx`
            logger.warning(
                f"Frame index {frame_idx} for FrameGroup already exists in this RecordingSession. Overwriting."
            )
            session._frame_group_by_frame_idx.pop(frame_idx)


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
    _frame_group_by_frame_idx: Dict[int, FrameGroup] = field(factory=dict)

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
    def frame_groups(self) -> Dict[int, FrameGroup]:
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

        # Sort `CameraCluster._videos_by_session` by `Camcorder._video_by_session` order
        self.camera_cluster._videos_by_session[self].sort(
            key=camcorder._video_by_session[self].index
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

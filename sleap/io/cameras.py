"""Module for storing information for camera groups."""

from typing import List, Tuple, Optional

from attrs import define, field, cmp_using
from aniposelib.cameras import Camera
from aniposelib.cameras import CameraGroup
import numpy as np

@define
class Camcorder(Camera):
    """Class for storing information for camcorders.
    
    Attributes:
        matrix: Camera matrix.
        dist: Distortion coefficients.
        size: Image size.
        rvec: Rotation vector.
        tvec: Translation vector.
        name: Name of camera.
        extra_dist: Whether to use extra distortion coefficients.
    """

    matrix: np.ndarray = field(default=np.eye(3), eq=cmp_using(np.array_equal))
    dist: np.ndarray = field(default=np.eye(5), eq=cmp_using(np.array_equal))
    size: Optional[Tuple[int, int]] = field(default=None, eq=cmp_using(np.array_equal))
    rvec: np.ndarray = field(default=np.eye(3), eq=cmp_using(np.array_equal))
    tvec: np.ndarray = field(default=np.eye(3), eq=cmp_using(np.array_equal))
    name: Optional[str] = None
    extra_dist: bool = False

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, size={self.size})"

    @classmethod
    def from_dict(cls, d):
        """Creates a Camcorder object from a dictionary.
        
        Args:
            d: Dictionary with keys for matrix, dist, size, rvec, tvec, and name.
        
        Returns:
            Camcorder object.
        """
        cam = Camcorder()
        cam.load_dict(d)
        return cam

@define
class CameraCluster(CameraGroup):
    """Class for storing information for camera groups.
    
    Attributes:
        cameras: List of cameras.
        metadata: Set of metadata.
    """

    cameras: List[Camera] = field(factory=list)
    metadata: set = field(factory=set)

    def __attrs_post_init__(self):
        super().__init__(cameras=self.cameras, metadata=self.metadata)
    
    def __len__(self):
        return len(self.cameras)
    
    def __getitem__(self, idx):
        return self.cameras[idx]
    
    def __iter__(self):
        return iter(self.cameras)
    
    def __contains__(self, item):
        return item in self.cameras
    
    def __repr__(self):
        message = f"{self.__class__.__name__}(len={len(self)}: "
        for cam in self:
            message += f"{cam.name}, "
        return f"{message[:-2]})"

    @classmethod
    def load(cls, filename) -> "CameraCluster":
        """Loads cameras from a calibration.toml file.
        
        Args:
            filename: Path to calibration.toml file.
        
        Returns:
            CameraCluster object.
        """
        cam_group: CameraGroup = super().load(filename)
        cameras = [Camcorder(**cam.__dict__) for cam in cam_group.cameras]
        return cls(cameras=cameras, metadata=cam_group.metadata)
 
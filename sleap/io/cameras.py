"""Module for storing information for camera groups."""

from typing import List, Tuple, Optional, Union

from attrs import define, field, cmp_using
from aniposelib.cameras import Camera, FisheyeCamera, CameraGroup
import numpy as np

@define
class Camcorder:
    """Wrapper for Camera and FishEyeCamera classes.
    
    Attributes:
        camera: Camera or FishEyeCamera object.
    """

    camera: Optional[Union[Camera, FisheyeCamera]] = field(factory=None)

    def __eq__(self, other):
        if isinstance(other, Camcorder):
            for attr in vars(self):
                other_attr = getattr(other, attr)
                if isinstance(other_attr, np.ndarray):
                    if not np.array_equal(getattr(self, attr), other_attr):
                        return False
                elif getattr(self, attr) != other_attr:
                    return False
            return True
        return NotImplemented

    def __getattr__(self, attr):
        """Used to grab methods from `Camera` or `FishEyeCamera` objects."""
        return getattr(self.camera, attr)

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
        if 'fisheye' in d and d['fisheye']:
            cam = FisheyeCamera.from_dict(d)
        else:
            cam = Camera.from_dict(d)
        return Camcorder(cam)

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
        cameras = [Camcorder(cam) for cam in cam_group.cameras]
        return cls(cameras=cameras, metadata=cam_group.metadata)
 
"""Module for storing information for camera groups."""

from typing import List

import attrs
from aniposelib.cameras import CameraGroup
from aniposelib.cameras import Camera

@attrs.define
class CameraCluster(CameraGroup):
    """Class for storing information for camera groups.
    
    Attributes:
        cameras: List of cameras.
        metadata: Set of metadata.
    """

    cameras: List[Camera] = attrs.field(factory=list)
    metadata: set = attrs.field(factory=set)

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
        return cls(cameras=cam_group.cameras, metadata=cam_group.metadata)
 
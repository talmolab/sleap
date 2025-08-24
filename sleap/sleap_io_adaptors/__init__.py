"""
SLEAP-IO Adaptors Package

This package provides adaptor functions and classes to bridge between
sleap and sleap-io, implementing missing functionality and providing
compatibility layers.
"""

from .labels_adaptor import LabelsAdaptor
from .video_adaptor import VideoAdaptor, VideoBackendAdaptor
from .compatibility import patch_sleap_io_classes

__all__ = [
    "LabelsAdaptor",
    "VideoAdaptor",
    "VideoBackendAdaptor",
    "patch_sleap_io_classes",
]

# Auto-patch sleap-io classes when this package is imported

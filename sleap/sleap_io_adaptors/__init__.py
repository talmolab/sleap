"""SLEAP IO adaptors package for compatibility with sleap_io models.

This package provides compatibility methods and utilities for sleap_io models
to work seamlessly with the existing sleap codebase.
"""

from . import instance_utils
from . import skeleton_utils
from . import lf_labels_utils

__all__ = ["instance_utils", "skeleton_utils", "lf_labels_utils"]

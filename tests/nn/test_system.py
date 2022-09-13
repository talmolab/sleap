"""Tests for sleap.nn.system module.

Note: Most of this module cannot be tested effectively in CI since it expects GPUs to
be available.
"""

from sleap.nn.system import get_gpu_memory


def test_get_gpu_memory():
    # Make sure this doesn't throw an exception
    memory = get_gpu_memory()

"""Tests for sleap.nn.system module.

Note: Most of this module cannot be tested effectively in CI since it expects GPUs to
be available.
"""

from sleap.nn.system import get_gpu_memory
import os
import pytest


def test_get_gpu_memory():
    # Make sure this doesn't throw an exception
    memory = get_gpu_memory()


def test_get_gpu_memory_visible():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if cuda_visible_devices is None:
        pytest.skip("CUDA_VISIBLE_DEVICES not set.")

    elif cuda_visible_devices == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        gpu_memory = get_gpu_memory()

        assert len(gpu_memory) > 0
        assert len(gpu_memory) == 1

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    elif "," in cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        gpu_memory = get_gpu_memory()

        assert len(gpu_memory) > 0
        assert len(gpu_memory) == len(cuda_visible_devices.split(","))

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

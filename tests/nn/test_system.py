"""Tests for sleap.nn.system module.

Note: Most of this module cannot be tested effectively in CI since it expects GPUs to
be available.
"""

from sleap.nn.system import (
    get_gpu_memory,
    get_all_gpus,
    use_cpu_only,
    use_gpu,
    is_gpu_system,
)
import os
import pytest
import subprocess
import tensorflow as tf
import shutil
import platform


def test_get_gpu_memory():
    # Make sure this doesn't throw an exception
    memory = get_gpu_memory()


@pytest.mark.parametrize("cuda_visible_devices", ["0", "1", "0,1"])
def test_get_gpu_memory_visible(cuda_visible_devices):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not available.")

    # Get GPU indices from nvidia-smi
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    nvidia_indices = (
        subprocess.check_output(command).decode("utf-8").strip().split("\n")
    )

    # Set parameterized CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    gpu_memory = get_gpu_memory()

    if nvidia_indices == "0" or nvidia_indices == "1":
        assert len(gpu_memory) > 0
        assert len(gpu_memory) == 1

    elif nvidia_indices == "0,1":
        assert len(gpu_memory) > 0
        assert len(gpu_memory) == 2


def test_get_gpu_memory_no_nvidia_smi():
    # Backup current PATH
    old_path = os.environ["PATH"]

    # Set PATH to an empty string to simulate that nvidia-smi is not available
    os.environ["PATH"] = ""

    memory = get_gpu_memory()

    # Restore the original PATH
    os.environ["PATH"] = old_path

    assert memory == []


@pytest.mark.parametrize("cuda_visible_devices", ["invalid", "3,5", "-1"])
def test_get_gpu_memory_invalid_cuda_visible_devices(cuda_visible_devices):
    for value in cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = value

        memory = get_gpu_memory()

        # Cleanup CUDA_VISIBLE_DEVICES variable after the test
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        assert len(memory) == 0


def test_gpu_order_and_length():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not available.")

    # Get GPU indices from sleap.nn.system.get_all_gpus
    sleap_indices = [int(gpu.name.split(":")[-1]) for gpu in get_all_gpus()]

    # Get GPU indices from nvidia-smi
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    nvidia_indices = (
        subprocess.check_output(command).decode("utf-8").strip().split("\n")
    )
    nvidia_indices = [int(idx) for idx in nvidia_indices]

    # Assert that the order and length of GPU indices match
    assert sleap_indices == nvidia_indices


def test_gpu_device_order():
    """Indirectly tests GPU device order by ensuring environment variable is set."""

    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.skipif(
    not ("arm64" in platform.platform()),
    reason="Only test on macosx-arm64",
)
def test_reinitialize():
    """This test tries to change the devices after they have been initialized."""
    assert is_gpu_system()
    use_gpu(0)
    tf.zeros((1,)) + tf.ones((1,))
    # The following would normally throw:
    #   RuntimeError: Visible devices cannot be modified after being initialized
    use_cpu_only()

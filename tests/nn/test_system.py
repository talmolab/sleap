"""Tests for sleap.nn.system module.

Note: Most of this module cannot be tested effectively in CI since it expects GPUs to
be available.
"""

from sleap.nn.system import get_gpu_memory
from sleap.nn.system import get_all_gpus
import os
import pytest
import subprocess
import tensorflow as tf


def test_get_gpu_memory():
    # Make sure this doesn't throw an exception
    memory = get_gpu_memory()

@pytest.mark.parametrize("cuda_visible_devices", ["0", "1", "0,1"])
def test_get_gpu_memory_visible(cuda_visible_devices):
    # Get GPU indices from nvidia-smi
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    nvidia_indices = subprocess.check_output(command).decode('utf-8').strip().split('\n')

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    gpu_memory = get_gpu_memory()

    if nvidia_indices is None or nvidia_indices is []:
        pytest.skip("CUDA_VISIBLE_DEVICES not set.")

    elif nvidia_indices == "0" or nvidia_indices == "1":
        assert len(gpu_memory) > 0
        assert len(gpu_memory) == 1

    elif nvidia_indices == "0,1":
        assert len(gpu_memory) > 0
        assert len(gpu_memory) == 2

def test_gpu_order_and_length():
    # Get GPU indices from sleap.nn.system.get_all_gpus
    sleap_indices = [int(gpu.name.split(':')[-1]) for gpu in get_all_gpus()]
    
    # Get GPU indices from nvidia-smi
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    nvidia_indices = subprocess.check_output(command).decode('utf-8').strip().split('\n')
    nvidia_indices = [int(idx) for idx in nvidia_indices]

    # Assert that the order and length of GPU indices match
    assert sleap_indices == nvidia_indices
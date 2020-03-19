"""Utilities for working with the physical system (e.g., GPUs).

This module mostly provides convenience functions for changing the state of the runtime
environment by wrapping `tf.config` module functions.
"""

import tensorflow as tf
from typing import List, Optional, Text


def get_all_gpus() -> List[tf.config.PhysicalDevice]:
    """Return a list of GPUs including unavailable devices."""
    return tf.config.list_physical_devices("GPU")


def is_gpu_system() -> bool:
    """Return True if the system has discoverable GPUs."""
    return len(get_all_gpus()) > 0


def get_available_gpus() -> List[tf.config.PhysicalDevice]:
    """Return a list of available GPUs."""
    return tf.config.get_visible_devices("GPU")


def get_current_gpu() -> tf.config.PhysicalDevice:
    """Return the current (single) GPU device.

    Returns:
        The tf.config.PhysicalDevice for the available GPU.

        If no GPUs are available, returns None.

    Raises:
        ValueError: If multiple GPUs are available.
    """
    available_gpus = get_available_gpus()
    if len(available_gpus) == 0:
        return None
    elif len(available_gpus) == 1:
        return available_gpus[0]
    else:
        raise ValueError("Multiple GPUs are available.")


def use_cpu_only():
    """Hide GPUs from TensorFlow to ensure only the CPU is available."""
    tf.config.set_visible_devices([], "GPU")


def use_gpu(device_ind: int):
    """Make a single GPU available to TensorFlow.

    Args:
        device_ind: Index of the GPU within the list of system GPUs.
    """
    gpus = get_all_gpus()
    tf.config.set_visible_devices(gpus[device_ind], "GPU")


def use_first_gpu():
    """Make only the first GPU available to TensorFlow."""
    use_gpu(0)


def use_last_gpu():
    """Make only the last GPU available to TensorFlow."""
    use_gpu(-1)


def is_initialized(gpu: Optional[tf.config.PhysicalDevice] = None) -> bool:
    """Check if a physical GPU has been initialized without triggering initialization.

    Arguments:
        gpu: The GPU to check for initialization. If None, defaults to the current GPU.

    Returns:
        True if the GPU is initialized.

    Notes:
        Once initialized, the GPU cannot be hidden or change its memory policy.

        Initialization happens when a `tf.config.LogicalDevice` is created on the
        physical device, typically when the first tensor op runs.

        Checking if the GPU is initializing by querying the logical devices will trigger
        initialization, so this method provides an easy way of checking without
        modifying the system state.
    """
    if gpu is None:
        gpu = get_current_gpu()
    is_initialized = False
    try:
        # Get the current memory policy.
        current_policy = tf.config.experimental.get_memory_growth(gpu)

        # Try flipping it.
        tf.config.experimental.set_memory_growth(gpu, not current_policy)

        # Set it back if we were successful.
        tf.config.experimental.set_memory_growth(gpu, current_policy)

    except RuntimeError as ex:
        is_initialized = (
            len(ex.args) > 0
            and ex.args[0]
            == "Physical devices cannot be modified after being initialized"
        )
    return is_initialized


def disable_preallocation():
    """Disable preallocation of full GPU memory on all available GPUs.

    This enables memory growth policy so that TensorFlow will not pre-allocate all
    available GPU memory.

    Preallocation can be more efficient, but can lead to CUDA startup errors when the
    memory is not available (e.g., shared, multi-session and some *nix systems).

    See also: enable_gpu_preallocation
    """
    for gpu in get_available_gpus():
        tf.config.experimental.set_memory_growth(gpu, True)


def enable_preallocation():
    """Enable preallocation of full GPU memory on all available GPUs.

    This disables memory growth policy so that TensorFlow will pre-allocate all
    available GPU memory.

    Preallocation can be more efficient, but can lead to CUDA startup errors when the
    memory is not available (e.g., shared, multi-session and some *nix systems).

    See also: disable_gpu_preallocation
    """
    for gpu in get_available_gpus():
        tf.config.experimental.set_memory_growth(gpu, False)


def initialize_devices():
    """Initialize available physical devices as logical devices.

    If preallocation was enabled on the GPUs, this will trigger memory allocation.
    """
    tf.config.list_logical_devices()


def summary():
    """Print a summary of the state of the system."""
    gpus = get_available_gpus()
    all_gpus = get_all_gpus()
    if len(all_gpus) > 0:
        print(f"GPUs: {len(gpus)}/{len(all_gpus)} available")
        for gpu in all_gpus:
            print(f"  Device: {gpu.name}")
            print(f"         Available: {gpu in gpus}")
            print(f"        Initalized: {is_initialized(gpu)}")
            print(
                f"     Memory growth: {tf.config.experimental.get_memory_growth(gpu)}"
            )

    else:
        print("GPUs: None detected.")


def best_logical_device_name() -> Text:
    """Return the name of the best logical device for performance.

    This is particularly useful to use with `tf.device()` for explicit tensor placement.

    Returns:
        The name of the first logical GPU device if available, or alternatively the CPU.

    Notes:
        This will initialize the logical devices if they were not already!
    """
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 0:
        device_name = gpus[0].name
    else:
        cpus = tf.config.list_logical_devices("CPU")
        device_name = cpus[0].name
    return device_name

"""Load legacy metrics from SLEAP  <= v1.4.1"""

import numpy as np
import pickle


class MockSLEAPArray(np.ndarray):
    """
    Mock ndarray subclass to replace SLEAP array objects during unpickling.

    SLEAP's PointArray is an ndarray subclass, so we need this to be one too
    for numpy's unpickling to work properly.
    """

    def __new__(cls, shape=(0,), dtype=float):
        return np.ndarray.__new__(cls, shape, dtype)

    def __array_finalize__(self, obj):
        pass  # Required for ndarray subclasses


class MockSLEAPObject:
    """Mock class for non-array SLEAP objects."""

    pass


class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler that intercepts SLEAP class loading.

    When pickle tries to import sleap.instance.PointArray (which we don't have),
    we return our mock class instead so unpickling can complete.
    """

    def find_class(self, module, name):
        # Replace SLEAP classes with our mocks
        if "sleap" in module.lower():
            return MockSLEAPArray if "array" in name.lower() else MockSLEAPObject

        # Everything else loads normally
        return super().find_class(module, name)


def extract_arrays_from_object(obj, prefix="", arrays=None):
    """
    Recursively find and extract all numpy arrays and numeric values from an object.

    Searches through dicts, lists, object attributes, and nested arrays
    to find all numpy arrays and scalar numeric values.
    """
    if arrays is None:
        arrays = {}

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.dtype != np.dtype("object"):
            # Found a data array - store it
            arrays[prefix or "array"] = obj
        else:
            # Object array - recurse into each element
            for i, item in enumerate(obj.flat):
                extract_arrays_from_object(item, f"{prefix}[{i}]", arrays)

    # Handle scalar numeric values (int, float, np.number)
    elif isinstance(obj, (int, float, np.number)):
        arrays[prefix] = obj

    # Handle dictionaries - recurse into values
    elif isinstance(obj, dict):
        for key, val in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            extract_arrays_from_object(val, new_prefix, arrays)

    # Handle lists and tuples - recurse into elements
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            extract_arrays_from_object(item, f"{prefix}[{i}]", arrays)

    # Handle objects with attributes - recurse into __dict__
    elif hasattr(obj, "__dict__"):
        for key, val in obj.__dict__.items():
            if not key.startswith("_"):  # Skip private attributes
                new_prefix = f"{prefix}.{key}" if prefix else key
                extract_arrays_from_object(val, new_prefix, arrays)

    return arrays


def load_npz_extract_arrays(npz_file):
    """
    Load .npz file and extract all numpy arrays, even from pickled objects.

    Returns a dict mapping array names to numpy arrays and numeric values.
    """
    import io
    import zipfile
    import numpy.lib.format as fmt

    all_arrays = {}

    # Open the .npz file (which is actually a zip archive)
    with zipfile.ZipFile(npz_file, "r") as zf:
        for filename in zf.namelist():
            key = filename.replace(".npy", "")

            # Read the .npy file and check its dtype
            bio = io.BytesIO(zf.read(filename))
            version = fmt.read_magic(bio)

            # Use version-specific header reader (NumPy 2.x removed private functions)
            if version == (1, 0):
                shape, _, dtype = fmt.read_array_header_1_0(bio)
            elif version == (2, 0):
                shape, _, dtype = fmt.read_array_header_2_0(bio)
            else:
                raise ValueError(f"Unsupported .npy format version: {version}")

            if dtype == np.dtype("object"):
                # Pickled object - use our custom unpickler and extract arrays
                obj = CustomUnpickler(bio).load()
                extracted = extract_arrays_from_object(obj, prefix=key)
                all_arrays.update(extracted)

            else:
                # Regular numeric array - load directly
                bio.seek(0)
                all_arrays[key] = np.load(bio, allow_pickle=False)
    return all_arrays

"""
A miscellaneous set of utility functions. Try not to put things in here
unless they really have no other place.
"""

import os
import re

from pkg_resources import Requirement, resource_filename

import h5py as h5
import numpy as np
import attr
import psutil
import json
import rapidjson

from typing import Any, Dict, Hashable, Iterable, List, Optional


def json_loads(json_str: str) -> Dict:
    """
    A simple wrapper around the JSON decoder we are using.

    Args:
        json_str: JSON string to decode.

    Returns:
        Result of decoding JSON string.
    """
    try:
        return rapidjson.loads(json_str)
    except:
        return json.loads(json_str)


def json_dumps(d: Dict, filename: str = None):
    """
    A simple wrapper around the JSON encoder we are using.

    Args:
        d: The dict to write.
        filename: The filename to write to.

    Returns:
        None
    """

    encoder = rapidjson

    if filename:
        with open(filename, "w") as f:
            encoder.dump(d, f, ensure_ascii=False)
    else:
        return encoder.dumps(d)


def attr_to_dtype(cls: Any):
    """
    Converts classes with basic types to numpy composite dtypes.

    Arguments:
        cls: class to convert

    Returns:
        numpy dtype.
    """
    dtype_list = []
    for field in attr.fields(cls):
        if field.type == str:
            dtype_list.append((field.name, h5.special_dtype(vlen=str)))
        elif field.type is None:
            raise TypeError(
                f"numpy dtype for {cls} cannot be constructed because no "
                + "type information found. Make sure each field is type annotated."
            )
        elif field.type in [str, int, float, bool]:
            dtype_list.append((field.name, field.type))
        else:
            raise TypeError(
                f"numpy dtype for {cls} cannot be constructed because no "
                + f"{field.type} is not supported."
            )

    return np.dtype(dtype_list)


def usable_cpu_count() -> int:
    """
    Gets number of CPUs usable by the current process.

    Takes into consideration cpusets restrictions.

    Returns:
        The number of usable cpus
    """
    try:
        result = len(os.sched_getaffinity(0))
    except AttributeError:
        try:
            result = len(psutil.Process().cpu_affinity())
        except AttributeError:
            result = os.cpu_count()
    return result


def save_dict_to_hdf5(h5file: h5.File, path: str, dic: dict):
    """
    Saves dictionary to an HDF5 file.

    Calls itself recursively if items in dictionary are not
    `np.ndarray`, `np.int64`, `np.float64`, `str`, or bytes.
    Objects must be iterable.

    Args:
        h5file: The HDF5 filename object to save the data to.
            Assume it is open.
        path: The path to group save the dict under.
        dic: The dict to save.

    Raises:
        ValueError: If type for item in dict cannot be saved.


    Returns:
        None
    """
    for key, item in list(dic.items()):
        print(f"Saving {key}:")
        if item is None:
            h5file[path + key] = ""
        elif isinstance(item, bool):
            h5file[path + key] = int(item)
        elif isinstance(item, list):
            items_encoded = []
            for it in item:
                if isinstance(it, str):
                    items_encoded.append(it.encode("utf8"))
                else:
                    items_encoded.append(it)

            h5file[path + key] = np.asarray(items_encoded)
        elif isinstance(item, (str)):
            h5file[path + key] = item.encode("utf8")
        elif isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            save_dict_to_hdf5(h5file, path + key + "/", item)
        elif isinstance(item, int):
            h5file[path + key] = item
        else:
            raise ValueError("Cannot save %s type" % type(item))


def frame_list(frame_str: str) -> Optional[List[int]]:
    """
    Converts 'n-m' string to list of ints.

    Args:
        frame_str: string representing range

    Returns:
        List of ints, or None if string does not represent valid range.
    """

    # Handle ranges of frames. Must be of the form "1-200"
    if "-" in frame_str:
        min_max = frame_str.split("-")
        min_frame = int(min_max[0])
        max_frame = int(min_max[1])
        return list(range(min_frame, max_frame + 1))

    return [int(x) for x in frame_str.split(",")] if len(frame_str) else None


def uniquify(seq: Iterable[Hashable]) -> List:
    """
    Returns unique elements from list, preserving order.

    Note: This will not work on Python 3.5 or lower since dicts don't
    preserve order.

    Args:
        seq: The list to remove duplicates from.

    Returns:
        The unique elements from the input list extracted in original
        order.
    """

    # Raymond Hettinger
    # https://twitter.com/raymondh/status/944125570534621185
    return list(dict.fromkeys(seq))


def weak_filename_match(filename_a: str, filename_b: str) -> bool:
    """
    Check if paths probably point to same file.

    Compares the filename and names of two directories up.

    Args:
        filename_a: first path to check
        filename_b: path to check against first path

    Returns:
        True if the paths probably match.
    """
    # convert all path separators to /
    filename_a = filename_a.replace("\\", "/")
    filename_b = filename_b.replace("\\", "/")

    # remove unique pid so we can match tmp directories for same zip
    filename_a = re.sub("/tmp_\d+_", "tmp_", filename_a)
    filename_b = re.sub("/tmp_\d+_", "tmp_", filename_b)

    # check if last three parts of path match
    return filename_a.split("/")[-3:] == filename_b.split("/")[-3:]


def dict_cut(d: Dict, a: int, b: int) -> Dict:
    """
    Helper function for creating subdictionary by numeric indexing of items.

    Assumes that `dict.items()` will have a fixed order.

    Args:
        d: The dictionary to "split"
        a: Start index of range of items to include in result.
        b: End index of range of items to include in result.

    Returns:
        A dictionary that contains a subset of the items in the original dict.
    """
    return dict(list(d.items())[a:b])


def get_package_file(filename: str) -> str:
    """Returns full path to specified file within sleap package."""
    package_path = Requirement.parse("sleap")
    result = resource_filename(package_path, filename)
    return result


def get_config_file(shortname: str) -> str:
    """Returns full path to specified file in config directory of package."""
    local_path = f"sleap/config/{shortname}"
    return get_package_file(local_path)

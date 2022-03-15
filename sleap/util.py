"""
A miscellaneous set of utility functions. Try not to put things in here
unless they really have no other place.
"""

import os
import re
import shutil

from collections import defaultdict
from pkg_resources import Requirement, resource_filename

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import h5py as h5
import numpy as np
import attr
import psutil
import json
import rapidjson
import yaml

from typing import Any, Dict, Hashable, Iterable, List, Optional

import sleap.version as sleap_version


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

    # Handle ranges of frames. Must be of the form "1-200" (or "1,-200")
    if "-" in frame_str:
        min_max = frame_str.split("-")
        min_frame = int(min_max[0].rstrip(","))
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
    filename_a = re.sub(r"/tmp_\d+_", "tmp_", filename_a)
    filename_b = re.sub(r"/tmp_\d+_", "tmp_", filename_b)

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


def get_config_file(
    shortname: str, ignore_file_not_found: bool = False, get_defaults: bool = False
) -> str:
    """
    Returns the full path to the specified config file.

    The config file will be at ~/.sleap/<version>/<shortname>

    If that file doesn't yet exist, we'll look for a <shortname> file inside
    the package config directory (sleap/config) and copy the file into the
    user's config directory (creating the directory if needed).

    Args:
        shortname: The short filename, e.g., shortcuts.yaml
        ignore_file_not_found: If True, then return path for config file
            regardless of whether it exists.
        get_defaults: If True, then just return the path to default config file.

    Raises:
        FileNotFoundError: If the specified config file cannot be found.

    Returns:
        The full path to the specified config file.
    """

    if not get_defaults:
        desired_path = os.path.expanduser(
            f"~/.sleap/{sleap_version.__version__}/{shortname}"
        )

        # Make sure there's a ~/.sleap/<version>/ directory to store user version of the
        # config file.
        try:
            os.makedirs(os.path.expanduser(f"~/.sleap/{sleap_version.__version__}"))
        except FileExistsError:
            pass

        # If we don't care whether the file exists, just return the path
        if ignore_file_not_found:
            return desired_path

    # If we do care whether the file exists, check the package version of the
    # config file if we can't find the user version.

    if get_defaults or not os.path.exists(desired_path):
        package_path = get_package_file(f"sleap/config/{shortname}")
        if not os.path.exists(package_path):
            raise FileNotFoundError(
                f"Cannot locate {shortname} config file at {desired_path} or {package_path}."
            )

        if get_defaults:
            return package_path

        # Copy package version of config file into user config directory.
        shutil.copy(package_path, desired_path)

    return desired_path


def get_config_yaml(shortname: str, get_defaults: bool = False) -> dict:
    config_path = get_config_file(shortname, get_defaults=get_defaults)
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def save_config_yaml(shortname: str, data: Any) -> dict:
    yaml_path = get_config_file(shortname, ignore_file_not_found=True)
    with open(yaml_path, "w") as f:
        print(f"Saving config: {yaml_path}")
        yaml.dump(data, f)


def make_scoped_dictionary(
    flat_dict: Dict[str, Any], exclude_nones: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Converts dictionary with scoped keys to dictionary of dictionaries.

    Args:
        flat_dict: The dictionary to convert. Keys should be strings with
            `scope.foo` format.
        exclude_nodes: Whether to exclude items where value is None.

    Returns:
        Dictionary in which keys are `scope` and values are dictionary with
            `foo` (etc) as keys and original value of `scope.foo` as value.
    """
    scoped_dict = defaultdict(dict)

    for key, val in flat_dict.items():
        if "." in key and (not exclude_nones or val is not None):
            scope, subkey = key.split(".")

            scoped_dict[scope][subkey] = val

    return scoped_dict


def find_files_by_suffix(
    root_dir: str, suffix: str, prefix: str = "", depth: int = 0
) -> List[os.DirEntry]:
    """
    Returns list of files matching suffix, optionally searching in subdirs.

    Args:
        root_dir: Path to directory where we start searching
        suffix: File suffix to match (e.g., '.json')
        prefix: Optional file prefix to match
        depth: How many subdirectories deep to keep searching

    Returns:
        List of os.DirEntry objects.
    """

    with os.scandir(root_dir) as file_iterator:
        files = [file for file in file_iterator]

    subdir_paths = [file.path for file in files if file.is_dir()]
    matching_files = [
        file
        for file in files
        if file.is_file()
        and file.name.endswith(suffix)
        and (not prefix or file.name.startswith(prefix))
    ]

    if depth:
        for subdir in subdir_paths:
            matching_files.extend(
                find_files_by_suffix(subdir, suffix, prefix, depth=depth - 1)
            )

    return matching_files


def parse_uri_path(uri: str) -> str:
    """Parse a URI starting with 'file:///' to a posix path."""
    return Path(url2pathname(urlparse(unquote(uri)).path)).as_posix()

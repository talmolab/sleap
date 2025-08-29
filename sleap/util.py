"""A miscellaneous set of utility functions.

Note: to avoid circular imports, this file is used for utility functions that do not
depend on any other modules in the package.

Try not to put things in here unless they really have no other place.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
import cv2
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Hashable, Iterable, List, Optional

from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

try:
    from importlib.resources import files  # New in 3.9+
except ImportError:
    from importlib_resources import files  # TODO(LM): Upgrade to importlib.resources.

import attr
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import rapidjson
import rich.progress
import seaborn as sns
import yaml

import sleap.version as sleap_version

if TYPE_CHECKING:
    from rich.progress import Task


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: Task) -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.1f} FPS", style="progress.data.speed")


def json_loads(json_str: str) -> Dict:
    """A simple wrapper around the JSON decoder we are using.

    Args:
        json_str: JSON string to decode.

    Returns:
        Result of decoding JSON string.
    """
    try:
        return rapidjson.loads(json_str)
    except Exception:
        return json.loads(json_str)


def json_dumps(d: Dict, filename: str = None):
    """A simple wrapper around the JSON encoder we are using.

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
    """Converts classes with basic types to numpy composite dtypes.

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
    """Gets number of CPUs usable by the current process.

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
    """Saves dictionary to an HDF5 file.

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
    """Converts 'n-m' string to list of ints.

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


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resizes single image with shape (height, width, channels)."""
    height, width, channels = img.shape
    new_height, new_width = int(height // (1 / scale)), int(width // (1 / scale))

    # Note that OpenCV takes shape as (width, height).

    if channels == 1:
        # opencv doesn't want a single channel to have its own dimension
        img = cv2.resize(img[:, :], (new_width, new_height))[..., None]
    else:
        img = cv2.resize(img, (new_width, new_height))

    return img


def resize_images(images: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return images
    return np.stack([resize_image(img, scale) for img in images])


def uniquify(seq: Iterable[Hashable]) -> List:
    """Returns unique elements from list, preserving order.

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
    """Check if paths probably point to same file.

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
    """Helper function for creating subdictionary by numeric indexing of items.

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

    data_path: Path = files("sleap").joinpath(filename)
    return data_path.as_posix()


def get_config_file(
    shortname: str, ignore_file_not_found: bool = False, get_defaults: bool = False
) -> str:
    """Returns the full path to the specified config file.

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

    desired_path = Path.home() / f".sleap/{sleap_version.__version__}/{shortname}"

    # Make sure there's a ~/.sleap/<version>/ directory to store user version of
    # the config file.
    desired_path.parent.mkdir(parents=True, exist_ok=True)

    # If we don't care whether the file exists, just return the path
    if ignore_file_not_found:
        return desired_path

    # If we do care whether the file exists, check the package version of the
    # config file if we can't find the user version.
    if get_defaults or not desired_path.exists():
        package_path = get_package_file(f"config/{shortname}")
        package_path = Path(package_path)
        if not package_path.exists():
            raise FileNotFoundError(
                f"Cannot locate {shortname} config file at {desired_path} or "
                f"{package_path}."
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
    """Returns list of files matching suffix, optionally searching in subdirs.

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


def imgfig(
    size: float | tuple = 6, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Create a tight figure for image plotting.

    Args:
        size: Scalar or 2-tuple specifying the (width, height) of the figure in inches.
            If scalar, will assume equal width and height.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    fig = plt.figure(figsize=(scale * size[0], scale * size[1]), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def plot_img(
    img: np.ndarray, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Plot an image in a tight figure."""
    if hasattr(img, "numpy"):
        img = img.numpy()

    if img.shape[0] == 1:
        # Squeeze out batch singleton dimension.
        img = img.squeeze(axis=0)

    # Check if image is grayscale (single channel).
    grayscale = img.shape[-1] == 1
    if grayscale:
        # Squeeze out singleton channel.
        img = img.squeeze(axis=-1)

    # Normalize the range of pixel values.
    img_min = img.min()
    img_max = img.max()
    if img_min < 0.0 or img_max > 1.0:
        img = (img - img_min) / (img_max - img_min)

    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )

    ax = fig.gca()
    ax.imshow(
        img,
        cmap="gray" if grayscale else None,
        origin="upper",
        extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
    )
    return fig


def plot_instance(
    instance,
    skeleton=None,
    cmap=None,
    color_by_node=False,
    lw=2,
    ms=10,
    bbox=None,
    scale=1.0,
    **kwargs,
):
    """Plot a single instance with edge coloring."""
    if cmap is None:
        cmap = sns.color_palette("tab20")

    if skeleton is None and hasattr(instance, "skeleton"):
        skeleton = instance.skeleton

    if skeleton is None:
        color_by_node = True
    else:
        if len(skeleton.edges) == 0:
            color_by_node = True

    if hasattr(instance, "numpy"):
        inst_pts = instance.numpy()
    else:
        inst_pts = instance

    h_lines = []
    if color_by_node:
        for k, (x, y) in enumerate(inst_pts):
            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(x, y, ".", ms=ms, c=cmap[k % len(cmap)], **kwargs)
            h_lines.append(h_lines_k)

    else:
        for k, (src_node, dst_node) in enumerate(skeleton.edges):
            src_pt = instance.points_array[instance.skeleton.node_to_index(src_node)]
            dst_pt = instance.points_array[instance.skeleton.node_to_index(dst_node)]

            x = np.array([src_pt[0], dst_pt[0]])
            y = np.array([src_pt[1], dst_pt[1]])

            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(
                x, y, ".-", ms=ms, lw=lw, c=cmap[k % len(cmap)], **kwargs
            )

            h_lines.append(h_lines_k)

    return h_lines


def plot_instances(
    instances, skeleton=None, cmap=None, color_by_track=False, tracks=None, **kwargs
):
    """Plot a list of instances with identity coloring."""

    if cmap is None:
        cmap = sns.color_palette("tab10")

    if color_by_track and tracks is None:
        # Infer tracks for ordering if not provided.
        tracks = set()
        for instance in instances:
            tracks.add(instance.track)

        # Sort by spawned frame.
        tracks = sorted(list(tracks), key=lambda track: track.name)

    h_lines = []
    for i, instance in enumerate(instances):
        if color_by_track:
            if instance.track is None:
                raise ValueError(
                    "Instances must have a set track when coloring by track."
                )

            if instance.track not in tracks:
                raise ValueError("Instance has a track not found in specified tracks.")

            color = cmap[tracks.index(instance.track) % len(cmap)]

        else:
            # Color by identity (order in list).
            color = cmap[i % len(cmap)]

        h_lines_i = plot_instance(instance, skeleton=skeleton, cmap=[color], **kwargs)
        h_lines.append(h_lines_i)

    return h_lines

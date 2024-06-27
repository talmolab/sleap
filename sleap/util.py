"""A miscellaneous set of utility functions. 

Try not to put things in here unless they really have no other place.
"""

import base64
import json
import os
import re
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import attr
from attrs import field
from attrs.validators import is_callable, optional, and_
import h5py as h5
import numpy as np
import psutil
import rapidjson
import yaml

try:
    from importlib.resources import files  # New in 3.9+
except ImportError:
    from importlib_resources import files  # TODO(LM): Upgrade to importlib.resources.
from PIL import Image

import sleap.version as sleap_version

# TODO(LM): Open a PR to attrs to add this to the library, then remove once we upgrade.
@attr.s(repr=False, slots=True, hash=True)
class _DeepIterableConverter:
    member_converter: Callable = field(validator=is_callable())
    iterable_converter: Optional[Callable] = field(
        default=None, validator=optional(is_callable())
    )

    def __call__(self, value):
        """We use a callable class to be able to change the ``__repr__``."""

        new_value = []
        for member in value:
            new_value.append(self.member_converter(member))

        if self.iterable_converter is not None:
            return self.iterable_converter(new_value)
        else:
            return type(value)(new_value)

    def __repr__(self):
        iterable_identifier = (
            "" if self.iterable_converter is None else f" {self.iterable_converter!r}"
        )
        return (
            "<deep_iterable converter for{iterable_identifier}"
            " iterables of {member!r}>"
        ).format(
            iterable_identifier=iterable_identifier,
            member=self.member_converter,
        )


# TODO(LM): Open a PR to attrs to add this to the library, then remove once we upgrade.
def deep_iterable_converter(member_converter, iterable_converter=None):
    """A converter that performs deep conversion of an iterable.

    :param member_converter: Converter(s) to apply to iterable members
    :param iterable_converter: Converter to apply to iterable itself
        (optional)

    .. versionadded:: not added to attrs yet

    :raises TypeError: if any sub-converters fail
    """
    if isinstance(member_converter, (list, tuple)):
        member_converter = and_(*member_converter)
    return _DeepIterableConverter(member_converter, iterable_converter)


def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: Optional[float] = None,
    stddev: float = 0.025,
    use_cocoeval: bool = True,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        use_cocoeval: Indicates whether the OKS score is calculated like cocoeval
            method or not. True indicating the score is calculated using the
            cocoeval method (widely used and the code can be found here at
            https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192C5-L233C20)
            and False indicating the score is calculated using the method exactly
            as given in the paper referenced in the Notes below.
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement ** 2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    if use_cocoeval:
        # If use_cocoeval is True, then compute normalization factor according to cocoeval.
        spread_factor = (2 * stddev) ** 2
        scale_factor = 2 * (scale + np.spacing(1))
    else:
        # If use_cocoeval is False, then compute normalization factor according to the paper.
        spread_factor = stddev ** 2
        scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float64"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def json_loads(json_str: str) -> Dict:
    """A simple wrapper around the JSON decoder we are using.

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

    desired_path = None  # Handle case where get_defaults, but cannot find package_path

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
        package_path = get_package_file(f"config/{shortname}")
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


def decode_preview_image(img_b64: bytes) -> Image:
    """Decode a skeleton preview image byte string representation to a `PIL.Image`

    Args:
        img_b64: a byte string representation of a skeleton preview image

    Returns:
        A PIL.Image of the skeleton preview
    """
    bytes = base64.b64decode(img_b64)
    buffer = BytesIO(bytes)
    img = Image.open(buffer)
    return img

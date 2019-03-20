"""
A miscellaneous set of utility functions. Try not to put things in here
unless they really have no other place.
"""

import h5py as h5
import numpy as np
import attr

from typing import Callable

def attr_to_dtype(cls):
    dtype_list = []
    for field in attr.fields(cls):
        if field.type == str:
            dtype_list.append((field.name, h5.special_dtype(vlen=str)))
        elif field.type is None:
            raise TypeError(f"numpy dtype for {cls} cannot be constructed because no " +
                            "type information found. Make sure each field is type annotated.")
        elif field.type in [str, int, float, bool]:
            dtype_list.append((field.name, field.type))
        else:
            raise TypeError(f"numpy dtype for {cls} cannot be constructed because no " +
                            f"{field.type} is not supported.")

    return np.dtype(dtype_list)


def try_open_file(open: Callable, *args, **kwargs) -> object:
    """
    A quick little utility method to try to open a file with the
    full path, if that doesn't work, try with the basename, if
    that doesn't work, return None

    Args:
        open: A callable to invoke to open the file
        *args: The arguments to pass to the open callable
        **kwargs: The keyword arguments to pass to the open callable

    Returns:
        The return value of the callable open function or None if no
        success.
    """
    try:
        return open(*args, **kwargs)
    except FileNotFoundError:
        try:
            return open(*args, **kwargs)
        except FileNotFoundError:
            return None
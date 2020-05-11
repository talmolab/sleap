"""
Utilities for working with file paths.
"""

import os
from typing import Callable, Dict, List, Optional, Text, Tuple

from sleap import util


def list_file_missing(filenames):
    """Given a list of filenames, returns list of whether file exists."""
    return [not os.path.exists(filename) for filename in filenames]


def filenames_prefix_change(
    filenames,
    old_prefix,
    new_prefix,
    missing: bool = None,
    confirm_callback: Optional[Callable] = None,
):
    """
    Finds missing files by changing the initial part of paths.

    Args:
        filenames: The list of filenames, needn't all be missing.
        old_prefix: Initial part of path to replace.
        new_prefix: Initial part with which to replace it.
        missing: List of which files are known to be missing; if not given,
            then we'll check each file.
        confirm_callback: If given, then we'll call this before applying
            change to confirm that user wants to apply the change.

    Returns:
        None; `filenames` (and `missing`, if given) have new data.
    """

    if not filenames or not old_prefix or not new_prefix:
        return

    # Ask for confirmation if there's a confirmation callback given
    need_to_ask = True if callable(confirm_callback) else False

    # Try changing every filename unless we're given list of which are missing
    check = missing if missing else [True] * len(filenames)

    # Just to be on the safe side, make sure this list covers all filenames
    if len(check) < len(filenames):
        check.extend([True] * (len(filenames) - len(check)))

    for i, filename in enumerate(filenames):
        if check[i]:
            if filename.startswith(old_prefix):
                try_filename = filename.replace(old_prefix, new_prefix)
                try_filename = fix_path_separator(try_filename)

                if os.path.exists(try_filename):
                    # Check if user would like to apply change to all paths
                    # with the same initial segment.
                    if need_to_ask and not confirm_callback():
                        return

                    # We're still here, so we can go ahead and replace
                    need_to_ask = False
                    filenames[i] = try_filename
                    check[i] = False

                    # Save prefix change in config file so that it can be used
                    # automatically in the future
                    save_path_prefix_replacement(old_prefix, new_prefix)


def fix_path_separator(path: str):
    return path.replace("\\", "/")


def find_changed_subpath(old_path: str, new_path: str) -> Tuple[str, str]:
    """Finds the smallest initial section of path that was changed.

    Args:
        old_path: Old path
        new_path: New path

    Returns:
        (initial part of old path), (corresponding replacement in new path)
    """
    seps = ("/", "\\")

    # Find overlap at end of paths
    old_common = ""
    new_char_idx = -1
    for old_char_idx in range(len(old_path) - 1, 0, -1):
        old_char = old_path[old_char_idx]
        new_char = new_path[new_char_idx]
        if old_char == new_char or old_char in seps and new_char in seps:
            old_common = old_char + old_common
            new_char_idx -= 1
        else:
            break

    # Get the initial part of the old path which was replaced, and the
    # initial part of the new path which replaced it.
    old_initial = old_path[: old_char_idx + 1]
    new_inital = new_path[: new_char_idx + 1] if new_char_idx < -1 else new_path

    return (old_initial, new_inital)


def fix_paths_with_saved_prefix(
    filenames,
    missing: Optional[List[bool]] = None,
    path_prefix_conversions: Optional[Dict[Text, Text]] = None,
):
    if path_prefix_conversions is None:
        path_prefix_conversions = util.get_config_yaml("path_prefixes.yaml")

    if path_prefix_conversions is None:
        return

    for i, filename in enumerate(filenames):
        if missing is not None:
            if not missing[i]:
                continue
        elif os.path.exists(filename):
            continue

        for old_prefix, new_prefix in path_prefix_conversions.items():
            if filename.startswith(old_prefix):
                try_filename = filename.replace(old_prefix, new_prefix)
                try_filename = fix_path_separator(try_filename)

                if os.path.exists(try_filename):
                    filenames[i] = try_filename
                    if missing is not None:
                        missing[i] = False
                    continue


def save_path_prefix_replacement(old_prefix: str, new_prefix: str):
    data = util.get_config_yaml("path_prefixes.yaml") or dict()
    data[old_prefix] = new_prefix
    util.save_config_yaml("path_prefixes.yaml", data)

"""Miscellaneous utility functions for data processing."""

from typing import Any, List

def ensure_list(x: Any) -> List[Any]:
    """Convert the input into a list if it is not already."""
    if not isinstance(x, list):
        return [x]
    return x

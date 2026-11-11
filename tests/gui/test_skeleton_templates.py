"""Tests for shipped skeleton template files."""

import base64
import io
import json

from PIL import Image

import sleap.util
from sleap.gui.commands import OpenSkeleton


def test_centroid_template_loads_single_node():
    """The shipped centroid.json loads to a single-node, edge-less skeleton."""
    path = sleap.util.get_package_file("skeletons/centroid.json")
    skeleton = OpenSkeleton.load_skeleton(path)
    assert skeleton.node_names == ["centroid"]
    assert len(skeleton.edges) == 0


def test_centroid_template_preview_is_rgba():
    """The centroid.json preview image decodes to an RGBA PIL image."""
    path = sleap.util.get_package_file("skeletons/centroid.json")
    with open(path, "r") as f:
        skeleton_data = json.load(f)

    b64 = skeleton_data["preview_image"]["py/b64"]
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert image.mode == "RGBA"

import pytest

from sleap_io import Skeleton, load_skeleton

TEST_FLY_LEGS_SKELETON = "tests/data/skeleton/fly_skeleton_legs.json"
TEST_FLY_LEGS_SKELETON_DICT = "tests/data/skeleton/fly_skeleton_legs_pystate_dict.json"


@pytest.fixture
def fly_legs_skeleton_json():
    """Path to fly_skeleton_legs.json

    This skeleton json has py/state in tuple format.
    """
    return TEST_FLY_LEGS_SKELETON


@pytest.fixture
def fly_legs_skeleton_dict_json():
    """Path to fly_skeleton_legs_pystate_dict.json

    This skeleton json has py/state dict format.
    """
    return TEST_FLY_LEGS_SKELETON_DICT


@pytest.fixture
def stickman():
    # Make a skeleton with a space in its name to test things.
    node_names = [
        "head",
        "neck",
        "body",
        "right-arm",
        "left-arm",
        "right-leg",
        "left-leg",
    ]
    stickman = Skeleton()
    stickman.add_nodes(node_names)

    edges = [
        ("neck", "head"),
        ("body", "neck"),
        ("body", "right-arm"),
        ("body", "left-arm"),
        ("body", "right-leg"),
        ("body", "left-leg"),
    ]
    stickman.add_edges(edges)
    stickman.add_symmetry("left-arm", "right-arm")
    stickman.add_symmetry("left-leg", "right-leg")

    return stickman


@pytest.fixture
def skeleton():
    # Create a simple skeleton object
    node_names = ["head", "thorax", "abdomen", "left-wing", "right-wing"]
    skeleton = Skeleton(nodes=node_names, name="Fly")

    edges = [
        ("head", "thorax"),
        ("thorax", "abdomen"),
        ("thorax", "left-wing"),
        ("thorax", "right-wing"),
    ]
    skeleton.add_edges(edges)
    skeleton.add_symmetry("left-wing", "right-wing")

    return skeleton


@pytest.fixture
def flies13_skeleton():
    return load_skeleton("sleap/skeletons/flies13.json")

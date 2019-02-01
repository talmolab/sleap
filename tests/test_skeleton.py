import os
import pytest

from leap.skeleton import Skeleton

JSON_TEST_FILENAME = 'tests/test_skeleton.json'

@pytest.fixture(autouse=True)
def cleanup_file():
    yield
    if os.path.isfile(JSON_TEST_FILENAME):
        os.remove(JSON_TEST_FILENAME)

@pytest.fixture
def skeleton():

    # Create a simple skeleton object
    skeleton = Skeleton("Bug")
    skeleton.add_node(name="head")
    skeleton.add_node(name="thorax")
    skeleton.add_node(name="abdomen")
    skeleton.add_node(name="left-wing")
    skeleton.add_node(name="right-wing")
    skeleton.add_edge(source="head", destination="thorax")
    skeleton.add_edge(source="thorax", destination="abdomen")
    skeleton.add_edge(source="thorax", destination="left-wing")
    skeleton.add_edge(source="thorax", destination="right-wing")
    skeleton.add_symmetry(node1="left-wing", node2="right-wing")

    return skeleton

def test_add_dupe_node(skeleton):
    """
    Test if adding a node with the same name to skeleton throws an exception.
    """
    with pytest.raises(ValueError):
        skeleton.add_node(name="head")

def test_add_dupe_edge(skeleton):
    """
    Test if adding a duplicate edge to skeleton throws an exception.
    """
    with pytest.raises(ValueError):
        skeleton.add_edge(source="head", destination="thorax")

def test_no_node_edge(skeleton):
    """
    Test if adding an edge with a non-existent node to the skeleton throws an exception.
    """
    with pytest.raises(ValueError):
        skeleton.add_edge(source="non-existent-node-name", destination="thorax")
    with pytest.raises(ValueError):
        skeleton.add_edge(source="head", destination="non-existent-node-name")

def test_json(skeleton):
    """
    Test saving and loading a Skeleton object in JSON.
    """

    # Save it to a JSON file
    skeleton.save_json(JSON_TEST_FILENAME)

    # Load the JSON object back in
    skeleton_copy = Skeleton.load_json(JSON_TEST_FILENAME)

    # Make sure we get back the same skeleton we saved.
    assert(skeleton == skeleton_copy)


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

def test_remove_node(skeleton):
    """
    Test whether we can delete nodes successfully.
    """
    skeleton.add_node("test_node1")
    skeleton.add_node("test_node2")
    skeleton.add_edge("test_node1", "test_node2")
    skeleton.delete_node("test_node1")

    assert not skeleton.graph.has_node("test_node1")
    assert not skeleton.graph.has_edge("test_node1", "test_node2")
    assert skeleton.graph.has_node("test_node2")

def test_remove_node_non_exist(skeleton):
    """
    Test whether deleting a non-existent node throws and exception.
    """
    with pytest.raises(ValueError):
        skeleton.delete_node("non-existent-node")

def test_no_node_edge(skeleton):
    """
    Test if adding an edge with a non-existent node to the skeleton throws an exception.
    """
    with pytest.raises(ValueError):
        skeleton.add_edge(source="non-existent-node-name", destination="thorax")
    with pytest.raises(ValueError):
        skeleton.add_edge(source="head", destination="non-existent-node-name")

def test_getitem_node(skeleton):
    """
    Test whether we can access nodes of the skeleton via subscript notation.
    """

    # Make sure attempting to get a non-existent node throws exception
    with pytest.raises(ValueError):
        skeleton["non_exist_node"]

    # Now try to get the head node
    assert(skeleton["head"] is not None)

def test_node_rename(skeleton):
    """
    Test if we can rename a node in place.
    """

    skeleton.relabel_nodes({"head": "new_head_name"})

    # Make sure the old "head" doesn't exist
    with pytest.raises(ValueError):
        skeleton["head"]

    # Make sure new head as the correct name
    assert(skeleton["new_head_name"] is not None)


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


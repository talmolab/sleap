import os
import copy

import pytest

from sleap.skeleton import Skeleton


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

    assert not skeleton.has_node("test_node1")
    assert not skeleton.has_edge("test_node1", "test_node2")
    assert skeleton.has_node("test_node2")


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

    # Make sure new head has the correct name
    assert(skeleton["new_head_name"] is not None)


def test_eq():
    s1 = Skeleton("s1")
    s1.add_nodes(['1','2','3','4','5','6'])
    s1.add_edge('1', '2')
    s1.add_edge('3', '4')
    s1.add_edge('5', '6')
    s1.add_symmetry('3', '6')


    # Make a copy check that they are equal
    s2 = copy.deepcopy(s1)
    assert s1 == s2

    # Add an edge, check that they are not equal
    s2 = copy.deepcopy(s1)
    s2.add_edge('5', '1')
    assert s1 != s2

    # Add a symmetry edge, not equal
    s2 = copy.deepcopy(s1)
    s2.add_symmetry('5', '1')
    assert s1 != s2

    # Delete a node
    s2 = copy.deepcopy(s1)
    s2.delete_node('5')
    assert s1 != s2

    # Delete and edge, not equal
    s2 = copy.deepcopy(s1)
    s2.delete_edge('1', '2')
    assert s1 != s2

    # FIXME: Probably shouldn't test it this way.
    # Add a value to a nodes dict, make sure they are not equal. This is touching the
    # internal _graph storage of the skeleton which probably should be avoided. But, I
    # wanted to test this just in case we add attributes to the nodes in the future.
    s2 = copy.deepcopy(s1)
    s2._graph.nodes['1']['test'] = 5
    assert s1 != s2

def test_symmetry():
    s1 = Skeleton("s1")
    s1.add_nodes(['1','2','3','4','5','6'])
    s1.add_edge('1', '2')
    s1.add_edge('3', '4')
    s1.add_edge('5', '6')
    s1.add_symmetry('1', '5')
    s1.add_symmetry('3', '6')

    assert s1.get_symmetry("1") == "5"
    assert s1.get_symmetry("5") == "1"

    assert s1.get_symmetry("3") == "6"

    # Cannot add more than one symmetry to a node
    with pytest.raises(ValueError):
        s1.add_symmetry('1', '6')
    with pytest.raises(ValueError):
        s1.add_symmetry('6', '1')

def test_json(skeleton, tmpdir):
    """
    Test saving and loading a Skeleton object in JSON.
    """
    JSON_TEST_FILENAME = os.path.join(tmpdir, 'skeleton.json')

    # Save it to a JSON file
    skeleton.save_json(JSON_TEST_FILENAME)

    # Load the JSON object back in
    skeleton_copy = Skeleton.load_json(JSON_TEST_FILENAME)

    # Make sure we get back the same skeleton we saved.
    assert(skeleton == skeleton_copy)


def test_hdf5(skeleton, stickman, tmpdir):
    filename = os.path.join(tmpdir, 'skeleton.h5')

    if os.path.isfile(filename):
        os.remove(filename)

    # Save both skeletons to the HDF5 file
    skeleton.save_hdf5(filename)
    stickman.save_hdf5(filename)

    # Load the all the skeletons as a list
    sk_list = Skeleton.load_all_hdf5(filename)

    # Lets check that they are equal to what we saved, this checks the order too.
    assert skeleton == sk_list[0]
    assert stickman == sk_list[1]

    # Check load to dict as well
    sk_dict = Skeleton.load_all_hdf5(filename, return_dict=True)
    assert skeleton == sk_dict[skeleton.name]
    assert stickman == sk_dict[stickman.name]

    # Check individual load
    assert Skeleton.load_hdf5(filename, skeleton.name) == skeleton
    assert Skeleton.load_hdf5(filename, stickman.name) == stickman

    # Check overwrite save and save list
    Skeleton.save_all_hdf5(filename, [skeleton, stickman])
    assert Skeleton.load_hdf5(filename, skeleton.name) == skeleton
    assert Skeleton.load_hdf5(filename, stickman.name) == stickman

    # Make sure we can't load a non-existent skeleton
    with pytest.raises(KeyError):
        Skeleton.load_hdf5(filename, 'BadName')

    # Make sure we can't save skeletons with the same name
    with pytest.raises(ValueError):
        Skeleton.save_all_hdf5(filename, [skeleton, Skeleton(name=skeleton.name)])


def test_name_change(skeleton):
    new_skeleton = Skeleton.rename_skeleton(skeleton, "New Fly")

    import networkx as nx

    def dict_match(dict1, dict2):
        return dict1 == dict2

    # Make sure the graphs are the same, not exact but clo
    assert nx.is_isomorphic(new_skeleton._graph, skeleton._graph, node_match=dict_match)

    # Make sure the skeletons are different (because of name)
    assert new_skeleton != skeleton

    # Make sure they hash different
    assert hash(new_skeleton) != hash(skeleton)

    # Make sure sets work
    assert len({new_skeleton, skeleton}) == 2

    # Make sure changing the name causues problems
    with pytest.raises(NotImplementedError):
        skeleton.name = "Test"

def test_graph_property(skeleton):
    assert [node for node in skeleton.graph.nodes()] == skeleton.node_names

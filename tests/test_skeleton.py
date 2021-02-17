import os
import copy

import pytest

from sleap.skeleton import Skeleton


def test_add_dupe_node(skeleton):
    """
    Test if adding a node with the same name to skeleton throws an exception.
    """
    with pytest.raises(ValueError):
        skeleton.add_node("head")


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
    assert skeleton["head"] is not None


def test_contains_node(skeleton):
    """
    Test whether __contains__ overload returns presence of nodes by name.
    """
    assert "head" in skeleton
    assert "not head" not in skeleton


def test_node_rename(skeleton):
    """
    Test if we can rename a node in place.
    """

    skeleton.relabel_nodes({"head": "new_head_name"})

    # Make sure the old "head" doesn't exist
    with pytest.raises(ValueError):
        skeleton["head"]

    # Make sure new head has the correct name
    assert skeleton["new_head_name"] is not None


def test_eq():
    s1 = Skeleton("s1")
    s1.add_nodes(["1", "2", "3", "4", "5", "6"])
    s1.add_edge("1", "2")
    s1.add_edge("3", "4")
    s1.add_edge("5", "6")
    s1.add_symmetry("3", "6")

    # Make a copy check that they are equal
    s2 = copy.deepcopy(s1)
    assert s1.matches(s2)

    # Add an edge, check that they are not equal
    s2 = copy.deepcopy(s1)
    s2.add_edge("5", "1")
    assert not s1.matches(s2)

    # Add a symmetry edge, not equal
    s2 = copy.deepcopy(s1)
    s2.add_symmetry("5", "1")
    assert not s1.matches(s2)

    # Delete a node
    s2 = copy.deepcopy(s1)
    s2.delete_node("5")
    assert not s1.matches(s2)

    # Delete and edge, not equal
    s2 = copy.deepcopy(s1)
    s2.delete_edge("1", "2")
    assert not s1.matches(s2)

    # FIXME: Probably shouldn't test it this way.
    # Add a value to a nodes dict, make sure they are not equal. This is touching the
    # internal _graph storage of the skeleton which probably should be avoided. But, I
    # wanted to test this just in case we add attributes to the nodes in the future.
    # UPDATE: Test is currently disabled.
    # Now that nodes are keyed to `Node`, we can't access by name.
    # Also, we can't directly check graph identity, since we want identity modulo `Node`
    # identity.
    # s2 = copy.deepcopy(s1)
    # s2._graph.nodes['1']['test'] = 5
    # assert s1 != s2


def test_symmetry():
    s1 = Skeleton("s1")
    s1.add_nodes(["1", "2", "3", "4", "5", "6"])
    s1.add_edge("1", "2")
    s1.add_edge("3", "4")
    s1.add_edge("5", "6")
    s1.add_symmetry("1", "5")
    s1.add_symmetry("3", "6")

    assert (s1.nodes[0], s1.nodes[4]) in s1.symmetries
    assert (s1.nodes[2], s1.nodes[5]) in s1.symmetries
    assert len(s1.symmetries) == 2

    assert (0, 4) in s1.symmetric_inds
    assert (2, 5) in s1.symmetric_inds
    assert len(s1.symmetric_inds) == 2

    assert s1.get_symmetry("1").name == "5"
    assert s1.get_symmetry("5").name == "1"

    assert s1.get_symmetry("3").name == "6"

    # Cannot add more than one symmetry to a node
    with pytest.raises(ValueError):
        s1.add_symmetry("1", "6")
    with pytest.raises(ValueError):
        s1.add_symmetry("6", "1")

    s1.delete_symmetry("1", "5")
    assert s1.get_symmetry("1") is None

    with pytest.raises(ValueError):
        s1.delete_symmetry("1", "5")


def test_json(skeleton, tmpdir):
    """
    Test saving and loading a Skeleton object in JSON.
    """
    JSON_TEST_FILENAME = os.path.join(tmpdir, "skeleton.json")

    # Save it to a JSON filename
    skeleton.save_json(JSON_TEST_FILENAME)

    # Load the JSON object back in
    skeleton_copy = Skeleton.load_json(JSON_TEST_FILENAME)

    # Make sure we get back the same skeleton we saved.
    assert skeleton.matches(skeleton_copy)


def test_hdf5(skeleton, stickman, tmpdir):
    filename = os.path.join(tmpdir, "skeleton.h5")

    if os.path.isfile(filename):
        os.remove(filename)

    # Save both skeletons to the HDF5 filename
    skeleton.save_hdf5(filename)
    stickman.save_hdf5(filename)

    # Load the all the skeletons as a list
    sk_list = Skeleton.load_all_hdf5(filename)

    # Lets check that they are equal to what we saved, this checks the order too.
    assert skeleton.matches(sk_list[0])
    assert stickman.matches(sk_list[1])

    # Check load to dict as well
    sk_dict = Skeleton.load_all_hdf5(filename, return_dict=True)
    assert skeleton.matches(sk_dict[skeleton.name])
    assert stickman.matches(sk_dict[stickman.name])

    # Check individual load
    assert Skeleton.load_hdf5(filename, skeleton.name).matches(skeleton)
    assert Skeleton.load_hdf5(filename, stickman.name).matches(stickman)

    # Check overwrite save and save list
    Skeleton.save_all_hdf5(filename, [skeleton, stickman])
    assert Skeleton.load_hdf5(filename, skeleton.name).matches(skeleton)
    assert Skeleton.load_hdf5(filename, stickman.name).matches(stickman)

    # Make sure we can't load a non-existent skeleton
    with pytest.raises(KeyError):
        Skeleton.load_hdf5(filename, "BadName")

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
    assert [node for node in skeleton.graph.nodes()] == skeleton.nodes


def test_load_mat_format():
    skeleton = Skeleton.load_mat(
        "tests/data/skeleton/leap_mat_format/skeleton_legs.mat"
    )

    # Check some stuff about the skeleton we loaded
    assert len(skeleton.nodes) == 24
    assert len(skeleton.edges) == 23

    # The node and edge list that should be present in skeleton_legs.mat
    node_names = [
        "head",
        "neck",
        "thorax",
        "abdomen",
        "wingL",
        "wingR",
        "forelegL1",
        "forelegL2",
        "forelegL3",
        "forelegR1",
        "forelegR2",
        "forelegR3",
        "midlegL1",
        "midlegL2",
        "midlegL3",
        "midlegR1",
        "midlegR2",
        "midlegR3",
        "hindlegL1",
        "hindlegL2",
        "hindlegL3",
        "hindlegR1",
        "hindlegR2",
        "hindlegR3",
    ]

    edges = [
        [2, 1],
        [1, 0],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 15],
        [15, 16],
        [16, 17],
        [2, 18],
        [18, 19],
        [19, 20],
        [2, 21],
        [21, 22],
        [22, 23],
    ]

    assert [n.name for n in skeleton.nodes] == node_names

    # Check the edges and their order
    for i, edge in enumerate(skeleton.edge_names):
        assert tuple(edges[i]) == (
            skeleton.node_to_index(edge[0]),
            skeleton.node_to_index(edge[1]),
        )


def test_arborescence():
    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # linear: a -> b -> c
    skeleton.add_edge("a", "b")
    skeleton.add_edge("b", "c")

    assert skeleton.is_arborescence

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # two branches from a: a -> b and a -> c
    skeleton.add_edge("a", "b")
    skeleton.add_edge("a", "c")

    assert skeleton.is_arborescence

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # no edges so too many roots
    assert not skeleton.is_arborescence
    assert sorted((n.name for n in skeleton.root_nodes)) == ["a", "b", "c"]

    # still too many roots: a and c
    skeleton.add_edge("a", "b")

    assert not skeleton.is_arborescence
    assert sorted((n.name for n in skeleton.root_nodes)) == ["a", "c"]

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # cycle
    skeleton.add_edge("a", "b")
    skeleton.add_edge("b", "c")
    skeleton.add_edge("c", "a")

    assert not skeleton.is_arborescence
    assert len(skeleton.cycles) == 1
    assert len(skeleton.root_nodes) == 0

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")
    skeleton.add_node("d")

    # diamond, too many sources leading to d
    skeleton.add_edge("a", "b")
    skeleton.add_edge("a", "c")
    skeleton.add_edge("b", "d")
    skeleton.add_edge("c", "d")

    assert not skeleton.is_arborescence
    assert len(skeleton.cycles) == 0
    assert len(skeleton.root_nodes) == 1
    assert len(skeleton.in_degree_over_one) == 1

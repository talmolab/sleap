from sleap_io.model.skeleton import Skeleton
from sleap.sleap_io_adaptors.skeleton_utils import (
    is_arborescence,
    cycles,
    in_degree_over_one,
    root_nodes,
)


def test_arborescence():
    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # linear: a -> b -> c
    skeleton.add_edge("a", "b")
    skeleton.add_edge("b", "c")

    assert is_arborescence(skeleton)

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # two branches from a: a -> b and a -> c
    skeleton.add_edge("a", "b")
    skeleton.add_edge("a", "c")

    assert is_arborescence(skeleton)

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # no edges so too many roots
    assert not is_arborescence(skeleton)

    # still too many roots: a and c
    skeleton.add_edge("a", "b")

    assert not is_arborescence(skeleton)

    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")

    # cycle
    skeleton.add_edge("a", "b")
    skeleton.add_edge("b", "c")
    skeleton.add_edge("c", "a")

    assert not is_arborescence(skeleton)
    assert len(cycles(skeleton)) == 1
    assert len(root_nodes(skeleton)) == 0

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

    assert not is_arborescence(skeleton)
    assert len(cycles(skeleton)) == 0
    assert len(root_nodes(skeleton)) == 1
    assert len(in_degree_over_one(skeleton)) == 1

    # symmetry edges should be ignored
    skeleton = Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    skeleton.add_node("c")
    skeleton.add_edge("a", "b")
    skeleton.add_edge("b", "c")
    skeleton.add_symmetry("a", "c")
    assert is_arborescence(skeleton)

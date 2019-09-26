import pytest

from sleap.skeleton import Skeleton


@pytest.fixture
def stickman():

    # Make a skeleton with a space in its name to test things.
    stickman = Skeleton("Stick man")
    stickman.add_nodes(
        ["head", "neck", "body", "right-arm", "left-arm", "right-leg", "left-leg"]
    )
    stickman.add_edge("neck", "head")
    stickman.add_edge("body", "neck")
    stickman.add_edge("body", "right-arm")
    stickman.add_edge("body", "left-arm")
    stickman.add_edge("body", "right-leg")
    stickman.add_edge("body", "left-leg")
    stickman.add_symmetry(node1="left-arm", node2="right-arm")
    stickman.add_symmetry(node1="left-leg", node2="right-leg")

    return stickman


@pytest.fixture
def skeleton():

    # Create a simple skeleton object
    skeleton = Skeleton("Fly")
    skeleton.add_node("head")
    skeleton.add_node("thorax")
    skeleton.add_node("abdomen")
    skeleton.add_node("left-wing")
    skeleton.add_node("right-wing")
    skeleton.add_edge(source="head", destination="thorax")
    skeleton.add_edge(source="thorax", destination="abdomen")
    skeleton.add_edge(source="thorax", destination="left-wing")
    skeleton.add_edge(source="thorax", destination="right-wing")
    skeleton.add_symmetry(node1="left-wing", node2="right-wing")

    return skeleton

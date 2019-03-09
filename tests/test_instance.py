import pytest
import math
import numpy as np

from sleap.instance import Instance, Point
from sleap.io.video import Video
from  sleap.skeleton import Skeleton

@pytest.fixture
def fly_skeleton():

    # Create a simple skeleton object
    skeleton = Skeleton("Fly")
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

def test_instance_node_get_set_item(fly_skeleton):
    """
    Test basic get item and set item functionality of instances.
    """
    instance = Instance(skeleton=fly_skeleton, video=None, frame_idx=0)
    instance["head"].x = 20
    instance["head"].y = 50

    instance["left-wing"] = Point(x=30, y=40, visible=False)

    assert instance["head"].x == 20
    assert instance["head"].y == 50

    assert instance["left-wing"] == Point(x=30, y=40, visible=False)

    thorax_point = instance["thorax"]
    assert math.isnan(thorax_point.x) and math.isnan(thorax_point.y)


def test_instance_node_multi_get_set_item(fly_skeleton):
    """
    Test basic get item and set item functionality of instances.
    """
    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3,6)}

    instance1 = Instance(skeleton=fly_skeleton, video=None, frame_idx=0, points=points)
    instance2 = Instance(skeleton=fly_skeleton, video=None, frame_idx=0)

    instance1[node_names] = list(points.values())

    x_values = [p.x for p in instance1[node_names]]
    y_values = [p.y for p in instance1[node_names]]

    assert np.allclose(x_values, [1, 2, 3])
    assert np.allclose(y_values, [4, 5, 6])

def test_non_exist_node(fly_skeleton):
    """
    Test is instances throw key errors for nodes that don't exist in the skeleton.
    """
    instance = Instance(skeleton=fly_skeleton, video=None, frame_idx=0)

    with pytest.raises(KeyError):
        instance["non-existent-node"].x = 1

    with pytest.raises(KeyError):
        instance = Instance(skeleton=fly_skeleton, video=None, frame_idx=0, points = {"non-exist": Point()})


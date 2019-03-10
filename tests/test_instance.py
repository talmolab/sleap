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

def test_instance_point_iter(fly_skeleton):
    """
    Test iteration methods over instances.
    """
    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3, 6)}

    instance = Instance(skeleton=fly_skeleton, video=None, frame_idx=0, points=points)

    assert [node for node in instance.nodes()] == ['head', 'left-wing', 'right-wing']
    assert np.allclose([p.x for p in instance.points()], [1, 2, 3])
    assert np.allclose([p.y for p in instance.points()], [4, 5, 6])

    # Make sure we can iterate over tuples
    for (node, point) in instance.nodes_points():
        assert points[node] == point


def test_instance_to_pandas_df(fly_skeleton):
    """
    Test generating pandas DataFrames from lists of instances.
    """

    # Generate some instances
    NUM_INSTANCES = 500
    NUM_COLS = 8

    instances = []
    for i in range(NUM_INSTANCES):
        instance = Instance(skeleton=fly_skeleton, video=None, frame_idx=i)
        instance['head'] = Point(i*1, i*2)
        instance['left-wing'] = Point(10 + i * 1, 10 + i * 2)
        instance['right-wing'] = Point(20 + i * 1, 20 + i * 2)

        # Lets make an NaN entry to test skip_nan as well
        instance['thorax']

        instances.append(instance)

    df = Instance.to_pandas_df(instances)

    # Check to make sure we got the expected shape
    assert df.shape == (3*NUM_INSTANCES, NUM_COLS)

    # Check skip_nan is working
    assert Instance.to_pandas_df(instances, skip_nan=False).shape == (4*NUM_INSTANCES, NUM_COLS)

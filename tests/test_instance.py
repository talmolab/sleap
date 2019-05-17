import os
import math
import copy

import pytest
import numpy as np

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame

def test_instance_node_get_set_item(skeleton):
    """
    Test basic get item and set item functionality of instances.
    """
    instance = Instance(skeleton=skeleton)
    instance["head"].x = 20
    instance["head"].y = 50

    instance["left-wing"] = Point(x=30, y=40, visible=False)

    assert instance["head"].x == 20
    assert instance["head"].y == 50

    assert instance["left-wing"] == Point(x=30, y=40, visible=False)

    thorax_point = instance["thorax"]
    assert math.isnan(thorax_point.x) and math.isnan(thorax_point.y)


def test_instance_node_multi_get_set_item(skeleton):
    """
    Test basic get item and set item functionality of instances.
    """
    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3,6)}

    instance1 = Instance(skeleton=skeleton, points=points)

    instance1[node_names] = list(points.values())

    x_values = [p.x for p in instance1[node_names]]
    y_values = [p.y for p in instance1[node_names]]

    assert np.allclose(x_values, [1, 2, 3])
    assert np.allclose(y_values, [4, 5, 6])


def test_non_exist_node(skeleton):
    """
    Test is instances throw key errors for nodes that don't exist in the skeleton.
    """
    instance = Instance(skeleton=skeleton)

    with pytest.raises(KeyError):
        instance["non-existent-node"].x = 1

    with pytest.raises(KeyError):
        instance = Instance(skeleton=skeleton, points = {"non-exist": Point()})


def test_instance_point_iter(skeleton):
    """
    Test iteration methods over instances.
    """
    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3, 6)}

    instance = Instance(skeleton=skeleton, points=points)

    assert [node.name for node in instance.nodes] == ['head', 'left-wing', 'right-wing']
    assert np.allclose([p.x for p in instance.points()], [1, 2, 3])
    assert np.allclose([p.y for p in instance.points()], [4, 5, 6])

    # Make sure we can iterate over tuples
    for (node, point) in instance.nodes_points:
        assert points[node.name] == point


def test_instance_to_pandas_df(skeleton, instances):
    """
    Test generating pandas DataFrames from lists of instances.
    """

    # How many columns are supposed to be in point DataFrame
    NUM_COLS = 9

    NUM_INSTANCES = len(instances)

    df = Instance.to_pandas_df(instances)

    # Check to make sure we got the expected shape
    assert df.shape == (3*NUM_INSTANCES, NUM_COLS)

    # Check skip_nan is working
    assert Instance.to_pandas_df(instances, skip_nan=False).shape == (4*NUM_INSTANCES, NUM_COLS)

# Skip HDF5 saving of instances now because tracks are not saved properly
@pytest.mark.skip
def test_hdf5(instances, tmpdir):
    out_dir = tmpdir
    path = os.path.join(out_dir, 'dataset.h5')

    if os.path.isfile(path):
        os.remove(path)

    Instance.save_hdf5(file=path, instances=instances)

    assert os.path.isfile(path)

    # Make a deep copy, because we are gonna drop nan points in place.
    # and I don't want to change the fixture.
    instances_copy = copy.deepcopy(instances)

    # Drop the NaN points
    Instance.drop_all_nan_points(instances_copy)

    # Make sure we can overwrite
    Instance.save_hdf5(file=path, instances=instances_copy[0:100], skip_nan=False)

    # Load the data back
    instances2 = Instance.load_hdf5(file=path)

    # Check that we get back the same instances
    for i in range(len(instances2)):
        assert instances_copy[i].matches(instances2[i])


def test_skeleton_node_name_change():
    """
    Test that and instance is not broken after a node on the
    skeleton has its name changed.
    """

    s = Skeleton("Test")
    s.add_nodes(['a', 'b', 'c', 'd', 'e'])
    s.add_edge('a', 'b')

    instance = Instance(s)
    instance['a'] = Point(1,2)
    instance['b'] = Point(3,4)

    # Rename the node
    s.relabel_nodes({'a': 'A'})

    # Reference to the old node name should raise a KeyError
    with pytest.raises(KeyError):
        instance['a'].x = 2

    # Make sure the A now references the same point on the instance
    assert instance['A'] == Point(1, 2)
    assert instance['b'] == Point(3, 4)

def test_instance_comparison(skeleton):

    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3,6)}

    instance1 = Instance(skeleton=skeleton, points=points)
    instance2 = copy.deepcopy(instance1)

    assert instance1.matches(instance1)

    assert instance1 != instance2

    assert instance1.matches(instance2)

    instance2["head"].x = 42
    assert not instance1.matches(instance2)

    instance2 = copy.deepcopy(instance1)
    instance2.skeleton.add_node('extra_node')
    assert not instance1.matches(instance2)

def test_points_array(skeleton):
    """ Test conversion of instances to points array"""

    node_names = ["left-wing", "head", "right-wing"]
    points = {"head": Point(1, 4), "left-wing": Point(2, 5), "right-wing": Point(3, 6)}

    instance1 = Instance(skeleton=skeleton, points=points)

    pts = instance1.points_array()

    assert pts.shape == (len(skeleton.nodes), 2)
    assert np.allclose(pts[skeleton.node_to_index('left-wing'), :], [2, 5])
    assert np.allclose(pts[skeleton.node_to_index('head'), :], [1, 4])
    assert np.allclose(pts[skeleton.node_to_index('right-wing'), :], [3, 6])
    assert np.isnan(pts[skeleton.node_to_index('thorax'), :]).all()

    # Now change a point, make sure it is reflected
    instance1['head'].x = 0
    instance1['thorax'] = Point(1, 2)
    pts = instance1.points_array()
    assert np.allclose(pts[skeleton.node_to_index('head'), :], [0, 4])
    assert np.allclose(pts[skeleton.node_to_index('thorax'), :], [1, 2])

    # Now use the cached version and make sure changes are not
    # reflected
    pts = instance1.points_array(cached=True)
    assert np.allclose(pts[skeleton.node_to_index('thorax'), :], [1, 2])
    instance1['thorax'] = Point(1, 6)
    pts = instance1.points_array(cached=True)
    assert np.allclose(pts[skeleton.node_to_index('thorax'), :], [1, 2])

    # Now drop the cache and make sure changes are reflected.
    pts = instance1.points_array()
    assert np.allclose(pts[skeleton.node_to_index('thorax'), :], [1, 6])

def test_instance_labeled_frame_ref(skeleton, centered_pair_vid):
    """
    Test whether links between labeled frames and instances are kept
    """
    instances = [Instance(skeleton=skeleton) for i in range(3)]

    frame = LabeledFrame(video=centered_pair_vid, frame_idx=0, instances=instances)

    assert frame.instances[0].frame == frame
    assert frame[0].frame == frame
    assert frame[0].frame_idx == 0


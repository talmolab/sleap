import os
import pytest
import numpy as np

from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame, PredictedInstance
from sleap.io.video import Video, MediaVideo
from sleap.io.dataset import Labels, load_labels_json_old
from sleap.gui.suggestions import VideoFrameSuggestions

TEST_H5_DATASET = 'tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5'

def _check_labels_match(expected_labels, other_labels, format = 'png'):
    """
    A utitlity function to check whether to sets of labels match.
    This doesn't directly compares some things (like video objects).

    Args:
        expected_labels: The expected labels
        other_labels: The labels to check against expected

    Returns:
        True for match, False otherwise.
    """

    # Check the top level objects
    for x, y in zip(expected_labels.skeletons, other_labels.skeletons):

        # Inline the skeleton matches check to see if we can get a better
        # idea of why this test fails non-deterministically. The callstack
        # doesn't go deeper than the method call in pytest for some reason.
        # assert x.matches(y). The code below is weird because it is converted
        # from Skeleton.__eq__.
        self = x
        other = y

        # First check names, duh!
        if other.name != self.name:
            assert False

        def dict_match(dict1, dict2):
            return dict1 == dict2

        # Check if the graphs are iso-morphic
        import networkx as nx
        is_isomorphic = nx.is_isomorphic(self._graph, other._graph, node_match=dict_match)

        if not is_isomorphic:
            assert False

        # Now check that the nodes have the same labels and order. They can have
        # different weights I guess?!
        for node1, node2 in zip(self._graph.nodes, other._graph.nodes):
            if node1.name != node2.name:
                assert False

    for x, y in zip(expected_labels.tracks, other_labels.tracks):
        assert x.name == y.name and x.spawned_on == y.spawned_on

    # Check that we have the same thing
    for expected_label, label in zip(expected_labels.labels, other_labels.labels):

        assert expected_label.frame_idx == label.frame_idx

        frame_idx = label.frame_idx

        frame_data = label.video.get_frame(frame_idx)[0:15, 0:15, :]
        expected_frame_data = expected_label.video.get_frame(frame_idx)[0:15, 0:15, :]

        # Compare the first frames of the videos, do it on a small sub-region to
        # make the test reasonable in time.
        if format is 'png':
            assert np.allclose(frame_data, expected_frame_data)

        # Compare the instances
        assert all(i1.matches(i2) for (i1, i2) in zip(expected_label.instances, label.instances))

        # This test takes to long, break after 20 or so.
        if frame_idx > 20:
            break


def test_labels_json(tmpdir, multi_skel_vid_labels):
    json_file_path = os.path.join(tmpdir, 'dataset.json')

    if os.path.isfile(json_file_path):
        os.remove(json_file_path)

    # Save to json
    Labels.save_json(labels=multi_skel_vid_labels, filename=json_file_path)

    # Make sure the filename is there
    assert os.path.isfile(json_file_path)

    # Lets load the labels back in and make sure we haven't lost anything.
    loaded_labels = Labels.load_json(json_file_path)

    # Check that we have the same thing
    _check_labels_match(multi_skel_vid_labels, loaded_labels)

    # Check that we don't have the very same objects
    assert not multi_skel_vid_labels.skeletons[0] is loaded_labels.skeletons[0]
    assert not multi_skel_vid_labels.nodes[3] in loaded_labels.nodes
    assert not multi_skel_vid_labels.videos[0] is loaded_labels.videos[0]

    # Reload json using objects from original labels
    # We'll also test load_file() here
    loaded_labels = Labels.load_file(json_file_path, match_to=multi_skel_vid_labels)

    # Check that we now do have the same objects
    assert multi_skel_vid_labels.skeletons[0] in loaded_labels.skeletons
    assert multi_skel_vid_labels.nodes[3] in loaded_labels.nodes
    assert multi_skel_vid_labels.videos[0] in loaded_labels.videos

def test_load_labels_json_old(tmpdir):
    new_file_path = os.path.join(tmpdir, 'centered_pair_v2.json')

    # Function to run some checks on loaded labels
    def check_labels(labels):
        skel_node_names = ['head', 'neck', 'thorax', 'abdomen', 'wingL',
                           'wingR', 'forelegL1', 'forelegL2', 'forelegL3',
                           'forelegR1', 'forelegR2', 'forelegR3', 'midlegL1',
                           'midlegL2', 'midlegL3', 'midlegR1', 'midlegR2',
                           'midlegR3', 'hindlegL1', 'hindlegL2', 'hindlegL3',
                           'hindlegR1', 'hindlegR2', 'hindlegR3']

        # Do some basic checks
        assert len(labels) == 70

        # Make sure we only create one video object and it works
        assert len({label.video for label in labels}) == 1
        assert labels[0].video.get_frame(0).shape == (384, 384, 1)

        # Check some frame objects.
        assert labels[0].frame_idx == 0
        assert labels[40].frame_idx == 573

        # Check the skeleton
        assert labels[0].instances[0].skeleton.node_names == skel_node_names

    labels = Labels.load_json("tests/data/json_format_v1/centered_pair.json")
    check_labels(labels)

    # Save out to new JSON format
    Labels.save_json(labels, new_file_path)

    # Reload and check again.
    new_labels = Labels.load_json(new_file_path)
    check_labels(new_labels)


def test_label_accessors(centered_pair_labels):
    labels = centered_pair_labels

    video = labels.videos[0]
    assert len(labels.find(video)) == 70
    assert labels[video] == labels.find(video)

    f = labels.frames(video, from_frame_idx=1)
    assert next(f).frame_idx == 15
    assert next(f).frame_idx == 31

    f = labels.frames(video, from_frame_idx=31, reverse=True)
    assert next(f).frame_idx == 15

    f = labels.frames(video, from_frame_idx=0, reverse=True)
    assert next(f).frame_idx == 1092
    next(f)
    next(f)
    # test that iterator now has fewer items left
    assert len(list(f)) == 70-3

    assert labels.instance_count(video, 15) == 2
    assert labels.instance_count(video, 7) == 0

    assert labels[0].video == video
    assert labels[0].frame_idx == 0

    assert labels[61].video == video
    assert labels[61].frame_idx == 954

    assert len(labels.find(video, frame_idx=954)) == 1
    assert len(labels.find(video, 954)) == 1
    assert labels.find(video, 954)[0] == labels[61]
    assert labels.find_first(video) == labels[0]
    assert labels.find_first(video, 954) == labels[61]
    assert labels.find_last(video) == labels[69]
    assert labels[video, 954] == labels[61]
    assert labels[video, 0] == labels[0]
    assert labels[video] == labels.labels

    assert len(labels.find(video, 101)) == 0
    assert labels.find_first(video, 101) is None
    with pytest.raises(KeyError):
        labels[video, 101]

    dummy_video = Video(backend=MediaVideo)
    assert len(labels.find(dummy_video)) == 0
    with pytest.raises(KeyError):
        labels[dummy_video]


def test_label_mutability():
    dummy_video = Video(backend=MediaVideo)
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance,])

    labels = Labels()
    labels.append(dummy_frame)

    assert dummy_video in labels.videos
    assert dummy_video in labels
    assert dummy_skeleton in labels.skeletons
    assert dummy_skeleton in labels
    assert dummy_frame in labels.labeled_frames
    assert dummy_frame in labels
    assert (dummy_video, 0) in labels
    assert (dummy_video, 1) not in labels

    dummy_video2 = Video(backend=MediaVideo)
    dummy_skeleton2 = Skeleton(name="dummy2")
    dummy_instance2 = Instance(dummy_skeleton2)
    dummy_frame2 = LabeledFrame(dummy_video2, frame_idx=0, instances=[dummy_instance2,])
    assert dummy_video2 not in labels
    assert dummy_skeleton2 not in labels
    assert dummy_frame2 not in labels

    labels.append(dummy_frame2)
    assert dummy_video2 in labels
    assert dummy_frame2 in labels

    labels.remove_video(dummy_video2)
    assert dummy_video2 not in labels
    assert dummy_frame2 not in labels
    assert len(labels.find(dummy_video2)) == 0

    assert len(labels) == 1
    labels.append(LabeledFrame(dummy_video, frame_idx=0))
    assert len(labels) == 1

    dummy_frames = [LabeledFrame(dummy_video, frame_idx=i) for i in range(10)]
    dummy_frames2 = [LabeledFrame(dummy_video2, frame_idx=i) for i in range(10)]

    for f in dummy_frames + dummy_frames2:
        labels.append(f)

    assert(len(labels) == 20)
    labels.remove_video(dummy_video2)
    assert(len(labels) == 10)

    assert len(labels.find(dummy_video)) == 10
    assert dummy_frame in labels
    assert all([label in labels for label in dummy_frames[1:]])

    assert dummy_video2 not in labels
    assert len(labels.find(dummy_video2)) == 0
    assert all([label not in labels for label in dummy_frames2])

    labels.remove_video(dummy_video)
    assert len(labels.find(dummy_video)) == 0

    dummy_frames3 = [LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance,]) for _ in range(10)]
    labels.labeled_frames.extend(dummy_frames3)
    assert len(labels) == 10
    assert len(labels.labeled_frames[0].instances) == 1
    labels.merge_matching_frames()
    assert len(labels) == 1
    assert len(labels.labeled_frames[0].instances) == 10

def skeleton_ids_from_label_instances(labels):
    return list(map(id, (lf.instances[0].skeleton for lf in labels.labeled_frames)))

def test_duplicate_skeletons_serializing():
    vid = Video.from_filename("foo.mp4")

    skeleton_a = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    lf_a = LabeledFrame(vid, frame_idx=2, instances=[Instance(skeleton_a)])
    lf_b = LabeledFrame(vid, frame_idx=3, instances=[Instance(skeleton_b)])

    new_labels = Labels(labeled_frames=[lf_a, lf_b])
    new_labels_json = new_labels.to_dict()

def test_distinct_skeletons_serializing():
    vid = Video.from_filename("foo.mp4")

    skeleton_a = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b.add_node("foo")

    lf_a = LabeledFrame(vid, frame_idx=2, instances=[Instance(skeleton_a)])
    lf_b = LabeledFrame(vid, frame_idx=3, instances=[Instance(skeleton_b)])

    new_labels = Labels(labeled_frames=[lf_a, lf_b])

    # Make sure we can serialize this
    new_labels_json = new_labels.to_dict()

def test_unify_skeletons():
    vid = Video.from_filename("foo.mp4")

    skeleton_a = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    lf_a = LabeledFrame(vid, frame_idx=2, instances=[Instance(skeleton_a)])
    lf_b = LabeledFrame(vid, frame_idx=3, instances=[Instance(skeleton_b)])

    labels = Labels()
    labels.extend_from([lf_a], unify=True)
    labels.extend_from([lf_b], unify=True)
    ids = skeleton_ids_from_label_instances(labels)

    # Make sure that skeleton_b got replaced with skeleton_a when we
    # added the frame with "unify" set
    assert len(set(ids)) == 1

    # Make sure we can serialize this
    labels.to_dict()

def test_dont_unify_skeletons():
    vid = Video.from_filename("foo.mp4")

    skeleton_a = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")
    skeleton_b = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    lf_a = LabeledFrame(vid, frame_idx=2, instances=[Instance(skeleton_a)])
    lf_b = LabeledFrame(vid, frame_idx=3, instances=[Instance(skeleton_b)])

    labels = Labels(labeled_frames=[lf_a])
    labels.extend_from([lf_b], unify=False)
    ids = skeleton_ids_from_label_instances(labels)

    # Make sure we still have two distinct skeleton objects
    assert len(set(ids)) == 2

    # Make sure we can serialize this
    labels.to_dict()

def test_instance_access():
    labels = Labels()

    dummy_skeleton = Skeleton()
    dummy_video = Video(backend=MediaVideo)
    dummy_video2 = Video(backend=MediaVideo)

    for i in range(10):
        labels.append(LabeledFrame(dummy_video, frame_idx=i, instances=[Instance(dummy_skeleton), Instance(dummy_skeleton)]))
    for i in range(10):
        labels.append(LabeledFrame(dummy_video2, frame_idx=i, instances=[Instance(dummy_skeleton), Instance(dummy_skeleton), Instance(dummy_skeleton)]))

    assert len(labels.all_instances) == 50
    assert len(list(labels.instances(video=dummy_video))) == 20
    assert len(list(labels.instances(video=dummy_video2))) == 30

def test_suggestions(small_robot_mp4_vid):
    dummy_video = small_robot_mp4_vid
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance,])

    labels = Labels()
    labels.append(dummy_frame)

    suggestions = dict()
    suggestions[dummy_video] = VideoFrameSuggestions.suggest(
                                dummy_video,
                                params=dict(method="random", per_video=13))
    labels.set_suggestions(suggestions)

    assert len(labels.get_video_suggestions(dummy_video)) == 13


def test_load_labels_mat(mat_labels):
    assert len(mat_labels.nodes) == 6
    assert len(mat_labels) == 43


@pytest.mark.parametrize("format", ['png', 'mjpeg/avi'])
def test_save_labels_with_frame_data(multi_skel_vid_labels, tmpdir, format):
    """
    Test saving and loading a labels dataset with frame data included
    as JSON.
    """

    # Lets take a subset of the labels so this doesn't take too long
    multi_skel_vid_labels.labeled_frames = multi_skel_vid_labels.labeled_frames[5:30]

    filename = os.path.join(tmpdir, 'test.json')
    Labels.save_json(multi_skel_vid_labels, filename=filename, save_frame_data=True, frame_data_format=format)

    # Load the data back in
    loaded_labels = Labels.load_json(f"{filename}.zip")

    # Check that we have the same thing
    _check_labels_match(multi_skel_vid_labels, loaded_labels, format=format)

    # Make sure we can load twice
    loaded_labels = Labels.load_json(f"{filename}.zip")


def test_labels_hdf5(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, 'test.h5')

    Labels.save_hdf5(filename=filename, labels=labels)

    loaded_labels = Labels.load_hdf5(filename=filename)

    _check_labels_match(labels, loaded_labels)


def test_labels_predicted_hdf5(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, 'test.h5')

    # Lets promote some of these Instances to predicted instances
    for label in labels:
        for i, instance in enumerate(label.instances):
            if i % 2 == 0:
                label.instances[i] = PredictedInstance.from_instance(instance, 0.3)

    # Lets also add some from_predicted values
    for label in labels:
        label.instances[1].from_predicted = label.instances[0]

    # Try adding a node to the skeleton
    labels.skeletons[0].add_node("new node")

    # Save and compare the results
    Labels.save_hdf5(filename=filename, labels=labels)
    loaded_labels = Labels.load_hdf5(filename=filename)
    _check_labels_match(labels, loaded_labels)

    # Try deleting nodes from the skeleton
    node = labels.skeletons[0].nodes[-1]
    labels.skeletons[0].delete_node(node)
    node = labels.skeletons[0].nodes[-1]
    labels.skeletons[0].delete_node(node)

    # Save and compare the results
    Labels.save_hdf5(filename=filename, labels=labels)
    loaded_labels = Labels.load_hdf5(filename=filename)
    _check_labels_match(labels, loaded_labels)

def test_labels_append_hdf5(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, 'test.h5')

    # Save each frame of the Labels dataset one by one in append
    # mode
    for label in labels:

        # Just do the first 20 to speed things up
        if label.frame_idx > 20:
            break

        Labels.save_hdf5(filename=filename, labels=Labels([label]), append=True)

    # Now load the dataset and make sure we get the same thing we started
    # with.
    loaded_labels = Labels.load_hdf5(filename=filename)

    _check_labels_match(labels, loaded_labels)


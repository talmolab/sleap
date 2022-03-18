import os
import pytest
import numpy as np
from pathlib import Path

import sleap
from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame, PredictedInstance, Track
from sleap.io.video import Video, MediaVideo
from sleap.io.dataset import Labels, load_file
from sleap.io.legacy import load_labels_json_old
from sleap.gui.suggestions import VideoFrameSuggestions, SuggestionFrame

TEST_H5_DATASET = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"


def _check_labels_match(expected_labels, other_labels, format="png"):
    """
    A utility function to check whether to sets of labels match.
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

        is_isomorphic = nx.is_isomorphic(
            self._graph, other._graph, node_match=dict_match
        )

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
        if format is "png":
            assert np.allclose(frame_data, expected_frame_data)

        # Compare the instances
        assert all(
            i1.matches(i2)
            for (i1, i2) in zip(expected_label.instances, label.instances)
        )

        # This test takes to long, break after 20 or so.
        if frame_idx > 20:
            break


def test_labels_json(tmpdir, multi_skel_vid_labels):
    json_file_path = os.path.join(tmpdir, "dataset.json")

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
    new_file_path = os.path.join(tmpdir, "centered_pair_v2.json")

    # Function to run some checks on loaded labels
    def check_labels(labels):
        skel_node_names = [
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
    assert len(list(f)) == 70 - 3

    assert labels.instance_count(video, 15) == 2
    assert labels.instance_count(video, 7) == 0

    assert labels[0].video == video
    assert labels[0].frame_idx == 0

    assert labels[61].video == video
    assert labels[61].frame_idx == 954

    assert labels[np.int64(0)] == labels[0]
    assert labels[np.int64(61)] == labels[61]
    assert labels[np.array([0, 61])] == labels[[0, 61]]

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


def test_scalar_properties():
    # Scalar
    dummy_video = Video(backend=MediaVideo)
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance])

    labels = Labels()
    labels.append(dummy_frame)

    assert labels.video == dummy_video
    assert labels.skeleton == dummy_skeleton

    # Empty
    labels = Labels()
    with pytest.raises(ValueError):
        labels.video
    with pytest.raises(ValueError):
        labels.skeleton

    # More than one video
    dummy_skeleton = Skeleton()
    labels = Labels()
    labels.append(
        LabeledFrame(
            Video(backend=MediaVideo), frame_idx=0, instances=[Instance(dummy_skeleton)]
        )
    )
    labels.append(
        LabeledFrame(
            Video(backend=MediaVideo), frame_idx=0, instances=[Instance(dummy_skeleton)]
        )
    )
    assert labels.skeleton == dummy_skeleton
    with pytest.raises(ValueError):
        labels.video

    # More than one skeleton
    dummy_video = Video(backend=MediaVideo)
    labels = Labels()
    labels.append(
        LabeledFrame(dummy_video, frame_idx=0, instances=[Instance(Skeleton())])
    )
    labels.append(
        LabeledFrame(dummy_video, frame_idx=1, instances=[Instance(Skeleton())])
    )
    assert labels.video == dummy_video
    with pytest.raises(ValueError):
        labels.skeleton


def test_has_missing_videos():
    labels = Labels()
    labels.add_video(Video.from_filename("small_robot.mp4"))
    assert labels.has_missing_videos

    labels = Labels()
    labels.add_video(Video.from_filename("tests/data/videos/small_robot.mp4"))
    assert not labels.has_missing_videos


def test_label_mutability():
    dummy_video = Video(backend=MediaVideo)
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance])

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
    dummy_frame2 = LabeledFrame(dummy_video2, frame_idx=0, instances=[dummy_instance2])
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

    assert len(labels) == 20
    labels.remove_video(dummy_video2)
    assert len(labels) == 10

    assert len(labels.find(dummy_video)) == 10
    assert dummy_frame in labels
    assert all([label in labels for label in dummy_frames[1:]])

    assert dummy_video2 not in labels
    assert len(labels.find(dummy_video2)) == 0
    assert all([label not in labels for label in dummy_frames2])

    labels.remove_video(dummy_video)
    assert len(labels.find(dummy_video)) == 0


def test_labels_merge():
    dummy_video = Video(backend=MediaVideo)
    dummy_skeleton = Skeleton()
    dummy_skeleton.add_node("node")

    labels = Labels()
    dummy_frames = []

    # Add 10 instances with different points (so they aren't "redundant")
    for i in range(10):
        instance = Instance(skeleton=dummy_skeleton, points=dict(node=Point(i, i)))
        dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[instance])
        dummy_frames.append(dummy_frame)

    labels.labeled_frames.extend(dummy_frames)
    assert len(labels) == 10
    assert len(labels.labeled_frames[0].instances) == 1

    labels.merge_matching_frames()
    assert len(labels) == 1
    assert len(labels.labeled_frames[0].instances) == 10


def test_complex_merge():
    dummy_video_a = Video.from_filename("foo.mp4")
    dummy_video_b = Video.from_filename("foo.mp4")

    dummy_skeleton_a = Skeleton()
    dummy_skeleton_a.add_node("node")

    dummy_skeleton_b = Skeleton()
    dummy_skeleton_b.add_node("node")

    dummy_instances_a = []
    dummy_instances_a.append(
        Instance(skeleton=dummy_skeleton_a, points=dict(node=Point(1, 1)))
    )
    dummy_instances_a.append(
        Instance(skeleton=dummy_skeleton_a, points=dict(node=Point(2, 2)))
    )

    labels_a = Labels()
    labels_a.append(
        LabeledFrame(dummy_video_a, frame_idx=0, instances=dummy_instances_a)
    )

    dummy_instances_b = []
    dummy_instances_b.append(
        Instance(skeleton=dummy_skeleton_b, points=dict(node=Point(1, 1)))
    )
    dummy_instances_b.append(
        Instance(skeleton=dummy_skeleton_b, points=dict(node=Point(3, 3)))
    )

    labels_b = Labels()
    labels_b.append(
        LabeledFrame(dummy_video_b, frame_idx=0, instances=dummy_instances_b)
    )  # conflict
    labels_b.append(
        LabeledFrame(dummy_video_b, frame_idx=1, instances=dummy_instances_b)
    )  # clean

    merged, extra_a, extra_b = Labels.complex_merge_between(labels_a, labels_b)

    # Check that we have the cleanly merged frame
    assert dummy_video_a in merged
    assert len(merged[dummy_video_a]) == 1  # one merged frame
    assert len(merged[dummy_video_a][1]) == 2  # with two instances

    # Check that labels_a includes redundant and clean
    assert len(labels_a.labeled_frames) == 2
    assert len(labels_a.labeled_frames[0].instances) == 1
    assert labels_a.labeled_frames[0].instances[0].points[0].x == 1
    assert len(labels_a.labeled_frames[1].instances) == 2
    assert labels_a.labeled_frames[1].instances[0].points[0].x == 1
    assert labels_a.labeled_frames[1].instances[1].points[0].x == 3

    # Check that extra_a/b includes the appropriate conflicting instance
    assert len(extra_a) == 1
    assert len(extra_b) == 1
    assert len(extra_a[0].instances) == 1
    assert len(extra_b[0].instances) == 1
    assert extra_a[0].instances[0].points[0].x == 2
    assert extra_b[0].instances[0].points[0].x == 3

    # Check that objects were unified
    assert extra_a[0].video == extra_b[0].video

    # Check resolving the conflict using new
    Labels.finish_complex_merge(labels_a, extra_b)
    assert len(labels_a.labeled_frames) == 2
    assert len(labels_a.labeled_frames[0].instances) == 2
    assert labels_a.labeled_frames[0].instances[1].points[0].x == 3


def test_merge_predictions():
    dummy_video_a = Video.from_filename("foo.mp4")
    dummy_video_b = Video.from_filename("foo.mp4")

    dummy_skeleton_a = Skeleton()
    dummy_skeleton_a.add_node("node")

    dummy_skeleton_b = Skeleton()
    dummy_skeleton_b.add_node("node")

    dummy_instances_a = []
    dummy_instances_a.append(
        Instance(skeleton=dummy_skeleton_a, points=dict(node=Point(1, 1)))
    )
    dummy_instances_a.append(
        Instance(skeleton=dummy_skeleton_a, points=dict(node=Point(2, 2)))
    )

    labels_a = Labels()
    labels_a.append(
        LabeledFrame(dummy_video_a, frame_idx=0, instances=dummy_instances_a)
    )

    dummy_instances_b = []
    dummy_instances_b.append(
        Instance(skeleton=dummy_skeleton_b, points=dict(node=Point(1, 1)))
    )
    dummy_instances_b.append(
        PredictedInstance(
            skeleton=dummy_skeleton_b, points=dict(node=Point(3, 3)), score=1
        )
    )

    labels_b = Labels()
    labels_b.append(
        LabeledFrame(dummy_video_b, frame_idx=0, instances=dummy_instances_b)
    )

    # Frames have one redundant instance (perfect match) and all the
    # non-matching instances are different types (one predicted, one not).
    merged, extra_a, extra_b = Labels.complex_merge_between(labels_a, labels_b)
    assert len(merged[dummy_video_a]) == 1
    assert len(merged[dummy_video_a][0]) == 1  # the predicted instance was merged
    assert not extra_a
    assert not extra_b


def test_merge_with_package(min_labels_robot, tmpdir):
    # Add a suggestion and save with images.
    labels = min_labels_robot
    labels.suggestions.append(
        sleap.io.dataset.SuggestionFrame(video=labels.video, frame_idx=1)
    )
    pkg_path = os.path.join(tmpdir, "test.pkg.slp")
    assert len(labels.predicted_instances) == 0
    labels.save(pkg_path, with_images=True, embed_suggested=True)

    # Load package.
    labels_pkg = sleap.load_file(pkg_path)
    assert isinstance(labels_pkg.video.backend, sleap.io.video.HDF5Video)
    assert labels_pkg.video.backend.has_embedded_images
    assert isinstance(
        labels_pkg.video.backend._source_video.backend, sleap.io.video.MediaVideo
    )
    assert len(labels_pkg.predicted_instances) == 0

    # Add prediction.
    inst = labels_pkg.user_instances[0]
    pts = inst.numpy()
    inst_pr = sleap.PredictedInstance.from_pointsarray(
        pts,
        skeleton=labels_pkg.skeleton,
        point_confidences=np.zeros(len(pts)),
        instance_score=1.0,
    )
    labels_pkg.append(
        sleap.LabeledFrame(
            video=labels_pkg.suggestions[0].video,
            frame_idx=labels_pkg.suggestions[0].frame_idx,
            instances=[inst_pr],
        )
    )

    # Save labels without image data.
    preds_path = pkg_path + ".predictions.slp"
    labels_pkg.save(preds_path)

    # Load predicted labels created from package.
    labels_pr = sleap.load_file(preds_path)
    assert len(labels_pr.predicted_instances) == 1

    # Merge with base labels.
    base_video_path = labels.video.backend.filename
    merged, extra_base, extra_new = sleap.Labels.complex_merge_between(
        labels, labels_pr
    )
    assert len(labels.videos) == 1
    assert labels.video.backend.filename == base_video_path
    assert len(labels.predicted_instances) == 1
    assert len(extra_base) == 0
    assert len(extra_new) == 0
    assert labels.predicted_instances[0].frame.frame_idx == 1

    # Merge predictions to package instead.
    labels_pkg = sleap.load_file(pkg_path)
    labels_pr = sleap.load_file(preds_path)
    assert len(labels_pkg.predicted_instances) == 0
    base_video_path = labels_pkg.video.backend.filename
    merged, extra_base, extra_new = sleap.Labels.complex_merge_between(
        labels_pkg, labels_pr
    )
    assert len(labels_pkg.videos) == 1
    assert labels_pkg.video.backend.filename == base_video_path
    assert len(labels_pkg.predicted_instances) == 1
    assert len(extra_base) == 0
    assert len(extra_new) == 0
    assert labels_pkg.predicted_instances[0].frame.frame_idx == 1


def test_merge_with_skeleton_conflict(min_labels, tmpdir):
    # Save out base labels
    base_labels = min_labels.copy()
    base_labels.save(f"{tmpdir}/base_labels.slp")

    # Merge labels with a renamed node
    labels = base_labels.copy()
    labels[0].frame_idx = 1
    labels.skeleton.relabel_node("A", "a")
    labels.save(f"{tmpdir}/labels.renamed_node.slp")

    labels = base_labels.copy()
    merged, extra_base, extra_new = sleap.Labels.complex_merge_between(
        labels, sleap.load_file(f"{tmpdir}/labels.renamed_node.slp")
    )
    assert len(extra_base) == 0
    assert len(extra_new) == 0
    assert labels.skeleton.node_names == ["A", "B", "a"]
    assert np.isnan(labels[0][0].numpy()).any(axis=1).tolist() == [False, False, True]
    assert np.isnan(labels[1][0].numpy()).any(axis=1).tolist() == [True, False, False]

    # Merge labels with a new node
    labels = base_labels.copy()
    labels[0].frame_idx = 1
    labels.skeleton.add_node("C")
    inst = labels[0][0]
    inst["C"] = sleap.instance.Point(x=1, y=2, visible=True)
    labels.save(f"{tmpdir}/labels.new_node.slp")

    labels = base_labels.copy()
    merged, extra_base, extra_new = sleap.Labels.complex_merge_between(
        labels, sleap.load_file(f"{tmpdir}/labels.new_node.slp")
    )
    assert len(extra_base) == 0
    assert len(extra_new) == 0
    assert labels.skeleton.node_names == ["A", "B", "C"]
    assert np.isnan(labels[0][0].numpy()).any(axis=1).tolist() == [False, False, True]
    assert np.isnan(labels[1][0].numpy()).any(axis=1).tolist() == [False, False, False]

    # Merge labels with a deleted node
    labels = base_labels.copy()
    labels[0].frame_idx = 1
    labels.skeleton.delete_node("A")
    labels.save(f"{tmpdir}/labels.deleted_node.slp")

    labels = base_labels.copy()
    merged, extra_base, extra_new = sleap.Labels.complex_merge_between(
        labels, sleap.load_file(f"{tmpdir}/labels.deleted_node.slp")
    )
    assert len(extra_base) == 0
    assert len(extra_new) == 0
    assert labels.skeleton.node_names == ["A", "B"]
    assert np.isnan(labels[0][0].numpy()).any(axis=1).tolist() == [False, False]
    assert np.isnan(labels[1][0].numpy()).any(axis=1).tolist() == [True, False]
    assert (labels[0][0].numpy()[1] == labels[1][0].numpy()[1]).all()


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
        labels.append(
            LabeledFrame(
                dummy_video,
                frame_idx=i,
                instances=[Instance(dummy_skeleton), Instance(dummy_skeleton)],
            )
        )
    for i in range(10):
        labels.append(
            LabeledFrame(
                dummy_video2,
                frame_idx=i,
                instances=[
                    Instance(dummy_skeleton),
                    Instance(dummy_skeleton),
                    Instance(dummy_skeleton),
                ],
            )
        )

    assert len(labels.all_instances) == 50
    assert len(list(labels.instances(video=dummy_video))) == 20
    assert len(list(labels.instances(video=dummy_video2))) == 30


def test_basic_suggestions(small_robot_mp4_vid):
    dummy_video = small_robot_mp4_vid
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance])

    labels = Labels()
    labels.append(dummy_frame)

    suggestions = VideoFrameSuggestions.suggest(
        labels=labels, params=dict(method="sample", per_video=13)
    )
    labels.set_suggestions(suggestions)

    assert len(labels.get_video_suggestions(dummy_video)) == 13


def test_deserialize_suggestions(small_robot_mp4_vid, tmpdir):
    dummy_video = small_robot_mp4_vid
    dummy_skeleton = Skeleton()
    dummy_instance = Instance(dummy_skeleton)
    dummy_frame = LabeledFrame(dummy_video, frame_idx=0, instances=[dummy_instance])

    labels = Labels()
    labels.append(dummy_frame)

    suggestions = VideoFrameSuggestions.suggest(
        labels=labels, params=dict(method="sample", per_video=13)
    )
    labels.set_suggestions(suggestions)

    filename = os.path.join(tmpdir, "new_suggestions.h5")
    Labels.save_file(filename=filename, labels=labels)

    new_suggestion_labels = Labels.load_file(filename)
    assert len(suggestions) == len(new_suggestion_labels.suggestions)
    assert [frame.frame_idx for frame in suggestions] == [
        frame.frame_idx for frame in new_suggestion_labels.suggestions
    ]


def test_load_labels_mat(mat_labels):
    assert len(mat_labels.nodes) == 6
    assert len(mat_labels) == 43


@pytest.mark.parametrize("format", ["png", "mjpeg/avi"])
def test_save_labels_with_frame_data(multi_skel_vid_labels, tmpdir, format):
    """
    Test saving and loading a labels dataset with frame data included
    as JSON.
    """

    # Lets take a subset of the labels so this doesn't take too long
    multi_skel_vid_labels.labeled_frames = multi_skel_vid_labels.labeled_frames[5:30]

    filename = os.path.join(tmpdir, "test.json")
    Labels.save_json(
        multi_skel_vid_labels,
        filename=filename,
        save_frame_data=True,
        frame_data_format=format,
        # compress=True,
    )

    print(filename, os.path.exists(filename + ".zip"))

    # Load the data back in
    loaded_labels = Labels.load_json(f"{filename}.zip")

    # Check that we have the same thing
    _check_labels_match(multi_skel_vid_labels, loaded_labels, format=format)

    # Make sure we can load twice
    loaded_labels = Labels.load_json(f"{filename}.zip")


def test_save_labels_and_frames_hdf5(multi_skel_vid_labels, tmpdir):
    # Lets take a subset of the labels so this doesn't take too long
    labels = multi_skel_vid_labels
    labels.labeled_frames = labels.labeled_frames[5:30]

    filename = os.path.join(tmpdir, "test.h5")

    Labels.save_hdf5(filename=filename, labels=labels, save_frame_data=True)

    loaded_labels = Labels.load_hdf5(filename=filename)

    _check_labels_match(labels, loaded_labels)

    # Rename file (after closing videos)
    for vid in loaded_labels.videos:
        vid.close()
    filerename = os.path.join(tmpdir, "test_rename.h5")
    os.rename(filename, filerename)

    # Make sure we open can after rename
    loaded_labels = Labels.load_hdf5(filename=filerename)


def test_save_frame_data_hdf5(min_labels_slp, tmpdir):
    labels = Labels(min_labels_slp.labeled_frames)
    labels.append(LabeledFrame(video=labels.video, frame_idx=1))
    labels.suggestions.append(SuggestionFrame(video=labels.video, frame_idx=2))

    fn = os.path.join(tmpdir, "test_user_only.slp")
    labels.save_frame_data_hdf5(
        fn,
        format="png",
        user_labeled=True,
        all_labeled=False,
        suggested=False,
    )
    assert Video.from_filename(fn, dataset="video0").embedded_frame_inds == [0]

    fn = os.path.join(tmpdir, "test_all_labeled.slp")
    labels.save_frame_data_hdf5(
        fn,
        format="png",
        user_labeled=False,
        all_labeled=True,
        suggested=False,
    )
    assert Video.from_filename(fn, dataset="video0").embedded_frame_inds == [0, 1]

    fn = os.path.join(tmpdir, "test_suggested.slp")
    labels.save_frame_data_hdf5(
        fn,
        format="png",
        user_labeled=False,
        all_labeled=False,
        suggested=True,
    )
    assert Video.from_filename(fn, dataset="video0").embedded_frame_inds == [2]

    fn = os.path.join(tmpdir, "test_all.slp")
    labels.save_frame_data_hdf5(
        fn,
        format="png",
        user_labeled=False,
        all_labeled=True,
        suggested=True,
    )
    assert Video.from_filename(fn, dataset="video0").embedded_frame_inds == [0, 1, 2]


def test_save_labels_with_images(min_labels_slp, tmpdir):
    labels = Labels(min_labels_slp.labeled_frames)
    labels.append(LabeledFrame(video=labels.video, frame_idx=1))
    labels.suggestions.append(SuggestionFrame(video=labels.video, frame_idx=2))

    fn = os.path.join(tmpdir, "test_user_only.slp")
    labels.save(
        fn,
        with_images=True,
        embed_all_labeled=False,
        embed_suggested=False,
    )
    assert Labels.load_file(fn).video.embedded_frame_inds == [0]

    fn = os.path.join(tmpdir, "test_all_labeled.slp")
    labels.save(
        fn,
        with_images=True,
        embed_all_labeled=True,
        embed_suggested=False,
    )
    assert Labels.load_file(fn).video.embedded_frame_inds == [0, 1]

    fn = os.path.join(tmpdir, "test_suggested.slp")
    labels.save(
        fn,
        with_images=True,
        embed_all_labeled=False,
        embed_suggested=True,
    )
    assert Labels.load_file(fn).video.embedded_frame_inds == [0, 2]

    fn = os.path.join(tmpdir, "test_all.slp")
    labels.save(
        fn,
        with_images=True,
        embed_all_labeled=True,
        embed_suggested=True,
    )
    assert Labels.load_file(fn).video.embedded_frame_inds == [0, 1, 2]


def test_labels_hdf5(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, "test.h5")

    Labels.save_hdf5(filename=filename, labels=labels)

    loaded_labels = Labels.load_hdf5(filename=filename)

    _check_labels_match(labels, loaded_labels)


def test_labels_predicted_hdf5(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, "test.h5")

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
    filename = os.path.join(tmpdir, "test.h5")

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


def test_hdf5_from_predicted(multi_skel_vid_labels, tmpdir):
    labels = multi_skel_vid_labels
    filename = os.path.join(tmpdir, "test.h5")

    # Add some predicted instances to create from_predicted links
    for frame_num, frame in enumerate(labels):
        if frame_num % 20 == 0:
            frame.instances[0].from_predicted = PredictedInstance.from_instance(
                frame.instances[0], float(frame_num)
            )
            frame.instances.append(frame.instances[0].from_predicted)

    # Save and load, compare the results
    Labels.save_hdf5(filename=filename, labels=labels)
    loaded_labels = Labels.load_hdf5(filename=filename)

    for frame_num, frame in enumerate(loaded_labels):
        if frame_num % 20 == 0:
            assert frame.instances[0].from_predicted.score == float(frame_num)


def test_hdf5_empty_save(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")
    Labels.save_hdf5(filename=filename, labels=labels)

    dummy_video = Video.from_filename("foo.mp4")
    labels.videos.append(dummy_video)
    Labels.save_hdf5(filename=filename, labels=labels)


def test_makedirs(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "new/dirs/test.h5")
    Labels.save_file(filename=filename, labels=labels)


def test_multivideo_tracks():
    vid_a = Video.from_filename("foo.mp4")
    vid_b = Video.from_filename("bar.mp4")

    skeleton = Skeleton.load_json("tests/data/skeleton/fly_skeleton_legs.json")

    track_a = Track(spawned_on=2, name="A")
    track_b = Track(spawned_on=3, name="B")

    inst_a = Instance(track=track_a, skeleton=skeleton)
    inst_b = Instance(track=track_b, skeleton=skeleton)

    lf_a = LabeledFrame(vid_a, frame_idx=2, instances=[inst_a])
    lf_b = LabeledFrame(vid_b, frame_idx=3, instances=[inst_b])

    labels = Labels(labeled_frames=[lf_a, lf_b])

    # Try setting video B instance to track used in video A
    labels.track_swap(vid_b, new_track=track_a, old_track=track_b, frame_range=(3, 4))

    assert inst_b.track == track_a


def test_many_tracks_hdf5(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")

    labels.tracks = [Track(spawned_on=i, name=f"track {i}") for i in range(4000)]

    Labels.save_hdf5(filename=filename, labels=labels)


def test_many_videos_hdf5(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")

    labels.videos = [Video.from_filename(f"video {i}.mp4") for i in range(3000)]

    Labels.save_hdf5(filename=filename, labels=labels)


def test_many_suggestions_hdf5(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")
    video = Video.from_filename("foo.mp4")
    labels.videos = [video]

    labels.suggestions = [SuggestionFrame(video, i) for i in range(3000)]

    Labels.save_hdf5(filename=filename, labels=labels)


def test_path_fix(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")

    # Add a video without a full path
    labels.add_video(Video.from_filename("small_robot.mp4"))

    Labels.save_hdf5(filename=filename, labels=labels)

    # Pass the path to the directory with the video
    labels = Labels.load_file(filename, video_search="tests/data/videos/")

    # Make sure we got the actual video path by searching that directory
    assert len(labels.videos) == 1
    assert labels.videos[0].filename == "tests/data/videos/small_robot.mp4"


def test_path_fix_with_new_full_path(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")

    # Add video with bad filename
    labels.add_video(Video.from_filename("foo.mp4"))

    Labels.save_hdf5(filename=filename, labels=labels)

    # Pass list of full video paths to use instead of those in labels
    labels = Labels.load_file(
        filename, video_search=["tests/data/videos/small_robot.mp4"]
    )

    # Make sure we got the actual video path by searching that directory
    assert len(labels.videos) == 1
    assert labels.videos[0].filename == "tests/data/videos/small_robot.mp4"


def test_load_file(tmpdir):
    labels = Labels()
    filename = os.path.join(tmpdir, "test.h5")
    labels.add_video(Video.from_filename("small_robot.mp4"))
    Labels.save_hdf5(filename=filename, labels=labels)

    # Fix video path from full path
    labels = load_file(filename, search_paths="tests/data/videos/small_robot.mp4")
    assert Path(labels.video.filename).samefile("tests/data/videos/small_robot.mp4")

    # No auto-detect
    labels = load_file(filename, detect_videos=False)
    assert labels.video.filename == "small_robot.mp4"

    # Fix video path by searching in the labels folder
    tmpvid = tmpdir.join("small_robot.mp4")
    tmpvid.write("")  # dummy file
    assert load_file(filename).video.filename == tmpvid
    assert load_file(filename, search_paths=str(tmpdir)).video.filename == tmpvid
    assert load_file(filename, search_paths=str(tmpvid)).video.filename == tmpvid


def test_local_path_save(tmpdir, monkeypatch):

    filename = "test.h5"

    # Set current working directory (monkeypatch isolates other tests)
    monkeypatch.chdir(tmpdir)

    # Try saving with relative path
    Labels.save_file(filename=filename, labels=Labels())

    assert os.path.exists(os.path.join(tmpdir, filename))


def test_slp_file(min_labels_slp, min_labels):
    assert min_labels.videos[0].filename == min_labels_slp.videos[0].filename


def test_provenance(tmpdir):
    labels = Labels(provenance=dict(source="test_provenance"))
    filename = os.path.join(tmpdir, "test.slp")

    # Add a video without a full path
    labels.add_video(Video.from_filename("small_robot.mp4"))

    Labels.save_file(filename=filename, labels=labels)

    labels = Labels.load_file(filename)
    print(labels.provenance)
    assert labels.provenance["source"] == "test_provenance"


def test_has_frame():
    video = Video(backend=MediaVideo)
    labels = Labels([LabeledFrame(video=video, frame_idx=0)])

    assert labels.has_frame(labels[0])
    assert labels.has_frame(labels[0], use_cache=False)
    assert labels.has_frame(LabeledFrame(video=video, frame_idx=0))
    assert labels.has_frame(video=video, frame_idx=0)
    assert labels.has_frame(video=video, frame_idx=0, use_cache=False)
    assert not labels.has_frame(LabeledFrame(video=video, frame_idx=1))
    assert not labels.has_frame(LabeledFrame(video=video, frame_idx=1), use_cache=False)
    assert not labels.has_frame(video=video, frame_idx=1)
    with pytest.raises(ValueError):
        labels.has_frame()
    with pytest.raises(ValueError):
        labels.has_frame(video=video)
    with pytest.raises(ValueError):
        labels.has_frame(frame_idx=1)


@pytest.fixture
def removal_test_labels():
    skeleton = Skeleton()
    video = Video(backend=MediaVideo(filename="test"))
    lf_user_only = LabeledFrame(
        video=video, frame_idx=0, instances=[Instance(skeleton=skeleton)]
    )
    lf_pred_only = LabeledFrame(
        video=video, frame_idx=1, instances=[PredictedInstance(skeleton=skeleton)]
    )
    lf_both = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[Instance(skeleton=skeleton), PredictedInstance(skeleton=skeleton)],
    )
    labels = Labels([lf_user_only, lf_pred_only, lf_both])
    return labels


def test_copy(removal_test_labels):
    new_labels = removal_test_labels.copy()
    new_labels[0].instances = []
    new_labels.remove_frame(new_labels[-1])
    assert len(removal_test_labels[0].instances) == 1
    assert len(removal_test_labels) == 3


def test_remove_user_instances(removal_test_labels):
    labels = removal_test_labels
    assert len(labels) == 3

    labels.remove_user_instances()
    assert len(labels) == 2
    assert labels[0].frame_idx == 1
    assert not labels[0].has_user_instances
    assert labels[0].has_predicted_instances
    assert labels[1].frame_idx == 2
    assert not labels[1].has_user_instances
    assert labels[1].has_predicted_instances


def test_remove_user_instances_with_new_labels(removal_test_labels):
    labels = removal_test_labels
    assert len(labels) == 3

    new_labels = Labels(
        [
            LabeledFrame(
                video=labels.video,
                frame_idx=0,
                instances=[Instance(skeleton=labels.skeleton)],
            )
        ]
    )
    labels.remove_user_instances(new_labels=new_labels)
    assert len(labels) == 2
    assert labels[0].frame_idx == 1
    assert not labels[0].has_user_instances
    assert labels[0].has_predicted_instances
    assert labels[1].frame_idx == 2
    assert labels[1].has_user_instances
    assert labels[1].has_predicted_instances


def test_remove_predictions(removal_test_labels):
    labels = removal_test_labels
    assert len(labels) == 3

    labels.remove_predictions()
    assert len(labels) == 2
    assert labels[0].frame_idx == 0
    assert labels[0].has_user_instances
    assert not labels[0].has_predicted_instances
    assert labels[1].frame_idx == 2
    assert labels[1].has_user_instances
    assert not labels[1].has_predicted_instances


def test_remove_predictions_with_new_labels(removal_test_labels):
    labels = removal_test_labels
    assert len(labels) == 3

    new_labels = Labels(
        [
            LabeledFrame(
                video=labels.video,
                frame_idx=1,
                instances=[PredictedInstance(skeleton=labels.skeleton)],
            )
        ]
    )
    labels.remove_predictions(new_labels=new_labels)
    assert len(labels) == 2
    assert labels[0].frame_idx == 0
    assert labels[0].has_user_instances
    assert not labels[0].has_predicted_instances
    assert labels[1].frame_idx == 2
    assert labels[1].has_user_instances
    assert labels[1].has_predicted_instances


def test_labels_numpy(centered_pair_predictions):
    trx = centered_pair_predictions.numpy(video=None, all_frames=False, untracked=False)
    assert trx.shape == (1100, 27, 24, 2)

    trx = centered_pair_predictions.numpy(video=None, all_frames=True, untracked=False)
    assert trx.shape == (1100, 27, 24, 2)

    # Remove the first labeled frame
    centered_pair_predictions.remove_frame(centered_pair_predictions[0])
    assert len(centered_pair_predictions) == 1099

    trx = centered_pair_predictions.numpy(video=None, all_frames=False, untracked=False)
    assert trx.shape == (1099, 27, 24, 2)

    trx = centered_pair_predictions.numpy(video=None, all_frames=True, untracked=False)
    assert trx.shape == (1100, 27, 24, 2)

    labels_single = Labels(
        [
            LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=[lf.instances[0]]
            )
            for lf in centered_pair_predictions
        ]
    )
    assert labels_single.numpy().shape == (1100, 1, 24, 2)

    assert centered_pair_predictions.numpy(untracked=True).shape == (1100, 5, 24, 2)
    for lf in centered_pair_predictions:
        for inst in lf:
            inst.track = None
    centered_pair_predictions.tracks = []
    assert centered_pair_predictions.numpy(untracked=False).shape == (1100, 0, 24, 2)


def test_remove_track(centered_pair_predictions):
    labels = centered_pair_predictions

    track = labels.tracks[-1]
    track_insts = [inst for inst in labels.instances() if inst.track == track]
    labels.remove_track(track)
    assert track not in labels.tracks
    assert all(inst.track != track for inst in labels.instances())

    track = labels.tracks[0]
    track_insts = [inst for inst in labels.instances() if inst.track == track]
    labels.remove_track(track)
    assert track not in labels.tracks
    assert all(inst.track != track for inst in labels.instances())


def test_remove_all_tracks(centered_pair_predictions):
    labels = centered_pair_predictions
    labels.remove_all_tracks()
    assert len(labels.tracks) == 0
    assert all(inst.track is None for inst in labels.instances())


def test_remove_empty_frames(min_labels):
    min_labels.append(sleap.LabeledFrame(video=min_labels.video, frame_idx=2))
    assert len(min_labels) == 2
    assert len(min_labels[-1]) == 0
    min_labels.remove_empty_frames()
    assert len(min_labels) == 1
    assert len(min_labels[0]) == 2


def test_remove_empty_instances(min_labels):
    for inst in min_labels.labeled_frames[0].instances:
        for pt in inst.points:
            pt.visible = False
    min_labels.remove_empty_instances(keep_empty_frames=True)
    assert len(min_labels) == 1
    assert len(min_labels[0]) == 0


def test_remove_empty_instances_and_frames(min_labels):
    for inst in min_labels.labeled_frames[0].instances:
        for pt in inst.points:
            pt.visible = False
    min_labels.remove_empty_instances(keep_empty_frames=False)
    assert len(min_labels) == 0


def test_merge_nodes(min_labels):
    labels = min_labels.copy()
    labels.skeleton.add_node("a")

    inst = labels[0][0]
    inst["A"] = Point(x=np.nan, y=np.nan, visible=False)
    inst["a"] = Point(x=1, y=2, visible=True)
    inst = labels[0][1]
    inst["A"] = Point(x=0, y=1, visible=False)
    inst["a"] = Point(x=1, y=2, visible=True)

    labels.merge_nodes("A", "a")

    assert labels.skeleton.node_names == ["A", "B"]

    inst = labels[0][0]
    assert inst["A"].x == 1 and inst["A"].y == 2
    assert len(inst.nodes) == 2
    inst = labels[0][1]
    assert inst["A"].x == 1 and inst["A"].y == 2
    assert len(inst.nodes) == 2


def test_split(centered_pair_predictions):
    labels_a, labels_b = centered_pair_predictions.split(0.8)
    assert len(labels_a) == 880
    assert len(labels_b) == 220

    assert (
        len(
            np.intersect1d(
                [lf.frame_idx for lf in labels_a], [lf.frame_idx for lf in labels_b]
            )
        )
        == 0
    )

    labels_a, labels_b = centered_pair_predictions.extract([0]).split(0.8)
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a[0] != labels_b[0]
    assert labels_a[0].frame_idx == labels_b[0].frame_idx

    labels_a, labels_b = centered_pair_predictions.extract([0]).split(0.8, copy=False)
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a[0] == labels_b[0]


def test_remove_untracked_instances(min_tracks_2node_labels):
    """Test removal of untracked instances and empty frames.

    Args:
        min_tracks_2node_labels: Labels object which contains user labeled frames with
        tracked instances.
    """
    labels = min_tracks_2node_labels

    # Preprocessing
    labels.labeled_frames[0].instances[0].track = None
    labels.labeled_frames[1].instances = []
    assert any(
        [inst.track is None for lf in labels.labeled_frames for inst in lf.instances]
    )
    assert any([len(lf.instances) == 0 for lf in labels.labeled_frames])

    # Test function with remove_empty_frames=False
    labels.remove_untracked_instances(remove_empty_frames=False)
    assert all(
        [
            inst.track is not None
            for lf in labels.labeled_frames
            for inst in lf.instances
        ]
    )
    assert any([len(lf.instances) == 0 for lf in labels.labeled_frames])

    # Test function with remove_empty_frames=True
    labels.remove_untracked_instances(remove_empty_frames=True)
    assert all([len(lf.instances) > 0 for lf in labels.labeled_frames])

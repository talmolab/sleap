import numpy as np

from sleap.info.feature_suggestions import (
    FrameItem,
    FrameGroupSet,
    ItemStack,
    FeatureSuggestionPipeline,
)


def test_frame_item(small_robot_mp4_vid):
    item = FrameItem(video=small_robot_mp4_vid, frame_idx=12)
    assert np.all(
        item.get_raw_image(scale=1.0)[:15, :15, 0]
        == small_robot_mp4_vid[12][:15, :15, 0]
    )

    assert item.get_raw_image(scale=1).shape == (
        1,
        small_robot_mp4_vid.height,
        small_robot_mp4_vid.width,
        small_robot_mp4_vid.channels,
    )

    assert item.get_raw_image(scale=0.5).shape == (
        1,
        small_robot_mp4_vid.height // 2,
        small_robot_mp4_vid.width // 2,
        small_robot_mp4_vid.channels,
    )


def test_frame_group_set(small_robot_mp4_vid):
    groupset = FrameGroupSet(method="testing")

    items = [FrameItem(small_robot_mp4_vid, i) for i in range(10)]
    more_items = [FrameItem(small_robot_mp4_vid, i) for i in range(13, 16)]

    for item in items:
        groupset.append_to_group(group=item.frame_idx % 3, item=item)

    assert groupset.get_item_group(items[0]) == 0
    assert groupset.get_item_group(items[1]) == 1
    assert groupset.get_item_group(items[2]) == 2
    assert groupset.get_item_group(items[3]) == 0

    assert set(groupset.all_items) == set(items)

    groupset.extend_group_items(group=3, item_list=more_items)
    assert groupset.get_item_group(more_items[0]) == 3
    assert set(groupset.all_items) == set(items).union(more_items)

    sampled_groupset = groupset.sample(per_group=2)

    for group, item_list in sampled_groupset.groups:
        assert len(item_list) == 2


def test_item_stack(centered_pair_vid, small_robot_mp4_vid):
    stack = ItemStack()

    videos = [centered_pair_vid, small_robot_mp4_vid]
    stack.make_sample_group(videos, samples_per_video=3, sample_method="stride")
    stack.get_all_items_from_group()

    # Make sure that we got the right frame items
    assert len(stack.items) == 6
    assert stack.items[0].frame_idx == 0
    assert stack.items[1].frame_idx == centered_pair_vid.frames // 3

    assert stack.items[3].frame_idx == 0
    assert stack.items[4].frame_idx == small_robot_mp4_vid.frames // 3

    stack.get_raw_images(scale=0.1)

    # Make sure an item owns its row of data
    assert stack.get_item_data_idxs(stack.items[1]) == (1,)
    assert stack.get_item_by_data_row(3) == stack.items[3]

    # Make sure that we loaded correctly sized data
    i = len(stack.items)
    h = max(centered_pair_vid.height // 10, small_robot_mp4_vid.height // 10)
    w = max(centered_pair_vid.width // 10, small_robot_mp4_vid.width // 10)
    c = max(centered_pair_vid.channels, small_robot_mp4_vid.channels)
    assert stack.data.shape == (i, h, w, c)
    assert stack.get_item_data(stack.items[1]).shape == (1, h, w, c)

    stack.flatten()

    # Make sure that data was correctly flattened
    assert stack.data.shape == (i, h * w * c)

    stack.pca(n_components=3)

    # Make sure that we have right number of dimensions after pca
    assert stack.data.shape == (i, 3)

    stack.kmeans(n_clusters=2)

    # Make sure we have right number of groups after kmeans
    assert len(stack.current_groupset.group_data.keys()) == 2

    stack.sample_groups(samples_per_group=1)

    for group, item_list in stack.current_groupset.groups:
        assert len(item_list) == 1

    stack.get_all_items_from_group()

    # Make sure we got 1 item from each of the 2 groups
    assert len(stack.items) == 2

    frame_items = stack.to_suggestion_frames(group_offset=10)
    assert frame_items[0].frame_idx == stack.items[0].frame_idx
    assert frame_items[0].group == 10
    assert frame_items[1].frame_idx == stack.items[1].frame_idx
    assert frame_items[1].group == 11


def test_brisk_suggestions(centered_pair_vid):
    # TODO: add test for brisk_bag_of_features
    pass


def test_feature_suggestion_pipeline(centered_pair_vid):
    videos = [centered_pair_vid]
    pipeline = FeatureSuggestionPipeline(
        per_video=5,
        scale=0.1,
        sample_method="random",
        feature_type="raw",
        n_components=3,
        n_clusters=2,
        per_cluster=1,
    )

    suggestions = pipeline.get_suggestion_frames(videos)

    assert len(suggestions) == 2

import pytest
from unittest.mock import Mock

def switch_frame(self, video):
    """Jump to the last labeled frame or maintain the same frame index if the video is long enough."""
    current_frame_idx = self.state["frame_idx"]

    if video.num_frames > current_frame_idx:
        # if the new video is long enough, stay on the same frame
        self.state["frame_idx"] = current_frame_idx
        # old logic
    else:
        # if the new video is not long enough, find the last labeled frame
        last_label = self.labels.find_last(video)
        if last_label is not None:
            self.state["frame_idx"] = last_label.frame_idx
        else:
            self.state["frame_idx"] = 0

@pytest.fixture
def context():
    ctx = Mock()
    ctx.state = {"frame_idx": 3}
    ctx.labels = Mock()
    return ctx

# on a frame that both videos have, make sure frame index remains the same when switching (new logic)
def test_switch_frame_same_index(context):
    video1 = Mock()
    video1.num_frames = 10
    video2 = Mock()
    video2.num_frames = 10

    # Switch to video2, expecting the frame index to remain the same
    switch_frame(context, video2)

    assert context.state["frame_idx"] == 3

# on a frame that is way out of range for the shorter video without any labels, make sure the frame index switches to the first frame (old logic)
def test_switch_frame_out_of_range_no_labels(context):
    video1 = Mock()
    video1.num_frames = 10
    video2 = Mock()
    video2.num_frames = 3

    # no labels found
    context.labels.find_last.return_value = None

    # Switch to video2, expecting the frame index to switch to the first frame
    switch_frame(context, video2)

    assert context.state["frame_idx"] == 0

# on a frame that is way out of range for the shorter video with labels, make sure the frame index switches to the last labeled frame (old logic)
def test_switch_frame_out_of_range_with_labels(context):
    video1 = Mock()
    video1.num_frames = 10
    video2 = Mock()
    video2.num_frames = 3

    # Last label found at frame 2
    last_label = Mock()
    last_label.frame_idx = 2
    context.labels.find_last.return_value = last_label

    # Switch to video2, expecting the frame index to switch to the last labeled frame
    switch_frame(context, video2)

    assert context.state["frame_idx"] == 2

# go to the last frame in the shorter video, switch to the longer video, try switching back to the shorter video (edge case)
def test_switch_frame_last_to_longer_and_back(context):
    video1 = Mock()
    video1.num_frames = 3
    video2 = Mock()
    video2.num_frames = 10

    # Go to the last frame in the shorter video
    context.state["frame_idx"] = 2

    # Switch to the longer video
    switch_frame(context, video2)
    assert context.state["frame_idx"] == 2

    # Switch back to the shorter video
    switch_frame(context, video1)
    assert context.state["frame_idx"] == 2

# go to the last frame in the shorter video, switch to the longer video, move one frame up, try switching back to the shorter video (edge case)
def test_switch_frame_last_to_longer_and_up_and_back(context):
    video1 = Mock()
    video1.num_frames = 3
    video2 = Mock()
    video2.num_frames = 10

    # Last label found at frame 2
    last_label = Mock()
    last_label.frame_idx = 2
    context.labels.find_last.return_value = last_label

    # Go to the last frame in the shorter video
    context.state["frame_idx"] = 2

    # Switch to the longer video
    switch_frame(context, video2)
    assert context.state["frame_idx"] == 2

    # Move one frame up in the longer video
    context.state["frame_idx"] += 1
    assert context.state["frame_idx"] == 3

    # Switch back to the shorter video
    switch_frame(context, video1)
    assert context.state["frame_idx"] == 2

from sleap.nn.utils import VideoLoader
import numpy as np


def test_grayscale_video():
    vid = VideoLoader(filename="tests/data/videos/small_robot.mp4",)
    assert vid.shape[-1] == 3

    vid = VideoLoader(filename="tests/data/videos/small_robot.mp4", grayscale=True)
    assert vid.shape[-1] == 1


def test_dummy_video():
    vid = VideoLoader(filename="tests/data/videos/small_robot.mp4", dummy=True)

    x = vid.load_frames([1, 3, 5])
    assert x.shape == (3, 320, 560, 3)
    assert np.all(x == 0)

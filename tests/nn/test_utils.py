from sleap.nn.utils import VideoLoader


def test_grayscale_video():
    vid = VideoLoader(filename="tests/data/videos/small_robot.mp4",)
    assert vid.shape[-1] == 3

    vid = VideoLoader(filename="tests/data/videos/small_robot.mp4", grayscale=True)
    assert vid.shape[-1] == 1

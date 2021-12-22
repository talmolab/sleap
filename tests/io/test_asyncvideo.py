import pytest
import sys
from sleap import Video
from sleap.io.asyncvideo import AsyncVideo


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="ZMQ testing breaks locally on Windows"
)
def test_async_video(centered_pair_vid, small_robot_mp4_vid):
    async_video = AsyncVideo.from_video(centered_pair_vid, frames_per_chunk=23)

    all_idxs = []
    for idxs, frames in async_video.chunks:
        assert len(idxs) in (23, 19)  # 19 for last chunk
        all_idxs.extend(idxs)

        assert frames.shape[0] == len(idxs)
        assert frames.shape[1:] == centered_pair_vid.shape[1:]

    assert len(all_idxs) == centered_pair_vid.num_frames

    # make sure we can load another video (i.e., previous video closed)

    async_video = AsyncVideo.from_video(
        small_robot_mp4_vid, frame_idxs=range(0, 10, 2), frames_per_chunk=10
    )

    for idxs, frames in async_video.chunks:
        # there should only be single chunk
        assert idxs == list(range(0, 10, 2))

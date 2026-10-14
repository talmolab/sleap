from sleap.gui.widgets.video_worker import FrameLoaderThread


def test_request_frame_refreshes_copy_on_grayscale_change(qtbot, small_robot_mp4_vid):
    """Worker should re-copy the video after its grayscale setting changes (#2882)."""
    video = small_robot_mp4_vid
    worker = FrameLoaderThread()

    video.grayscale = False
    worker.request_frame(video, 0)
    first_copy = worker.local_video_copy
    assert first_copy.backend_metadata.get("grayscale") is False

    # Simulate "Toggle Grayscale"
    video.grayscale = True
    worker.request_frame(video, 0)
    assert worker.local_video_copy is not first_copy
    assert worker.local_video_copy.backend_metadata.get("grayscale") is True

    # Same setting again should reuse the existing copy
    copy_after_toggle = worker.local_video_copy
    worker.request_frame(video, 0)
    assert worker.local_video_copy is copy_after_toggle

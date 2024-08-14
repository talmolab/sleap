import os
import cv2
from pathlib import Path
from sleap.io.videowriter import VideoWriter, VideoWriterOpenCV, VideoWriterImageio


def test_video_writer(tmpdir, small_robot_mp4_vid):
    out_path = os.path.join(tmpdir, "clip.avi")

    # Make sure video writer works
    writer = VideoWriter.safe_builder(
        out_path,
        height=small_robot_mp4_vid.height,
        width=small_robot_mp4_vid.width,
        fps=small_robot_mp4_vid.fps,
    )

    writer.add_frame(small_robot_mp4_vid[0][0])
    writer.add_frame(small_robot_mp4_vid[1][0])

    writer.close()

    assert os.path.exists(out_path)


def test_cv_video_writer(tmpdir, small_robot_mp4_vid):
    out_path = os.path.join(tmpdir, "clip.avi")

    # Make sure OpenCV video writer works
    writer = VideoWriterOpenCV(
        out_path,
        height=small_robot_mp4_vid.height,
        width=small_robot_mp4_vid.width,
        fps=small_robot_mp4_vid.fps,
    )

    writer.add_frame(small_robot_mp4_vid[0][0])
    writer.add_frame(small_robot_mp4_vid[1][0])

    writer.close()

    assert os.path.exists(out_path)


def test_imageio_video_writer(tmpdir, small_robot_mp4_vid):
    out_path = Path(tmpdir) / "clip.avi"

    # Make sure imageio video writer works
    writer = VideoWriterImageio(
        out_path,
        height=small_robot_mp4_vid.height,
        width=small_robot_mp4_vid.width,
        fps=small_robot_mp4_vid.fps,
    )

    writer.add_frame(small_robot_mp4_vid[0][0])
    writer.add_frame(small_robot_mp4_vid[1][0])

    writer.close()

    assert os.path.exists(out_path)
    # Check attributes
    assert writer.height == small_robot_mp4_vid.height
    assert writer.width == small_robot_mp4_vid.width
    assert writer.fps == small_robot_mp4_vid.fps
    assert writer.filename == out_path
    assert writer.crf == 21
    assert writer.preset == "superfast"

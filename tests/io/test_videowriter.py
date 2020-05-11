import os
from sleap.io.videowriter import VideoWriter, VideoWriterOpenCV


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

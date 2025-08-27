# import os
# import cv2
# from pathlib import Path
# from sleap.io.videowriter import (
#     VideoWriter,
#     VideoWriterOpenCV,
#     VideoWriterImageio,
#     resize_images,
# )


# def test_video_writer(tmpdir, small_robot_mp4_vid):
#     out_path = os.path.join(tmpdir, "clip.avi")

#     # Make sure video writer works
#     writer = VideoWriter.safe_builder(
#         out_path,
#         height=small_robot_mp4_vid.height,
#         width=small_robot_mp4_vid.width,
#         fps=small_robot_mp4_vid.fps,
#     )

#     writer.add_frame(small_robot_mp4_vid[0][0])
#     writer.add_frame(small_robot_mp4_vid[1][0])

#     writer.close()

#     assert os.path.exists(out_path)


# def test_cv_video_writer(tmpdir, small_robot_mp4_vid):
#     out_path = os.path.join(tmpdir, "clip.avi")

#     # Make sure OpenCV video writer works
#     writer = VideoWriterOpenCV(
#         out_path,
#         height=small_robot_mp4_vid.height,
#         width=small_robot_mp4_vid.width,
#         fps=small_robot_mp4_vid.fps,
#     )

#     writer.add_frame(small_robot_mp4_vid[0][0])
#     writer.add_frame(small_robot_mp4_vid[1][0])

#     writer.close()

#     assert os.path.exists(out_path)


# def test_imageio_video_writer_avi(tmpdir, small_robot_mp4_vid):
#     out_path = Path(tmpdir) / "clip.avi"

#     # Make sure imageio video writer works
#     writer = VideoWriterImageio(
#         out_path,
#         height=small_robot_mp4_vid.height,
#         width=small_robot_mp4_vid.width,
#         fps=small_robot_mp4_vid.fps,
#     )

#     writer.add_frame(small_robot_mp4_vid[0][0])
#     writer.add_frame(small_robot_mp4_vid[1][0])

#     writer.close()

#     assert os.path.exists(out_path)
#     # Check attributes
#     assert writer.height == small_robot_mp4_vid.height
#     assert writer.width == small_robot_mp4_vid.width
#     assert writer.fps == small_robot_mp4_vid.fps
#     assert writer.filename == out_path
#     assert writer.crf == 21
#     assert writer.preset == "superfast"


# def test_imageio_video_writer_odd_size(tmpdir, movenet_video):
#     out_path = Path(tmpdir) / "clip.mp4"

#     # Reduce the size of the video frames by 1 pixel in each dimension
#     reduced_height = movenet_video.height - 1
#     reduced_width = movenet_video.width - 1

#     # Initialize the writer with the reduced dimensions
#     writer = VideoWriterImageio(
#         out_path,
#         height=reduced_height,
#         width=reduced_width,
#         fps=movenet_video.fps,
#     )

#     # Resize frames and add them to the video
#     for i in range(len(movenet_video) - 1):
#         frame = movenet_video[i][0]  # Access the actual frame object
#         reduced_frame = cv2.resize(frame, (reduced_width, reduced_height))
#         writer.add_frame(reduced_frame)

#     writer.close()

#     # Assertions to validate the test
#     assert os.path.exists(out_path)
#     assert writer.height == reduced_height
#     assert writer.width == reduced_width
#     assert writer.fps == movenet_video.fps
#     assert writer.filename == out_path
#     assert writer.crf == 21
#     assert writer.preset == "superfast"


# def test_resize(small_robot_mp4_vid):
#     imgs = small_robot_mp4_vid[:4]

#     resized_imgs = resize_images(imgs, 0.25)

#     assert resized_imgs.shape[0] == imgs.shape[0]
#     assert resized_imgs.shape[1] == imgs.shape[1] // 4
#     assert resized_imgs.shape[2] == imgs.shape[2] // 4
#     assert resized_imgs.shape[3] == imgs.shape[3]

import numpy as np
import os
import pytest
import cv2
from sleap.io.dataset import Labels
from sleap.io.visuals import (
    save_labeled_video,
    resize_images,
    VideoMarkerThread,
    main as sleap_render,
)


def test_resize(small_robot_mp4_vid):
    imgs = small_robot_mp4_vid[:4]

    resized_imgs = resize_images(imgs, 0.25)

    assert resized_imgs.shape[0] == imgs.shape[0]
    assert resized_imgs.shape[1] == imgs.shape[1] // 4
    assert resized_imgs.shape[2] == imgs.shape[2] // 4
    assert resized_imgs.shape[3] == imgs.shape[3]


def test_serial_pipeline(centered_pair_predictions, tmpdir):
    frames = [0, 1, 2]
    video_idx = 0
    scale = 0.25

    video = centered_pair_predictions.videos[video_idx]
    frame_images = video.get_frames(frames)

    # Make sure we can resize
    small_images = resize_images(frame_images, scale=scale)

    _, height, width, _ = small_images.shape

    assert height == video.height // (1 / scale)
    assert width == video.width // (1 / scale)

    marker_thread = VideoMarkerThread(
        in_q=None,
        out_q=None,
        labels=centered_pair_predictions,
        video_idx=video_idx,
        scale=scale,
        color_manager=None,
    )

    # Make sure we can mark images
    marked_image_list = marker_thread._mark_images(
        frame_indices=frames,
        frame_images=small_images,
    )

    # There's a point at 201, 186 (i.e. 50.25, 46.5), so make sure it got marked
    assert not np.allclose(
        marked_image_list[0][44:48, 48:52, 0], small_images[0, 44:48, 48:52, 0]
    )

    # Make sure no change where nothing marked
    assert np.allclose(
        marked_image_list[0][10:20, :10, 0], small_images[0, 10:20, :10, 0]
    )


@pytest.mark.parametrize("background", ["original", "black", "white", "grey"])
def test_sleap_render_with_different_backgrounds(background):
    args = (
        f"-o test_{background}.avi -f 2 --scale 1.2 --frames 1,2 --video-index 0 "
        f"--background {background} "
        "tests/data/json_format_v2/centered_pair_predictions.json".split()
    )
    sleap_render(args)
    assert (
        os.path.exists(f"test_{background}.avi")
        and os.path.getsize(f"test_{background}.avi") > 0
    )

    # Check if the background is set correctly if not original background
    if background != "original":
        saved_video_path = f"test_{background}.avi"
        cap = cv2.VideoCapture(saved_video_path)
        ret, frame = cap.read()

        # Calculate mean color of the channels
        b, g, r = cv2.split(frame)
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        # Set threshold values. Color is white if greater than white threshold, black
        # if less than grey threshold and grey if in between both threshold values.
        white_threshold = 240
        grey_threshold = 40

        # Check if the average color is white, grey, or black
        if all(val > white_threshold for val in [mean_b, mean_g, mean_r]):
            background_color = "white"
        elif all(val < grey_threshold for val in [mean_b, mean_g, mean_r]):
            background_color = "black"
        else:
            background_color = "grey"
        assert background_color == background


def test_sleap_render(centered_pair_predictions):
    args = (
        "-o testvis.avi -f 2 --scale 1.2 --frames 1,2 --video-index 0 "
        "tests/data/json_format_v2/centered_pair_predictions.json".split()
    )
    sleap_render(args)
    assert os.path.exists("testvis.avi")


@pytest.mark.parametrize("crop", ["Half", "Quarter", None])
def test_write_visuals(tmpdir, centered_pair_predictions: Labels, crop: str):
    video = centered_pair_predictions.videos[0]

    # Determine crop size relative to original size and scale
    crop_size_xy = None
    w = int(video.backend.width)
    h = int(video.backend.height)
    if crop == "Half":
        crop_size_xy = (w // 2, h // 2)
    elif crop == "Quarter":
        crop_size_xy = (w // 4, h // 4)

    path = os.path.join(tmpdir, "clip.avi")
    save_labeled_video(
        filename=path,
        labels=centered_pair_predictions,
        video=video,
        frames=(0, 1, 2),
        fps=15,
        edge_is_wedge=True,
        crop_size_xy=crop_size_xy,
    )
    assert os.path.exists(path)

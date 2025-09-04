import numpy as np
import os
import pytest
from sleap_io import Labels
from sleap.io.visuals import (
    save_labeled_video,
    VideoMarkerThread,
    main as sleap_render,
)
from sleap.util import resize_images


def test_serial_pipeline(centered_pair_predictions, tmpdir):
    frames = [0, 1, 2]
    video_idx = 0
    scale = 0.25

    video = centered_pair_predictions.videos[video_idx]
    frame_images = video[frames]

    # Make sure we can resize
    small_images = resize_images(frame_images, scale=scale)

    _, height, width, _ = small_images.shape

    assert height == video.shape[1] // (1 / scale)
    assert width == video.shape[2] // (1 / scale)

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


def test_sleap_render(centered_pair_predictions_slp_path, tmp_path):
    output_video = tmp_path / "testvis.avi"
    args = (
        f"-o {output_video} -f 2 --scale 1.2 --frames 1,2 --video-index 0 "
        f"{centered_pair_predictions_slp_path}".split()
    )
    sleap_render(args)
    assert output_video.exists()


@pytest.mark.parametrize("crop", ["Half", "Quarter", None])
def test_write_visuals(tmpdir, centered_pair_predictions: Labels, crop: str):
    video = centered_pair_predictions.videos[0]

    # Determine crop size relative to original size and scale
    crop_size_xy = None
    w = int(video.shape[2])
    h = int(video.shape[1])
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

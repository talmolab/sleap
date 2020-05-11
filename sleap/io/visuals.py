"""
Module for generating videos with visual annotation overlays.
"""

from sleap.io.video import Video
from sleap.io.videowriter import VideoWriter
from sleap.io.dataset import Labels
from sleap.util import usable_cpu_count

import cv2
import os
import numpy as np
import math
from time import time, clock
from typing import Iterable, List, Tuple

from queue import Queue
from threading import Thread

import logging

logger = logging.getLogger(__name__)

# Object that signals shutdown
_sentinel = object()


def reader(out_q: Queue, video: Video, frames: List[int], scale: float = 1.0):
    """Read frame images from video and send them into queue.

    Args:
        out_q: Queue to send (list of frame indexes, ndarray of frame images)
            for chunks of video.
        video: The `Video` object to read.
        frames: Full list frame indexes we want to read.
        scale: Output scale for frame images.

    Returns:
        None.
    """

    cv2.setNumThreads(usable_cpu_count())

    total_count = len(frames)
    chunk_size = 64
    chunk_count = math.ceil(total_count / chunk_size)

    logger.info(f"Chunks: {chunk_count}, chunk size: {chunk_size}")

    i = 0
    for chunk_i in range(chunk_count):

        # Read the next chunk of frames
        frame_start = chunk_size * chunk_i
        frame_end = min(frame_start + chunk_size, total_count)
        frames_idx_chunk = frames[frame_start:frame_end]

        t0 = clock()

        # Safely load frames from video, skipping frames we can't load
        loaded_chunk_idxs, video_frame_images = video.get_frames_safely(
            frames_idx_chunk
        )

        if scale != 1.0:
            video_frame_images = resize_images(video_frame_images, scale)

        elapsed = clock() - t0
        fps = len(loaded_chunk_idxs) / elapsed
        logger.debug(f"reading chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        out_q.put((loaded_chunk_idxs, video_frame_images))

    # send _sentinal object into queue to signal that we're done
    out_q.put(_sentinel)


def marker(in_q: Queue, out_q: Queue, labels: Labels, video_idx: int, scale: float):
    """Annotate frame images (draw instances).

    Args:
        in_q: Queue with (list of frame indexes, ndarray of frame images).
        out_q: Queue to send annotated images as
            (images, h, w, channels) ndarray.
        labels: the `Labels` object from which to get data for annotating.
        video_idx: index of `Video` in `labels.videos` list.

    Returns:
        None.
    """

    cv2.setNumThreads(usable_cpu_count())

    chunk_i = 0
    while True:
        data = in_q.get()

        if data is _sentinel:
            # no more data to be received so stop
            in_q.put(_sentinel)
            break

        frames_idx_chunk, video_frame_images = data

        t0 = clock()

        imgs = mark_images(
            frame_indices=frames_idx_chunk,
            frame_images=video_frame_images,
            video_idx=video_idx,
            labels=labels,
            scale=scale,
        )

        elapsed = clock() - t0
        fps = len(imgs) / elapsed
        logger.debug(f"drawing chunk {chunk_i} in {elapsed} s = {fps} fps")
        chunk_i += 1
        out_q.put(imgs)

    # send _sentinal object into queue to signal that we're done
    out_q.put(_sentinel)


def writer(
    in_q: Queue,
    progress_queue: Queue,
    filename: str,
    fps: float,
    img_w_h: Tuple[int, int],
):
    """Write annotated images to video.

    Args:
        in_q: Queue with annotated images as (images, h, w, channels) ndarray
        progress_queue: Queue to send progress as
            (total frames written: int, elapsed time: float).
            Send (-1, elapsed time) when done.
        filename: full path to output video
        fps: frames per second for output video
        img_w_h: (w, h) for output video (note width first for opencv)

    Returns:
        None.
    """

    cv2.setNumThreads(usable_cpu_count())

    w, h = img_w_h

    writer_object = VideoWriter.safe_builder(filename, height=h, width=w, fps=fps)

    start_time = clock()
    total_elapsed = 0
    total_frames_written = 0

    i = 0
    while True:
        data = in_q.get()

        if data is _sentinel:
            # no more data to be received so stop
            in_q.put(_sentinel)
            break

        t0 = clock()
        for img in data:
            writer_object.add_frame(img, bgr=True)

        elapsed = clock() - t0
        fps = len(data) / elapsed
        logger.debug(f"writing chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        total_frames_written += len(data)
        total_elapsed = clock() - start_time
        progress_queue.put((total_frames_written, total_elapsed))

    writer_object.close()
    # send (-1, time) to signal done
    progress_queue.put((-1, total_elapsed))


def save_labeled_video(
    filename: str,
    labels: Labels,
    video: Video,
    frames: Iterable[int],
    fps: int = 15,
    scale: float = 1.0,
    gui_progress: bool = False,
):
    """Function to generate and save video with annotations.

    Args:
        filename: Output filename.
        labels: The dataset from which to get data.
        video: The source :class:`Video` we want to annotate.
        frames: List of frames to include in output video.
        fps: Frames per second for output video.
        gui_progress: Whether to show Qt GUI progress dialog.

    Returns:
        None.
    """
    print(f"Writing video with {len(frames)} frame images...")

    output_width_height = (int(video.width * scale), int(video.height * scale))

    t0 = clock()

    q1 = Queue(maxsize=10)
    q2 = Queue(maxsize=10)
    progress_queue = Queue()

    thread_read = Thread(target=reader, args=(q1, video, frames, scale))
    thread_mark = Thread(
        target=marker, args=(q1, q2, labels, labels.videos.index(video), scale)
    )
    thread_write = Thread(
        target=writer, args=(q2, progress_queue, filename, fps, output_width_height),
    )

    thread_read.start()
    thread_mark.start()
    thread_write.start()

    progress_win = None
    if gui_progress:
        from PySide2 import QtWidgets, QtCore

        progress_win = QtWidgets.QProgressDialog(
            f"Generating video with {len(frames)} frames...", "Cancel", 0, len(frames)
        )
        progress_win.setMinimumWidth(300)
        progress_win.setWindowModality(QtCore.Qt.WindowModal)

    while True:
        frames_complete, elapsed = progress_queue.get()
        if frames_complete == -1:
            break
        if progress_win is not None and progress_win.wasCanceled():
            break
        fps = frames_complete / elapsed
        remaining_frames = len(frames) - frames_complete
        remaining_time = remaining_frames / fps

        if gui_progress:
            progress_win.setValue(frames_complete)
        else:
            print(
                f"Finished {frames_complete} frames in {elapsed} s, fps = {fps}, approx {remaining_time} s remaining"
            )

    elapsed = clock() - t0
    fps = len(frames) / elapsed
    print(f"Done in {elapsed} s, fps = {fps}.")


def mark_images(frame_indices, frame_images, video_idx, labels, scale):
    imgs = []
    for i, frame_idx in enumerate(frame_indices):
        img = get_frame_image(
            video_frame=frame_images[i],
            video_idx=video_idx,
            frame_idx=frame_idx,
            labels=labels,
            scale=scale,
        )

        imgs.append(img)
    return imgs


def get_frame_image(
    video_frame: np.ndarray,
    video_idx: int,
    frame_idx: int,
    labels: Labels,
    scale: float,
) -> np.ndarray:
    """Returns single annotated frame image.

    Args:
        video_frame: The ndarray of the frame image.
        video_idx: Index of video in :attribute:`Labels.videos` list.
        frame_idx: Index of frame in video.
        labels: The dataset from which to get data.

    Returns:
        ndarray of frame image with visual annotations added.
    """

    # Use OpenCV to convert to BGR color image
    video_frame = img_to_cv(video_frame)

    # Add the instances to the image
    plot_instances_cv(video_frame, video_idx, frame_idx, labels, scale)

    return video_frame


def img_to_cv(img: np.ndarray) -> np.ndarray:
    """Prepares frame image as needed for opencv."""
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert grayscale to BGR
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def plot_instances_cv(
    img: np.ndarray, video_idx: int, frame_idx: int, labels: Labels, scale: float
) -> np.ndarray:
    """Adds visuals annotations to single frame image.

    Args:
        img: The ndarray of the frame image.
        video_idx: Index of video in :attribute:`Labels.videos` list.
        frame_idx: Index of frame in video.
        labels: The dataset from which to get data.

    Returns:
        ndarray of frame image with visual annotations added.
    """
    cmap = [
        [0, 114, 189],
        [217, 83, 25],
        [237, 177, 32],
        [126, 47, 142],
        [119, 172, 48],
        [77, 190, 238],
        [162, 20, 47],
    ]
    lfs = labels.find(labels.videos[video_idx], frame_idx)

    if len(lfs) == 0:
        return

    count_no_track = 0
    for i, instance in enumerate(lfs[0].instances_to_show):
        if instance.track in labels.tracks:
            track_idx = labels.tracks.index(instance.track)
        else:
            # Instance without track
            track_idx = len(labels.tracks) + count_no_track
            count_no_track += 1

        # Get color for instance and convert RGB to BGR for OpenCV
        inst_color = cmap[track_idx % len(cmap)][::-1]

        plot_instance_cv(img, instance, inst_color, scale=scale)


def has_nans(*vals):
    return any((np.isnan(val) for val in vals))


def plot_instance_cv(
    img: np.ndarray,
    instance: "Instance",
    color: Iterable[int],
    unscaled_marker_radius: float = 4,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Add visual annotations for single instance.

    Args:
        img: The ndarray of the frame image.
        instance: The :class:`Instance` to add to frame image.
        color: (r, g, b) color for this instance.
        unscaled_marker_radius: Radius of marker for instance points (nodes).
        scale: 

    Returns:
        ndarray of frame image with visual annotations for instance added.
    """

    # Get matrix of all point locations
    points_array = instance.points_array

    # Rescale point locations
    points_array *= scale

    marker_radius = max(1, int(unscaled_marker_radius // (1 / scale)))

    for x, y in points_array:
        # Make sure this is a valid and visible point
        if not has_nans(x, y):
            # Convert to ints for opencv (now that we know these aren't nans)
            x, y = int(x), int(y)
            # Draw circle to mark node
            cv2.circle(
                img, (x, y), marker_radius, color, lineType=cv2.LINE_AA,
            )

    for (src, dst) in instance.skeleton.edge_inds:
        # Get points for the nodes connected by this edge
        src_x, src_y = points_array[src]
        dst_x, dst_y = points_array[dst]

        # Make sure that both nodes are present in this instance before drawing edge
        if not has_nans(src_x, src_y, dst_x, dst_y):

            # Convert to ints for opencv
            src_x, src_y = int(src_x), int(src_y)
            dst_x, dst_y = int(dst_x), int(dst_y)

            # Draw line to mark edge between nodes
            cv2.line(
                img, (src_x, src_y), (dst_x, dst_y), color, lineType=cv2.LINE_AA,
            )


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resizes single image with shape (height, width, channels)."""
    height, width, channels = img.shape
    new_height, new_width = int(height // (1 / scale)), int(width // (1 / scale))

    # Note that OpenCV takes shape as (width, height).

    if channels == 1:
        # opencv doesn't want a single channel to have its own dimension
        img = cv2.resize(img[:, :], (new_width, new_height))[..., None]
    else:
        img = cv2.resize(img, (new_width, new_height))

    return img


def resize_images(images: np.ndarray, scale: float) -> np.ndarray:
    return np.stack([resize_image(img, scale) for img in images])


def main_cli():
    import argparse
    from sleap.util import frame_list

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="The output filename for the video",
    )
    parser.add_argument("-f", "--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--scale", type=float, default=1.0, help="Output image scale")
    parser.add_argument(
        "--frames",
        type=frame_list,
        default="",
        help="list of frames to predict. Either comma separated list (e.g. 1,2,3) or "
        "a range separated by hyphen (e.g. 1-3). (default is entire video)",
    )
    parser.add_argument(
        "--video-index", type=int, default=0, help="Index of video in labels dataset"
    )
    args = parser.parse_args()

    labels = Labels.load_file(
        args.data_path, video_search=[os.path.dirname(args.data_path)]
    )

    if args.video_index >= len(labels.videos):
        raise IndexError(f"There is no video with index {args.video_index}.")

    vid = labels.videos[args.video_index]

    if args.frames is None:
        frames = sorted([lf.frame_idx for lf in labels if len(lf.instances)])
    else:
        frames = args.frames

    filename = args.output or args.data_path + ".avi"

    save_labeled_video(
        filename=filename,
        labels=labels,
        video=vid,
        frames=frames,
        fps=args.fps,
        scale=args.scale,
    )

    print(f"Video saved as: {filename}")


if __name__ == "__main__":
    main_cli()

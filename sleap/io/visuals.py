from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.util import usable_cpu_count

import cv2
import numpy as np
import math
from time import time, clock
from typing import List

from queue import Queue
from threading import Thread

import logging
logger = logging.getLogger(__name__)

# Object that signals shutdown
_sentinel = object()

def reader(out_q: Queue, video: Video, frames: List[int]):
    """Read frame images from video and send them into queue.

    Args:
        out_q: Queue to send (list of frame indexes, ndarray of frame images)
            for chunks of video.
        video: the `Video` object to read
        frames: full list frame indexes we want to read
    """

    cv2.setNumThreads(usable_cpu_count())

    total_count = len(frames)
    chunk_size = 64
    chunk_count = math.ceil(total_count/chunk_size)

    logger.info(f"Chunks: {chunk_count}, chunk size: {chunk_size}")

    i = 0
    for chunk_i in range(chunk_count):

        # Read the next chunk of frames
        frame_start = chunk_size * chunk_i
        frame_end = min(frame_start + chunk_size, total_count)
        frames_idx_chunk = frames[frame_start:frame_end]

        t0 = clock()

        # Load frames from video
        video_frame_images = video[frames_idx_chunk]

        elapsed = clock() - t0
        fps = len(frames_idx_chunk)/elapsed
        logger.debug(f"reading chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        out_q.put((frames_idx_chunk, video_frame_images))

    # send _sentinal object into queue to signal that we're done
    out_q.put(_sentinel)

def marker(in_q: Queue, out_q: Queue, labels: Labels):
    """Annotate frame images (draw instances).

    Args:
        in_q: Queue with (list of frame indexes, ndarray of frame images)
        out_q: Queue to send annotated images as (images, h, w, channels) ndarray
        labels: The `Labels` object from which to get data for annotating.

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
        imgs = []
        for i, frame_idx in enumerate(frames_idx_chunk):
            img = get_frame_image(
                        video_frame=video_frame_images[i],
                        frame_idx=frame_idx,
                        labels=labels)

            imgs.append(img)
        elapsed = clock() - t0
        fps = len(imgs)/elapsed
        logger.debug(f"drawing chunk {chunk_i} in {elapsed} s = {fps} fps")
        chunk_i += 1
        out_q.put(imgs)

    # send _sentinal object into queue to signal that we're done
    out_q.put(_sentinel)

def writer(in_q: Queue, progress_queue: Queue,
           filename: str, fps: int, img_w_h: tuple):
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

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, fps, img_w_h)

    start_time = clock()
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
            out.write(img)

        elapsed = clock() - t0
        fps = len(data)/elapsed
        logger.debug(f"writing chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        total_frames_written += len(data)
        total_elapsed = clock() - start_time
        progress_queue.put((total_frames_written, total_elapsed))

    out.release()
    # send (-1, time) to signal done
    progress_queue.put((-1, total_elapsed))

def save_labeled_video(
            filename: str,
            labels: Labels,
            video: Video,
            frames: List[int],
            fps: int=15,
            gui_progress: bool=False):
    """Function to generate and save video with annotations."""
    output_size = (video.height, video.width)

    print(f"Writing video with {len(frames)} frame images...")

    t0 = clock()

    q1 = Queue()
    q2 = Queue()
    progress_queue = Queue()

    thread_read = Thread(target=reader, args=(q1, video, frames,))
    thread_mark = Thread(target=marker, args=(q1, q2, labels,))
    thread_write = Thread(target=writer, args=(
                                            q2, progress_queue, filename,
                                            fps, (video.width, video.height),
                                            ))

    thread_read.start()
    thread_mark.start()
    thread_write.start()

    progress_win = None
    if gui_progress:
        from PySide2 import QtWidgets, QtCore

        progress_win = QtWidgets.QProgressDialog(
                            f"Generating video with {len(frames)} frames...",
                            "Cancel",
                            0, len(frames))
        progress_win.setMinimumWidth(300)
        progress_win.setWindowModality(QtCore.Qt.WindowModal)

    while True:
        frames_complete, elapsed = progress_queue.get()
        if frames_complete == -1:
            break
        if progress_win is not None and progress_win.wasCanceled():
            break
        fps = frames_complete/elapsed
        remaining_frames = len(frames) - frames_complete
        remaining_time = remaining_frames/fps

        if gui_progress:
            progress_win.setValue(frames_complete)
        else:
            print(f"Finished {frames_complete} frames in {elapsed} s, fps = {fps}, approx {remaining_time} s remaining")

    elapsed = clock() - t0
    fps = len(frames)/elapsed
    print(f"Done in {elapsed} s, fps = {fps}.")

def img_to_cv(img):
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert grayscale to BGR
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def get_frame_image(video_frame, frame_idx, labels):
    img = img_to_cv(video_frame)
    plot_instances_cv(img, frame_idx, labels)
    return img

def _point_int_tuple(point):
    return int(point.x), int(point.y)

def plot_instances_cv(img, frame_idx, labels):
    cmap = ([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ])
    lfs = [label for label in labels.labels if label.video == labels.videos[0] and label.frame_idx == frame_idx]

    if len(lfs) == 0: return

    count_no_track = 0
    for i, instance in enumerate(lfs[0].instances_to_show):
        if instance.track in labels.tracks:
            track_idx = labels.tracks.index(instance.track)
        else:
            # Instance without track
            track_idx = len(labels.tracks) + count_no_track
            count_no_track += 1

        inst_color = cmap[track_idx%len(cmap)]

        plot_instance_cv(img, instance, inst_color)

def plot_instance_cv(img, instance, color, marker_radius=4):

    # RGB -> BGR for cv2
    cv_color = color[::-1]

    for (node, point) in instance.nodes_points:
        # plot node at point
        if point.visible and not point.isnan():
            cv2.circle(img,
                    _point_int_tuple(point),
                    marker_radius,
                    cv_color,
                    lineType=cv2.LINE_AA)
    for (src, dst) in instance.skeleton.edges:
        # Make sure that both nodes are present in this instance before drawing edge
        if src in instance and dst in instance:
            if instance[src].visible and instance[dst].visible \
                and not instance[src].isnan() and not instance[dst].isnan():

                cv2.line(
                        img,
                        _point_int_tuple(instance[src]),
                        _point_int_tuple(instance[dst]),
                        cv_color,
                        lineType=cv2.LINE_AA)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='The output filename for the video')
    parser.add_argument('-f', '--fps', type=int, default=15,
                        help='Frames per second')
    args = parser.parse_args()

    labels = Labels.load_json(args.data_path)
    vid = labels.videos[0]

    frames = [lf.frame_idx for lf in labels if len(lf.instances)]

#     if len(frames) > 20:
#         frames = frames[:20]

    filename = args.output or args.data_path + ".avi"

    save_labeled_video(filename=filename,
                       labels=labels,
                       video=labels.videos[0],
                       frames=frames,
                       fps=args.fps)

    print(f"Video saved as: {filename}")

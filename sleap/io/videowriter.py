"""
Module for writing avi/mp4 videos.

Usage: ::

   > writer = VideoWriter.safe_builder(filename, height, width, fps)
   > writer.add_frame(img)
   > writer.close()

"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from time import perf_counter
from queue import Queue
from threading import Thread

import cv2
import logging
import numpy as np
import imageio.v2 as iio

from sleap.io.video import Video
from sleap.util import usable_cpu_count

logger = logging.getLogger(__name__)


# Object that signals shutdown
_sentinel = object()


class VideoWriter(ABC):
    """Abstract base class for writing avi/mp4 videos."""

    @abstractmethod
    def __init__(self, filename, height, width, fps):
        pass

    @abstractmethod
    def add_frame(self, img):
        pass

    @abstractmethod
    def close(self):
        pass

    @staticmethod
    def safe_builder(filename, height, width, fps):
        """Builds VideoWriter based on available dependencies."""
        if VideoWriter.can_use_ffmpeg():
            return VideoWriterImageio(filename, height, width, fps)
        else:
            return VideoWriterOpenCV(filename, height, width, fps)

    @staticmethod
    def can_use_ffmpeg():
        """Check if ffmpeg is available for writing videos."""
        try:
            import imageio_ffmpeg as ffmpeg
        except ImportError:
            return False

        try:
            # Try to get the version of the ffmpeg plugin
            ffmpeg_version = ffmpeg.get_ffmpeg_version()
            if ffmpeg_version:
                return True
        except Exception:
            return False

        return False


class VideoWriterOpenCV(VideoWriter):
    """Writes video using OpenCV as wrapper for ffmpeg."""

    def __init__(self, filename, height, width, fps):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def add_frame(self, img, bgr: bool = False):
        if not bgr and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._writer.write(img)

    def close(self):
        self._writer.release()


class VideoWriterImageio(VideoWriter):
    """Writes video using imageio as a wrapper for ffmpeg.

    Attributes:
        filename: Path to video file to save to.
        height: Height of movie frames.
        width: Width of movie frames.
        fps: Playback framerate to save at.
        crf: Compression rate factor to control lossiness of video. Values go from
            2 to 32, with numbers in the 18 to 30 range being most common. Lower values
            mean less compressed/higher quality.
        preset: Name of the libx264 preset to use (default: "superfast").
    """

    def __init__(
        self, filename, height, width, fps, crf: int = 21, preset: str = "superfast"
    ):
        self.filename = filename
        self.height = height
        self.width = width
        self.fps = fps
        self.crf = crf
        self.preset = preset

        # Imageio's ffmpeg writer parameters
        # https://imageio.readthedocs.io/en/stable/examples.html#writing-videos-with-ffmpeg-and-vaapi
        # Use `ffmpeg -h encoder=libx264`` to see all options for libx264 output_params
        # output_params must be a list of strings
        # iio.help(name='FFMPEG') to test
        self.writer = iio.get_writer(
            filename,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            output_params=[
                "-preset",
                preset,
                "-crf",
                str(crf),
            ],
        )

    def add_frame(self, img, bgr: bool = False):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.writer.append_data(img)

    def close(self):
        self.writer.close()


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
    if scale == 1.0:
        return images
    return np.stack([resize_image(img, scale) for img in images])


def augment_background(images: np.ndarray, background: str | None) -> np.ndarray:
    """Augment the background of the video frame images.

    Args:
        images: The ndarray of the frame images.
        background: output video background. Either original, black, white, grey

    Raises:
        ValueError: If the background color is not one of the specified options.

    Returns:
        ndarray of frame images with background augmented.
    """
    if background is None or background == "original":
        return images

    # Get color to fill.
    fill_values = {"black": 0, "grey": 127, "white": 255}
    if background not in fill_values:
        raise ValueError(
            f"Invalid background color: {background}. "
            f"Options include: {', '.join(fill_values.keys())}"
        )

    # Fill the images with the specified color.
    fill = fill_values[background]
    images = images * 0 + fill
    return images


def reader(
    out_q: Queue[tuple[list[int], np.ndarray]],
    video: Video,
    frames: list[int],
    scale: float = 1.0,
    background: str | None = None,
):
    """Read frame images from video and send them into queue.

    Args:
        out_q: Queue to send (list of frame indexes, ndarray of frame images) for chunks
            of video.
        video: The `Video` object to read.
        frames: Full list frame indexes we want to read.
        scale: Output scale for frame images.
        background: Output video background. Either "original", "black", "white",
            "grey", or None. If None, the original background is used.

    Returns:
        None.
    """
    # TODO: Handle case where exception is raised in reader thread - currently stalls.
    background = background.lower() if background is not None else background
    cv2.setNumThreads(usable_cpu_count())

    total_count = len(frames)
    chunk_size = 64
    chunk_count = math.ceil(total_count / chunk_size)

    try:
        logger.info(f"Chunks: {chunk_count}, chunk size: {chunk_size}")
        i = 0
        for chunk_i in range(chunk_count):
            # Read the next chunk of frames
            frame_start = chunk_size * chunk_i
            frame_end = min(frame_start + chunk_size, total_count)
            frames_idx_chunk = frames[frame_start:frame_end]

            # Timer start.
            t0 = perf_counter()

            # Safely load frames from video, skipping frames we can't load
            loaded_chunk_idxs, video_frame_images = video.get_frames_safely(
                frames_idx_chunk
            )
            if not loaded_chunk_idxs:
                print(f"No frames could be loaded from chunk {chunk_i}")
                i += 1
                continue

            # Adjust images.
            video_frame_images = augment_background(
                images=video_frame_images, background=background
            )
            video_frame_images = resize_images(images=video_frame_images, scale=scale)

            # Timer end.
            elapsed = perf_counter() - t0
            fps = len(loaded_chunk_idxs) / elapsed
            logger.debug(f"Reading chunk {i} in {elapsed} s = {fps} fps")

            # Send chunk of images into queue
            i += 1
            out_q.put((loaded_chunk_idxs, video_frame_images))

    except Exception as e:
        raise e

    finally:
        # send _sentinal object into queue to signal that we're done
        out_q.put(_sentinel)


def writer(
    in_q: Queue[np.ndarray],
    progress_queue: Queue[tuple[int, float]],
    filename: str,
    fps: float,
):
    """Write images to video.

    Image size is determined by the first image received in queue.

    Args:
        in_q: Queue with images as numpy array of shape (images, h, w, channels).
        progress_queue: Queue to send progress as a tuple of (total frames written: int,
            elapsed time: float). Send (-1, elapsed time) when done.
        filename: Full path to output video.
        fps: Frames per second for output video.

    Returns:
        None.
    """

    cv2.setNumThreads(usable_cpu_count())

    writer_object = None
    total_elapsed = 0
    total_frames_written = 0
    start_time = perf_counter()
    try:
        i = 0
        while True:
            # Retrieve chunk of images from queue.
            data = in_q.get()
            if data is _sentinel:
                # No more data to be received so stop.
                in_q.put(_sentinel)
                break

            # Initialize writer object if not already done.
            if writer_object is None and len(data) > 0:
                h, w = data[0].shape[:2]
                writer_object = VideoWriter.safe_builder(
                    filename, height=h, width=w, fps=fps
                )

            # Start elapsed time.
            t0 = perf_counter()

            # Write chunk of images to video.
            for img in data:
                writer_object.add_frame(img, bgr=True)

            # End elapsed time.
            elapsed = perf_counter() - t0
            fps = len(data) / elapsed
            logger.debug(f"writing chunk {i} in {elapsed} s = {fps} fps")
            i += 1

            # Send progress to queue.
            total_frames_written += len(data)
            total_elapsed = perf_counter() - start_time
            progress_queue.put((total_frames_written, total_elapsed))

    except Exception as e:
        # Stop receiving data
        in_q.put(_sentinel)
        raise e

    finally:
        if writer_object is not None:
            writer_object.close()
        # Send (-1, time) to signal done
        progress_queue.put((-1, total_elapsed))


def progress_feedback(
    progress_queue: Queue,
    frames: list[int],
    gui_progress: bool = False,
):
    """Function to monitor progress of video generation.
    Args:
        progress_queue: Queue to get progress updates from.
        frames: List of frames to include in output video.
        gui_progress: Whether to show Qt GUI progress dialog. If False, print to
            console.

    Returns:
        None.
    """
    progress_win = None
    if gui_progress:
        from qtpy import QtCore, QtWidgets

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
                f"Finished {frames_complete} frames in {elapsed:.1f} s, "
                f"fps = {round(fps)}, approx {remaining_time:.1f} s remaining"
            )


def default_intermediate_target(
    in_queue: Queue[tuple[list[int], np.ndarray]], out_queue: Queue[np.ndarray]
):
    """Default target for intermediate threads.

    This function passes the frames from the input queue to the output queue without
    any processing. It is used as a placeholder when no intermediate threads are
    provided.

    Args:
        in_queue: Queue to receive frames from reader thread. Contains tuples of
            (list of frame indexes, ndarray of frame images).
        out_queue: Queue to send frames to writer thread. Contains ndarray of frame
            images.

    Returns:
        None
    """
    cv2.setNumThreads(usable_cpu_count())
    try:
        chunk_i = 0
        while True:
            # Retrieve chunk of images from queue.
            data = in_queue.get()
            if data is _sentinel:
                # No more data to be received so stop.
                in_queue.put(_sentinel)
                break

            frame_idx_chunk, images = data

            # Send chunk of images into output queue.
            chunk_i += 1
            out_queue.put(images)
    except Exception as e:
        # Stop receiving data
        in_queue.put(_sentinel)
        raise e
    finally:
        # Send _sentinal object into queue to signal that we're done
        out_queue.put(_sentinel)


def write_video(
    filename: str,
    video: Video,
    frames: list[int],
    fps: int = 15,
    scale: float = 1.0,
    background: str | None = None,
    gui_progress: bool = False,
    in_queue: Queue[tuple[list[int], np.ndarray]] | None = None,
    out_queue: Queue[np.ndarray] | None = None,
    intermediate_threads: list[Thread] | None = None,
):
    """Function to generate and save augmented `video`.

    Args:
        filename: Output filename.
        video: The source `Video` we want to augment.
        frames: List of frame indices of the original `video` to include in output
            video.
        fps: Frames per second for output video.
        scale: Scale of output video. 1.0 means no scaling, 0.5 means half size, etc.
        background: Output video background. Either "original", "black", "white",
            "grey", or None. If None, the original background is used.
        gui_progress: Whether to show Qt GUI progress dialog.
        in_queue: Queue to receive frames from reader thread. Contains tuples of
            (list of frame indexes, ndarray of frame images).
        out_queue: Queue to send frames to writer thread.
        intermediate_threads: List of threads to run in between reader and writer
            threads. These threads should consume the frames from the reader thread (via
            `in_queue`) and produce frames for the writer thread (via `out_queue`). This
            function will start the threads and wait for them to finish before closing
            the writer thread. If None, a default thread will be created that passes the
            frames from the reader thread to the writer thread without any processing.

    Returns:
        None.
    """
    print(f"Writing video with {len(frames)} frame images...")

    t0 = perf_counter()

    # Create queues for reading, writing, and progress feedback.
    q1 = in_queue or Queue(maxsize=10)
    q2 = out_queue or Queue(maxsize=10)
    progress_queue = Queue()

    # Create threads for reading and writing.
    thread_read = Thread(target=reader, args=(q1, video, frames, scale, background))
    thread_write = Thread(
        target=writer,
        args=(q2, progress_queue, filename, fps),
    )

    # Start the reader thread.
    thread_read.start()

    # If no intermediate threads are provided, create a thread to pass input to output.
    if intermediate_threads is None:
        intermediate_thread = Thread(
            target=default_intermediate_target,
            args=(q1, q2),
        )
        intermediate_threads = [intermediate_thread]

    # Start the intermediate threads.
    for thread in intermediate_threads:
        thread.start()

    # Start the writer thread.
    thread_write.start()

    # Wait for threads to finish.
    progress_feedback(
        progress_queue=progress_queue,
        frames=frames,
        gui_progress=gui_progress,
    )

    # We're done!
    elapsed = perf_counter() - t0
    fps = len(frames) / elapsed
    print(f"Done in {elapsed} s, fps = {fps}.")

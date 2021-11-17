"""
Module for generating videos with visual annotation overlays.
"""

from sleap.io.video import Video
from sleap.io.videowriter import VideoWriter
from sleap.io.dataset import Labels
from sleap.gui.color import ColorManager
from sleap.util import usable_cpu_count

import cv2
import os
import numpy as np
import math
from collections import deque
from time import perf_counter
from typing import List, Optional, Tuple

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

        t0 = perf_counter()

        # Safely load frames from video, skipping frames we can't load
        loaded_chunk_idxs, video_frame_images = video.get_frames_safely(
            frames_idx_chunk
        )

        if not loaded_chunk_idxs:
            print(f"No frames could be loaded from chunk {chunk_i}")
            i += 1
            continue

        if scale != 1.0:
            video_frame_images = resize_images(video_frame_images, scale)

        elapsed = perf_counter() - t0
        fps = len(loaded_chunk_idxs) / elapsed
        logger.debug(f"reading chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        out_q.put((loaded_chunk_idxs, video_frame_images))

    # send _sentinal object into queue to signal that we're done
    out_q.put(_sentinel)


def writer(
    in_q: Queue,
    progress_queue: Queue,
    filename: str,
    fps: float,
):
    """Write annotated images to video.

    Image size is determined by the first image received in queue.

    Args:
        in_q: Queue with annotated images as (images, h, w, channels) ndarray
        progress_queue: Queue to send progress as
            (total frames written: int, elapsed time: float).
            Send (-1, elapsed time) when done.
        filename: full path to output video
        fps: frames per second for output video

    Returns:
        None.
    """

    cv2.setNumThreads(usable_cpu_count())

    writer_object = None
    total_elapsed = 0
    total_frames_written = 0
    start_time = perf_counter()
    i = 0
    while True:
        data = in_q.get()

        if data is _sentinel:
            # no more data to be received so stop
            in_q.put(_sentinel)
            break

        if writer_object is None and data:
            h, w = data[0].shape[:2]
            writer_object = VideoWriter.safe_builder(
                filename, height=h, width=w, fps=fps
            )

        t0 = perf_counter()
        for img in data:
            writer_object.add_frame(img, bgr=True)

        elapsed = perf_counter() - t0
        fps = len(data) / elapsed
        logger.debug(f"writing chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        total_frames_written += len(data)
        total_elapsed = perf_counter() - start_time
        progress_queue.put((total_frames_written, total_elapsed))

    writer_object.close()
    # send (-1, time) to signal done
    progress_queue.put((-1, total_elapsed))


class VideoMarkerThread(Thread):
    """Annotate frame images (draw instances).

    Args:
        in_q: Queue with (list of frame indexes, ndarray of frame images).
        out_q: Queue to send annotated images as
            (images, h, w, channels) ndarray.
        labels: the `Labels` object from which to get data for annotating.
        video_idx: index of `Video` in `labels.videos` list.
        scale: scale of image (so we can scale point locations to match)
        show_edges: whether to draw lines between nodes
        color_manager: ColorManager object which determine what colors to use
            for what instance/node/edge
    """

    def __init__(
        self,
        in_q: Queue,
        out_q: Queue,
        labels: Labels,
        video_idx: int,
        scale: float,
        show_edges: bool = True,
        crop_size_xy: Optional[Tuple[int, int]] = None,
        color_manager: Optional[ColorManager] = None,
    ):
        super(VideoMarkerThread, self).__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.labels = labels
        self.video_idx = video_idx
        self.scale = scale
        self.show_edges = show_edges

        if color_manager is None:
            color_manager = ColorManager(labels=labels)
            color_manager.color_predicted = True

        self.color_manager = color_manager

        self.node_line_width = self.color_manager.get_item_type_pen_width("node")
        self.edge_line_width = self.color_manager.get_item_type_pen_width("edge")

        # fixme: these widths are based on *screen* pixels, so we'll adjust
        #  them since we want *video* pixels.
        self.node_line_width = max(1, self.node_line_width // 2)
        self.edge_line_width = max(1, self.node_line_width // 2)

        unscaled_marker_radius = 3
        self.marker_radius = max(1, int(unscaled_marker_radius // (1 / scale)))

        self.edge_line_width *= 2
        self.marker_radius *= 2
        self.alpha = 0.6

        self.crop = False
        if crop_size_xy:
            self.crop = True
            self.crop_w, self.crop_h = crop_size_xy
            self._crop_centers = deque(maxlen=5)  # use running avg for smoother crops
        else:
            self.crop_h = 0
            self.crop_w = 0
            self._crop_centers = []

    def run(self):
        # when thread starts, start loop to receive images (from reader),
        # draw things on the images, and pass them along (to writer)
        self.marker()

    def marker(self):
        cv2.setNumThreads(usable_cpu_count())

        chunk_i = 0
        while True:
            data = self.in_q.get()

            if data is _sentinel:
                # no more data to be received so stop
                self.in_q.put(_sentinel)
                break

            frames_idx_chunk, video_frame_images = data

            t0 = perf_counter()

            imgs = self._mark_images(
                frame_indices=frames_idx_chunk,
                frame_images=video_frame_images,
            )

            elapsed = perf_counter() - t0
            fps = len(imgs) / elapsed
            logger.debug(f"drawing chunk {chunk_i} in {elapsed} s = {fps} fps")
            chunk_i += 1
            self.out_q.put(imgs)

        # send _sentinal object into queue to signal that we're done
        self.out_q.put(_sentinel)

    def _mark_images(self, frame_indices, frame_images):
        imgs = []
        for i, frame_idx in enumerate(frame_indices):
            img = self._mark_single_frame(
                video_frame=frame_images[i], frame_idx=frame_idx
            )

            imgs.append(img)
        return imgs

    def _mark_single_frame(self, video_frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Returns single annotated frame image.

        Args:
            video_frame: The ndarray of the frame image.
            frame_idx: Index of frame in video.

        Returns:
            ndarray of frame image with visual annotations added.
        """

        # Use OpenCV to convert to BGR color image
        video_frame = img_to_cv(video_frame)

        # Add the instances to the image
        overlay = self._plot_instances_cv(video_frame.copy(), frame_idx)

        return cv2.addWeighted(overlay, self.alpha, video_frame, 1 - self.alpha, 0)

    def _plot_instances_cv(
        self,
        img: np.ndarray,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """Adds visuals annotations to single frame image.

        Args:
            img: The ndarray of the frame image.
            frame_idx: Index of frame in video.

        Returns:
            ndarray of frame image with visual annotations added.
        """
        labels = self.labels
        video_idx = self.video_idx

        lfs = labels.find(labels.videos[video_idx], frame_idx)

        if len(lfs) == 0:
            return self._crop_frame(img) if self.crop else img

        instances = lfs[0].instances_to_show

        offset = None
        if self.crop:
            img, offset = self._crop_frame(img, instances)

        for instance in instances:
            self._plot_instance_cv(img, instance, offset)

        return img

    def _get_crop_center(
        self, img: np.ndarray, instances: Optional[List["Instance"]] = None
    ) -> Tuple[int, int]:
        if instances:
            centroids = np.array([inst.centroid for inst in instances])
            center_xy = np.median(centroids, axis=0)

            self._crop_centers.append(center_xy)
        elif not self._crop_centers:
            # no crops so far and no instances yet so just use image center
            img_w, img_h = img.shape[:2]
            center_xy = img_w // 2, img_h // 2

            self._crop_centers.append(center_xy)

        # use a running average of the last N centers to smooth movement
        center_xy = tuple(np.mean(np.stack(self._crop_centers), axis=0))

        return center_xy

    def _crop_frame(
        self, img: np.ndarray, instances: Optional[List["Instance"]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        center_xy = self._get_crop_center(img, instances)
        return self._crop_img(img, center_xy)

    def _crop_img(
        self, img: np.ndarray, center_xy: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        img_w, img_h = img.shape[:2]  # fixme?
        center_x, center_y = center_xy

        # Adjust center (on original coordinates) to scaled image coordinages
        center_x = center_x // (1 / self.scale)
        center_y = center_y // (1 / self.scale)

        # Find center, ensuring we're within top/left bounds for image
        crop_x0 = max(0, int(center_x - self.crop_w // 2))
        crop_y0 = max(0, int(center_y - self.crop_h // 2))

        # And ensure that we're within bottom/right bounds for image
        if crop_x0 + self.crop_w > img_w:
            crop_x0 = img_w - self.crop_w
        if crop_y0 + self.crop_h > img_h:
            crop_y0 = img_h - self.crop_h

        offset = crop_x0, crop_y0
        crop_x1 = crop_x0 + self.crop_w
        crop_y1 = crop_y0 + self.crop_h

        img = img[crop_y0:crop_y1, crop_x0:crop_x1, ...]

        return img, offset

    def _plot_instance_cv(
        self,
        img: np.ndarray,
        instance: "Instance",
        offset: Optional[Tuple[int, int]] = None,
        fill: bool = True,
    ):
        """
        Add visual annotations for single instance.

        Args:
            img: The ndarray of the frame image.
            instance: The :class:`Instance` to add to frame image.

        Returns:
            None; modifies img in place.
        """

        scale = self.scale
        nodes = instance.skeleton.nodes

        # Get matrix of all point locations
        points_array = instance.points_array

        # Rescale point locations
        points_array *= scale

        # Shift point locations (offset is for *scaled* coordinates)
        if offset:
            points_array -= offset

        for node_idx, (x, y) in enumerate(points_array):

            node = nodes[node_idx]
            node_color_bgr = self.color_manager.get_item_color(node, instance)[::-1]

            # Make sure this is a valid and visible point
            if not has_nans(x, y):
                # Convert to ints for opencv (now that we know these aren't nans)
                x, y = int(x), int(y)

                # Draw circle to mark node
                cv2.circle(
                    img=img,
                    center=(x, y),
                    radius=int(self.marker_radius),
                    color=node_color_bgr,
                    thickness=cv2.FILLED if fill else self.node_line_width,
                    lineType=cv2.FILLED if fill else cv2.LINE_AA,
                )

        if self.show_edges:
            for (src, dst) in instance.skeleton.edge_inds:
                # Get points for the nodes connected by this edge
                src_x, src_y = points_array[src]
                dst_x, dst_y = points_array[dst]

                edge = (nodes[src], nodes[dst])
                edge_color_bgr = self.color_manager.get_item_color(edge, instance)[::-1]

                # Make sure that both nodes are present in this instance before drawing edge
                if not has_nans(src_x, src_y, dst_x, dst_y):

                    # Convert to ints for opencv
                    src_x, src_y = int(src_x), int(src_y)
                    dst_x, dst_y = int(dst_x), int(dst_y)

                    # Draw line to mark edge between nodes
                    cv2.line(
                        img=img,
                        pt1=(src_x, src_y),
                        pt2=(dst_x, dst_y),
                        color=edge_color_bgr,
                        thickness=int(self.edge_line_width),
                        lineType=cv2.LINE_AA,
                    )


def save_labeled_video(
    filename: str,
    labels: Labels,
    video: Video,
    frames: List[int],
    fps: int = 15,
    scale: float = 1.0,
    crop_size_xy: Optional[Tuple[int, int]] = None,
    show_edges: bool = True,
    color_manager: Optional[ColorManager] = None,
    gui_progress: bool = False,
):
    """Function to generate and save video with annotations.

    Args:
        filename: Output filename.
        labels: The dataset from which to get data.
        video: The source :class:`Video` we want to annotate.
        frames: List of frames to include in output video.
        fps: Frames per second for output video.
        scale: scale of image (so we can scale point locations to match)
        crop_size_xy: size of crop around instances, or None for full images
        show_edges: whether to draw lines between nodes
        color_manager: ColorManager object which determine what colors to use
            for what instance/node/edge
        gui_progress: Whether to show Qt GUI progress dialog.

    Returns:
        None.
    """
    print(f"Writing video with {len(frames)} frame images...")

    t0 = perf_counter()

    q1 = Queue(maxsize=10)
    q2 = Queue(maxsize=10)
    progress_queue = Queue()

    thread_read = Thread(target=reader, args=(q1, video, frames, scale))
    thread_mark = VideoMarkerThread(
        in_q=q1,
        out_q=q2,
        labels=labels,
        video_idx=labels.videos.index(video),
        scale=scale,
        show_edges=show_edges,
        crop_size_xy=crop_size_xy,
        color_manager=color_manager,
    )
    thread_write = Thread(
        target=writer,
        args=(q2, progress_queue, filename, fps),
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

    elapsed = perf_counter() - t0
    fps = len(frames) / elapsed
    print(f"Done in {elapsed} s, fps = {fps}.")


def has_nans(*vals):
    return any((np.isnan(val) for val in vals))


def img_to_cv(img: np.ndarray) -> np.ndarray:
    """Prepares frame image as needed for opencv."""
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert grayscale to BGR
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


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
    parser.add_argument("-f", "--fps", type=int, default=25, help="Frames per second")
    parser.add_argument("--scale", type=float, default=1.0, help="Output image scale")
    parser.add_argument(
        "--crop", type=str, default="", help="Crop size as <width>,<height>"
    )
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

    try:
        crop_size_xy = list(map(int, args.crop.split(",")))
    except:
        crop_size_xy = None

    save_labeled_video(
        filename=filename,
        labels=labels,
        video=vid,
        frames=frames,
        fps=args.fps,
        scale=args.scale,
        crop_size_xy=crop_size_xy,
    )

    print(f"Video saved as: {filename}")


if __name__ == "__main__":
    main_cli()

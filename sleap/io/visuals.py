"""
Module for generating videos with visual annotation overlays.

.. deprecated::
    This module is deprecated. Use ``sleap_io.render_video()`` and
    ``sleap_io.render_image()`` instead for improved rendering with
    more customization options. See the sleap-io documentation for details.

The legacy rendering code in this module is maintained for backwards
compatibility but will be removed in a future release.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from queue import Queue, Empty
from threading import Thread
from time import perf_counter
from typing import List, Optional, Tuple
import os

import cv2
import numpy as np

from sleap.gui.color import ColorManager
from sleap_io.model.instance import Instance
from sleap_io import Video, Labels, LabeledFrame
from sleap.sleap_io_adaptors.video_utils import _sentinel
from sleap.sleap_io_adaptors.lf_labels_utils import (
    load_labels_video_search,
    get_instances_to_show,
)
from sleap_io import save_video
from sleap.util import usable_cpu_count

logger = logging.getLogger(__name__)


class VideoMarkerThread(Thread):
    """Annotate frame images (draw instances).

    .. deprecated::
        Use ``sleap_io.render_video()`` instead for improved rendering.

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
        edge_is_wedge: bool = False,
        marker_size: int = 4,
        crop_size_xy: Optional[Tuple[int, int]] = None,
        color_manager: Optional[ColorManager] = None,
        palette: str = "standard",
        distinctly_color: str = "instances",
    ):
        warnings.warn(
            "VideoMarkerThread is deprecated. Use sleap_io.render_video() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super(VideoMarkerThread, self).__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.labels = labels
        self.video_idx = video_idx
        self.scale = scale
        self.show_edges = show_edges
        self.edge_is_wedge = edge_is_wedge
        self.exception = None  # Store any exception that occurs in the thread

        if color_manager is None:
            color_manager = ColorManager(labels=labels, palette=palette)
            color_manager.color_predicted = True
            color_manager.distinctly_color = distinctly_color

        self.color_manager = color_manager

        self.node_line_width = self.color_manager.get_item_type_pen_width("node")
        self.edge_line_width = self.color_manager.get_item_type_pen_width("edge")

        # fixme: these widths are based on *screen* pixels, so we'll adjust
        #  them since we want *video* pixels.
        self.node_line_width = max(1, self.node_line_width // 2)
        self.edge_line_width = max(1, self.node_line_width // 2)

        self.marker_radius = max(1, int(marker_size // (1 / scale)))

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

        try:
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
        except Exception as e:
            # Store exception for main thread to check
            self.exception = e
            # Stop receiving data
            self.in_q.put(_sentinel)

        finally:
            # Send _sentinel object into queue to signal that we're done
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
        """Return single annotated frame image.

        Args:
            video_frame: The ndarray of the frame image.
            frame_idx: Index of frame in video.

        Returns:
            ndarray of frame image with visual annotations added (in RGB format).
        """
        # Use OpenCV to convert to BGR color image
        video_frame = img_to_cv(video_frame)

        # Add the instances to the image
        overlay = self._plot_instances_cv(video_frame.copy(), frame_idx)

        # Crop video_frame to same size as overlay
        video_frame_cropped = (
            self._crop_frame(video_frame.copy())[0] if self.crop else video_frame
        )

        result = cv2.addWeighted(
            overlay, self.alpha, video_frame_cropped, 1 - self.alpha, 0
        )

        # Convert back to RGB for imageio/FFMPEG writes (which expect RGB, not BGR)
        if result.shape[-1] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result

    def _plot_instances_cv(
        self,
        img: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Add visual annotations to single frame image.

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
            return self._crop_frame(img)[0] if self.crop else img

        instances = get_instances_to_show(lfs[0])

        offset = None
        if self.crop:
            img, offset = self._crop_frame(img, instances)

        for instance in instances:
            self._plot_instance_cv(img, instance, offset, frame=lfs[0])

        return img

    def _get_crop_center(
        self, img: np.ndarray, instances: Optional[List["Instance"]] = None
    ) -> Tuple[int, int]:
        if instances:
            centroids = np.array(
                [np.nanmedian(inst.numpy(), axis=0) for inst in instances]
            )
            center_xy = np.nanmedian(centroids, axis=0)
            self._crop_centers.append(center_xy)

        elif not self._crop_centers:
            # no crops so far and no instances yet so just use image center
            img_w, img_h = img.shape[:2]
            center_xy = img_w // 2, img_h // 2

            self._crop_centers.append(center_xy)

        # use a running average of the last N centers to smooth movement
        center_xy = tuple(np.nanmean(np.stack(self._crop_centers), axis=0))

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
        frame: Optional[LabeledFrame] = None,
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
        from sleap.sleap_io_adaptors.instance_utils import instance_get_points_array

        points_array = instance_get_points_array(instance)

        # Rescale point locations
        points_array *= scale

        # Shift point locations (offset is for *scaled* coordinates)
        if offset:
            points_array -= offset

        for node_idx, (x, y) in enumerate(points_array):
            node = nodes[node_idx]
            node_color_bgr = self.color_manager.get_item_color(
                node, instance, frame=frame
            )[::-1]

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
            for src, dst in instance.skeleton.edge_inds:
                # Get points for the nodes connected by this edge
                src_x, src_y = points_array[src]
                dst_x, dst_y = points_array[dst]

                edge = (nodes[src], nodes[dst])
                edge_color_bgr = self.color_manager.get_item_color(
                    edge, instance, frame=frame
                )[::-1]

                # Make sure that both nodes are present in this instance before
                # drawing edge
                if not has_nans(src_x, src_y, dst_x, dst_y):
                    # Convert to ints for opencv
                    src_x, src_y = int(src_x), int(src_y)
                    dst_x, dst_y = int(dst_x), int(dst_y)

                    if self.edge_is_wedge:
                        r = self.marker_radius / 2

                        # Get vector from source to destination
                        vec_x = dst_x - src_x
                        vec_y = dst_y - src_y
                        mag = (pow(vec_x, 2) + pow(vec_y, 2)) ** 0.5
                        vec_x = int(r * vec_x / mag)
                        vec_y = int(r * vec_y / mag)

                        # Define the wedge
                        src_1 = [src_x - vec_y, src_y + vec_x]
                        dst = [dst_x, dst_y]
                        src_2 = [src_x + vec_y, src_y - vec_x]
                        pts = np.array([src_1, dst, src_2])

                        # Draw the wedge
                        cv2.fillPoly(
                            img=img,
                            pts=[pts],
                            color=edge_color_bgr,
                            lineType=cv2.LINE_AA,
                        )

                    else:
                        # Draw line to mark edge between nodes
                        cv2.line(
                            img=img,
                            pt1=(src_x, src_y),
                            pt2=(dst_x, dst_y),
                            color=edge_color_bgr,
                            thickness=int(self.edge_line_width),
                            lineType=cv2.LINE_AA,
                        )


class VideoReaderThread(Thread):
    """Thread for reading video frames without blocking the main thread.

    Args:
        video: The video to read frames from.
        frames: List of frame indices to read.
        out_q: Queue to send (frame_indices, frame_images) tuples.
        chunk_size: Number of frames to read per chunk.
    """

    def __init__(
        self,
        video: Video,
        frames: list[int],
        out_q: Queue,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.video = video
        self.frames = frames
        self.out_q = out_q
        self.chunk_size = chunk_size
        self.exception = None

    def run(self):
        try:
            for i0 in range(0, len(self.frames), self.chunk_size):
                i1 = min(i0 + self.chunk_size, len(self.frames))
                frame_inds = self.frames[i0:i1]
                frame_imgs = self.video[frame_inds]
                self.out_q.put((frame_inds, frame_imgs))
        except Exception as e:
            self.exception = e
        finally:
            self.out_q.put(_sentinel)


def save_labeled_video(
    filename: str,
    labels: Labels,
    video: Video,
    frames: list[int],
    fps: int = 15,
    scale: float = 1.0,
    crop_size_xy: tuple[int, int] | None = None,
    background: str = "original",
    show_edges: bool = True,
    edge_is_wedge: bool = False,
    marker_size: int = 4,
    color_manager: ColorManager | None = None,
    palette: str = "standard",
    distinctly_color: str = "instances",
    gui_progress: bool = False,
    chunk_size: int = 64,
):
    """Function to generate and save video with annotations.

    .. deprecated::
        Use ``sleap_io.render_video()`` instead for improved rendering with
        more customization options including color schemes, marker shapes,
        and real-time preview support.

        Example migration::

            import sleap_io as sio

            # Old way (deprecated):
            save_labeled_video(filename, labels, video, frames, fps=30)

            # New way:
            sio.render_video(
                labels,
                filename,
                video=video,
                frame_inds=frames,
                fps=30,
                color_by="track",
                palette="tableau10",
            )

    Args:
        filename: Output filename.
        labels: The dataset from which to get data.
        video: The source :class:`Video` we want to annotate.
        frames: List of frames to include in output video.
        fps: Frames per second for output video.
        scale: scale of image (so we can scale point locations to match)
        crop_size_xy: size of crop around instances, or None for full images
        background: output video background. Either original, black, white, grey
        show_edges: whether to draw lines between nodes
        edge_is_wedge: whether to draw edges as wedges (draw as line if False)
        marker_size: Size of marker in pixels before scaling by `scale`
        color_manager: ColorManager object which determine what colors to use
            for what instance/node/edge
        palette: SLEAP color palette to use. Options include: "alphabet", "five+",
            "solarized", or "standard". Only used if `color_manager` is None.
        distinctly_color: Specify how to color instances. Options include: "instances",
            "edges", and "nodes". Only used if `color_manager` is None.
        gui_progress: Whether to show Qt GUI progress dialog.

    Returns:
        None.
    """
    warnings.warn(
        "save_labeled_video() is deprecated. Use sleap_io.render_video() instead "
        "for improved rendering with more customization options.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Set up GUI progress dialog if requested
    progress_win = None
    canceled = False
    if gui_progress:
        try:
            from qtpy import QtWidgets

            progress_win = QtWidgets.QProgressDialog(
                "Exporting labeled video...", "Cancel", 0, len(frames)
            )
            progress_win.setWindowTitle("Export Progress")
            progress_win.setMinimumDuration(0)
            progress_win.setValue(0)
            progress_win.show()
            QtWidgets.QApplication.instance().processEvents()
        except Exception:
            # If Qt is not available, continue without progress dialog
            progress_win = None

    # Create queues for thread communication
    reader_q = Queue(maxsize=10)  # Reader -> Marker
    marker_q = Queue(maxsize=10)  # Marker -> Main

    # Start video reader thread
    reader_thread = VideoReaderThread(
        video=video,
        frames=frames,
        out_q=reader_q,
        chunk_size=chunk_size,
    )
    reader_thread.start()

    # Start marker thread
    marker_thread = VideoMarkerThread(
        in_q=reader_q,
        out_q=marker_q,
        labels=labels,
        video_idx=labels.videos.index(video),
        scale=scale,
        show_edges=show_edges,
        edge_is_wedge=edge_is_wedge,
        marker_size=marker_size,
        crop_size_xy=crop_size_xy,
        color_manager=color_manager,
        palette=palette,
        distinctly_color=distinctly_color,
    )
    marker_thread.start()

    # Collect annotated frames from the output queue
    annotated_frames = []
    frames_processed = 0
    while True:
        # Use timeout to allow GUI event processing
        try:
            imgs = marker_q.get(timeout=0.1)
        except Empty:
            # Check for exceptions in worker threads
            if reader_thread.exception is not None:
                raise reader_thread.exception
            if marker_thread.exception is not None:
                raise marker_thread.exception
            # Process GUI events while waiting
            if progress_win is not None:
                from qtpy import QtWidgets

                QtWidgets.QApplication.instance().processEvents()
                if progress_win.wasCanceled():
                    canceled = True
                    break
            continue

        if imgs is _sentinel:
            break

        annotated_frames.extend(imgs)
        frames_processed += len(imgs)

        # Update progress dialog
        if progress_win is not None:
            from qtpy import QtWidgets

            progress_win.setValue(frames_processed)
            progress_win.setLabelText(
                f"Exporting labeled video...<br>"
                f"{frames_processed}/{len(frames)} frames "
                f"(<b>{(frames_processed / len(frames)) * 100:.1f}%</b>)"
            )
            QtWidgets.QApplication.instance().processEvents()
            if progress_win.wasCanceled():
                canceled = True
                break

    # Wait for threads to finish
    reader_thread.join(timeout=5.0)
    marker_thread.join(timeout=5.0)

    # Check for exceptions in worker threads
    if reader_thread.exception is not None:
        raise reader_thread.exception
    if marker_thread.exception is not None:
        raise marker_thread.exception

    # If canceled, clean up and return
    if canceled:
        if progress_win is not None:
            progress_win.close()
        return

    # Save video at end after getting annotated frames
    if progress_win is not None:
        progress_win.setLabelText("Writing video file...")
        from qtpy import QtWidgets

        QtWidgets.QApplication.instance().processEvents()

    save_video(
        frames=annotated_frames,
        filename=filename,
        fps=fps,
    )

    if progress_win is not None:
        progress_win.setValue(len(frames))
        progress_win.close()


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


def main(args: list = None):
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
    # parser.add_argument(
    #     "--crop", type=str, default="", help="Crop size as <width>,<height>"
    # )
    parser.add_argument(
        "--frames",
        type=frame_list,
        default="",
        help="list of frames to predict. Either comma separated list (e.g. 1,2,3) or "
        "a range separated by hyphen (e.g. 1-3). (default is entire video)",
    )
    parser.add_argument(
        "--video-index",
        type=int,
        default=0,
        help="Index of video in labels dataset (default: 0)",
    )
    parser.add_argument(
        "--show_edges",
        type=int,
        default=1,
        help="Whether to draw lines between nodes (default: 1)",
    )
    parser.add_argument(
        "--edge_is_wedge",
        type=int,
        default=0,
        help="Whether to draw edges as wedges (default: 0)",
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=4,
        help="Size of marker in pixels before scaling by `scale` (default: 4)",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="standard",
        help=(
            "SLEAP color palette to use Options include: 'alphabet', 'five+', "
            "'solarized', or 'standard' (default: 'standard')"
        ),
    )
    parser.add_argument(
        "--distinctly_color",
        type=str,
        default="instances",
        help=(
            "Specify how to color instances. Options include: 'instances', "
            "'edges', and 'nodes' (default: 'nodes')"
        ),
    )
    # parser.add_argument(
    #     "--background",
    #     type=str,
    #     default="original",
    #     help=(
    #         "Specify the type of background to be used to save the videos."
    #         "Options for background: original, black, white and grey"
    #     ),
    # )
    args = parser.parse_args(args=args)
    labels = load_labels_video_search(
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

    # try:
    #     crop_size_xy = list(map(int, args.crop.split(",")))
    # except Exception:
    #     crop_size_xy = None

    save_labeled_video(
        filename=filename,
        labels=labels,
        video=vid,
        frames=frames,
        fps=args.fps,
        scale=args.scale,
        crop_size_xy=None,  # default value since argument is commented out
        show_edges=args.show_edges > 0,
        edge_is_wedge=args.edge_is_wedge > 0,
        marker_size=args.marker_size,
        palette=args.palette,
        distinctly_color=args.distinctly_color,
        background="original",  # default value since argument is commented out
    )

    print(f"Video saved as: {filename}")


# if __name__ == "__main__":
#    main()

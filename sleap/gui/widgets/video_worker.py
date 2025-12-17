"""
Simple, clean worker thread implementation for video frame loading.

This avoids the complex signal/slot system that was causing QBasicTimer errors.
"""

import time
import queue
import threading
from collections import deque

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from sleap.gui.widgets.video import ndarray_to_qimage
from copy import deepcopy
import sleap_io as sio


class FrameLoaderThread(QThread):
    """
    Simple thread that loads frames using a queue-based approach.

    This avoids complex signal/slot connections that cause threading issues.
    """

    # Signal emitted when a frame is ready
    frameReady = Signal(int, QImage)  # (frame_idx, qimage)

    def __init__(self):
        super().__init__()
        self.request_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.current_video = None
        self.local_video_copy = None

        # Performance tracking
        self._frame_load_times = deque(maxlen=100)
        self._dropped_frames = 0

        # Debug mode flag for logging
        self.debug_mode = False

    def set_debug_mode(self, value: bool):
        self.debug_mode = value

    def _prepopulate_shape_cache(self, video: sio.Video):
        """Pre-populate the backend's cached shape from backend_metadata.

        For ImageVideo backends, accessing backend.shape triggers cv2.imread()
        to read a frame and determine dimensions. On network filesystems, this
        is very slow. The .slp file stores shape in backend_metadata, so we can
        use that to pre-populate the backend's _cached_shape.
        """
        if video.backend is None:
            return

        if (
            hasattr(video.backend, "_cached_shape")
            and video.backend._cached_shape is None
            and "shape" in video.backend_metadata
            and video.backend_metadata["shape"] is not None
        ):
            video.backend._cached_shape = tuple(video.backend_metadata["shape"])

    def run(self):
        """Main thread loop - processes frame requests from the queue."""

        while not self.stop_flag.is_set():
            try:
                # Wait for a request with timeout
                video, frame_idx = self.request_queue.get(timeout=0.01)

                if self.debug_mode:
                    print(f"[THREAD] Got frame request: {frame_idx}")

                # Collect all pending requests to find the latest one (LIFO)
                pending_requests = [(video, frame_idx)]  # Start with current request
                while not self.request_queue.empty():
                    try:
                        pending_video, pending_idx = self.request_queue.get_nowait()
                        pending_requests.append((pending_video, pending_idx))
                        self._dropped_frames += 1
                        if self.debug_mode:
                            print(f"[THREAD] Found pending request: {pending_idx}")
                    except queue.Empty:
                        break

                # Process only the latest (most recent) request
                latest_video, latest_frame_idx = pending_requests[-1]
                if self.debug_mode and len(pending_requests) > 1:
                    dropped_count = len(pending_requests) - 1
                    print(
                        f"[THREAD] Processing latest frame {latest_frame_idx}, "
                        f"dropped {dropped_count} older requests"
                    )

                # Process the frame
                self._process_frame(latest_video, latest_frame_idx)

            except queue.Empty:
                # No requests, continue waiting
                continue
            except Exception as e:
                print(f"[THREAD] Error in worker loop: {e}")

        pass  # Thread stopped

    def _process_frame(self, video, frame_idx: int):
        """Load and emit a frame."""
        try:
            start_time = time.time()

            if self.debug_mode:
                print(f"[THREAD] Loading frame {frame_idx}")

            # Load the frame
            frame = video[frame_idx]

            if frame is not None:
                # Handle 4-channel images (RGBA/BGRA)
                # sleap-io's opencv backend loads 4-channel images as BGRA but
                # doesn't convert to RGBA. Since we don't need alpha for display,
                # strip it and convert BGR to RGB if needed.
                if frame.ndim == 3 and frame.shape[-1] == 4:
                    # Drop alpha and swap BGR to RGB (opencv loads as BGRA)
                    frame = frame[..., 2::-1]  # BGRA -> RGB (takes channels 2,1,0)

                # Convert to QImage
                qimage = ndarray_to_qimage(frame, copy=True)

                # Emit the result
                self.frameReady.emit(frame_idx, qimage)

                if self.debug_mode:
                    print(f"[THREAD] Emitted frame {frame_idx}")

                # Track performance
                load_time = time.time() - start_time
                self._frame_load_times.append(load_time)

                # Log performance stats periodically
                if self.debug_mode and len(self._frame_load_times) == 100:
                    avg_time = sum(self._frame_load_times) / 100
                    dropped = self._dropped_frames
                    print(f"[PERF] Avg load: {avg_time:.3f}s, Dropped: {dropped}")
            else:
                if self.debug_mode:
                    print(f"[THREAD] Frame {frame_idx} was None")

        except Exception as e:
            print(f"[THREAD] Error processing frame {frame_idx}: {e}")

    def request_frame(self, video: sio.Video, frame_idx: int):
        """Request a frame to be loaded (called from main thread)."""
        if self.debug_mode:
            print(f"[MAIN] Requesting frame {frame_idx}")

        # Update the current video if a new one was provided
        if self.current_video is not video:
            if self.debug_mode:
                print("[MAIN] Switching to new video")

            # Retain original state
            reopen = video.is_open
            open_backend = video.open_backend

            # Clear backend directly instead of calling close() to avoid
            # triggering imread on network filesystems. close() tries to
            # access backend.shape to save metadata, which for ImageVideo
            # triggers cv2.imread() - very slow on network drives.
            if video.backend is not None:
                video.backend = None
            video.open_backend = False

            # Update the reference
            self.current_video = video

            # Make a thread-local copy
            self.local_video_copy = deepcopy(video)

            # Open the backend immediately and pre-populate shape cache
            # to avoid imread on network filesystems when reading frames
            self.local_video_copy.open_backend = True
            if self.local_video_copy.exists():
                self.local_video_copy.open()
                self._prepopulate_shape_cache(self.local_video_copy)

            # Restore the original state in the incoming video
            self.current_video.open_backend = open_backend
            if reopen:
                self.current_video.open()
                # Pre-populate shape cache to avoid imread on network filesystems
                self._prepopulate_shape_cache(self.current_video)

        self.request_queue.put((self.local_video_copy, frame_idx))

        if self.debug_mode:
            queue_size = self.request_queue.qsize()
            print(f"[MAIN] Frame {frame_idx} added to queue (size: {queue_size})")

    def stop(self):
        """Stop the worker thread."""
        self.stop_flag.set()
        self.quit()
        if not self.wait(2000):
            self.terminate()
            self.wait()

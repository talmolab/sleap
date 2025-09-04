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

            # Close the backend
            video.close()
            video.open_backend = False

            # Update the reference
            self.current_video = video

            # Make a thread-local copy
            self.local_video_copy = deepcopy(video)

            # Set it to open the backend on first read
            self.local_video_copy.open_backend = True

            # Restore the original state in the incoming video
            self.current_video.open_backend = open_backend
            if reopen:
                self.current_video.open()

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

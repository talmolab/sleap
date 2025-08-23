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

        # Performance tracking
        self._frame_load_times = deque(maxlen=100)
        self._dropped_frames = 0

    def run(self):
        """Main thread loop - processes frame requests from the queue."""

        while not self.stop_flag.is_set():
            try:
                # Wait for a request with timeout
                video, frame_idx = self.request_queue.get(timeout=0.01)

                # Clear any pending requests (only process latest)
                while not self.request_queue.empty():
                    try:
                        _, pending_idx = self.request_queue.get_nowait()
                        self._dropped_frames += 1
                    except queue.Empty:
                        break

                # Process the frame
                self._process_frame(video, frame_idx)

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

            # Load the frame
            frame = video.get_frame(frame_idx)

            if frame is not None:
                # Convert to QImage
                qimage = ndarray_to_qimage(frame, copy=True)

                # Emit the result
                self.frameReady.emit(frame_idx, qimage)

                # Track performance
                load_time = time.time() - start_time
                self._frame_load_times.append(load_time)

                # Log performance stats periodically (uncomment for debugging)
                # if len(self._frame_load_times) == 100:
                #     avg_time = sum(self._frame_load_times) / 100
                #     dropped = self._dropped_frames
                #     print(f"[PERF] Avg load: {avg_time:.3f}s, Dropped: {dropped}")

        except Exception as e:
            print(f"[THREAD] Error processing frame {frame_idx}: {e}")

    def request_frame(self, video, frame_idx: int):
        """Request a frame to be loaded (called from main thread)."""
        self.request_queue.put((video, frame_idx))

    def stop(self):
        """Stop the worker thread."""
        self.stop_flag.set()
        self.quit()
        if not self.wait(2000):
            self.terminate()
            self.wait()

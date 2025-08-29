import logging
import sys


# Setup logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Import submodules we want available at top-level
from sleap.version import __version__, versions

# temp fix for MediaVideo
from sleap_io.io.video_reading import MediaVideo
import multiprocessing
import cv2
import numpy as np
import imageio.v3 as iio

MediaVideo._lock = multiprocessing.RLock()


def _read_frame(self, frame_idx: int) -> np.ndarray:
    """Read a single frame from the video.
    Args:
        frame_idx: Index of frame to read.
    Returns:
        The frame as a numpy array of shape `(height, width, channels)`.
    Notes:
        This does not apply grayscale conversion. It is recommended to use the
        `get_frame` method of the `VideoBackend` class instead.
    """
    failed = False
    with MediaVideo._lock:
        if self.plugin == "opencv":
            if self.reader.get(cv2.CAP_PROP_POS_FRAMES) != frame_idx:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, img = self.reader.read()
        elif self.plugin == "pyav" or self.plugin == "FFMPEG":
            if self.keep_open:
                img = self.reader.read(index=frame_idx)
            else:
                with iio.imopen(self.filename, "r", plugin=self.plugin) as reader:
                    img = reader.read(index=frame_idx)

    success = (not failed) and (img is not None)
    if not success:
        raise IndexError(f"Failed to read frame index {frame_idx}.")
    return img


MediaVideo._read_frame = _read_frame

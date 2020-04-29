"""
Module for writing avi/mp4 videos.

Usage:

> writer = VideoWriter.safe_builder(filename, height, width, fps)
> writer.add_frame(img)
> writer.close()
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np


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
        if VideoWriter.can_use_skvideo():
            return VideoWriterSkvideo(filename, height, width, fps)
        else:
            return VideoWriterOpenCV(filename, height, width, fps)

    @staticmethod
    def can_use_skvideo():
        # See if we can import skvideo
        try:
            import skvideo
        except ImportError:
            return False

        # See if skvideo can find FFMPEG
        if skvideo.getFFmpegVersion() != "0.0.0":
            return True

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


class VideoWriterSkvideo(VideoWriter):
    """Writes video using scikit-video as wrapper for ffmpeg."""

    def __init__(self, filename, height, width, fps):
        import skvideo.io

        fps = str(fps)
        self._writer = skvideo.io.FFmpegWriter(
            filename,
            inputdict={"-r": fps,},
            outputdict={
                "-c:v": "libx264",
                "-preset": "superfast",
                "-g": "1",
                # % grouping keyframe interval
                "-framerate": fps,
                "-crf": "15",
                "-pix_fmt": "yuv420p",
            },
        )

    def add_frame(self, img, bgr: bool = False):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._writer.writeFrame(img)

    def close(self):
        self._writer.close()

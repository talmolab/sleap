"""
Video Adaptor for SLEAP-IO

Provides missing methods and functionality that exist in sleap Video
but are not available in sleap-io Video.
"""

from typing import Tuple
import numpy as np

from sleap_io.model.video import Video as SleapIOVideo
from sleap_io.io.video_reading import VideoBackend as SleapIOVideoBackend


class VideoAdaptor:
    """Adaptor class that provides missing functionality for sleap-io Video."""

    @staticmethod
    def from_filename(filename: str) -> SleapIOVideo:
        """Create a Video from filename using sleap-io backend.

        This provides a compatible interface similar to sleap Video.from_filename()
        but uses sleap-io for the actual video creation.

        Args:
            filename: Path to the video file

        Returns:
            Video object
        """
        return SleapIOVideo.from_filename(filename)

    @staticmethod
    def from_hdf5(filename: str, dataset: str = "box") -> SleapIOVideo:
        """Create a Video from HDF5 file using sleap-io backend.

        Args:
            filename: Path to the HDF5 file
            dataset: Name of the dataset within the HDF5 file

        Returns:
            Video object
        """
        return SleapIOVideo.from_hdf5(filename, dataset)

    @staticmethod
    def from_media(filename: str) -> SleapIOVideo:
        """Create a Video from media file using sleap-io backend.

        Args:
            filename: Path to the media file

        Returns:
            Video object
        """
        return SleapIOVideo.from_media(filename)

    @staticmethod
    def from_numpy(array: np.ndarray) -> SleapIOVideo:
        """Create a Video from numpy array using sleap-io backend.

        Args:
            array: Numpy array with shape (frames, height, width, channels)

        Returns:
            Video object
        """
        return SleapIOVideo.from_numpy(array)

    @staticmethod
    def from_images(image_paths: list) -> SleapIOVideo:
        """Create a Video from list of image paths using sleap-io backend.

        Args:
            image_paths: List of paths to image files

        Returns:
            Video object
        """
        return SleapIOVideo.from_images(image_paths)

    @staticmethod
    def from_imgstore(filename: str, dataset: str = "box") -> SleapIOVideo:
        """Create a Video from imgstore using sleap-io backend.

        Args:
            filename: Path to the imgstore file
            dataset: Name of the dataset

        Returns:
            Video object
        """
        return SleapIOVideo.from_imgstore(filename, dataset)

    @staticmethod
    def from_single_image(filename: str) -> SleapIOVideo:
        """Create a Video from single image using sleap-io backend.

        Args:
            filename: Path to the image file

        Returns:
            Video object
        """
        return SleapIOVideo.from_single_image(filename)

    @staticmethod
    def create_dummy(filename: str) -> SleapIOVideo:
        """Create a dummy Video using sleap-io backend.

        Args:
            filename: Path for the dummy video

        Returns:
            Video object
        """
        return SleapIOVideo.create_dummy(filename)


class VideoBackendAdaptor:
    """Adaptor class that provides missing functionality for sleap-io VideoBackend."""

    @staticmethod
    def get_frame(backend: SleapIOVideoBackend, idx: int) -> np.ndarray:
        """Get a frame from the video backend.

        Args:
            backend: The video backend
            idx: Frame index

        Returns:
            Frame as numpy array with shape (height, width, channels)
        """
        return backend.get_frame(idx)

    @staticmethod
    def get_frames(backend: SleapIOVideoBackend, indices: list) -> np.ndarray:
        """Get multiple frames from the video backend.

        Args:
            backend: The video backend
            indices: List of frame indices

        Returns:
            Frames as numpy array with shape (len(indices), height, width, channels)
        """
        frames = []
        for idx in indices:
            frames.append(backend.get_frame(idx))
        return np.array(frames)

    @staticmethod
    def get_frame_range(
        backend: SleapIOVideoBackend, start: int, end: int
    ) -> np.ndarray:
        """Get a range of frames from the video backend.

        Args:
            backend: The video backend
            start: Starting frame index (inclusive)
            end: Ending frame index (exclusive)

        Returns:
            Frames as numpy array with shape (end-start, height, width, channels)
        """
        frames = []
        for idx in range(start, end):
            frames.append(backend.get_frame(idx))
        return np.array(frames)

    @staticmethod
    def get_frame_at_time(backend: SleapIOVideoBackend, time: float) -> np.ndarray:
        """Get a frame at a specific time.

        Args:
            backend: The video backend
            time: Time in seconds

        Returns:
            Frame as numpy array
        """
        # Convert time to frame index
        if hasattr(backend, "fps") and backend.fps > 0:
            frame_idx = int(time * backend.fps)
            frame_idx = max(0, min(frame_idx, backend.frames - 1))
            return backend.get_frame(frame_idx)
        else:
            raise ValueError("Video backend does not have FPS information")

    @staticmethod
    def get_frame_timestamp(backend: SleapIOVideoBackend, idx: int) -> float:
        """Get the timestamp for a frame.

        Args:
            backend: The video backend
            idx: Frame index

        Returns:
            Timestamp in seconds
        """
        if hasattr(backend, "fps") and backend.fps > 0:
            return idx / backend.fps
        else:
            raise ValueError("Video backend does not have FPS information")

    @staticmethod
    def get_frame_count(backend: SleapIOVideoBackend) -> int:
        """Get the total number of frames.

        Args:
            backend: The video backend

        Returns:
            Number of frames
        """
        return backend.frames

    @staticmethod
    def get_video_info(backend: SleapIOVideoBackend) -> dict:
        """Get comprehensive video information.

        Args:
            backend: The video backend

        Returns:
            Dictionary containing video information
        """
        info = {
            "frames": backend.frames,
            "width": backend.width,
            "height": backend.height,
            "channels": backend.channels,
        }

        # Add optional attributes if available
        if hasattr(backend, "fps"):
            info["fps"] = backend.fps

        if hasattr(backend, "duration"):
            info["duration"] = backend.duration

        if hasattr(backend, "filename"):
            info["filename"] = backend.filename

        return info

    @staticmethod
    def is_valid_frame_index(backend: SleapIOVideoBackend, idx: int) -> bool:
        """Check if a frame index is valid.

        Args:
            backend: The video backend
            idx: Frame index to check

        Returns:
            True if index is valid, False otherwise
        """
        return 0 <= idx < backend.frames

    @staticmethod
    def get_frame_shape(backend: SleapIOVideoBackend) -> Tuple[int, int, int]:
        """Get the shape of individual frames.

        Args:
            backend: The video backend

        Returns:
            Tuple of (height, width, channels)
        """
        return (backend.height, backend.width, backend.channels)

    @staticmethod
    def get_video_shape(backend: SleapIOVideoBackend) -> Tuple[int, int, int, int]:
        """Get the shape of the entire video.

        Args:
            backend: The video backend

        Returns:
            Tuple of (frames, height, width, channels)
        """
        return (backend.frames, backend.height, backend.width, backend.channels)

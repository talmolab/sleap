"""Video utilities for sleap-io integration."""

from sleap_io import Video
from sleap_io.io.video_reading import MediaVideo, HDF5Video, ImageVideo, TiffVideo
from typing import Tuple

# Object that signals shutdown
_sentinel = object()


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


def available_video_exts() -> Tuple[str]:
    """Return tuple of supported video extensions.

    Returns:
        Tuple of supported video extensions.
    """
    return MediaVideo.EXTS + HDF5Video.EXTS + ImageVideo.EXTS + TiffVideo.EXTS


def video_util_reset(video: Video, filename: str = None, grayscale: bool = None):
    """Reloads the video.

    Returns:
        None
    """
    if filename is not None:
        video.replace_filename(filename, open=False)
        # No apparent 'test frame'.

    # If none, auto-detects based on first frame load.
    video.grayscale = grayscale

    # no apparent 'bgr' attribute (removed in sleap-io?)

    # potential breaking change
    if (filename is not None) or (grayscale is not None):
        # In sleap-io, keep_open is a backend attribute, not a video attribute
        if hasattr(video, "backend") and hasattr(video.backend, "keep_open"):
            video.backend.keep_open = (
                False  # Reader depends on both filename and grayscale
            )


def get_last_frame_idx(video=None):
    """Get the last frame index for a specific video.
    This function recreates the functionality of labels.get_last_frame_idx(video)
    from the original SLEAP codebase.
    """
    if video is None:
        return None
    if hasattr(video.backend, "source_inds"):
        source_inds = video.backend.source_inds
        return max(source_inds)
    else:
        return len(video) - 1


def video_get_frames(video: Video) -> int:
    """Get frame count for backward compatibility.

    Args:
        video: Video object to get frame count from.

    Returns:
        Number of frames in the video.
    """
    return len(video)


def video_get_height(video: Video) -> int:
    """Get video height for backward compatibility.

    Args:
        video: Video object to get height from.

    Returns:
        Height of video frames in pixels.
    """
    return video.shape[1] if len(video.shape) > 1 else None


def video_get_width(video: Video) -> int:
    """Get video width for backward compatibility.

    Args:
        video: Video object to get width from.

    Returns:
        Width of video frames in pixels.
    """
    return video.shape[2] if len(video.shape) > 2 else None


def video_get_channels(video: Video) -> int:
    """Get video channels for backward compatibility.

    Args:
        video: Video object to get channels from.

    Returns:
        Number of channels in video frames.
    """
    return video.shape[3] if len(video.shape) > 3 else 3


def video_get_frame(video: Video, idx: int):
    """Get frame by index for backward compatibility.

    Args:
        video: Video object to get frame from.
        idx: Frame index to retrieve.

    Returns:
        Frame data as numpy array.
    """
    return video[idx]

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
        video.keep_open = False  # Reader depends on both filename and grayscale


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
        return video.backend.frames - 1

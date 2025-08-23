"""
Compatibility Layer for SLEAP-IO

This module provides monkey-patching and compatibility layers to make
sleap-io classes work seamlessly with existing sleap code.
"""

from sleap_io.model.labels import Labels as SleapIOLabels
from sleap_io.model.video import Video as SleapIOVideo
from sleap_io.io.video_reading import VideoBackend as SleapIOVideoBackend

from .labels_adaptor import LabelsAdaptor
from .video_adaptor import VideoAdaptor, VideoBackendAdaptor


def patch_sleap_io_classes():
    """Apply all compatibility patches to sleap-io classes."""
    patch_labels_class()
    patch_video_class()
    patch_video_backend_class()


def patch_labels_class():
    """Add missing methods to sleap-io Labels class."""

    # Add make_gui_video_callback method
    if not hasattr(SleapIOLabels, "make_gui_video_callback"):
        SleapIOLabels.make_gui_video_callback = classmethod(
            LabelsAdaptor.make_gui_video_callback
        )

    # Add make_video_callback method
    if not hasattr(SleapIOLabels, "make_video_callback"):
        SleapIOLabels.make_video_callback = classmethod(
            LabelsAdaptor.make_video_callback
        )

    # Add load_file method
    if not hasattr(SleapIOLabels, "load_file"):
        SleapIOLabels.load_file = classmethod(LabelsAdaptor.load_file)

    # Add save_file method
    if not hasattr(SleapIOLabels, "save_file"):
        SleapIOLabels.save_file = LabelsAdaptor.save_file

    # Add from_sleap_labels method
    if not hasattr(SleapIOLabels, "from_sleap_labels"):
        SleapIOLabels.from_sleap_labels = classmethod(LabelsAdaptor.from_sleap_labels)

    # Add to_sleap_labels method
    if not hasattr(SleapIOLabels, "to_sleap_labels"):
        SleapIOLabels.to_sleap_labels = LabelsAdaptor.to_sleap_labels

    # Add other missing methods that exist in sleap-io
    if not hasattr(SleapIOLabels, "load_nwb"):
        SleapIOLabels.load_nwb = classmethod(LabelsAdaptor.load_nwb)

    if not hasattr(SleapIOLabels, "load_dlc"):
        SleapIOLabels.load_dlc = classmethod(LabelsAdaptor.load_dlc)

    if not hasattr(SleapIOLabels, "load_coco"):
        SleapIOLabels.load_coco = classmethod(LabelsAdaptor.load_coco)

    if not hasattr(SleapIOLabels, "load_ultralytics"):
        SleapIOLabels.load_ultralytics = classmethod(LabelsAdaptor.load_ultralytics)

    if not hasattr(SleapIOLabels, "load_jabs"):
        SleapIOLabels.load_jabs = classmethod(LabelsAdaptor.load_jabs)

    if not hasattr(SleapIOLabels, "load_labelstudio"):
        SleapIOLabels.load_labelstudio = classmethod(LabelsAdaptor.load_labelstudio)

    # Add methods that don't exist in sleap-io but are called in commands.py
    if not hasattr(SleapIOLabels, "load_alphatracker"):
        SleapIOLabels.load_alphatracker = classmethod(LabelsAdaptor.load_alphatracker)

    if not hasattr(SleapIOLabels, "from_deepposekit"):
        SleapIOLabels.from_deepposekit = classmethod(LabelsAdaptor.from_deepposekit)


def patch_video_class():
    """Add missing methods to sleap-io Video class."""

    # Add from_filename method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_filename"):
        SleapIOVideo.from_filename = classmethod(VideoAdaptor.from_filename)

    # Add from_hdf5 method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_hdf5"):
        SleapIOVideo.from_hdf5 = classmethod(VideoAdaptor.from_hdf5)

    # Add from_media method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_media"):
        SleapIOVideo.from_media = classmethod(VideoAdaptor.from_media)

    # Add from_numpy method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_numpy"):
        SleapIOVideo.from_numpy = classmethod(VideoAdaptor.from_numpy)

    # Add from_images method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_images"):
        SleapIOVideo.from_images = classmethod(VideoAdaptor.from_images)

    # Add from_imgstore method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_imgstore"):
        SleapIOVideo.from_imgstore = classmethod(VideoAdaptor.from_imgstore)

    # Add from_single_image method if it doesn't exist
    if not hasattr(SleapIOVideo, "from_single_image"):
        SleapIOVideo.from_single_image = classmethod(VideoAdaptor.from_single_image)

    # Add create_dummy method if it doesn't exist
    if not hasattr(SleapIOVideo, "create_dummy"):
        SleapIOVideo.create_dummy = classmethod(VideoAdaptor.create_dummy)


def patch_video_backend_class():
    """Add missing methods to sleap-io VideoBackend class."""

    # Add get_frames method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frames"):
        SleapIOVideoBackend.get_frames = VideoBackendAdaptor.get_frames

    # Add get_frame_range method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frame_range"):
        SleapIOVideoBackend.get_frame_range = VideoBackendAdaptor.get_frame_range

    # Add get_frame_at_time method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frame_at_time"):
        SleapIOVideoBackend.get_frame_at_time = VideoBackendAdaptor.get_frame_at_time

    # Add get_frame_timestamp method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frame_timestamp"):
        SleapIOVideoBackend.get_frame_timestamp = (
            VideoBackendAdaptor.get_frame_timestamp
        )

    # Add get_frame_count method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frame_count"):
        SleapIOVideoBackend.get_frame_count = VideoBackendAdaptor.get_frame_count

    # Add get_video_info method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_video_info"):
        SleapIOVideoBackend.get_video_info = VideoBackendAdaptor.get_video_info

    # Add is_valid_frame_index method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "is_valid_frame_index"):
        SleapIOVideoBackend.is_valid_frame_index = (
            VideoBackendAdaptor.is_valid_frame_index
        )

    # Add get_frame_shape method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_frame_shape"):
        SleapIOVideoBackend.get_frame_shape = VideoBackendAdaptor.get_frame_shape

    # Add get_video_shape method if it doesn't exist
    if not hasattr(SleapIOVideoBackend, "get_video_shape"):
        SleapIOVideoBackend.get_video_shape = VideoBackendAdaptor.get_video_shape


# Auto-patch when module is imported
patch_sleap_io_classes()

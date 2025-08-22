"""
Dialog for exporting clip; shows message depending on available encoder.
"""

from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog

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

class ExportClipDialog(FormBuilderModalDialog):
    def __init__(self, form_name=None):

        form_name = form_name or "video_clip_form"
        super().__init__(form_name=form_name)

        _can_use_ffmpeg = can_use_ffmpeg()

        if _can_use_ffmpeg:
            message = (
                "<i><b>MP4</b> file will be encoded using system ffmpeg "
                "via imageio (preferred option).</i>"
            )
        else:
            message = (
                "<i>Unable to use ffmpeg via imageio. <b>AVI</b> file will be "
                "encoding using OpenCV.</i>"
            )

        self.add_message(message)

        self.setWindowTitle("Export Clip Options")

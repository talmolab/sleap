"""
Dialog for exporting clip; shows message depending on available encoder.
"""

from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog


class ExportClipDialog(FormBuilderModalDialog):
    def __init__(self, form_name=None):
        from sleap.sleap_io_adaptors.video_utils import can_use_ffmpeg

        form_name = form_name or "video_clip_form"
        super().__init__(form_name=form_name)

        can_use_ffmpeg = can_use_ffmpeg()

        if can_use_ffmpeg:
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

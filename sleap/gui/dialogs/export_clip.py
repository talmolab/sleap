"""
Dialog for exporting clip; shows message depending on available encoder.
"""

from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog


class ExportClipDialog(FormBuilderModalDialog):
    def __init__(self):
        from sleap.io.videowriter import VideoWriter

        super().__init__(form_name="labeled_clip_form")

        can_use_ffmpeg = VideoWriter.can_use_ffmpeg()

        if can_use_ffmpeg:
            message = (
                "<i><b>MP4</b> file will be encoded using "
                "system ffmpeg via imageio (preferred option).</i>"
            )
        else:
            message = (
                "<i>Unable to use ffpmeg via imageio. "
                "<b>AVI</b> file will be encoding using OpenCV.</i>"
            )

        self.add_message(message)

        self.setWindowTitle("Export Clip Options")

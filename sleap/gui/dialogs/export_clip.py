from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog


class ExportClipDialog(FormBuilderModalDialog):
    def __init__(self):
        from sleap.gui.dialogs.formbuilder import YamlFormWidget
        from sleap.io.videowriter import VideoWriter

        form_widget = YamlFormWidget.from_name("labeled_clip_form")
        super().__init__(form_widget=form_widget)

        can_use_skvideo = VideoWriter.can_use_skvideo()

        if can_use_skvideo:
            message = (
                "<i><b>MP4</b> file will be encoded using "
                "system ffmpeg via scikit-video (preferred option).</i>"
            )
        else:
            message = (
                "<i>Unable to use ffpmeg via scikit-video. "
                "<b>AVI</b> file will be encoding using OpenCV.</i>"
            )

        self.add_message(message)

        self.setWindowTitle("Export Clip Options")

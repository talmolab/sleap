"""
Dialog for exporting clip; shows message depending on available encoder.
"""

from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog
from qtpy import QtWidgets

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

class ExportClipAndLabelsDialog(FormBuilderModalDialog):
    def __init__(self, video_fps=30):
        from sleap.io.videowriter import VideoWriter

        # Initialize with a blank widget (no YAML needed)
        super().__init__(form_widget=QtWidgets.QWidget())

        self.setWindowTitle("Export Clip Options")

        # FPS Field
        self.fps_input = QtWidgets.QSpinBox()
        self.fps_input.setRange(1, 240)
        self.fps_input.setValue(video_fps)  # Set default FPS from video
        self.add_message("Frames per second:")
        self.layout().insertWidget(2, self.fps_input)

        # Open when done Checkbox
        self.open_when_done = QtWidgets.QCheckBox("Open file after saving")
        self.layout().insertWidget(3, self.open_when_done)

        # Video format message
        can_use_ffmpeg = VideoWriter.can_use_ffmpeg()
        if can_use_ffmpeg:
            message = (
                "<i><b>MP4</b> file will be encoded using "
                "system ffmpeg via imageio (preferred option).</i>"
            )
        else:
            message = (
                "<i>Unable to use ffmpeg via imageio. "
                "<b>AVI</b> file will be encoded using OpenCV.</i>"
            )
        self.add_message(message)

    def on_accept(self):
        """Retrieve the form results and accept the dialog."""
        self._results = {
            "fps": self.fps_input.value(),
            "open_when_done": self.open_when_done.isChecked(),
        }
        self.accept()

    def get_results(self):
        """Get results as a dictionary."""
        self._results = {
            "fps": self.fps_input.value(),
            "open_when_done": self.open_when_done.isChecked(),
        }
        return self._results
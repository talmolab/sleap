"""Frame range dialog."""
from qtpy import QtWidgets
from sleap.gui.dialogs.formbuilder import FormBuilderModalDialog
from typing import Optional


class FrameRangeDialog(FormBuilderModalDialog):
    def __init__(self, max_frame_idx: Optional[int] = None, title: str = "Frame Range"):

        super().__init__(form_name="frame_range_form")
        min_frame_idx_field = self.form_widget.fields["min_frame_idx"]
        max_frame_idx_field = self.form_widget.fields["max_frame_idx"]

        if max_frame_idx is not None:
            min_frame_idx_field.setRange(1, max_frame_idx)
            min_frame_idx_field.setValue(1)

            max_frame_idx_field.setRange(1, max_frame_idx)
            max_frame_idx_field.setValue(max_frame_idx)

        min_frame_idx_field.valueChanged.connect(self._update_max_frame_range)
        max_frame_idx_field.valueChanged.connect(self._update_min_frame_range)

        self.setWindowTitle(title)

    def _update_max_frame_range(self, value):
        min_frame_idx_field = self.form_widget.fields["min_frame_idx"]
        max_frame_idx_field = self.form_widget.fields["max_frame_idx"]

        max_frame_idx_field.setRange(value, max_frame_idx_field.maximum())

    def _update_min_frame_range(self, value):
        min_frame_idx_field = self.form_widget.fields["min_frame_idx"]
        max_frame_idx_field = self.form_widget.fields["max_frame_idx"]

        min_frame_idx_field.setRange(min_frame_idx_field.minimum(), value)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    dialog = FrameRangeDialog(max_frame_idx=100)
    print(dialog.get_results())

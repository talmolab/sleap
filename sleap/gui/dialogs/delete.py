"""
Dialog for deleting various subsets of instances in dataset.
"""

from sleap import LabeledFrame, Instance
from sleap.gui.dialogs import formbuilder

from PySide2 import QtCore, QtWidgets

from typing import List, Text, Tuple


class DeleteDialog(QtWidgets.QDialog):
    """
    Dialog for deleting various subsets of instances in dataset.

    Args:
        context: The `CommandContext` from which this dialog is being
            shown. The context provides both a `labels` (`Labels`) and a
            `state` (`GuiState`).

    """

    # NOTE: use type by name (rather than importing CommandContext) to avoid
    # circular includes.
    def __init__(
        self,
        context: "CommandContext",
        *args,
        **kwargs,
    ):

        super(DeleteDialog, self).__init__(*args, **kwargs)

        self.context = context

        # Layout for main form and buttons
        self.form_widget = self._make_form_widget()
        buttons_layout_widget = self._make_button_widget()

        # Layout for entire dialog
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.form_widget)
        layout.addWidget(buttons_layout_widget)

        self.setLayout(layout)

        self.accepted.connect(self.delete)

    def _make_form_widget(self):
        self.tracks = self.context.labels.tracks

        widget = QtWidgets.QGroupBox()
        layout = QtWidgets.QFormLayout()

        self.instance_type_menu = formbuilder.FieldComboWidget()
        self.frames_menu = formbuilder.FieldComboWidget()
        self.tracks_menu = formbuilder.FieldComboWidget()

        instance_type_options = [
            "predicted instances",
            "user instances",
            "all instances",
        ]

        frame_options = [
            "current frame",
            "current video",
        ]
        if len(self.context.labels.videos) > 1:
            frame_options.append("all videos")
        if self.context.state["has_frame_range"]:
            frame_options.extend(
                ["selected clip", "current video except for selected clip"]
            )

        if self.tracks:
            track_options = [
                "any track identity (including none)",
                "no track identity set",
            ]
            self._track_idx_offset = len(track_options)
            track_options.extend([track.name for track in self.tracks])
        else:
            self._track_idx_offset = 0
            track_options = []

        self.instance_type_menu.set_options(instance_type_options)
        self.frames_menu.set_options(frame_options)
        self.tracks_menu.set_options(track_options)

        layout.addRow("Delete", self.instance_type_menu)
        layout.addRow("in", self.frames_menu)

        if self.tracks:
            layout.addRow("with", self.tracks_menu)

        widget.setLayout(layout)
        return widget

    def _make_button_widget(self):
        # Layout for buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.delete_button = buttons.addButton(
            "Delete", QtWidgets.QDialogButtonBox.AcceptRole
        )

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        # Connect actions for buttons
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        return buttons_layout_widget

    def get_selected_track(self):
        track_menu_idx = self.tracks_menu.currentIndex()
        track_idx = track_menu_idx - self._track_idx_offset

        if 0 <= track_idx < len(self.tracks):
            return self.tracks[track_idx]

        return None

    def get_frames_instances(
        self, instance_type_value: Text, frames_value: Text, tracks_value: Text
    ) -> List[Tuple[LabeledFrame, Instance]]:
        """Get list of instances based on drop-down menu options selected."""

        def inst_condition(inst):
            if instance_type_value.startswith("predicted"):
                if not hasattr(inst, "score"):
                    return False
            elif instance_type_value.startswith("user"):
                if hasattr(inst, "score"):
                    return False

            if tracks_value.startswith("any"):
                # print("match any track")
                pass
            elif tracks_value.startswith("no"):
                # print("match None track")
                if inst.track is not None:
                    return False
            else:
                track_to_match = self.get_selected_track()
                if track_to_match:
                    if inst.track != track_to_match:
                        return False

            return True

        labels = self.context.labels

        lf_list = []
        if frames_value == "current frame":
            lf_list = labels.find(
                video=self.context.state["video"],
                frame_idx=self.context.state["frame_idx"],
            )
        elif frames_value == "current video":
            lf_list = labels.find(
                video=self.context.state["video"],
            )
        elif frames_value == "all videos":
            lf_list = labels.labeled_frames
        elif frames_value == "selected clip":
            clip_range = range(*self.context.state["frame_range"])
            print(clip_range)
            lf_list = labels.find(
                video=self.context.state["video"], frame_idx=clip_range
            )
        elif frames_value == "current video except for selected clip":
            clip_range = range(*self.context.state["frame_range"])
            lf_list = [
                lf
                for lf in labels.labeled_frames
                if (
                    lf.video != self.context.state["video"]
                    or lf.frame_idx not in clip_range
                )
            ]
        else:
            raise ValueError(f"Invalid frames_value: {frames_value}")

        lf_inst_list = [
            (lf, inst) for lf in lf_list for inst in lf if inst_condition(inst)
        ]

        return lf_inst_list

    def delete(self):
        instance_type_value = self.instance_type_menu.value()
        frames_value = self.frames_menu.value()
        tracks_value = self.tracks_menu.value()

        lf_inst_list = self.get_frames_instances(
            instance_type_value=instance_type_value,
            frames_value=frames_value,
            tracks_value=tracks_value,
        )

        # print(len(lf_inst_list))
        # print(instance_type_value)
        # print(frames_value)
        # print(tracks_value)
        self._delete(lf_inst_list)

    def _delete(self, lf_inst_list: List[Tuple[LabeledFrame, Instance]]):
        # Delete the instances
        for lf, inst in lf_inst_list:
            self.context.labels.remove_instance(lf, inst, in_transaction=True)
            if not lf.instances:
                self.context.labels.remove(lf)

        # Update caches since we skipped doing this after each deletion
        self.context.labels.update_cache()

        # Log update
        self.context.changestack_push("delete instances")


if __name__ == "__main__":

    app = QtWidgets.QApplication([])

    from sleap import Labels
    from sleap.gui.commands import CommandContext

    labels = Labels.load_file(
        "tests/data/json_format_v2/centered_pair_predictions.json"
    )
    context = CommandContext.from_labels(labels)
    context.state["frame_idx"] = 123
    context.state["video"] = labels.videos[0]
    context.state["has_frame_range"] = True
    context.state["frame_range"] = (10, 20)

    win = DeleteDialog(context=context)
    win.show()

    app.exec_()

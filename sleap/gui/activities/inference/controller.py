from typing import Text, List, Optional

import attr
from PySide2.QtWidgets import QWidget

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
    ModelType,
    TrackerType,
    Verbosity,
)
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.gui.widgets.videos_table import VideosTableModel


@attr.s(auto_attribs=True)
class InferenceGuiController(object):
    model: InferenceGuiModel
    logging_enabled: bool = True

    # Getters

    @staticmethod
    def get_model_type_names() -> List[Text]:
        return [mt.value for mt in ModelType]

    @staticmethod
    def get_tracking_method_names() -> List[Text]:
        return [tm.value[0] for tm in TrackerType]

    @staticmethod
    def get_verbosity_names() -> List[Text]:
        return [v.value[0] for v in Verbosity]

    def get_video_table_model(self) -> VideosTableModel:
        return self.model.videos.videos_table_model

    # Setters
    def set_model_type(self, model_type: str,
                           single_instance_model_widget: QWidget,
                           bottom_up_model_widget: QWidget,
                           top_down_centroid_model_widget: QWidget,
                           top_down_centered_instance_model_widget: QWidget) -> None:
        self.model.models.model_type = InferenceGuiController.lookup_enum(ModelType, model_type)
        if self.model.models.model_type.value == ModelType.SINGLE_INSTANCE.value:
            single_instance_model_widget.setDisabled(False)
            bottom_up_model_widget.setDisabled(True)
            top_down_centroid_model_widget.setDisabled(True)
            top_down_centered_instance_model_widget.setDisabled(True)
        elif self.model.models.model_type.value == ModelType.BOTTOM_UP.value:
            single_instance_model_widget.setDisabled(True)
            bottom_up_model_widget.setDisabled(False)
            top_down_centroid_model_widget.setDisabled(True)
            top_down_centered_instance_model_widget.setDisabled(True)
        elif self.model.models.model_type.value == ModelType.TOP_DOWN.value:
            self.log(f"BlaE+++")
            single_instance_model_widget.setDisabled(True)
            bottom_up_model_widget.setDisabled(True)
            top_down_centroid_model_widget.setDisabled(False)
            top_down_centered_instance_model_widget.setDisabled(False)
        self.log(f"Model type changed to {self.model.models.model_type}")

    def set_tracking_enabled(self,
                             tracking_enabled: bool,
                             tracking_method_widget: QWidget,
                             tracking_window_size_widget: QWidget):
        self.model.instances.enable_tracking = tracking_enabled
        tracking_method_widget.setEnabled(self.model.instances.enable_tracking)
        tracking_window_size_widget.setEnabled(self.model.instances.enable_tracking)
        self.log(f"Tracking enabled changed to {self.model.instances.enable_tracking}")

    # Actions

    def run(self) -> None:
        for v in self.model.videos.videos_table_model.items:
            cmd = f"sleap-track {v['Path']}"

            if self.model.models.model_type == ModelType.TOP_DOWN:
                cmd += f" -m {self.model.models.centroid_model.path}"
                cmd += f" -m {self.model.models.centered_instance_model.path}"
            elif self.model.models.model_type == ModelType.BOTTOM_UP:
                cmd += f" -m {self.model.models.bottom_up_model.path}"
            elif self.model.models.model_type == ModelType.SINGLE_INSTANCE:
                cmd += f" -m {self.model.models.single_instance_model.path}"

            cmd += (
                f" --tracking.tracker {self.model.instances.tracking_method.value[1]}"
            )
            cmd += f" --tracking.track_window {self.model.instances.tracking_window}"
            cmd += f" --tracking.target_instance_count {self.model.instances.max_num_instances}"

            cmd += f" -o {self.model.output.output_file_path}"
            cmd += f" --verbosity {self.model.output.verbosity.value[1]}"
            if not self.model.output.include_empty_frames:
                cmd += " --no-empty-frames"
            self._execute(cmd)

    def save(self):
        self.log(f"Save stub:\n{attr.asdict(self.model)}")

    def export(self):
        self.log(f"Export stub...")

    def load(self):
        self.log(f"Load stub...")

    def _execute(self, cmd: Text):
        self.log(f"Execute stub:\n{cmd}")

    # Utils

    @staticmethod
    def lookup_enum(enum, value: str, value_index: Optional[int] = None) -> object:
        if value_index is None:
            filtered = [v for v in enum if v.value == value]
        else:
            filtered = [v[value_index] for v in enum if v.value == value]
        if len(filtered) == 1:
            return filtered[0]
        else:
            raise ValueError(f"Enum {enum} has {len(filtered)} matching values for {value}")

    def log(self, message: str) -> None:
        if self.logging_enabled:
            print(f"+++ {message}")

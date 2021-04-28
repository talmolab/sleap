import os
from typing import Text, Optional

import attr

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
)
from sleap.gui.activities.inference.enums import ModelType, Verbosity, TrackerType
from sleap.gui.widgets.videos_table import VideosTableModel


@attr.s(auto_attribs=True)
class InferenceGuiController(object):
    model: InferenceGuiModel
    logging_enabled: bool = True

    # Getters

    def get_model_type(self) -> ModelType:
        return self.model.models.model_type

    def get_single_instance_model_path(self) -> str:
        return self.model.models.single_instance_model.path if self.model.models.single_instance_model else None

    def get_bottom_up_model_path(self) -> str:
        return self.model.models.bottom_up_model.path if self.model.models.bottom_up_model else None

    def get_top_down_centroid_model_path(self) -> str:
        return self.model.models.centroid_model.path if self.model.models.centroid_model else None

    def get_top_down_centered_instance_model_path(self) -> str:
        return self.model.models.centered_instance_model.path if self.model.models.centered_instance_model else None

    def get_video_table_model(self) -> VideosTableModel:
        return self.model.videos.videos_table_model

    def get_max_num_instances_in_frame(self) -> int:
        return self.model.instances.max_num_instances

    def get_enable_tracking(self) -> bool:
        return self.model.instances.enable_tracking

    def get_tracking_method(self) -> TrackerType:
        return self.model.instances.tracking_method

    def get_tracking_window_size(self) -> int:
        return self.model.instances.tracking_window

    def get_ouput_dir_path(self) -> str:
        return os.path.split(self.model.output.output_file_path)[0] if self.model.output.output_file_path else None

    def get_output_file_name(self) -> str:
        return os.path.split(self.model.output.output_file_path)[1] if self.model.output.output_file_path else None

    def get_include_empty_frames(self) -> bool:
        return self.model.output.include_empty_frames

    def get_verbosity(self) -> Verbosity:
        return self.model.output.verbosity


    # Setters

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
                f" --tracking.tracker {self.model.instances.tracking_method.arg()}"
            )
            cmd += f" --tracking.track_window {self.model.instances.tracking_window}"
            cmd += f" --tracking.target_instance_count {self.model.instances.max_num_instances}"

            cmd += f" -o {self.model.output.output_file_path}"
            cmd += f" --verbosity {self.model.output.verbosity.arg()}"
            if not self.model.output.include_empty_frames:
                cmd += " --no-empty-frames"
            self._execute(cmd)

    def save(self, content: dict) -> None:
        self.log(f"Save stub:\n current model: {attr.asdict(self.model)}\n content: {content}")

    def export(self):
        self.log(f"Export stub...")

    def load(self):
        self.log(f"Load stub...")

    def _execute(self, cmd: Text):
        self.log(f"Execute stub:\n{cmd}")

    # Utils

    def log(self, message: str) -> None:
        if self.logging_enabled:
            print(f"+++ {message}")

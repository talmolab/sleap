import os
from typing import Text, List

import attr

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
)
from sleap.gui.activities.inference.enums import ModelType, Verbosity, TrackerType


@attr.s(auto_attribs=True)
class InferenceGuiController(object):
    model: InferenceGuiModel
    logging_enabled: bool = True

    # Getters

    def get_model_type(self) -> ModelType:
        return self.model.models.model_type

    def get_single_instance_model_path(self) -> str:
        return self.model.models.single_instance_model_path

    def get_bottom_up_model_path(self) -> str:
        return self.model.models.bottom_up_model_path

    def get_top_down_centroid_model_path(self) -> str:
        return self.model.models.centroid_model_path

    def get_top_down_centered_instance_model_path(self) -> str:
        return self.model.models.centered_instance_model_path

    def get_video_paths(self) -> List[str]:
        return self.model.videos.paths

    def get_max_num_instances_in_frame(self) -> int:
        return self.model.instances.max_num_instances

    def get_enable_tracking(self) -> bool:
        return self.model.instances.enable_tracking

    def get_tracking_method(self) -> TrackerType:
        return self.model.instances.tracking_method

    def get_tracking_window_size(self) -> int:
        return self.model.instances.tracking_window

    def get_output_dir_path(self) -> str:
        return self.model.output.output_dir_path

    def get_output_file_suffix(self) -> str:
        return self.model.output.output_file_suffix

    def get_include_empty_frames(self) -> bool:
        return self.model.output.include_empty_frames

    def get_verbosity(self) -> Verbosity:
        return self.model.output.verbosity

    # Setters

    # Actions

    def run(self) -> None:
        for v in self.get_video_paths():
            cmd = f"sleap-track {v}"

            if self.get_model_type() == ModelType.TOP_DOWN:
                cmd += f" -m {os.path.dirname(self.get_top_down_centroid_model_path())}"
                cmd += f" -m {os.path.dirname(self.get_top_down_centered_instance_model_path())}"
            elif self.get_model_type() == ModelType.BOTTOM_UP:
                cmd += f" -m {os.path.dirname(self.get_bottom_up_model_path())}"
            elif self.get_model_type() == ModelType.SINGLE_INSTANCE:
                cmd += f" -m {os.path.dirname(self.get_single_instance_model_path())}"

            if self.get_enable_tracking():
                cmd += f" --tracking.tracker {self.get_tracking_method().arg()}"
                cmd += f" --tracking.track_window {self.get_tracking_window_size()}"
                cmd += f" --tracking.target_instance_count {self.get_max_num_instances_in_frame()}"

            output_file_path = f"{os.path.basename(v)}{self.get_output_file_suffix()}"
            if self.get_output_dir_path():
                output_file_path = os.path.join(self.get_output_dir_path(), output_file_path)
            cmd += f" -o {output_file_path}"
            cmd += f" --verbosity {self.get_verbosity().arg()}"
            if not self.get_include_empty_frames():
                cmd += " --no-empty-frames"
            self._execute(cmd)

    def save(self, content: dict) -> None:
        self.log(
            f"Save stub:\n current model: {attr.asdict(self.model)}\n content: {content}"
        )

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

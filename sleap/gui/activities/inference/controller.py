from typing import Text, Optional

import attr

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
)
from sleap.gui.activities.inference.enums import ModelType
from sleap.gui.widgets.videos_table import VideosTableModel


@attr.s(auto_attribs=True)
class InferenceGuiController(object):
    model: InferenceGuiModel
    logging_enabled: bool = True

    # Getters

    def get_video_table_model(self) -> VideosTableModel:
        return self.model.videos.videos_table_model

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

import json
from typing import Text, List

import attr

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
    ModelType,
    TrackerType,
    Verbosity,
)
from sleap.gui.widgets.videos_table import VideosTableModel


@attr.s(auto_attribs=True)
class InferenceGuiController(object):
    model: InferenceGuiModel

    @staticmethod
    def model_type_names() -> List[Text]:
        return [mt.value for mt in ModelType]

    @staticmethod
    def tracking_method_names() -> List[Text]:
        return [tm.value[0] for tm in TrackerType]

    @staticmethod
    def verbosity_names() -> List[Text]:
        return [v.value[0] for v in Verbosity]

    def video_table_model(self) -> VideosTableModel:
        return self.model.videos.videos_table_model

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
        print(f"+++ Save stub:\n{attr.asdict(self.model)}")

    def export(self):
        print("+++ Export stub...")

    def load(self):
        print("+++ Load stub...")

    def _execute(self, cmd: Text):
        print(f"+++ Execute stub:\n{cmd}")

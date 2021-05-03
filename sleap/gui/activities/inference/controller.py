import os
import subprocess
import time
from typing import Text, List, Callable, Optional

import attr

from sleap.gui.activities.inference.model import (
    InferenceGuiModel,
)
from sleap.gui.activities.inference.enums import ModelType, Verbosity, TrackerType
from sleap.gui.learning.runners import kill_process


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

    def set_model_type(self, value: ModelType) -> None:
        self.model.models.model_type = value

    def set_single_instance_model_path(self, value: str) -> None:
        self.model.models.single_instance_model_path = value

    def set_bottom_up_model_path(self, value: str) -> None:
        self.model.models.bottom_up_model_path = value

    def set_top_down_centroid_model_path(self, value: str) -> None:
        self.model.models.centroid_model_path = value

    def set_top_down_centered_instance_model_path(self, value: str) -> None:
        self.model.models.centered_instance_model_path = value

    def set_video_paths(self, value: List[str]) -> None:
        self.model.videos.paths = value

    def set_max_num_instances_in_frame(self, value: int) -> None:
        self.model.instances.max_num_instances = value

    def set_enable_tracking(self, value: bool) -> None:
        self.model.instances.enable_tracking = value

    def set_tracking_method(self, value: TrackerType) -> None:
        self.model.instances.tracking_method = value

    def set_tracking_window_size(self, value: int) -> None:
        self.model.instances.tracking_window = value

    def set_output_dir_path(self, value: str) -> None:
        self.model.output.output_dir_path = value

    def set_output_file_suffix(self, value: str) -> None:
        self.model.output.output_file_suffix = value

    def set_include_empty_frames(self, value: bool) -> None:
        self.model.output.include_empty_frames = value

    def set_verbosity(self, value: Verbosity) -> None:
        self.model.output.verbosity = value

    # Actions

    def run(self, content: dict) -> None:
        self.save(content=content)

        def output_consumer(output_line: str) -> int:
            self.log(f"Output line: {output_line}")
            return False

        for v in self.get_video_paths():
            cmd_args = ["sleap-track", v]

            if self.get_model_type() == ModelType.TOP_DOWN:
                cmd_args.extend(["-m", os.path.dirname(self.get_top_down_centroid_model_path())])
                cmd_args.extend(["-m", os.path.dirname(self.get_top_down_centered_instance_model_path())])
            elif self.get_model_type() == ModelType.BOTTOM_UP:
                cmd_args.extend(["-m", os.path.dirname(self.get_bottom_up_model_path())])
            elif self.get_model_type() == ModelType.SINGLE_INSTANCE:
                cmd_args.extend(["-m", os.path.dirname(self.get_single_instance_model_path())])

            if self.get_enable_tracking():
                cmd_args.extend(["--tracking.tracker", self.get_tracking_method().arg()])
                cmd_args.extend(["--tracking.track_window", self.get_tracking_window_size()])
                cmd_args.extend(["--tracking.target_instance_count", self.get_max_num_instances_in_frame()])

            output_file_path = f"{os.path.basename(v)}{self.get_output_file_suffix()}"
            if self.get_output_dir_path():
                output_file_path = os.path.join(self.get_output_dir_path(), output_file_path)
            cmd_args.extend(["-o", output_file_path])

            cmd_args.extend(["--verbosity", self.get_verbosity().arg()])

            if not self.get_include_empty_frames():
                cmd_args.append("--no-empty-frames")
            self._execute(cmd_args=cmd_args, output_consumer=lambda s: output_consumer(s))

    def save(self, content: dict) -> None:
        self.log(f"Saving: {content}")
        self.set_model_type(ModelType.from_display(content["model_type"])),
        self.set_single_instance_model_path(content["single_instance_model"]),
        self.set_bottom_up_model_path(content["bottom_up_model"]),
        self.set_top_down_centroid_model_path(content["top_down_centroid_model"]),
        self.set_top_down_centered_instance_model_path(content["top_down_centered_instance_model"]),
        self.set_video_paths(content["video_paths"]),
        self.set_max_num_instances_in_frame(content["max_num_instances_in_frame"]),
        self.set_enable_tracking(content["enable_tracking"]),
        self.set_tracking_method(TrackerType.from_display(content["tracking_method"])),
        self.set_tracking_window_size(content["tracking_window_size"]),
        self.set_output_dir_path(content["output_dir_path"]),
        self.set_output_file_suffix(content["output_file_suffix"]),
        self.set_include_empty_frames(content["include_empty_frames"]),
        self.set_verbosity(Verbosity.from_display(content["verbosity"])),

    def export(self):
        self.log(f"Export stub...")

    def load(self):
        self.log(f"Load stub...")

    def _execute(self, cmd_args: List[str], output_consumer: Optional[Callable[[str], int]] = None) -> int:
        print(f"Executing: {' '.join(cmd_args)}\n")
        with subprocess.Popen(cmd_args, stdout=subprocess.PIPE) as proc:
            while proc.poll() is None:
                output_line = proc.stdout.readline().decode().rstrip()
                if output_consumer is not None:
                    cancel_request = output_consumer(output_line)
                    if cancel_request:
                        self.log(f"Received cancel request from output consumer. Cancelling {proc.pid}..")
                        kill_process(proc.pid)
                        return -1
                else:
                    self.log(f"Output line:{output_line}")
                time.sleep(0.1)
            self.log(f"Process return code: {proc.returncode}")
            return proc.returncode

    # Utils

    def log(self, message: str) -> None:
        if self.logging_enabled:
            print(f"+++ {message}")

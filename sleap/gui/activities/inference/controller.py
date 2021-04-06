from typing import List, Text

from sleap.gui.activities.inference.model import InferenceGuiModel


class InferenceGuiController(object):
    model: InferenceGuiModel

    def run(
            self,
            video_paths: List[Text],
            model_paths: List[Text],
            output_path: Text,
            tracking_mode: Text = "none",
            verbosity: Text = "json",
            no_empty_frames: bool = True
    ):
        for v in video_paths:
            cmd = f"sleap-track {v}"
            for m in model_paths:
                cmd += f" -m {m}"
            cmd += f" -o {output_path}"
            cmd += f" --tracking.tracker {tracking_mode}"
            cmd += f" --verbosity {verbosity}"
            if no_empty_frames:
                cmd += " --no-empty-frames"
            self._execute(cmd)

    def save(self):
        print("+++ Save stub...")

    def export(self):
        print("+++ Export stub...")

    def load(self):
        print("+++ Load stub...")

    def _execute(self, cmd: Text):
        print(f"+++ Execute stub: {cmd}")

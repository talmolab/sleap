from typing import Optional, Text, List

import attr

from sleap.gui.activities.inference.enums import ModelType, TrackerType, Verbosity
from sleap.gui.learning.configs import ConfigFileInfo


@attr.s(auto_attribs=True)
class TrainedModels(object):
    model_type: Optional[ModelType] = ModelType.TOP_DOWN
    single_instance_model: Optional[ConfigFileInfo] = None
    bottom_up_model: Optional[ConfigFileInfo] = None
    centroid_model: Optional[ConfigFileInfo] = None
    centered_instance_model: Optional[ConfigFileInfo] = None


@attr.s(auto_attribs=True)
class Videos(object):
    paths: List[str] = []


@attr.s(auto_attribs=True)
class Instances(object):
    max_num_instances: int = 2
    enable_tracking: bool = False
    tracking_method: TrackerType = TrackerType.SIMPLE
    tracking_window: int = 5


@attr.s(auto_attribs=True)
class Output(object):
    output_file_path: Optional[Text] = None
    include_empty_frames: bool = False
    verbosity: Verbosity = Verbosity.JSON


@attr.s(auto_attribs=True)
class InferenceGuiModel(object):
    models: TrainedModels = TrainedModels()
    videos: Videos = Videos()
    instances: Instances = Instances()
    output: Output = Output()

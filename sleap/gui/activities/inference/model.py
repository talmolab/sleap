from enum import Enum
from typing import Optional, Text, List

import attr

from sleap.gui.learning.configs import ConfigFileInfo


class ModelType(Enum):
    SINGLE_INSTANCE = "Single Instance"
    BOTTOM_UP = "Multi Instance Bottom Up"
    TOP_DOWN = "Multi Instance Top Down"


@attr.s(auto_attribs=True)
class TrainedModels(object):
    model_type: Optional[ModelType] = ModelType.TOP_DOWN
    single_instance_model: Optional[ConfigFileInfo] = None
    bottom_up_model: Optional[ConfigFileInfo] = None
    centroid_model: Optional[ConfigFileInfo] = None
    centered_instance_model: Optional[ConfigFileInfo] = None


@attr.s(auto_attribs=True)
class VideoMetadata(object):
    path: Text
    frames: int
    image_size: Text
    from_frame: int
    to_frame: int


@attr.s(auto_attribs=True)
class Videos(object):
    video_metadata_list: List[VideoMetadata] = []


@attr.s(auto_attribs=True)
class Instances(object):
    class TrackerType(Enum):
        SIMPLE = "Simple"
        FLOW = "Flow Shift"
        KALMAN = "Kalman Filter"

    max_num_instances: int = 2
    enable_tracking: bool = False
    tracking_method: TrackerType = TrackerType.SIMPLE
    tracking_window: int = 5


class Verbosity(Enum):
    JSON = "Json"
    RICH = "Rich"
    NONE = "None"


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

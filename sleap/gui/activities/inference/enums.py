from enum import Enum


class ModelType(Enum):
    SINGLE_INSTANCE = "Single Instance"
    BOTTOM_UP = "Multi Instance Bottom Up"
    TOP_DOWN = "Multi Instance Top Down"


class TrackerType(Enum):
    SIMPLE = "Simple", "simple"
    FLOW = "Flow Shift", "flow"
    KALMAN = "Kalman Filter", "kalman"


class Verbosity(Enum):
    JSON = "Json", "json"
    RICH = "Rich", "rich"
    NONE = "None", "none"

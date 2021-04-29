from enum import Enum


class ModelType(Enum):
    TOP_DOWN = "Multi Instance Top Down"
    BOTTOM_UP = "Multi Instance Bottom Up"
    SINGLE_INSTANCE = "Single Instance"

    def display(self):
        return self.value


class TrackerType(Enum):
    SIMPLE = "Simple", "simple"
    FLOW = "Flow Shift", "flow"
    KALMAN = "Kalman Filter", "kalman"

    def display(self):
        return self.value[0]

    def arg(self):
        return self.value[1]


class Verbosity(Enum):
    JSON = "Json", "json"
    RICH = "Rich", "rich"
    NONE = "None", "none"

    def display(self):
        return self.value[0]

    def arg(self):
        return self.value[1]

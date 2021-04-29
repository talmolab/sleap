from enum import Enum


class BaseDisplayEnum(Enum):
    def display(self) -> str:
        return self.value if isinstance(self.value, str) else self.value[0]

    def arg(self) -> str:
        return self.value[1]

    @classmethod
    def from_display(cls, value: str) -> Enum:
        for m in cls:
            if m.display() == value:
                return m


class ModelType(BaseDisplayEnum):
    TOP_DOWN = "Multi Instance Top Down"
    BOTTOM_UP = "Multi Instance Bottom Up"
    SINGLE_INSTANCE = "Single Instance"


class TrackerType(BaseDisplayEnum):
    SIMPLE = "Simple", "simple"
    FLOW = "Flow Shift", "flow"
    KALMAN = "Kalman Filter", "kalman"


class Verbosity(BaseDisplayEnum):
    JSON = "Json", "json"
    RICH = "Rich", "rich"
    NONE = "None", "none"


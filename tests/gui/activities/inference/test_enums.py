from sleap.gui.activities.inference.enums import *


def test_display():
    # not tuple
    assert ModelType.TOP_DOWN.display() == "Multi Instance Top Down"
    # tuple
    assert TrackerType.SIMPLE.display() == "Simple"


def test_arg():
    assert TrackerType.SIMPLE.arg() == "simple"
    assert Verbosity.JSON.arg() == "json"


def test_from_display():
    assert ModelType.from_display("Single Instance") == ModelType.SINGLE_INSTANCE
    assert TrackerType.from_display("Simple") == TrackerType.SIMPLE
    assert TrackerType.from_display("na") is None


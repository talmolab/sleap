import sleap
import pytest


@pytest.fixture
def min_centroid_model_path():
    return "tests/data/models/minimal_instance.UNet.centroid"


@pytest.fixture
def min_centered_instance_model_path():
    return "tests/data/models/minimal_instance.UNet.centered_instance"


@pytest.fixture
def min_bottomup_model_path():
    return "tests/data/models/minimal_instance.UNet.bottomup"


@pytest.fixture
def min_single_instance_robot_model_path():
    return "tests/data/models/minimal_robot.UNet.single_instance"

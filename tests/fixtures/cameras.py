"""Camera fixtures for pytest."""

import pytest

@pytest.fixture
def min_session_calibration_toml_path():
    return "tests\data\cameras\minimal_session\calibration.toml"
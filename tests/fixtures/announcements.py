import pytest

TEST_BULLETIN_JSON = "tests/data/announcement_checker_bulletin/test_bulletin.json"

@pytest.fixture
def bulletin_json_path():
    return TEST_BULLETIN_JSON

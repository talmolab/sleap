import pandas as pd
from sleap.gui.web import (
    ReleaseChecker,
    Release,
    AnnouncementChecker,
    get_analytics_data,
    ping_analytics,
)
import pytest
from sleap.gui.app import create_app
import json
import os


def test_release_from_json():
    rls = Release.from_json(
        {
            "html_url": "https://github.com/talmolab/sleap/releases/tag/v1.0.10a7",
            "tag_name": "v1.0.10a7",
            "name": "SLEAP v1.0.10a7",
            "prerelease": True,
            "published_at": "2020-11-05T19:14:57Z",
            "body": "Body text",
        }
    )

    assert rls.date == pd.to_datetime("2020-11-05T19:14:57Z")
    assert rls.title == "SLEAP v1.0.10a7"
    assert rls.version == "v1.0.10a7"
    assert rls.prerelease
    assert rls.url == "https://github.com/talmolab/sleap/releases/tag/v1.0.10a7"
    assert rls.description == "Body text"


def test_release_checker():
    rls_stable = Release.from_json(
        {
            "html_url": "https://github.com/talmolab/sleap/releases/tag/v1.0.9",
            "tag_name": "v1.0.9",
            "name": "SLEAP v1.0.9",
            "prerelease": False,
            "published_at": "2020-09-04T17:00:52Z",
            "body": "Body text",
        }
    )
    rls_pre = Release.from_json(
        {
            "html_url": "https://github.com/talmolab/sleap/releases/tag/v1.0.10a7",
            "tag_name": "v1.0.10a7",
            "name": "SLEAP v1.0.10a7",
            "prerelease": True,
            "published_at": "2020-11-05T19:14:57Z",
            "body": "Body text",
        }
    )
    rls_test = Release.from_json(
        {
            "html_url": "https://github.com/talmolab/sleap/releases/tag/v1.0.10a8",
            "tag_name": "v1.0.10a8",
            "name": "SLEAP v1.0.10a8",
            "prerelease": True,
            "published_at": "2020-11-06T19:14:57Z",
            "body": "Do not use this release. This is a test.",
        }
    )

    checker = ReleaseChecker(releases=[rls_stable, rls_pre, rls_test])
    checker.checked = True  # TODO: Mock request?

    assert checker.latest_release == rls_pre
    assert checker.latest_prerelease == rls_pre
    assert checker.latest_stable == rls_stable

    assert checker.get_release("v1.0.9") == rls_stable

    with pytest.raises(ValueError):
        checker.get_release("abc")

    assert len(checker.releases) == 2
    assert checker.releases[0] != rls_test
    assert checker.releases[1] != rls_test


def test_announcementchecker():

    BULLETIN_JSON_PATH = "D:\\TalmoLab\\sleap\\tests\\data\\announcement_checker_bulletin\\test_bulletin.json"
    app = create_app()
    app.state = {}
    app.state["announcement last seen date"] = "10/10/2023"
    checker = AnnouncementChecker(app=app, bulletin_json_path=BULLETIN_JSON_PATH)

    # Check if the announcement checker gets the correct date from the app
    assert checker.previous_announcement_date == "10/10/2023"

    # Create dummy JSON file to check
    bulletin_data = [
        {"title": "title1", "date": "10/12/2023", "content": "New announcement"},
        {"title": "title2", "date": "10/07/2023", "content": "Old Announcment"},
    ]
    with open(BULLETIN_JSON_PATH, "w") as test_file:
        json.dump(bulletin_data, test_file)

    # Check if latest announcement is fetched
    announcement = checker.get_latest_announcement()
    assert announcement == ("10/12/2023", "New announcement")

    checker.update_announcement()
    assert app.state["announcement last seen date"] == "10/12/2023"
    assert app.state["announcement"] == "New announcement"

    # Delete the JSON file
    os.remove(BULLETIN_JSON_PATH)


def test_get_analytics_data():
    analytics_data = get_analytics_data()
    assert "platform" in analytics_data

import pandas as pd
from sleap.gui.release_checker import ReleaseChecker, Release
import pytest


def test_release_from_json():
    rls = Release.from_json(
        {
            "html_url": "https://github.com/murthylab/sleap/releases/tag/v1.0.10a7",
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
    assert rls.url == "https://github.com/murthylab/sleap/releases/tag/v1.0.10a7"
    assert rls.description == "Body text"


def test_release_checker():
    rls_stable = Release.from_json(
        {
            "html_url": "https://github.com/murthylab/sleap/releases/tag/v1.0.9",
            "tag_name": "v1.0.9",
            "name": "SLEAP v1.0.9",
            "prerelease": False,
            "published_at": "2020-09-04T17:00:52Z",
            "body": "Body text",
        }
    )
    rls_pre = Release.from_json(
        {
            "html_url": "https://github.com/murthylab/sleap/releases/tag/v1.0.10a7",
            "tag_name": "v1.0.10a7",
            "name": "SLEAP v1.0.10a7",
            "prerelease": True,
            "published_at": "2020-11-05T19:14:57Z",
            "body": "Body text",
        }
    )

    checker = ReleaseChecker(releases=[rls_stable, rls_pre])
    checker.checked = True  # TODO: Mock request?

    checker.latest_release == rls_pre
    checker.latest_prerelease == rls_pre
    checker.latest_stable == rls_stable

    checker.get_release("v1.0.9") == rls_stable

    with pytest.raises(ValueError):
        checker.get_release("abc")

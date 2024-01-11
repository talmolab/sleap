"""Module for checking for new releases on GitHub."""

import attr
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime


REPO_ID = "talmolab/sleap"
ANALYTICS_ENDPOINT = "https://analytics.sleap.ai/ping"
# TODO: Change the Bulletin URL to the main website before deploying
BULLETIN_JSON_ENDPOINT = "https://sleap.ai/develop/docs/_static/bulletin.json"


@attr.s(auto_attribs=True)
class Release:
    title: str = attr.ib(order=False)
    version: str = attr.ib(order=False)
    prerelease: bool = attr.ib(order=False)
    date: pd.Timestamp
    url: str = attr.ib(order=False)
    description: str = attr.ib(order=False)

    @classmethod
    def from_json(cls, data: Dict) -> "Release":
        """Construct a release from a JSON-decoded response."""
        return cls(
            title=data["name"],
            version=data["tag_name"],
            prerelease=data["prerelease"],
            date=pd.to_datetime(data["published_at"]),
            url=data["html_url"],
            description=data["body"],
        )


def filter_test_releases(releases: List[Release]) -> List[Release]:
    """Filter test releases out of a list of `Release`s.

    Args:
        releases: A list of `Release`s.

    Returns:
        The filtered list of `Release`s. Any `Release` that has a description
        containing the string `"Do not use this release. This is a test."` will be
        excluded.
    """
    # Exclude releases tagged with test string.
    return [
        rls
        for rls in releases
        if "Do not use this release. This is a test." not in rls.description
    ]


@attr.s(auto_attribs=True)
class ReleaseChecker:
    """Checker for new releases of SLEAP on GitHub.

    This uses the GitHub REST API:
    https://docs.github.com/en/rest/releases/releases#list-releases

    Attributes:
        repo_id: The name of the repository (defaults to: "talmolab/sleap")
        releases: A list of `Release`s from querying GitHub.
        checked: Indicates whether the releases page has been checked.
    """

    repo_id: str = REPO_ID
    releases: List[Release] = attr.ib(factory=list, converter=filter_test_releases)
    checked: bool = attr.ib(default=False, init=False)

    def check_for_releases(self) -> bool:
        """Check online for new releases.

        Returns:
            `True` if new releases were found, or `False` if no new releases or was not
            able to connect to the web.
        """
        try:
            self.checked = True
            response = requests.get(
                f"https://api.github.com/repos/{self.repo_id}/releases"
            )
        except (requests.ConnectionError, requests.Timeout):
            return False

        try:
            self.releases = [Release.from_json(r) for r in response.json()]
        except:
            return False

        return True

    @property
    def latest_release(self) -> Release:
        """Return latest release."""
        if not self.checked:
            self.check_for_releases()
        releases = sorted(self.releases)
        if len(releases) == 0:
            return None
        else:
            return releases[-1]

    @property
    def latest_stable(self) -> Release:
        """Return latest stable release."""
        if not self.checked:
            self.check_for_releases()
        releases = sorted([rls for rls in self.releases if not rls.prerelease])
        if len(releases) == 0:
            return None
        else:
            return releases[-1]

    @property
    def latest_prerelease(self) -> Release:
        """Return latest prerelease."""
        if not self.checked:
            self.check_for_releases()
        releases = sorted([rls for rls in self.releases if rls.prerelease])
        if len(releases) == 0:
            return None
        else:
            return releases[-1]

    def get_release(self, version: str) -> Release:
        """Get a release by version tag string.

        Args:
            version: Release version tag (e.g., "v1.0.9")

        Returns:
            The `Release` object with the associated version number.
        """
        if not self.checked:
            self.check_for_releases()

        for rls in self.releases:
            if rls.version == version:
                return rls

        raise ValueError(
            f"Release version was not found: {version}. "
            "Check the page online for a full listing: "
            f"https://github.com/{self.repo_id}"
        )


@attr.s(auto_attribs=True)
class AnnouncementChecker:
    """Checker for new announcements on the bulletin page of sleap."""

    state: "GuiState"
    _previous_announcement_date: str = None
    bulletin_json_data: Optional[List[Dict[str, str]]] = None
    json_data_url: str = BULLETIN_JSON_ENDPOINT
    checked: bool = attr.ib(default=False, init=False)

    @property
    def previous_announcement_date(self):
        _previous_announcement_date = self.state["announcement last seen date"]
        return _previous_announcement_date

    def check_for_bulletin_data(self) -> Optional[List[Dict]]:
        """Reads the bulletin data from the JSON file endpoint."""
        try:
            self.checked = True
            self.bulletin_json_data = requests.get(self.json_data_url).json()
        except (requests.ConnectionError, requests.Timeout):
            self.bulletin_json_data = None

    def new_announcement_available(self) -> bool:
        """Check if latest announcement is available."""
        if not self.checked:
            self.check_for_bulletin_data()
        if self.bulletin_json_data:
            if self.previous_announcement_date:
                latest_date = datetime.strptime(
                    self.bulletin_json_data[0]["date"], "%m/%d/%Y"
                )
                previous_date = datetime.strptime(
                    self.previous_announcement_date, "%m/%d/%Y"
                )
                if latest_date > previous_date:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def update_latest_announcement(self) -> Optional[Tuple[str, str, str]]:
        """Return latest announcements on the releases page not seen by user."""
        if self.new_announcement_available():
            announcement_markdown = ""
            for announcement in self.bulletin_json_data:
                announcement_content = "\n".join(announcement["content"].split("\n"))
                announcement_markdown += (
                    "## " + announcement["title"] + "\n" + announcement_content + "\n"
                )
            self.state["announcement"] = announcement_markdown
            self.state["announcement last seen date"] = self.bulletin_json_data[0][
                "date"
            ]


def get_analytics_data() -> Dict[str, Any]:
    """Gather data to be transmitted to analytics backend."""
    import os
    import sleap
    import tensorflow as tf
    from pathlib import Path
    import platform

    return {
        "sleap_version": sleap.__version__,
        "python_version": platform.python_version(),
        "tf_version": tf.__version__,
        "conda_env": Path(os.environ.get("CONDA_PREFIX", "")).stem,
        "platform": platform.platform(),
    }


def ping_analytics():
    """Ping analytics service with anonymous usage data.

    Notes:
        This only gets called when the GUI starts and obeys user preferences for data
        collection.

        See https://sleap.ai/help.html#usage-data for more information.
    """
    import threading

    analytics_data = get_analytics_data()

    def _ping_analytics():
        try:
            response = requests.post(
                ANALYTICS_ENDPOINT,
                json=analytics_data,
            )
        except (requests.ConnectionError, requests.Timeout):
            pass

    # Fire and forget.
    threading.Thread(target=_ping_analytics).start()

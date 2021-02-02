"""Module for checking for new releases on GitHub."""


import attr
import pandas as pd
import requests
from typing import List, Dict, Optional


REPO_ID = "murthylab/sleap"


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
    https://docs.github.com/en/free-pro-team@latest/rest/reference/repos#releases

    Attributes:
        repo_id: The name of the repository (defaults to: "murthylab/sleap")
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

        self.releases = [Release.from_json(r) for r in response.json()]

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

"""
Script to create tagged release of HEAD on develop for "nightly" builds.

The tag will be the version number in

https://github.com/murthylab/sleap/blob/develop/sleap/version.py

so you should bump the version number and push that to develop
before running this script. The version number should be

X.Y.ZaN

so that pypi knows this is a pre-release and shouldn't be installed
unless explicitly requested.

The script will prompt you for a GitHub access token.

When a tagged release is pushed to GitHub, this triggers AppVeyor
to build and upload the conda and pypi packages. For tags on develop,
the package is uploaded to anaconda with the "dev" label, so it won't
be installed by default unless the user explicitly requests

sleap/label/dev

as the channel (instead of "sleap").
"""

import datetime
import re
from os import path
import github
from getpass import getpass


def main(publish, draft):
    token = ""
    while not token:
        token = getpass("Access token for GitHub:")

    try:
        g = github.Github(token)
        repo = g.get_repo("murthylab/sleap")
    except github.BadCredentialsException as e:
        print("Unable to connect to the repo with those credentials.")
        return

    dev_branch = repo.get_branch("develop")

    # Get version from sleap/version.py in develop branch on github
    version_file = repo.get_contents(
        "sleap/version.py", ref=dev_branch.commit.sha
    ).decoded_content.decode()
    sleap_version = re.search("\d.+(?=['\"])", version_file).group(0)

    print(f"Releasing {dev_branch.commit.sha} (develop HEAD) as {sleap_version}...")

    if publish:
        x = repo.create_git_tag_and_release(
            tag=f"v{sleap_version}",
            tag_message="",
            release_name=f"SLEAP v{sleap_version}",
            release_message=f"Automated release of {sleap_version} on {datetime.date.today().isoformat()}",
            object=dev_branch.commit.sha,
            type="tree",
            draft=draft,
            prerelease=True,
        )

        print("Tag/release created on GitHub.")
        print(x)
    else:
        print("Just testing connection, no tag/release pushed.")


if __name__ == "__main__":
    main(publish=True, draft=False)

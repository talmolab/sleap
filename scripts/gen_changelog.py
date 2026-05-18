import re
import requests
import mkdocs_gen_files
import os

OWNER = "talmolab"
REPO = "sleap"
GH_TOKEN = os.environ.get("GH_TOKEN", None)
if GH_TOKEN is None:
    GH_TOKEN = os.environ.get("GH_TOKEN_READ_ONLY", None)
if GH_TOKEN is None:
    GH_TOKEN = os.environ.get("GITHUB_TOKEN", None)
if GH_TOKEN is None:
    import subprocess

    proc = subprocess.run("gh auth token", shell=True, capture_output=True)
    GH_TOKEN = proc.stdout.decode().strip()

if GH_TOKEN is None:
    print("Warning: No GitHub token found, rate limits may be exceeded.")


def fetch_release_notes(owner, repo, github_token=None):
    """
    Fetches the release notes for all releases of a GitHub repository.

    Parameters:
    - owner (str): The owner of the repository.
    - repo (str): The name of the repository.
    - github_token (str, optional): GitHub personal access token for authentication (if required).

    Returns:
    - list of dict: A list of releases, each containing "tag_name" and "body" of the release.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    headers["Accept"] = "application/vnd.github.html+json"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch releases: {response.status_code}, {response.text}"
        )

    releases = response.json()

    release_notes = []
    for release in releases:
        release_notes.append(
            {
                "tag_name": release["tag_name"],
                "body_html": release["body_html"],
            }
        )

    return release_notes


releases = fetch_release_notes(OWNER, REPO, GH_TOKEN)

with mkdocs_gen_files.open("changelog.md", "w") as page:
    contents = ["# Changelog\n"]

    for release in releases:
        # Title
        url = f"https://github.com/talmolab/sleap/releases/tag/{release['tag_name']}"
        contents.append(f"## [{release['tag_name']}]({url})\n")

        # Body
        contents.append(release["body_html"])

        contents.append("\n")

    page.writelines(contents)

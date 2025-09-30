"""Generate changelog page from git history."""

import subprocess
from pathlib import Path

import mkdocs_gen_files


def get_changelog():
    """Extract changelog from git tags and commits."""
    try:
        # Get all tags sorted by version
        result = subprocess.run(
            ["git", "tag", "--sort=-v:refname"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = result.stdout.strip().split("\n")

        changelog = "# Changelog\n\n"

        # For each tag, get the date and commits since previous tag
        for i, tag in enumerate(tags):
            if not tag:
                continue

            # Get tag date
            date_result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", tag],
                capture_output=True,
                text=True,
                check=True,
            )
            date = date_result.stdout.strip().split()[0]

            changelog += f"## {tag} ({date})\n\n"

            # Get commits between this tag and the previous one
            if i < len(tags) - 1 and tags[i + 1]:
                range_spec = f"{tags[i + 1]}..{tag}"
            else:
                # For the oldest tag, show all commits up to that tag
                range_spec = tag

            commits_result = subprocess.run(
                ["git", "log", "--oneline", "--no-merges", range_spec],
                capture_output=True,
                text=True,
                check=True,
            )

            commits = commits_result.stdout.strip().split("\n")
            for commit in commits:
                if commit:
                    # Format: hash message -> - message
                    parts = commit.split(" ", 1)
                    if len(parts) == 2:
                        changelog += f"- {parts[1]}\n"

            changelog += "\n"

        return changelog

    except subprocess.CalledProcessError:
        # Fallback if git commands fail
        return "# Changelog\n\nNo changelog available yet.\n"


# Generate changelog
changelog_content = get_changelog()

with mkdocs_gen_files.open("CHANGELOG.md", "w") as f:
    f.write(changelog_content)
